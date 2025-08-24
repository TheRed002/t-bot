"""
BacktestService - Comprehensive Service Layer for Backtesting Framework

This service provides enterprise-grade backtesting capabilities by integrating
all service layers, implementing result caching, and providing advanced
performance analytics with statistical robustness.

Dependencies:
- P-001: Core types, exceptions, config
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
- P-008: Risk management service
- P-009: Data service layer
- P-010: Execution service layer
- P-011: Strategy service layer
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pandas as pd
import redis.asyncio as redis
from pydantic import BaseModel, Field, field_validator

from src.backtesting.analysis import MonteCarloAnalyzer, WalkForwardAnalyzer
from src.backtesting.attribution import PerformanceAttributor
from src.backtesting.data_replay import DataReplayManager
from src.backtesting.engine import BacktestResult
from src.backtesting.metrics import MetricsCalculator
from src.backtesting.simulator import TradeSimulator
from src.backtesting.utils import convert_market_records_to_dataframe
from src.core.base.component import BaseComponent
from src.core.config import Config
from src.core.exceptions import BacktestError, ServiceError
from src.data.types import DataRequest
from src.error_handling.decorators import (
    with_circuit_breaker,
    with_error_context,
    with_fallback,
    with_retry,
)
from src.utils.decimal_utils import safe_decimal
from src.utils.decorators import time_execution
from src.utils.validators import ValidationFramework


class BacktestRequest(BaseModel):
    """Comprehensive backtest request model with validation."""

    strategy_config: dict[str, Any] = Field(..., description="Strategy configuration")
    symbols: list[str] = Field(..., min_length=1, description="Symbols to backtest")
    exchange: str = Field(default="binance", description="Exchange to use for backtesting")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: Decimal = Field(
        default_factory=lambda: safe_decimal("10000"),
        ge=safe_decimal("100"),
        description="Initial capital",
    )
    timeframe: str = Field(default="1h", description="Data timeframe")

    # Trading parameters
    commission_rate: Decimal = Field(
        default_factory=lambda: safe_decimal("0.001"), description="Commission rate"
    )
    slippage_rate: Decimal = Field(
        default_factory=lambda: safe_decimal("0.0005"), description="Slippage rate"
    )
    enable_shorting: bool = Field(default=False, description="Enable short selling")
    max_open_positions: int = Field(default=5, ge=1, le=20, description="Max open positions")

    # Risk management
    risk_config: dict[str, Any] = Field(default_factory=dict, description="Risk management config")
    position_sizing_method: str = Field(
        default="equal_weight", description="Position sizing method"
    )

    # Analysis options
    enable_monte_carlo: bool = Field(default=True, description="Enable Monte Carlo analysis")
    enable_walk_forward: bool = Field(default=True, description="Enable walk-forward analysis")
    monte_carlo_runs: int = Field(default=1000, ge=100, le=10000, description="Monte Carlo runs")

    # Caching
    use_cache: bool = Field(default=True, description="Use result caching")
    cache_ttl_hours: int = Field(default=24, ge=1, le=168, description="Cache TTL in hours")

    @field_validator("end_date")
    @classmethod
    def validate_dates(cls, v: datetime, info) -> datetime:
        """Validate date range."""
        if "start_date" in info.data and v <= info.data["start_date"]:
            raise ValueError("End date must be after start date")
        return v

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: list[str]) -> list[str]:
        """Validate symbol format."""
        for symbol in v:
            try:
                ValidationFramework.validate_symbol(symbol)
            except ValueError as e:
                raise ValueError(f"Invalid symbol {symbol}: {e}")
        return v


class BacktestCacheEntry(BaseModel):
    """Cache entry for backtest results."""

    request_hash: str = Field(..., description="Request hash for cache key")
    result: BacktestResult = Field(..., description="Backtest result")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Cache metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_hours: int = Field(default=24, description="TTL in hours")

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        age = datetime.now(timezone.utc) - self.created_at
        return age > timedelta(hours=self.ttl_hours)


class BacktestService(BaseComponent):
    """
    Comprehensive BacktestService integrating all service layers.

    This service provides:
    - Service layer integration for data, execution, risk management
    - Advanced result caching with Redis backend
    - Monte Carlo and walk-forward analysis
    - Performance attribution and metrics
    - Statistical validation and robustness testing
    """

    def __init__(self, config: Config, **services):
        """Initialize BacktestService with dependency injection."""
        # Convert config to dict if needed
        config_dict = None
        if config:
            config_dict = config.dict() if hasattr(config, "dict") else {}

        super().__init__(name="BacktestService", config=config_dict)
        self.config = config

        # Service dependencies (injected)
        self.data_service = services.get("DataService")
        self.execution_service = services.get("ExecutionService")
        self.risk_service = services.get("RiskService")
        self.strategy_service = services.get("StrategyService")
        self.capital_service = services.get("CapitalService")
        self.ml_service = services.get("MLService")

        # Validate critical service dependencies
        if self.data_service and not hasattr(self.data_service, "get_market_data"):
            self.logger.warning("DataService instance missing get_market_data method")
            self.data_service = None

        # Backtesting components
        self.metrics_calculator = MetricsCalculator()
        self.monte_carlo_analyzer: MonteCarloAnalyzer | None = None
        self.walk_forward_analyzer: WalkForwardAnalyzer | None = None
        self.performance_attributor: PerformanceAttributor | None = None
        self.data_replay_manager: DataReplayManager | None = None
        self.trade_simulator: TradeSimulator | None = None

        # Caching
        self._redis_client: redis.Redis | None = None
        self._memory_cache: dict[str, BacktestCacheEntry] = {}
        self._cache_config = self._setup_cache_config()

        # State tracking
        self._active_backtests: dict[str, dict[str, Any]] = {}
        self._initialized = False
        self._cache_available = False

    def _setup_cache_config(self) -> dict[str, Any]:
        """Setup caching configuration."""
        # Use proper config access pattern
        if hasattr(self.config, "backtest_cache"):
            cache_config = self.config.backtest_cache
        elif hasattr(self.config, "get"):
            cache_config = self.config.get("backtest_cache", {})
        else:
            cache_config = {}

        return {
            "redis_host": cache_config.get("redis_host", "localhost"),
            "redis_port": cache_config.get("redis_port", 6379),
            "redis_db": cache_config.get("redis_db", 1),
            "redis_password": cache_config.get("redis_password"),
            "memory_cache_size": cache_config.get("memory_cache_size", 100),
            "default_ttl_hours": cache_config.get("default_ttl_hours", 24),
        }

    async def initialize(self) -> None:
        """Initialize the BacktestService with all components."""
        try:
            if self._initialized:
                return

            self.logger.info("Initializing BacktestService...")

            # Initialize service dependencies
            await self._initialize_services()

            # Initialize backtesting components
            await self._initialize_backtesting_components()

            # Initialize caching
            await self._initialize_caching()

            self._initialized = True
            self.logger.info("BacktestService initialized successfully")

        except Exception as e:
            self.logger.error(f"BacktestService initialization failed: {e}")
            raise ServiceError(f"Initialization failed: {e}")

    async def _initialize_services(self) -> None:
        """Initialize service dependencies via dependency injection."""
        # Services are injected via constructor, just initialize them if needed
        if self.data_service and hasattr(self.data_service, "initialize"):
            await self.data_service.initialize()

        if self.execution_service and hasattr(self.execution_service, "initialize"):
            await self.execution_service.initialize()

        if self.risk_service and hasattr(self.risk_service, "initialize"):
            await self.risk_service.initialize()

        if self.strategy_service and hasattr(self.strategy_service, "initialize"):
            await self.strategy_service.initialize()

        if self.capital_service and hasattr(self.capital_service, "initialize"):
            await self.capital_service.initialize()

        if self.ml_service and hasattr(self.ml_service, "initialize"):
            await self.ml_service.initialize()

        self.logger.info("Service dependencies initialized")

    async def _initialize_backtesting_components(self) -> None:
        """Initialize backtesting-specific components."""
        # Initialize analyzers
        self.monte_carlo_analyzer = MonteCarloAnalyzer(self.config)
        self.walk_forward_analyzer = WalkForwardAnalyzer(self.config)
        self.performance_attributor = PerformanceAttributor(self.config)

        # Initialize data replay and simulation
        self.data_replay_manager = DataReplayManager(self.config)

        # Create SimulationConfig from Config for TradeSimulator
        from src.backtesting.simulator import SimulationConfig

        sim_config = SimulationConfig()  # Use defaults, can be enhanced later
        self.trade_simulator = TradeSimulator(sim_config)

        self.logger.info("Backtesting components initialized")

    async def _initialize_caching(self) -> None:
        """Initialize Redis caching for results."""
        try:
            self._redis_client = redis.Redis(
                host=self._cache_config["redis_host"],
                port=self._cache_config["redis_port"],
                db=self._cache_config["redis_db"],
                password=self._cache_config["redis_password"],
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )

            # Test connection
            await self._redis_client.ping()
            self._cache_available = True
            self.logger.info("Redis cache initialized")

        except Exception as e:
            self.logger.warning(f"Redis cache initialization failed, using memory only: {e}")
            self._redis_client = None
            self._cache_available = False

    @time_execution
    @with_error_context(component="backtesting", operation="service_run_backtest")
    @with_circuit_breaker(failure_threshold=3, recovery_timeout=60)
    @with_retry(max_attempts=3)
    async def run_backtest(self, request: BacktestRequest) -> BacktestResult:
        """
        Run comprehensive backtest with all service integrations.

        Args:
            request: BacktestRequest with all parameters

        Returns:
            BacktestResult with comprehensive metrics and analysis
        """
        if not self._initialized:
            await self.initialize()

        backtest_id = str(uuid.uuid4())
        self.logger.info(f"Starting backtest {backtest_id}", request=request.dict())

        # Check cache first
        if request.use_cache:
            cached_result = await self._get_cached_result(request)
            if cached_result:
                self.logger.info(f"Returning cached result for backtest {backtest_id}")
                return cached_result

        # Track backtest execution
        self._active_backtests[backtest_id] = {
            "request": request,
            "start_time": datetime.now(timezone.utc),
            "stage": "preparing",
            "progress": 0,
        }

        # Execute backtest pipeline
        result = await self._execute_backtest_pipeline(backtest_id, request)

        # Cache result if requested
        if request.use_cache:
            await self._cache_result(request, result)

        # Clean up
        if backtest_id in self._active_backtests:
            del self._active_backtests[backtest_id]

        self.logger.info(f"Backtest {backtest_id} completed successfully")
        return result

    async def _execute_backtest_pipeline(
        self, backtest_id: str, request: BacktestRequest
    ) -> BacktestResult:
        """Execute the complete backtest pipeline."""
        # Stage 1: Data preparation
        await self._update_backtest_stage(backtest_id, "data_preparation", 10)
        market_data = await self._prepare_market_data(request)

        # Stage 2: Strategy initialization
        await self._update_backtest_stage(backtest_id, "strategy_initialization", 20)
        strategy = await self._initialize_strategy(request.strategy_config)

        # Stage 3: Risk management setup
        await self._update_backtest_stage(backtest_id, "risk_setup", 30)
        risk_manager = await self._setup_risk_management(request.risk_config)

        # Stage 4: Core simulation
        await self._update_backtest_stage(backtest_id, "simulation", 40)
        simulation_result = await self._run_core_simulation(
            request, strategy, risk_manager, market_data
        )

        # Stage 5: Advanced analysis
        await self._update_backtest_stage(backtest_id, "advanced_analysis", 70)
        advanced_metrics = await self._run_advanced_analysis(
            request, simulation_result, market_data
        )

        # Stage 6: Results consolidation
        await self._update_backtest_stage(backtest_id, "results_consolidation", 90)
        final_result = await self._consolidate_results(simulation_result, advanced_metrics, request)

        await self._update_backtest_stage(backtest_id, "completed", 100)
        return final_result

    async def _prepare_market_data(self, request: BacktestRequest) -> dict[str, pd.DataFrame]:
        """Prepare market data for all symbols."""
        market_data = {}

        if not self.data_service:
            raise BacktestError("DataService not configured", "BACKTEST_010")

        for symbol in request.symbols:
            # Use DataService to get historical data
            data_request = DataRequest(
                symbol=symbol,
                exchange=request.exchange,
                start_time=request.start_date,
                end_time=request.end_date,
                use_cache=True,
            )

            try:
                records = await self.data_service.get_market_data(data_request)
            except Exception as e:
                self.logger.error(f"Failed to get market data for {symbol}: {e}")
                continue

            if records:
                # Convert to pandas DataFrame using shared utility
                df = convert_market_records_to_dataframe(records)
                market_data[symbol] = df

                self.logger.debug(f"Prepared {len(df)} data points for {symbol}")
            else:
                self.logger.warning(f"No data available for {symbol}")

        return market_data

    async def _initialize_strategy(self, strategy_config: dict[str, Any]) -> Any:
        """Initialize strategy using StrategyService."""
        from src.strategies.interfaces import BaseStrategyInterface

        # Validate strategy config before creation
        if not strategy_config:
            raise BacktestError("Strategy configuration is empty", "BACKTEST_006")

        if "strategy_type" not in strategy_config:
            raise BacktestError("Strategy type not specified in config", "BACKTEST_007")

        # Create strategy through service
        strategy = await self.strategy_service.create_strategy(strategy_config)

        # Validate created strategy
        if not strategy:
            raise BacktestError(
                f"Failed to create strategy of type: {strategy_config.get('strategy_type')}",
                "BACKTEST_008",
            )

        if not isinstance(strategy, BaseStrategyInterface):
            raise BacktestError(
                f"Invalid strategy type returned: {type(strategy).__name__}", "BACKTEST_009"
            )

        return strategy

    async def _setup_risk_management(self, risk_config: dict[str, Any]) -> Any:
        """Setup risk management using RiskManagementService."""
        return await self.risk_service.create_risk_manager(risk_config)

    async def _run_core_simulation(
        self,
        request: BacktestRequest,
        strategy: Any,
        risk_manager: Any,
        market_data: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        """Run core simulation using TradeSimulator."""
        from src.backtesting.simulator import SimulationConfig

        sim_config = SimulationConfig(
            initial_capital=float(request.initial_capital),
            commission_rate=float(request.commission_rate),
            slippage_rate=float(request.slippage_rate),
            enable_shorting=request.enable_shorting,
            max_positions=request.max_open_positions,
        )

        return await self.trade_simulator.run_simulation(
            config=sim_config,
            strategy=strategy,
            risk_manager=risk_manager,
            market_data=market_data,
            start_date=request.start_date,
            end_date=request.end_date,
        )

    async def _run_advanced_analysis(
        self,
        request: BacktestRequest,
        simulation_result: dict[str, Any],
        market_data: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        """Run advanced analysis including Monte Carlo and walk-forward."""
        advanced_metrics = {}

        # Monte Carlo analysis
        if request.enable_monte_carlo and self.monte_carlo_analyzer:
            try:
                mc_results = await self.monte_carlo_analyzer.run_analysis(
                    simulation_result=simulation_result,
                    market_data=market_data,
                    num_runs=request.monte_carlo_runs,
                )
                advanced_metrics["monte_carlo"] = mc_results
            except Exception as e:
                self.logger.error(f"Monte Carlo analysis failed: {e}")
                advanced_metrics["monte_carlo"] = None

        # Walk-forward analysis
        if request.enable_walk_forward and self.walk_forward_analyzer:
            try:
                wf_results = await self.walk_forward_analyzer.run_analysis(
                    strategy_config=request.strategy_config,
                    market_data=market_data,
                    start_date=request.start_date,
                    end_date=request.end_date,
                )
                advanced_metrics["walk_forward"] = wf_results
            except Exception as e:
                self.logger.error(f"Walk-forward analysis failed: {e}")
                advanced_metrics["walk_forward"] = None

        # Performance attribution
        if self.performance_attributor:
            try:
                attribution = await self.performance_attributor.analyze(
                    simulation_result, market_data
                )
                advanced_metrics["attribution"] = attribution
            except Exception as e:
                self.logger.error(f"Performance attribution failed: {e}")
                advanced_metrics["attribution"] = None

        return advanced_metrics

    async def _consolidate_results(
        self,
        simulation_result: dict[str, Any],
        advanced_metrics: dict[str, Any],
        request: BacktestRequest,
    ) -> BacktestResult:
        """Consolidate all results into final BacktestResult."""
        # Calculate comprehensive metrics
        equity_curve = simulation_result.get("equity_curve", [])
        trades = simulation_result.get("trades", [])
        daily_returns = simulation_result.get("daily_returns", [])
        initial_capital = float(request.initial_capital)

        # Base metrics from MetricsCalculator
        base_metrics = self.metrics_calculator.calculate_all(
            equity_curve=equity_curve,
            trades=trades,
            daily_returns=daily_returns,
            initial_capital=initial_capital,
        )

        # Combine with advanced metrics
        all_metrics = {**base_metrics, **advanced_metrics}

        # Create BacktestResult
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in trades if t.get("pnl", 0) <= 0]

        return BacktestResult(
            total_return=all_metrics.get("total_return", safe_decimal("0")),
            annual_return=all_metrics.get("annual_return", safe_decimal("0")),
            sharpe_ratio=all_metrics.get("sharpe_ratio", 0.0),
            sortino_ratio=all_metrics.get("sortino_ratio", 0.0),
            max_drawdown=all_metrics.get("max_drawdown", safe_decimal("0")),
            win_rate=all_metrics.get("win_rate", 0.0),
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=all_metrics.get("avg_win", safe_decimal("0")),
            avg_loss=all_metrics.get("avg_loss", safe_decimal("0")),
            profit_factor=all_metrics.get("profit_factor", 0.0),
            volatility=all_metrics.get("volatility", 0.0),
            var_95=all_metrics.get("var_95", safe_decimal("0")),
            cvar_95=all_metrics.get("cvar_95", safe_decimal("0")),
            equity_curve=equity_curve,
            trades=trades,
            daily_returns=daily_returns,
            metadata={
                "request": request.dict(),
                "advanced_metrics": advanced_metrics,
                "backtest_duration": str(request.end_date - request.start_date),
                "symbols": request.symbols,
                "service_version": "2.0.0",
            },
        )

    async def _update_backtest_stage(self, backtest_id: str, stage: str, progress: int) -> None:
        """Update backtest execution stage."""
        if backtest_id in self._active_backtests:
            self._active_backtests[backtest_id]["stage"] = stage
            self._active_backtests[backtest_id]["progress"] = progress
            self.logger.debug(f"Backtest {backtest_id} stage: {stage} ({progress}%)")

    def _generate_request_hash(self, request: BacktestRequest) -> str:
        """Generate hash for cache key from request."""
        import hashlib

        # Create deterministic string from request
        cache_data = {
            "strategy_config": request.strategy_config,
            "symbols": sorted(request.symbols),
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "initial_capital": str(request.initial_capital),
            "timeframe": request.timeframe,
            "commission_rate": str(request.commission_rate),
            "slippage_rate": str(request.slippage_rate),
            "enable_shorting": request.enable_shorting,
            "max_open_positions": request.max_open_positions,
            "risk_config": request.risk_config,
            "position_sizing_method": request.position_sizing_method,
        }

        cache_string = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.sha256(cache_string.encode()).hexdigest()

    @with_fallback(default_return=None)
    async def _get_cached_result(self, request: BacktestRequest) -> BacktestResult | None:
        """Get cached backtest result if available."""
        request_hash = self._generate_request_hash(request)

        # Check memory cache first
        if request_hash in self._memory_cache:
            entry = self._memory_cache[request_hash]
            if not entry.is_expired():
                return entry.result
            else:
                del self._memory_cache[request_hash]

        # Check Redis cache
        if self._redis_client:
            cache_key = f"backtest_result:{request_hash}"
            cached_data = await self._redis_client.get(cache_key)

            if cached_data:
                entry_data = json.loads(cached_data)
                entry = BacktestCacheEntry(**entry_data)

                if not entry.is_expired():
                    # Update memory cache
                    if len(self._memory_cache) < self._cache_config["memory_cache_size"]:
                        self._memory_cache[request_hash] = entry
                    return entry.result
                else:
                    # Remove expired entry
                    await self._redis_client.delete(cache_key)

        return None

    @with_fallback(default_return=None)
    async def _cache_result(self, request: BacktestRequest, result: BacktestResult) -> None:
        """Cache backtest result."""
        request_hash = self._generate_request_hash(request)

        cache_entry = BacktestCacheEntry(
            request_hash=request_hash,
            result=result,
            metadata={
                "symbols": request.symbols,
                "timeframe": request.timeframe,
                "duration": str(request.end_date - request.start_date),
            },
            ttl_hours=request.cache_ttl_hours,
        )

        # Update memory cache
        if len(self._memory_cache) < self._cache_config["memory_cache_size"]:
            self._memory_cache[request_hash] = cache_entry

        # Update Redis cache
        if self._redis_client:
            cache_key = f"backtest_result:{request_hash}"
            cache_data = cache_entry.dict(default=str)
            cache_json = json.dumps(cache_data, default=str)

            ttl_seconds = request.cache_ttl_hours * 3600
            await self._redis_client.setex(cache_key, ttl_seconds, cache_json)

    async def get_active_backtests(self) -> dict[str, dict[str, Any]]:
        """Get status of all active backtests."""
        return self._active_backtests.copy()

    async def cancel_backtest(self, backtest_id: str) -> bool:
        """Cancel an active backtest."""
        if backtest_id in self._active_backtests:
            self._active_backtests[backtest_id]["stage"] = "cancelled"
            self.logger.info(f"Backtest {backtest_id} cancelled")
            return True
        return False

    async def clear_cache(self, pattern: str = "*") -> int:
        """Clear cached results matching pattern."""
        cleared_count = 0

        # Clear memory cache
        if pattern == "*":
            cleared_count += len(self._memory_cache)
            self._memory_cache.clear()
        else:
            keys_to_remove = [k for k in self._memory_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._memory_cache[key]
            cleared_count += len(keys_to_remove)

        # Clear Redis cache
        if self._redis_client:
            try:
                cache_pattern = f"backtest_result:{pattern}"
                keys = await self._redis_client.keys(cache_pattern)
                if keys:
                    await self._redis_client.delete(*keys)
                    cleared_count += len(keys)
            except Exception as e:
                self.logger.error(f"Redis cache clear failed: {e}")

        self.logger.info(f"Cleared {cleared_count} cache entries")
        return cleared_count

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "memory_cache": {
                "entries": len(self._memory_cache),
                "max_size": self._cache_config["memory_cache_size"],
            },
            "redis_cache": {"available": self._cache_available},
        }

        if self._redis_client:
            try:
                redis_info = await self._redis_client.info("memory")
                stats["redis_cache"]["memory_usage"] = redis_info.get(
                    "used_memory_human", "unknown"
                )

                # Count backtest result keys
                keys = await self._redis_client.keys("backtest_result:*")
                stats["redis_cache"]["entries"] = len(keys)

            except Exception as e:
                stats["redis_cache"]["error"] = str(e)

        return stats

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            "status": "healthy",
            "initialized": self._initialized,
            "services": {},
            "cache": await self.get_cache_stats(),
            "active_backtests": len(self._active_backtests),
        }

        # Check service dependencies
        services_to_check = [
            ("data_service", self.data_service),
            ("execution_service", self.execution_service),
            ("risk_service", self.risk_service),
            ("strategy_service", self.strategy_service),
            ("capital_service", self.capital_service),
            ("ml_service", self.ml_service),
        ]

        for service_name, service in services_to_check:
            if service:
                try:
                    service_health = await service.health_check()
                    health["services"][service_name] = service_health.get("status", "unknown")
                except Exception as e:
                    health["services"][service_name] = f"unhealthy: {e}"
                    health["status"] = "degraded"
            else:
                health["services"][service_name] = "not_initialized"
                health["status"] = "degraded"

        return health

    async def cleanup(self) -> None:
        """Cleanup service resources."""
        try:
            # Cancel any active backtests
            for backtest_id in list(self._active_backtests.keys()):
                await self.cancel_backtest(backtest_id)

            # Close Redis connection
            if self._redis_client:
                await self._redis_client.close()

            # Clear caches
            self._memory_cache.clear()

            # Cleanup service dependencies
            services_to_cleanup = [
                self.data_service,
                self.execution_service,
                self.risk_service,
                self.strategy_service,
                self.capital_service,
                self.ml_service,
            ]

            for service in services_to_cleanup:
                if service and hasattr(service, "cleanup"):
                    await service.cleanup()

            self._initialized = False
            self.logger.info("BacktestService cleanup completed")

        except Exception as e:
            self.logger.error(f"BacktestService cleanup error: {e}")
