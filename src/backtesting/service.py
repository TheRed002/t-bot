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

import asyncio
import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.backtesting.utils import convert_market_records_to_dataframe

if TYPE_CHECKING:
    from src.backtesting.analysis import MonteCarloAnalyzer, WalkForwardAnalyzer
    from src.backtesting.attribution import PerformanceAttributor
    from src.backtesting.data_replay import DataReplayManager
    from src.backtesting.engine import BacktestResult
    from src.backtesting.metrics import MetricsCalculator
    from src.backtesting.simulator import TradeSimulator
from src.core.base.interfaces import HealthCheckResult, HealthStatus
from src.core.base.service import BaseService
from src.core.config import Config
from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.data.types import DataRequest
from src.utils.backtesting_decorators import service_operation
from src.utils.backtesting_validators import (
    validate_date_range,
    validate_is_expired,
    validate_symbol_list,
)
from src.utils.config_conversion import convert_config_to_dict
from src.utils.decimal_utils import to_decimal
from src.utils.messaging_patterns import ErrorPropagationMixin


class BacktestRequest(BaseModel):
    """Comprehensive backtest request model with validation."""

    strategy_config: dict[str, Any] = Field(..., description="Strategy configuration")
    symbols: list[str] = Field(..., min_length=1, description="Symbols to backtest")
    exchange: str = Field(default="binance", description="Exchange to use for backtesting")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: Decimal = Field(
        default_factory=lambda: to_decimal("10000"),
        description="Initial capital",
    )
    timeframe: str = Field(default="1h", description="Data timeframe")

    # Trading parameters
    commission_rate: Decimal = Field(
        default_factory=lambda: to_decimal("0.001"), description="Commission rate"
    )
    slippage_rate: Decimal = Field(
        default_factory=lambda: to_decimal("0.0005"), description="Slippage rate"
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
        return validate_date_range(v, info)

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: list[str]) -> list[str]:
        """Validate symbol format."""
        return validate_symbol_list(v)


class BacktestCacheEntry(BaseModel):
    """Cache entry for backtest results."""

    request_hash: str = Field(..., description="Request hash for cache key")
    result: Any = Field(..., description="Backtest result")  # Use Any to avoid circular dependency
    metadata: dict[str, Any] = Field(default_factory=dict, description="Cache metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_hours: int = Field(default=24, description="TTL in hours")

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return validate_is_expired({"created_at": self.created_at, "ttl_hours": self.ttl_hours})


class BacktestService(BaseService, ErrorPropagationMixin):
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
        # Convert config to dict using shared utility
        config_dict = convert_config_to_dict(config)
        super().__init__(name="BacktestService", config=config_dict)
        self.config = config

        # Use core logging
        self._logger = get_logger(self.__class__.__name__)

        # Initialize injector first
        self._injector = services.get("injector")
        if not self._injector:
            try:
                from src.core.dependency_injection import get_global_injector

                self._injector = get_global_injector()
            except Exception:
                self._injector = None

        # Service dependencies (injected) - use interfaces to decouple from concrete implementations
        # Try to resolve from injector if services are not provided directly
        injector = self._injector

        self.data_service = services.get("DataService")
        if not self.data_service and injector:
            try:
                self.data_service = injector.resolve("DataService")
            except Exception:
                self.data_service = None

        self.execution_service = services.get("ExecutionService")
        if not self.execution_service and injector:
            try:
                self.execution_service = injector.resolve("ExecutionService")
            except Exception:
                self.execution_service = None

        self.risk_service = services.get("RiskService")
        if not self.risk_service and injector:
            try:
                self.risk_service = injector.resolve("RiskService")
            except Exception:
                self.risk_service = None

        self.strategy_service = services.get("StrategyService")
        if not self.strategy_service and injector:
            try:
                self.strategy_service = injector.resolve("StrategyService")
            except Exception:
                self.strategy_service = None

        self.capital_service = services.get("CapitalService")
        if not self.capital_service and injector:
            try:
                self.capital_service = injector.resolve("CapitalService")
            except Exception:
                self.capital_service = None

        self.ml_service = services.get("MLService")
        if not self.ml_service and injector:
            try:
                self.ml_service = injector.resolve("MLService")
            except Exception:
                self.ml_service = None

        self.repository = services.get("BacktestRepositoryInterface")
        if not self.repository and injector:
            try:
                self.repository = injector.resolve("BacktestRepositoryInterface")
            except Exception:
                self.repository = None

        # Log service availability
        available_services = [
            name
            for name, service in [
                ("DataService", self.data_service),
                ("ExecutionService", self.execution_service),
                ("RiskService", self.risk_service),
                ("StrategyService", self.strategy_service),
                ("CapitalService", self.capital_service),
                ("MLService", self.ml_service),
                ("Repository", self.repository),
            ]
            if service is not None
        ]

        self._logger.info(
            "BacktestService initialized with services", available_services=available_services
        )  # type: ignore

        # Backtesting components - will be injected during initialization
        self.metrics_calculator: MetricsCalculator | None = None
        self.monte_carlo_analyzer: MonteCarloAnalyzer | None = None
        self.walk_forward_analyzer: WalkForwardAnalyzer | None = None
        self.performance_attributor: PerformanceAttributor | None = None
        self.data_replay_manager: DataReplayManager | None = None
        self.trade_simulator: TradeSimulator | None = None

        # Injector already initialized above

        # Caching abstraction - inject cache service instead of direct Redis
        self._cache_service = services.get("CacheService")
        if not self._cache_service and self._injector:
            try:
                self._cache_service = self._injector.resolve("CacheService")
            except Exception:
                self._cache_service = None
        self._memory_cache: dict[str, BacktestCacheEntry] = {}

        # State tracking
        self._active_backtests: dict[str, dict[str, Any]] = {}
        self._initialized = False
        self._cache_available = False

    async def initialize(self) -> None:
        """Initialize the BacktestService with all components."""
        return await self.execute_with_monitoring("initialize", self._initialize_impl)

    async def _initialize_impl(self) -> None:
        """Internal initialization implementation."""
        if self._initialized:
            return

        self._logger.info("Initializing BacktestService...")  # type: ignore

        # Initialize service dependencies
        await self._initialize_services()

        # Initialize backtesting components
        await self._initialize_backtesting_components()

        # Initialize cache service if available
        if self._cache_service and hasattr(self._cache_service, "initialize"):
            await self._cache_service.initialize()

        self._initialized = True
        self._logger.info("BacktestService initialized successfully")  # type: ignore

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

        self._logger.info("Service dependencies initialized")  # type: ignore

    async def _initialize_backtesting_components(self) -> None:
        """Initialize backtesting-specific components using dependency injection."""
        if not self._injector:
            raise ServiceError(
                "Dependency injector not available - cannot initialize backtesting components",
                error_code="BACKTEST_014",
            )

        try:
            # Use dependency injection to create components
            self.metrics_calculator = self._injector.resolve("MetricsCalculator")
            self.monte_carlo_analyzer = self._injector.resolve("MonteCarloAnalyzer")
            self.walk_forward_analyzer = self._injector.resolve("WalkForwardAnalyzer")
            self.performance_attributor = self._injector.resolve("PerformanceAttributor")
            self.data_replay_manager = self._injector.resolve("DataReplayManager")
            self.trade_simulator = self._injector.resolve("TradeSimulator")

            self._logger.info("Backtesting components initialized via dependency injection")  # type: ignore
        except Exception as e:
            self._logger.error(f"Failed to initialize backtesting components: {e}")  # type: ignore
            raise ServiceError(f"Component initialization failed: {e}") from e

    async def run_backtest(self, request: BacktestRequest) -> "BacktestResult":
        """Run comprehensive backtest using BaseService execution patterns."""
        return await self.execute_with_monitoring("run_backtest", self._run_backtest_impl, request)

    async def run_backtest_from_dict(self, request_data: dict[str, Any]) -> "BacktestResult":
        """Run backtest from dictionary request data (for controller layer)."""
        try:
            # Validate and create request object - business logic stays in service
            request = BacktestRequest(**request_data)
            return await self.run_backtest(request)
        except Exception as e:
            self._logger.error(f"Failed to process backtest request: {e}")  # type: ignore
            raise ServiceError(f"Invalid backtest request: {e}") from e

    async def serialize_result(self, result: "BacktestResult") -> dict[str, Any]:
        """Serialize BacktestResult for API response (moved from controller)."""
        try:
            return {
                "total_return": float(result.total_return),
                "annual_return": float(result.annual_return),
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "max_drawdown": float(result.max_drawdown),
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
                "winning_trades": result.winning_trades,
                "losing_trades": result.losing_trades,
                "avg_win": float(result.avg_win),
                "avg_loss": float(result.avg_loss),
                "profit_factor": result.profit_factor,
                "volatility": result.volatility,
                "var_95": float(result.var_95),
                "cvar_95": float(result.cvar_95),
                "equity_curve": result.equity_curve,
                "trades": result.trades,
                "daily_returns": result.daily_returns,
                "metadata": result.metadata,
            }
        except Exception as e:
            self._logger.error(f"Error serializing backtest result: {e}")  # type: ignore
            raise ServiceError(f"Failed to serialize result: {e}") from e

    @service_operation(service_name="backtesting", operation="service_run_backtest")
    async def _run_backtest_impl(self, request: BacktestRequest) -> "BacktestResult":
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
        self._logger.info(f"Starting backtest {backtest_id}", request=request.model_dump())  # type: ignore

        # Check cache first
        if request.use_cache:
            cached_result = await self._get_cached_result(request)
            if cached_result:
                self._logger.info(f"Returning cached result for backtest {backtest_id}")  # type: ignore
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

        self._logger.info(f"Backtest {backtest_id} completed successfully")  # type: ignore
        return result

    async def _execute_backtest_pipeline(
        self, backtest_id: str, request: BacktestRequest
    ) -> "BacktestResult":
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
        """Prepare market data for backtesting."""
        market_data = {}

        if not self.data_service:
            raise ServiceError(
                "DataService not configured for backtesting", error_code="BACKTEST_012"
            )

        for symbol in request.symbols:
            data_request = DataRequest(
                symbol=symbol,
                exchange=request.exchange,
                start_time=request.start_date,
                end_time=request.end_date,
                limit=10000,
                cache_ttl=3600,
                use_cache=True,
            )

            try:
                records = await self.data_service.get_market_data(data_request)
            except Exception as e:
                self._logger.error(f"Failed to get market data for {symbol}: {e}")  # type: ignore
                raise ServiceError(f"Failed to get market data for {symbol}: {e}") from e

            if records:
                df = convert_market_records_to_dataframe(records)
                market_data[symbol] = df
                self._logger.info(f"Market data prepared for {symbol}: {len(df)} data points")  # type: ignore
            else:
                self._logger.warning(f"No data available for {symbol}")  # type: ignore

        return market_data

    async def _initialize_strategy(self, strategy_config: dict[str, Any]) -> Any:
        """Initialize strategy for backtesting."""
        # Basic validation
        if not strategy_config:
            raise ValidationError(
                "Strategy configuration is empty",
                field_name="strategy_config",
                field_value=None,
                error_code="BACKTEST_007",
            )

        if "strategy_type" not in strategy_config:
            raise ValidationError(
                "Strategy type not specified in config",
                field_name="strategy_type",
                field_value=None,
                error_code="BACKTEST_008",
            )

        if self.strategy_service is None:
            raise ServiceError("Strategy service not configured", error_code="BACKTEST_009")

        try:
            strategy = await self.strategy_service.create_strategy(strategy_config)
        except Exception as e:
            self._logger.error(f"Failed to create strategy: {e}")  # type: ignore
            raise ServiceError(f"Failed to create strategy: {e}") from e

        if not strategy:
            raise ServiceError(
                f"Failed to create strategy of type: {strategy_config.get('strategy_type')}",
                error_code="BACKTEST_010",
            )

        return strategy

    async def _setup_risk_management(self, risk_config: dict[str, Any]) -> Any:
        """Setup risk management for backtesting."""
        if self.risk_service is None:
            self._logger.info("Risk service not configured - risk management disabled for backtest")  # type: ignore
            return None

        try:
            return await self.risk_service.create_risk_manager(risk_config)
        except Exception as e:
            self._logger.error(f"Failed to setup risk management: {e}")  # type: ignore
            raise ServiceError(
                f"Failed to setup risk management: {e}", error_code="BACKTEST_013"
            ) from e

    async def _run_core_simulation(
        self,
        request: BacktestRequest,
        strategy: Any,
        risk_manager: Any,
        market_data: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        """Run core simulation."""
        # Use factory to create BacktestEngine with proper service dependencies
        if self._injector:
            engine_factory = self._injector.resolve("BacktestEngineFactory")
        else:
            raise ServiceError(
                "Dependency injector not available for engine creation", error_code="BACKTEST_015"
            )

        # Create engine configuration
        from src.backtesting.engine import BacktestConfig

        engine_config = BacktestConfig(
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            symbols=request.symbols,
            exchange=request.exchange,
            timeframe=request.timeframe,
            commission=request.commission_rate,
            slippage=request.slippage_rate,
            enable_shorting=request.enable_shorting,
            max_open_positions=request.max_open_positions,
        )

        # Create engine with dependencies
        engine = engine_factory(
            config=engine_config,
            strategy=strategy,
            risk_manager=risk_manager,
            data_service=self.data_service,
            execution_engine_service=self.execution_service,
        )

        # Run the backtest using the engine
        try:
            result = await engine.run()

            # Convert BacktestResult to dict for service layer
            return {
                "equity_curve": result.equity_curve,
                "trades": result.trades,
                "daily_returns": result.daily_returns,
                "positions": {},  # Engine handles position management internally
                "execution_stats": {
                    "total_trades": result.total_trades,
                    "winning_trades": result.winning_trades,
                    "win_rate": result.win_rate,
                },
                "result": result,  # Keep full result for metrics
            }
        except Exception as e:
            self._logger.error(f"BacktestEngine execution failed: {e}")  # type: ignore
            raise ServiceError(f"Simulation engine failed: {e}", error_code="BACKTEST_016") from e

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
                self._logger.error(f"Monte Carlo analysis failed: {e}")  # type: ignore
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
                self._logger.error(f"Walk-forward analysis failed: {e}")  # type: ignore
                advanced_metrics["walk_forward"] = None

        # Performance attribution
        if self.performance_attributor:
            try:
                attribution = await self.performance_attributor.analyze(
                    simulation_result, market_data
                )
                advanced_metrics["attribution"] = attribution
            except Exception as e:
                self._logger.error(f"Performance attribution failed: {e}")  # type: ignore
                advanced_metrics["attribution"] = None

        return advanced_metrics

    async def _consolidate_results(
        self,
        simulation_result: dict[str, Any],
        advanced_metrics: dict[str, Any],
        request: BacktestRequest,
    ) -> "BacktestResult":
        """Consolidate all results into final BacktestResult."""
        # Check if we have a BacktestResult from engine
        if "result" in simulation_result:
            # Engine already produced a BacktestResult - enhance it with advanced metrics
            base_result = simulation_result["result"]

            # Add advanced metrics to metadata
            metadata = base_result.metadata.copy()
            metadata["advanced_metrics"] = advanced_metrics

            # Create enhanced result
            from src.backtesting.engine import BacktestResult

            return BacktestResult(
                total_return=base_result.total_return,
                annual_return=base_result.annual_return,
                sharpe_ratio=base_result.sharpe_ratio,
                sortino_ratio=base_result.sortino_ratio,
                max_drawdown=base_result.max_drawdown,
                win_rate=base_result.win_rate,
                total_trades=base_result.total_trades,
                winning_trades=base_result.winning_trades,
                losing_trades=base_result.losing_trades,
                avg_win=base_result.avg_win,
                avg_loss=base_result.avg_loss,
                profit_factor=base_result.profit_factor,
                volatility=base_result.volatility,
                var_95=base_result.var_95,
                cvar_95=base_result.cvar_95,
                equity_curve=base_result.equity_curve,
                trades=base_result.trades,
                daily_returns=base_result.daily_returns,
                metadata=metadata,
            )
        else:
            # Fallback to legacy consolidation logic
            equity_curve = simulation_result.get("equity_curve", [])
            trades = simulation_result.get("trades", [])
            daily_returns = simulation_result.get("daily_returns", [])
            initial_capital = request.initial_capital

            # Base metrics from MetricsCalculator
            if self.metrics_calculator:
                base_metrics = self.metrics_calculator.calculate_all(
                    equity_curve=equity_curve,
                    trades=trades,
                    daily_returns=daily_returns,
                    initial_capital=float(initial_capital),
                )
            else:
                # Fallback basic metrics
                base_metrics = {
                    "total_return": to_decimal("0"),
                    "annual_return": to_decimal("0"),
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "max_drawdown": to_decimal("0"),
                    "win_rate": 0.0,
                    "avg_win": to_decimal("0"),
                    "avg_loss": to_decimal("0"),
                    "profit_factor": 0.0,
                    "volatility": 0.0,
                    "var_95": to_decimal("0"),
                    "cvar_95": to_decimal("0"),
                }

            # Combine with advanced metrics
            all_metrics = {**base_metrics, **advanced_metrics}

            # Create BacktestResult
            winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
            losing_trades = [t for t in trades if t.get("pnl", 0) <= 0]

            from src.backtesting.engine import BacktestResult

            return BacktestResult(
                total_return=all_metrics.get("total_return", to_decimal("0")),
                annual_return=all_metrics.get("annual_return", to_decimal("0")),
                sharpe_ratio=all_metrics.get("sharpe_ratio", 0.0),
                sortino_ratio=all_metrics.get("sortino_ratio", 0.0),
                max_drawdown=all_metrics.get("max_drawdown", to_decimal("0")),
                win_rate=all_metrics.get("win_rate", 0.0),
                total_trades=len(trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                avg_win=all_metrics.get("avg_win", to_decimal("0")),
                avg_loss=all_metrics.get("avg_loss", to_decimal("0")),
                profit_factor=all_metrics.get("profit_factor", 0.0),
                volatility=all_metrics.get("volatility", 0.0),
                var_95=all_metrics.get("var_95", to_decimal("0")),
                cvar_95=all_metrics.get("cvar_95", to_decimal("0")),
                equity_curve=equity_curve,
                trades=trades,
                daily_returns=daily_returns,
                metadata={
                    "request": request.model_dump(),
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
            self._logger.info(
                "Backtest progress", backtest_id=backtest_id, stage=stage, progress=progress
            )  # type: ignore

    def _generate_request_hash(self, request: BacktestRequest) -> str:
        """Generate hash for cache key from request."""
        # Delegate hash generation to utility service to avoid business logic in service
        try:
            from src.utils.serialization_utilities import HashGenerator

            return HashGenerator.generate_backtest_hash(request)
        except ImportError:
            # Fallback implementation
            import hashlib

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

    async def _get_cached_result(self, request: BacktestRequest) -> "BacktestResult | None":
        """Get cached backtest result if available."""
        if not self._cache_service:
            return None

        request_hash = self._generate_request_hash(request)
        cache_key = f"backtest_result:{request_hash}"

        try:
            cached_data = await self._cache_service.get(cache_key)
            if cached_data:
                entry = BacktestCacheEntry(**cached_data)
                if not entry.is_expired():
                    return entry.result
                else:
                    # Remove expired entry
                    await self._cache_service.delete(cache_key)
        except Exception as e:
            self._logger.warning(f"Cache read error: {e}")  # type: ignore

        return None

    async def _cache_result(self, request: BacktestRequest, result: "BacktestResult") -> None:
        """Cache backtest result and persist to repository."""
        if not self._cache_service:
            return

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

        cache_key = f"backtest_result:{request_hash}"
        ttl_seconds = request.cache_ttl_hours * 3600

        try:
            await self._cache_service.set(cache_key, cache_entry.model_dump(), ttl_seconds)
        except Exception as e:
            self._logger.warning(f"Cache write error: {e}")  # type: ignore

        # Persist to repository for long-term storage
        await self._persist_result(request, result)

    async def _persist_result(
        self, request: BacktestRequest, result: "BacktestResult"
    ) -> str | None:
        """Persist backtest result to repository."""
        if not self.repository:
            self._logger.warning("Repository not available for result persistence")  # type: ignore
            return None

        try:
            result_data = await self.serialize_result(result)
            request_data = request.model_dump()

            result_id = await self.repository.save_backtest_result(result_data, request_data)
            self._logger.info(f"Persisted backtest result with ID: {result_id}")  # type: ignore
            return result_id

        except Exception as e:
            self._logger.error(f"Failed to persist backtest result: {e}")  # type: ignore
            return None

    async def get_active_backtests(self) -> dict[str, dict[str, Any]]:
        """Get status of all active backtests."""
        return await self.execute_with_monitoring(
            "get_active_backtests", self._get_active_backtests_impl
        )

    async def _get_active_backtests_impl(self) -> dict[str, dict[str, Any]]:
        """Implementation for getting active backtests."""
        return self._active_backtests.copy()

    async def cancel_backtest(self, backtest_id: str) -> bool:
        """Cancel an active backtest."""
        return await self.execute_with_monitoring(
            "cancel_backtest", self._cancel_backtest_impl, backtest_id
        )

    async def _cancel_backtest_impl(self, backtest_id: str) -> bool:
        """Implementation for cancelling backtest."""
        if backtest_id in self._active_backtests:
            self._active_backtests[backtest_id]["stage"] = "cancelled"
            self._logger.info(f"Backtest {backtest_id} cancelled")  # type: ignore
            return True
        return False

    async def clear_cache(self, pattern: str = "*") -> int:
        """Clear cached results matching pattern."""
        return await self.execute_with_monitoring("clear_cache", self._clear_cache_impl, pattern)

    async def _clear_cache_impl(self, pattern: str = "*") -> int:
        """Implementation for clearing cache."""
        cleared_count = 0

        if self._cache_service:
            try:
                cache_pattern = f"backtest_result:{pattern}"
                cleared_count = await self._cache_service.clear_pattern(cache_pattern)
            except Exception as e:
                self._logger.error(f"Cache clear failed: {e}")  # type: ignore

        # Clear memory cache
        if pattern == "*":
            cleared_count += len(self._memory_cache)
            self._memory_cache.clear()
        else:
            keys_to_remove = [k for k in self._memory_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._memory_cache[key]
            cleared_count += len(keys_to_remove)

        self._logger.info(f"Cleared {cleared_count} cache entries")  # type: ignore
        return cleared_count

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "memory_cache": {
                "entries": len(self._memory_cache),
            },
            "cache_service": {"available": self._cache_service is not None},
        }

        if self._cache_service:
            try:
                cache_stats = await self._cache_service.get_stats()
                stats["cache_service"].update(cache_stats)  # type: ignore
            except Exception as e:
                stats["cache_service"]["error"] = str(e)  # type: ignore

        return stats

    async def get_backtest_result(self, result_id: str) -> dict[str, Any] | None:
        """Get a specific backtest result by ID."""
        if not self.repository:
            raise ServiceError("Repository not available", error_code="SERVICE_001")

        try:
            return await self.repository.get_backtest_result(result_id)
        except Exception as e:
            self._logger.error(f"Failed to get backtest result {result_id}: {e}")  # type: ignore
            raise ServiceError(
                f"Failed to retrieve backtest result: {e}", error_code="SERVICE_002"
            ) from e

    async def list_backtest_results(
        self, limit: int = 50, offset: int = 0, strategy_type: str | None = None
    ) -> list[dict[str, Any]]:
        """List backtest results with filtering."""
        if not self.repository:
            raise ServiceError("Repository not available", error_code="SERVICE_003")

        try:
            return await self.repository.list_backtest_results(limit, offset, strategy_type)
        except Exception as e:
            self._logger.error(f"Failed to list backtest results: {e}")  # type: ignore
            raise ServiceError(
                f"Failed to list backtest results: {e}", error_code="SERVICE_004"
            ) from e

    async def delete_backtest_result(self, result_id: str) -> bool:
        """Delete a specific backtest result by ID."""
        if not self.repository:
            raise ServiceError("Repository not available", error_code="SERVICE_005")

        try:
            return await self.repository.delete_backtest_result(result_id)
        except Exception as e:
            self._logger.error(f"Failed to delete backtest result {result_id}: {e}")  # type: ignore
            raise ServiceError(
                f"Failed to delete backtest result: {e}", error_code="SERVICE_006"
            ) from e

    async def health_check(self) -> HealthCheckResult:
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
            ("repository", self.repository),
        ]

        services_dict: dict[str, str] = {}
        for service_name, service in services_to_check:
            if service:
                try:
                    service_health = await service.health_check()
                    if hasattr(service_health, "status"):
                        services_dict[service_name] = str(service_health.status)
                    elif isinstance(service_health, dict):
                        services_dict[service_name] = service_health.get("status", "unknown")
                    else:
                        services_dict[service_name] = str(service_health)
                except Exception as e:
                    services_dict[service_name] = f"unhealthy: {e}"
                    health["status"] = "degraded"
            else:
                services_dict[service_name] = "not_initialized"
                health["status"] = "degraded"

        health["services"] = services_dict

        # Convert to HealthCheckResult
        from datetime import timezone

        status = HealthStatus.HEALTHY if health["status"] == "healthy" else HealthStatus.UNHEALTHY
        return HealthCheckResult(
            status=status,
            message=f"BacktestService status: {health['status']}",
            details=health,
            check_time=datetime.now(timezone.utc),
        )

    async def cleanup(self) -> None:
        """Cleanup service resources with proper async coordination."""
        try:
            # Cancel any active backtests concurrently
            if self._active_backtests:
                cancel_tasks = []
                for backtest_id in list(self._active_backtests.keys()):
                    task = asyncio.create_task(self.cancel_backtest(backtest_id))
                    cancel_tasks.append(task)

                if cancel_tasks:
                    # Wait for all cancellations with timeout
                    await asyncio.wait_for(
                        asyncio.gather(*cancel_tasks, return_exceptions=True), timeout=10.0
                    )

            # Cleanup services concurrently where possible
            async_cleanup_tasks = []

            # Cache service cleanup
            if self._cache_service and hasattr(self._cache_service, "cleanup"):
                async_cleanup_tasks.append(
                    self._safe_async_cleanup("cache_service", self._cache_service.cleanup())
                )

            # Data replay manager cleanup
            if self.data_replay_manager and hasattr(self.data_replay_manager, "cleanup"):
                async_cleanup_tasks.append(
                    self._safe_async_cleanup(
                        "data_replay_manager", self.data_replay_manager.cleanup()
                    )
                )

            # Service dependencies cleanup
            services_to_cleanup = [
                ("data_service", self.data_service),
                ("execution_service", self.execution_service),
                ("risk_service", self.risk_service),
                ("strategy_service", self.strategy_service),
                ("capital_service", self.capital_service),
                ("ml_service", self.ml_service),
                ("repository", self.repository),
            ]

            for service_name, service in services_to_cleanup:
                if service and hasattr(service, "cleanup"):
                    async_cleanup_tasks.append(
                        self._safe_async_cleanup(service_name, service.cleanup())
                    )

            # Execute all async cleanups concurrently with timeout
            if async_cleanup_tasks:
                await asyncio.wait_for(
                    asyncio.gather(*async_cleanup_tasks, return_exceptions=True), timeout=30.0
                )

            # Synchronous cleanup for components that don't support async
            if self.trade_simulator and hasattr(self.trade_simulator, "cleanup"):
                try:
                    self.trade_simulator.cleanup()
                except Exception as sync_cleanup_error:
                    self._logger.warning(f"Trade simulator cleanup error: {sync_cleanup_error}")  # type: ignore

            # Clear caches
            self._memory_cache.clear()

            self._initialized = False
            self._logger.info("BacktestService cleanup completed")  # type: ignore

        except asyncio.TimeoutError:
            self._logger.error("BacktestService cleanup timed out")  # type: ignore
        except Exception as e:
            self._logger.error(f"BacktestService cleanup error: {e}")  # type: ignore
            # Don't re-raise cleanup errors to avoid masking original issues

    async def _safe_async_cleanup(self, service_name: str, cleanup_coro) -> None:
        """Safely execute async cleanup with error handling."""
        try:
            await cleanup_coro
            self._logger.debug(f"{service_name} cleanup completed")  # type: ignore
        except Exception as cleanup_error:
            self._logger.warning(f"{service_name} cleanup error: {cleanup_error}")  # type: ignore
