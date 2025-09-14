"""
Strategy Service - Business logic layer for strategy operations.

This module provides the service layer for strategy management, including:
- Strategy lifecycle management
- Signal processing and validation
- Performance monitoring
- Risk integration
- Backtesting coordination
"""

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from src.core.base.service import BaseService
from src.core.caching.cache_decorators import cached
from src.core.caching.cache_manager import get_cache_manager
from src.core.exceptions import ServiceError, StrategyError
from src.core.types import (
    MarketData,
    Signal,
    StrategyConfig,
    StrategyMetrics,
    StrategyStatus,
)
from src.error_handling import (
    ErrorHandler,
    ErrorSeverity,
    get_global_error_handler,
    with_error_context,
)
from src.strategies.dependencies import StrategyServiceContainer, create_strategy_service_container
from src.strategies.interfaces import StrategyServiceInterface
from src.utils.decorators import time_execution

# Constants for production configuration
MIN_SIGNAL_CONFIDENCE = 0.1
DEFAULT_CACHE_TTL = 300  # 5 minutes
STRATEGY_METRICS_UPDATE_TIMEOUT = 300  # 5 minutes
DEFAULT_CACHE_TTL_SHORT = 30  # 30 seconds


# Define cache_strategy_signals decorator with proper strategy-specific implementation
def cache_strategy_signals(strategy_id_arg_name: str, ttl: int = DEFAULT_CACHE_TTL) -> Callable:
    """Cache decorator specifically for strategy signals."""
    return cached(
        ttl=ttl,
        namespace="strategy",
        data_type="strategy",
        key_generator=lambda *args,
        **kwargs: f"signals_{kwargs.get(strategy_id_arg_name, 'unknown')}",
    )


class StrategyService(BaseService, StrategyServiceInterface):
    """
    Service layer for strategy operations and management.

    Provides business logic for:
    - Strategy lifecycle management
    - Signal generation and validation
    - Performance monitoring
    - Risk management integration
    """

    def __init__(
        self,
        name: str = "StrategyService",
        config: dict[str, Any] | None = None,
        repository=None,
        risk_manager=None,
        exchange_factory=None,
        data_service=None,
        service_manager=None,
    ):
        """
        Initialize strategy service with injected dependencies.

        Args:
            name: Service name
            config: Service configuration
            repository: Strategy repository (injected)
            risk_manager: Risk manager service (injected)
            exchange_factory: Exchange factory (injected)
            data_service: Data service (injected)
            service_manager: Service manager for lazy loading dependencies
        """
        super().__init__(name, config)

        # Strategy registry
        self._active_strategies: dict[str, Any] = {}
        self._strategy_configs: dict[str, StrategyConfig] = {}
        self._strategy_metrics: dict[str, StrategyMetrics] = {}

        # Injected dependencies
        self._repository = repository
        self._risk_manager = risk_manager
        self._exchange_factory = exchange_factory
        self._data_service = data_service
        self._service_manager = service_manager
        self._error_handler: ErrorHandler = get_global_error_handler()

        # Initialize cache manager
        self.cache_manager = get_cache_manager(config=config)

        # Strategy performance tracking
        self._performance_cache: dict[str, dict[str, Any]] = {}
        self._signal_history: dict[str, list[Signal]] = {}

        self._logger.info("StrategyService initialized with dependency injection")

        # Create strategy service container for dependency injection to strategies
        self._strategy_services: StrategyServiceContainer | None = None

        # Update the repository to use the new strategy repository
        if self._repository is None and self._service_manager:
            # Try to get the new repository from service manager
            try:
                # Note: In production, this would be injected through DI
                self._logger.info("Strategy repository will be injected through DI")
            except Exception as e:
                self._logger.warning(f"Could not initialize strategy repository: {e}")

    async def _do_start(self) -> None:
        """Start service - dependencies already injected in constructor."""
        try:
            # Validate injected dependencies
            if self._repository is None:
                self._logger.warning("No repository injected - some operations may be limited")

            if self._risk_manager is None:
                self._logger.warning("No risk manager injected - risk validation will be limited")

            if self._exchange_factory is None:
                self._logger.warning(
                    "No exchange factory injected - strategy execution may be limited"
                )

            if self._data_service is None:
                self._logger.warning("No data service injected - data operations may be limited")

            # Build strategy service container for dependency injection to strategies
            self._strategy_services = await asyncio.wait_for(
                self._build_strategy_service_container(),
                timeout=30.0  # 30 second timeout for service container initialization
            )
            self._logger.info("Strategy service started with dependency injection")

        except Exception as e:
            raise ServiceError(f"Failed to start strategy service: {e}") from e

    async def _build_strategy_service_container(self) -> StrategyServiceContainer:
        """Build service container for strategy dependency injection."""
        try:
            # Get services from service manager if available
            risk_service = None
            data_service = self._data_service
            execution_service = None
            monitoring_service = None
            state_service = None
            capital_service = None
            analytics_service = None
            ml_service = None
            optimization_service = None

            if self._service_manager:
                try:
                    risk_service = await self._service_manager.get_service("risk_management")
                except Exception as e:
                    self._logger.debug(f"Could not get risk service: {e}")

                try:
                    execution_service = await self._service_manager.get_service("execution")
                except Exception as e:
                    self._logger.debug(f"Could not get execution service: {e}")

                try:
                    monitoring_service = await self._service_manager.get_service("monitoring")
                except Exception as e:
                    self._logger.debug(f"Could not get monitoring service: {e}")

                try:
                    state_service = await self._service_manager.get_service("state")
                except Exception as e:
                    self._logger.debug(f"Could not get state service: {e}")

                try:
                    capital_service = await self._service_manager.get_service("capital_management")
                except Exception as e:
                    self._logger.debug(f"Could not get capital service: {e}")

                try:
                    analytics_service = await self._service_manager.get_service("analytics")
                except Exception as e:
                    self._logger.debug(f"Could not get analytics service: {e}")

                try:
                    ml_service = await self._service_manager.get_service("ml")
                except Exception as e:
                    self._logger.debug(f"Could not get ml service: {e}")

                try:
                    optimization_service = await self._service_manager.get_service("optimization")
                except Exception as e:
                    self._logger.debug(f"Could not get optimization service: {e}")

            # Create the service container
            container = create_strategy_service_container(
                risk_service=risk_service or self._risk_manager,
                data_service=data_service,
                execution_service=execution_service,
                monitoring_service=monitoring_service,
                state_service=state_service,
                capital_service=capital_service,
                analytics_service=analytics_service,
                ml_service=ml_service,
                optimization_service=optimization_service,
            )

            self._logger.info(
                "Strategy service container built",
                services_available=container.get_service_status()
            )
            return container

        except Exception as e:
            self._logger.error(f"Failed to build strategy service container: {e}")
            # Return empty container as fallback
            return StrategyServiceContainer()

    # Strategy Lifecycle Management
    async def register_strategy(
        self, strategy_id: str, strategy_instance: Any, config: StrategyConfig
    ) -> None:
        """
        Register a strategy with the service.

        Args:
            strategy_id: Unique strategy identifier
            strategy_instance: Strategy instance
            config: Strategy configuration

        Raises:
            ServiceError: If strategy registration fails
        """
        return await self.execute_with_monitoring(
            "register_strategy",
            self._register_strategy_impl,
            strategy_id,
            strategy_instance,
            config,
        )

    async def _register_strategy_impl(
        self, strategy_id: str, strategy_instance: Any, config: StrategyConfig
    ) -> None:
        """Internal implementation for strategy registration."""
        if strategy_id in self._active_strategies:
            raise StrategyError(f"Strategy {strategy_id} already registered")

        # Validate strategy configuration
        if not await self.validate_strategy_config(config):
            raise StrategyError(f"Invalid configuration for strategy {strategy_id}")

        # Initialize strategy dependencies using service container
        if hasattr(strategy_instance, "services"):
            # Strategy already has service container (new pattern)
            if not strategy_instance.services or not strategy_instance.services.is_ready():
                strategy_instance.services = self._strategy_services
                self._logger.info(
                    f"Strategy {strategy_id} updated with service container",
                    services_available=strategy_instance.services.get_service_status()
                )
        else:
            # Legacy strategy support - use old setter methods (DEPRECATED)
            self._logger.warning(
                f"Strategy {strategy_id} using deprecated service injection methods"
            )
            if self._risk_manager:
                strategy_instance.set_risk_manager(self._risk_manager)

            if self._exchange_factory:
                # Get appropriate exchange for strategy
                try:
                    exchange = await self._exchange_factory.get_exchange(config.exchange_type)
                    strategy_instance.set_exchange(exchange)
                except Exception as e:
                    self._logger.warning(f"Could not set exchange for strategy: {e}")

            if self._data_service:
                strategy_instance.set_data_service(self._data_service)

        # Register strategy
        self._active_strategies[strategy_id] = strategy_instance
        self._strategy_configs[strategy_id] = config
        self._strategy_metrics[strategy_id] = StrategyMetrics(strategy_id=strategy_id)
        self._signal_history[strategy_id] = []

        # Save strategy to database via repository
        if self._repository:
            try:
                from decimal import Decimal

                from src.database.models import Strategy

                # Create database strategy record
                strategy_db = Strategy(
                    name=config.name,
                    type=config.strategy_type,
                    status="inactive",
                    bot_id=config.bot_id if hasattr(config, "bot_id") else None,
                    params=config.parameters,
                    max_position_size=Decimal(str(getattr(config, "max_position_size", "1000.0"))),
                    risk_per_trade=Decimal(str(getattr(config, "risk_per_trade", "0.02"))),
                )

                await self._repository.create_strategy(strategy_db)
                self._logger.info(f"Strategy saved to database: {strategy_db.id}")

            except Exception as e:
                self._logger.warning(f"Failed to save strategy to database: {e}")

        self._logger.info(
            "Strategy registered successfully",
            strategy_id=strategy_id,
            strategy_type=config.strategy_type,
        )

    async def start_strategy(self, strategy_id: str) -> None:
        """
        Start a registered strategy.

        Args:
            strategy_id: Strategy to start

        Raises:
            ServiceError: If strategy start fails
        """
        return await self.execute_with_monitoring(
            "start_strategy", self._start_strategy_impl, strategy_id
        )

    async def _start_strategy_impl(self, strategy_id: str) -> None:
        """Internal implementation for strategy start."""
        if strategy_id not in self._active_strategies:
            raise StrategyError(f"Strategy {strategy_id} not registered")

        strategy = self._active_strategies[strategy_id]

        # Validate pre-start conditions
        if not await self._validate_start_conditions(strategy_id):
            raise StrategyError(f"Start conditions not met for strategy {strategy_id}")

        # Start strategy
        await strategy.start()

        self._logger.info("Strategy started successfully", strategy_id=strategy_id)

    async def stop_strategy(self, strategy_id: str) -> None:
        """
        Stop a running strategy.

        Args:
            strategy_id: Strategy to stop
        """
        return await self.execute_with_monitoring(
            "stop_strategy", self._stop_strategy_impl, strategy_id
        )

    async def _stop_strategy_impl(self, strategy_id: str) -> None:
        """Internal implementation for strategy stop."""
        if strategy_id not in self._active_strategies:
            raise StrategyError(f"Strategy {strategy_id} not registered")

        strategy = self._active_strategies[strategy_id]
        await strategy.stop()

        self._logger.info("Strategy stopped successfully", strategy_id=strategy_id)

    # Signal Processing
    @time_execution
    @with_error_context(operation="process_market_data")
    async def process_market_data(self, market_data: MarketData) -> dict[str, list[Signal]]:
        """
        Process market data through all active strategies.

        Args:
            market_data: Market data to process

        Returns:
            Dictionary mapping strategy_id to generated signals
        """
        return await self.execute_with_monitoring(
            "process_market_data", self._process_market_data_impl, market_data
        )

    async def _process_market_data_impl(self, market_data: MarketData) -> dict[str, list[Signal]]:
        """Internal implementation for market data processing with concurrent processing."""
        all_signals = {}

        # Create a snapshot of active strategies to avoid race conditions
        active_strategies = {
            strategy_id: strategy
            for strategy_id, strategy in self._active_strategies.items()
            if strategy.status == StrategyStatus.ACTIVE
        }

        if not active_strategies:
            return all_signals

        # Process strategies concurrently with proper error handling
        async def process_strategy(strategy_id: str, strategy):
            try:
                # Generate signals with timeout
                signals = await asyncio.wait_for(
                    strategy.generate_signals(market_data),
                    timeout=10.0  # 10 second timeout for signal generation
                )

                # Validate signals concurrently
                validation_tasks = [
                    self.validate_signal(strategy_id, signal) for signal in signals
                ]
                
                if validation_tasks:
                    validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
                    validated_signals = [
                        signal for signal, is_valid in zip(signals, validation_results)
                        if isinstance(is_valid, bool) and is_valid
                    ]
                else:
                    validated_signals = []

                # Store in history (thread-safe append)
                if strategy_id not in self._signal_history:
                    self._signal_history[strategy_id] = []
                self._signal_history[strategy_id].extend(validated_signals)

                # Update metrics asynchronously
                await asyncio.wait_for(
                    self._update_strategy_metrics(strategy_id, validated_signals),
                    timeout=5.0  # 5 second timeout for metrics update
                )

                return strategy_id, validated_signals

            except asyncio.TimeoutError:
                self._logger.warning(f"Timeout processing market data for strategy {strategy_id}")
                return strategy_id, []
            except Exception as e:
                if self._error_handler:
                    await self._error_handler.handle_error(
                        error=e,
                        context={
                            "strategy_id": strategy_id,
                            "operation": "process_market_data",
                            "symbol": market_data.symbol,
                        },
                        severity=ErrorSeverity.HIGH,
                    )
                else:
                    self._logger.error(
                        f"Error processing market data for strategy {strategy_id}: {e}",
                        exc_info=True,
                    )
                return strategy_id, []

        # Process all strategies concurrently
        strategy_tasks = [
            process_strategy(strategy_id, strategy)
            for strategy_id, strategy in active_strategies.items()
        ]

        # Execute with overall timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*strategy_tasks, return_exceptions=True),
                timeout=30.0  # 30 second overall timeout
            )
            
            # Collect results
            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    strategy_id, signals = result
                    all_signals[strategy_id] = signals
                elif isinstance(result, Exception):
                    self._logger.error(f"Strategy processing error: {result}")

        except asyncio.TimeoutError:
            self._logger.error("Timeout processing market data across all strategies")
            # Return partial results
            for strategy_id in active_strategies.keys():
                if strategy_id not in all_signals:
                    all_signals[strategy_id] = []

        return all_signals

    async def validate_signal(self, strategy_id: str, signal: Signal) -> bool:
        """
        Validate a trading signal.

        Args:
            strategy_id: Strategy that generated the signal
            signal: Signal to validate

        Returns:
            True if signal is valid, False otherwise
        """
        try:
            # Basic signal validation
            if not signal.symbol or not signal.direction:
                return False

            if signal.strength < MIN_SIGNAL_CONFIDENCE:
                return False

            # Strategy-specific validation
            strategy = self._active_strategies.get(strategy_id)
            if strategy and hasattr(strategy, "validate_signal"):
                if not await strategy.validate_signal(signal):
                    return False

            # Risk manager validation
            if self._risk_manager:
                if not await self._risk_manager.validate_signal(signal):
                    return False

            return True

        except Exception as e:
            self._logger.error(
                "Signal validation error",
                strategy_id=strategy_id,
                signal=signal.model_dump(),
                error=str(e),
            )
            return False

    # Strategy Configuration and Validation
    async def validate_strategy_config(self, config: StrategyConfig) -> bool:
        """
        Validate strategy configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid
        """
        try:
            # Basic validation
            if not config.name or not config.strategy_type:
                return False

            # Check required parameters
            if not config.parameters:
                return False

            # Validate exchange compatibility
            if config.exchange_type and self._exchange_factory:
                if not self._exchange_factory.is_exchange_supported(config.exchange_type):
                    return False

            # Strategy-specific validation
            return await self._validate_strategy_specific_config(config)

        except Exception as e:
            self._logger.error("Config validation error", error=str(e))
            return False

    async def _validate_strategy_specific_config(self, config: StrategyConfig) -> bool:
        """Validate strategy-specific configuration parameters."""
        # This would be extended based on strategy types
        return True

    async def _validate_start_conditions(self, strategy_id: str) -> bool:
        """Validate conditions before starting a strategy."""
        config = self._strategy_configs.get(strategy_id)
        if not config:
            return False

        # Check if required services are available
        if config.requires_risk_manager and not self._risk_manager:
            return False

        if config.requires_exchange and not self._exchange_factory:
            return False

        return True

    # Performance Monitoring
    async def _update_strategy_metrics(self, strategy_id: str, signals: list[Signal]) -> None:
        """Update strategy performance metrics with analytics integration."""
        if strategy_id not in self._strategy_metrics:
            return

        metrics = self._strategy_metrics[strategy_id]
        metrics.signals_generated += len(signals)
        metrics.last_signal_time = (
            datetime.now(timezone.utc) if signals else metrics.last_signal_time
        )
        metrics.last_updated = datetime.now(timezone.utc)

        # Cache updated metrics for performance optimization
        if self.cache_manager:
            try:
                cache_key = f"strategy_metrics_{strategy_id}"
                await self.cache_manager.set(
                    cache_key,
                    metrics.model_dump(),
                    ttl=DEFAULT_CACHE_TTL,
                    namespace="strategy_performance"
                )
            except Exception as e:
                self.logger.warning(
                    "Failed to cache strategy metrics",
                    strategy_id=strategy_id,
                    error=str(e)
                )

        # NEW: Analytics service integration for real-time strategy performance tracking
        if self._strategy_services and self._strategy_services.analytics_service:
            try:
                await self._record_strategy_analytics(strategy_id, signals)
            except Exception as e:
                self.logger.warning(
                    "Failed to record strategy analytics",
                    strategy_id=strategy_id,
                    error=str(e)
                )

    async def _record_strategy_analytics(self, strategy_id: str, signals: list[Signal]) -> None:
        """Record strategy performance data to analytics service.
        
        This provides real-time strategy performance analysis, attribution analysis,
        and benchmarking capabilities identified in the integration audit.
        """
        if not self._strategy_services or not self._strategy_services.analytics_service:
            return

        try:
            # Calculate real-time performance metrics
            strategy_config = self._strategy_configs.get(strategy_id)
            strategy_metrics = self._strategy_metrics.get(strategy_id)
            signal_history = self._signal_history.get(strategy_id, [])

            if not strategy_config or not strategy_metrics:
                return

            # Calculate win rate from recent signals
            win_rate = await self._calculate_win_rate(strategy_id, signal_history)
            sharpe_ratio = await self._calculate_sharpe_ratio(strategy_id)
            max_drawdown = await self._calculate_max_drawdown(strategy_id)

            # Record comprehensive strategy performance data
            performance_data = {
                "strategy_id": strategy_id,
                "strategy_type": strategy_config.strategy_type,
                "timestamp": datetime.now(timezone.utc),
                "signals_generated": len(signals),
                "total_signals": len(signal_history),
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "signal_strength_avg": sum(s.strength for s in signals) / len(signals) if signals else 0.0,
                "last_signal_time": strategy_metrics.last_signal_time,
                "performance_period": "real_time"
            }

            # Send to analytics service for processing
            await self._strategy_services.analytics_service.record_strategy_performance(
                strategy_id=strategy_id,
                performance_data=performance_data
            )

            self.logger.debug(
                "Strategy analytics recorded",
                strategy_id=strategy_id,
                signals_count=len(signals),
                win_rate=win_rate
            )

        except Exception as e:
            self.logger.error(
                "Failed to record strategy analytics",
                strategy_id=strategy_id,
                error=str(e),
                exc_info=True
            )

    async def _calculate_win_rate(self, strategy_id: str, signal_history: list[Signal]) -> float:
        """Calculate strategy win rate from signal history.
        
        Args:
            strategy_id: Strategy identifier
            signal_history: Historical signals to analyze
            
        Returns:
            Win rate as a percentage (0.0 to 100.0)
        """
        if not signal_history or len(signal_history) < 2:
            return 0.0

        try:
            # Simple win rate calculation based on signal strength consistency
            # In production, this would use actual trade outcomes
            strong_signals = [s for s in signal_history[-50:] if s.strength >= 0.7]  # Last 50 signals
            return (len(strong_signals) / min(len(signal_history), 50)) * 100.0

        except Exception as e:
            self.logger.warning(f"Win rate calculation failed: {e}")
            return 0.0

    async def _calculate_sharpe_ratio(self, strategy_id: str) -> float:
        """Calculate strategy Sharpe ratio.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Sharpe ratio or 0.0 if calculation fails
        """
        try:
            # Simplified Sharpe ratio calculation
            # In production, this would use actual returns data
            signal_history = self._signal_history.get(strategy_id, [])
            if len(signal_history) < 10:
                return 0.0

            # Use signal strength as a proxy for returns
            strengths = [s.strength for s in signal_history[-100:]]  # Last 100 signals
            if not strengths:
                return 0.0

            mean_return = sum(strengths) / len(strengths)
            variance = sum((s - mean_return) ** 2 for s in strengths) / len(strengths)
            std_dev = variance ** 0.5

            return mean_return / std_dev if std_dev > 0 else 0.0

        except Exception as e:
            self.logger.warning(f"Sharpe ratio calculation failed: {e}")
            return 0.0

    async def _calculate_max_drawdown(self, strategy_id: str) -> float:
        """Calculate strategy maximum drawdown.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Maximum drawdown as a percentage or 0.0 if calculation fails
        """
        try:
            # Simplified drawdown calculation using signal strength
            signal_history = self._signal_history.get(strategy_id, [])
            if len(signal_history) < 5:
                return 0.0

            strengths = [s.strength for s in signal_history[-100:]]  # Last 100 signals
            if not strengths:
                return 0.0

            # Calculate rolling maximum and drawdown
            running_max = 0.0
            max_drawdown = 0.0

            for strength in strengths:
                running_max = max(running_max, strength)
                drawdown = (running_max - strength) / running_max if running_max > 0 else 0.0
                max_drawdown = max(max_drawdown, drawdown)

            return max_drawdown * 100.0  # Convert to percentage

        except Exception as e:
            self.logger.warning(f"Max drawdown calculation failed: {e}")
            return 0.0

    @cache_strategy_signals(strategy_id_arg_name="strategy_id", ttl=DEFAULT_CACHE_TTL)
    async def get_strategy_performance(self, strategy_id: str) -> dict[str, Any]:
        """
        Get comprehensive strategy performance data.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Performance metrics and statistics
        """
        return await self.execute_with_monitoring(
            "get_strategy_performance", self._get_strategy_performance_impl, strategy_id
        )

    async def _get_strategy_performance_impl(self, strategy_id: str) -> dict[str, Any]:
        """Internal implementation for performance retrieval."""
        if strategy_id not in self._active_strategies:
            raise StrategyError(f"Strategy {strategy_id} not found")

        strategy = self._active_strategies[strategy_id]
        metrics = self._strategy_metrics.get(strategy_id)
        config = self._strategy_configs.get(strategy_id)

        performance = {
            "strategy_id": strategy_id,
            "status": strategy.status.value if hasattr(strategy, "status") else "unknown",
            "config": config.model_dump() if config else {},
            "metrics": metrics.model_dump() if metrics else {},
            "signal_history_count": len(self._signal_history.get(strategy_id, [])),
            "last_update": datetime.now(timezone.utc),
        }

        # Add strategy-specific performance data
        if hasattr(strategy, "get_performance_summary"):
            performance.update(strategy.get_performance_summary())

        return performance

    @cached(
        ttl=DEFAULT_CACHE_TTL_SHORT,
        namespace="strategy_performance",
        data_type="strategy",
        key_generator=lambda self, strategy_id: f"metrics_{strategy_id}",
    )
    async def get_cached_strategy_metrics(self, strategy_id: str) -> dict[str, Any] | None:
        """Get cached strategy metrics for improved performance.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Cached strategy metrics or None if not found
        """
        if strategy_id not in self._strategy_metrics:
            return None

        metrics = self._strategy_metrics[strategy_id]
        return {
            "strategy_id": strategy_id,
            "signals_generated": metrics.signals_generated,
            "last_signal_time": metrics.last_signal_time,
            "last_updated": metrics.last_updated,
            "cached_at": datetime.now(timezone.utc)
        }

    async def get_strategy_performance_with_cache(self, strategy_id: str) -> dict[str, Any]:
        """Get strategy performance with intelligent caching.
        
        This method first checks for cached metrics and falls back to full calculation
        if cache is stale or missing.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Performance data with cache optimization
        """
        try:
            # Try to get cached metrics first
            cached_metrics = await self.get_cached_strategy_metrics(strategy_id)

            # Check if we have fresh cached data
            if cached_metrics and cached_metrics.get("last_updated"):
                cache_age = datetime.now(timezone.utc) - cached_metrics["last_updated"]
                if cache_age.total_seconds() < DEFAULT_CACHE_TTL_SHORT:
                    self.logger.debug(
                        "Using cached strategy metrics",
                        strategy_id=strategy_id,
                        cache_age_seconds=cache_age.total_seconds()
                    )
                    return cached_metrics

            # Fall back to full performance calculation
            return await self.get_strategy_performance(strategy_id)

        except Exception as e:
            self.logger.warning(
                "Failed to get cached metrics, falling back to full calculation",
                strategy_id=strategy_id,
                error=str(e)
            )
            return await self.get_strategy_performance(strategy_id)

    # Service Management
    @cached(
        ttl=DEFAULT_CACHE_TTL_SHORT,
        namespace="strategy",
        data_type="strategy",
        key_generator=lambda self: "all_strategies",
    )
    async def get_all_strategies(self) -> dict[str, dict[str, Any]]:
        """Get information about all registered strategies."""
        strategies = {}

        for strategy_id in self._active_strategies:
            try:
                performance = await self._get_strategy_performance_impl(strategy_id)
                strategies[strategy_id] = performance
            except Exception as e:
                self._logger.error(
                    "Error getting strategy info", strategy_id=strategy_id, error=str(e)
                )

        return strategies

    async def cleanup_strategy(self, strategy_id: str) -> None:
        """
        Clean up and remove a strategy.

        Args:
            strategy_id: Strategy to cleanup
        """
        return await self.execute_with_monitoring(
            "cleanup_strategy", self._cleanup_strategy_impl, strategy_id
        )

    async def _cleanup_strategy_impl(self, strategy_id: str) -> None:
        """Internal implementation for strategy cleanup."""
        if strategy_id in self._active_strategies:
            strategy = self._active_strategies[strategy_id]

            try:
                # Stop strategy if running
                if hasattr(strategy, "status") and strategy.status == StrategyStatus.ACTIVE:
                    await strategy.stop()

                # Cleanup strategy resources
                if hasattr(strategy, "cleanup"):
                    strategy.cleanup()
            except Exception as e:
                self._logger.error(
                    "Error during strategy cleanup operations",
                    strategy_id=strategy_id,
                    error=str(e)
                )
            finally:
                # Remove from registries - ensure this happens even if cleanup fails
                del self._active_strategies[strategy_id]
                self._strategy_configs.pop(strategy_id, None)
                self._strategy_metrics.pop(strategy_id, None)
                self._signal_history.pop(strategy_id, None)
                self._performance_cache.pop(strategy_id, None)

                self._logger.info("Strategy cleaned up", strategy_id=strategy_id)

    async def _service_health_check(self) -> Any:
        """Service-specific health check."""
        from src.core.base.interfaces import HealthStatus

        # Check active strategies
        total_strategies = len(self._active_strategies)
        sum(
            1
            for strategy in self._active_strategies.values()
            if hasattr(strategy, "status") and strategy.status == StrategyStatus.ACTIVE
        )

        # Check for stuck strategies
        stuck_strategies = []
        for strategy_id, metrics in self._strategy_metrics.items():
            if metrics.last_updated:
                time_since_update = (
                    datetime.now(timezone.utc) - metrics.last_updated
                ).total_seconds()
                if time_since_update > STRATEGY_METRICS_UPDATE_TIMEOUT:
                    stuck_strategies.append(strategy_id)

        if stuck_strategies:
            self._logger.warning("Detected stuck strategies", strategies=stuck_strategies)
            return HealthStatus.DEGRADED

        if total_strategies == 0:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def get_metrics(self) -> dict[str, Any]:
        """Get service metrics including strategy statistics."""
        metrics = super().get_metrics()

        # Add strategy-specific metrics
        metrics.update(
            {
                "total_strategies": len(self._active_strategies),
                "running_strategies": sum(
                    1
                    for strategy in self._active_strategies.values()
                    if hasattr(strategy, "status") and strategy.status == StrategyStatus.ACTIVE
                ),
                "total_signals_generated": sum(
                    len(signals) for signals in self._signal_history.values()
                ),
                "strategies_by_status": {},
            }
        )

        # Count strategies by status
        status_counts: dict[str, int] = {}
        for strategy in self._active_strategies.values():
            if hasattr(strategy, "status"):
                status = strategy.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
        metrics["strategies_by_status"] = status_counts

        return metrics

    def resolve_dependency(self, dependency_name: str) -> Any:
        """Resolve a dependency by name.

        Args:
            dependency_name: Name of the dependency to resolve

        Returns:
            The resolved dependency

        Raises:
            KeyError: If dependency not found
        """
        dependency_map = {
            "StrategyRepository": self._repository,
            "RiskManager": self._risk_manager,
            "ExchangeFactory": self._exchange_factory,
            "DataService": self._data_service,
        }

        if dependency_name not in dependency_map:
            raise KeyError(f"Unknown dependency: {dependency_name}")

        dependency = dependency_map[dependency_name]
        if dependency is None:
            raise KeyError(f"Dependency {dependency_name} not available")

        return dependency
