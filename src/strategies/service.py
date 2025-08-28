"""
Strategy Service - Business logic layer for strategy operations.

This module provides the service layer for strategy management, including:
- Strategy lifecycle management
- Signal processing and validation
- Performance monitoring
- Risk integration
- Backtesting coordination
"""

from datetime import datetime, timezone
from typing import Any

from src.core.base.service import BaseService
from src.core.caching import (
    cache_strategy_signals,
    cached,
    get_cache_manager,
)
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
from src.utils.decorators import time_execution


class StrategyService(BaseService):
    """
    Service layer for strategy operations and management.

    Provides business logic for:
    - Strategy lifecycle management
    - Signal generation and validation
    - Performance monitoring
    - Risk management integration
    - Backtesting coordination
    """

    def __init__(self, name: str = "StrategyService", config: dict[str, Any] | None = None):
        """
        Initialize strategy service.

        Args:
            name: Service name
            config: Service configuration
        """
        super().__init__(name, config)

        # Strategy registry
        self._active_strategies: dict[str, Any] = {}
        self._strategy_configs: dict[str, StrategyConfig] = {}
        self._strategy_metrics: dict[str, StrategyMetrics] = {}

        # Service dependencies (will be injected)
        self._risk_manager = None
        self._exchange_factory = None
        self._data_service = None
        self._backtesting_engine = None
        self._error_handler: ErrorHandler = get_global_error_handler()

        # Initialize cache manager
        self.cache_manager = get_cache_manager(config=config)

        # Strategy performance tracking
        self._performance_cache: dict[str, dict[str, Any]] = {}
        self._signal_history: dict[str, list[Signal]] = {}

        self._logger.info("StrategyService initialized")

    async def _do_start(self) -> None:
        """Initialize service dependencies."""
        try:
            # Resolve dependencies if available
            try:
                self._risk_manager = self.resolve_dependency("RiskService")
            except (KeyError, AttributeError, ImportError) as e:
                self._logger.warning("RiskService not available", error=str(e))
                self._risk_manager = None
            except Exception as e:
                self._logger.warning("Unexpected error resolving RiskService", error=str(e))
                self._risk_manager = None

            try:
                self._exchange_factory = self.resolve_dependency("ExchangeFactory")
            except (KeyError, AttributeError, ImportError) as e:
                self._logger.warning("ExchangeFactory not available", error=str(e))
                self._exchange_factory = None
            except Exception as e:
                self._logger.warning("Unexpected error resolving ExchangeFactory", error=str(e))
                self._exchange_factory = None

            try:
                self._data_service = self.resolve_dependency("DataService")
            except (KeyError, AttributeError, ImportError) as e:
                self._logger.warning("DataService not available", error=str(e))
                self._data_service = None
            except Exception as e:
                self._logger.warning("Unexpected error resolving DataService", error=str(e))
                self._data_service = None

            # Optional dependencies
            try:
                self._backtesting_engine = self.resolve_dependency("BacktestService")
            except (KeyError, AttributeError, ImportError) as e:
                self._logger.warning("BacktestService not available", error=str(e))
                self._backtesting_engine = None
            except Exception as e:
                self._logger.warning("Unexpected error resolving BacktestService", error=str(e))
                self._backtesting_engine = None

            self._logger.info("Strategy service dependencies resolved")

        except Exception as e:
            raise ServiceError(f"Failed to initialize strategy service: {e}")

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

        # Initialize strategy dependencies
        if self._risk_manager:
            strategy_instance.set_risk_manager(self._risk_manager)

        if self._exchange_factory:
            # Get appropriate exchange for strategy
            exchange = await self._exchange_factory.get_exchange(config.exchange_type)
            strategy_instance.set_exchange(exchange)

        # Register strategy
        self._active_strategies[strategy_id] = strategy_instance
        self._strategy_configs[strategy_id] = config
        self._strategy_metrics[strategy_id] = StrategyMetrics(strategy_id=strategy_id)
        self._signal_history[strategy_id] = []

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
        """Internal implementation for market data processing."""
        all_signals = {}

        # Process data through each active strategy
        for strategy_id, strategy in self._active_strategies.items():
            if strategy.status != StrategyStatus.RUNNING:
                continue

            try:
                # Generate signals
                signals = await strategy.generate_signals(market_data)

                # Validate signals
                validated_signals = []
                for signal in signals:
                    if await self.validate_signal(strategy_id, signal):
                        validated_signals.append(signal)

                all_signals[strategy_id] = validated_signals

                # Store in history
                self._signal_history[strategy_id].extend(validated_signals)

                # Update metrics
                await self._update_strategy_metrics(strategy_id, validated_signals)

            except Exception as e:
                await self._error_handler.handle_error(
                    error=e,
                    context={
                        "strategy_id": strategy_id,
                        "operation": "process_market_data",
                        "symbol": market_data.symbol,
                    },
                    severity=ErrorSeverity.HIGH,
                )
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

            if signal.confidence < 0.1:  # Minimum confidence threshold
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
        """Update strategy performance metrics."""
        if strategy_id not in self._strategy_metrics:
            return

        metrics = self._strategy_metrics[strategy_id]
        metrics.signals_generated += len(signals)
        metrics.last_signal_time = (
            datetime.now(timezone.utc) if signals else metrics.last_signal_time
        )
        metrics.last_updated = datetime.now(timezone.utc)

    @cache_strategy_signals(strategy_id_arg_name="strategy_id", ttl=300)  # Cache for 5 minutes
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

    # Backtesting Integration
    async def run_backtest(
        self, strategy_id: str, backtest_config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Run backtest for a strategy.

        Args:
            strategy_id: Strategy to backtest
            backtest_config: Backtest configuration

        Returns:
            Backtest results
        """
        return await self.execute_with_monitoring(
            "run_backtest", self._run_backtest_impl, strategy_id, backtest_config
        )

    async def _run_backtest_impl(
        self, strategy_id: str, backtest_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Internal implementation for backtest execution."""
        if not self._backtesting_engine:
            raise ServiceError("Backtesting engine not available")

        if strategy_id not in self._active_strategies:
            raise StrategyError(f"Strategy {strategy_id} not found")

        strategy = self._active_strategies[strategy_id]

        # Configure backtest
        from src.backtesting.engine import BacktestConfig

        config = BacktestConfig(**backtest_config)

        # Run backtest
        from src.backtesting.engine import BacktestEngine

        engine = BacktestEngine(config=config, strategy=strategy, risk_manager=self._risk_manager)

        result = await engine.run()

        self._logger.info(
            "Backtest completed",
            strategy_id=strategy_id,
            total_return=result.total_return,
            sharpe_ratio=result.sharpe_ratio,
        )

        return result.model_dump()

    # Service Management
    @cached(
        ttl=30,
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

            # Stop strategy if running
            if hasattr(strategy, "status") and strategy.status == StrategyStatus.RUNNING:
                await strategy.stop()

            # Cleanup strategy resources
            if hasattr(strategy, "cleanup"):
                strategy.cleanup()

            # Remove from registries
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
            if hasattr(strategy, "status") and strategy.status == StrategyStatus.RUNNING
        )

        # Check for stuck strategies
        stuck_strategies = []
        for strategy_id, metrics in self._strategy_metrics.items():
            if metrics.last_updated:
                time_since_update = (
                    datetime.now(timezone.utc) - metrics.last_updated
                ).total_seconds()
                if time_since_update > 300:  # 5 minutes
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
                    if hasattr(strategy, "status") and strategy.status == StrategyStatus.RUNNING
                ),
                "total_signals_generated": sum(
                    len(signals) for signals in self._signal_history.values()
                ),
                "strategies_by_status": {},
            }
        )

        # Count strategies by status
        status_counts = {}
        for strategy in self._active_strategies.values():
            if hasattr(strategy, "status"):
                status = strategy.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
        metrics["strategies_by_status"] = status_counts

        return metrics
