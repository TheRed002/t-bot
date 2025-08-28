"""
Refactored Base Strategy Interface - Enhanced architecture with service layers.

This module provides the enhanced base class that ALL strategies must inherit from.
The interface defines the contract that all strategy implementations must follow.

Enhanced features:
- Service layer integration
- Dependency injection
- Validation framework
- Backtesting interface
- Performance monitoring
- Metrics tracking

CRITICAL: All strategy implementations (P-012, P-013A-E, P-019) MUST inherit from this exact interface.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.base import BaseComponent

# MANDATORY: Import from P-001
from src.core.types import (
    MarketData,
    Position,
    Signal,
    StrategyConfig,
    StrategyMetrics,
    StrategyStatus,
    StrategyType,
)

# MANDATORY: Import from P-030 - Monitoring infrastructure
from src.monitoring import MetricsCollector, get_tracer

# Import alerting if available
try:
    from src.monitoring.alerting import (
        Alert,
        AlertSeverity,
        get_alert_manager,
    )

    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False
    Alert = None
    AlertSeverity = None

# MANDATORY: Import from P-002A - Error handling integration
from src.error_handling import (
    ErrorHandler,
    ErrorSeverity,
    get_global_error_handler,
    with_circuit_breaker,
    with_error_context,
    with_retry,
)
from src.error_handling.recovery_scenarios import (
    DataFeedInterruptionRecovery,
    OrderRejectionRecovery,
)

# MANDATORY: Import from P-003+ - Use exchange interfaces
from src.exchanges.base import BaseExchange

# MANDATORY: Import from P-008+ - Use risk management
from src.risk_management.base import BaseRiskManager

# New imports for refactored architecture
from src.strategies.interfaces import (
    BaseStrategyInterface,
)
from src.strategies.validation import ValidationFramework

# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution


class BaseStrategy(BaseComponent, BaseStrategyInterface, ABC):
    """Base strategy interface that ALL strategies must inherit from.

    Enhanced with:
    - Service layer integration
    - Dependency injection
    - Validation framework
    - Backtesting capabilities
    - Performance monitoring
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize strategy with configuration.

        Args:
            config: Strategy configuration dictionary
        """
        super().__init__()  # Initialize BaseComponent
        self.config = StrategyConfig(**config)
        self._name: str = self.config.name  # Use config name for unique identification
        self._version: str = "1.0.0"
        self._status: StrategyStatus = StrategyStatus.STOPPED
        self.metrics: StrategyMetrics = StrategyMetrics(strategy_id=self.config.strategy_id)

        # Service dependencies (injected)
        self._risk_manager: BaseRiskManager | None = None
        self._exchange: BaseExchange | None = None
        self._data_service: Any | None = None
        self._validation_framework: ValidationFramework | None = None
        self._error_handler: ErrorHandler = get_global_error_handler()
        self._metrics_collector: MetricsCollector | None = None

        # Recovery scenarios for trading operations
        from src.core.config import get_config

        recovery_config = get_config()
        self._order_rejection_recovery = OrderRejectionRecovery(recovery_config)
        self._data_feed_recovery = DataFeedInterruptionRecovery(recovery_config)

        # Performance monitoring (legacy - will be migrated to MetricsCollector)
        self._performance_metrics: dict[str, Any] = {
            "total_signals": 0,
            "valid_signals": 0,
            "execution_count": 0,
            "last_performance_update": datetime.now(timezone.utc),
        }

        # Backtesting mode
        self._is_backtesting: bool = False
        self._backtest_config: dict[str, Any] | None = None
        self._backtest_metrics: dict[str, Any] = {}

        # Signal history for analysis
        self._signal_history: list[Signal] = []
        self._max_signal_history = 1000

        # Logger is provided by BaseComponent via LoggerMixin
        self.logger.info(
            "Strategy initialized", strategy=self.name, config=self.config.model_dump()
        )

    @property
    @abstractmethod
    def strategy_type(self) -> StrategyType:
        """Get the strategy type."""
        pass

    @property
    def name(self) -> str:
        """Get the strategy name."""
        return self._name

    @property
    def version(self) -> str:
        """Get the strategy version."""
        return self._version

    @property
    def status(self) -> StrategyStatus:
        """Get the strategy status."""
        return self._status

    @abstractmethod
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
        """Internal signal generation implementation.

        Args:
            data: Market data for signal generation

        Returns:
            List of trading signals
        """
        pass

    @time_execution
    @with_error_context(operation="generate_signals")
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def generate_signals(self, data: MarketData) -> list[Signal]:
        """Generate trading signals from market data.

        MANDATORY: All implementations must:
        1. Validate input data
        2. Return empty list on errors (graceful degradation)
        3. Apply confidence thresholds
        4. Log signal generation events

        Args:
            data: Market data for signal generation

        Returns:
            List of trading signals
        """
        # Create tracing span if available
        tracer = get_tracer(__name__) if "get_tracer" in globals() else None
        span_context = None

        if tracer:
            span_context = tracer.start_as_current_span(
                "strategy.generate_signals",
                attributes={
                    "strategy.name": self.name,
                    "strategy.type": self.strategy_type.value,
                    "market.symbol": data.symbol if data else "unknown",
                },
            )
            span_context.__enter__()

        try:
            # Validate market data if validation framework is available
            if self._validation_framework:
                validation_result = await self._validation_framework.validate_market_conditions(
                    data
                )
                if not validation_result.is_valid:
                    self.logger.warning(
                        "Market conditions validation failed",
                        strategy=self.name,
                        errors=validation_result.errors,
                    )
                    if validation_result.errors:  # Hard failures
                        return []

            # Generate signals
            signals = await self._generate_signals_impl(data)

            # Validate and filter signals
            validated_signals = []
            for signal in signals:
                if await self._validate_and_process_signal(signal, data):
                    validated_signals.append(signal)

            # Update performance metrics
            self._performance_metrics["total_signals"] += len(signals)
            self._performance_metrics["valid_signals"] += len(validated_signals)

            # Update monitoring metrics if available
            if self._metrics_collector and hasattr(self._metrics_collector, "trading_metrics"):
                # Report total signals generated
                if hasattr(self._metrics_collector.trading_metrics, "signals_generated"):
                    self._metrics_collector.trading_metrics.signals_generated.labels(
                        strategy=self.name, strategy_type=self.strategy_type.value
                    ).inc(len(signals))

                # Report valid signals
                if hasattr(self._metrics_collector.trading_metrics, "signals_validated"):
                    self._metrics_collector.trading_metrics.signals_validated.labels(
                        strategy=self.name, strategy_type=self.strategy_type.value
                    ).inc(len(validated_signals))

            # Store signal history
            self._add_to_signal_history(validated_signals)

            return validated_signals

        except Exception as e:
            await self._error_handler.handle_error(
                error=e,
                context={
                    "strategy": self.name,
                    "operation": "generate_signals",
                    "symbol": data.symbol if data else "unknown",
                },
                severity=ErrorSeverity.HIGH,
            )
            # Record error in span if available
            if span_context and hasattr(span_context, "__exit__"):
                span = span_context.__self__ if hasattr(span_context, "__self__") else None
                if span and hasattr(span, "record_exception"):
                    span.record_exception(e)

            # Fire alert for critical errors if alerting is available
            if ALERTING_AVAILABLE and AlertSeverity:
                alert_manager = get_alert_manager()
                if alert_manager:
                    try:
                        alert = Alert(
                            name=f"strategy_error_{self.name}",
                            severity=AlertSeverity.HIGH,
                            description=f"Strategy {self.name} failed to generate signals: {e!s}",
                            labels={
                                "strategy": self.name,
                                "strategy_type": self.strategy_type.value,
                                "operation": "generate_signals",
                                "symbol": data.symbol if data else "unknown",
                                "error_type": type(e).__name__,
                            },
                        )
                        await alert_manager.fire_alert(alert)
                    except Exception as alert_error:
                        self.logger.error(
                            "Failed to fire alert", strategy=self.name, error=str(alert_error)
                        )

            return []  # Graceful degradation
        finally:
            # Close tracing span if available
            if span_context and hasattr(span_context, "__exit__"):
                span_context.__exit__(None, None, None)

    @abstractmethod
    async def validate_signal(self, signal: Signal) -> bool:
        """Validate signal before execution.

        MANDATORY: Check signal confidence, direction, timestamp

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid, False otherwise
        """
        pass

    async def _validate_and_process_signal(self, signal: Signal, market_data: MarketData) -> bool:
        """Internal method to validate and process signal with all frameworks.

        Args:
            signal: Signal to validate
            market_data: Current market data

        Returns:
            True if signal is valid and should be processed
        """
        try:
            # Strategy-specific validation
            if not await self.validate_signal(signal):
                return False

            # Framework validation if available
            if self._validation_framework:
                validation_result = await self._validation_framework.validate_signal(
                    signal, market_data
                )
                if not validation_result.is_valid:
                    self.logger.debug(
                        "Signal validation failed",
                        strategy=self.name,
                        signal_id=getattr(signal, "id", "unknown"),
                        errors=validation_result.errors,
                    )
                    return False

            return True

        except Exception as e:
            await self._error_handler.handle_error(
                error=e,
                context={
                    "strategy": self.name,
                    "operation": "validate_signal",
                    "signal_id": getattr(signal, "id", "unknown"),
                },
                severity=ErrorSeverity.MEDIUM,
            )
            return False

    @abstractmethod
    def get_position_size(self, signal: Signal) -> Decimal:
        """Calculate position size for signal.

        MANDATORY: Integrate with risk management

        Args:
            signal: Signal for position sizing

        Returns:
            Position size as Decimal
        """
        pass

    @abstractmethod
    def should_exit(self, position: Position, data: MarketData) -> bool:
        """Determine if position should be closed.

        MANDATORY: Check stop loss, take profit, time exits

        Args:
            position: Current position
            data: Current market data

        Returns:
            True if position should be closed, False otherwise
        """
        pass

    # Standard methods that can be overridden
    @with_retry(max_attempts=3, base_delay=1.0)
    async def pre_trade_validation(self, signal: Signal) -> bool:
        """Pre-trade validation hook.

        Args:
            signal: Signal to validate

        Returns:
            True if validation passes, False otherwise
        """
        if not await self.validate_signal(signal):
            self.logger.warning(
                "Signal validation failed", strategy=self.name, signal=signal.model_dump()
            )
            return False

        if self._risk_manager:
            risk_valid = await self._risk_manager.validate_signal(signal)
            if not risk_valid:
                self.logger.warning(
                    "Risk validation failed", strategy=self.name, signal=signal.model_dump()
                )
                return False

        return True

    async def post_trade_processing(self, trade_result: Any) -> None:
        """Post-trade processing hook.

        Args:
            trade_result: Result of trade execution
        """
        # Update metrics
        self.metrics.total_trades += 1
        self._performance_metrics["execution_count"] += 1

        # TODO: Remove in production - Debug logging
        self.logger.debug("Post-trade processing", strategy=self.name, trade_result=trade_result)

        # Log trade result
        if hasattr(trade_result, "pnl") and trade_result.pnl:
            if trade_result.pnl > 0:
                self.metrics.winning_trades += 1
            else:
                self.metrics.losing_trades += 1

            self.metrics.total_pnl += trade_result.pnl

            # Update win rate
            if self.metrics.total_trades > 0:
                self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades

            # Report to monitoring if available
            if self._metrics_collector and hasattr(self._metrics_collector, "trading_metrics"):
                # Report PnL
                if hasattr(self._metrics_collector.trading_metrics, "trade_pnl"):
                    self._metrics_collector.trading_metrics.trade_pnl.labels(
                        strategy=self.name, strategy_type=self.strategy_type.value
                    ).observe(float(trade_result.pnl))

                # Report trade completion
                if hasattr(self._metrics_collector.trading_metrics, "trades_completed"):
                    self._metrics_collector.trading_metrics.trades_completed.labels(
                        strategy=self.name,
                        strategy_type=self.strategy_type.value,
                        status="win" if trade_result.pnl > 0 else "loss",
                    ).inc()

        self.metrics.last_updated = datetime.now(timezone.utc)

    # Dependency injection methods
    def set_risk_manager(self, risk_manager: BaseRiskManager) -> None:
        """Set risk manager for strategy.

        Args:
            risk_manager: Risk manager instance
        """
        self._risk_manager = risk_manager
        self.logger.info("Risk manager set", strategy=self.name)

    def set_exchange(self, exchange: BaseExchange) -> None:
        """Set exchange for strategy.

        Args:
            exchange: Exchange instance
        """
        self._exchange = exchange
        self.logger.info("Exchange set", strategy=self.name, exchange=exchange.__class__.__name__)

    def set_data_service(self, data_service: Any) -> None:
        """Set data service for strategy.

        Args:
            data_service: Data service instance
        """
        self._data_service = data_service
        self.logger.info("Data service set", strategy=self.name)

    def set_validation_framework(self, validation_framework: ValidationFramework) -> None:
        """Set validation framework for strategy.

        Args:
            validation_framework: Validation framework instance
        """
        self._validation_framework = validation_framework
        self.logger.info("Validation framework set", strategy=self.name)

    def set_metrics_collector(self, metrics_collector: MetricsCollector) -> None:
        """Set metrics collector for strategy monitoring.

        Args:
            metrics_collector: MetricsCollector instance
        """
        self._metrics_collector = metrics_collector
        self.logger.info("Metrics collector set", strategy=self.name)

    def get_strategy_info(self) -> dict[str, Any]:
        """Get strategy information.

        Returns:
            Dictionary with strategy information
        """
        return {
            "name": self.name,
            "type": self.strategy_type.value,
            "version": self.version,
            "status": self.status.value,
            "config": self.config.model_dump(),
            "metrics": self.metrics.model_dump(),
        }

    async def initialize(self, config: StrategyConfig) -> None:
        """Initialize the strategy with configuration.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.name = config.name
        await self._on_initialize()
        self.logger.info("Strategy initialized", strategy=self.name)

    @with_error_context(operation="start_strategy")
    async def start(self) -> None:
        """Start the strategy."""
        self._status = StrategyStatus.STARTING
        self.logger.info("Starting strategy", strategy=self.name)

        try:
            await self._on_start()
            self._status = StrategyStatus.RUNNING
            self.logger.info("Strategy started successfully", strategy=self.name)
        except Exception as e:
            self._status = StrategyStatus.ERROR
            await self._error_handler.handle_error(
                error=e,
                context={
                    "strategy": self.name,
                    "operation": "start",
                    "previous_status": "STARTING",
                },
                severity=ErrorSeverity.CRITICAL,
            )

            # Fire critical alert for strategy start failure
            if ALERTING_AVAILABLE and AlertSeverity:
                alert_manager = get_alert_manager()
                if alert_manager:
                    try:
                        alert = Alert(
                            name=f"strategy_start_failed_{self.name}",
                            severity=AlertSeverity.CRITICAL,
                            description=f"Strategy {self.name} failed to start: {e!s}",
                            labels={
                                "strategy": self.name,
                                "strategy_type": self.strategy_type.value,
                                "operation": "start",
                                "error_type": type(e).__name__,
                            },
                        )
                        await alert_manager.fire_alert(alert)
                    except (AttributeError, ImportError, KeyError) as e:
                        # Alert system not available or misconfigured - continue
                        pass
                    except Exception as e:
                        # Unexpected alerting error - don't fail start due to alerting issues
                        pass

            raise

    @with_error_context(operation="stop_strategy")
    async def stop(self) -> None:
        """Stop the strategy."""
        self._status = StrategyStatus.STOPPED
        self.logger.info("Stopping strategy", strategy=self.name)

        try:
            await self._on_stop()
            self.logger.info("Strategy stopped successfully", strategy=self.name)
        except Exception as e:
            await self._error_handler.handle_error(
                error=e,
                context={
                    "strategy": self.name,
                    "operation": "stop",
                },
                severity=ErrorSeverity.HIGH,
            )
            raise

    async def pause(self) -> None:
        """Pause the strategy."""
        if self._status == StrategyStatus.RUNNING:
            self._status = StrategyStatus.PAUSED
            self.logger.info("Strategy paused", strategy=self.name)

    async def resume(self) -> None:
        """Resume the strategy."""
        if self._status == StrategyStatus.PAUSED:
            self._status = StrategyStatus.RUNNING
            self.logger.info("Strategy resumed", strategy=self.name)

    # Backtesting interface implementation
    async def prepare_for_backtest(self, config: dict[str, Any]) -> None:
        """Prepare strategy for backtesting mode.

        Args:
            config: Backtesting configuration
        """
        self._is_backtesting = True
        self._backtest_config = config
        self._backtest_metrics = {
            "signals_generated": 0,
            "signals_executed": 0,
            "start_time": datetime.now(timezone.utc),
        }
        await self._on_backtest_prepare()
        self.logger.info("Strategy prepared for backtesting", strategy=self.name)

    async def process_historical_data(self, data: MarketData) -> list[Signal]:
        """Process historical data during backtesting.

        Args:
            data: Historical market data

        Returns:
            List of generated signals
        """
        if not self._is_backtesting:
            raise RuntimeError("Strategy not in backtesting mode")

        signals = await self.generate_signals(data)
        self._backtest_metrics["signals_generated"] += len(signals)
        return signals

    async def get_backtest_metrics(self) -> dict[str, Any]:
        """Get strategy-specific backtest metrics.

        Returns:
            Dictionary with backtest metrics
        """
        return {
            **self._backtest_metrics,
            "performance_metrics": self._performance_metrics,
            "signal_history_count": len(self._signal_history),
        }

    # Performance monitoring interface
    def get_real_time_metrics(self) -> dict[str, Any]:
        """Get real-time performance metrics.

        Returns:
            Current performance metrics
        """
        return {
            **self._performance_metrics,
            "strategy_status": self.status.value,
            "config_summary": {
                "name": self.config.name,
                "type": self.strategy_type.value,
                "id": self.config.strategy_id,
            },
            "dependencies_available": {
                "risk_manager": self._risk_manager is not None,
                "exchange": self._exchange is not None,
                "data_service": self._data_service is not None,
                "validation_framework": self._validation_framework is not None,
            },
        }

    def _add_to_signal_history(self, signals: list[Signal]) -> None:
        """Add signals to history for analysis.

        Args:
            signals: Signals to add to history
        """
        self._signal_history.extend(signals)

        # Maintain history size limit
        if len(self._signal_history) > self._max_signal_history:
            self._signal_history = self._signal_history[-self._max_signal_history :]

    # Optional lifecycle hooks that can be overridden
    async def _on_initialize(self) -> None:
        """Called during strategy initialization. Override for custom setup."""
        pass

    async def _on_start(self) -> None:
        """Called when strategy starts. Override for custom initialization."""
        pass

    async def _on_stop(self) -> None:
        """Called when strategy stops. Override for custom cleanup."""
        pass

    async def _on_backtest_prepare(self) -> None:
        """Called when preparing for backtesting. Override for custom setup."""
        pass

    def update_config(self, new_config: dict[str, Any]) -> None:
        """Update strategy configuration.

        Args:
            new_config: New configuration dictionary
        """
        old_config = self.config.model_dump()
        self.config = StrategyConfig(**new_config)
        self.logger.info(
            "Strategy config updated",
            strategy=self.name,
            old_config=old_config,
            new_config=self.config.model_dump(),
        )

    async def get_state(self) -> dict[str, Any]:
        """Get current strategy state.

        Returns:
            Dictionary containing strategy state information
        """
        return {
            "strategy_id": self.config.strategy_id,
            "strategy_type": self.strategy_type.value,
            "name": self.name,
            "status": self.status.value,
            "version": self.version,
            "parameters": self.config.parameters,
            "metrics": self.metrics.model_dump() if hasattr(self.metrics, "model_dump") else {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary for the strategy.

        Returns:
            Performance summary dictionary
        """
        return {
            "strategy_name": self.name,
            "strategy_type": self.strategy_type.value,
            "status": self.status.value,
            "total_trades": self.metrics.total_trades,
            "winning_trades": self.metrics.winning_trades,
            "losing_trades": self.metrics.losing_trades,
            "win_rate": self.metrics.win_rate,
            "total_pnl": float(self.metrics.total_pnl),
            "sharpe_ratio": self.metrics.sharpe_ratio,
            "max_drawdown": self.metrics.max_drawdown,
            "last_updated": self.metrics.last_updated.isoformat(),
            "performance_metrics": self._performance_metrics,
            "signal_stats": {
                "total_generated": self._performance_metrics.get("total_signals", 0),
                "valid_signals": self._performance_metrics.get("valid_signals", 0),
                "validation_rate": (
                    self._performance_metrics.get("valid_signals", 0)
                    / max(self._performance_metrics.get("total_signals", 1), 1)
                ),
                "history_count": len(self._signal_history),
            },
        }

    def cleanup(self) -> None:
        """Cleanup strategy resources."""
        try:
            if self._status == StrategyStatus.RUNNING:
                # Note: This should be await self.stop() in async context
                # For now, just change status
                self._status = StrategyStatus.STOPPED

            # Clear signal history
            self._signal_history.clear()

            # Reset metrics
            self._performance_metrics = {
                "total_signals": 0,
                "valid_signals": 0,
                "execution_count": 0,
                "last_performance_update": datetime.now(timezone.utc),
            }

            self.logger.info("Strategy cleanup completed", strategy=self.name)
        except Exception as e:
            # Use synchronous error logging during cleanup
            self._error_handler.logger.error(
                "Error during strategy cleanup",
                strategy=self.name,
                error=str(e),
                extra={"severity": ErrorSeverity.MEDIUM.value},
            )
        finally:
            super().cleanup()  # Call parent cleanup
