"""Risk monitoring with observer pattern to eliminate duplication."""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from src.core.dependency_injection import injectable
from src.core.types.risk import RiskAlert, RiskLevel, RiskMetrics
from src.utils.decorators import UnifiedDecorator as dec

# Module level logger
logger = logging.getLogger(__name__)


class RiskEvent(Enum):
    """Risk event types."""

    DRAWDOWN_EXCEEDED = "drawdown_exceeded"
    VAR_EXCEEDED = "var_exceeded"
    LOSS_LIMIT_REACHED = "loss_limit_reached"
    POSITION_LIMIT_REACHED = "position_limit_reached"
    CORRELATION_HIGH = "correlation_high"
    RISK_LEVEL_CHANGED = "risk_level_changed"
    LIQUIDITY_LOW = "liquidity_low"
    VOLATILITY_SPIKE = "volatility_spike"


class RiskObserver(ABC):
    """Abstract base class for risk observers."""

    @abstractmethod
    async def on_risk_event(self, event: RiskEvent, data: dict[str, Any]) -> None:
        """
        Handle risk event.

        Args:
            event: Type of risk event
            data: Event data
        """
        pass


class LoggingObserver(RiskObserver):
    """Observer that logs risk events."""

    def __init__(self):
        self._logger = logger

    async def on_risk_event(self, event: RiskEvent, data: dict[str, Any]) -> None:
        """Log risk event."""
        self._logger.warning(f"Risk event: {event.value}", event_data=data)


class AlertingObserver(RiskObserver):
    """Observer that creates alerts for risk events."""

    def __init__(self, alert_callback: Callable | None = None):
        """
        Initialize alerting observer.

        Args:
            alert_callback: Optional callback for alerts
        """
        self.alert_callback = alert_callback
        self.alerts: list[RiskAlert] = []
        self._logger = logger

    async def on_risk_event(self, event: RiskEvent, data: dict[str, Any]) -> None:
        """Create alert for risk event."""
        alert = RiskAlert(
            timestamp=datetime.utcnow(),
            event_type=event.value,
            severity=self._determine_severity(event),
            message=self._format_message(event, data),
            data=data,
        )

        self.alerts.append(alert)

        # Execute callback if provided
        if self.alert_callback:
            await self.alert_callback(alert)

        self._logger.info(f"Risk alert created: {alert.message}")

    def _determine_severity(self, event: RiskEvent) -> str:
        """Determine alert severity based on event type."""
        critical_events = {RiskEvent.LOSS_LIMIT_REACHED, RiskEvent.DRAWDOWN_EXCEEDED}

        high_events = {
            RiskEvent.VAR_EXCEEDED,
            RiskEvent.POSITION_LIMIT_REACHED,
            RiskEvent.LIQUIDITY_LOW,
        }

        if event in critical_events:
            return "CRITICAL"
        elif event in high_events:
            return "HIGH"
        else:
            return "MEDIUM"

    def _format_message(self, event: RiskEvent, data: dict[str, Any]) -> str:
        """Format alert message."""
        messages = {
            RiskEvent.DRAWDOWN_EXCEEDED: f"Drawdown {data.get('drawdown', 0):.2%} exceeds limit",
            RiskEvent.VAR_EXCEEDED: f"VaR {data.get('var', 0):.2%} exceeded",
            RiskEvent.LOSS_LIMIT_REACHED: f"Loss limit {data.get('loss', 0):.2%} reached",
            RiskEvent.POSITION_LIMIT_REACHED: f"Position limit reached: {data.get('positions', 0)} positions",
            RiskEvent.CORRELATION_HIGH: f"Correlation {data.get('correlation', 0):.2f} is high",
            RiskEvent.RISK_LEVEL_CHANGED: f"Risk level changed to {data.get('new_level', 'UNKNOWN')}",
            RiskEvent.LIQUIDITY_LOW: f"Low liquidity detected: {data.get('liquidity_ratio', 0):.2f}",
            RiskEvent.VOLATILITY_SPIKE: f"Volatility spike detected: {data.get('volatility', 0):.2%}",
        }

        return messages.get(event, f"Risk event: {event.value}")


class CircuitBreakerObserver(RiskObserver):
    """Observer that implements circuit breaker functionality."""

    def __init__(self, break_callback: Callable | None = None):
        """
        Initialize circuit breaker observer.

        Args:
            break_callback: Callback when circuit breaks
        """
        self.break_callback = break_callback
        self.is_broken = False
        self.break_time: datetime | None = None
        self.break_duration = timedelta(minutes=15)
        self._logger = logger

    async def on_risk_event(self, event: RiskEvent, data: dict[str, Any]) -> None:
        """Handle circuit breaker logic."""
        # Events that trigger circuit breaker
        break_events = {RiskEvent.LOSS_LIMIT_REACHED, RiskEvent.DRAWDOWN_EXCEEDED}

        if event in break_events and not self.is_broken:
            await self._trigger_circuit_breaker(event, data)

        # Check if circuit breaker should reset
        if self.is_broken and self.break_time:
            if datetime.utcnow() - self.break_time > self.break_duration:
                await self._reset_circuit_breaker()

    async def _trigger_circuit_breaker(self, event: RiskEvent, data: dict[str, Any]) -> None:
        """Trigger circuit breaker."""
        self.is_broken = True
        self.break_time = datetime.utcnow()

        self._logger.critical(f"Circuit breaker triggered by {event.value}", data=data)

        if self.break_callback:
            await self.break_callback(event, data)

    async def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker."""
        self.is_broken = False
        self.break_time = None
        self._logger.info("Circuit breaker reset")


@injectable(singleton=True)
class RiskMonitor:
    """
    Centralized risk monitor with observer pattern.

    This eliminates duplication of risk monitoring logic across modules.
    """

    def __init__(self):
        """Initialize risk monitor."""
        self._observers: set[RiskObserver] = set()
        self._thresholds: dict[str, Decimal] = {}
        self._metrics_history: list[RiskMetrics] = []
        self._last_risk_level = RiskLevel.LOW
        self._monitoring_task: asyncio.Task | None = None
        self._running = False
        self._logger = logger

        # Default observers
        self.add_observer(LoggingObserver())
        self.add_observer(AlertingObserver())
        self.add_observer(CircuitBreakerObserver())

        # Default thresholds
        self._set_default_thresholds()

    def _set_default_thresholds(self) -> None:
        """Set default monitoring thresholds."""
        self._thresholds = {
            "max_drawdown": Decimal("0.2"),
            "max_var": Decimal("0.1"),
            "max_daily_loss": Decimal("0.05"),
            "max_correlation": Decimal("0.7"),
            "min_liquidity": Decimal("0.2"),
            "max_positions": Decimal("10"),
            "volatility_spike": Decimal("2.0"),  # 2x normal volatility
        }

    def add_observer(self, observer: RiskObserver) -> None:
        """
        Add a risk observer.

        Args:
            observer: Observer to add
        """
        self._observers.add(observer)
        self._logger.debug(f"Added risk observer: {observer.__class__.__name__}")

    def remove_observer(self, observer: RiskObserver) -> None:
        """
        Remove a risk observer.

        Args:
            observer: Observer to remove
        """
        self._observers.discard(observer)

    async def notify_observers(self, event: RiskEvent, data: dict[str, Any]) -> None:
        """
        Notify all observers of a risk event.

        Args:
            event: Risk event type
            data: Event data
        """
        for observer in self._observers:
            try:
                await observer.on_risk_event(event, data)
            except Exception as e:
                self._logger.error(f"Observer error: {e}")

    @dec.enhance(log=True, monitor=True)
    async def monitor_metrics(self, metrics: RiskMetrics) -> None:
        """
        Monitor risk metrics and trigger events.

        Args:
            metrics: Current risk metrics
        """
        # Store metrics history
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > 100:
            self._metrics_history = self._metrics_history[-100:]

        # Check drawdown
        if metrics.current_drawdown > self._thresholds["max_drawdown"]:
            await self.notify_observers(
                RiskEvent.DRAWDOWN_EXCEEDED, {"drawdown": float(metrics.current_drawdown)}
            )

        # Check VaR
        if metrics.var_1d > self._thresholds["max_var"]:
            await self.notify_observers(RiskEvent.VAR_EXCEEDED, {"var": float(metrics.var_1d)})

        # Check correlation
        if metrics.correlation_risk > self._thresholds["max_correlation"]:
            await self.notify_observers(
                RiskEvent.CORRELATION_HIGH, {"correlation": float(metrics.correlation_risk)}
            )

        # Check risk level change
        if metrics.risk_level != self._last_risk_level:
            await self.notify_observers(
                RiskEvent.RISK_LEVEL_CHANGED,
                {"old_level": self._last_risk_level.value, "new_level": metrics.risk_level.value},
            )
            self._last_risk_level = metrics.risk_level

        # Check volatility spike
        if len(self._metrics_history) > 10:
            recent_volatility = self._calculate_recent_volatility()
            historical_volatility = self._calculate_historical_volatility()

            if recent_volatility > historical_volatility * float(
                self._thresholds["volatility_spike"]
            ):
                await self.notify_observers(
                    RiskEvent.VOLATILITY_SPIKE,
                    {"volatility": recent_volatility, "historical": historical_volatility},
                )

    def _calculate_recent_volatility(self) -> float:
        """Calculate recent volatility from metrics history."""
        if len(self._metrics_history) < 5:
            return 0.0

        recent_values = [float(m.portfolio_value) for m in self._metrics_history[-5:]]

        import numpy as np

        returns = np.diff(recent_values) / recent_values[:-1]
        return np.std(returns) if len(returns) > 0 else 0.0

    def _calculate_historical_volatility(self) -> float:
        """Calculate historical volatility from metrics history."""
        if len(self._metrics_history) < 20:
            return 0.01  # Default low volatility

        values = [float(m.portfolio_value) for m in self._metrics_history]

        import numpy as np

        returns = np.diff(values) / values[:-1]
        return np.std(returns) if len(returns) > 0 else 0.01

    async def monitor_portfolio(self, portfolio_data: dict[str, Any]) -> None:
        """
        Monitor portfolio-level risks.

        Args:
            portfolio_data: Portfolio data
        """
        # Check position count
        position_count = len(portfolio_data.get("positions", []))
        if position_count >= self._thresholds["max_positions"]:
            await self.notify_observers(
                RiskEvent.POSITION_LIMIT_REACHED, {"positions": position_count}
            )

        # Check daily loss
        daily_pnl = Decimal(str(portfolio_data.get("daily_pnl", 0)))
        total_value = Decimal(str(portfolio_data.get("total_value", 1)))

        if daily_pnl < 0:
            daily_loss = abs(daily_pnl) / total_value
            if daily_loss > self._thresholds["max_daily_loss"]:
                await self.notify_observers(
                    RiskEvent.LOSS_LIMIT_REACHED, {"loss": float(daily_loss)}
                )

        # Check liquidity
        liquidity_ratio = Decimal(str(portfolio_data.get("liquidity_ratio", 1)))
        if liquidity_ratio < self._thresholds["min_liquidity"]:
            await self.notify_observers(
                RiskEvent.LIQUIDITY_LOW, {"liquidity_ratio": float(liquidity_ratio)}
            )

    def set_threshold(self, key: str, value: Decimal) -> None:
        """
        Set a monitoring threshold.

        Args:
            key: Threshold key
            value: Threshold value
        """
        self._thresholds[key] = value
        self._logger.info(f"Set threshold {key} = {value}")

    def get_thresholds(self) -> dict[str, Decimal]:
        """Get current thresholds."""
        return self._thresholds.copy()

    async def start_monitoring(self, interval: int = 60) -> None:
        """
        Start continuous monitoring.

        Args:
            interval: Monitoring interval in seconds
        """
        if self._running:
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
        self._logger.info(f"Started risk monitoring with {interval}s interval")

    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        self._logger.info("Stopped risk monitoring")

    async def _monitoring_loop(self, interval: int) -> None:
        """Continuous monitoring loop."""
        while self._running:
            try:
                # This would fetch current metrics from the system
                # For now, it's a placeholder
                await asyncio.sleep(interval)
            except Exception as e:
                self._logger.error(f"Monitoring error: {e}")

    def get_alerts(self) -> list[RiskAlert]:
        """Get recent risk alerts."""
        for observer in self._observers:
            if isinstance(observer, AlertingObserver):
                return observer.alerts[-100:]  # Last 100 alerts
        return []
