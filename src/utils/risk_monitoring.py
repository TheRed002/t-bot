"""
Centralized risk monitoring utilities to eliminate observer pattern duplication.

This module provides unified risk monitoring patterns, eliminating duplication
across multiple risk monitoring implementations.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from src.core.logging import get_logger
from src.core.types.risk import RiskAlert, RiskLevel, RiskMetrics
from src.utils.decimal_utils import ZERO, safe_divide, to_decimal

logger = get_logger(__name__)


class RiskEventType(Enum):
    """Standardized risk event types."""

    DRAWDOWN_EXCEEDED = "drawdown_exceeded"
    VAR_EXCEEDED = "var_exceeded"
    LOSS_LIMIT_REACHED = "loss_limit_reached"
    POSITION_LIMIT_REACHED = "position_limit_reached"
    CORRELATION_HIGH = "correlation_high"
    RISK_LEVEL_CHANGED = "risk_level_changed"
    LIQUIDITY_LOW = "liquidity_low"
    VOLATILITY_SPIKE = "volatility_spike"
    EMERGENCY_STOP_TRIGGERED = "emergency_stop_triggered"
    CIRCUIT_BREAKER_OPENED = "circuit_breaker_opened"


class RiskEventSeverity(Enum):
    """Risk event severity levels."""

    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class RiskEvent:
    """Standardized risk event structure."""

    def __init__(
        self,
        event_type: RiskEventType,
        severity: RiskEventSeverity,
        message: str,
        data: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ):
        """
        Initialize risk event.

        Args:
            event_type: Type of risk event
            severity: Event severity level
            message: Human-readable message
            data: Additional event data
            timestamp: Event timestamp (defaults to now)
        """
        self.event_type = event_type
        self.severity = severity
        self.message = message
        self.data = data or {}
        self.timestamp = timestamp or datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


class RiskObserver(ABC):
    """Abstract base for risk event observers."""

    @abstractmethod
    async def handle_event(self, event: RiskEvent) -> None:
        """
        Handle a risk event.

        Args:
            event: Risk event to handle
        """
        pass

    @abstractmethod
    def get_observer_id(self) -> str:
        """Get unique observer identifier."""
        pass


class LoggingRiskObserver(RiskObserver):
    """Observer that logs risk events."""

    def __init__(self, observer_id: str = "logging_observer") -> None:
        """
        Initialize logging observer.

        Args:
            observer_id: Unique identifier for this observer
        """
        self.observer_id = observer_id
        self._logger = logger

    async def handle_event(self, event: RiskEvent) -> None:
        """Log the risk event with appropriate level."""
        log_data = {
            "event_type": event.event_type.value,
            "event_data": event.data,
            "timestamp": event.timestamp.isoformat(),
        }

        if event.severity == RiskEventSeverity.CRITICAL:
            self._logger.critical(event.message, extra=log_data)
        elif event.severity == RiskEventSeverity.HIGH:
            self._logger.error(event.message, extra=log_data)
        elif event.severity == RiskEventSeverity.WARNING:
            self._logger.warning(event.message, extra=log_data)
        else:
            self._logger.info(event.message, extra=log_data)

    def get_observer_id(self) -> str:
        """Get observer ID."""
        return self.observer_id


class AlertingRiskObserver(RiskObserver):
    """Observer that creates and stores risk alerts."""

    def __init__(
        self,
        observer_id: str = "alerting_observer",
        alert_callback: Callable[[RiskAlert], None] | None = None,
    ):
        """
        Initialize alerting observer.

        Args:
            observer_id: Unique identifier
            alert_callback: Optional callback for alerts
        """
        self.observer_id = observer_id
        self.alert_callback = alert_callback
        self.alerts: list[RiskAlert] = []
        self._logger = logger

    async def handle_event(self, event: RiskEvent) -> None:
        """Create alert from risk event."""
        try:
            alert = RiskAlert(
                timestamp=event.timestamp,
                event_type=event.event_type.value,
                severity=event.severity.value,
                message=event.message,
                data=event.data,
            )

            # Store alert
            self.alerts.append(alert)

            # Maintain reasonable alert history size
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-100:]  # Keep last 100

            # Execute callback if provided
            if self.alert_callback:
                try:
                    if asyncio.iscoroutinefunction(self.alert_callback):
                        await self.alert_callback(alert)
                    else:
                        self.alert_callback(alert)
                except Exception as e:
                    self._logger.error(f"Alert callback error: {e}")

            self._logger.debug(f"Risk alert created: {alert.message}")

        except Exception as e:
            self._logger.error(f"Error creating risk alert: {e}")

    def get_observer_id(self) -> str:
        """Get observer ID."""
        return self.observer_id

    def get_recent_alerts(self, limit: int = 50) -> list[RiskAlert]:
        """Get recent alerts."""
        return self.alerts[-limit:] if self.alerts else []


class CircuitBreakerObserver(RiskObserver):
    """Observer that implements circuit breaker functionality."""

    def __init__(
        self,
        observer_id: str = "circuit_breaker_observer",
        break_callback: Callable[[RiskEvent], None] | None = None,
        break_duration_minutes: int = 15,
    ):
        """
        Initialize circuit breaker observer.

        Args:
            observer_id: Unique identifier
            break_callback: Callback when circuit breaks
            break_duration_minutes: Circuit break duration
        """
        self.observer_id = observer_id
        self.break_callback = break_callback
        self.break_duration_minutes = break_duration_minutes
        self.is_broken = False
        self.break_time: datetime | None = None
        self.break_count = 0
        self._logger = logger

        # Events that trigger circuit breaker
        self.break_events = {
            RiskEventType.LOSS_LIMIT_REACHED,
            RiskEventType.DRAWDOWN_EXCEEDED,
            RiskEventType.EMERGENCY_STOP_TRIGGERED,
        }

    async def handle_event(self, event: RiskEvent) -> None:
        """Handle circuit breaker logic."""
        try:
            # Check if circuit should reset
            if self.is_broken and self._should_reset_circuit():
                await self._reset_circuit_breaker()

            # Check if event should trigger circuit breaker
            if (
                event.event_type in self.break_events
                and event.severity in [RiskEventSeverity.HIGH, RiskEventSeverity.CRITICAL]
                and not self.is_broken
            ):
                await self._trigger_circuit_breaker(event)

        except Exception as e:
            self._logger.error(f"Circuit breaker error: {e}")

    def _should_reset_circuit(self) -> bool:
        """Check if circuit breaker should reset."""
        if not self.break_time:
            return False

        elapsed_minutes = (datetime.now(timezone.utc) - self.break_time).total_seconds() / 60
        return elapsed_minutes >= self.break_duration_minutes

    async def _trigger_circuit_breaker(self, event: RiskEvent) -> None:
        """Trigger circuit breaker."""
        self.is_broken = True
        self.break_time = datetime.now(timezone.utc)
        self.break_count += 1

        self._logger.critical(
            f"Circuit breaker triggered by {event.event_type.value}",
            extra={"break_count": self.break_count, "event_data": event.data},
        )

        # Execute break callback
        if self.break_callback:
            try:
                if asyncio.iscoroutinefunction(self.break_callback):
                    await self.break_callback(event)
                else:
                    self.break_callback(event)
            except Exception as e:
                self._logger.error(f"Circuit breaker callback error: {e}")

    async def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker."""
        self.is_broken = False
        self.break_time = None
        self._logger.info(f"Circuit breaker reset after {self.break_duration_minutes} minutes")

    def get_observer_id(self) -> str:
        """Get observer ID."""
        return self.observer_id

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "is_broken": self.is_broken,
            "break_time": self.break_time.isoformat() if self.break_time else None,
            "break_count": self.break_count,
            "break_duration_minutes": self.break_duration_minutes,
        }


class UnifiedRiskMonitor:
    """
    Centralized risk monitor that unifies all monitoring patterns.

    This eliminates duplication of risk monitoring logic across modules.
    """

    def __init__(self) -> None:
        """Initialize unified risk monitor."""
        self._observers: dict[str, RiskObserver] = {}
        self._thresholds: dict[str, Decimal] = {}
        self._metrics_history: list[RiskMetrics] = []
        self._last_risk_level = RiskLevel.LOW
        self._monitoring_active = False
        self._logger = logger

        # Initialize default observers
        self.add_observer(LoggingRiskObserver())
        self.add_observer(AlertingRiskObserver())
        self.add_observer(CircuitBreakerObserver())

        # Set default thresholds
        self._set_default_thresholds()

    def _set_default_thresholds(self) -> None:
        """Set default monitoring thresholds."""
        self._thresholds = {
            "max_drawdown": to_decimal("0.2"),  # 20%
            "max_var_1d": to_decimal("0.1"),  # 10%
            "max_daily_loss": to_decimal("0.05"),  # 5%
            "max_correlation": to_decimal("0.7"),  # 70%
            "min_liquidity": to_decimal("0.2"),  # 20%
            "max_positions": to_decimal("10"),  # 10 positions
            "volatility_spike_multiplier": to_decimal("2.0"),  # 2x normal
        }

    def add_observer(self, observer: RiskObserver) -> None:
        """
        Add a risk observer.

        Args:
            observer: Observer to add
        """
        observer_id = observer.get_observer_id()
        self._observers[observer_id] = observer
        self._logger.debug(f"Added risk observer: {observer_id}")

    def remove_observer(self, observer_id: str) -> None:
        """
        Remove a risk observer.

        Args:
            observer_id: ID of observer to remove
        """
        if observer_id in self._observers:
            del self._observers[observer_id]
            self._logger.debug(f"Removed risk observer: {observer_id}")

    async def notify_observers(self, event: RiskEvent) -> None:
        """
        Notify all observers of a risk event.

        Args:
            event: Risk event to broadcast
        """
        for observer_id, observer in self._observers.items():
            try:
                await observer.handle_event(event)
            except Exception as e:
                self._logger.error(f"Observer {observer_id} error: {e}")

    async def monitor_metrics(self, metrics: RiskMetrics) -> None:
        """
        Monitor risk metrics and generate events.

        Args:
            metrics: Current risk metrics
        """
        try:
            # Store metrics history
            self._metrics_history.append(metrics)
            if len(self._metrics_history) > 1000:
                self._metrics_history = self._metrics_history[-100:]  # Keep recent

            # Check various risk conditions
            await self._check_drawdown(metrics)
            await self._check_var(metrics)
            await self._check_correlation(metrics)
            await self._check_risk_level_change(metrics)
            await self._check_volatility_spike(metrics)

        except Exception as e:
            self._logger.error(f"Risk monitoring error: {e}")

    async def monitor_portfolio(self, portfolio_data: dict[str, Any]) -> None:
        """
        Monitor portfolio-level risks.

        Args:
            portfolio_data: Portfolio data dictionary
        """
        try:
            await self._check_position_limits(portfolio_data)
            await self._check_daily_loss(portfolio_data)
            await self._check_liquidity(portfolio_data)

        except Exception as e:
            self._logger.error(f"Portfolio monitoring error: {e}")

    # Risk condition checkers

    async def _check_drawdown(self, metrics: RiskMetrics) -> None:
        """Check drawdown threshold."""
        if (
            metrics.current_drawdown is not None
            and metrics.current_drawdown > self._thresholds["max_drawdown"]
        ):
            event = RiskEvent(
                event_type=RiskEventType.DRAWDOWN_EXCEEDED,
                severity=RiskEventSeverity.CRITICAL,
                message=f"Drawdown {metrics.current_drawdown:.2%} exceeds limit",
                data={"drawdown": float(metrics.current_drawdown)},
            )
            await self.notify_observers(event)

    async def _check_var(self, metrics: RiskMetrics) -> None:
        """Check VaR threshold."""
        if metrics.var_1d > self._thresholds["max_var_1d"]:
            severity = (
                RiskEventSeverity.CRITICAL
                if metrics.var_1d > self._thresholds["max_var_1d"] * to_decimal("1.5")
                else RiskEventSeverity.HIGH
            )
            event = RiskEvent(
                event_type=RiskEventType.VAR_EXCEEDED,
                severity=severity,
                message=f"VaR {metrics.var_1d} exceeds threshold",
                data={"var_1d": float(metrics.var_1d)},
            )
            await self.notify_observers(event)

    async def _check_correlation(self, metrics: RiskMetrics) -> None:
        """Check correlation risk."""
        if (
            hasattr(metrics, "correlation_risk")
            and metrics.correlation_risk is not None
            and metrics.correlation_risk > self._thresholds["max_correlation"]
        ):
            event = RiskEvent(
                event_type=RiskEventType.CORRELATION_HIGH,
                severity=RiskEventSeverity.WARNING,
                message=f"High correlation risk detected: {metrics.correlation_risk:.2f}",
                data={"correlation": float(metrics.correlation_risk)},
            )
            await self.notify_observers(event)

    async def _check_risk_level_change(self, metrics: RiskMetrics) -> None:
        """Check for risk level changes."""
        if metrics.risk_level != self._last_risk_level:
            severity = (
                RiskEventSeverity.CRITICAL
                if metrics.risk_level == RiskLevel.CRITICAL
                else RiskEventSeverity.WARNING
            )
            event = RiskEvent(
                event_type=RiskEventType.RISK_LEVEL_CHANGED,
                severity=severity,
                message=f"Risk level changed from {self._last_risk_level.value} to {metrics.risk_level.value}",
                data={
                    "old_level": self._last_risk_level.value,
                    "new_level": metrics.risk_level.value,
                },
            )
            await self.notify_observers(event)
            self._last_risk_level = metrics.risk_level

    async def _check_volatility_spike(self, metrics: RiskMetrics) -> None:
        """Check for volatility spikes."""
        if len(self._metrics_history) > 20:
            # Simple volatility spike detection
            recent_values = [float(m.portfolio_value) for m in self._metrics_history[-10:]]
            historical_values = [float(m.portfolio_value) for m in self._metrics_history[-20:-10]]

            if len(recent_values) > 1 and len(historical_values) > 1:
                import numpy as np

                recent_vol = np.std(recent_values)
                historical_vol = np.std(historical_values)

                if historical_vol > 0 and recent_vol / historical_vol > float(
                    self._thresholds["volatility_spike_multiplier"]
                ):
                    event = RiskEvent(
                        event_type=RiskEventType.VOLATILITY_SPIKE,
                        severity=RiskEventSeverity.HIGH,
                        message=f"Volatility spike detected: {recent_vol / historical_vol:.1f}x increase",
                        data={"volatility_multiplier": recent_vol / historical_vol},
                    )
                    await self.notify_observers(event)

    async def _check_position_limits(self, portfolio_data: dict[str, Any]) -> None:
        """Check position count limits."""
        positions = portfolio_data.get("positions", [])
        if len(positions) >= int(self._thresholds["max_positions"]):
            event = RiskEvent(
                event_type=RiskEventType.POSITION_LIMIT_REACHED,
                severity=RiskEventSeverity.HIGH,
                message=f"Position limit reached: {len(positions)} positions",
                data={"position_count": len(positions)},
            )
            await self.notify_observers(event)

    async def _check_daily_loss(self, portfolio_data: dict[str, Any]) -> None:
        """Check daily loss limits."""
        daily_pnl = to_decimal(str(portfolio_data.get("daily_pnl", 0)))
        total_value = to_decimal(str(portfolio_data.get("total_value", 1)))

        if daily_pnl < ZERO and total_value > ZERO:
            daily_loss_pct = safe_divide(abs(daily_pnl), total_value, ZERO)
            if daily_loss_pct > self._thresholds["max_daily_loss"]:
                event = RiskEvent(
                    event_type=RiskEventType.LOSS_LIMIT_REACHED,
                    severity=RiskEventSeverity.CRITICAL,
                    message=f"Daily loss {daily_loss_pct:.2%} exceeds limit",
                    data={"daily_loss": float(daily_loss_pct)},
                )
                await self.notify_observers(event)

    async def _check_liquidity(self, portfolio_data: dict[str, Any]) -> None:
        """Check liquidity thresholds."""
        liquidity_ratio = to_decimal(str(portfolio_data.get("liquidity_ratio", 1)))
        if liquidity_ratio < self._thresholds["min_liquidity"]:
            event = RiskEvent(
                event_type=RiskEventType.LIQUIDITY_LOW,
                severity=RiskEventSeverity.WARNING,
                message=f"Low liquidity: {liquidity_ratio:.2%}",
                data={"liquidity_ratio": float(liquidity_ratio)},
            )
            await self.notify_observers(event)

    # Configuration methods

    def set_threshold(self, key: str, value: Decimal) -> None:
        """Set monitoring threshold."""
        self._thresholds[key] = value
        self._logger.info(f"Updated threshold {key} = {value}")

    def get_thresholds(self) -> dict[str, Decimal]:
        """Get current thresholds."""
        return self._thresholds.copy()

    def get_observer_status(self) -> dict[str, Any]:
        """Get status of all observers."""
        status = {}
        for observer_id, observer in self._observers.items():
            if hasattr(observer, "get_status"):
                status[observer_id] = observer.get_status()
            else:
                status[observer_id] = {"type": type(observer).__name__}
        return status

    async def trigger_emergency_stop(self, reason: str) -> None:
        """Manually trigger emergency stop event."""
        event = RiskEvent(
            event_type=RiskEventType.EMERGENCY_STOP_TRIGGERED,
            severity=RiskEventSeverity.CRITICAL,
            message=f"Emergency stop triggered: {reason}",
            data={"reason": reason},
        )
        await self.notify_observers(event)


# Global instance for singleton pattern
_unified_risk_monitor = UnifiedRiskMonitor()


def get_unified_risk_monitor() -> UnifiedRiskMonitor:
    """Get the global unified risk monitor instance."""
    return _unified_risk_monitor
