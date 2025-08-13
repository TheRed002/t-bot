"""
Circuit Breaker System for P-009 Risk Management.

This module implements comprehensive circuit breakers that automatically halt trading
when risk thresholds are exceeded or system failures are detected.

CRITICAL: This integrates with P-008 (risk management), P-001 (types, exceptions, config),
P-002A (error handling), and P-007A (utils) components.

Circuit Breaker Types:
- Daily loss limit breaker (default 5%)
- Portfolio drawdown breaker (default 10%)
- Volatility spike detection
- Model confidence degradation
- System error rate monitoring
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from src.core.config import Config
from src.core.exceptions import (
    CircuitBreakerTriggeredError,
)
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import (
    CircuitBreakerEvent,
    CircuitBreakerType,
    CircuitBreakerStatus,
)

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

# MANDATORY: Import from P-008
from src.risk_management.base import BaseRiskManager

# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution
from src.utils.helpers import calculate_volatility

logger = get_logger(__name__)


"""Use core CircuitBreakerStatus for breaker states (no local duplication)."""

# Backward compatibility for tests expecting CircuitBreakerState symbol
CircuitBreakerState = CircuitBreakerStatus


# Use CircuitBreakerType from core types instead of defining new enum


# CircuitBreakerEvent is already defined in src.core.types


class BaseCircuitBreaker(ABC):
    """
    Abstract base class for circuit breakers.

    All circuit breakers must implement this interface to ensure consistent
    behavior across different risk monitoring scenarios.
    """

    def __init__(self, config: Config, risk_manager: BaseRiskManager):
        """
        Initialize circuit breaker with configuration and risk manager.

        Args:
            config: Application configuration
            risk_manager: Risk manager for calculations
        """
        self.config = config
        self.risk_config = config.risk
        self.risk_manager = risk_manager
        self.error_handler = ErrorHandler(config)
        self.logger = logger.bind(component="circuit_breaker")

        # Circuit breaker state
        self.state = CircuitBreakerStatus.CLOSED
        self.trigger_time: datetime | None = None
        self.recovery_time: datetime | None = None
        self.trigger_count = 0
        self.max_trigger_count = 5

        # Event history
        self.events: list[CircuitBreakerEvent] = []
        self.max_events = 100

        # Recovery settings
        self.recovery_timeout = timedelta(minutes=30)  # 30 minutes default
        self.test_interval = timedelta(minutes=5)  # Test every 5 minutes

        self.logger.info("Circuit breaker initialized", breaker_type=self.__class__.__name__)

    @abstractmethod
    @time_execution
    async def check_condition(self, data: dict[str, Any]) -> bool:
        """
        Check if circuit breaker condition is triggered.

        Args:
            data: Data required for condition checking

        Returns:
            bool: True if circuit breaker should trigger
        """
        pass

    @abstractmethod
    async def get_threshold_value(self) -> Decimal:
        """
        Get the current threshold value for this circuit breaker.

        Returns:
            Decimal: Current threshold value
        """
        pass

    @abstractmethod
    async def get_current_value(self, data: dict[str, Any]) -> Decimal:
        """
        Get the current value being monitored.

        Args:
            data: Data required for value calculation

        Returns:
            Decimal: Current monitored value
        """
        pass

    @time_execution
    async def evaluate(self, data: dict[str, Any]) -> bool:
        """
        Evaluate circuit breaker condition and update state.

        Args:
            data: Data required for evaluation

        Returns:
            bool: True if circuit breaker is triggered
        """
        try:
            if self.state == CircuitBreakerStatus.OPEN:
                # Check if recovery timeout has passed
                if self.trigger_time and datetime.now() - self.trigger_time > self.recovery_timeout:
                    self.state = CircuitBreakerStatus.HALF_OPEN
                    self.logger.info(
                        "Circuit breaker transitioning to half-open state",
                        breaker_type=self.__class__.__name__,
                    )

            # Check condition
            if await self.check_condition(data):
                await self._trigger_circuit_breaker(data)
                return True

            # If half-open and condition passes, close circuit breaker
            if self.state == CircuitBreakerStatus.HALF_OPEN:
                await self._close_circuit_breaker()

            return False

        except Exception as e:
            self.logger.error(
                "Circuit breaker evaluation failed",
                error=str(e),
                breaker_type=self.__class__.__name__,
            )
            # Route through central error handler with context
            try:
                if hasattr(self.error_handler, "create_error_context"):
                    context = self.error_handler.create_error_context(
                        e, component="circuit_breaker", operation="evaluate"
                    )
                    await self.error_handler.handle_error(e, context)
            except Exception as handler_error:
                self.logger.error("Error handler failed", error=str(handler_error))

            # Re-raise CircuitBreakerTriggeredError to allow tests to catch it
            if isinstance(e, CircuitBreakerTriggeredError):
                raise
            return False

    async def _trigger_circuit_breaker(self, data: dict[str, Any]) -> None:
        """Trigger the circuit breaker and log the event."""
        if self.state == CircuitBreakerStatus.CLOSED:
            self.state = CircuitBreakerStatus.OPEN
            self.trigger_time = datetime.now()
            self.trigger_count += 1

            # Create event record
            current_value = await self.get_current_value(data)
            threshold_value = await self.get_threshold_value()

            # Use the breaker_type class attribute defined in each circuit
            # breaker class
            breaker_type = getattr(
                self.__class__, "breaker_type", CircuitBreakerType.MANUAL_TRIGGER
            )

            event = CircuitBreakerEvent(
                trigger_type=breaker_type,
                threshold=threshold_value,
                actual_value=current_value,
                timestamp=datetime.now(),
                description=f"Circuit breaker triggered: {current_value} > {threshold_value}",
                metadata={"trigger_count": self.trigger_count},
            )

            self.events.append(event)
            if len(self.events) > self.max_events:
                self.events.pop(0)

            self.logger.warning(
                "Circuit breaker triggered",
                breaker_type=self.__class__.__name__,
                current_value=current_value,
                threshold=threshold_value,
                trigger_count=self.trigger_count,
            )

            # Raise exception to halt trading
            raise CircuitBreakerTriggeredError(
                f"Circuit breaker triggered: {self.__class__.__name__}",
                error_code="CIRCUIT_BREAKER_TRIGGERED",
                details={
                    "breaker_type": self.__class__.__name__,
                    "current_value": current_value,
                    "threshold": threshold_value,
                    "trigger_count": self.trigger_count,
                },
            )

    async def _close_circuit_breaker(self) -> None:
        """Close the circuit breaker after successful recovery."""
        self.state = CircuitBreakerStatus.CLOSED
        self.recovery_time = datetime.now()

        self.logger.info(
            "Circuit breaker closed after recovery",
            breaker_type=self.__class__.__name__,
            recovery_time=self.recovery_time,
        )

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "state": self.state.value,
            "trigger_count": self.trigger_count,
            "trigger_time": self.trigger_time.isoformat() if self.trigger_time else None,
            "recovery_time": self.recovery_time.isoformat() if self.recovery_time else None,
            "events_count": len(self.events),
        }

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.state = CircuitBreakerStatus.CLOSED
        self.trigger_time = None
        self.recovery_time = None
        self.trigger_count = 0
        self.events.clear()

        self.logger.info("Circuit breaker reset", breaker_type=self.__class__.__name__)


class DailyLossLimitBreaker(BaseCircuitBreaker):
    """
    Circuit breaker for daily loss limit monitoring.

    Triggers when daily portfolio loss exceeds the configured threshold.
    Default threshold is 5% of portfolio value.
    """

    breaker_type = CircuitBreakerType.DAILY_LOSS_LIMIT

    def __init__(self, config: Config, risk_manager: BaseRiskManager):
        super().__init__(config, risk_manager)
        self.threshold_pct = self.risk_config.max_daily_loss_pct
        self.logger = logger.bind(breaker_type="daily_loss_limit")

    async def get_threshold_value(self) -> Decimal:
        """Get daily loss limit threshold."""
        return Decimal(str(self.threshold_pct))

    async def get_current_value(self, data: dict[str, Any]) -> Decimal:
        """Calculate current daily loss percentage."""
        portfolio_value = data.get("portfolio_value", Decimal("0"))
        daily_pnl = data.get("daily_pnl", Decimal("0"))

        if portfolio_value <= 0:
            return Decimal("0")

        daily_loss_pct = abs(min(daily_pnl, Decimal("0"))) / portfolio_value
        return daily_loss_pct

    @time_execution
    async def check_condition(self, data: dict[str, Any]) -> bool:
        """Check if daily loss limit is exceeded."""
        current_value = await self.get_current_value(data)
        threshold_value = await self.get_threshold_value()

        return current_value > threshold_value


class DrawdownLimitBreaker(BaseCircuitBreaker):
    """
    Circuit breaker for portfolio drawdown monitoring.

    Triggers when portfolio drawdown exceeds the configured threshold.
    Default threshold is 15% of peak portfolio value.
    """

    breaker_type = CircuitBreakerType.DRAWDOWN_LIMIT

    def __init__(self, config: Config, risk_manager: BaseRiskManager):
        super().__init__(config, risk_manager)
        self.threshold_pct = self.risk_config.max_drawdown_pct
        self.logger = logger.bind(breaker_type="drawdown_limit")

    async def get_threshold_value(self) -> Decimal:
        """Get drawdown limit threshold."""
        return Decimal(str(self.threshold_pct))

    async def get_current_value(self, data: dict[str, Any]) -> Decimal:
        """Calculate current drawdown percentage."""
        current_value = data.get("current_portfolio_value", Decimal("0"))
        peak_value = data.get("peak_portfolio_value", current_value)

        if peak_value <= 0:
            return Decimal("0")

        # Calculate drawdown: (peak - current) / peak
        # If current > peak, drawdown should be 0 (no drawdown)
        if current_value >= peak_value:
            return Decimal("0")

        drawdown_pct = (peak_value - current_value) / peak_value
        return drawdown_pct

    @time_execution
    async def check_condition(self, data: dict[str, Any]) -> bool:
        """Check if drawdown limit is exceeded."""
        current_value = await self.get_current_value(data)
        threshold_value = await self.get_threshold_value()

        return current_value > threshold_value


class VolatilitySpikeBreaker(BaseCircuitBreaker):
    """
    Circuit breaker for volatility spike detection.

    Triggers when market volatility exceeds normal levels, indicating
    potentially dangerous market conditions.
    """

    breaker_type = CircuitBreakerType.VOLATILITY_SPIKE

    def __init__(self, config: Config, risk_manager: BaseRiskManager):
        super().__init__(config, risk_manager)
        self.volatility_threshold = Decimal("0.05")  # 5% daily volatility
        self.lookback_period = 20  # 20 days for volatility calculation
        self.logger = logger.bind(breaker_type="volatility_spike")

    async def get_threshold_value(self) -> Decimal:
        """Get volatility spike threshold."""
        return self.volatility_threshold

    async def get_current_value(self, data: dict[str, Any]) -> Decimal:
        """Calculate current market volatility."""
        price_history = data.get("price_history", [])

        if len(price_history) < 2:
            return Decimal("0")

        # Calculate daily returns
        returns = []
        for i in range(1, len(price_history)):
            prev_price = Decimal(str(price_history[i - 1]))
            curr_price = Decimal(str(price_history[i]))

            if prev_price > 0:
                daily_return = (curr_price - prev_price) / prev_price
                returns.append(daily_return)

        if not returns:
            return Decimal("0")

        # Calculate volatility (standard deviation of returns)
        try:
            vol_float = calculate_volatility([float(r) for r in returns])
            return Decimal(str(vol_float))
        except Exception:
            return Decimal("0")

    @time_execution
    async def check_condition(self, data: dict[str, Any]) -> bool:
        """Check if volatility spike is detected."""
        current_value = await self.get_current_value(data)
        threshold_value = await self.get_threshold_value()

        return current_value > threshold_value


class ModelConfidenceBreaker(BaseCircuitBreaker):
    """
    Circuit breaker for model confidence degradation.

    Triggers when ML model confidence falls below acceptable levels,
    indicating potential model drift or unreliable predictions.
    """

    breaker_type = CircuitBreakerType.MODEL_CONFIDENCE

    def __init__(self, config: Config, risk_manager: BaseRiskManager):
        super().__init__(config, risk_manager)
        self.confidence_threshold = Decimal("0.3")  # 30% minimum confidence
        self.logger = logger.bind(breaker_type="model_confidence")

    async def get_threshold_value(self) -> Decimal:
        """Get minimum confidence threshold."""
        return self.confidence_threshold

    async def get_current_value(self, data: dict[str, Any]) -> Decimal:
        """Get current model confidence level."""
        model_confidence = data.get("model_confidence", Decimal("1.0"))
        return model_confidence

    @time_execution
    async def check_condition(self, data: dict[str, Any]) -> bool:
        """Check if model confidence is below threshold."""
        current_value = await self.get_current_value(data)
        threshold_value = await self.get_threshold_value()

        return current_value < threshold_value


class SystemErrorRateBreaker(BaseCircuitBreaker):
    """
    Circuit breaker for system error rate monitoring.

    Triggers when system error rate exceeds acceptable levels,
    indicating potential system instability.
    """

    breaker_type = CircuitBreakerType.SYSTEM_ERROR_RATE

    def __init__(self, config: Config, risk_manager: BaseRiskManager):
        super().__init__(config, risk_manager)
        self.error_rate_threshold = Decimal("0.1")  # 10% error rate
        self.error_window = timedelta(minutes=15)  # 15-minute window
        self.logger = logger.bind(breaker_type="system_error_rate")

        # Error tracking
        self.error_times: list[datetime] = []

    async def get_threshold_value(self) -> Decimal:
        """Get error rate threshold."""
        return self.error_rate_threshold

    async def get_current_value(self, data: dict[str, Any]) -> Decimal:
        """Calculate current error rate."""
        current_time = datetime.now()
        window_start = current_time - self.error_window

        # Clean old errors
        self.error_times = [t for t in self.error_times if t > window_start]

        # Add new error if provided
        if data.get("error_occurred", False):
            self.error_times.append(current_time)

        total_requests = data.get("total_requests", 1)
        error_count = len(self.error_times)

        if total_requests <= 0:
            return Decimal("0")

        error_rate = Decimal(str(error_count)) / Decimal(str(total_requests))
        return error_rate

    @time_execution
    async def check_condition(self, data: dict[str, Any]) -> bool:
        """Check if error rate exceeds threshold."""
        current_value = await self.get_current_value(data)
        threshold_value = await self.get_threshold_value()

        return current_value > threshold_value


class CircuitBreakerManager:
    """
    Manager for all circuit breakers in the system.

    Coordinates multiple circuit breakers and provides unified interface
    for circuit breaker management and monitoring.
    """

    def __init__(self, config: Config, risk_manager: BaseRiskManager):
        """
        Initialize circuit breaker manager.

        Args:
            config: Application configuration
            risk_manager: Risk manager for calculations
        """
        self.config = config
        self.risk_manager = risk_manager
        self.logger = logger.bind(component="circuit_breaker_manager")

        # Initialize all circuit breakers
        self.circuit_breakers: dict[str, BaseCircuitBreaker] = {
            "daily_loss_limit": DailyLossLimitBreaker(config, risk_manager),
            "drawdown_limit": DrawdownLimitBreaker(config, risk_manager),
            "volatility_spike": VolatilitySpikeBreaker(config, risk_manager),
            "model_confidence": ModelConfidenceBreaker(config, risk_manager),
            "system_error_rate": SystemErrorRateBreaker(config, risk_manager),
        }

        self.logger.info(
            "Circuit breaker manager initialized", breaker_count=len(self.circuit_breakers)
        )

    @time_execution
    async def evaluate_all(self, data: dict[str, Any]) -> dict[str, bool]:
        """
        Evaluate all circuit breakers.

        Args:
            data: Data required for circuit breaker evaluation

        Returns:
            Dict[str, bool]: Results for each circuit breaker
        """
        results = {}
        triggered_breakers = []

        for name, breaker in self.circuit_breakers.items():
            try:
                triggered = await breaker.evaluate(data)
                results[name] = triggered

                if triggered:
                    triggered_breakers.append(name)
                    self.logger.warning("Circuit breaker triggered", breaker_name=name)

            except CircuitBreakerTriggeredError as e:
                # Circuit breaker was triggered, record it and continue
                triggered_breakers.append(name)
                results[name] = True
                self.logger.error("Circuit breaker triggered", breaker_name=name, error=str(e))
            except Exception as e:
                self.logger.error(
                    "Circuit breaker evaluation failed", breaker_name=name, error=str(e)
                )
                results[name] = False

        # If any circuit breakers were triggered, raise an exception
        if triggered_breakers:
            raise CircuitBreakerTriggeredError(
                f"Circuit breakers triggered: {', '.join(triggered_breakers)}"
            )

        return results

    @time_execution
    async def get_status(self) -> dict[str, Any]:
        """Get status of all circuit breakers."""
        status = {}

        for name, breaker in self.circuit_breakers.items():
            status[name] = breaker.get_status()

        return status

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self.circuit_breakers.values():
            breaker.reset()

        self.logger.info("All circuit breakers reset")

    def get_triggered_breakers(self) -> list[str]:
        """Get list of currently triggered circuit breakers."""
        triggered = []

        for name, breaker in self.circuit_breakers.items():
            if breaker.state == CircuitBreakerStatus.OPEN:
                triggered.append(name)

        return triggered

    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed (no circuit breakers triggered)."""
        return len(self.get_triggered_breakers()) == 0
