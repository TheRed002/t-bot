"""
Comprehensive error handling framework for the trading bot.

This module provides error categorization, severity classification, retry policies,
circuit breaker integration, and error context preservation for debugging and recovery.

CRITICAL: This module integrates with P-001 core exceptions and P-002 database
for error logging and will be used by all subsequent prompts.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any

from src.core.config import Config

# MANDATORY: Import from P-001 core framework
from src.core.exceptions import (
    DataError,
    ExchangeError,
    ExecutionError,
    ModelError,
    RiskManagementError,
    SecurityError,
    StateConsistencyError,
    TradingBotError,
    ValidationError,
)
from src.core.logging import get_logger
from src.core.types import ErrorPattern

# MANDATORY: Import from P-007A utils framework
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and escalation."""

    CRITICAL = "critical"  # System failure, data corruption, security breach
    HIGH = "high"  # Trading halted, model failure, risk limit breach
    MEDIUM = "medium"  # Performance degradation, data quality issues
    LOW = "low"  # Configuration warnings, minor validation errors


@dataclass
class ErrorContext:
    """Context information for error tracking and recovery."""

    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    component: str
    operation: str
    error: Exception  # Add the actual error object
    user_id: str | None = None
    bot_id: str | None = None
    symbol: str | None = None
    order_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    stack_trace: str | None = None
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3


class CircuitBreaker:
    """Circuit breaker pattern implementation for preventing cascading failures."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise TradingBotError("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"

            raise e

    def should_transition_to_half_open(self) -> bool:
        """Check if circuit breaker should transition to HALF_OPEN state."""
        if self.state == "OPEN" and self.last_failure_time:
            return time.time() - self.last_failure_time > self.recovery_timeout
        return False


class ErrorHandler:
    """Comprehensive error handling and recovery system."""

    def __init__(self, config: Config):
        self.config = config
        self.error_patterns: dict[str, ErrorPattern] = {}
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.retry_policies: dict[str, dict[str, Any]] = {
            "network_errors": {
                "max_attempts": 5,
                "backoff_strategy": "exponential",
                "base_delay": 1,
                "max_delay": 60,
                "jitter": True,
            },
            "api_rate_limits": {
                "max_attempts": 3,
                "backoff_strategy": "linear",
                "base_delay": 5,
                "respect_retry_after": True,
            },
            "database_errors": {
                "max_attempts": 3,
                "backoff_strategy": "exponential",
                "base_delay": 0.5,
                "max_delay": 10,
            },
        }

        # Initialize circuit breakers for critical components
        self._initialize_circuit_breakers()

    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for critical system components."""
        self.circuit_breakers = {
            "api_calls": CircuitBreaker(failure_threshold=5, recovery_timeout=30),
            "database_connections": CircuitBreaker(failure_threshold=3, recovery_timeout=15),
            "exchange_connections": CircuitBreaker(failure_threshold=3, recovery_timeout=20),
            "model_inference": CircuitBreaker(failure_threshold=2, recovery_timeout=60),
        }

    @time_execution
    def classify_error(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on error type and context."""
        if isinstance(error, StateConsistencyError | SecurityError):
            return ErrorSeverity.CRITICAL
        elif isinstance(
            error, RiskManagementError | ExchangeError | ConnectionError | ExecutionError
        ):
            return ErrorSeverity.HIGH
        elif isinstance(error, DataError | ModelError):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, ValidationError):
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM

    @time_execution
    def create_error_context(
        self, error: Exception, component: str, operation: str, **kwargs
    ) -> ErrorContext:
        """Create error context for tracking and recovery."""
        import uuid

        # Extract kwargs that should be passed to ErrorContext
        context_kwargs = {}
        for key in ["user_id", "bot_id", "symbol", "order_id"]:
            if key in kwargs:
                context_kwargs[key] = kwargs.pop(key)

        return ErrorContext(
            error_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            severity=self.classify_error(error),
            component=component,
            operation=operation,
            error=error,
            details={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "kwargs": kwargs,
            },
            stack_trace=self._get_stack_trace(),
            **context_kwargs,
        )

    def _get_stack_trace(self) -> str:
        """Get current stack trace for debugging."""
        import traceback

        return "".join(traceback.format_stack())

    @time_execution
    async def handle_error(
        self, error: Exception, context: ErrorContext, recovery_strategy: Callable | None = None
    ) -> bool:
        """Handle error with appropriate recovery strategy."""

        # Log error with structured data
        logger.error(
            "Error occurred",
            error_id=context.error_id,
            severity=context.severity.value,
            component=context.component,
            operation=context.operation,
            error_type=type(error).__name__,
            error_message=str(error),
            details=context.details,
        )

        # Update error patterns
        self._update_error_patterns(context)

        # Check if error should trigger circuit breaker
        circuit_breaker_key = self._get_circuit_breaker_key(context)
        if circuit_breaker_key and circuit_breaker_key in self.circuit_breakers:
            try:
                self.circuit_breakers[circuit_breaker_key].call(lambda: self._raise_error(error))
            except TradingBotError:
                logger.warning(
                    "Circuit breaker triggered",
                    circuit_breaker=circuit_breaker_key,
                    error_id=context.error_id,
                )

        # Attempt recovery if strategy provided and not at max attempts
        if recovery_strategy and context.recovery_attempts < context.max_recovery_attempts:
            try:
                context.recovery_attempts += 1
                # Accept both class instances with execute_recovery and callables
                if hasattr(recovery_strategy, "execute_recovery"):
                    await recovery_strategy.execute_recovery(context.__dict__)
                else:
                    await recovery_strategy(context)
                logger.info(
                    "Recovery successful",
                    error_id=context.error_id,
                    recovery_attempts=context.recovery_attempts,
                )
                return True
            except Exception as recovery_error:
                logger.error(
                    "Recovery failed",
                    error_id=context.error_id,
                    recovery_error=str(recovery_error),
                    recovery_attempts=context.recovery_attempts,
                )

        # Escalate if critical or high severity
        if context.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            await self._escalate_error(context)

        return False

    def _update_error_patterns(self, context: ErrorContext):
        """Update error pattern tracking for analytics."""
        pattern_key = f"{context.component}:{type(context.error).__name__}"

        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = ErrorPattern(
                pattern_id=pattern_key,
                pattern_type="frequency",
                component=context.component,
                error_type=type(context.error).__name__,
                frequency=0.0,
                severity=context.severity.value,
                first_detected=context.timestamp,
                last_detected=context.timestamp,
                occurrence_count=0,
                confidence=0.8,
                description=f"Error pattern for {context.component}",
                suggested_action="Monitor and investigate",
            )

        self.error_patterns[pattern_key].occurrence_count += 1
        self.error_patterns[pattern_key].last_detected = context.timestamp

    @time_execution
    def _get_circuit_breaker_key(self, context: ErrorContext) -> str | None:
        """Determine which circuit breaker should be triggered."""
        if "api" in context.component.lower():
            return "api_calls"
        elif "database" in context.component.lower():
            return "database_connections"
        elif "exchange" in context.component.lower():
            return "exchange_connections"
        elif "model" in context.component.lower():
            return "model_inference"
        return None

    def _raise_error(self, error: Exception):
        """Helper method to raise error for circuit breaker."""
        raise error

    async def _escalate_error(self, context: ErrorContext):
        """Escalate critical errors for immediate attention."""
        escalation_message = {
            "error_id": context.error_id,
            "severity": context.severity.value,
            "component": context.component,
            "operation": context.operation,
            "timestamp": context.timestamp.isoformat(),
            "details": context.details,
        }

        logger.critical("Error escalation required", escalation_data=escalation_message)

        # TODO: Implement notification channels (Discord, email, SMS)
        # This will be implemented in P-031 (Alerting and Notification System)

    def get_retry_policy(self, error_type: str) -> dict[str, Any]:
        """Get retry policy for specific error type."""
        return self.retry_policies.get(error_type, self.retry_policies["network_errors"])

    def get_circuit_breaker_status(self) -> dict[str, str]:
        """Get status of all circuit breakers."""
        return {name: breaker.state for name, breaker in self.circuit_breakers.items()}

    def get_error_patterns(self) -> dict[str, ErrorPattern]:
        """Get current error patterns for analysis."""
        return self.error_patterns.copy()


def error_handler_decorator(
    component: str, operation: str, recovery_strategy: Callable | None = None, **kwargs
):
    """Decorator for automatic error handling and recovery."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Get error handler from config or create new one
                from src.core.config import Config

                config = Config()
                handler = ErrorHandler(config)

                context = handler.create_error_context(
                    error=e, component=component, operation=operation, **kwargs
                )

                await handler.handle_error(e, context, recovery_strategy)
                raise e

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get error handler from config or create new one
                from src.core.config import Config

                config = Config()
                handler = ErrorHandler(config)

                context = handler.create_error_context(
                    error=e, component=component, operation=operation, **kwargs
                )

                # For sync functions, we can't await, so just log
                logger.error(
                    "Error in sync function",
                    error_id=context.error_id,
                    severity=context.severity.value,
                    component=context.component,
                    operation=context.operation,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                raise e

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
