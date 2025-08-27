"""
Comprehensive error handling framework for the trading bot.

This module provides error categorization, severity classification, retry policies,
circuit breaker integration, and error context preservation for debugging and recovery.

CRITICAL: This module integrates with P-001 core exceptions and P-002 database
for error logging and will be used by all subsequent prompts.
"""

import asyncio
from decimal import Decimal
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.error_handling.recovery import RecoveryStrategy

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

# Import consolidated ErrorContext from context module
from src.error_handling.context import ErrorContext, ErrorSeverity
from src.error_handling.security_rate_limiter import (
    get_security_rate_limiter,
    record_recovery_failure,
)
from src.error_handling.security_sanitizer import (
    SensitivityLevel,
    get_security_sanitizer,
)

# MANDATORY: Import from P-007A utils framework
from src.utils.decorators import time_execution


@dataclass
class ErrorPattern:
    """Represents a detected error pattern."""

    pattern_id: str
    pattern_type: str  # frequency, correlation, trend, anomaly
    component: str
    error_type: str
    frequency: float  # errors per hour
    severity: str
    first_detected: datetime
    last_detected: datetime
    occurrence_count: int
    confidence: float  # 0.0 to 1.0
    description: str
    suggested_action: str
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert ErrorPattern to dictionary for serialization."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "component": self.component,
            "error_type": self.error_type,
            "frequency": self.frequency,
            "severity": self.severity,
            "first_detected": self.first_detected.isoformat(),
            "last_detected": self.last_detected.isoformat(),
            "occurrence_count": self.occurrence_count,
            "confidence": self.confidence,
            "description": self.description,
            "suggested_action": self.suggested_action,
            "is_active": self.is_active,
        }


class CircuitBreaker:
    """Circuit breaker pattern implementation for preventing cascading failures."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.RLock()

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if (
                    self.last_failure_time
                    and time.time() - self.last_failure_time > self.recovery_timeout
                ):
                    self.state = "HALF_OPEN"
                else:
                    raise TradingBotError("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            with self._lock:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
            return result
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"

            raise e

    def should_transition_to_half_open(self) -> bool:
        """Check if circuit breaker should transition to HALF_OPEN state."""
        with self._lock:
            if self.state == "OPEN" and self.last_failure_time:
                return time.time() - self.last_failure_time > self.recovery_timeout
            return False


def ensure_timezone_aware(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware (UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class ErrorPatternCache:
    """Optimized error pattern storage with size limits and TTL."""

    def __init__(self, max_patterns: int = 1000, ttl_hours: int = 24):
        self.max_patterns = max_patterns
        self.ttl_hours = ttl_hours
        self._patterns: OrderedDict[str, ErrorPattern] = OrderedDict()
        self._last_cleanup = datetime.now(timezone.utc)
        self._cleanup_interval_minutes = 30

    def add_pattern(self, pattern: ErrorPattern) -> None:
        """Add or update error pattern with automatic cleanup."""
        # Periodic cleanup
        if self._should_cleanup():
            self._cleanup_expired()

        # Remove LRU patterns if at capacity
        while len(self._patterns) >= self.max_patterns:
            self._patterns.popitem(last=False)

        # Add/update pattern
        self._patterns[pattern.pattern_id] = pattern
        self._patterns.move_to_end(pattern.pattern_id)  # Mark as recently used

    def get_pattern(self, pattern_id: str) -> ErrorPattern | None:
        """Get pattern and mark as recently used."""
        if pattern_id not in self._patterns:
            return None

        pattern = self._patterns[pattern_id]
        # Check if expired
        current_time_utc = datetime.now(timezone.utc)
        # Ensure both datetimes are timezone-aware for comparison
        pattern_time = ensure_timezone_aware(pattern.first_detected)
        age_hours = (current_time_utc - pattern_time).total_seconds() / 3600
        if age_hours > self.ttl_hours:
            del self._patterns[pattern_id]
            return None

        # Mark as recently used
        self._patterns.move_to_end(pattern_id)
        return pattern

    def get_all_patterns(self) -> dict[str, ErrorPattern]:
        """Get all active patterns."""
        self._cleanup_expired()
        return dict(self._patterns)

    def _should_cleanup(self) -> bool:
        """Check if cleanup is needed."""
        current_time_utc = datetime.now(timezone.utc)
        # Ensure both datetimes are timezone-aware for comparison
        last_cleanup_time = ensure_timezone_aware(self._last_cleanup)
        minutes_since_cleanup = (current_time_utc - last_cleanup_time).total_seconds() / 60
        return minutes_since_cleanup >= self._cleanup_interval_minutes

    def _cleanup_expired(self) -> None:
        """Remove expired patterns."""
        current_time = datetime.now(timezone.utc)
        expired_keys = []

        for pattern_id, pattern in self._patterns.items():
            # Ensure both datetimes are timezone-aware for comparison
            pattern_time = ensure_timezone_aware(pattern.first_detected)
            age_hours = (current_time - pattern_time).total_seconds() / 3600
            if age_hours > self.ttl_hours:
                expired_keys.append(pattern_id)

        for key in expired_keys:
            del self._patterns[key]

        self._last_cleanup = current_time

    def cleanup_expired(self) -> None:
        """Public method to remove expired patterns."""
        self._cleanup_expired()

    def size(self) -> int:
        """Get current number of patterns."""
        return len(self._patterns)

    def get_last_cleanup(self) -> datetime:
        """Get the last cleanup timestamp."""
        return self._last_cleanup


class ErrorHandler:
    """Comprehensive error handling and recovery system with optimized memory management."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(self.__class__.__module__)

        # Security components
        self.sanitizer = get_security_sanitizer()
        self.rate_limiter = get_security_rate_limiter()

        # Optimized error pattern storage
        self.error_patterns = ErrorPatternCache(max_patterns=1000, ttl_hours=24)
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        # Performance monitoring
        self._operation_count = 0
        self._last_performance_log = datetime.now(timezone.utc)
        self._performance_log_interval_minutes = 10
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
        # Ensure we have a valid exception type
        error_type = type(error)
        if not isinstance(error_type, type):
            return ErrorSeverity.MEDIUM

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
        """Create error context for tracking and recovery with security sanitization."""
        # Extract kwargs that should be passed to ErrorContext
        context_kwargs = {}
        for key in ["user_id", "bot_id", "symbol", "order_id"]:
            if key in kwargs:
                context_kwargs[key] = kwargs.pop(key)

        # Determine sensitivity level based on error classification
        severity = self.classify_error(error)
        sensitivity_level = self._get_sensitivity_level(severity, component)

        # Sanitize error message and details
        sanitized_error_message = self.sanitizer.sanitize_error_message(
            str(error), sensitivity_level
        )
        sanitized_kwargs = self.sanitizer.sanitize_context(kwargs, sensitivity_level)
        sanitized_stack_trace = self.sanitizer.sanitize_stack_trace(
            self._get_stack_trace(), sensitivity_level
        )

        return ErrorContext(
            error_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            component=component,
            operation=operation,
            error=error,
            details={
                "error_type": type(error).__name__,
                "error_message": sanitized_error_message,
                "kwargs": sanitized_kwargs,
                "sensitivity_level": sensitivity_level.value,
            },
            stack_trace=sanitized_stack_trace,
            **context_kwargs,
        )

    def _get_stack_trace(self) -> str:
        """Get current stack trace for debugging."""
        return "".join(traceback.format_stack())

    @time_execution
    async def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        recovery_strategy: "RecoveryStrategy | Callable[..., Any] | None" = None,
    ) -> bool:
        """Handle error with appropriate recovery strategy and security controls."""

        # Check rate limits before processing
        rate_limit_result = await self.rate_limiter.check_rate_limit(
            component=context.component,
            operation="error_recovery",
            context={"error_id": context.error_id},
        )

        if not rate_limit_result.allowed:
            self.logger.warning(
                "Error recovery rate limited",
                error_id=context.error_id,
                component=context.component,
                reason=rate_limit_result.reason,
                retry_after=rate_limit_result.suggested_retry_after,
            )
            # Still log the error but don't attempt recovery

        # Get sanitized data for logging
        sensitivity_level = self._get_sensitivity_level(context.severity, context.component)
        sanitized_error_message = self.sanitizer.sanitize_error_message(
            str(error), sensitivity_level
        )
        sanitized_details = self.sanitizer.sanitize_context(context.details, sensitivity_level)

        # Log error with sanitized structured data
        self.logger.error(
            "Error occurred",
            error_id=context.error_id,
            severity=context.severity.value,
            component=context.component,
            operation=context.operation,
            error_type=type(error).__name__,
            error_message=sanitized_error_message,
            details=sanitized_details,
            rate_limited=not rate_limit_result.allowed,
        )

        # Update error patterns
        self._update_error_patterns(context)

        # Check if error should trigger circuit breaker
        circuit_breaker_key = self._get_circuit_breaker_key(context)
        if circuit_breaker_key and circuit_breaker_key in self.circuit_breakers:
            try:
                self.circuit_breakers[circuit_breaker_key].call(lambda: self._raise_error(error))
            except TradingBotError:
                self.logger.warning(
                    "Circuit breaker triggered",
                    circuit_breaker=circuit_breaker_key,
                    error_id=context.error_id,
                )

        # Attempt recovery if strategy provided, not at max attempts, and not rate limited
        recovery_attempted = False
        if (
            recovery_strategy
            and context.recovery_attempts < context.max_recovery_attempts
            and rate_limit_result.allowed
        ):
            try:
                context.recovery_attempts += 1
                # Accept both class instances with execute_recovery and callables
                try:
                    if hasattr(recovery_strategy, "execute_recovery"):
                        recovery_data = context.__dict__ if hasattr(context, "__dict__") else context
                        await recovery_strategy.execute_recovery(recovery_data)
                    else:
                        await recovery_strategy(context)
                except AttributeError as e:
                    self.logger.error(
                        "Recovery strategy does not have required interface",
                        error_id=context.error_id,
                        strategy_type=type(recovery_strategy).__name__,
                        error=str(e),
                    )
                    raise

                # Sanitize recovery success message
                self.logger.info(
                    "Recovery successful",
                    error_id=context.error_id,
                    component=context.component,
                    recovery_attempts=context.recovery_attempts,
                )
                recovery_attempted = True
                return True
            except Exception as recovery_error:
                # Record recovery failure for rate limiting
                record_recovery_failure(
                    component=context.component,
                    operation="error_recovery",
                    error_severity=context.severity.value,
                )

                # Sanitize recovery error message
                sanitized_recovery_error = self.sanitizer.sanitize_error_message(
                    str(recovery_error), sensitivity_level
                )

                self.logger.error(
                    "Recovery failed",
                    error_id=context.error_id,
                    component=context.component,
                    recovery_error=sanitized_recovery_error,
                    recovery_attempts=context.recovery_attempts,
                )
                recovery_attempted = True

        # Record failure if recovery wasn't attempted or rate limited
        if not recovery_attempted or not rate_limit_result.allowed:
            record_recovery_failure(
                component=context.component,
                operation="error_handling",
                error_severity=context.severity.value,
            )

        # Escalate if critical or high severity with sanitized context
        if context.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            await self._escalate_error(context)

        return False

    def handle_error_sync(
        self,
        error: Exception,
        component: str,
        operation: str,
        recovery_strategy: Callable | None = None,
        **kwargs,
    ) -> bool:
        """
        Synchronous version of handle_error for use in non-async contexts.
        
        Args:
            error: The exception to handle
            component: Component where error occurred
            operation: Operation being performed
            recovery_strategy: Optional recovery strategy
            **kwargs: Additional context data
            
        Returns:
            bool: True if error was recovered, False otherwise
        """
        import asyncio
        
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            # Already in async context, create task
            future = asyncio.ensure_future(
                self.handle_error(error, component, operation, recovery_strategy, **kwargs)
            )
            # Can't wait synchronously in running loop
            self.logger.warning(
                "handle_error_sync called from async context, scheduled as task",
                component=component,
                operation=operation,
            )
            return False  # Can't wait for result
        except RuntimeError:
            # No running loop, safe to use run_until_complete
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.handle_error(error, component, operation, recovery_strategy, **kwargs)
                )
            finally:
                loop.close()
                asyncio.set_event_loop(None)

    def _update_error_patterns(self, context: ErrorContext) -> None:
        """Update error pattern tracking for analytics with optimized storage."""
        pattern_key = f"{context.component}:{type(context.error).__name__}"

        # Get existing pattern or create new one
        existing_pattern = self.error_patterns.get_pattern(pattern_key)

        if existing_pattern is None:
            new_pattern = ErrorPattern(
                pattern_id=pattern_key,
                pattern_type="frequency",
                component=context.component,
                error_type=type(context.error).__name__,
                frequency=1.0,
                severity=context.severity.value,
                first_detected=context.timestamp,
                last_detected=context.timestamp,
                occurrence_count=1,
                confidence=0.8,
                description=f"Error pattern for {context.component}",
                suggested_action="Monitor and investigate",
            )
            self.error_patterns.add_pattern(new_pattern)
        else:
            # Update existing pattern
            existing_pattern.occurrence_count += 1
            existing_pattern.last_detected = context.timestamp
            existing_pattern.frequency = self._calculate_frequency(existing_pattern)
            self.error_patterns.add_pattern(existing_pattern)  # Update position in LRU

        # Log performance metrics periodically
        self._operation_count += 1
        self._log_performance_metrics()

    def _calculate_frequency(self, pattern: ErrorPattern) -> float:
        """Calculate error frequency per hour."""
        time_diff = (pattern.last_detected - pattern.first_detected).total_seconds()
        # Ensure minimum time to avoid division by zero
        if time_diff <= 0:
            # If no time has passed, return the count as instantaneous frequency
            return float(pattern.occurrence_count)
        hours = time_diff / 3600
        # Ensure minimum of 0.1 hours to prevent very high frequencies
        hours = max(hours, 0.1)
        return pattern.occurrence_count / hours

    def _log_performance_metrics(self) -> None:
        """Log performance metrics periodically."""
        now = datetime.now(timezone.utc)
        minutes_since_log = (now - self._last_performance_log).total_seconds() / 60

        if minutes_since_log >= self._performance_log_interval_minutes:
            self.logger.info(
                "ErrorHandler performance metrics",
                operations_processed=self._operation_count,
                error_patterns_count=self.error_patterns.size(),
                circuit_breakers_count=len(self.circuit_breakers),
                time_window_minutes=minutes_since_log,
            )
            self._operation_count = 0
            self._last_performance_log = now

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
        """Escalate critical errors for immediate attention with secure data handling."""

        # Determine escalation sensitivity level (always HIGH for escalations)
        escalation_sensitivity = SensitivityLevel.HIGH

        # Sanitize escalation data to prevent sensitive information leakage
        sanitized_details = self.sanitizer.sanitize_context(context.details, escalation_sensitivity)

        escalation_message = {
            "error_id": context.error_id,
            "severity": context.severity.value,
            "component": context.component,
            "operation": context.operation,
            "timestamp": context.timestamp.isoformat(),
            "details": sanitized_details,
            "security_note": "Escalation data has been sanitized for security",
        }

        self.logger.critical("Error escalation required", escalation_data=escalation_message)

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
        return self.error_patterns.get_all_patterns()

    def get_memory_usage_stats(self) -> dict[str, Any]:
        """Get memory usage statistics for monitoring."""
        return {
            "error_patterns_count": self.error_patterns.size(),
            "circuit_breakers_count": len(self.circuit_breakers),
            "operations_processed": self._operation_count,
            "last_cleanup": self.error_patterns.get_last_cleanup().isoformat(),
        }

    async def cleanup_resources(self) -> None:
        """Manual cleanup of resources for memory management."""
        # Force cleanup of expired patterns
        self.error_patterns.cleanup_expired()

        # Remove inactive circuit breakers
        inactive_breakers = []
        for name, breaker in self.circuit_breakers.items():
            with breaker._lock:
                if (
                    breaker.last_failure_time
                    and (time.time() - breaker.last_failure_time) > 3600  # 1 hour
                ):
                    inactive_breakers.append(name)

        for name in inactive_breakers:
            del self.circuit_breakers[name]

        self.logger.info(
            "Resource cleanup completed",
            patterns_remaining=self.error_patterns.size(),
            circuit_breakers_remaining=len(self.circuit_breakers),
            inactive_breakers_removed=len(inactive_breakers),
        )

    def _get_sensitivity_level(self, severity: ErrorSeverity, component: str) -> SensitivityLevel:
        """Determine sensitivity level based on error severity and component."""
        # Financial/trading components always get high sensitivity
        financial_components = ["trading", "exchange", "wallet", "payment", "order", "position"]

        if any(keyword in component.lower() for keyword in financial_components):
            return SensitivityLevel.CRITICAL

        # Map error severity to sensitivity level
        severity_mapping = {
            ErrorSeverity.CRITICAL: SensitivityLevel.CRITICAL,
            ErrorSeverity.HIGH: SensitivityLevel.HIGH,
            ErrorSeverity.MEDIUM: SensitivityLevel.MEDIUM,
            ErrorSeverity.LOW: SensitivityLevel.LOW,
        }

        return severity_mapping.get(severity, SensitivityLevel.MEDIUM)


# Module-level error handler instance
_error_handler: ErrorHandler | None = None


def init_error_handler(config: Config) -> None:
    """Initialize the module-level error handler instance."""
    global _error_handler
    _error_handler = ErrorHandler(config)


def get_error_handler() -> ErrorHandler:
    """Get the error handler instance, creating one if needed."""
    global _error_handler
    if _error_handler is None:
        # Create a default instance if not initialized
        from src.core.config import Config

        _error_handler = ErrorHandler(Config())
    return _error_handler


def error_handler_decorator(
    component: str,
    operation: str,
    recovery_strategy: "RecoveryStrategy | Callable[..., Any] | None" = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Decorator for automatic error handling and recovery."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                handler = get_error_handler()

                context = handler.create_error_context(
                    error=e, component=component, operation=operation, **kwargs
                )

                await handler.handle_error(e, context, recovery_strategy)
                raise e

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = get_error_handler()

                context = handler.create_error_context(
                    error=e, component=component, operation=operation, **kwargs
                )

                # For sync functions, we can't await, so just log
                logger = get_logger(func.__module__)
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
