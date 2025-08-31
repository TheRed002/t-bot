"""Recovery strategies using Protocol for standardized interfaces."""

import asyncio
import time
from collections.abc import Callable
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Protocol, runtime_checkable

from src.core import BaseComponent
from src.error_handling.security_rate_limiter import (
    get_security_rate_limiter,
    record_recovery_failure,
)
from src.error_handling.security_sanitizer import (
    SensitivityLevel,
    get_security_sanitizer,
)
from src.utils.error_categorization import is_retryable_error


@runtime_checkable
class RecoveryStrategy(Protocol):
    """Protocol for all recovery strategies with proper type annotations."""

    def can_recover(self, error: Exception, context: dict[str, Any]) -> bool:
        """Check if this strategy can handle the error."""
        ...

    async def recover(self, error: Exception, context: dict[str, Any]) -> Any:
        """Execute recovery logic asynchronously."""
        ...

    @property
    def max_attempts(self) -> int:
        """Maximum recovery attempts."""
        ...


class RetryRecovery(BaseComponent):
    """Retry recovery strategy with exponential backoff."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: Decimal = Decimal("1.0"),
        max_delay: Decimal = Decimal("60.0"),
        exponential_base: Decimal = Decimal("2.0"),
    ) -> None:
        """
        Initialize retry recovery.

        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Initial delay between retries
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff
        """
        super().__init__(name="RetryRecovery", config={})
        self._max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def can_recover(self, error: Exception, context: dict[str, Any]) -> bool:
        """Check if retry is appropriate."""
        # Check if we've exceeded max attempts
        current_attempts = context.get("retry_count", 0)
        if current_attempts >= self._max_attempts:
            return False

        # Check for non-retryable errors
        error_msg = str(error).lower()
        return is_retryable_error(error_msg)

    async def recover(self, error: Exception, context: dict[str, Any]) -> dict[str, Any]:
        """Execute retry with exponential backoff and security controls."""
        retry_count = context.get("retry_count", 0)
        component = context.get("component", "unknown")

        # Check rate limits for retry attempts
        rate_limiter = get_security_rate_limiter()
        rate_limit_result = await rate_limiter.check_rate_limit(
            component=component,
            operation="retry_recovery",
            context={"retry_count": retry_count, "error_type": type(error).__name__},
        )

        if not rate_limit_result.allowed:
            self._logger.warning(
                "Retry recovery rate limited",
                component=component,
                retry_count=retry_count,
                reason=rate_limit_result.reason,
            )
            return {
                "action": "rate_limited",
                "delay": rate_limit_result.suggested_retry_after,
                "retry_count": retry_count,
            }

        # Calculate delay with high precision decimal calculation
        from decimal import localcontext

        with localcontext() as ctx:
            ctx.prec = 8
            ctx.rounding = ROUND_HALF_UP
            delay = min(self.base_delay * (self.exponential_base**retry_count), self.max_delay)

        # Sanitize error information for logging
        sanitizer = get_security_sanitizer()
        sanitized_error_msg = sanitizer.sanitize_error_message(str(error), SensitivityLevel.MEDIUM)

        self._logger.info(
            "Retrying after delay",
            delay=delay,
            attempt=retry_count + 1,
            max_attempts=self._max_attempts,
            component=component,
            error_type=type(error).__name__,
            sanitized_error=sanitized_error_msg,
        )

        return {"action": "retry", "delay": str(delay), "retry_count": retry_count + 1}

    @property
    def max_attempts(self) -> int:
        """Maximum recovery attempts."""
        return self._max_attempts


class CircuitBreakerRecovery(BaseComponent):
    """Circuit breaker recovery strategy."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: Decimal = Decimal("60.0"),
        half_open_requests: int = 1,
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            timeout: Time before attempting half-open state
            half_open_requests: Requests allowed in half-open state
        """
        super().__init__(name="CircuitBreakerRecovery", config={})
        self._max_attempts = 1  # Circuit breaker doesn't retry
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_requests = half_open_requests

        # Circuit breaker state
        self.state = "closed"  # closed, open, half_open
        self.failure_count = 0
        self.last_failure_time: Decimal | None = None
        self.half_open_count = 0

    def can_recover(self, error: Exception, context: dict[str, Any]) -> bool:
        """Check if circuit breaker can handle this."""
        # Circuit breaker can handle any error
        return True

    async def recover(self, error: Exception, context: dict[str, Any]) -> dict[str, Any]:
        """Execute circuit breaker logic with security monitoring."""
        current_time = Decimal(str(time.time()))
        component = context.get("component", "unknown")

        # Record failure for security monitoring
        if self.state == "closed":
            record_recovery_failure(
                component=component, operation="circuit_breaker", error_severity="medium"
            )

        if self.state == "closed":
            self.failure_count += 1
            self.last_failure_time = current_time

            if self.failure_count >= self.failure_threshold:
                self.state = "open"

                # Sanitize error information for logging
                sanitizer = get_security_sanitizer()
                sanitized_error_msg = sanitizer.sanitize_error_message(
                    str(error), SensitivityLevel.MEDIUM
                )

                self._logger.warning(
                    "Circuit breaker opened",
                    component=component,
                    failure_count=self.failure_count,
                    error_type=type(error).__name__,
                    sanitized_error=sanitized_error_msg,
                )
                return {
                    "action": "circuit_break",
                    "state": "open",
                    "wait_time": str(self.timeout),
                }

            return {"action": "proceed"}

        elif self.state == "open":
            if self.last_failure_time and (current_time - self.last_failure_time) >= self.timeout:
                self.state = "half_open"
                self.half_open_count = 0
                self._logger.info("Circuit breaker entering half-open state")
                return {"action": "test", "state": "half_open"}

            remaining_time = (
                self.timeout - (current_time - self.last_failure_time)
                if self.last_failure_time is not None
                else self.timeout
            )
            return {
                "action": "reject",
                "state": "open",
                "remaining_time": str(remaining_time),
            }

        elif self.state == "half_open":
            self.half_open_count += 1

            if context.get("success", False):
                # Success in half-open state, close circuit
                self.state = "closed"
                self.failure_count = 0
                self._logger.info("Circuit breaker closed after successful test")
                return {"action": "proceed", "state": "closed"}

            if self.half_open_count >= self.half_open_requests:
                # Failed in half-open, reopen circuit
                self.state = "open"
                self.last_failure_time = current_time
                self._logger.warning("Circuit breaker reopened after half-open test failure")
                return {
                    "action": "circuit_break",
                    "state": "open",
                    "wait_time": str(self.timeout),
                }

            return {"action": "test", "state": "half_open"}

        return {"action": "reject", "reason": "unknown_state"}

    @property
    def max_attempts(self) -> int:
        """Maximum recovery attempts."""
        return self._max_attempts


class FallbackRecovery(BaseComponent):
    """Fallback to alternative implementation."""

    def __init__(self, fallback_function: Callable[..., Any], max_attempts: int = 1) -> None:
        """
        Initialize fallback recovery.

        Args:
            fallback_function: Function to call as fallback
            max_attempts: Maximum fallback attempts
        """
        super().__init__(name="FallbackRecovery", config={})
        self.fallback_function = fallback_function
        self._max_attempts = max_attempts

    def can_recover(self, error: Exception, context: dict[str, Any]) -> bool:
        """Check if fallback is available."""
        return self.fallback_function is not None

    async def recover(self, error: Exception, context: dict[str, Any]) -> dict[str, Any]:
        """Execute fallback function with security controls."""
        component = context.get("component", "unknown")

        # Check rate limits for fallback attempts
        rate_limiter = get_security_rate_limiter()
        rate_limit_result = await rate_limiter.check_rate_limit(
            component=component,
            operation="fallback_recovery",
            context={"error_type": type(error).__name__},
        )

        if not rate_limit_result.allowed:
            self._logger.warning(
                "Fallback recovery rate limited",
                component=component,
                reason=rate_limit_result.reason,
            )
            return {"action": "rate_limited", "delay": rate_limit_result.suggested_retry_after}

        # Sanitize error information for logging
        sanitizer = get_security_sanitizer()
        sanitized_error_msg = sanitizer.sanitize_error_message(str(error), SensitivityLevel.MEDIUM)

        self._logger.info(
            "Using fallback recovery",
            component=component,
            error_type=type(error).__name__,
            sanitized_error=sanitized_error_msg,
        )

        try:
            # Sanitize arguments before passing to fallback
            args = context.get("args", {})
            sanitized_args = sanitizer.sanitize_context(args, SensitivityLevel.MEDIUM)

            if asyncio.iscoroutinefunction(self.fallback_function):
                # Execute async fallback with timeout to prevent blocking and memory leaks
                fallback_task = None
                try:
                    # Create task to allow for proper cancellation
                    fallback_task = asyncio.create_task(self.fallback_function(**sanitized_args))

                    result = await asyncio.wait_for(fallback_task, timeout=30.0)
                    return {"action": "fallback_complete", "result": result}

                except asyncio.TimeoutError:
                    self._logger.warning(
                        "Async fallback function timeout", component=component, timeout=30.0
                    )
                    # Ensure task is cancelled to prevent resource leaks
                    if fallback_task and not fallback_task.done():
                        fallback_task.cancel()
                        try:
                            await asyncio.wait_for(fallback_task, timeout=5.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass  # Task cleanup complete
                    return {"action": "fallback_timeout"}

                except asyncio.CancelledError:
                    # Ensure proper cleanup on cancellation
                    if fallback_task and not fallback_task.done():
                        fallback_task.cancel()
                        try:
                            await asyncio.wait_for(fallback_task, timeout=1.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass  # Task cleanup complete
                    # Re-raise cancellation to maintain proper async task cleanup
                    raise

            else:
                # Execute sync fallback in executor to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                executor = None
                executor_future = None

                try:
                    # Create executor future for proper cancellation handling
                    executor_future = loop.run_in_executor(
                        executor, lambda: self.fallback_function(**sanitized_args)
                    )

                    result = await asyncio.wait_for(executor_future, timeout=30.0)
                    return {"action": "fallback_complete", "result": result}

                except asyncio.TimeoutError:
                    self._logger.warning(
                        "Sync fallback function timeout", component=component, timeout=30.0
                    )
                    # Cancel executor future to prevent resource leaks
                    if executor_future and not executor_future.done():
                        executor_future.cancel()
                        try:
                            await asyncio.wait_for(executor_future, timeout=5.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass  # Executor cleanup complete
                    return {"action": "fallback_timeout"}

                except asyncio.CancelledError:
                    # Ensure proper executor cleanup on cancellation
                    if executor_future and not executor_future.done():
                        executor_future.cancel()
                        try:
                            await asyncio.wait_for(executor_future, timeout=1.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass  # Executor cleanup complete
                    # Re-raise cancellation to maintain proper async task cleanup
                    raise
        except Exception as e:
            # Record fallback failure
            record_recovery_failure(
                component=component, operation="fallback_execution", error_severity="high"
            )

            # Sanitize fallback error message
            sanitized_fallback_error = sanitizer.sanitize_error_message(
                str(e), SensitivityLevel.HIGH
            )

            self._logger.error(
                "Fallback recovery failed",
                component=component,
                fallback_error_type=type(e).__name__,
                sanitized_fallback_error=sanitized_fallback_error,
            )
            return {"action": "fallback_failed", "error": sanitized_fallback_error}

    @property
    def max_attempts(self) -> int:
        """Maximum recovery attempts."""
        return self._max_attempts
