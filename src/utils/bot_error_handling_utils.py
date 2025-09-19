"""
Shared error handling patterns for bot management services.

Extracted from duplicated error handling code across multiple bot management services.
"""

import asyncio
from functools import wraps
from typing import Any, Callable, TypeVar, Awaitable
from contextlib import asynccontextmanager

from src.core.logging import get_logger
from src.core.exceptions import (
    ServiceError,
    ValidationError,
    ExecutionError,
    NetworkError,
    DatabaseConnectionError
)

logger = get_logger(__name__)

T = TypeVar('T')


class BotServiceErrorHandler:
    """Standardized error handling for bot services."""

    @staticmethod
    def handle_service_operation(
        operation_name: str,
        bot_id: str | None = None,
        reraise: bool = True
    ):
        """
        Decorator for standardized service operation error handling.

        Args:
            operation_name: Name of the operation for logging
            bot_id: Optional bot ID for context
            reraise: Whether to reraise exceptions as ServiceError

        Usage:
            @BotServiceErrorHandler.handle_service_operation("start_bot", bot_id)
            async def start_bot(self, bot_id: str) -> bool:
                # Implementation
        """
        def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                try:
                    return await func(*args, **kwargs)

                except ValidationError as e:
                    context = f" for bot {bot_id}" if bot_id else ""
                    logger.error(f"Validation error in {operation_name}{context}: {e}")
                    if reraise:
                        raise ServiceError(f"{operation_name} validation failed: {e}") from e
                    raise

                except ExecutionError as e:
                    context = f" for bot {bot_id}" if bot_id else ""
                    logger.error(f"Execution error in {operation_name}{context}: {e}")
                    if reraise:
                        raise ServiceError(f"{operation_name} execution failed: {e}") from e
                    raise

                except NetworkError as e:
                    context = f" for bot {bot_id}" if bot_id else ""
                    logger.error(f"Network error in {operation_name}{context}: {e}")
                    if reraise:
                        raise ServiceError(f"{operation_name} network error: {e}") from e
                    raise

                except DatabaseConnectionError as e:
                    context = f" for bot {bot_id}" if bot_id else ""
                    logger.error(f"Database error in {operation_name}{context}: {e}")
                    if reraise:
                        raise ServiceError(f"{operation_name} database error: {e}") from e
                    raise

                except ServiceError:
                    # Already a ServiceError, just re-raise
                    raise

                except Exception as e:
                    context = f" for bot {bot_id}" if bot_id else ""
                    logger.error(f"Unexpected error in {operation_name}{context}: {e}")
                    if reraise:
                        raise ServiceError(f"{operation_name} failed unexpectedly: {e}") from e
                    raise

            return wrapper
        return decorator

    @staticmethod
    @asynccontextmanager
    async def error_context(operation_name: str, bot_id: str | None = None):
        """
        Async context manager for error handling with logging.

        Usage:
            async with BotServiceErrorHandler.error_context("resource_allocation", bot_id):
                # Code that might fail
        """
        context = f" for bot {bot_id}" if bot_id else ""
        try:
            yield
        except Exception as e:
            logger.error(f"Error in {operation_name}{context}: {e}")
            raise

    @staticmethod
    def safe_operation(
        operation_name: str,
        default_return: Any = None,
        log_errors: bool = True
    ):
        """
        Decorator for safe operations that should not fail the entire service.

        Args:
            operation_name: Name of the operation for logging
            default_return: Value to return on error
            log_errors: Whether to log errors

        Usage:
            @BotServiceErrorHandler.safe_operation("cleanup", default_return=False)
            async def cleanup_resources(self) -> bool:
                # Implementation that might fail
        """
        def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T | Any]]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T | Any:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if log_errors:
                        logger.error(f"Safe operation {operation_name} failed: {e}")
                    return default_return

            return wrapper
        return decorator


class RetryManager:
    """Manages retry logic for bot operations."""

    @staticmethod
    def with_retry(
        max_attempts: int = 3,
        delay_seconds: float = 1.0,
        exponential_backoff: bool = True,
        retryable_exceptions: tuple = (NetworkError, DatabaseConnectionError)
    ):
        """
        Decorator for adding retry logic to operations.

        Args:
            max_attempts: Maximum number of retry attempts
            delay_seconds: Initial delay between retries
            exponential_backoff: Whether to use exponential backoff
            retryable_exceptions: Tuple of exception types to retry on

        Usage:
            @RetryManager.with_retry(max_attempts=5, delay_seconds=2.0)
            async def connect_to_exchange(self) -> bool:
                # Implementation that might need retries
        """
        def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                last_exception = None

                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)

                    except retryable_exceptions as e:
                        last_exception = e
                        if attempt == max_attempts - 1:
                            # Last attempt, don't retry
                            break

                        # Calculate delay
                        if exponential_backoff:
                            delay = delay_seconds * (2 ** attempt)
                        else:
                            delay = delay_seconds

                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay}s..."
                        )

                        await asyncio.sleep(delay)

                    except Exception as e:
                        # Non-retryable exception, fail immediately
                        logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise

                # All retries exhausted
                logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
                if last_exception:
                    raise last_exception
                else:
                    raise RuntimeError(f"Operation {func.__name__} failed after {max_attempts} attempts")

            return wrapper
        return decorator


class CircuitBreaker:
    """Simple circuit breaker implementation for bot operations."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def __call__(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """Use circuit breaker as a decorator."""
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await self.call(func, *args, **kwargs)
        return wrapper

    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute function with circuit breaker logic."""
        # Check circuit state
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise ServiceError(
                    f"Circuit breaker is OPEN for {func.__name__}. "
                    f"Try again after {self.recovery_timeout}s"
                )

        try:
            result = await func(*args, **kwargs)

            # Success - reset failure count
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise

        except Exception as e:
            # Unexpected exception - don't count as circuit breaker failure
            logger.error(f"Unexpected exception in circuit breaker for {func.__name__}: {e}")
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt circuit reset."""
        import time
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )

    def _on_success(self) -> None:
        """Handle successful execution."""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self) -> None:
        """Handle failed execution."""
        import time

        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures. "
                f"Will retry after {self.recovery_timeout}s"
            )


class ErrorAggregator:
    """Aggregate and analyze errors across bot operations."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._logger = get_logger(f"{__name__}.{service_name}")
        self._errors: list[dict[str, Any]] = []
        self._error_counts: dict[str, int] = {}

    def record_error(
        self,
        operation_name: str,
        error: Exception,
        bot_id: str | None = None,
        context: dict[str, Any] | None = None
    ) -> None:
        """Record an error for analysis."""
        try:
            import time

            error_record = {
                'timestamp': time.time(),
                'operation_name': operation_name,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'bot_id': bot_id,
                'context': context or {}
            }

            self._errors.append(error_record)

            # Update error counts
            error_key = f"{operation_name}:{type(error).__name__}"
            self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1

            # Keep only last 1000 errors
            if len(self._errors) > 1000:
                self._errors = self._errors[-1000:]

        except Exception as e:
            self._logger.error(f"Failed to record error: {e}")

    def get_error_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get summary of errors in the specified time period."""
        try:
            import time
            cutoff_time = time.time() - (hours * 3600)

            recent_errors = [
                error for error in self._errors
                if error['timestamp'] >= cutoff_time
            ]

            # Count errors by type
            error_type_counts = {}
            operation_counts = {}
            bot_error_counts = {}

            for error in recent_errors:
                error_type = error['error_type']
                operation = error['operation_name']
                bot_id = error['bot_id']

                error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
                operation_counts[operation] = operation_counts.get(operation, 0) + 1

                if bot_id:
                    bot_error_counts[bot_id] = bot_error_counts.get(bot_id, 0) + 1

            return {
                'service_name': self.service_name,
                'time_period_hours': hours,
                'total_errors': len(recent_errors),
                'error_types': error_type_counts,
                'operations': operation_counts,
                'bots': bot_error_counts,
                'most_common_error': max(error_type_counts.items(), key=lambda x: x[1]) if error_type_counts else None,
                'most_problematic_operation': max(operation_counts.items(), key=lambda x: x[1]) if operation_counts else None
            }

        except Exception as e:
            self._logger.error(f"Failed to generate error summary: {e}")
            return {
                'service_name': self.service_name,
                'error': str(e)
            }

    def get_bot_error_history(self, bot_id: str, hours: int = 24) -> list[dict[str, Any]]:
        """Get error history for a specific bot."""
        try:
            import time
            cutoff_time = time.time() - (hours * 3600)

            bot_errors = [
                error for error in self._errors
                if error['bot_id'] == bot_id and error['timestamp'] >= cutoff_time
            ]

            return sorted(bot_errors, key=lambda x: x['timestamp'], reverse=True)

        except Exception as e:
            self._logger.error(f"Failed to get bot error history for {bot_id}: {e}")
            return []


# Global error aggregator instance
_global_error_aggregator: ErrorAggregator | None = None


def get_error_aggregator(service_name: str | None = None) -> ErrorAggregator:
    """Get or create global error aggregator."""
    global _global_error_aggregator

    if _global_error_aggregator is None:
        _global_error_aggregator = ErrorAggregator(service_name or "BotManagement")

    return _global_error_aggregator


# Convenience decorators using the global error aggregator
def handle_bot_operation(operation_name: str, bot_id: str | None = None):
    """Convenience decorator using global error aggregator."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        error_handler = BotServiceErrorHandler.handle_service_operation(operation_name, bot_id)
        aggregator = get_error_aggregator()

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await error_handler(func)(*args, **kwargs)
            except Exception as e:
                aggregator.record_error(operation_name, e, bot_id)
                raise

        return wrapper
    return decorator


def safe_bot_operation(operation_name: str, default_return: Any = None):
    """Convenience decorator for safe operations with error aggregation."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T | Any]]:
        safe_handler = BotServiceErrorHandler.safe_operation(operation_name, default_return)
        aggregator = get_error_aggregator()

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T | Any:
            try:
                return await safe_handler(func)(*args, **kwargs)
            except Exception as e:
                aggregator.record_error(operation_name, e)
                return default_return

        return wrapper
    return decorator