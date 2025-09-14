"""
Simple error handling decorators.

Simplified error handling with basic retry, circuit breaker, and fallback patterns.
"""

import asyncio
import functools
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar, cast

from src.core.exceptions import (
    DatabaseConnectionError,
    NetworkError,
    ServiceError,
)
from src.core.logging import get_logger

F = TypeVar("F", bound=Callable[..., Any])
logger = get_logger(__name__)


class FallbackStrategy(Enum):
    """Fallback strategies for error handling."""

    NONE = "none"
    RETURN_NONE = "return_none"
    RETURN_EMPTY = "return_empty"
    RAISE_ERROR = "raise_error"


@dataclass
class RetryConfig:
    """Simple retry configuration."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential: bool = True


@dataclass
class CircuitBreakerConfig:
    """Simple circuit breaker configuration."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type | None = None


@dataclass
class FallbackConfig:
    """Simple fallback configuration."""

    strategy: FallbackStrategy = FallbackStrategy.RETURN_NONE
    fallback_value: Any = None
    fallback_function: Callable | None = None
    default_value: Any = None


# Simple global state for circuit breakers and metrics
_circuit_breakers: dict[str, Any] = {}
_error_counts: dict[str, int] = {}
_active_handlers: set[str] = set()


def error_handler(
    retry_config: RetryConfig | None = None,
    circuit_breaker_config: CircuitBreakerConfig | None = None,
    fallback_config: FallbackConfig | None = None,
    enable_metrics: bool = True,
    enable_logging: bool = True,
    **kwargs,
):
    """Error handler decorator."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **func_kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"

            # Register handler
            _active_handlers.add(func_name)

            # Check circuit breaker
            if circuit_breaker_config and _should_circuit_break(func_name, circuit_breaker_config):
                return _handle_fallback(fallback_config)

            last_error = None
            max_attempts = retry_config.max_attempts if retry_config else 1

            for attempt in range(max_attempts):
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **func_kwargs)
                    else:
                        result = func(*args, **func_kwargs)

                    # Reset error count on success
                    _error_counts[func_name] = 0
                    return result

                except Exception as e:
                    last_error = e
                    _error_counts[func_name] = _error_counts.get(func_name, 0) + 1

                    if enable_logging:
                        logger.warning(f"Error in {func_name} (attempt {attempt + 1}): {e}")

                    # Check if should retry
                    exceptions = kwargs.get("exceptions")
                    should_retry = _should_retry(e, retry_config, exceptions)
                    if attempt < max_attempts - 1 and should_retry:
                        delay = _calculate_delay(attempt, retry_config)
                        await asyncio.sleep(delay)
                        continue
                    else:
                        break

            # All retries failed
            if fallback_config:
                return _handle_fallback(fallback_config)
            else:
                raise last_error

        @functools.wraps(func)
        def sync_wrapper(*args, **func_kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"

            # Register handler
            _active_handlers.add(func_name)

            # Check circuit breaker
            if circuit_breaker_config and _should_circuit_break(func_name, circuit_breaker_config):
                return _handle_fallback(fallback_config)

            last_error = None
            max_attempts = retry_config.max_attempts if retry_config else 1

            for attempt in range(max_attempts):
                try:
                    result = func(*args, **func_kwargs)

                    # Reset error count on success
                    _error_counts[func_name] = 0
                    return result

                except Exception as e:
                    last_error = e
                    _error_counts[func_name] = _error_counts.get(func_name, 0) + 1

                    if enable_logging:
                        logger.warning(f"Error in {func_name} (attempt {attempt + 1}): {e}")

                    # Check if should retry
                    exceptions = kwargs.get("exceptions")
                    should_retry = _should_retry(e, retry_config, exceptions)
                    if attempt < max_attempts - 1 and should_retry:
                        delay = _calculate_delay(attempt, retry_config)
                        time.sleep(delay)
                        continue
                    else:
                        break

            # All retries failed
            if fallback_config:
                return _handle_fallback(fallback_config)
            else:
                raise last_error

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)

    return decorator


def _should_circuit_break(func_name: str, config: CircuitBreakerConfig) -> bool:
    """Check if circuit breaker should be open."""
    error_count = _error_counts.get(func_name, 0)
    return error_count >= config.failure_threshold


def _should_retry(
    error: Exception, config: RetryConfig | None, exceptions: tuple | None = None
) -> bool:
    """Check if error should trigger retry."""
    if not config:
        return False

    # If specific exceptions provided, use those
    if exceptions:
        return isinstance(error, exceptions)

    # Default retry on network and database errors
    return isinstance(error, NetworkError | DatabaseConnectionError | ServiceError)


def _calculate_delay(attempt: int, config: RetryConfig | None) -> float:
    """Calculate retry delay."""
    if not config:
        return 0

    if config.exponential:
        delay = config.base_delay * (2**attempt)
    else:
        delay = config.base_delay

    return min(delay, config.max_delay)


def _handle_fallback(config: FallbackConfig | None) -> Any:
    """Handle fallback strategy."""
    if not config:
        return None

    if config.strategy == FallbackStrategy.RETURN_NONE:
        return None
    elif config.strategy == FallbackStrategy.RETURN_EMPTY:
        return {}
    elif config.strategy == FallbackStrategy.RAISE_ERROR:
        raise ServiceError("Circuit breaker open")
    elif config.fallback_function:
        return config.fallback_function()
    else:
        return config.fallback_value


# Convenience decorators that wrap enhanced_error_handler
def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
    exceptions: tuple | None = None,
    backoff_factor: float | None = None,
):
    """
    Retry decorator with configurable retry logic.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential: Whether to use exponential backoff
        exceptions: Tuple of exceptions to retry on
        backoff_factor: Backward compatibility parameter (unused)

    Returns:
        Decorated function with retry logic
    """
    retry_config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential=exponential,
    )
    return error_handler(retry_config=retry_config, exceptions=exceptions)


def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type | None = None,
    **kwargs,
):
    """
    Circuit breaker decorator to prevent cascading failures.

    Args:
        failure_threshold: Number of failures before circuit opens
        recovery_timeout: Time in seconds before attempting recovery
        expected_exception: Backward compatibility parameter (unused)
        **kwargs: Additional configuration parameters

    Returns:
        Decorated function with circuit breaker logic
    """
    circuit_config = CircuitBreakerConfig(
        failure_threshold=failure_threshold, recovery_timeout=recovery_timeout
    )
    return error_handler(circuit_breaker_config=circuit_config)


def with_fallback(
    strategy: FallbackStrategy = FallbackStrategy.RETURN_NONE,
    fallback_value: Any = None,
    fallback_function: Callable | None = None,
    default_value: Any = None,
):
    """
    Fallback decorator for graceful error handling.

    Args:
        strategy: Fallback strategy to use
        fallback_value: Value to return on failure
        fallback_function: Function to call for fallback value
        default_value: Default value (overrides fallback_value if provided)

    Returns:
        Decorated function with fallback logic
    """
    # Use default_value if provided, otherwise use fallback_value
    actual_fallback_value = default_value if default_value is not None else fallback_value

    fallback_config = FallbackConfig(
        strategy=strategy, fallback_value=actual_fallback_value, fallback_function=fallback_function
    )
    return error_handler(fallback_config=fallback_config)


def with_error_context(**context_kwargs):
    """
    Error context decorator for adding contextual information to exceptions.

    Args:
        **context_kwargs: Context information to add to errors

    Returns:
        Decorated function with error context
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                # Add context to error
                if hasattr(e, "context"):
                    e.context.update(context_kwargs)
                else:
                    e.context = context_kwargs
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Add context to error
                if hasattr(e, "context"):
                    e.context.update(context_kwargs)
                else:
                    e.context = context_kwargs
                raise

        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)

    return decorator


def get_active_handler_count() -> int:
    """Get count of active error handlers."""
    return len(_active_handlers)


def shutdown_all_error_handlers() -> None:
    """Shutdown all error handlers."""
    _active_handlers.clear()
    _error_counts.clear()
    _circuit_breakers.clear()


# Backward compatibility aliases
circuit_breaker = with_circuit_breaker
retry = with_retry
fallback = with_fallback

# Additional backward compatibility function
def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple | None = None,
):
    """
    Retry with exponential backoff - backward compatibility function.
    
    This is an alias for with_retry with exponential backoff.
    """
    return with_retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential=True,
        exceptions=exceptions,
        backoff_factor=backoff_factor,
    )
