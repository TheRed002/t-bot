"""
Error handling decorators for resilient operation.

This module provides decorators for adding error handling capabilities
to functions and methods, including circuit breakers and retry logic.

IMPORTANT: This module imports core decorators from src.utils.decorators to avoid
duplication and maintains backward compatibility through re-exports.
"""

import asyncio
import functools
from collections.abc import Callable
from typing import Any, TypeVar

from src.core.logging import get_logger

# Import existing decorators from utils to avoid duplication
from src.utils.decorators import circuit_breaker, retry

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# Re-export utils decorators with error_handling naming convention for backward compatibility
def with_circuit_breaker(
    failure_threshold: int = 5, recovery_timeout: float = 60.0, expected_exception: type = Exception
) -> Callable[[F], F]:
    """
    Decorator to add circuit breaker functionality to a function.

    This is a wrapper around the utils.decorators.circuit_breaker to maintain
    backward compatibility with the error_handling module naming convention.

    Args:
        failure_threshold: Number of failures before opening
        recovery_timeout: Seconds to wait before attempting recovery
        expected_exception: Exception type to catch

    Returns:
        Decorated function
    """
    return circuit_breaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception,
    )


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator to add retry logic to a function.

    This is a wrapper around the utils.decorators.retry to maintain
    backward compatibility with the error_handling module naming convention.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Backoff multiplier for each retry
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated function
    """
    return retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
        exceptions=exceptions,
    )


def with_fallback(
    fallback_value: Any = None,
    fallback_function: Callable | None = None,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator to add fallback logic to a function.

    Args:
        fallback_value: Value to return on failure
        fallback_function: Function to call on failure
        exceptions: Tuple of exceptions to catch

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"Function {func.__name__} failed: {e}. Using fallback.")
                    if fallback_function:
                        if asyncio.iscoroutinefunction(fallback_function):
                            return await fallback_function(*args, **kwargs)
                        else:
                            return fallback_function(*args, **kwargs)
                    return fallback_value

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"Function {func.__name__} failed: {e}. Using fallback.")
                    if fallback_function:
                        return fallback_function(*args, **kwargs)
                    return fallback_value

            return sync_wrapper  # type: ignore[return-value]

    return decorator
