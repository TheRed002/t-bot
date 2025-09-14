"""
Shared decorator patterns for backtesting module.

This module contains common decorator combinations used across backtesting components
to eliminate duplication and ensure consistency.
"""

from functools import wraps
from typing import Any, Callable

from src.error_handling.decorators import with_circuit_breaker, with_error_context, with_retry
from src.utils.decorators import time_execution


def backtesting_operation(
    component: str = "backtesting",
    operation: str = "",
    max_retries: int = 3,
    circuit_breaker: bool = True,
    failure_threshold: int = 3,
    recovery_timeout: int = 60,
    time_it: bool = True
):
    """
    Combined decorator for typical backtesting operations.

    Args:
        component: Component name for error context
        operation: Operation name for error context
        max_retries: Maximum retry attempts
        circuit_breaker: Whether to use circuit breaker
        failure_threshold: Circuit breaker failure threshold
        recovery_timeout: Circuit breaker recovery timeout
        time_it: Whether to time the execution

    Returns:
        Decorated function with applied decorators
    """
    def decorator(func: Callable) -> Callable:
        # Start with the base function
        decorated_func = func

        # Apply decorators in reverse order (innermost first)
        if time_it:
            decorated_func = time_execution(decorated_func)

        # Add error context
        decorated_func = with_error_context(
            component=component,
            operation=operation or func.__name__
        )(decorated_func)

        # Add circuit breaker if requested
        if circuit_breaker:
            decorated_func = with_circuit_breaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )(decorated_func)

        # Add retry logic
        if max_retries > 0:
            decorated_func = with_retry(max_attempts=max_retries)(decorated_func)

        return decorated_func

    return decorator


def data_loading_operation(
    operation: str = "data_loading",
    max_retries: int = 3
):
    """
    Specialized decorator for data loading operations.

    Args:
        operation: Specific operation name
        max_retries: Maximum retry attempts

    Returns:
        Decorated function with data loading specific decorators
    """
    return backtesting_operation(
        component="data_loading",
        operation=operation,
        max_retries=max_retries,
        circuit_breaker=True,
        failure_threshold=3,
        recovery_timeout=60,
        time_it=True
    )


def trading_operation(
    operation: str = "trading",
    max_retries: int = 1  # Trading operations usually need fewer retries
):
    """
    Specialized decorator for trading operations.

    Args:
        operation: Specific operation name
        max_retries: Maximum retry attempts (lower for trading)

    Returns:
        Decorated function with trading specific decorators
    """
    return backtesting_operation(
        component="trading",
        operation=operation,
        max_retries=max_retries,
        circuit_breaker=False,  # Trading operations might not want circuit breaker
        time_it=True
    )


def service_operation(
    service_name: str = "backtesting",
    operation: str = "",
    max_retries: int = 3
):
    """
    Specialized decorator for service layer operations.

    Args:
        service_name: Name of the service
        operation: Specific operation name
        max_retries: Maximum retry attempts

    Returns:
        Decorated function with service specific decorators
    """
    return backtesting_operation(
        component=service_name,
        operation=operation,
        max_retries=max_retries,
        circuit_breaker=True,
        failure_threshold=3,
        recovery_timeout=60,
        time_it=True
    )


def analysis_operation(
    operation: str = "analysis",
    max_retries: int = 2
):
    """
    Specialized decorator for analysis operations.

    Args:
        operation: Specific operation name
        max_retries: Maximum retry attempts

    Returns:
        Decorated function with analysis specific decorators
    """
    return backtesting_operation(
        component="backtesting_analysis",
        operation=operation,
        max_retries=max_retries,
        circuit_breaker=False,  # Analysis operations usually don't need circuit breaker
        time_it=True
    )