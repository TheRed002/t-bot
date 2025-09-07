"""Error handling utilities for analytics module."""

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any

from src.core.base import BaseComponent
from src.core.exceptions import AnalyticsError, ServiceError, ValidationError


class AnalyticsErrorHandler(BaseComponent):
    """Centralized error handling utilities for analytics."""

    def __init__(self):
        super().__init__()

    def handle_analytics_error(
        self, error: Exception, operation: str, context: dict | None = None, reraise: bool = True
    ) -> Exception | None:
        """Standard error handling for analytics operations.

        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
            context: Additional context for debugging
            reraise: Whether to reraise the exception

        Returns:
            The processed exception if reraise=False
        """
        context = context or {}

        error_msg = f"Analytics operation '{operation}' failed: {error!s}"

        if isinstance(error, (ValidationError, ServiceError, AnalyticsError)):
            # Already a proper application error
            self.logger.error(
                error_msg, extra={"context": context, "error_type": type(error).__name__}
            )
            if reraise:
                raise error
            return error
        else:
            # Wrap generic exceptions
            analytics_error = AnalyticsError(error_msg, details=context)
            self.logger.error(error_msg, extra={"context": context, "original_error": str(error)})
            if reraise:
                raise analytics_error from error
            return analytics_error

    def safe_execute(self, operation: str, context: dict | None = None):
        """Decorator for safe execution of analytics operations."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    return self.handle_analytics_error(e, operation, context, reraise=False)

            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return self.handle_analytics_error(e, operation, context, reraise=False)

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def log_and_continue(self, error: Exception, operation: str, default_return: Any = None) -> Any:
        """Log error and continue with default return value.

        Args:
            error: The exception that occurred
            operation: Description of the operation
            default_return: Value to return on error

        Returns:
            The default return value
        """
        self.logger.warning(
            f"Analytics operation '{operation}' failed, continuing with default: {error!s}"
        )
        return default_return

    def validate_and_handle(
        self,
        condition: bool,
        error_message: str,
        operation: str,
        validation_context: dict | None = None,
    ) -> None:
        """Validate condition and handle error with consistent messaging.

        Args:
            condition: Condition that must be True
            error_message: Error message if condition fails
            operation: Description of the operation
            validation_context: Additional validation context

        Raises:
            ValidationError: If condition is False
        """
        if not condition:
            context = {"operation": operation}
            if validation_context:
                context.update(validation_context)

            self.handle_analytics_error(
                ValidationError(error_message, details=context), operation, context
            )
