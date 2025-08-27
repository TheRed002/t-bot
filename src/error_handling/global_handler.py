"""Global error handler for consistent error management across modules."""

import asyncio
import sys
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from functools import wraps
from typing import Any

from src.core.logging import get_logger
from src.error_handling.context import ErrorContextFactory
from src.error_handling.factory import ErrorHandlerChain, ErrorHandlerFactory

logger = get_logger(__name__)


class GlobalErrorHandler:
    """
    Global error handler that provides consistent error handling across the application.

    This eliminates duplication of error handling logic and ensures all errors
    are handled consistently.
    """

    def __init__(self):
        """Initialize global error handler."""
        self._logger = logger
        self._error_callbacks: list[Callable] = []
        self._critical_callbacks: list[Callable] = []
        self._recovery_strategies: dict[type, Callable] = {}

        # Initialize error handler chain
        self._setup_error_handlers()

        # Statistics
        self._error_count = 0
        self._errors_by_type: dict[str, int] = {}
        self._last_error_time: datetime | None = None

    def _setup_error_handlers(self):
        """Set up the error handler chain."""
        # Register only handlers that don't have database dependencies
        try:
            from src.error_handling.handlers.validation import (
                DataValidationErrorHandler,
                ValidationErrorHandler,
            )

            ErrorHandlerFactory.register("validation", ValidationErrorHandler)
            ErrorHandlerFactory.register("data_validation", DataValidationErrorHandler)
        except ImportError:
            logger.debug("Validation error handlers not available")

        try:
            from src.error_handling.handlers.network import (
                NetworkErrorHandler,
                RateLimitErrorHandler,
            )

            ErrorHandlerFactory.register("network", NetworkErrorHandler)
            ErrorHandlerFactory.register("rate_limit", RateLimitErrorHandler)
        except ImportError:
            logger.debug("Network error handlers not available")

        # Create handler chain with registered handlers
        self._handler_chain = ErrorHandlerChain(["validation", "data_validation", "network"])

    def register_database_handler(self):
        """Register database error handler separately to avoid circular dependencies."""
        try:
            from src.error_handling.handlers.database import DatabaseErrorHandler

            ErrorHandlerFactory.register("database", DatabaseErrorHandler)

            # Update the handler chain if it exists
            if hasattr(self, "_handler_chain"):
                handlers = ["validation", "data_validation", "network", "database"]
                self._handler_chain = ErrorHandlerChain(handlers)

            logger.debug("Database error handler registered")
        except ImportError:
            logger.debug("Database error handler not available")

    def register_error_callback(self, callback: Callable[[Exception, dict], None]):
        """
        Register a callback to be called on any error.

        Args:
            callback: Function taking (error, context) parameters
        """
        self._error_callbacks.append(callback)

    def register_critical_callback(self, callback: Callable[[Exception, dict], None]):
        """
        Register a callback to be called on critical errors.

        Args:
            callback: Function taking (error, context) parameters
        """
        self._critical_callbacks.append(callback)

    def register_recovery_strategy(self, error_type: type, strategy: Callable):
        """
        Register a recovery strategy for a specific error type.

        Args:
            error_type: Type of error to handle
            strategy: Recovery function
        """
        self._recovery_strategies[error_type] = strategy

    async def handle_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        severity: str = "error",
    ) -> dict[str, Any]:
        """
        Handle an error with appropriate logging and recovery.

        Args:
            error: The exception to handle
            context: Additional context information
            severity: Error severity level

        Returns:
            Dict containing error details and recovery information
        """
        # Update statistics
        self._error_count += 1
        error_type = type(error).__name__
        self._errors_by_type[error_type] = self._errors_by_type.get(error_type, 0) + 1
        self._last_error_time = datetime.now(timezone.utc)

        # Create error context
        error_context = ErrorContextFactory.create_context(error, context or {})

        # Log the error
        self._logger.error(
            f"Error handled: {error_type}",
            error_message=str(error),
            error_type=error_type,
            severity=severity,
            context=error_context.to_dict(),
            exc_info=error if severity == "critical" else None,
        )

        # Execute callbacks
        for callback in self._error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error, error_context.to_dict())
                else:
                    callback(error, error_context.to_dict())
            except Exception as cb_error:
                self._logger.error(
                    f"Error in callback: {cb_error}",
                    callback=callback.__name__,
                    original_error=str(error),
                )

        # Execute critical callbacks if needed
        if severity == "critical":
            for callback in self._critical_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(error, error_context.to_dict())
                    else:
                        callback(error, error_context.to_dict())
                except Exception as cb_error:
                    self._logger.error(
                        f"Error in critical callback: {cb_error}",
                        callback=callback.__name__,
                        original_error=str(error),
                    )

        # Attempt recovery if strategy exists
        recovery_result = None
        if type(error) in self._recovery_strategies:
            strategy = self._recovery_strategies[type(error)]
            try:
                if asyncio.iscoroutinefunction(strategy):
                    recovery_result = await strategy(error, error_context)
                else:
                    recovery_result = strategy(error, error_context)
            except Exception as recovery_error:
                self._logger.error(
                    f"Recovery strategy failed: {recovery_error}",
                    strategy=strategy.__name__,
                    original_error=str(error),
                )

        # Use handler chain for additional processing
        handler_result = await self._handler_chain.handle(error, error_context.to_dict())

        return {
            "error_type": error_type,
            "error_message": str(error),
            "severity": severity,
            "context": error_context.to_dict(),
            "recovery_attempted": recovery_result is not None,
            "recovery_result": recovery_result,
            "handler_result": handler_result,
            "timestamp": self._last_error_time.isoformat() if self._last_error_time else None,
        }

    def handle_error_sync(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        severity: str = "error",
    ) -> dict[str, Any]:
        """
        Synchronous version of handle_error for use in non-async contexts.
        
        Args:
            error: The exception to handle
            context: Additional context about the error
            severity: Error severity level
            
        Returns:
            dict: Error handling result
        """
        import asyncio
        
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            # Already in async context, create task
            future = asyncio.ensure_future(
                self.handle_error(error, context, severity)
            )
            # Can't wait synchronously in running loop
            self._logger.warning(
                "handle_error_sync called from async context, scheduled as task",
                error_type=type(error).__name__,
            )
            return {
                "error": str(error),
                "severity": severity,
                "context": context,
                "recovery_attempted": False,
                "handler_result": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except RuntimeError:
            # No running loop, safe to use run_until_complete
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.handle_error(error, context, severity)
                )
            finally:
                loop.close()
                asyncio.set_event_loop(None)

    def handle_exception_hook(self, exc_type, exc_value, exc_traceback):
        """
        Global exception hook for unhandled exceptions.

        This is set as sys.excepthook to catch all unhandled exceptions.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            # Allow keyboard interrupt to work normally
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Log critical unhandled exception
        self._logger.critical(
            "Unhandled exception",
            exc_type=exc_type.__name__,
            exc_value=str(exc_value),
            exc_traceback=traceback.format_tb(exc_traceback),
        )

        # Create error context
        context = {
            "unhandled": True,
            "exc_type": exc_type.__name__,
            "traceback": traceback.format_tb(exc_traceback),
        }

        # Handle the error asynchronously with proper task management
        error_task = asyncio.create_task(self.handle_error(exc_value, context, severity="critical"))
        # Add done callback to log any exceptions from the error handling itself
        error_task.add_done_callback(self._log_error_handler_exception)

    def _log_error_handler_exception(self, task: asyncio.Task) -> None:
        """Log exceptions that occur in error handling tasks."""
        if task.exception():
            self._logger.critical(
                "Exception in error handler task",
                handler_exception=str(task.exception()),
                exc_info=task.exception(),
            )

    def install_global_handler(self):
        """Install this handler as the global exception handler."""
        sys.excepthook = self.handle_exception_hook

    def error_handler_decorator(
        self,
        severity: str = "error",
        reraise: bool = True,
        default_return: Any = None,
    ):
        """
        Decorator for consistent error handling.

        Args:
            severity: Error severity level
            reraise: Whether to reraise the exception after handling
            default_return: Default value to return on error (if not reraising)

        Example:
            @global_handler.error_handler_decorator(severity="warning", reraise=False)
            async def risky_operation():
                ...
        """

        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    context = {
                        "function": func.__name__,
                        "args": str(args)[:200],  # Truncate for safety
                        "kwargs": str(kwargs)[:200],
                    }
                    await self.handle_error(e, context, severity)
                    if reraise:
                        raise
                    return default_return

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = {
                        "function": func.__name__,
                        "args": str(args)[:200],  # Truncate for safety
                        "kwargs": str(kwargs)[:200],
                    }
                    # Use asyncio.create_task for async handling with proper task management
                    error_task = asyncio.create_task(self.handle_error(e, context, severity))
                    # Add done callback to log any exceptions from the error handling itself
                    error_task.add_done_callback(self._log_error_handler_exception)
                    if reraise:
                        raise
                    return default_return

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def get_statistics(self) -> dict[str, Any]:
        """
        Get error handling statistics.

        Returns:
            Dict containing error counts and patterns
        """
        return {
            "total_errors": self._error_count,
            "errors_by_type": dict(self._errors_by_type),
            "last_error_time": self._last_error_time.isoformat() if self._last_error_time else None,
            "registered_callbacks": len(self._error_callbacks),
            "registered_critical_callbacks": len(self._critical_callbacks),
            "registered_recovery_strategies": list(self._recovery_strategies.keys()),
        }

    def reset_statistics(self):
        """Reset error statistics."""
        self._error_count = 0
        self._errors_by_type = {}
        self._last_error_time = None


# Create singleton instance
_global_handler: GlobalErrorHandler | None = None


def get_global_error_handler() -> GlobalErrorHandler:
    """Get the global error handler instance."""
    global _global_handler
    if _global_handler is None:
        _global_handler = GlobalErrorHandler()
    return _global_handler


def register_with_di(container: Any) -> None:
    """Register GlobalErrorHandler with dependency injection container.

    Args:
        container: The dependency injection container
    """
    try:
        # Register as singleton
        container.register(GlobalErrorHandler, get_global_error_handler(), singleton=True)
    except Exception:
        # If registration fails, just log and continue
        # The get_global_error_handler() function will still work
        logger.debug("Failed to register GlobalErrorHandler with DI container")
