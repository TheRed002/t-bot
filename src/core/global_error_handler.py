"""Global error handler for consistent error management across modules."""

import asyncio
import sys
import traceback
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import Any

from src.core.dependency_injection import injectable
from src.core.logging import get_logger
from src.error_handling.context import ErrorContextFactory
from src.error_handling.factory import ErrorHandlerChain, ErrorHandlerFactory

logger = get_logger(__name__)


@injectable(singleton=True)
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
        # Import and register default handlers
        from src.error_handling.handlers.database import DatabaseErrorHandler
        from src.error_handling.handlers.network import NetworkErrorHandler
        from src.error_handling.handlers.validation import (
            DataValidationErrorHandler,
            ValidationErrorHandler,
        )

        ErrorHandlerFactory.register("network", NetworkErrorHandler)
        ErrorHandlerFactory.register("validation", ValidationErrorHandler)
        ErrorHandlerFactory.register("database", DatabaseErrorHandler)
        ErrorHandlerFactory.register("data_validation", DataValidationErrorHandler)

        # Create default chain
        self._handler_chain = ErrorHandlerChain(["network", "database", "validation"])

    def handle_error(
        self, error: Exception, context: dict[str, Any] | None = None, critical: bool = False
    ) -> dict[str, Any]:
        """
        Handle an error with appropriate strategy.

        Args:
            error: The exception to handle
            context: Additional context information
            critical: Whether this is a critical error

        Returns:
            Result of error handling
        """
        # Update statistics
        self._error_count += 1
        error_type = type(error).__name__
        self._errors_by_type[error_type] = self._errors_by_type.get(error_type, 0) + 1
        self._last_error_time = datetime.utcnow()

        # Create error context
        full_context = ErrorContextFactory.create(error, **(context or {}))

        # Log the error
        self._logger.error(f"Error handled: {error_type}: {error!s}", extra=full_context)

        # Handle through chain
        result = self._handler_chain.handle(error, full_context)

        # Execute callbacks
        if critical:
            self._execute_critical_callbacks(error, full_context)
        else:
            self._execute_error_callbacks(error, full_context)

        # Apply recovery strategy if available
        recovery_strategy = self._recovery_strategies.get(type(error))
        if recovery_strategy:
            try:
                recovery_result = recovery_strategy(error, full_context)
                result["recovery"] = recovery_result
            except Exception as recovery_error:
                self._logger.error(f"Recovery strategy failed: {recovery_error}")

        return result

    def error_handler(
        self, reraise: bool = False, critical: bool = False, context: dict[str, Any] | None = None
    ):
        """
        Decorator for error handling.

        Args:
            reraise: Whether to reraise the error after handling
            critical: Whether to treat as critical error
            context: Additional context to include
        """

        def decorator(func):
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Add function context
                    error_context = {
                        "function": func.__name__,
                        "module": func.__module__,
                        **(context or {}),
                    }

                    # Handle error
                    self.handle_error(e, error_context, critical=critical)

                    if reraise:
                        raise

                    # Return default value
                    return None

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Add function context
                    error_context = {
                        "function": func.__name__,
                        "module": func.__module__,
                        **(context or {}),
                    }

                    # Handle error
                    self.handle_error(e, error_context, critical=critical)

                    if reraise:
                        raise

                    # Return default value
                    return None

            # Return appropriate wrapper
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def register_error_callback(self, callback: Callable) -> None:
        """
        Register a callback for error events.

        Args:
            callback: Function to call on error
        """
        self._error_callbacks.append(callback)
        self._logger.debug(f"Registered error callback: {callback.__name__}")

    def register_critical_callback(self, callback: Callable) -> None:
        """
        Register a callback for critical errors.

        Args:
            callback: Function to call on critical error
        """
        self._critical_callbacks.append(callback)
        self._logger.debug(f"Registered critical callback: {callback.__name__}")

    def register_recovery_strategy(self, error_type: type, strategy: Callable) -> None:
        """
        Register a recovery strategy for an error type.

        Args:
            error_type: Type of error
            strategy: Recovery strategy function
        """
        self._recovery_strategies[error_type] = strategy
        self._logger.debug(f"Registered recovery strategy for {error_type.__name__}")

    def _execute_error_callbacks(self, error: Exception, context: dict[str, Any]) -> None:
        """Execute error callbacks."""
        for callback in self._error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(error, context))
                else:
                    callback(error, context)
            except Exception as e:
                self._logger.error(f"Error callback failed: {e}")

    def _execute_critical_callbacks(self, error: Exception, context: dict[str, Any]) -> None:
        """Execute critical error callbacks."""
        for callback in self._critical_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(error, context))
                else:
                    callback(error, context)
            except Exception as e:
                self._logger.error(f"Critical callback failed: {e}")

    def get_statistics(self) -> dict[str, Any]:
        """Get error statistics."""
        return {
            "total_errors": self._error_count,
            "errors_by_type": self._errors_by_type.copy(),
            "last_error_time": self._last_error_time.isoformat() if self._last_error_time else None,
        }

    def reset_statistics(self) -> None:
        """Reset error statistics."""
        self._error_count = 0
        self._errors_by_type.clear()
        self._last_error_time = None

    def install_exception_hook(self) -> None:
        """Install global exception hook."""

        def exception_hook(exc_type, exc_value, exc_traceback):
            """Global exception hook."""
            if issubclass(exc_type, KeyboardInterrupt):
                # Allow keyboard interrupt to work normally
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            # Format traceback
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            tb_string = "".join(tb_lines)

            # Handle the error
            self.handle_error(
                exc_value, context={"traceback": tb_string, "uncaught": True}, critical=True
            )

        sys.excepthook = exception_hook
        self._logger.info("Global exception hook installed")


# Convenience functions
_global_handler = GlobalErrorHandler()


def handle_error(
    error: Exception, context: dict[str, Any] | None = None, critical: bool = False
) -> dict[str, Any]:
    """Handle an error globally."""
    return _global_handler.handle_error(error, context, critical)


def error_handler(
    reraise: bool = False, critical: bool = False, context: dict[str, Any] | None = None
):
    """Decorator for error handling."""
    return _global_handler.error_handler(reraise, critical, context)


def register_error_callback(callback: Callable) -> None:
    """Register an error callback."""
    _global_handler.register_error_callback(callback)


def register_critical_callback(callback: Callable) -> None:
    """Register a critical error callback."""
    _global_handler.register_critical_callback(callback)


def register_recovery_strategy(error_type: type, strategy: Callable) -> None:
    """Register a recovery strategy."""
    _global_handler.register_recovery_strategy(error_type, strategy)


def get_error_statistics() -> dict[str, Any]:
    """Get error statistics."""
    return _global_handler.get_statistics()


def install_exception_hook() -> None:
    """Install global exception hook."""
    _global_handler.install_exception_hook()
