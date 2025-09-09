"""
Shared utilities for core module patterns.

This module provides common functionality extracted from core module duplications:
- Dependency injection patterns
- Connection management
- Resource lifecycle management
- Health check utilities
- Logging helpers

These utilities promote DRY principles while maintaining proper separation of concerns.
"""

import asyncio
import inspect
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, TypeVar, get_type_hints

from src.core.base.interfaces import HealthStatus
from src.core.exceptions import (
    ComponentError,
    ConnectionError as CoreConnectionError,
    DependencyError,
)
from src.core.logging import get_logger

T = TypeVar("T")


class DependencyInjectionMixin:
    """Mixin providing common dependency injection patterns."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dependency_container: Any | None = None
        self._auto_inject = True

    def configure_dependencies(self, container: Any) -> None:
        """
        Configure component dependencies.

        Args:
            container: Dependency injection container
        """
        # Support both DependencyContainer and DependencyInjector
        if hasattr(container, "get_container"):
            # This is a DependencyInjector, get the actual container
            self._dependency_container = container.get_container()
        else:
            # This is a DependencyContainer
            self._dependency_container = container

        self._auto_inject = True  # Enable auto-injection when container is available

        if hasattr(self, "_logger"):
            self._logger.debug(
                "Dependencies configured", component=getattr(self, "_name", self.__class__.__name__)
            )

    def get_dependencies(self) -> list[str]:
        """
        Get list of required dependencies.

        Returns:
            List of dependency names
        """
        return []  # Override in subclasses as needed

    def _resolve_dependency(self, type_name: str, param_name: str) -> Any:
        """
        Resolve dependency using service locator pattern.

        Args:
            type_name: Type name to resolve
            param_name: Parameter name to resolve

        Returns:
            Resolved dependency instance

        Raises:
            DependencyError: If dependency cannot be resolved
        """
        if not self._dependency_container:
            raise DependencyError("No dependency container configured")

        # Try resolving by type name first
        if hasattr(self._dependency_container, "get") and type_name in self._dependency_container:
            return self._dependency_container.get(type_name)
        elif hasattr(self._dependency_container, "resolve"):
            try:
                return self._dependency_container.resolve(type_name)
            except Exception:
                pass  # Try parameter name fallback

        # Try resolving by parameter name
        if hasattr(self._dependency_container, "get") and param_name in self._dependency_container:
            return self._dependency_container.get(param_name)
        elif hasattr(self._dependency_container, "resolve"):
            return self._dependency_container.resolve(param_name)

        raise DependencyError(f"Cannot resolve dependency: {type_name} or {param_name}")

    def _inject_dependencies_into_kwargs(
        self, target_callable: Callable, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Inject dependencies into keyword arguments for a callable.

        Args:
            target_callable: The callable to inject dependencies for
            kwargs: Current keyword arguments

        Returns:
            Updated keyword arguments with injected dependencies
        """
        if not self._auto_inject or not self._dependency_container:
            return kwargs

        try:
            # Get callable signature
            if inspect.isclass(target_callable):
                signature = inspect.signature(target_callable.__init__)
            else:
                signature = inspect.signature(target_callable)

            # Get type hints
            try:
                type_hints = get_type_hints(target_callable)
            except (NameError, AttributeError):
                type_hints = {}

            # Inject dependencies based on parameter types
            for param_name, param in signature.parameters.items():
                if param_name not in kwargs and param_name != "self" and param_name in type_hints:
                    param_type = type_hints[param_name]

                    try:
                        # Use service locator pattern for dependency resolution
                        dependency = self._resolve_dependency(param_type.__name__, param_name)
                        kwargs[param_name] = dependency

                        if hasattr(self, "_logger"):
                            self._logger.debug(
                                "Dependency injected",
                                parameter=param_name,
                                dependency_type=param_type.__name__,
                            )

                    except Exception as e:
                        # Dependency injection is optional, log for debugging
                        if hasattr(self, "_logger"):
                            self._logger.debug(
                                "Failed to inject dependency (optional)",
                                parameter=param_name,
                                error=str(e),
                                error_type=type(e).__name__,
                            )
                        continue

            return kwargs

        except Exception as e:
            if hasattr(self, "_logger"):
                self._logger.warning("Dependency injection failed", error=str(e))
            return kwargs


class ConnectionManagerMixin:
    """Mixin providing common connection management patterns."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._connection_lock = asyncio.Lock()
        self._connection_retries = 0
        self._max_connection_retries = 3
        self._connection_retry_delay = 1.0
        self._shutdown_requested = False

    async def _ensure_connection(self, client_attr: str, reconnect_method: str) -> None:
        """
        Ensure connection is available with proper error handling.

        Args:
            client_attr: Attribute name for the client object
            reconnect_method: Method name to call for reconnection
        """
        if self._shutdown_requested:
            raise ComponentError("Component is shutting down")

        async with self._connection_lock:
            client = getattr(self, client_attr, None)

            # Check if client is connected and healthy
            if client is None or not await self._is_connection_healthy(client):
                reconnect_func = getattr(self, reconnect_method)
                await reconnect_func()

    async def _is_connection_healthy(self, client: Any) -> bool:
        """
        Check if connection is healthy. Override in subclasses.

        Args:
            client: Client object to check

        Returns:
            True if healthy, False otherwise
        """
        return client is not None

    async def _reconnect_with_backoff(self, connect_method: str, test_method: str = "ping") -> None:
        """
        Reconnect with exponential backoff.

        Args:
            connect_method: Method name to call for connection
            test_method: Method name to call for testing connection
        """
        logger = getattr(self, "_logger", None) or get_logger(self.__class__.__name__)

        for attempt in range(self._max_connection_retries):
            try:
                connect_func = getattr(self, connect_method)
                await connect_func()

                # Test the connection if test method is available
                if hasattr(self, test_method):
                    test_func = getattr(self, test_method)
                    await asyncio.wait_for(test_func(), timeout=2.0)

                self._connection_retries = 0  # Reset on success
                logger.info("Successfully reconnected")
                return

            except Exception as e:
                self._connection_retries += 1
                wait_time = self._connection_retry_delay * (2**attempt)

                if attempt < self._max_connection_retries - 1:
                    logger.warning(
                        f"Connection attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All connection attempts failed: {e}")
                    raise CoreConnectionError(
                        f"Failed to connect after {self._max_connection_retries} attempts: {e}"
                    ) from e


class LifecycleManagerMixin:
    """Mixin providing common lifecycle management patterns."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._start_time: datetime | None = None
        self._stop_time: datetime | None = None
        self._is_running = False
        self._shutdown_requested = False
        self._background_tasks: set[asyncio.Task] = set()

    async def _start_lifecycle(self) -> None:
        """Start lifecycle management."""
        if self._is_running:
            return

        self._start_time = datetime.now(timezone.utc)
        self._is_running = True
        self._shutdown_requested = False

        logger = getattr(self, "_logger", None) or get_logger(self.__class__.__name__)
        logger.debug("Lifecycle started", component=getattr(self, "_name", self.__class__.__name__))

    async def _stop_lifecycle(self) -> None:
        """Stop lifecycle management with cleanup."""
        if not self._is_running:
            return

        self._shutdown_requested = True
        self._stop_time = datetime.now(timezone.utc)

        # Cancel background tasks
        await self._cancel_background_tasks()

        self._is_running = False

        logger = getattr(self, "_logger", None) or get_logger(self.__class__.__name__)
        logger.debug("Lifecycle stopped", component=getattr(self, "_name", self.__class__.__name__))

    def _create_background_task(self, coro, name: str | None = None) -> asyncio.Task:
        """
        Create and track background task.

        Args:
            coro: Coroutine to run as background task
            name: Optional task name

        Returns:
            Created task
        """
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)

        # Remove task from set when done
        def cleanup_task(t: asyncio.Task) -> None:
            self._background_tasks.discard(t)

        task.add_done_callback(cleanup_task)
        return task

    async def _cancel_background_tasks(self) -> None:
        """Cancel all background tasks gracefully."""
        if not self._background_tasks:
            return

        logger = getattr(self, "_logger", None) or get_logger(self.__class__.__name__)
        logger.debug(f"Canceling {len(self._background_tasks)} background tasks")

        # Cancel all tasks
        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()

        # Wait for cancellation with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._background_tasks, return_exceptions=True), timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning("Background task cancellation timed out")

        self._background_tasks.clear()

    @property
    def uptime(self) -> float | None:
        """Get component uptime in seconds."""
        if self._start_time is None:
            return None

        end_time = self._stop_time or datetime.now(timezone.utc)
        return (end_time - self._start_time).total_seconds()


class HealthCheckMixin:
    """Mixin providing common health check patterns."""

    async def basic_health_check(self) -> HealthStatus:
        """
        Perform basic health check.

        Returns:
            Health status
        """
        try:
            # Check if component is running (if lifecycle is supported)
            if hasattr(self, "_is_running") and not self._is_running:
                return HealthStatus.UNHEALTHY

            # Check if shutdown was requested
            if hasattr(self, "_shutdown_requested") and self._shutdown_requested:
                return HealthStatus.UNHEALTHY

            # Check connection health if connection is supported
            if hasattr(self, "_is_connection_healthy"):
                client = getattr(self, "_client", None) or getattr(self, "client", None)
                if client and not await self._is_connection_healthy(client):
                    return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            logger = getattr(self, "_logger", None) or get_logger(self.__class__.__name__)
            logger.error("Health check failed", error=str(e))
            return HealthStatus.UNHEALTHY


class ResourceCleanupMixin:
    """Mixin providing common resource cleanup patterns."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cleanup_callbacks: list[Callable] = []

    def register_cleanup_callback(self, callback: Callable) -> None:
        """
        Register cleanup callback.

        Args:
            callback: Cleanup function to call on shutdown
        """
        self._cleanup_callbacks.append(callback)

    async def _cleanup_resources(self) -> None:
        """Execute all registered cleanup callbacks."""
        logger = getattr(self, "_logger", None) or get_logger(self.__class__.__name__)

        for callback in reversed(self._cleanup_callbacks):  # Reverse order for cleanup
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.warning(f"Cleanup callback failed: {e}")

        self._cleanup_callbacks.clear()


class LoggingHelperMixin:
    """Mixin providing structured logging helpers."""

    def _log_operation_start(self, operation: str, **context) -> datetime:
        """
        Log operation start with context.

        Args:
            operation: Operation name
            **context: Additional context

        Returns:
            Start time for duration calculation
        """
        logger = getattr(self, "_logger", None) or get_logger(self.__class__.__name__)
        start_time = datetime.now(timezone.utc)

        logger.debug(
            f"{operation} started",
            operation=operation,
            component=getattr(self, "_name", self.__class__.__name__),
            **context,
        )

        return start_time

    def _log_operation_success(self, operation: str, start_time: datetime, **context) -> None:
        """
        Log operation success with duration.

        Args:
            operation: Operation name
            start_time: Operation start time
            **context: Additional context
        """
        logger = getattr(self, "_logger", None) or get_logger(self.__class__.__name__)
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        logger.info(
            f"{operation} completed successfully",
            operation=operation,
            component=getattr(self, "_name", self.__class__.__name__),
            duration_seconds=duration,
            **context,
        )

    def _log_operation_error(
        self, operation: str, start_time: datetime, error: Exception, **context
    ) -> None:
        """
        Log operation error with duration.

        Args:
            operation: Operation name
            start_time: Operation start time
            error: Exception that occurred
            **context: Additional context
        """
        logger = getattr(self, "_logger", None) or get_logger(self.__class__.__name__)
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        logger.error(
            f"{operation} failed",
            operation=operation,
            component=getattr(self, "_name", self.__class__.__name__),
            duration_seconds=duration,
            error=str(error),
            error_type=type(error).__name__,
            **context,
        )


# Utility context managers
@asynccontextmanager
async def managed_connection(
    connection_manager: Any, client_attr: str, reconnect_method: str
) -> AsyncGenerator[Any, None]:
    """
    Context manager for ensuring connection availability.

    Args:
        connection_manager: Object with connection management capabilities
        client_attr: Attribute name for the client
        reconnect_method: Method name for reconnection
    """
    await connection_manager._ensure_connection(client_attr, reconnect_method)
    try:
        yield getattr(connection_manager, client_attr)
    except Exception:
        # Connection might be lost, will be handled on next call
        raise


@asynccontextmanager
async def operation_logging(
    logger_mixin: Any, operation: str, **context: Any
) -> AsyncGenerator[None, None]:
    """
    Context manager for operation logging.

    Args:
        logger_mixin: Object with logging capabilities
        operation: Operation name
        **context: Additional context
    """
    start_time = logger_mixin._log_operation_start(operation, **context)
    try:
        yield
        logger_mixin._log_operation_success(operation, start_time, **context)
    except Exception as e:
        logger_mixin._log_operation_error(operation, start_time, e, **context)
        raise


class BaseUtilityMixin(
    DependencyInjectionMixin,
    ConnectionManagerMixin,
    LifecycleManagerMixin,
    HealthCheckMixin,
    ResourceCleanupMixin,
    LoggingHelperMixin,
):
    """Combined utility mixin with all common patterns."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    async def initialize(self) -> None:
        """Initialize component with all mixins."""
        await self._start_lifecycle()

    async def shutdown(self) -> None:
        """Shutdown component with cleanup."""
        await self._cleanup_resources()
        await self._stop_lifecycle()
