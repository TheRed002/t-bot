"""
Base component implementation providing common functionality.

This module contains the enhanced BaseComponent class that replaces the
basic implementation in src/base.py with comprehensive lifecycle management,
health checks, and monitoring capabilities.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from src.core.base.interfaces import (
    Configurable,
    HealthCheckable,
    HealthCheckResult,
    HealthStatus,
    Injectable,
    Lifecycle,
    Loggable,
    Monitorable,
)
from src.core.exceptions import ComponentError, ConfigurationError
from src.core.logging import get_logger
from src.core.types.base import ConfigDict


class BaseComponent(
    Lifecycle,
    HealthCheckable,
    Injectable,
    Loggable,
    Monitorable,
    Configurable,
):
    """
    Enhanced base component with complete lifecycle management.

    Provides:
    - Async lifecycle management (start/stop/restart)
    - Health check framework
    - Dependency injection support
    - Structured logging with correlation IDs
    - Metrics collection
    - Configuration management
    - Resource cleanup
    - Error handling with context

    Example:
        ```python
        class MyComponent(BaseComponent):
            async def _do_start(self):
                # Custom startup logic
                await self.initialize_resources()

            async def _do_stop(self):
                # Custom cleanup logic
                await self.cleanup_resources()

            async def _health_check_internal(self):
                # Component-specific health checks
                if self.is_healthy():
                    return HealthStatus.HEALTHY
                return HealthStatus.UNHEALTHY
        ```
    """

    def __init__(
        self,
        name: str | None = None,
        config: ConfigDict | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize base component.

        Args:
            name: Component name for logging and identification
            config: Initial configuration dictionary
            correlation_id: Request correlation ID for tracing
        """
        self._name = name or self.__class__.__name__
        self._correlation_id = correlation_id or str(uuid.uuid4())
        self._logger = get_logger(self.__class__.__module__).bind(
            component=self._name,
            correlation_id=self._correlation_id,
        )

        # Lifecycle state
        self._is_running = False
        self._is_starting = False
        self._is_stopping = False
        self._start_time: datetime | None = None
        self._stop_time: datetime | None = None

        # Configuration
        self._config: ConfigDict = config or {}

        # Dependencies
        self._dependencies: set[str] = set()
        self._dependency_container: Any | None = None

        # Metrics
        self._metrics: dict[str, Any] = {
            "start_count": 0,
            "stop_count": 0,
            "restart_count": 0,
            "error_count": 0,
            "health_check_count": 0,
            "uptime_seconds": 0,
        }

        # Health check cache
        self._last_health_check: HealthCheckResult | None = None
        self._health_check_interval = 30  # seconds

        self._logger.debug("Component initialized", name=self._name)

    @property
    def name(self) -> str:
        """Get component name."""
        return self._name

    @property
    def logger(self) -> Any:
        """Get logger instance for this component."""
        return self._logger

    @property
    def correlation_id(self) -> str:
        """Get correlation ID for request tracing."""
        return self._correlation_id

    @property
    def is_running(self) -> bool:
        """Check if component is currently running."""
        return self._is_running

    @property
    def is_starting(self) -> bool:
        """Check if component is currently starting."""
        return self._is_starting

    @property
    def is_stopping(self) -> bool:
        """Check if component is currently stopping."""
        return self._is_stopping

    @property
    def uptime(self) -> float:
        """Get component uptime in seconds."""
        if not self._start_time:
            return 0.0
        end_time = self._stop_time or datetime.now(timezone.utc)
        return (end_time - self._start_time).total_seconds()

    # Lifecycle Management
    async def start(self) -> None:
        """
        Start the component and initialize resources.

        Raises:
            ComponentError: If component fails to start
        """
        if self._is_running:
            self._logger.warning("Component already running", name=self._name)
            return

        if self._is_starting:
            self._logger.warning("Component already starting", name=self._name)
            return

        self._is_starting = True
        self._start_time = datetime.now(timezone.utc)

        try:
            self._logger.info("Starting component", name=self._name)

            # Validate configuration before starting
            if not self.validate_config(self._config):
                raise ConfigurationError(f"Invalid configuration for {self._name}")

            # Perform component-specific startup
            await self._do_start()

            self._is_running = True
            self._metrics["start_count"] += 1

            self._logger.info("Component started successfully", name=self._name)

        except Exception as e:
            self._metrics["error_count"] += 1
            self._logger.error(
                "Failed to start component",
                name=self._name,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ComponentError(f"Failed to start {self._name}: {e}") from e

        finally:
            self._is_starting = False

    async def stop(self) -> None:
        """
        Stop the component and cleanup resources.

        Raises:
            ComponentError: If component fails to stop gracefully
        """
        if not self._is_running:
            self._logger.warning("Component not running", name=self._name)
            return

        if self._is_stopping:
            self._logger.warning("Component already stopping", name=self._name)
            return

        self._is_stopping = True
        self._stop_time = datetime.now(timezone.utc)

        try:
            self._logger.info("Stopping component", name=self._name)

            # Perform component-specific cleanup
            await self._do_stop()

            self._is_running = False
            self._metrics["stop_count"] += 1
            self._metrics["uptime_seconds"] += self.uptime

            self._logger.info("Component stopped successfully", name=self._name)

        except Exception as e:
            self._metrics["error_count"] += 1
            self._logger.error(
                "Error during component shutdown",
                name=self._name,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ComponentError(f"Failed to stop {self._name}: {e}") from e

        finally:
            self._is_stopping = False

    async def restart(self) -> None:
        """
        Restart the component.

        Raises:
            ComponentError: If component fails to restart
        """
        self._logger.info("Restarting component", name=self._name)

        if self._is_running:
            await self.stop()

        await asyncio.sleep(0.1)  # Brief pause between stop and start
        await self.start()

        self._metrics["restart_count"] += 1
        self._logger.info("Component restarted successfully", name=self._name)

    # Health Check Framework
    async def health_check(self) -> HealthCheckResult:
        """
        Perform comprehensive health check.

        Returns:
            HealthCheckResult: Current health status
        """
        self._metrics["health_check_count"] += 1
        check_time = datetime.now(timezone.utc)

        try:
            # Basic health checks
            if not self._is_running:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="Component is not running",
                    check_time=check_time,
                )

            if self._is_starting or self._is_stopping:
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message="Component is transitioning state",
                    check_time=check_time,
                )

            # Component-specific health check
            internal_status = await self._health_check_internal()

            # Combine with system metrics
            details = {
                "uptime_seconds": self.uptime,
                "metrics": self._metrics.copy(),
                "configuration_valid": self.validate_config(self._config),
            }

            result = HealthCheckResult(
                status=internal_status,
                details=details,
                message=f"Component {self._name} health check completed",
                check_time=check_time,
            )

            self._last_health_check = result
            return result

        except Exception as e:
            self._metrics["error_count"] += 1
            self._logger.error(
                "Health check failed",
                name=self._name,
                error=str(e),
                error_type=type(e).__name__,
            )

            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check error: {e}",
                details={"error": str(e), "error_type": type(e).__name__},
                check_time=check_time,
            )

    async def ready_check(self) -> HealthCheckResult:
        """
        Check if component is ready to serve requests.

        Returns:
            HealthCheckResult: Readiness status
        """
        if not self._is_running or self._is_starting or self._is_stopping:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message="Component is not ready",
            )

        # Component-specific readiness check
        return await self._readiness_check_internal()

    async def live_check(self) -> HealthCheckResult:
        """
        Check if component is alive and responsive.

        Returns:
            HealthCheckResult: Liveness status
        """
        try:
            # Basic responsiveness test
            start_time = datetime.now(timezone.utc)
            await asyncio.sleep(0.001)  # Minimal async operation
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            if response_time > 1.0:  # More than 1 second is concerning
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message="Component responding slowly",
                    details={"response_time_seconds": response_time},
                )

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Component is responsive",
                details={"response_time_seconds": response_time},
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Liveness check failed: {e}",
            )

    # Configuration Management
    def configure(self, config: ConfigDict) -> None:
        """
        Configure component with provided settings.

        Args:
            config: Configuration dictionary

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not self.validate_config(config):
            raise ConfigurationError(f"Invalid configuration for {self._name}")

        old_config = self._config.copy()
        self._config.update(config)

        self._logger.info(
            "Component configuration updated",
            name=self._name,
            changes=set(config.keys()) - set(old_config.keys()),
        )

        # Allow component to react to configuration changes
        self._on_config_changed(old_config, self._config)

    def get_config(self) -> ConfigDict:
        """Get current component configuration."""
        return self._config.copy()

    def validate_config(self, config: ConfigDict) -> bool:
        """
        Validate configuration settings.

        Args:
            config: Configuration to validate

        Returns:
            bool: True if configuration is valid
        """
        # Basic validation - override in subclasses for specific validation
        return isinstance(config, dict)

    # Dependency Injection
    def configure_dependencies(self, container: Any) -> None:
        """Configure component dependencies."""
        self._dependency_container = container
        self._logger.debug(
            "Dependencies configured",
            name=self._name,
            dependencies=list(self._dependencies),
        )

    def get_dependencies(self) -> list[str]:
        """Get list of required dependencies."""
        return list(self._dependencies)

    def add_dependency(self, dependency_name: str) -> None:
        """Add a dependency requirement."""
        self._dependencies.add(dependency_name)

    def remove_dependency(self, dependency_name: str) -> None:
        """Remove a dependency requirement."""
        self._dependencies.discard(dependency_name)

    # Metrics and Monitoring
    def get_metrics(self) -> dict[str, Any]:
        """Get current component metrics."""
        current_metrics = self._metrics.copy()
        current_metrics.update(
            {
                "is_running": self._is_running,
                "uptime_current_seconds": self.uptime if self._is_running else 0,
                "last_health_check": (
                    self._last_health_check.to_dict() if self._last_health_check else None
                ),
            }
        )
        return current_metrics

    def reset_metrics(self) -> None:
        """Reset component metrics."""
        self._metrics = {
            "start_count": 0,
            "stop_count": 0,
            "restart_count": 0,
            "error_count": 0,
            "health_check_count": 0,
            "uptime_seconds": 0,
        }
        self._logger.info("Component metrics reset", name=self._name)

    # Context Manager Support
    @asynccontextmanager
    async def lifecycle_context(self):
        """Async context manager for automatic lifecycle management."""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()

    # Protected methods for subclass customization
    async def _do_start(self) -> None:
        """Override in subclasses for custom startup logic."""
        pass

    async def _do_stop(self) -> None:
        """Override in subclasses for custom cleanup logic."""
        pass

    async def _health_check_internal(self) -> HealthStatus:
        """Override in subclasses for component-specific health checks."""
        return HealthStatus.HEALTHY

    async def _readiness_check_internal(self) -> HealthCheckResult:
        """Override in subclasses for component-specific readiness checks."""
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="Component is ready",
        )

    def _on_config_changed(self, old_config: ConfigDict, new_config: ConfigDict) -> None:
        """Override in subclasses to react to configuration changes."""
        pass

    # String representation
    def __repr__(self) -> str:
        """String representation of component."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self._name}', "
            f"running={self._is_running}, "
            f"uptime={self.uptime:.2f}s"
            f")"
        )


# Legacy compatibility - enhanced version of the original BaseComponent
class EnhancedBaseComponent(BaseComponent):
    """
    Enhanced version of the original BaseComponent with backward compatibility.

    This class maintains the original API while adding new functionality.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with backward compatibility."""
        super().__init__(*args, **kwargs)
        self._initialized = False  # Original property

    @property
    def initialized(self) -> bool:
        """Check if component is initialized (legacy compatibility)."""
        return self._is_running

    def initialize(self) -> None:
        """Initialize the component (legacy compatibility)."""
        # This should be replaced with async start() in new code
        self._initialized = True
        self.logger.debug(f"{self.__class__.__name__} initialized")

    def cleanup(self) -> None:
        """Cleanup the component (legacy compatibility)."""
        # This should be replaced with async stop() in new code
        self._initialized = False
        self.logger.debug(f"{self.__class__.__name__} cleaned up")
