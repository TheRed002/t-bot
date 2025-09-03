"""
Dependency injection container for monitoring services.

This module provides a proper dependency injection implementation
to replace the service locator anti-pattern with constructor injection.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from src.core import get_logger
from src.core.exceptions import MonitoringError, ServiceError

if TYPE_CHECKING:
    from src.monitoring.alerting import Alert, AlertManager
    from src.monitoring.dashboards import GrafanaDashboardManager
    from src.monitoring.interfaces import (
        AlertServiceInterface,
        DashboardServiceInterface,
        MetricsServiceInterface,
        MonitoringServiceInterface,
        PerformanceServiceInterface,
    )
    from src.monitoring.metrics import MetricsCollector
    from src.monitoring.performance import PerformanceProfiler

logger = get_logger(__name__)

T = TypeVar("T")


def create_factory(cls: type[T], *deps: type) -> Callable[[], T]:
    """
    Create a simple factory function that resolves dependencies from the container.
    
    Args:
        cls: The class to instantiate
        *deps: Dependency types to resolve from container
        
    Returns:
        Factory function that creates instances with resolved dependencies
    """
    def factory() -> T:
        container = get_monitoring_container()
        resolved_deps = []
        for dep_type in deps:
            try:
                resolved_deps.append(container.resolve(dep_type))
            except (KeyError, ValueError):
                # Skip optional dependencies
                pass
        return cls(*resolved_deps)
    return factory


class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collection."""

    def increment_counter(
        self, name: str, labels: dict[str, str] | None = None, value: float = 1.0, namespace: str = "tbot"
    ) -> None: ...

    def set_gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None, namespace: str = "tbot"
    ) -> None: ...

    def observe_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None, namespace: str = "tbot"
    ) -> None: ...


class AlertManagerProtocol(Protocol):
    """Protocol for alert management."""

    async def fire_alert(self, alert: Alert) -> None: ...

    async def resolve_alert(self, fingerprint: str) -> None: ...


class PerformanceProfilerProtocol(Protocol):
    """Protocol for performance profiling."""

    def start_operation(self, name: str, metadata: dict[str, Any] | None = None) -> Any: ...
    def end_operation(self, operation_id: str) -> None: ...
    def record_metric(
        self, name: str, value: float, tags: dict[str, str] | None = None
    ) -> None: ...


@dataclass
class ServiceBinding:
    """Represents a service binding in the DI container."""

    interface: type
    implementation: type | None = None
    factory: Callable[..., Any] | None = None
    instance: Any | None = None
    singleton: bool = True
    dependencies: list[type] = field(default_factory=list)


class DIContainer:
    """
    Dependency injection container for monitoring services.

    This container supports:
    - Constructor injection
    - Singleton and transient lifetimes
    - Factory functions
    - Automatic dependency resolution
    """

    def __init__(self):
        """Initialize the DI container."""
        self._bindings: dict[type, ServiceBinding] = {}
        self._resolving: set[type] = set()  # Track circular dependencies

    def register(
        self,
        interface: type[T],
        implementation: type[T] | None = None,
        factory: Callable[..., T] | None = None,
        singleton: bool = True,
    ) -> None:
        """
        Register a service binding.

        Args:
            interface: The interface/protocol type
            implementation: The concrete implementation class
            factory: Optional factory function
            singleton: Whether to create singleton instance

        Raises:
            ValueError: If neither implementation nor factory provided
        """
        if implementation is None and factory is None:
            raise ValueError("Must provide either implementation or factory")

        # Extract dependencies from constructor
        dependencies = []
        if implementation:
            sig = inspect.signature(implementation.__init__)
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param.annotation != inspect.Parameter.empty:
                    dependencies.append(param.annotation)

        binding = ServiceBinding(
            interface=interface,
            implementation=implementation,
            factory=factory,
            singleton=singleton,
            dependencies=dependencies,
        )

        self._bindings[interface] = binding
        impl_name = getattr(implementation, '__name__', 'factory') if implementation else "factory"
        interface_name = getattr(interface, '__name__', str(interface))
        logger.debug(f"Registered {interface_name} -> {impl_name}")

    def resolve(self, interface: type[T]) -> T:
        """
        Resolve a service instance.

        Args:
            interface: The interface type to resolve

        Returns:
            Instance of the requested service

        Raises:
            ValueError: If service not registered or circular dependency detected
        """
        if interface not in self._bindings:
            raise ValueError(f"No binding registered for {interface.__name__}")

        if interface in self._resolving:
            raise ValueError(f"Circular dependency detected for {interface.__name__}")

        binding = self._bindings[interface]

        # Return existing singleton if available
        if binding.singleton and binding.instance is not None:
            return binding.instance

        try:
            self._resolving.add(interface)

            if binding.factory:
                # Use factory function
                instance = binding.factory()
            else:
                # Resolve dependencies
                resolved_deps: dict[str, Any] = {}
                for dep_type in binding.dependencies:
                    if dep_type in self._bindings and binding.implementation is not None:
                        param_name = self._get_param_name(binding.implementation, dep_type)
                        resolved_deps[param_name] = self.resolve(dep_type)

                # Create instance with resolved dependencies
                if binding.implementation is not None:
                    instance = binding.implementation(**resolved_deps)
                else:
                    raise ValueError(f"No implementation found for {interface}")

            # Store singleton instance
            if binding.singleton:
                binding.instance = instance

            return instance

        finally:
            self._resolving.remove(interface)

    def _get_param_name(self, cls: type, param_type: type) -> str:
        """Get parameter name for a given type in constructor."""
        init_method = cls.__init__
        sig = inspect.signature(init_method)
        for param_name, param in sig.parameters.items():
            if param.annotation == param_type:
                return param_name
        raise ValueError(f"No parameter of type {param_type.__name__} in {cls.__name__}")

    def clear(self) -> None:
        """Clear all bindings and instances."""
        self._bindings.clear()
        self._resolving.clear()


# Global container instance
_container = DIContainer()


def get_monitoring_container() -> DIContainer:
    """Get the monitoring-specific DI container instance."""
    return _container


def setup_monitoring_dependencies() -> None:
    """
    Configure default monitoring dependencies using proper constructor injection.

    This function sets up the standard bindings for monitoring services using
    proper dependency injection with constructor parameters.
    """
    from src.monitoring.alerting import AlertManager, NotificationConfig
    from src.monitoring.dashboards import GrafanaDashboardManager
    from src.monitoring.interfaces import (
        AlertServiceInterface,
        DashboardServiceInterface,
        MetricsServiceInterface,
        MonitoringServiceInterface,
        PerformanceServiceInterface,
    )
    from src.monitoring.metrics import MetricsCollector
    from src.monitoring.performance import PerformanceProfiler
    from src.monitoring.services import (
        DefaultAlertService,
        DefaultDashboardService,
        DefaultMetricsService,
        DefaultPerformanceService,
        MonitoringService,
    )

    container = get_monitoring_container()

    # Register MetricsCollector with factory for proper initialization
    def metrics_collector_factory() -> MetricsCollector:
        """Factory function for creating MetricsCollector with proper configuration."""
        return MetricsCollector()

    container.register(MetricsCollector, factory=metrics_collector_factory, singleton=True)

    # AlertManager requires NotificationConfig - use factory pattern for complex construction
    def alert_manager_factory() -> AlertManager:
        """Factory function for creating AlertManager with proper configuration."""
        return AlertManager(NotificationConfig())

    container.register(AlertManager, factory=alert_manager_factory, singleton=True)

    # PerformanceProfiler with factory for complex dependency injection
    def performance_profiler_factory() -> PerformanceProfiler:
        """Factory function for creating PerformanceProfiler with injected dependencies."""
        metrics_collector = container.resolve(MetricsCollector)
        alert_manager = container.resolve(AlertManager)

        return PerformanceProfiler(
            metrics_collector=metrics_collector,
            alert_manager=alert_manager
        )

    container.register(PerformanceProfiler, factory=performance_profiler_factory, singleton=True)

    # Register GrafanaDashboardManager with environment-based config
    def dashboard_manager_factory() -> GrafanaDashboardManager:
        """Factory function for creating GrafanaDashboardManager with environment config."""
        import os

        from src.monitoring.dashboards import GrafanaDashboardManager

        grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3000")
        api_key = os.getenv("GRAFANA_API_KEY", "")

        # Try to inject error handler if available from DI container
        error_handler = None
        try:
            error_handler = container.resolve("ErrorHandler")
        except (KeyError, ValueError):
            # Error handler is optional dependency
            pass

        return GrafanaDashboardManager(grafana_url, api_key, error_handler)

    container.register(GrafanaDashboardManager, factory=dashboard_manager_factory, singleton=True)

    # Register service implementations using factory pattern for proper dependency injection
    container.register(DefaultMetricsService, factory=create_factory(DefaultMetricsService, MetricsCollector), singleton=True)
    container.register(DefaultAlertService, factory=create_factory(DefaultAlertService, AlertManager), singleton=True)
    container.register(DefaultPerformanceService, factory=create_factory(DefaultPerformanceService, PerformanceProfiler), singleton=True)
    container.register(DefaultDashboardService, factory=create_factory(DefaultDashboardService, GrafanaDashboardManager), singleton=True)

    # Register service interfaces to implementations using factory pattern for proper delegation
    def metrics_service_interface_factory() -> MetricsServiceInterface:
        return container.resolve(DefaultMetricsService)

    def alert_service_interface_factory() -> AlertServiceInterface:
        return container.resolve(DefaultAlertService)

    def performance_service_interface_factory() -> PerformanceServiceInterface:
        return container.resolve(DefaultPerformanceService)

    def dashboard_service_interface_factory() -> DashboardServiceInterface:
        return container.resolve(DefaultDashboardService)

    container.register(MetricsServiceInterface, factory=metrics_service_interface_factory, singleton=True)
    container.register(AlertServiceInterface, factory=alert_service_interface_factory, singleton=True)
    container.register(PerformanceServiceInterface, factory=performance_service_interface_factory, singleton=True)
    container.register(DashboardServiceInterface, factory=dashboard_service_interface_factory, singleton=True)

    # Register composite monitoring service using factory for dependency injection
    def monitoring_service_factory() -> MonitoringService:
        alert_service = container.resolve(AlertServiceInterface)
        metrics_service = container.resolve(MetricsServiceInterface)
        performance_service = container.resolve(PerformanceServiceInterface)
        return MonitoringService(alert_service, metrics_service, performance_service)

    container.register(MonitoringService, factory=monitoring_service_factory, singleton=True)

    def monitoring_service_interface_factory() -> MonitoringServiceInterface:
        return container.resolve(MonitoringService)

    container.register(MonitoringServiceInterface, factory=monitoring_service_interface_factory, singleton=True)

    logger.info("Monitoring dependencies configured with constructor injection")


# Factory functions following proper dependency injection pattern
def create_metrics_collector() -> MetricsCollector:
    """Create MetricsCollector instance using DI container."""
    from src.monitoring.metrics import MetricsCollector
    container = get_monitoring_container()
    try:
        return container.resolve(MetricsCollector)
    except (ServiceError, MonitoringError, KeyError, ValueError) as e:
        logger.warning(f"Failed to resolve MetricsCollector from DI container: {e}")
        return MetricsCollector()


def create_alert_manager() -> AlertManager:
    """Create AlertManager instance using DI container."""
    from src.monitoring.alerting import AlertManager, NotificationConfig
    container = get_monitoring_container()
    try:
        return container.resolve(AlertManager)
    except (ServiceError, MonitoringError, KeyError, ValueError) as e:
        logger.warning(f"Failed to resolve AlertManager from DI container: {e}")
        return AlertManager(NotificationConfig())


def create_performance_profiler() -> PerformanceProfiler:
    """Create PerformanceProfiler instance using DI container."""
    from src.monitoring.alerting import AlertManager, NotificationConfig
    from src.monitoring.metrics import MetricsCollector
    from src.monitoring.performance import PerformanceProfiler
    container = get_monitoring_container()
    try:
        return container.resolve(PerformanceProfiler)
    except (ServiceError, MonitoringError, KeyError, ValueError) as e:
        logger.warning(f"Failed to resolve PerformanceProfiler from DI container: {e}")
        # Create dependencies through DI container
        try:
            metrics_collector = container.resolve(MetricsCollector)
            alert_manager = container.resolve(AlertManager)
            return PerformanceProfiler(metrics_collector=metrics_collector, alert_manager=alert_manager)
        except Exception:
            # Final fallback with required dependencies
            from src.monitoring.metrics import MetricsCollector
            from src.monitoring.performance import PerformanceProfiler
            return PerformanceProfiler(
                metrics_collector=MetricsCollector(),
                alert_manager=AlertManager(NotificationConfig())
            )


def create_monitoring_service():
    """Create monitoring service using DI container."""
    container = get_monitoring_container()
    try:
        return container.resolve(MonitoringServiceInterface)
    except (ServiceError, MonitoringError, KeyError, ValueError) as e:
        logger.warning(f"Failed to resolve MonitoringServiceInterface from DI container: {e}")
        # Use DI to resolve service dependencies instead of creating them directly
        try:
            alert_service = container.resolve(AlertServiceInterface)
            metrics_service = container.resolve(MetricsServiceInterface)
            performance_service = container.resolve(PerformanceServiceInterface)
            from src.monitoring.services import MonitoringService
            return MonitoringService(alert_service, metrics_service, performance_service)
        except Exception:
            # Create services through container with proper dependency chain
            from src.monitoring.services import (
                DefaultAlertService,
                DefaultMetricsService,
                DefaultPerformanceService,
                MonitoringService,
            )
            try:
                alert_service = DefaultAlertService(container.resolve(AlertManager))
                metrics_service = DefaultMetricsService(container.resolve(MetricsCollector))
                performance_service = DefaultPerformanceService(container.resolve(PerformanceProfiler))
                return MonitoringService(alert_service, metrics_service, performance_service)
            except Exception:
                # Final fallback - direct instantiation
                alert_service = create_alert_service()
                metrics_service = create_metrics_service()
                performance_service = create_performance_service()
                return MonitoringService(alert_service, metrics_service, performance_service)


def create_alert_service():
    """Create alert service using DI container."""
    container = get_monitoring_container()
    try:
        return container.resolve(AlertServiceInterface)
    except (ServiceError, MonitoringError, KeyError, ValueError) as e:
        logger.warning(f"Failed to resolve AlertServiceInterface from DI container: {e}")
        # Use DI to resolve alert manager dependency
        from src.monitoring.alerting import AlertManager, NotificationConfig
        from src.monitoring.services import DefaultAlertService
        try:
            alert_manager = container.resolve(AlertManager)
            return DefaultAlertService(alert_manager)
        except Exception:
            # Final fallback with dependency injection
            return DefaultAlertService(AlertManager(NotificationConfig()))


def create_metrics_service():
    """Create metrics service using DI container."""
    container = get_monitoring_container()
    try:
        return container.resolve(MetricsServiceInterface)
    except (ServiceError, MonitoringError, KeyError, ValueError) as e:
        logger.warning(f"Failed to resolve MetricsServiceInterface from DI container: {e}")
        # Use DI to resolve metrics collector dependency
        from src.monitoring.metrics import MetricsCollector
        from src.monitoring.services import DefaultMetricsService
        try:
            metrics_collector = container.resolve(MetricsCollector)
            return DefaultMetricsService(metrics_collector)
        except Exception:
            # Final fallback with dependency injection
            return DefaultMetricsService(MetricsCollector())


def create_performance_service():
    """Create performance service using DI container."""
    container = get_monitoring_container()
    try:
        return container.resolve(PerformanceServiceInterface)
    except (ServiceError, MonitoringError, KeyError, ValueError) as e:
        logger.warning(f"Failed to resolve PerformanceServiceInterface from DI container: {e}")
        from src.monitoring.services import DefaultPerformanceService
        # Use DI to resolve performance profiler dependency
        try:
            performance_profiler = container.resolve(PerformanceProfiler)
            return DefaultPerformanceService(performance_profiler)
        except Exception:
            # Final fallback - create profiler through factory
            performance_profiler = create_performance_profiler()
            return DefaultPerformanceService(performance_profiler)


def create_dashboard_service():
    """Create dashboard service using DI container."""
    container = get_monitoring_container()
    try:
        return container.resolve(DashboardServiceInterface)
    except (ServiceError, MonitoringError, KeyError, ValueError) as e:
        logger.warning(f"Failed to resolve DashboardServiceInterface from DI container: {e}")
        from src.monitoring.services import DefaultDashboardService
        # Use DI to resolve dashboard manager dependency
        try:
            dashboard_manager = container.resolve(GrafanaDashboardManager)
            return DefaultDashboardService(dashboard_manager)
        except Exception:
            # Final fallback - create manager through factory
            dashboard_manager = create_dashboard_manager()
            return DefaultDashboardService(dashboard_manager)


def create_dashboard_manager() -> GrafanaDashboardManager:
    """Create dashboard manager using DI container."""
    import os

    from src.monitoring.dashboards import GrafanaDashboardManager
    container = get_monitoring_container()
    try:
        return container.resolve(GrafanaDashboardManager)
    except (ServiceError, MonitoringError, KeyError, ValueError) as e:
        logger.warning(f"Failed to resolve GrafanaDashboardManager from DI container: {e}")
        grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3000")
        api_key = os.getenv("GRAFANA_API_KEY", "")
        return GrafanaDashboardManager(grafana_url, api_key)
