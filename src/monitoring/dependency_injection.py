"""
Dependency injection container for monitoring services.

This module provides a proper dependency injection implementation
to replace the service locator anti-pattern with constructor injection.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from src.core import get_logger
from src.monitoring.alerting import AlertManager
from src.monitoring.metrics import MetricsCollector
from src.monitoring.performance import PerformanceProfiler

logger = get_logger(__name__)

T = TypeVar("T")


class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collection."""

    async def increment_counter(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None: ...

    async def set_gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None: ...

    async def observe_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None: ...


class AlertManagerProtocol(Protocol):
    """Protocol for alert management."""

    async def fire_alert(
        self,
        rule_name: str,
        description: str,
        labels: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
    ) -> None: ...

    async def resolve_alert(self, rule_name: str, labels: dict[str, str] | None = None) -> None: ...


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
        impl_name = implementation.__name__ if implementation else "factory"
        logger.debug(f"Registered {interface.__name__} -> {impl_name}")

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
                resolved_deps = {}
                for dep_type in binding.dependencies:
                    if dep_type in self._bindings:
                        param_name = self._get_param_name(binding.implementation, dep_type)
                        resolved_deps[param_name] = self.resolve(dep_type)

                # Create instance with resolved dependencies
                instance = binding.implementation(**resolved_deps)

            # Store singleton instance
            if binding.singleton:
                binding.instance = instance

            return instance

        finally:
            self._resolving.remove(interface)

    def _get_param_name(self, cls: type, param_type: type) -> str:
        """Get parameter name for a given type in constructor."""
        sig = inspect.signature(cls.__init__)
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


def get_container() -> DIContainer:
    """Get the global DI container instance."""
    return _container


def setup_monitoring_dependencies() -> None:
    """
    Configure default monitoring dependencies.

    This function sets up the standard bindings for monitoring services.
    """
    container = get_container()

    # Register core monitoring services
    container.register(MetricsCollectorProtocol, MetricsCollector)
    container.register(AlertManagerProtocol, AlertManager)
    container.register(PerformanceProfilerProtocol, PerformanceProfiler)

    logger.info("Monitoring dependencies configured")


# Convenience functions for migration from service locator pattern
def create_metrics_collector() -> MetricsCollector:
    """Create a new MetricsCollector instance using DI."""
    return get_container().resolve(MetricsCollector)


def create_alert_manager() -> AlertManager:
    """Create a new AlertManager instance using DI."""
    return get_container().resolve(AlertManager)


def create_performance_profiler(
    metrics_collector: MetricsCollectorProtocol | None = None,
    alert_manager: AlertManagerProtocol | None = None,
) -> PerformanceProfiler:
    """
    Create a new PerformanceProfiler instance using DI.

    Args:
        metrics_collector: Optional metrics collector override
        alert_manager: Optional alert manager override

    Returns:
        New PerformanceProfiler instance
    """
    if metrics_collector or alert_manager:
        # Create with specific dependencies
        return PerformanceProfiler(metrics_collector=metrics_collector, alert_manager=alert_manager)
    else:
        # Use DI container
        return get_container().resolve(PerformanceProfiler)
