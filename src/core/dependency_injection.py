"""Dependency injection system to break circular dependencies."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from threading import Lock
from typing import Any, TypeVar

from src.core.exceptions import ComponentError, DependencyError
from src.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class DependencyContainer:
    """Container for managing dependencies."""

    def __init__(self) -> None:
        """Initialize dependency container."""
        self._services: dict[str, Any] = {}
        self._factories: dict[str, Callable] = {}
        self._singletons: dict[str, Any] = {}
        self._lock = Lock()
        self._logger = logger

    def register(self, name: str, service: Any | Callable, singleton: bool = False) -> None:
        """
        Register a service or factory.

        Args:
            name: Service name
            service: Service instance or factory function
            singleton: Whether to treat as singleton
        """
        with self._lock:
            if callable(service) and not inspect.isclass(service):
                # Register as factory
                self._factories[name] = service
                if singleton:
                    self._singletons[name] = None
            else:
                # Register as service
                self._services[name] = service
                if singleton:
                    self._singletons[name] = service

            self._logger.debug(f"Registered service: {name} (singleton={singleton})")

    def register_class(
        self, name: str, cls: type[T], *args, singleton: bool = False, **kwargs
    ) -> None:
        """
        Register a class for lazy instantiation.

        Args:
            name: Service name
            cls: Class to instantiate
            *args: Positional arguments for instantiation
            singleton: Whether to treat as singleton
            **kwargs: Keyword arguments for instantiation
        """

        def factory():
            return cls(*args, **kwargs)

        self.register(name, factory, singleton=singleton)

    def get(self, name: str) -> Any:
        """
        Get a service by name.

        Args:
            name: Service name

        Returns:
            Service instance

        Raises:
            DependencyError: If service not found or instantiation fails
        """
        with self._lock:
            # Check if it's a direct service
            if name in self._services:
                return self._services[name]

            # Check if it's a singleton that's already created
            if name in self._singletons and self._singletons[name] is not None:
                return self._singletons[name]

            # Check if it's a factory
            if name in self._factories:
                try:
                    instance = self._factories[name]()

                    # Cache if singleton
                    if name in self._singletons:
                        self._singletons[name] = instance

                    return instance
                except Exception as e:
                    raise DependencyError(
                        f"Failed to instantiate service '{name}' from factory",
                        dependency_name=name,
                        error_code="DEP_003",
                        suggested_action="Check factory function implementation and dependencies",
                        context={"factory_error": str(e)},
                    ) from e

            raise DependencyError(
                f"Service '{name}' not registered",
                dependency_name=name,
                error_code="DEP_001",
                suggested_action="Register the service before attempting to resolve it",
                context={
                    "available_services": list(self._services.keys()) + list(self._factories.keys())
                },
            )

    def has(self, name: str) -> bool:
        """Check if service is registered."""
        return name in self._services or name in self._factories

    def clear(self) -> None:
        """Clear all registered services."""
        with self._lock:
            self._services.clear()
            self._factories.clear()
            self._singletons.clear()


class DependencyInjector:
    """
    Dependency injector for automatic dependency resolution.

    This eliminates circular dependencies and manual wiring.
    """

    _instance: DependencyInjector | None = None
    _lock = Lock()

    def __new__(cls) -> DependencyInjector:
        """Singleton pattern with proper thread safety."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize dependency injector."""
        with self._lock:
            if not hasattr(self, "_initialized"):
                self._container = DependencyContainer()
                self._logger = logger
                self._initialized = True

    def register(self, name: str | None = None, singleton: bool = False):
        """
        Decorator to register a service.

        Args:
            name: Service name (defaults to class name)
            singleton: Whether to treat as singleton
        """

        def decorator(cls_or_func):
            service_name = name or cls_or_func.__name__

            if inspect.isclass(cls_or_func):
                # Register class
                self._container.register_class(service_name, cls_or_func, singleton=singleton)
            else:
                # Register function/instance
                self._container.register(service_name, cls_or_func, singleton=singleton)

            return cls_or_func

        return decorator

    def inject(self, func: Callable) -> Callable:
        """
        Decorator to inject dependencies into function.

        Dependencies are resolved by parameter name.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Get function signature
                sig = inspect.signature(func)

                # Resolve dependencies
                for param_name, param in sig.parameters.items():
                    # Skip if already provided
                    if param_name in kwargs:
                        continue

                    # Try to resolve from container
                    try:
                        if self._container.has(param_name):
                            kwargs[param_name] = self._container.get(param_name)
                        elif param.annotation != param.empty:
                            # Try to resolve by type annotation
                            type_name = param.annotation.__name__
                            if self._container.has(type_name):
                                kwargs[param_name] = self._container.get(type_name)
                    except DependencyError as e:
                        # Re-raise with additional context about the injection
                        raise DependencyError(
                            f"Failed to inject dependency '{param_name}' into function "
                            f"'{func.__name__}'",
                            dependency_name=param_name,
                            error_code="DEP_004",
                            suggested_action=(
                                "Ensure dependency is registered before calling function"
                            ),
                            context={
                                "function_name": func.__name__,
                                "parameter_name": param_name,
                                "original_error": str(e),
                            },
                        ) from e

                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, DependencyError):
                    raise
                # Wrap other exceptions with context
                raise ComponentError(
                    f"Error executing function '{func.__name__}' with dependency injection",
                    component_name=func.__name__,
                    error_code="COMP_001",
                    suggested_action="Check function implementation and dependencies",
                    context={"function_error": str(e)},
                ) from e

        return wrapper

    def resolve(self, name: str) -> Any:
        """
        Resolve a dependency by name.

        Args:
            name: Service name

        Returns:
            Service instance
        Raises:
            DependencyError: If service cannot be resolved
        """
        try:
            return self._container.get(name)
        except Exception as e:
            if isinstance(e, DependencyError):
                raise
            raise DependencyError(
                f"Unexpected error resolving dependency '{name}'",
                dependency_name=name,
                error_code="DEP_005",
                suggested_action="Check service registration and container state",
                context={"resolution_error": str(e)},
            ) from e

    def register_service(self, name: str, service: Any, singleton: bool = False) -> None:
        """
        Register a service directly.

        Args:
            name: Service name
            service: Service instance or factory
            singleton: Whether to treat as singleton

        Raises:
            DependencyError: If registration fails
        """
        try:
            self._container.register(name, service, singleton=singleton)
        except Exception as e:
            raise DependencyError(
                f"Failed to register service '{name}'",
                dependency_name=name,
                error_code="DEP_006",
                suggested_action="Check service implementation and registration parameters",
                context={"service_type": type(service).__name__, "registration_error": str(e)},
            ) from e

    def register_factory(self, name: str, factory: Callable, singleton: bool = False) -> None:
        """
        Register a factory function.

        Args:
            name: Service name
            factory: Factory function
            singleton: Whether to treat as singleton

        Raises:
            DependencyError: If registration fails
        """
        try:
            if not callable(factory):
                raise ValueError("Factory must be callable")
            self._container.register(name, factory, singleton=singleton)
        except Exception as e:
            raise DependencyError(
                f"Failed to register factory '{name}'",
                dependency_name=name,
                error_code="DEP_007",
                suggested_action="Ensure factory is a valid callable function",
                context={"factory_type": type(factory).__name__, "registration_error": str(e)},
            ) from e

    def has_service(self, name: str) -> bool:
        """Check if service is registered."""
        return self._container.has(name)

    def clear(self) -> None:
        """Clear all registered services."""
        self._container.clear()

    @classmethod
    def get_instance(cls) -> DependencyInjector:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_container(self) -> DependencyContainer:
        """Get the dependency container."""
        return self._container


# Global injector instance
injector = DependencyInjector.get_instance()


# Decorators for convenience
def injectable(name: str | None = None, singleton: bool = False):
    """Decorator to mark a class as injectable."""
    return injector.register(name=name, singleton=singleton)


def inject(func: Callable) -> Callable:
    """Decorator to inject dependencies into a function."""
    return injector.inject(func)


# Production usage examples moved to documentation


# Service locator pattern support
class ServiceLocator:
    """Service locator for easy access to services."""

    def __init__(self, injector: DependencyInjector):
        self._injector = injector

    def __getattr__(self, name: str) -> Any:
        """Get service by attribute access."""
        try:
            return self._injector.resolve(name)
        except DependencyError as e:
            raise AttributeError(f"Service '{name}' not found") from e
        except Exception as e:
            raise DependencyError(
                f"Failed to resolve service '{name}'",
                dependency_name=name,
                error_code="DEP_002",
                suggested_action="Check service registration and dependencies",
            ) from e


# Global service locator
services = ServiceLocator(injector)


def get_container() -> DependencyContainer:
    """Get the global dependency container."""
    return injector.get_container()
