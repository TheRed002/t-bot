"""
Service Registry for T-Bot Trading System.

This module provides a central registry for all system services,
enabling dependency injection and service discovery.
"""

import inspect
from abc import ABC, abstractmethod
from typing import Any

from src.base import BaseComponent


class ServiceInterface(ABC):
    """Base interface for all services."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        pass


class ServiceRegistry(BaseComponent):
    """Central registry for all system services."""

    def __init__(self):
        super().__init__()
        self._services: dict[str, Any] = {}
        self._interfaces: dict[str, type] = {}
        self._initialized: set = set()

    def register_service(self, name: str, service: Any, interface: type | None = None) -> None:
        """
        Register a service in the registry.

        Args:
            name: Service name
            service: Service instance
            interface: Optional service interface for type checking
        """
        if interface:
            if not self._implements_interface(service, interface):
                raise ValueError(
                    f"Service {name} does not implement interface {interface.__name__}"
                )
            self._interfaces[name] = interface

        self._services[name] = service
        self.logger.info(f"Registered service: {name}")

    def get_service(self, name: str) -> Any:
        """
        Get a service by name.

        Args:
            name: Service name

        Returns:
            Service instance

        Raises:
            KeyError: If service not found
        """
        if name not in self._services:
            raise KeyError(f"Service not found: {name}")
        return self._services[name]

    def has_service(self, name: str) -> bool:
        """Check if a service is registered."""
        return name in self._services

    async def initialize_all(self) -> None:
        """Initialize all registered services."""
        for name, service in self._services.items():
            if name not in self._initialized:
                try:
                    if hasattr(service, "initialize"):
                        await service.initialize()
                    self._initialized.add(name)
                    self.logger.info(f"Initialized service: {name}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize service {name}: {e}")
                    raise

    async def cleanup_all(self) -> None:
        """Cleanup all services."""
        for name, service in self._services.items():
            try:
                if hasattr(service, "cleanup"):
                    await service.cleanup()
                self.logger.info(f"Cleaned up service: {name}")
            except Exception as e:
                self.logger.error(f"Failed to cleanup service {name}: {e}")
        self._initialized.clear()

    def list_services(self) -> dict[str, str]:
        """List all registered services."""
        return {name: type(service).__name__ for name, service in self._services.items()}

    def get_all_service_names(self) -> list[str]:
        """Get all registered service names."""
        return list(self._services.keys())

    def _implements_interface(self, service: Any, interface: type) -> bool:
        """Check if a service implements the required interface."""
        if not inspect.isclass(interface) or not issubclass(interface, ABC):
            return True  # Not an abstract interface

        # Check if all abstract methods are implemented
        interface_methods = set()
        for method_name in dir(interface):
            method = getattr(interface, method_name)
            if getattr(method, "__isabstractmethod__", False):
                interface_methods.add(method_name)

        service_methods = set(dir(service))
        return interface_methods.issubset(service_methods)


# Global service registry instance
_global_registry: ServiceRegistry | None = None


def get_service_registry() -> ServiceRegistry:
    """Get or create the global service registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ServiceRegistry()
    return _global_registry


def register_service(name: str, service: Any, interface: type | None = None) -> None:
    """Register a service in the global registry."""
    registry = get_service_registry()
    registry.register_service(name, service, interface)


def get_service(name: str) -> Any:
    """Get a service from the global registry."""
    registry = get_service_registry()
    return registry.get_service(name)
