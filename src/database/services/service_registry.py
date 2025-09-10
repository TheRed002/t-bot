"""Service registry for managing service layer dependencies."""

from collections.abc import Callable
from typing import Any, TypeVar

from src.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class ServiceRegistry:
    """Registry for managing service instances and dependencies."""

    def __init__(self) -> None:
        """Initialize service registry."""
        self._services: dict[str, Any] = {}
        self._service_factories: dict[str, Callable[[], Any]] = {}

    def register_service(self, name: str, service_instance: Any) -> None:
        """
        Register a service instance.

        Args:
            name: Service name
            service_instance: Service instance to register
        """
        self._services[name] = service_instance
        logger.info(f"Service registered: {name}")

    def register_factory(self, name: str, factory: Callable[[], Any]) -> None:
        """
        Register a service factory.

        Args:
            name: Service name
            factory: Factory function to create service
        """
        self._service_factories[name] = factory
        logger.info(f"Service factory registered: {name}")

    def get_service(self, name: str) -> Any:
        """
        Get service by name, creating it if necessary.

        Args:
            name: Service name

        Returns:
            Service instance

        Raises:
            KeyError: If service not found
        """
        # Return existing service if available
        if name in self._services:
            return self._services[name]

        # Create service using factory if available
        if name in self._service_factories:
            factory = self._service_factories[name]
            service = factory()
            self._services[name] = service
            logger.info(f"Service created using factory: {name}")
            return service

        raise KeyError(f"Service not found: {name}")

    def has_service(self, name: str) -> bool:
        """
        Check if service is available.

        Args:
            name: Service name

        Returns:
            True if service is available
        """
        return name in self._services or name in self._service_factories

    def clear_services(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._service_factories.clear()
        logger.info("All services cleared from registry")

    def list_services(self) -> list[str]:
        """
        List all available service names.

        Returns:
            List of service names
        """
        return list(set(self._services.keys()) | set(self._service_factories.keys()))


# Global service registry instance
service_registry = ServiceRegistry()
