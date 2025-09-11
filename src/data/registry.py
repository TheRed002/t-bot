"""
Data Service Registry - Event-Driven Service Discovery

This module provides a registry pattern for data services to reduce coupling
and enable event-driven communication between data services.
"""

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from src.core.base import BaseComponent

ServiceType = TypeVar("ServiceType")


class ServiceRegistry(BaseComponent, Generic[ServiceType]):
    """
    Generic service registry for managing service instances and dependencies.

    This registry helps break circular dependencies by providing a central
    location for service discovery and event-based communication.
    """

    def __init__(self):
        """Initialize the service registry."""
        super().__init__()
        self._services: dict[str, ServiceType] = {}
        self._service_metadata: dict[str, dict[str, Any]] = {}
        self._event_handlers: dict[str, list[Callable[[dict[str, Any]], None]]] = {}

    def register_service(
        self, name: str, service: ServiceType, metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Register a service in the registry.

        Args:
            name: Service name/identifier
            service: Service instance
            metadata: Optional service metadata
        """
        if name in self._services:
            self.logger.warning(f"Service {name} is already registered, replacing existing")

        self._services[name] = service
        self._service_metadata[name] = metadata or {}

        self.logger.info(f"Registered service: {name}")

        # Emit service registered event
        self._emit_event(
            "service.registered", {"service_name": name, "metadata": self._service_metadata[name]}
        )

    def get_service(self, name: str) -> ServiceType | None:
        """
        Get a service from the registry.

        Args:
            name: Service name/identifier

        Returns:
            Service instance or None if not found
        """
        service = self._services.get(name)
        if service is None:
            self.logger.warning(f"Service {name} not found in registry")

        return service

    def unregister_service(self, name: str) -> bool:
        """
        Unregister a service from the registry.

        Args:
            name: Service name/identifier

        Returns:
            True if service was unregistered, False if not found
        """
        if name not in self._services:
            return False

        # Emit service unregistering event
        self._emit_event(
            "service.unregistering",
            {"service_name": name, "metadata": self._service_metadata[name]},
        )

        del self._services[name]
        del self._service_metadata[name]

        self.logger.info(f"Unregistered service: {name}")
        return True

    def list_services(self) -> dict[str, dict[str, Any]]:
        """
        List all registered services with their metadata.

        Returns:
            Dictionary mapping service names to their metadata
        """
        return dict(self._service_metadata)

    def subscribe_to_event(self, event_name: str, handler: Callable[[dict[str, Any]], None]) -> None:
        """
        Subscribe to registry events.

        Args:
            event_name: Name of the event to subscribe to
            handler: Event handler function
        """
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []

        self._event_handlers[event_name].append(handler)

    def _emit_event(self, event_name: str, event_data: dict[str, Any]) -> None:
        """
        Emit an event to all subscribers.

        Args:
            event_name: Name of the event
            event_data: Event data
        """
        handlers = self._event_handlers.get(event_name, [])
        for handler in handlers:
            try:
                handler(event_data)
            except Exception as e:
                self.logger.error(f"Error in event handler for {event_name}: {e}")

    async def cleanup(self) -> None:
        """Cleanup registry resources."""
        try:
            # Emit cleanup events for all services
            for service_name in list(self._services.keys()):
                self._emit_event(
                    "service.cleanup",
                    {
                        "service_name": service_name,
                        "metadata": self._service_metadata[service_name],
                    },
                )

            self._services.clear()
            self._service_metadata.clear()
            self._event_handlers.clear()

            self.logger.info("Service registry cleanup completed")

        except Exception as e:
            self.logger.error(f"Service registry cleanup error: {e}")


# Note: Use dependency injection to get registry instance instead of global
# data_service_registry: ServiceRegistry[Any] = ServiceRegistry()
