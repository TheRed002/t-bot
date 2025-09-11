"""
Tests for ServiceRegistry

Comprehensive test coverage for the data service registry including:
- Service registration and retrieval
- Service metadata management
- Event emission and subscription
- Error handling and cleanup
- Generic type handling
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from src.data.registry import ServiceRegistry


class MockService:
    """Mock service for testing."""

    def __init__(self, name: str):
        self.name = name

    def do_something(self):
        return f"Service {self.name} working"


class TestServiceRegistry:
    """Test ServiceRegistry class."""

    def test_initialization(self):
        """Test registry initialization."""
        registry = ServiceRegistry[MockService]()

        assert registry._services == {}
        assert registry._service_metadata == {}
        assert registry._event_handlers == {}

    def test_register_service_basic(self):
        """Test basic service registration."""
        registry = ServiceRegistry[MockService]()
        service = MockService("test-service")

        registry.register_service("test", service)

        assert "test" in registry._services
        assert registry._services["test"] == service
        assert "test" in registry._service_metadata
        assert registry._service_metadata["test"] == {}

    def test_register_service_with_metadata(self):
        """Test service registration with metadata."""
        registry = ServiceRegistry[MockService]()
        service = MockService("test-service")
        metadata = {"version": "1.0", "type": "mock"}

        registry.register_service("test", service, metadata)

        assert registry._service_metadata["test"] == metadata

    def test_register_service_replacement(self):
        """Test service replacement with warning."""
        registry = ServiceRegistry[MockService]()
        service1 = MockService("service1")
        service2 = MockService("service2")

        # Register first service
        registry.register_service("test", service1)

        # Replace service (should log warning)
        registry.register_service("test", service2)

        assert registry._services["test"] == service2

    def test_register_service_emits_event(self):
        """Test service registration emits event."""
        registry = ServiceRegistry[MockService]()
        service = MockService("test-service")
        metadata = {"version": "1.0"}

        # Mock event emission
        with patch.object(registry, "_emit_event") as mock_emit:
            registry.register_service("test", service, metadata)

            mock_emit.assert_called_once_with(
                "service.registered",
                {"service_name": "test", "metadata": metadata}
            )

    def test_get_service_existing(self):
        """Test getting existing service."""
        registry = ServiceRegistry[MockService]()
        service = MockService("test-service")

        registry.register_service("test", service)

        result = registry.get_service("test")
        assert result == service

    def test_get_service_not_found(self):
        """Test getting non-existent service."""
        registry = ServiceRegistry[MockService]()

        result = registry.get_service("nonexistent")

        assert result is None

    def test_unregister_service_existing(self):
        """Test unregistering existing service."""
        registry = ServiceRegistry[MockService]()
        service = MockService("test-service")
        metadata = {"version": "1.0"}

        registry.register_service("test", service, metadata)

        with patch.object(registry, "_emit_event") as mock_emit:
            result = registry.unregister_service("test")

            assert result is True
            assert "test" not in registry._services
            assert "test" not in registry._service_metadata

            mock_emit.assert_called_once_with(
                "service.unregistering",
                {"service_name": "test", "metadata": metadata}
            )

    def test_unregister_service_not_found(self):
        """Test unregistering non-existent service."""
        registry = ServiceRegistry[MockService]()

        result = registry.unregister_service("nonexistent")

        assert result is False

    def test_list_services_empty(self):
        """Test listing services when registry is empty."""
        registry = ServiceRegistry[MockService]()

        services = registry.list_services()

        assert services == {}

    def test_list_services_populated(self):
        """Test listing services when registry has services."""
        registry = ServiceRegistry[MockService]()
        service1 = MockService("service1")
        service2 = MockService("service2")
        metadata1 = {"version": "1.0"}
        metadata2 = {"version": "2.0"}

        registry.register_service("test1", service1, metadata1)
        registry.register_service("test2", service2, metadata2)

        services = registry.list_services()

        assert services == {
            "test1": metadata1,
            "test2": metadata2
        }

    def test_list_services_returns_dict_copy(self):
        """Test that list_services returns a dict copy."""
        registry = ServiceRegistry[MockService]()
        service = MockService("test-service")
        metadata = {"version": "1.0"}

        registry.register_service("test", service, metadata)

        services = registry.list_services()

        # Modify the returned dict structure
        services["new_service"] = {"test": "value"}

        # Original service registry should not have the new service
        assert "new_service" not in registry._service_metadata

        # But modifying metadata values will still affect original (shallow copy)
        services["test"]["new_field"] = "modified"
        assert "new_field" in registry._service_metadata["test"]

    def test_subscribe_to_event_new_event(self):
        """Test subscribing to new event."""
        registry = ServiceRegistry[MockService]()
        handler = Mock()

        registry.subscribe_to_event("test.event", handler)

        assert "test.event" in registry._event_handlers
        assert handler in registry._event_handlers["test.event"]

    def test_subscribe_to_event_existing_event(self):
        """Test subscribing to existing event."""
        registry = ServiceRegistry[MockService]()
        handler1 = Mock()
        handler2 = Mock()

        registry.subscribe_to_event("test.event", handler1)
        registry.subscribe_to_event("test.event", handler2)

        assert len(registry._event_handlers["test.event"]) == 2
        assert handler1 in registry._event_handlers["test.event"]
        assert handler2 in registry._event_handlers["test.event"]

    def test_emit_event_no_handlers(self):
        """Test emitting event with no handlers."""
        registry = ServiceRegistry[MockService]()

        # Should not raise exception
        registry._emit_event("nonexistent.event", {"data": "test"})

    def test_emit_event_with_handlers(self):
        """Test emitting event with handlers."""
        registry = ServiceRegistry[MockService]()
        handler1 = Mock()
        handler2 = Mock()
        event_data = {"service_name": "test", "data": "value"}

        registry.subscribe_to_event("test.event", handler1)
        registry.subscribe_to_event("test.event", handler2)

        registry._emit_event("test.event", event_data)

        handler1.assert_called_once_with(event_data)
        handler2.assert_called_once_with(event_data)

    def test_emit_event_handler_exception(self):
        """Test event emission with handler exception."""
        registry = ServiceRegistry[MockService]()
        handler_error = Mock(side_effect=Exception("Handler error"))
        handler_success = Mock()
        event_data = {"data": "test"}

        registry.subscribe_to_event("test.event", handler_error)
        registry.subscribe_to_event("test.event", handler_success)

        # Should not raise exception even with handler error
        registry._emit_event("test.event", event_data)

        # Error handler should fail but not prevent success handler
        handler_error.assert_called_once_with(event_data)
        handler_success.assert_called_once_with(event_data)

    @pytest.mark.asyncio
    async def test_cleanup_empty_registry(self):
        """Test cleanup of empty registry."""
        registry = ServiceRegistry[MockService]()

        await registry.cleanup()

        assert registry._services == {}
        assert registry._service_metadata == {}
        assert registry._event_handlers == {}

    @pytest.mark.asyncio
    async def test_cleanup_with_services(self):
        """Test cleanup of registry with services."""
        registry = ServiceRegistry[MockService]()
        service1 = MockService("service1")
        service2 = MockService("service2")
        metadata1 = {"version": "1.0"}
        metadata2 = {"version": "2.0"}

        registry.register_service("test1", service1, metadata1)
        registry.register_service("test2", service2, metadata2)

        # Add event handler
        handler = Mock()
        registry.subscribe_to_event("test.event", handler)

        with patch.object(registry, "_emit_event") as mock_emit:
            await registry.cleanup()

            # Should emit cleanup events for all services
            assert mock_emit.call_count == 2
            cleanup_calls = [call for call in mock_emit.call_args_list if call[0][0] == "service.cleanup"]
            assert len(cleanup_calls) == 2

        assert registry._services == {}
        assert registry._service_metadata == {}
        assert registry._event_handlers == {}

    @pytest.mark.asyncio
    async def test_cleanup_event_emission(self):
        """Test cleanup emits correct events."""
        registry = ServiceRegistry[MockService]()
        service = MockService("test-service")
        metadata = {"version": "1.0"}

        registry.register_service("test", service, metadata)

        with patch.object(registry, "_emit_event") as mock_emit:
            await registry.cleanup()

            # Check cleanup event was emitted
            cleanup_call = None
            for call in mock_emit.call_args_list:
                if call[0][0] == "service.cleanup":
                    cleanup_call = call
                    break

            assert cleanup_call is not None
            assert cleanup_call[0][1]["service_name"] == "test"
            assert cleanup_call[0][1]["metadata"] == metadata

    @pytest.mark.asyncio
    async def test_cleanup_exception_handling(self):
        """Test cleanup handles exceptions gracefully."""
        registry = ServiceRegistry[MockService]()
        service = MockService("test-service")

        registry.register_service("test", service)

        # Mock emit_event to raise exception
        with patch.object(registry, "_emit_event", side_effect=Exception("Emit error")):
            # Should not raise exception even with error
            await registry.cleanup()


class TestServiceRegistryGenericTypes:
    """Test generic type handling."""

    def test_generic_typing_mock_service(self):
        """Test registry with MockService type."""
        registry = ServiceRegistry[MockService]()
        service = MockService("typed-service")

        registry.register_service("typed", service)
        retrieved = registry.get_service("typed")

        assert isinstance(retrieved, MockService)
        assert retrieved.name == "typed-service"

    def test_generic_typing_any_type(self):
        """Test registry with Any type."""
        registry = ServiceRegistry[Any]()

        # Can store any type
        string_service = "string_service"
        dict_service = {"type": "dict"}

        registry.register_service("string", string_service)
        registry.register_service("dict", dict_service)

        assert registry.get_service("string") == string_service
        assert registry.get_service("dict") == dict_service

    def test_generic_typing_custom_class(self):
        """Test registry with custom class type."""

        class CustomService:
            def __init__(self, value: int):
                self.value = value

        registry = ServiceRegistry[CustomService]()
        service = CustomService(42)

        registry.register_service("custom", service)
        retrieved = registry.get_service("custom")

        assert isinstance(retrieved, CustomService)
        assert retrieved.value == 42


class TestServiceRegistryIntegration:
    """Integration tests for service registry."""

    def test_full_lifecycle(self):
        """Test full service lifecycle."""
        registry = ServiceRegistry[MockService]()
        service = MockService("lifecycle-service")
        metadata = {"version": "1.0", "type": "mock"}

        # Track events
        events_received = []

        def event_handler(event_data):
            events_received.append(event_data)

        registry.subscribe_to_event("service.registered", event_handler)
        registry.subscribe_to_event("service.unregistering", event_handler)

        # Register service
        registry.register_service("lifecycle", service, metadata)

        # Verify registration
        assert registry.get_service("lifecycle") == service
        assert registry.list_services()["lifecycle"] == metadata

        # Verify registration event
        assert len(events_received) == 1
        assert events_received[0]["service_name"] == "lifecycle"

        # Unregister service
        result = registry.unregister_service("lifecycle")

        # Verify unregistration
        assert result is True
        assert registry.get_service("lifecycle") is None
        assert "lifecycle" not in registry.list_services()

        # Verify unregistration event
        assert len(events_received) == 2
        assert events_received[1]["service_name"] == "lifecycle"

    def test_multiple_services_same_event(self):
        """Test multiple services subscribing to same event."""
        registry = ServiceRegistry[MockService]()

        # Create multiple event handlers
        handler1_calls = []
        handler2_calls = []

        def handler1(event_data):
            handler1_calls.append(event_data)

        def handler2(event_data):
            handler2_calls.append(event_data)

        registry.subscribe_to_event("service.registered", handler1)
        registry.subscribe_to_event("service.registered", handler2)

        # Register multiple services
        service1 = MockService("service1")
        service2 = MockService("service2")

        registry.register_service("test1", service1)
        registry.register_service("test2", service2)

        # Both handlers should receive both events
        assert len(handler1_calls) == 2
        assert len(handler2_calls) == 2

        # Verify event data
        assert handler1_calls[0]["service_name"] == "test1"
        assert handler1_calls[1]["service_name"] == "test2"
        assert handler2_calls[0]["service_name"] == "test1"
        assert handler2_calls[1]["service_name"] == "test2"

    @pytest.mark.asyncio
    async def test_cleanup_with_event_handlers(self):
        """Test cleanup preserves event functionality."""
        registry = ServiceRegistry[MockService]()

        # Setup services and handlers
        service = MockService("cleanup-test")
        cleanup_events = []

        def cleanup_handler(event_data):
            cleanup_events.append(event_data)

        registry.subscribe_to_event("service.cleanup", cleanup_handler)
        registry.register_service("cleanup-test", service)

        # Cleanup
        await registry.cleanup()

        # Verify cleanup event was received
        assert len(cleanup_events) == 1
        assert cleanup_events[0]["service_name"] == "cleanup-test"

        # Verify all data is cleared
        assert registry._services == {}
        assert registry._service_metadata == {}
        assert registry._event_handlers == {}


class TestDataServiceRegistryDI:
    """Test data service registry via dependency injection."""

    def test_registry_via_di_exists(self):
        """Test registry can be obtained via DI."""
        from src.data.di_registration import configure_data_dependencies, get_data_service_registry

        injector = configure_data_dependencies()
        registry = get_data_service_registry(injector)

        assert registry is not None
        assert isinstance(registry, ServiceRegistry)

    def test_registry_via_di_usable(self):
        """Test registry obtained via DI is usable."""
        from src.data.di_registration import configure_data_dependencies, get_data_service_registry

        injector = configure_data_dependencies()
        registry = get_data_service_registry(injector)

        service = MockService("di-test")

        registry.register_service("di-test", service)
        retrieved = registry.get_service("di-test")

        assert retrieved == service

        # Clean up
        registry.unregister_service("di-test")


class TestServiceRegistryEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_service_name(self):
        """Test registering service with empty name."""
        registry = ServiceRegistry[MockService]()
        service = MockService("empty-name-service")

        registry.register_service("", service)

        assert registry.get_service("") == service

    def test_none_service_registration(self):
        """Test registering None as service."""
        registry = ServiceRegistry[Any]()

        registry.register_service("none-service", None)

        assert registry.get_service("none-service") is None

    def test_special_characters_in_service_name(self):
        """Test service names with special characters."""
        registry = ServiceRegistry[MockService]()
        service = MockService("special-service")

        special_names = [
            "service.with.dots",
            "service-with-dashes",
            "service_with_underscores",
            "service/with/slashes",
            "service with spaces",
            "service@with@symbols",
        ]

        for name in special_names:
            registry.register_service(name, service)
            assert registry.get_service(name) == service

    def test_large_metadata(self):
        """Test registration with large metadata."""
        registry = ServiceRegistry[MockService]()
        service = MockService("large-metadata-service")

        # Create large metadata
        large_metadata = {
            f"key_{i}": f"value_{i}" * 100
            for i in range(1000)
        }

        registry.register_service("large-metadata", service, large_metadata)

        assert registry._service_metadata["large-metadata"] == large_metadata

    def test_concurrent_operations_simulation(self):
        """Test simulated concurrent operations."""
        registry = ServiceRegistry[MockService]()

        # Simulate rapid registration/unregistration
        for i in range(100):
            service = MockService(f"service_{i}")
            registry.register_service(f"service_{i}", service)

        # Verify all services registered
        assert len(registry._services) == 100

        # Unregister odd numbered services
        for i in range(1, 100, 2):
            registry.unregister_service(f"service_{i}")

        # Verify correct services remain
        assert len(registry._services) == 50
        for i in range(0, 100, 2):
            assert f"service_{i}" in registry._services

    def test_event_handler_modification_during_emission(self):
        """Test event handler list modification during emission."""
        registry = ServiceRegistry[MockService]()

        # Handler that adds another handler
        def self_modifying_handler(event_data):
            def new_handler(event_data):
                pass
            registry.subscribe_to_event("test.event", new_handler)

        registry.subscribe_to_event("test.event", self_modifying_handler)

        # Should not cause issues
        registry._emit_event("test.event", {"data": "test"})

        # Verify handler was added
        assert len(registry._event_handlers["test.event"]) == 2

    def test_memory_cleanup_after_many_operations(self):
        """Test memory is properly cleaned up after many operations."""
        registry = ServiceRegistry[MockService]()

        # Perform many operations
        for cycle in range(10):
            # Register many services
            for i in range(100):
                service = MockService(f"service_{cycle}_{i}")
                metadata = {"cycle": cycle, "index": i}
                registry.register_service(f"service_{cycle}_{i}", service, metadata)

            # Add event handlers
            for event in ["event1", "event2", "event3"]:
                handler = lambda data: None
                registry.subscribe_to_event(f"{event}_{cycle}", handler)

            # Clear all services
            service_names = list(registry._services.keys())
            for name in service_names:
                registry.unregister_service(name)

        # Verify final state
        assert len(registry._services) == 0
        assert len(registry._service_metadata) == 0
        # Event handlers remain (this is expected behavior)
        assert len(registry._event_handlers) > 0
