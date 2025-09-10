"""Tests for database service registry."""

import pytest
from unittest.mock import Mock

from src.database.services.service_registry import ServiceRegistry, service_registry


class TestServiceRegistry:
    """Test the ServiceRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a fresh ServiceRegistry instance for testing."""
        return ServiceRegistry()

    @pytest.fixture
    def mock_service(self):
        """Create a mock service for testing."""
        return Mock()

    @pytest.fixture
    def mock_factory(self, mock_service):
        """Create a mock factory function that returns a service."""
        return lambda: mock_service

    def test_init(self, registry):
        """Test ServiceRegistry initialization."""
        assert registry._services == {}
        assert registry._service_factories == {}

    def test_register_service(self, registry, mock_service):
        """Test registering a service instance."""
        # Act
        registry.register_service("test_service", mock_service)

        # Assert
        assert "test_service" in registry._services
        assert registry._services["test_service"] == mock_service

    def test_register_factory(self, registry, mock_factory):
        """Test registering a service factory."""
        # Act
        registry.register_factory("test_service", mock_factory)

        # Assert
        assert "test_service" in registry._service_factories
        assert registry._service_factories["test_service"] == mock_factory

    def test_get_service_existing_instance(self, registry, mock_service):
        """Test getting an existing service instance."""
        # Arrange
        registry.register_service("test_service", mock_service)

        # Act
        result = registry.get_service("test_service")

        # Assert
        assert result == mock_service

    def test_get_service_from_factory(self, registry, mock_service, mock_factory):
        """Test getting a service created from factory."""
        # Arrange
        registry.register_factory("test_service", mock_factory)

        # Act
        result = registry.get_service("test_service")

        # Assert
        assert result == mock_service
        # Service should now be cached
        assert "test_service" in registry._services
        assert registry._services["test_service"] == mock_service

    def test_get_service_factory_called_once(self, registry, mock_service):
        """Test that factory is only called once for service creation."""
        # Arrange
        call_count = 0
        def factory():
            nonlocal call_count
            call_count += 1
            return mock_service

        registry.register_factory("test_service", factory)

        # Act
        result1 = registry.get_service("test_service")
        result2 = registry.get_service("test_service")

        # Assert
        assert call_count == 1  # Factory should only be called once
        assert result1 == mock_service
        assert result2 == mock_service
        assert result1 is result2  # Same instance

    def test_get_service_not_found(self, registry):
        """Test getting a service that doesn't exist."""
        # Act & Assert
        with pytest.raises(KeyError, match="Service not found: nonexistent_service"):
            registry.get_service("nonexistent_service")

    def test_get_service_prefers_existing_instance_over_factory(self, registry, mock_service):
        """Test that existing instance is preferred over factory."""
        # Arrange
        existing_service = Mock()
        factory_service = Mock()
        factory = lambda: factory_service

        registry.register_service("test_service", existing_service)
        registry.register_factory("test_service", factory)

        # Act
        result = registry.get_service("test_service")

        # Assert
        assert result == existing_service
        assert result != factory_service

    def test_has_service_with_instance(self, registry, mock_service):
        """Test checking if service exists when instance is registered."""
        # Arrange
        registry.register_service("test_service", mock_service)

        # Act & Assert
        assert registry.has_service("test_service") is True
        assert registry.has_service("nonexistent") is False

    def test_has_service_with_factory(self, registry, mock_factory):
        """Test checking if service exists when factory is registered."""
        # Arrange
        registry.register_factory("test_service", mock_factory)

        # Act & Assert
        assert registry.has_service("test_service") is True
        assert registry.has_service("nonexistent") is False

    def test_has_service_with_both(self, registry, mock_service, mock_factory):
        """Test checking if service exists when both instance and factory are registered."""
        # Arrange
        registry.register_service("test_service", mock_service)
        registry.register_factory("test_service", mock_factory)

        # Act & Assert
        assert registry.has_service("test_service") is True

    def test_clear_services(self, registry, mock_service, mock_factory):
        """Test clearing all services."""
        # Arrange
        registry.register_service("service1", mock_service)
        registry.register_factory("service2", mock_factory)

        # Act
        registry.clear_services()

        # Assert
        assert registry._services == {}
        assert registry._service_factories == {}
        assert registry.has_service("service1") is False
        assert registry.has_service("service2") is False

    def test_list_services_empty(self, registry):
        """Test listing services when registry is empty."""
        # Act
        result = registry.list_services()

        # Assert
        assert result == []

    def test_list_services_with_instances(self, registry, mock_service):
        """Test listing services with registered instances."""
        # Arrange
        registry.register_service("service1", mock_service)
        registry.register_service("service2", Mock())

        # Act
        result = registry.list_services()

        # Assert
        assert set(result) == {"service1", "service2"}

    def test_list_services_with_factories(self, registry, mock_factory):
        """Test listing services with registered factories."""
        # Arrange
        registry.register_factory("service1", mock_factory)
        registry.register_factory("service2", lambda: Mock())

        # Act
        result = registry.list_services()

        # Assert
        assert set(result) == {"service1", "service2"}

    def test_list_services_with_both(self, registry, mock_service, mock_factory):
        """Test listing services with both instances and factories."""
        # Arrange
        registry.register_service("service1", mock_service)
        registry.register_factory("service2", mock_factory)
        registry.register_service("service3", Mock())
        registry.register_factory("service4", lambda: Mock())

        # Act
        result = registry.list_services()

        # Assert
        assert set(result) == {"service1", "service2", "service3", "service4"}

    def test_list_services_with_duplicate_names(self, registry, mock_service, mock_factory):
        """Test listing services with same name in both instance and factory."""
        # Arrange
        registry.register_service("service1", mock_service)
        registry.register_factory("service1", mock_factory)  # Same name
        registry.register_service("service2", Mock())

        # Act
        result = registry.list_services()

        # Assert
        assert set(result) == {"service1", "service2"}  # No duplicates

    def test_register_service_overwrites_existing(self, registry, mock_service):
        """Test that registering a service overwrites existing one."""
        # Arrange
        original_service = Mock()
        new_service = mock_service
        registry.register_service("test_service", original_service)

        # Act
        registry.register_service("test_service", new_service)

        # Assert
        assert registry.get_service("test_service") == new_service
        assert registry.get_service("test_service") != original_service

    def test_register_factory_overwrites_existing(self, registry):
        """Test that registering a factory overwrites existing one."""
        # Arrange
        original_service = Mock()
        new_service = Mock()
        original_factory = lambda: original_service
        new_factory = lambda: new_service
        
        registry.register_factory("test_service", original_factory)

        # Act
        registry.register_factory("test_service", new_factory)

        # Assert
        result = registry.get_service("test_service")
        assert result == new_service
        assert result != original_service

    def test_factory_exception_propagated(self, registry):
        """Test that exceptions from factory are propagated."""
        # Arrange
        def failing_factory():
            raise ValueError("Factory failed")

        registry.register_factory("test_service", failing_factory)

        # Act & Assert
        with pytest.raises(ValueError, match="Factory failed"):
            registry.get_service("test_service")

    def test_service_registry_behavior_after_factory_creation(self, registry, mock_service):
        """Test service registry behavior after a service is created from factory."""
        # Arrange
        factory = lambda: mock_service
        registry.register_factory("test_service", factory)

        # Act - Create service from factory
        result1 = registry.get_service("test_service")
        
        # Register a new factory with same name
        new_service = Mock()
        new_factory = lambda: new_service
        registry.register_factory("test_service", new_factory)
        
        # Get service again
        result2 = registry.get_service("test_service")

        # Assert
        assert result1 == mock_service
        assert result2 == mock_service  # Should still return cached instance, not new factory result
        assert result2 != new_service


class TestGlobalServiceRegistry:
    """Test the global service registry instance."""

    def test_global_service_registry_exists(self):
        """Test that global service registry exists."""
        assert service_registry is not None
        assert isinstance(service_registry, ServiceRegistry)

    def test_global_service_registry_is_singleton(self):
        """Test that global service registry is a singleton."""
        # Import again to get the same instance
        from src.database.services.service_registry import service_registry as registry2
        assert service_registry is registry2

    def test_global_service_registry_functionality(self):
        """Test basic functionality of global service registry."""
        # Arrange
        mock_service = Mock()
        
        # Clean up any existing services first
        original_services = service_registry._services.copy()
        original_factories = service_registry._service_factories.copy()
        
        try:
            service_registry.clear_services()
            
            # Act
            service_registry.register_service("test_global_service", mock_service)
            
            # Assert
            assert service_registry.has_service("test_global_service")
            assert service_registry.get_service("test_global_service") == mock_service
            
        finally:
            # Restore original state
            service_registry.clear_services()
            service_registry._services.update(original_services)
            service_registry._service_factories.update(original_factories)


class TestServiceRegistryEdgeCases:
    """Test edge cases for ServiceRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh ServiceRegistry instance for testing."""
        return ServiceRegistry()

    def test_register_service_with_none(self, registry):
        """Test registering None as a service."""
        # Act
        registry.register_service("none_service", None)

        # Assert
        assert registry.get_service("none_service") is None
        assert registry.has_service("none_service") is True

    def test_register_factory_returning_none(self, registry):
        """Test registering a factory that returns None."""
        # Arrange
        none_factory = lambda: None

        # Act
        registry.register_factory("none_service", none_factory)

        # Assert
        assert registry.get_service("none_service") is None
        assert registry.has_service("none_service") is True

    def test_empty_service_name(self, registry):
        """Test behavior with empty service name."""
        # Arrange
        mock_service = Mock()

        # Act
        registry.register_service("", mock_service)

        # Assert
        assert registry.get_service("") == mock_service
        assert registry.has_service("") is True

    def test_none_service_name(self, registry):
        """Test behavior with None as service name."""
        # Arrange
        mock_service = Mock()

        # Act
        registry.register_service(None, mock_service)

        # Assert
        assert registry.get_service(None) == mock_service
        assert registry.has_service(None) is True

    def test_service_name_collision_instance_then_factory(self, registry):
        """Test name collision when instance is registered first, then factory."""
        # Arrange
        instance_service = Mock()
        factory_service = Mock()
        factory = lambda: factory_service

        # Act
        registry.register_service("collision_service", instance_service)
        registry.register_factory("collision_service", factory)

        # Assert
        # Instance should take precedence
        result = registry.get_service("collision_service")
        assert result == instance_service
        assert result != factory_service

    def test_service_name_collision_factory_then_instance(self, registry):
        """Test name collision when factory is registered first, then instance."""
        # Arrange
        instance_service = Mock()
        factory_service = Mock()
        factory = lambda: factory_service

        # Act
        registry.register_factory("collision_service", factory)
        registry.register_service("collision_service", instance_service)

        # Assert
        # Instance should take precedence
        result = registry.get_service("collision_service")
        assert result == instance_service
        assert result != factory_service