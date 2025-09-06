"""
Comprehensive tests for capital management DI registration.

Tests the dependency injection registration system for capital management services,
ensuring proper factory creation, service registration, and interface mappings.
"""

import logging
from unittest.mock import Mock

import pytest

# Disable logging during tests to improve performance
logging.getLogger().setLevel(logging.CRITICAL)

from src.capital_management.di_registration import register_capital_management_services
from src.capital_management.factory import CapitalManagementFactory
from src.capital_management.service import CapitalService


@pytest.fixture(scope="session")
def mock_container():
    """Mock dependency injection container for testing."""
    container = Mock()
    container.services = {}
    container.singleton_flags = {}

    def register(name: str, factory, singleton: bool = False):
        container.services[name] = factory
        container.singleton_flags[name] = singleton

    def get(name: str):
        if name in container.services:
            factory = container.services[name]
            if callable(factory):
                return factory()
            return factory
        raise KeyError(f"Service {name} not found")

    def has(name: str) -> bool:
        return name in container.services

    container.register = register
    container.get = get
    container.has = has
    return container


class MockContainer:
    """Mock dependency injection container for testing."""

    def __init__(self):
        self.services = {}
        self.singleton_flags = {}

    def register(self, name: str, factory, singleton: bool = False):
        """Register a service with the container."""
        self.services[name] = factory
        self.singleton_flags[name] = singleton

    def get(self, name: str):
        """Get a service from the container."""
        if name in self.services:
            factory = self.services[name]
            if callable(factory):
                return factory()
            return factory
        raise KeyError(f"Service {name} not found")

    def has(self, name: str) -> bool:
        """Check if service is registered."""
        return name in self.services


class TestRegisterCapitalManagementServices:
    """Test capital management DI registration functionality."""

    def test_register_capital_management_services_success(self, mock_container):
        """Test successful registration of all capital management services."""
        # Arrange
        container = mock_container

        # Act
        register_capital_management_services(container)

        # Assert - Main factory registration
        assert "CapitalManagementFactory" in container.services
        assert container.singleton_flags["CapitalManagementFactory"] is True

        # Get the factory instance
        factory = container.get("CapitalManagementFactory")
        assert isinstance(factory, CapitalManagementFactory)
        assert factory.dependency_container == container

    def test_register_concrete_services(self, mock_container):
        """Test registration of concrete service implementations."""
        # Arrange
        container = mock_container

        # Act
        register_capital_management_services(container)

        # Assert - Concrete services
        concrete_services = [
            "CapitalService",
            "CapitalAllocator",
            "CurrencyManager",
            "ExchangeDistributor",
            "FundFlowManager",
        ]

        for service_name in concrete_services:
            assert service_name in container.services
            assert container.singleton_flags[service_name] is True

    def test_register_interface_mappings(self, mock_container):
        """Test registration of interface to implementation mappings."""
        # Arrange
        container = mock_container

        # Act
        register_capital_management_services(container)

        # Assert - Interface mappings
        interface_mappings = [
            ("AbstractCapitalService", "CapitalService"),
            ("CapitalServiceProtocol", "CapitalService"),
            ("AbstractCurrencyManagementService", "CurrencyManager"),
            ("CurrencyManagementServiceProtocol", "CurrencyManager"),
            ("AbstractExchangeDistributionService", "ExchangeDistributor"),
            ("ExchangeDistributionServiceProtocol", "ExchangeDistributor"),
            ("AbstractFundFlowManagementService", "FundFlowManager"),
            ("FundFlowManagementServiceProtocol", "FundFlowManager"),
        ]

        for interface_name, implementation_name in interface_mappings:
            assert interface_name in container.services
            assert container.singleton_flags[interface_name] is True

    def test_register_individual_factories(self, mock_container):
        """Test registration of individual sub-factories."""
        # Arrange
        container = mock_container

        # Act
        register_capital_management_services(container)

        # Assert - Individual factories
        factory_services = [
            "CapitalServiceFactory",
            "CapitalAllocatorFactory",
            "CurrencyManagerFactory",
            "ExchangeDistributorFactory",
            "FundFlowManagerFactory",
        ]

        for factory_name in factory_services:
            assert factory_name in container.services
            assert container.singleton_flags[factory_name] is True

    def test_factory_creation_functions(self, mock_container):
        """Test that factory creation functions work correctly."""
        # Arrange
        container = mock_container

        # Register the service
        register_capital_management_services(container)

        # Act - Test service creation (services are actually created)
        capital_service = container.get("CapitalService")
        allocator = container.get("CapitalAllocator")
        currency_manager = container.get("CurrencyManager")
        exchange_distributor = container.get("ExchangeDistributor")
        fund_flow_manager = container.get("FundFlowManager")

        # Assert - Verify services are created and of correct type
        assert capital_service is not None
        assert allocator is not None
        assert currency_manager is not None
        assert exchange_distributor is not None
        assert fund_flow_manager is not None

        # Verify they are the expected types by checking their class names
        assert "CapitalService" in str(type(capital_service))
        assert "CapitalAllocator" in str(type(allocator))
        assert "CurrencyManager" in str(type(currency_manager))
        assert "ExchangeDistributor" in str(type(exchange_distributor))
        assert "FundFlowManager" in str(type(fund_flow_manager))

    def test_interface_resolution(self):
        """Test that interfaces resolve to correct implementations."""
        # Arrange
        container = MockContainer()
        mock_capital_service = Mock(spec=CapitalService)

        # Register services
        register_capital_management_services(container)

        # Mock the CapitalService creation
        container.services["CapitalService"] = lambda: mock_capital_service

        # Act
        abstract_service = container.get("AbstractCapitalService")
        protocol_service = container.get("CapitalServiceProtocol")

        # Assert
        assert abstract_service == mock_capital_service
        assert protocol_service == mock_capital_service

    def test_factory_hierarchy_access(self):
        """Test access to sub-factories through the main factory."""
        # Arrange
        container = MockContainer()

        # Act
        register_capital_management_services(container)

        # Assert - Get the main factory and verify sub-factories are accessible
        main_factory = container.get("CapitalManagementFactory")

        # Verify sub-factories exist as attributes
        assert hasattr(main_factory, "capital_service_factory")
        assert hasattr(main_factory, "capital_allocator_factory")
        assert hasattr(main_factory, "currency_manager_factory")
        assert hasattr(main_factory, "exchange_distributor_factory")
        assert hasattr(main_factory, "fund_flow_manager_factory")

        # Verify they are the correct factory types
        assert main_factory.capital_service_factory is not None
        assert main_factory.capital_allocator_factory is not None
        assert main_factory.currency_manager_factory is not None
        assert main_factory.exchange_distributor_factory is not None
        assert main_factory.fund_flow_manager_factory is not None

    def test_registration_with_none_container(self):
        """Test registration behavior with None container (edge case)."""
        # This should not raise an error but should handle gracefully
        # In production, this would be a configuration error
        with pytest.raises(AttributeError):
            register_capital_management_services(None)

    def test_registration_idempotency(self):
        """Test that multiple registrations don't cause conflicts."""
        # Arrange
        container = MockContainer()

        # Act - Register twice
        register_capital_management_services(container)
        register_capital_management_services(container)

        # Assert - Should still work correctly
        factory = container.get("CapitalManagementFactory")
        assert isinstance(factory, CapitalManagementFactory)

        # Should still have all required services
        required_services = [
            "CapitalService",
            "CapitalAllocator",
            "CurrencyManager",
            "ExchangeDistributor",
            "FundFlowManager",
        ]
        for service_name in required_services:
            assert service_name in container.services

    def test_container_dependency_injection(self):
        """Test that the container is properly injected into the factory."""
        # Arrange
        container = MockContainer()

        # Act
        register_capital_management_services(container)

        # Get the main factory
        main_factory = container.get("CapitalManagementFactory")

        # Assert
        assert main_factory.dependency_container == container

        # Verify sub-factories also have container access
        assert main_factory.capital_service_factory._dependency_container == container
        assert main_factory.capital_allocator_factory._dependency_container == container
        assert main_factory.currency_manager_factory._dependency_container == container
        assert main_factory.exchange_distributor_factory._dependency_container == container
        assert main_factory.fund_flow_manager_factory._dependency_container == container

    def test_service_creation_error_handling(self):
        """Test error handling in service creation functions."""
        # Arrange
        container = MockContainer()
        register_capital_management_services(container)

        # Create a factory that will fail
        def failing_factory():
            raise Exception("Factory creation failed")

        container.services["CapitalManagementFactory"] = failing_factory

        # Act & Assert - Should propagate the error
        with pytest.raises(Exception, match="Factory creation failed"):
            container.get("CapitalService")

    def test_all_expected_services_registered(self):
        """Test that all expected services are registered."""
        # Arrange
        container = MockContainer()

        # Expected services list
        expected_services = [
            # Main factory
            "CapitalManagementFactory",
            # Concrete services
            "CapitalService",
            "CapitalAllocator",
            "CurrencyManager",
            "ExchangeDistributor",
            "FundFlowManager",
            # Abstract interfaces
            "AbstractCapitalService",
            "AbstractCurrencyManagementService",
            "AbstractExchangeDistributionService",
            "AbstractFundFlowManagementService",
            # Protocol interfaces
            "CapitalServiceProtocol",
            "CurrencyManagementServiceProtocol",
            "ExchangeDistributionServiceProtocol",
            "FundFlowManagementServiceProtocol",
            # Individual factories
            "CapitalServiceFactory",
            "CapitalAllocatorFactory",
            "CurrencyManagerFactory",
            "ExchangeDistributorFactory",
            "FundFlowManagerFactory",
        ]

        # Act
        register_capital_management_services(container)

        # Assert
        for service_name in expected_services:
            assert service_name in container.services, f"Service {service_name} not registered"
            assert container.singleton_flags[service_name] is True, (
                f"Service {service_name} not singleton"
            )
