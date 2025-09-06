"""
Comprehensive tests for capital management factory implementations.

Tests all factory classes for proper instance creation, dependency injection,
error handling, and factory pattern compliance.
"""

import logging
import unittest.mock
from typing import Any
from unittest.mock import Mock, patch

import pytest

# Disable logging during tests to improve performance
logging.getLogger().setLevel(logging.CRITICAL)

from src.capital_management.capital_allocator import CapitalAllocator
from src.capital_management.currency_manager import CurrencyManager
from src.capital_management.exchange_distributor import ExchangeDistributor
from src.capital_management.factory import (
    CapitalAllocatorFactory,
    CapitalManagementFactory,
    CapitalServiceFactory,
    CurrencyManagerFactory,
    ExchangeDistributorFactory,
    FundFlowManagerFactory,
)
from src.capital_management.fund_flow_manager import FundFlowManager
from src.capital_management.service import CapitalService
from src.core.exceptions import CreationError


class MockDependencyContainer:
    """Mock dependency injection container for testing."""

    def __init__(self):
        self.services = {}

    def get(self, service_name: str):
        if service_name in self.services:
            return self.services[service_name]
        raise Exception(f"Service {service_name} not found")

    def register_service(self, name: str, service: Any):
        self.services[name] = service

    def register(self, name: str, factory, singleton: bool = False):
        """Register a factory with the container."""
        self.services[name] = factory


class TestCapitalServiceFactory:
    """Test CapitalServiceFactory functionality."""

    def test_init_without_container(self):
        """Test factory initialization without dependency container."""
        # Act
        factory = CapitalServiceFactory()
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Assert
        assert factory._product_type == CapitalService
        assert factory._name == "CapitalServiceFactory"
        assert factory._correlation_id is not None  # Auto-generated UUID
        assert factory._dependency_container is None

    def test_init_with_container(self):
        """Test factory initialization with dependency container."""
        # Arrange
        container = MockDependencyContainer()
        correlation_id = "test-correlation-123"

        # Act
        factory = CapitalServiceFactory(
            dependency_container=container, correlation_id=correlation_id
        )
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Assert
        assert factory._dependency_container == container
        assert factory._correlation_id == correlation_id

    @patch("src.capital_management.service.CapitalService")
    def test_create_capital_service_without_dependencies(self, mock_capital_service):
        """Test creating CapitalService without any dependencies."""
        # Arrange
        mock_capital_service.__name__ = "CapitalService"
        mock_instance = Mock()
        mock_capital_service.return_value = mock_instance
        mock_capital_service.__name__ = "CapitalService"  # Set __name__ for factory
        factory = CapitalServiceFactory(correlation_id="test-123")
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Act
        result = factory.create("default")

        # Assert
        mock_capital_service.assert_called_once_with(
            capital_repository=None,
            audit_repository=None,
            state_service=None,
            correlation_id="test-123",
        )
        assert result == mock_instance

    @patch("src.capital_management.service.CapitalService")
    def test_create_capital_service_with_dependencies(self, mock_capital_service):
        """Test creating CapitalService with injected dependencies."""
        # Arrange
        mock_capital_service.__name__ = "CapitalService"
        mock_instance = Mock()
        mock_capital_service.return_value = mock_instance
        mock_capital_service.__name__ = "CapitalService"  # Set __name__ for factory

        mock_capital_repo = Mock()
        mock_audit_repo = Mock()
        mock_state_service = Mock()

        container = MockDependencyContainer()
        container.register_service("CapitalRepository", mock_capital_repo)
        container.register_service("AuditRepository", mock_audit_repo)
        container.register_service("StateService", mock_state_service)

        factory = CapitalServiceFactory(dependency_container=container, correlation_id="test-123")
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Act
        result = factory.create("default")

        # Assert
        mock_capital_service.assert_called_once_with(
            capital_repository=mock_capital_repo,
            audit_repository=mock_audit_repo,
            state_service=mock_state_service,
            correlation_id="test-123",
        )
        assert result == mock_instance

    @patch("src.capital_management.service.CapitalService")
    def test_create_capital_service_with_explicit_params(self, mock_capital_service):
        """Test creating CapitalService with explicit parameters."""
        # Arrange
        mock_capital_service.__name__ = "CapitalService"
        mock_instance = Mock()
        mock_capital_service.return_value = mock_instance
        mock_capital_service.__name__ = "CapitalService"  # Set __name__ for factory

        mock_capital_repo = Mock()
        mock_audit_repo = Mock()

        factory = CapitalServiceFactory()
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Act
        result = factory.create(
            "default", capital_repository=mock_capital_repo, audit_repository=mock_audit_repo
        )

        # Assert
        mock_capital_service.assert_called_once()
        # Verify call arguments (correlation_id will be auto-generated)
        call_args = mock_capital_service.call_args
        assert call_args.kwargs["capital_repository"] == mock_capital_repo
        assert call_args.kwargs["audit_repository"] == mock_audit_repo
        assert call_args.kwargs["state_service"] is None
        assert call_args.kwargs["correlation_id"] is not None  # Auto-generated


class TestCapitalAllocatorFactory:
    """Test CapitalAllocatorFactory functionality."""

    def test_init_without_container(self):
        """Test factory initialization without dependency container."""
        # Act
        factory = CapitalAllocatorFactory()
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Assert
        assert factory._product_type == CapitalAllocator
        assert factory._name == "CapitalAllocatorFactory"
        assert factory._dependency_container is None

    @patch("src.capital_management.capital_allocator.CapitalAllocator")
    def test_create_allocator_requires_capital_service(self, mock_allocator):
        """Test that CapitalAllocator creation requires CapitalService."""
        # Arrange
        mock_allocator.__name__ = "CapitalAllocator"
        factory = CapitalAllocatorFactory()
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Act & Assert
        with pytest.raises(CreationError, match="CapitalService is required"):
            factory.create("default")

    @patch("src.capital_management.capital_allocator.CapitalAllocator")
    def test_create_allocator_with_capital_service_from_container(self, mock_allocator):
        """Test creating CapitalAllocator with CapitalService from container."""
        # Arrange
        mock_allocator.__name__ = "CapitalAllocator"
        mock_instance = Mock()
        mock_allocator.return_value = mock_instance
        mock_allocator.__name__ = "CapitalAllocator"  # Set __name__ for factory

        mock_capital_service = Mock()
        container = MockDependencyContainer()
        container.register_service("CapitalService", mock_capital_service)

        factory = CapitalAllocatorFactory(dependency_container=container)
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Act
        result = factory.create("default")

        # Assert
        mock_allocator.assert_called_once_with(
            capital_service=mock_capital_service,
            config_service=None,
            risk_manager=None,
            trade_lifecycle_manager=None,
            validation_service=None,
        )
        assert result == mock_instance

    @patch("src.capital_management.capital_allocator.CapitalAllocator")
    def test_create_allocator_container_dependency_resolution(self, mock_allocator):
        """Test full dependency resolution from container."""
        # Arrange
        mock_allocator.__name__ = "CapitalAllocator"
        mock_instance = Mock()
        mock_allocator.return_value = mock_instance
        mock_allocator.__name__ = "CapitalAllocator"  # Set __name__ for factory

        mock_capital_service = Mock()
        mock_config_service = Mock()
        mock_risk_service = Mock()
        mock_trade_lifecycle = Mock()
        mock_validation = Mock()

        container = MockDependencyContainer()
        container.register_service("CapitalService", mock_capital_service)
        container.register_service("ConfigService", mock_config_service)
        container.register_service("RiskService", mock_risk_service)
        container.register_service("TradeLifecycleManager", mock_trade_lifecycle)
        container.register_service("ValidationService", mock_validation)

        factory = CapitalAllocatorFactory(dependency_container=container)
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Act
        result = factory.create("default")

        # Assert
        mock_allocator.assert_called_once_with(
            capital_service=mock_capital_service,
            config_service=mock_config_service,
            risk_manager=mock_risk_service,
            trade_lifecycle_manager=mock_trade_lifecycle,
            validation_service=mock_validation,
        )

    def test_create_allocator_missing_capital_service_in_container(self):
        """Test error when CapitalService not available in container."""
        # Arrange
        container = MockDependencyContainer()
        factory = CapitalAllocatorFactory(dependency_container=container)
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Act & Assert
        with pytest.raises(CreationError, match="CapitalService is required but not available"):
            factory.create("default")

    @patch("src.capital_management.capital_allocator.CapitalAllocator")
    def test_create_allocator_risk_manager_fallback(self, mock_allocator):
        """Test RiskManager fallback when RiskService not available."""
        # Arrange
        mock_allocator.__name__ = "CapitalAllocator"
        mock_instance = Mock()
        mock_allocator.return_value = mock_instance
        mock_allocator.__name__ = "CapitalAllocator"  # Set __name__ for factory

        mock_capital_service = Mock()
        mock_risk_manager = Mock()

        container = MockDependencyContainer()
        container.register_service("CapitalService", mock_capital_service)
        container.register_service("RiskManager", mock_risk_manager)

        factory = CapitalAllocatorFactory(dependency_container=container)
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Act
        result = factory.create("default")

        # Assert
        mock_allocator.assert_called_once_with(
            capital_service=mock_capital_service,
            config_service=None,
            risk_manager=mock_risk_manager,
            trade_lifecycle_manager=None,
            validation_service=None,
        )


class TestCurrencyManagerFactory:
    """Test CurrencyManagerFactory functionality."""

    def test_init_basic(self):
        """Test basic factory initialization."""
        # Act
        factory = CurrencyManagerFactory()
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Assert
        assert factory._product_type == CurrencyManager
        assert factory._name == "CurrencyManagerFactory"

    @patch("src.capital_management.currency_manager.CurrencyManager")
    def test_create_currency_manager_without_dependencies(self, mock_currency_manager):
        """Test creating CurrencyManager without dependencies."""
        # Arrange
        mock_currency_manager.__name__ = "CurrencyManager"
        mock_instance = Mock()
        mock_currency_manager.return_value = mock_instance
        mock_currency_manager.__name__ = "CurrencyManager"  # Set __name__ for factory
        factory = CurrencyManagerFactory(correlation_id="test-456")
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Act
        result = factory.create("default")

        # Assert
        mock_currency_manager.assert_called_once_with(
            exchange_data_service=None, validation_service=None, correlation_id="test-456"
        )
        assert result == mock_instance

    @patch("src.capital_management.currency_manager.CurrencyManager")
    def test_create_currency_manager_with_dependencies(self, mock_currency_manager):
        """Test creating CurrencyManager with dependencies from container."""
        # Arrange
        mock_currency_manager.__name__ = "CurrencyManager"
        mock_instance = Mock()
        mock_currency_manager.return_value = mock_instance
        mock_currency_manager.__name__ = "CurrencyManager"  # Set __name__ for factory

        mock_exchange_service = Mock()
        mock_validation_service = Mock()

        container = MockDependencyContainer()
        container.register_service("ExchangeDataService", mock_exchange_service)
        container.register_service("ValidationService", mock_validation_service)

        factory = CurrencyManagerFactory(dependency_container=container, correlation_id="test-789")
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Act
        result = factory.create("default")

        # Assert
        mock_currency_manager.assert_called_once_with(
            exchange_data_service=mock_exchange_service,
            validation_service=mock_validation_service,
            correlation_id="test-789",
        )


class TestExchangeDistributorFactory:
    """Test ExchangeDistributorFactory functionality."""

    def test_init_basic(self):
        """Test basic factory initialization."""
        # Act
        factory = ExchangeDistributorFactory()
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Assert
        assert factory._product_type == ExchangeDistributor
        assert factory._name == "ExchangeDistributorFactory"

    @patch("src.capital_management.exchange_distributor.ExchangeDistributor")
    def test_create_distributor_without_dependencies(self, mock_distributor):
        """Test creating ExchangeDistributor without dependencies."""
        # Arrange
        mock_distributor.__name__ = "ExchangeDistributor"
        mock_instance = Mock()
        mock_distributor.return_value = mock_instance
        mock_distributor.__name__ = "ExchangeDistributor"  # Set __name__ for factory
        factory = ExchangeDistributorFactory()
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Act
        result = factory.create("default")

        # Assert
        mock_distributor.assert_called_once_with(
            exchanges=None, validation_service=None, correlation_id=unittest.mock.ANY
        )

    @patch("src.capital_management.exchange_distributor.ExchangeDistributor")
    def test_create_distributor_with_exchange_registry(self, mock_distributor):
        """Test creating ExchangeDistributor with ExchangeRegistry from container."""
        # Arrange
        mock_distributor.__name__ = "ExchangeDistributor"
        mock_instance = Mock()
        mock_distributor.return_value = mock_instance
        mock_distributor.__name__ = "ExchangeDistributor"  # Set __name__ for factory

        mock_exchange_registry = {"binance": Mock(), "coinbase": Mock()}
        mock_validation_service = Mock()

        container = MockDependencyContainer()
        container.register_service("ExchangeRegistry", mock_exchange_registry)
        container.register_service("ValidationService", mock_validation_service)

        factory = ExchangeDistributorFactory(dependency_container=container)
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Act
        result = factory.create("default")

        # Assert
        mock_distributor.assert_called_once_with(
            exchanges=mock_exchange_registry,
            validation_service=mock_validation_service,
            correlation_id=unittest.mock.ANY,
        )

    @patch("src.capital_management.exchange_distributor.ExchangeDistributor")
    def test_create_distributor_empty_exchanges_fallback(self, mock_distributor):
        """Test ExchangeDistributor creation with empty exchanges fallback."""
        # Arrange
        mock_distributor.__name__ = "ExchangeDistributor"
        mock_instance = Mock()
        mock_distributor.return_value = mock_instance
        mock_distributor.__name__ = "ExchangeDistributor"  # Set __name__ for factory

        container = MockDependencyContainer()  # No ExchangeRegistry
        factory = ExchangeDistributorFactory(dependency_container=container)
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Act
        result = factory.create("default")

        # Assert
        mock_distributor.assert_called_once_with(
            exchanges={},  # Empty dict fallback
            validation_service=None,
            correlation_id=unittest.mock.ANY,
        )


class TestFundFlowManagerFactory:
    """Test FundFlowManagerFactory functionality."""

    def test_init_basic(self):
        """Test basic factory initialization."""
        # Act
        factory = FundFlowManagerFactory()
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Assert
        assert factory._product_type == FundFlowManager
        assert factory._name == "FundFlowManagerFactory"

    @patch("src.capital_management.fund_flow_manager.FundFlowManager")
    def test_create_fund_flow_manager_basic(self, mock_fund_flow):
        """Test creating FundFlowManager without dependencies."""
        # Arrange
        mock_fund_flow.__name__ = "FundFlowManager"
        mock_instance = Mock()
        mock_fund_flow.return_value = mock_instance
        mock_fund_flow.__name__ = "FundFlowManager"  # Set __name__ for factory
        factory = FundFlowManagerFactory()
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Act
        result = factory.create("default")

        # Assert
        mock_fund_flow.assert_called_once_with(
            cache_service=None,
            time_series_service=None,
            validation_service=None,
            correlation_id=unittest.mock.ANY,
        )

    @patch("src.capital_management.fund_flow_manager.FundFlowManager")
    def test_create_fund_flow_manager_with_dependencies(self, mock_fund_flow):
        """Test creating FundFlowManager with all dependencies."""
        # Arrange
        mock_fund_flow.__name__ = "FundFlowManager"
        mock_instance = Mock()
        mock_fund_flow.return_value = mock_instance
        mock_fund_flow.__name__ = "FundFlowManager"  # Set __name__ for factory

        mock_cache_service = Mock()
        mock_time_series = Mock()
        mock_validation = Mock()

        container = MockDependencyContainer()
        container.register_service("CacheService", mock_cache_service)
        container.register_service("TimeSeriesService", mock_time_series)
        container.register_service("ValidationService", mock_validation)

        factory = FundFlowManagerFactory(dependency_container=container)
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Act
        result = factory.create("default")

        # Assert
        mock_fund_flow.assert_called_once_with(
            cache_service=mock_cache_service,
            time_series_service=mock_time_series,
            validation_service=mock_validation,
            correlation_id=unittest.mock.ANY,
        )


class TestCapitalManagementFactory:
    """Test main CapitalManagementFactory functionality."""

    def test_init_without_dependencies(self):
        """Test initialization of main factory without dependencies."""
        # Act
        factory = CapitalManagementFactory()

        # Assert
        assert factory.dependency_container is None
        assert factory.correlation_id is None
        assert factory.capital_service_factory is not None
        assert factory.capital_allocator_factory is not None
        assert factory.currency_manager_factory is not None
        assert factory.exchange_distributor_factory is not None
        assert factory.fund_flow_manager_factory is not None

    def test_init_with_dependencies(self):
        """Test initialization with dependency container."""
        # Arrange
        container = MockDependencyContainer()
        correlation_id = "main-test-123"

        # Act
        factory = CapitalManagementFactory(
            dependency_container=container, correlation_id=correlation_id
        )

        # Assert
        assert factory.dependency_container == container
        assert factory.correlation_id == correlation_id

        # Verify sub-factories have correct dependencies
        assert factory.capital_service_factory._dependency_container == container
        assert factory.capital_service_factory._correlation_id == correlation_id
        assert factory.capital_allocator_factory._dependency_container == container
        assert factory.currency_manager_factory._dependency_container == container
        assert factory.exchange_distributor_factory._dependency_container == container
        assert factory.fund_flow_manager_factory._dependency_container == container

    def test_create_capital_service(self):
        """Test creating CapitalService through main factory."""
        # Arrange
        factory = CapitalManagementFactory()
        mock_service = Mock()
        factory.capital_service_factory.create = Mock(return_value=mock_service)

        # Act
        result = factory.create_capital_service(param1="value1")

        # Assert
        factory.capital_service_factory.create.assert_called_once_with("default", param1="value1")
        assert result == mock_service

    def test_create_capital_allocator(self):
        """Test creating CapitalAllocator through main factory."""
        # Arrange
        factory = CapitalManagementFactory()
        mock_allocator = Mock()
        factory.capital_allocator_factory.create = Mock(return_value=mock_allocator)

        # Act
        result = factory.create_capital_allocator(param2="value2")

        # Assert
        factory.capital_allocator_factory.create.assert_called_once_with("default", param2="value2")
        assert result == mock_allocator

    def test_create_currency_manager(self):
        """Test creating CurrencyManager through main factory."""
        # Arrange
        factory = CapitalManagementFactory()
        mock_manager = Mock()
        factory.currency_manager_factory.create = Mock(return_value=mock_manager)

        # Act
        result = factory.create_currency_manager()

        # Assert
        factory.currency_manager_factory.create.assert_called_once_with("default")
        assert result == mock_manager

    def test_create_exchange_distributor(self):
        """Test creating ExchangeDistributor through main factory."""
        # Arrange
        factory = CapitalManagementFactory()
        mock_distributor = Mock()
        factory.exchange_distributor_factory.create = Mock(return_value=mock_distributor)

        # Act
        result = factory.create_exchange_distributor()

        # Assert
        factory.exchange_distributor_factory.create.assert_called_once_with("default")
        assert result == mock_distributor

    def test_create_fund_flow_manager(self):
        """Test creating FundFlowManager through main factory."""
        # Arrange
        factory = CapitalManagementFactory()
        mock_fund_flow = Mock()
        factory.fund_flow_manager_factory.create = Mock(return_value=mock_fund_flow)

        # Act
        result = factory.create_fund_flow_manager()

        # Assert
        factory.fund_flow_manager_factory.create.assert_called_once_with("default")
        assert result == mock_fund_flow

    def test_register_factories(self):
        """Test registering factories with container."""
        # Arrange
        factory = CapitalManagementFactory()
        container = MockDependencyContainer()

        # Act
        factory.register_factories(container)

        # Assert - Verify all factories are registered
        factory_names = [
            "CapitalServiceFactory",
            "CapitalAllocatorFactory",
            "CurrencyManagerFactory",
            "ExchangeDistributorFactory",
            "FundFlowManagerFactory",
            "CapitalManagementFactory",
        ]

        for factory_name in factory_names:
            assert factory_name in container.services

        # Verify factories return correct instances
        assert container.get("CapitalServiceFactory")() == factory.capital_service_factory
        assert container.get("CapitalAllocatorFactory")() == factory.capital_allocator_factory
        assert container.get("CurrencyManagerFactory")() == factory.currency_manager_factory
        assert container.get("ExchangeDistributorFactory")() == factory.exchange_distributor_factory
        assert container.get("FundFlowManagerFactory")() == factory.fund_flow_manager_factory
        assert container.get("CapitalManagementFactory")() == factory


class TestFactoryErrorHandling:
    """Test error handling across all factories."""

    def test_capital_service_factory_create_error(self):
        """Test error handling in CapitalServiceFactory creation."""
        # Arrange
        container = MockDependencyContainer()
        factory = CapitalServiceFactory(dependency_container=container)
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        with patch(
            "src.capital_management.service.CapitalService",
            side_effect=Exception("Creation failed"),
        ):
            # Act & Assert
            with pytest.raises(Exception, match="Creation failed"):
                factory.create("default")

    def test_container_service_retrieval_error_handling(self):
        """Test handling of service retrieval errors from container."""
        # Arrange
        container = MockDependencyContainer()
        factory = CapitalServiceFactory(dependency_container=container)
        factory.configure_validation(
            validate_products=False
        )  # Disable validation for mocked classes

        # Act - Should not raise error, should use None for optional dependencies
        with patch("src.capital_management.service.CapitalService") as mock_service:
            mock_instance = Mock()
            mock_service.return_value = mock_instance

            result = factory.create("default")

            # Assert - Should have called with None values for missing dependencies
            mock_service.assert_called_once_with(
                capital_repository=None,
                audit_repository=None,
                state_service=None,
                correlation_id=unittest.mock.ANY,
            )

    def test_factory_inheritance_structure(self):
        """Test that factories inherit from BaseFactory correctly."""
        # Arrange & Act
        capital_service_factory = CapitalServiceFactory()
        capital_service_factory.configure_validation(validate_products=False)
        capital_allocator_factory = CapitalAllocatorFactory()
        capital_allocator_factory.configure_validation(validate_products=False)
        currency_manager_factory = CurrencyManagerFactory()
        currency_manager_factory.configure_validation(validate_products=False)
        exchange_distributor_factory = ExchangeDistributorFactory()
        exchange_distributor_factory.configure_validation(validate_products=False)
        fund_flow_manager_factory = FundFlowManagerFactory()
        fund_flow_manager_factory.configure_validation(validate_products=False)

        # Assert - All factories should have BaseFactory methods
        factories = [
            capital_service_factory,
            capital_allocator_factory,
            currency_manager_factory,
            exchange_distributor_factory,
            fund_flow_manager_factory,
        ]

        for factory in factories:
            assert hasattr(factory, "_product_type")
            assert hasattr(factory, "_name")
            assert hasattr(factory, "create")
            assert hasattr(factory, "register")

    def test_factory_correlation_id_propagation(self):
        """Test that correlation IDs are properly propagated."""
        # Arrange
        correlation_id = "propagation-test-456"

        # Act
        main_factory = CapitalManagementFactory(correlation_id=correlation_id)

        # Assert - All sub-factories should have the same correlation ID
        assert main_factory.capital_service_factory._correlation_id == correlation_id
        assert main_factory.capital_allocator_factory._correlation_id == correlation_id
        assert main_factory.currency_manager_factory._correlation_id == correlation_id
        assert main_factory.exchange_distributor_factory._correlation_id == correlation_id
        assert main_factory.fund_flow_manager_factory._correlation_id == correlation_id
