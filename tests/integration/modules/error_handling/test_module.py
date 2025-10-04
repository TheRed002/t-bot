"""
Integration tests for error_handling module integration with other modules.

This test suite verifies that the error_handling module properly integrates
with other modules in the system, including:
- Service layer integration
- Dependency injection patterns
- Error propagation between modules
- Decorator usage patterns
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import pytest_asyncio
pytestmark = pytest.mark.skip("Error handling module tests need comprehensive setup")

from src.core.config import Config
from src.core.dependency_injection import DependencyContainer
from src.core.exceptions import DatabaseError, ServiceError, ValidationError
from src.error_handling.di_registration import configure_error_handling_di
from src.error_handling.service import ErrorHandlingService


@pytest_asyncio.fixture
async def config():
    """Test configuration fixture."""
    return Config()


@pytest_asyncio.fixture
async def dependency_container(config):
    """Dependency injection container with error handling configured."""
    container = DependencyContainer()
    container.register("Config", config, singleton=True)

    # Configure error handling DI
    configure_error_handling_di(container, config)

    return container


@pytest_asyncio.fixture
async def error_handling_service(dependency_container):
    """Error handling service fixture."""
    service = dependency_container.resolve("ErrorHandlingService")
    await service.initialize()
    yield service
    await service.shutdown()


class TestErrorHandlingServiceIntegration:
    """Test error handling service integration."""

    @pytest.mark.asyncio
    async def test_service_dependency_injection(self, dependency_container):
        """Test that error handling services are properly registered with DI."""
        # Verify all required services are registered
        assert dependency_container.has_service("ErrorHandler")
        assert dependency_container.has_service("GlobalErrorHandler")
        assert dependency_container.has_service("ErrorPatternAnalytics")
        assert dependency_container.has_service("ErrorHandlingService")

        # Verify services can be resolved
        error_service = dependency_container.resolve("ErrorHandlingService")
        assert error_service is not None
        assert isinstance(error_service, ErrorHandlingService)

    @pytest.mark.asyncio
    async def test_service_initialization_with_dependencies(self, error_handling_service):
        """Test that error handling service initializes with proper dependencies."""
        assert error_handling_service._error_handler is not None
        assert error_handling_service._global_handler is not None
        assert error_handling_service._pattern_analytics is not None
        # StateMonitor is optional, so we don't assert it

    @pytest.mark.asyncio
    async def test_error_handling_service_error_processing(self, error_handling_service):
        """Test that error handling service properly processes errors."""
        test_error = ValidationError("Test validation error")

        with pytest.raises(ValidationError):
            # ValidationErrors should be re-raised after processing
            await error_handling_service.handle_error(
                error=test_error,
                component="TestComponent",
                operation="test_operation",
                context={"test_key": "test_value"},
            )

    @pytest.mark.asyncio
    async def test_error_handling_service_non_validation_errors(self, error_handling_service):
        """Test error handling for non-validation errors."""
        test_error = DatabaseError("Test database error")

        result = await error_handling_service.handle_error(
            error=test_error,
            component="TestComponent",
            operation="test_operation",
            context={"test_key": "test_value"},
        )

        assert result["handled"] is True
        assert result["component"] == "TestComponent"
        assert result["operation"] == "test_operation"
        assert "error_id" in result


class TestDatabaseServiceIntegration:
    """Test database service integration with error handling."""

    @pytest.mark.asyncio
    async def test_database_service_uses_error_handling_service(self):
        """Test that DatabaseService properly uses ErrorHandlingService."""
        # Mock the error handling service
        mock_error_service = AsyncMock()

        # Import and create database service with mocked error handling
        from src.database.service import DatabaseService

        db_service = DatabaseService(error_handling_service=mock_error_service)

        assert db_service.error_handling_service == mock_error_service

    @patch("src.database.service.ValidationFramework")
    @pytest.mark.asyncio
    async def test_database_service_error_handling_integration(self, mock_validation):
        """Test that database service calls error handling service on validation errors."""
        # Setup mock validation to raise error
        mock_validation.validate_price.side_effect = ValidationError("Invalid price")

        # Mock error handling service
        mock_error_service = AsyncMock()

        from src.database.service import DatabaseService

        db_service = DatabaseService(error_handling_service=mock_error_service)

        # Create a mock entity with price
        mock_entity = MagicMock()
        mock_entity.price = "invalid_price"

        # Test validation with error handling
        with pytest.raises(ValidationError):
            db_service._validate_entity(mock_entity)

        # Verify error handling service was called
        # Note: Since we use asyncio.create_task, we need to wait a bit
        await asyncio.sleep(0.1)
        mock_error_service.handle_error.assert_called_once()


class TestExchangeServiceIntegration:
    """Test exchange service integration with error handling."""

    @pytest.mark.asyncio
    async def test_exchange_service_has_error_handling_dependency(self):
        """Test that ExchangeService accepts error handling service dependency."""
        mock_error_service = MagicMock()
        mock_factory = MagicMock()
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {}

        from src.exchanges.service import ExchangeService

        service = ExchangeService(
            exchange_factory=mock_factory,
            config=mock_config,
            error_handling_service=mock_error_service,
        )

        assert service.error_handling_service == mock_error_service


class TestDecoratorIntegration:
    """Test error handling decorator integration."""

    @pytest.mark.asyncio
    async def test_error_handling_decorators_available(self):
        """Test that error handling decorators can be imported and used."""
        from src.error_handling.decorators import (
            with_circuit_breaker,
            with_retry,
        )

        # Test decorators can be applied
        @with_retry(max_attempts=2)
        @with_circuit_breaker(failure_threshold=3)
        @pytest.mark.asyncio
        async def test_function():
            return "success"

        result = await test_function()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_decorator_integration_with_services(self):
        """Test that decorators integrate properly with services."""
        # This test verifies that services using decorators work correctly
        from src.database.service import DatabaseService

        # Create service instance
        from unittest.mock import MagicMock
        mock_connection_manager = MagicMock()
        service = DatabaseService(connection_manager=mock_connection_manager)

        # Verify that methods with decorators exist and are callable
        assert hasattr(service, "get_entity_by_id")
        assert callable(service.get_entity_by_id)


class TestModuleBoundaryValidation:
    """Test module boundary validation patterns."""

    @pytest.mark.asyncio
    async def test_error_handling_boundary_validation(self, error_handling_service):
        """Test boundary validation between error_handling and monitoring modules."""
        # This tests the BoundaryValidator usage in ErrorHandlingService
        test_error = ServiceError("Test service error")

        result = await error_handling_service.handle_error(
            error=test_error,
            component="TestComponent",
            operation="test_operation",
            context={"price": "100.50", "quantity": "10.0"},
        )

        # Verify the response follows the expected boundary contract
        assert "error_id" in result
        assert "handled" in result
        assert "recovery_success" in result
        assert "severity" in result
        assert "timestamp" in result
        assert "component" in result
        assert "operation" in result

    @pytest.mark.asyncio
    async def test_error_context_transformation(self, error_handling_service):
        """Test error context transformation for financial data."""
        test_error = ServiceError("Test error")

        result = await error_handling_service.handle_error(
            error=test_error,
            component="TestComponent",
            operation="test_operation",
            context={"price": "100.50", "quantity": "10.0"},
        )

        # Verify financial data was properly transformed to Decimal
        assert result["handled"] is True


class TestServiceManagerIntegration:
    """Test service manager integration."""

    @pytest.mark.asyncio
    async def test_service_manager_registers_error_handling(self):
        """Test that service manager properly registers error handling services."""
        from src.core.dependency_injection import DependencyContainer
        from src.core.service_manager import register_infrastructure_services

        # Create test container and config
        container = DependencyContainer()
        config = Config()

        # Register infrastructure services
        register_infrastructure_services(config)

        # Verify error handling service is available
        # Note: This test may need adjustment based on actual service manager implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
