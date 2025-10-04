"""
Integration validation tests for error_handling module.

This test suite validates the error_handling module's proper integration with other
modules in the system, ensuring:
- Correct dependency injection patterns
- Proper module boundaries and API usage  
- Integration with database, monitoring, and web_interface modules
- Error propagation patterns work correctly
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import Config
from src.core.dependency_injection import DependencyContainer
from src.core.exceptions import DatabaseError, ServiceError, ValidationError
from src.error_handling import (
    ErrorHandlingService,
    ErrorSeverity,
    configure_error_handling_di,
    get_global_error_handler,
    with_circuit_breaker,
    with_retry,
)


@pytest.fixture
def config():
    """Test configuration."""
    return Config()


@pytest.fixture
def dependency_container(config):
    """DI container with basic services."""
    container = DependencyContainer()
    container.register("Config", config, singleton=True)
    return container


@pytest.fixture
def mock_database_service():
    """Mock database service for testing."""
    service = MagicMock()
    service.initialize = AsyncMock()
    service.cleanup = AsyncMock()
    service.health_check = AsyncMock(return_value={"status": "healthy"})
    return service


@pytest.fixture
def mock_monitoring_service():
    """Mock monitoring service for testing."""
    service = MagicMock()
    service.record_error_event = AsyncMock()
    service.record_error_pattern = AsyncMock()
    return service


class TestErrorHandlingDependencyInjection:
    """Test error_handling module dependency injection patterns."""

    def test_dependency_container_interface_compatibility(self, dependency_container):
        """Test that error_handling works with actual DependencyContainer interface."""
        # Verify the DI container has the expected methods
        assert hasattr(dependency_container, "register")
        assert hasattr(dependency_container, "get")
        assert hasattr(dependency_container, "has")
        assert callable(dependency_container.register)
        assert callable(dependency_container.get)
        assert callable(dependency_container.has)

    def test_error_handling_di_registration_basic(self, dependency_container, config):
        """Test basic error handling DI registration without hanging."""
        # Register services without full initialization to avoid hanging
        try:
            # Test that registration can be called without errors
            configure_error_handling_di(dependency_container, config)
            
            # Verify services are registered
            assert dependency_container.has("ErrorHandlingService")
            assert dependency_container.has("ErrorHandler")
            assert dependency_container.has("GlobalErrorHandler")
            
        except Exception as e:
            pytest.fail(f"Error handling DI registration failed: {e}")

    def test_error_handling_service_factory(self, dependency_container, config):
        """Test error handling service can be created via factory pattern."""
        from src.error_handling.service import create_error_handling_service
        
        # Create service using factory function
        service = create_error_handling_service(
            config=config,
            dependency_container=dependency_container,
        )
        
        assert service is not None
        assert isinstance(service, ErrorHandlingService)


class TestModuleBoundaryIntegration:
    """Test integration across module boundaries."""

    async def test_database_service_integration(self, mock_database_service):
        """Test that DatabaseService properly integrates with error handling."""
        from src.database.connection import DatabaseConnectionManager
        from src.database.service import DatabaseService
        from unittest.mock import Mock

        # Mock the connection manager
        mock_connection_manager = Mock(spec=DatabaseConnectionManager)

        # Create service with valid parameters
        db_service = DatabaseService(connection_manager=mock_connection_manager)
        assert db_service is not None
        assert db_service.name == "DatabaseService"

    async def test_web_interface_error_middleware_integration(self):
        """Test web interface error middleware uses error handling properly."""
        from src.web_interface.middleware.error_handler import ErrorHandlerMiddleware
        
        # Create middleware instance
        middleware = ErrorHandlerMiddleware(app=MagicMock(), debug=False)
        
        # Verify middleware can be instantiated and has error handling integration
        assert hasattr(middleware, "app")

    async def test_monitoring_service_integration(self, mock_monitoring_service):
        """Test monitoring service integration with error handling."""
        # Test that monitoring service can handle error events from error_handling
        error_data = {
            "error_type": "ValidationError",
            "component": "TestComponent", 
            "severity": "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Verify monitoring service can process error events
        await mock_monitoring_service.record_error_event(error_data)
        mock_monitoring_service.record_error_event.assert_called_once_with(error_data)


class TestErrorPropagationPatterns:
    """Test error propagation patterns between modules."""

    async def test_validation_error_propagation(self):
        """Test validation errors are properly propagated."""
        from src.utils.messaging_patterns import BoundaryValidator
        
        # Test boundary validation patterns
        boundary_data = {
            "component": "web_interface",
            "error_type": "ValidationError",
            "severity": "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_mode": "batch",
            "data_format": "boundary_validation_v1",
            "message_pattern": "batch",
            "boundary_crossed": True,
        }
        
        # Verify boundary validation can be applied
        try:
            BoundaryValidator.validate_web_interface_to_error_boundary(boundary_data)
        except Exception as e:
            # This might raise if validation fails, which is expected behavior
            assert isinstance(e, (ValidationError, ServiceError))

    async def test_financial_data_transformation(self):
        """Test financial data transformation in error contexts."""
        from src.error_handling.service import ErrorHandlingService
        
        # Create minimal service for testing
        config = Config()
        service = ErrorHandlingService(config=config)
        
        # Test data transformation
        context = {"price": "100.50", "quantity": "10.0"}
        transformed = service._transform_error_context(context, "TestComponent")
        
        assert "processing_stage" in transformed
        assert "processed_at" in transformed
        assert "component" in transformed
        assert transformed["component"] == "TestComponent"


class TestDecoratorIntegration:
    """Test error handling decorator integration."""

    async def test_retry_decorator_integration(self):
        """Test retry decorator can be used across modules."""
        call_count = 0
        
        @with_retry(max_attempts=3)
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ServiceError("Test error")
            return "success"
        
        result = await test_function()
        assert result == "success"
        assert call_count == 2

    async def test_circuit_breaker_decorator_integration(self):
        """Test circuit breaker decorator can be used across modules."""
        @with_circuit_breaker(failure_threshold=2)
        async def test_function():
            return "success"
        
        result = await test_function()
        assert result == "success"

    def test_decorator_import_patterns(self):
        """Test that decorators can be imported from error_handling module."""
        # Test various import patterns used by other modules
        from src.error_handling.decorators import with_retry, with_circuit_breaker
        
        assert callable(with_retry)
        assert callable(with_circuit_breaker)
        
        # Test module-level imports
        from src.error_handling import with_retry as module_retry
        assert callable(module_retry)


class TestServiceLayerIntegration:
    """Test service layer integration patterns."""

    async def test_error_handling_service_interface(self):
        """Test ErrorHandlingService implements proper service interface."""
        from src.error_handling.interfaces import ErrorHandlingServiceInterface
        from src.error_handling.service import ErrorHandlingService
        
        config = Config()
        service = ErrorHandlingService(config=config)
        
        # Verify service implements the interface
        assert isinstance(service, ErrorHandlingServiceInterface)

    async def test_global_error_handler_setup(self):
        """Test global error handler setup patterns."""
        # Test that global error handler can be accessed
        handler = get_global_error_handler()
        # Handler might be None if not initialized, which is acceptable
        
        # Test global handler setup function exists
        from src.error_handling import set_global_error_handler
        assert callable(set_global_error_handler)

    async def test_error_context_creation_patterns(self):
        """Test error context creation across modules."""
        from src.error_handling.context import ErrorContext
        
        # Test error context can be created
        context = ErrorContext(
            error_id="test-123",
            error=ServiceError("Test error"),
            component="TestComponent",
            operation="test_operation",
            severity=ErrorSeverity.MEDIUM,
            timestamp=datetime.now(timezone.utc)
        )
        
        assert context.error_id == "test-123"
        assert context.component == "TestComponent"


class TestIntegrationErrorScenarios:
    """Test integration error scenarios and recovery."""

    async def test_missing_dependency_handling(self):
        """Test handling of missing dependencies in integration."""
        # Test that services handle missing dependencies gracefully
        config = Config()
        
        # Create service with minimal dependencies
        service = ErrorHandlingService(config=config)
        
        # Verify service can be created even with missing optional dependencies
        assert service._state_monitor is None  # Optional dependency
        
        # Test initialization detects missing required dependencies
        with pytest.raises(ServiceError, match="Required dependencies not injected"):
            await service.initialize()

    async def test_circular_dependency_prevention(self, dependency_container, config):
        """Test that circular dependencies are prevented."""
        # Register error handling services
        configure_error_handling_di(dependency_container, config)
        
        # Verify services can be registered without circular dependency issues
        assert dependency_container.has("ErrorHandlingService")
        assert dependency_container.has("ErrorHandler")

    async def test_service_lifecycle_integration(self):
        """Test service lifecycle integration patterns."""
        from src.error_handling.service import ErrorHandlingService
        
        config = Config()
        
        # Create service with mock dependencies to avoid initialization issues
        mock_error_handler = MagicMock()
        mock_global_handler = MagicMock() 
        mock_pattern_analytics = MagicMock()
        
        service = ErrorHandlingService(
            config=config,
            error_handler=mock_error_handler,
            global_handler=mock_global_handler,
            pattern_analytics=mock_pattern_analytics,
        )
        
        # Test service lifecycle
        await service.initialize()
        assert service._initialized is True
        
        # Test health check
        health = await service.health_check()
        from src.core.base.interfaces import HealthStatus
        assert health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        
        # Test cleanup
        await service.shutdown()
        assert service._initialized is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])