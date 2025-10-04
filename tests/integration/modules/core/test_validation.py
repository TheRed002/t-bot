"""
Integration tests for core module validation and cross-module boundaries.

This test suite validates that the core module properly integrates with other modules
and that module boundaries are correctly respected throughout the codebase.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock

from src.core.base.service import BaseService
from src.core.dependency_injection import DependencyInjector, get_global_injector
from src.core.exceptions import ServiceError, DependencyError, ValidationError
from src.core.types.base import ConfigDict
from src.core.base.interfaces import HealthStatus


class MockRepository:
    """Mock repository for testing dependency injection."""
    
    def __init__(self):
        self.data = {}
        
    async def create(self, entity):
        entity.id = "test_id"
        self.data[entity.id] = entity
        return entity
        
    async def get(self, entity_id):
        return self.data.get(entity_id)


class TestCoreService(BaseService):
    """Test service that inherits from BaseService."""
    
    def __init__(self, name="TestService", config=None):
        super().__init__(name, config)
        self.repository = None
        
    def configure_dependencies(self, dependency_injector):
        """Configure dependencies using the DI container."""
        super().configure_dependencies(dependency_injector)
        try:
            self.repository = self.resolve_dependency("TestRepository")
        except DependencyError:
            # Repository is optional for this test
            pass
            
    async def test_operation(self, data):
        """Test operation that uses monitoring."""
        return await self.execute_with_monitoring(
            "test_operation",
            self._test_operation_impl,
            data
        )
        
    async def _test_operation_impl(self, data):
        """Implementation of test operation."""
        if not data.get("valid", True):
            raise ValidationError("Invalid test data")
        return {"result": "success", "data": data}


class TestCoreModuleIntegration:
    """Test suite for core module integration validation."""
    
    @pytest.fixture
    def injector(self):
        """Create a clean dependency injector for each test."""
        injector = DependencyInjector()
        injector.clear()
        yield injector
        injector.clear()
        
    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        return MockRepository()
        
    @pytest.fixture
    def test_service(self):
        """Create test service instance."""
        return TestCoreService()
        
    @pytest.mark.asyncio
    async def test_base_service_dependency_injection(self, injector, mock_repository, test_service):
        """Test that BaseService properly integrates with dependency injection."""
        # Register repository
        injector.register_service("TestRepository", mock_repository, singleton=True)
        
        # Configure service dependencies
        test_service.configure_dependencies(injector)
        
        # Verify dependency was resolved
        assert test_service.repository is mock_repository
        assert test_service._dependency_container is injector
        
    @pytest.mark.asyncio
    async def test_service_lifecycle_management(self, injector, test_service):
        """Test service lifecycle methods work correctly."""
        # Configure dependencies
        test_service.configure_dependencies(injector)
        
        # Test starting service
        await test_service.start()
        assert test_service.is_running
        
        # Test health check
        health = await test_service.health_check()
        assert health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        
        # Test stopping service
        await test_service.stop()
        assert not test_service.is_running
        
    @pytest.mark.asyncio
    async def test_service_error_handling(self, injector, test_service):
        """Test that service error handling works correctly."""
        test_service.configure_dependencies(injector)
        await test_service.start()
        
        # Test validation error propagation
        with pytest.raises(ValidationError):
            await test_service.test_operation({"valid": False})
            
        # Test successful operation
        result = await test_service.test_operation({"valid": True, "test": "data"})
        assert result["result"] == "success"
        assert result["data"]["test"] == "data"
        
        await test_service.stop()
        
    @pytest.mark.asyncio
    async def test_service_metrics_collection(self, injector, test_service):
        """Test that service metrics are properly collected."""
        test_service.configure_dependencies(injector)
        await test_service.start()
        
        # Perform some operations
        await test_service.test_operation({"test": "data1"})
        await test_service.test_operation({"test": "data2"})
        
        # Get metrics
        metrics = test_service.get_metrics()
        
        # Verify metrics structure
        assert "operations_count" in metrics
        assert "operations_success" in metrics
        assert "operations_error" in metrics
        assert "average_response_time" in metrics
        
        # Verify operation counts
        assert metrics["operations_count"] == 2
        assert metrics["operations_success"] == 2
        assert metrics["operations_error"] == 0
        
        await test_service.stop()
        
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, injector, test_service):
        """Test that circuit breaker works correctly."""
        test_service.configure_dependencies(injector)
        test_service.configure_circuit_breaker(enabled=True, threshold=3, timeout=60)
        await test_service.start()
        
        # Trigger multiple failures to trip circuit breaker
        for _ in range(4):
            try:
                await test_service.test_operation({"valid": False})
            except (ValidationError, ServiceError):
                pass
                
        # Circuit breaker should now be open
        with pytest.raises(ServiceError, match="Circuit breaker is OPEN"):
            await test_service.test_operation({"valid": True})
            
        await test_service.stop()
        
    @pytest.mark.asyncio
    async def test_dependency_error_handling(self, injector, test_service):
        """Test that dependency errors are handled properly."""
        # Don't register required dependencies
        test_service.configure_dependencies(injector)
        
        # Test missing dependency resolution
        with pytest.raises(DependencyError):
            test_service.resolve_dependency("NonexistentService")
            
    def test_global_injector_singleton(self):
        """Test that global injector maintains singleton pattern."""
        injector1 = get_global_injector()
        injector2 = get_global_injector()
        
        assert injector1 is injector2
        
    @pytest.mark.asyncio
    async def test_transactional_service_pattern(self, injector):
        """Test transactional service functionality."""
        from src.core.base.service import TransactionalService
        
        class TestTransactionalService(TransactionalService):
            def __init__(self):
                super().__init__(name="TestTransactionalService")
                
            async def test_transaction_operation(self, data):
                return await self.execute_in_transaction(
                    "test_transaction",
                    self._transaction_impl,
                    data
                )
                
            async def _transaction_impl(self, data):
                return {"result": "transaction_success", "data": data}
        
        service = TestTransactionalService()
        service.configure_dependencies(injector)
        
        # Without transaction manager, should fall back to regular execution
        result = await service.test_transaction_operation({"test": "data"})
        assert result["result"] == "transaction_success"
        
    @pytest.mark.asyncio
    async def test_service_registry_integration(self, injector):
        """Test service registry and locator patterns."""
        from src.core.dependency_injection import services, ServiceLocator
        
        # Register a test service
        test_data = {"test": "value"}
        injector.register_service("TestData", test_data, singleton=True)
        
        # Test service locator
        locator = ServiceLocator(injector)
        retrieved_data = locator.TestData
        assert retrieved_data == test_data
        
        # Test attribute error for missing service
        with pytest.raises(AttributeError):
            _ = locator.NonexistentService
            
    def test_injectable_decorator(self, injector):
        """Test injectable decorator functionality."""
        from src.core.dependency_injection import injectable, get_global_injector

        # Get the global injector that the decorator uses
        global_injector = get_global_injector()
        global_injector.clear()  # Clean slate for test

        @injectable("TestInjectableService", singleton=True)
        class TestInjectableService:
            def __init__(self):
                self.value = "injectable"

        # Service should be registered in global injector
        assert global_injector.has_service("TestInjectableService")

        # Resolve service
        service = global_injector.resolve("TestInjectableService")
        assert service.value == "injectable"

        # Clean up
        global_injector.clear()
        
    @pytest.mark.asyncio
    async def test_service_factory_pattern(self, injector):
        """Test service factory registration and resolution."""
        def create_test_service():
            return {"factory_created": True, "timestamp": "test"}
            
        injector.register_factory("FactoryService", create_test_service, singleton=False)
        
        # Resolve multiple times to verify factory behavior
        service1 = injector.resolve("FactoryService")
        service2 = injector.resolve("FactoryService")
        
        assert service1["factory_created"] is True
        assert service2["factory_created"] is True
        # For non-singleton, each resolution creates new instance
        assert service1 is not service2
        
    @pytest.mark.asyncio
    async def test_interface_registration(self, injector):
        """Test interface-based service registration."""
        from abc import ABC, abstractmethod
        
        class TestInterface(ABC):
            @abstractmethod
            def test_method(self):
                pass
                
        class TestImplementation(TestInterface):
            def test_method(self):
                return "implementation"
                
        def create_implementation():
            return TestImplementation()
            
        injector.register_interface(TestInterface, create_implementation, singleton=True)
        
        # Resolve by interface name
        impl = injector.resolve("TestInterface")
        assert impl.test_method() == "implementation"
        assert isinstance(impl, TestImplementation)
        
    def test_container_operations(self, injector):
        """Test container basic operations."""
        container = injector.get_container()
        
        # Test service registration and retrieval
        test_service = {"test": "container"}
        container.register("ContainerTest", test_service)
        
        # Test has method
        assert container.has("ContainerTest")
        assert "ContainerTest" in container
        
        # Test get method
        retrieved = container.get("ContainerTest")
        assert retrieved == test_service
        
        # Test clear
        container.clear()
        assert not container.has("ContainerTest")


class TestCoreModuleBoundaries:
    """Test module boundary validation."""
    
    @pytest.mark.asyncio
    async def test_core_exception_propagation(self):
        """Test that core exceptions are properly used across modules."""
        # Test that other modules properly use core exceptions
        from src.core.exceptions import ServiceError, ValidationError, DependencyError
        
        # These should be the standard exceptions used throughout the codebase
        assert issubclass(ServiceError, Exception)
        assert issubclass(ValidationError, Exception) 
        assert issubclass(DependencyError, Exception)
        
        # Test exception with context
        error = ServiceError(
            "Test error",
            details={"component": "test", "operation": "test"}
        )
        assert "Test error" in str(error)
        assert error.details["component"] == "test"
        
    def test_core_types_usage(self):
        """Test that core types are properly used across modules."""
        from src.core.types import (
            OrderRequest, OrderResponse, OrderStatus,
            Signal, StrategyConfig, BotConfiguration
        )
        
        # Test that core types can be instantiated
        order_request = OrderRequest(
            symbol="BTC/USD",
            side="buy",
            order_type="market",
            quantity=1.0
        )
        assert order_request.symbol == "BTC/USD"
        assert order_request.side.value == "buy"
        
    def test_base_service_inheritance(self):
        """Test that services properly inherit from BaseService."""
        # Create a minimal service that follows the pattern
        class MinimalService(BaseService):
            def __init__(self):
                super().__init__(name="MinimalService")
                
        service = MinimalService()
        
        # Test base functionality
        assert service.name == "MinimalService"
        assert hasattr(service, 'configure_dependencies')
        assert hasattr(service, 'resolve_dependency')
        assert hasattr(service, 'execute_with_monitoring')
        
    @pytest.mark.asyncio
    async def test_module_integration_contracts(self):
        """Test that module integration contracts are properly defined."""
        # Test that required interfaces exist
        from src.core.base.interfaces import ServiceComponent, HealthStatus
        
        # Test health status enum
        assert hasattr(HealthStatus, 'HEALTHY')
        assert hasattr(HealthStatus, 'DEGRADED')
        assert hasattr(HealthStatus, 'UNHEALTHY')
        
    def test_configuration_integration(self):
        """Test core configuration integration."""
        from src.core.config import Config
        
        # Test basic config loading
        config = Config()
        config_dict = config.to_dict()
        
        # Should have basic structure
        assert isinstance(config_dict, dict)
        
    @pytest.mark.asyncio
    async def test_logging_integration(self):
        """Test logging integration across modules."""
        from src.core.logging import get_logger
        
        logger = get_logger("test_module")
        
        # Test logger functionality
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'debug')


class TestModuleIntegrationPatterns:
    """Test common integration patterns."""
    
    @pytest.mark.asyncio
    async def test_service_to_service_communication(self):
        """Test service-to-service communication patterns."""
        injector = DependencyInjector()
        
        class ServiceA(BaseService):
            def __init__(self):
                super().__init__(name="ServiceA")
                self.service_b = None
                
            def configure_dependencies(self, dependency_injector):
                super().configure_dependencies(dependency_injector)
                try:
                    self.service_b = self.resolve_dependency("ServiceB")
                except DependencyError:
                    pass
                    
            async def call_service_b(self, data):
                if self.service_b:
                    return await self.service_b.process_data(data)
                return {"error": "ServiceB not available"}
        
        class ServiceB(BaseService):
            def __init__(self):
                super().__init__(name="ServiceB")
                
            async def process_data(self, data):
                return {"processed": True, "original": data}
        
        # Register services
        service_a = ServiceA()
        service_b = ServiceB()
        
        injector.register_service("ServiceA", service_a, singleton=True)
        injector.register_service("ServiceB", service_b, singleton=True)
        
        # Configure dependencies
        service_a.configure_dependencies(injector)
        
        # Test service communication
        result = await service_a.call_service_b({"test": "data"})
        assert result["processed"] is True
        assert result["original"]["test"] == "data"
        
        injector.clear()
        
    def test_repository_pattern_integration(self):
        """Test repository pattern integration with core."""
        from src.core.base.repository import BaseRepository
        
        class TestEntity:
            def __init__(self, id=None, name=None):
                self.id = id
                self.name = name
        
        class TestRepository(BaseRepository):
            def __init__(self):
                super().__init__(TestEntity, str, "TestRepository")
                self.data = {}

            async def create(self, entity):
                entity.id = f"test_{len(self.data)}"
                self.data[entity.id] = entity
                return entity

            async def get_by_id(self, entity_id):
                return self.data.get(entity_id)

            # Implement required abstract methods
            async def _create_entity(self, entity):
                return await self.create(entity)

            async def _get_entity_by_id(self, entity_id):
                return await self.get_by_id(entity_id)

            async def _update_entity(self, entity):
                if entity.id in self.data:
                    self.data[entity.id] = entity
                    return entity
                return None

            async def _delete_entity(self, entity_id):
                return self.data.pop(entity_id, None)

            async def _list_entities(self, **kwargs):
                return list(self.data.values())

            async def _count_entities(self, **kwargs):
                return len(self.data)
        
        repo = TestRepository()
        assert repo.entity_type == TestEntity
        assert repo.name == "TestRepository"
        
    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling integration across modules."""
        from src.core.exceptions import ValidationError, ServiceError
        
        class TestService(BaseService):
            def __init__(self):
                super().__init__(name="TestService")
                
            async def failing_operation(self):
                return await self.execute_with_monitoring(
                    "failing_operation",
                    self._failing_impl
                )
                
            async def _failing_impl(self):
                raise ValidationError("Test validation failure")
        
        service = TestService()
        
        # ValidationError should propagate through service layer
        with pytest.raises(ValidationError):
            await service.failing_operation()
            
    @pytest.mark.asyncio 
    async def test_async_context_management(self):
        """Test async context management in services."""
        class AsyncContextService(BaseService):
            def __init__(self):
                super().__init__(name="AsyncContextService")
                self.context_entered = False
                self.context_exited = False
                
            async def __aenter__(self):
                self.context_entered = True
                await self.start()
                return self
                
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.context_exited = True
                await self.stop()
                
        async with AsyncContextService() as service:
            assert service.context_entered is True
            assert service.is_running is True
            
        assert service.context_exited is True
        assert service.is_running is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])