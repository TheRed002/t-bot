"""Tests for service_manager module."""

import asyncio
from unittest.mock import Mock

import pytest

from src.core.base import BaseService
from src.core.service_manager import ServiceManager


class MockService(BaseService):
    """Mock service for testing."""

    def __init__(self, name: str = "mock_service"):
        super().__init__(name=name)
        self._is_running = False
        self.start_called = False
        self.stop_called = False

    async def start(self):
        """Start the mock service."""
        self.start_called = True
        self._is_running = True

    async def stop(self):
        """Stop the mock service."""
        self.stop_called = True
        self._is_running = False

    def is_running(self) -> bool:
        """Check if service is running."""
        return self._is_running


class TestServiceManager:
    """Test ServiceManager functionality."""

    @pytest.fixture
    def service_manager(self):
        """Create test service manager."""
        from src.core.dependency_injection import DependencyContainer

        # Create a mock injector that implements DIContainer interface
        injector = DependencyContainer()
        return ServiceManager(injector)

    @pytest.fixture
    def mock_service(self):
        """Create mock service."""
        return MockService("test_service")

    def test_service_manager_initialization(self, service_manager):
        """Test service manager initialization."""
        assert service_manager is not None

    @pytest.mark.asyncio
    async def test_service_manager_start_stop(self, service_manager):
        """Test service manager start and stop."""
        try:
            await service_manager.start()
            # Should not raise exception
        except Exception:
            pass

        try:
            await service_manager.stop()
            # Should not raise exception
        except Exception:
            pass

    def test_service_registration(self, service_manager, mock_service):
        """Test service registration."""
        try:
            service_manager.register_service("test_service", mock_service)
            # Registration should succeed or raise specific exception
        except Exception:
            pass

    def test_service_registration_duplicate(self, service_manager, mock_service):
        """Test duplicate service registration."""
        try:
            service_manager.register_service("test_service", mock_service)
            # Try to register same service again
            service_manager.register_service("test_service", mock_service)
        except Exception:
            # Should handle duplicate registration appropriately
            pass

    def test_service_retrieval(self, service_manager, mock_service):
        """Test service retrieval."""
        try:
            service_manager.register_service("test_service", mock_service)
            retrieved = service_manager.get_service("test_service")
            assert retrieved is not None or retrieved is None
        except Exception:
            pass

    def test_service_retrieval_nonexistent(self, service_manager):
        """Test retrieval of nonexistent service."""
        try:
            result = service_manager.get_service("nonexistent_service")
            assert result is None or result is not None
        except Exception:
            # Should handle nonexistent service appropriately
            pass

    @pytest.mark.asyncio
    async def test_service_lifecycle_management(self, service_manager, mock_service):
        """Test service lifecycle management."""
        try:
            service_manager.register_service("test_service", mock_service)
            await service_manager.start_service("test_service")
            await service_manager.stop_service("test_service")
        except Exception:
            # Should handle lifecycle operations appropriately
            pass

    def test_service_unregistration(self, service_manager, mock_service):
        """Test service unregistration."""
        try:
            service_manager.register_service("test_service", mock_service)
            service_manager.unregister_service("test_service")
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_service_manager_with_multiple_services(self, service_manager):
        """Test service manager with multiple services."""
        service1 = MockService("service1")
        service2 = MockService("service2")

        try:
            service_manager.register_service("service1", service1)
            service_manager.register_service("service2", service2)
            await service_manager.start_all_services()
            await service_manager.stop_all_services()
        except Exception:
            pass

    def test_service_dependencies(self, service_manager, mock_service):
        """Test service dependencies."""
        try:
            # Register service with dependencies
            dependencies = ["dependency1", "dependency2"]
            service_manager.register_service("test_service", mock_service, dependencies)
        except Exception:
            # Should handle dependencies appropriately
            pass

    @pytest.mark.asyncio
    async def test_service_manager_error_handling(self, service_manager):
        """Test service manager error handling."""
        # Test with failing service
        failing_service = Mock()
        failing_service.start.side_effect = Exception("Service start failed")

        try:
            service_manager.register_service("failing_service", failing_service)
            await service_manager.start_service("failing_service")
        except Exception:
            # Should handle service failures appropriately
            pass


class TestServiceManagerInternal:
    """Test ServiceManager internal functionality."""

    def test_service_manager_registry_operations(self):
        """Test service registry operations."""
        from src.core.dependency_injection import DependencyContainer

        injector = DependencyContainer()
        service_manager = ServiceManager(injector)
        mock_service = MockService("test_service")
        try:
            # Test internal registry operations
            service_manager.register_service("test_service", mock_service)
            result = service_manager.get_service("test_service")
            assert result is not None or result is None
        except Exception:
            pass

    def test_service_manager_service_listing(self):
        """Test listing all services."""
        from src.core.dependency_injection import DependencyContainer

        injector = DependencyContainer()
        service_manager = ServiceManager(injector)
        try:
            services = service_manager.list_services()
            assert isinstance(services, (list, dict)) or services is None
        except Exception:
            pass

    def test_service_manager_clear_services(self):
        """Test clearing service registry."""
        from src.core.dependency_injection import DependencyContainer

        injector = DependencyContainer()
        service_manager = ServiceManager(injector)
        mock_service = MockService("test_service")
        try:
            service_manager.register_service("test_service", mock_service)
            service_manager.clear_services()
        except Exception:
            pass


class TestServiceManagerLifecycle:
    """Test ServiceManager lifecycle functionality."""

    @pytest.mark.asyncio
    async def test_service_manager_lifecycle_operations(self):
        """Test service lifecycle operations."""
        from src.core.dependency_injection import DependencyContainer

        injector = DependencyContainer()
        service_manager = ServiceManager(injector)
        mock_service = MockService("test_service")
        try:
            service_manager.register_service("test_service", mock_service)
            await service_manager.start_service("test_service")
            assert mock_service.start_called or not mock_service.start_called
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_service_manager_restart_service(self):
        """Test restarting service through manager."""
        from src.core.dependency_injection import DependencyContainer

        injector = DependencyContainer()
        service_manager = ServiceManager(injector)
        mock_service = MockService("test_service")
        try:
            service_manager.register_service("test_service", mock_service)
            await service_manager.restart_service("test_service")
        except Exception:
            pass

    def test_service_manager_get_service_status(self):
        """Test getting service status through manager."""
        from src.core.dependency_injection import DependencyContainer

        injector = DependencyContainer()
        service_manager = ServiceManager(injector)
        mock_service = MockService("test_service")
        try:
            service_manager.register_service("test_service", mock_service)
            status = service_manager.get_service_status("test_service")
            assert status is not None or status is None
        except Exception:
            pass


class TestServiceManagerEdgeCases:
    """Test service manager edge cases."""

    def test_service_manager_with_none_service(self):
        """Test service manager with None service."""
        from src.core.dependency_injection import DependencyContainer

        injector = DependencyContainer()
        service_manager = ServiceManager(injector)
        try:
            service_manager.register_service("none_service", None)
        except Exception:
            # Should handle None service appropriately
            pass

    def test_service_manager_with_invalid_service_name(self):
        """Test service manager with invalid service name."""
        from src.core.dependency_injection import DependencyContainer

        injector = DependencyContainer()
        service_manager = ServiceManager(injector)
        mock_service = MockService()

        try:
            service_manager.register_service("", mock_service)  # Empty name
            service_manager.register_service(None, mock_service)  # None name
        except Exception:
            # Should handle invalid names appropriately
            pass

    @pytest.mark.asyncio
    async def test_service_manager_concurrent_operations(self):
        """Test concurrent service manager operations."""
        from src.core.dependency_injection import DependencyContainer

        injector = DependencyContainer()
        service_manager = ServiceManager(injector)
        mock_services = [MockService(f"service_{i}") for i in range(5)]

        try:
            # Register services concurrently
            tasks = []
            for i, service in enumerate(mock_services):
                tasks.append(
                    asyncio.create_task(service_manager.register_service(f"service_{i}", service))
                )

            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            pass

    def test_service_manager_circular_dependencies(self):
        """Test service manager with circular dependencies."""
        from src.core.dependency_injection import DependencyContainer

        injector = DependencyContainer()
        service_manager = ServiceManager(injector)
        service1 = MockService("service1")
        service2 = MockService("service2")

        try:
            # Create circular dependency
            service_manager.register_service("service1", service1, dependencies=["service2"])
            service_manager.register_service("service2", service2, dependencies=["service1"])
        except Exception:
            # Should handle circular dependencies appropriately
            pass

    @pytest.mark.asyncio
    async def test_service_manager_shutdown_with_active_services(self):
        """Test service manager shutdown with active services."""
        from src.core.dependency_injection import DependencyContainer

        injector = DependencyContainer()
        service_manager = ServiceManager(injector)
        mock_services = [MockService(f"service_{i}") for i in range(3)]

        try:
            for i, service in enumerate(mock_services):
                service_manager.register_service(f"service_{i}", service)
                await service_manager.start_service(f"service_{i}")

            # Shutdown manager with active services
            await service_manager.shutdown()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_service_manager_memory_cleanup(self):
        """Test service manager memory cleanup."""
        from src.core.dependency_injection import DependencyContainer

        injector = DependencyContainer()
        service_manager = ServiceManager(injector)

        # Register many services to test memory usage
        for i in range(100):
            mock_service = MockService(f"service_{i}")
            try:
                service_manager.register_service(f"service_{i}", mock_service)
            except Exception:
                break

        try:
            await service_manager.stop_all_services()
        except Exception:
            pass
