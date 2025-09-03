"""Simple tests for monitoring dependency injection module."""

import pytest
from unittest.mock import Mock, patch
from src.monitoring.dependency_injection import (
    DIContainer,
    create_factory,
    get_monitoring_container,
    setup_monitoring_dependencies,
)


class TestDIContainer:
    """Test DIContainer basic functionality."""

    def test_di_container_init(self):
        """Test DIContainer initialization."""
        container = DIContainer()
        
        assert container._bindings == {}
        assert container._resolving == set()

    def test_register_factory(self):
        """Test register_factory method."""
        container = DIContainer()
        
        def test_factory():
            return "test_instance"
        
        class TestService: pass
        
        container.register(TestService, factory=test_factory)
        
        assert TestService in container._bindings
        assert container._bindings[TestService].factory is test_factory

    def test_register_factory_singleton(self):
        """Test register_factory with singleton=True."""
        container = DIContainer()
        
        def test_factory():
            return "test_singleton"
        
        class TestService: pass
        
        container.register(TestService, factory=test_factory, singleton=True)
        
        instance1 = container.resolve(TestService)
        instance2 = container.resolve(TestService)
        
        assert instance1 == instance2  # Same instance

    def test_register_instance(self):
        """Test registering an instance using factory."""
        container = DIContainer()
        test_instance = Mock()
        
        class TestService: pass
        
        container.register(TestService, factory=lambda: test_instance)
        
        resolved = container.resolve(TestService)
        assert resolved is test_instance

    def test_resolve_factory(self):
        """Test resolve method with factory."""
        container = DIContainer()
        
        def test_factory():
            return Mock()
        
        class TestService: pass
        
        container.register(TestService, factory=test_factory)
        
        instance = container.resolve(TestService)
        assert instance is not None

    def test_resolve_nonexistent_service(self):
        """Test resolve method with nonexistent service."""
        container = DIContainer()
        
        class NonexistentService: pass
        
        with pytest.raises(ValueError):
            container.resolve(NonexistentService)

    def test_is_registered(self):
        """Test checking if service is registered."""
        container = DIContainer()
        
        class TestService: pass
        
        assert TestService not in container._bindings
        
        container.register(TestService, factory=lambda: Mock())
        
        assert TestService in container._bindings


class TestCreateFactory:
    """Test create_factory function."""

    def test_create_factory_basic(self):
        """Test create_factory with basic class."""
        class TestService:
            def __init__(self):
                self.value = "test"
        
        factory = create_factory(TestService)
        
        with patch('src.monitoring.dependency_injection.get_monitoring_container') as mock_get_container:
            mock_container = Mock()
            mock_get_container.return_value = mock_container
            
            instance = factory()
            assert isinstance(instance, TestService)
            assert instance.value == "test"

    def test_create_factory_with_dependencies(self):
        """Test create_factory with dependencies."""
        class Dependency:
            pass
        
        class TestService:
            def __init__(self, dep: Dependency):
                self.dep = dep
        
        factory = create_factory(TestService, Dependency)
        
        with patch('src.monitoring.dependency_injection.get_monitoring_container') as mock_get_container:
            mock_container = Mock()
            mock_dependency = Dependency()
            mock_container.resolve.return_value = mock_dependency
            mock_get_container.return_value = mock_container
            
            instance = factory()
            assert isinstance(instance, TestService)
            assert instance.dep is mock_dependency


class TestGlobalFunctions:
    """Test global monitoring container functions."""

    def test_get_monitoring_container_singleton(self):
        """Test get_monitoring_container returns singleton."""
        container1 = get_monitoring_container()
        container2 = get_monitoring_container()
        
        assert container1 is container2

    @patch('src.monitoring.dependency_injection.logger')
    def test_setup_monitoring_dependencies(self, mock_logger):
        """Test setup_monitoring_dependencies."""
        # Clear any existing container
        container = get_monitoring_container()
        container.clear()  # Clear existing registrations
        
        setup_monitoring_dependencies()
        
        # Check that some services are registered by type
        from src.monitoring.metrics import MetricsCollector
        assert MetricsCollector in container._bindings


class TestServiceRegistration:
    """Test service registration utilities."""

    def test_register_metrics_collector(self):
        """Test registering MetricsCollector."""
        container = DIContainer()
        
        def metrics_factory():
            from src.monitoring.metrics import MetricsCollector
            return MetricsCollector()
        
        from src.monitoring.metrics import MetricsCollector
        container.register(MetricsCollector, factory=metrics_factory, singleton=True)
        
        # Should be able to resolve
        collector1 = container.resolve(MetricsCollector)
        collector2 = container.resolve(MetricsCollector)
        
        assert collector1 is collector2  # Singleton behavior

    def test_register_alert_manager(self):
        """Test registering AlertManager."""
        container = DIContainer()
        
        def alert_factory():
            from src.monitoring.alerting import AlertManager, NotificationConfig
            return AlertManager(NotificationConfig())
        
        from src.monitoring.alerting import AlertManager
        container.register(AlertManager, factory=alert_factory, singleton=True)
        
        # Should be able to resolve
        manager = container.resolve(AlertManager)
        assert manager is not None


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_resolve_with_factory_exception(self):
        """Test resolve when factory raises exception."""
        container = DIContainer()
        
        def failing_factory():
            raise Exception("Factory failed")
        
        class FailingService: pass
        
        container.register(FailingService, factory=failing_factory)
        
        with pytest.raises(Exception, match="Factory failed"):
            container.resolve(FailingService)

    def test_create_factory_with_missing_dependencies(self):
        """Test create_factory when dependencies can't be resolved."""
        class Dependency:
            pass
        
        class TestService:
            def __init__(self, dep=None):
                self.dep = dep
        
        factory = create_factory(TestService, Dependency)
        
        with patch('src.monitoring.dependency_injection.get_monitoring_container') as mock_get_container:
            mock_container = Mock()
            mock_container.resolve.side_effect = KeyError("Dependency not found")
            mock_get_container.return_value = mock_container
            
            # Should still work, just without the dependency
            instance = factory()
            assert isinstance(instance, TestService)
            assert instance.dep is None  # No dependency injected