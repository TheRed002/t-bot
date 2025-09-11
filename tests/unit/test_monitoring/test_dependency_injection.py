"""
Test suite for monitoring dependency injection module.

Tests cover dependency registration and resolution for monitoring components.
"""



from src.monitoring.dependency_injection import (
    DIContainer,
    get_monitoring_container,
    setup_monitoring_dependencies,
)


class TestDependencyRegistration:
    """Test dependency registration functionality."""

    def test_setup_monitoring_dependencies(self):
        """Test setup monitoring dependencies."""
        # Should not raise any exceptions
        setup_monitoring_dependencies()

    def test_get_container(self):
        """Test get container function."""
        # Import the module to ensure it's initialized
        import src.monitoring.dependency_injection as di_module
        
        # Ensure container is properly initialized
        if not hasattr(di_module, '_container') or di_module._container is None:
            di_module._container = DIContainer()
            
        container = get_monitoring_container()
        assert container is not None
        assert isinstance(container, DIContainer)

    def test_di_container_creation(self):
        """Test DIContainer creation."""
        container = DIContainer()
        assert container is not None

    def test_di_container_resolve_metrics(self):
        """Test DIContainer resolve metrics service."""
        container = DIContainer()

        # Should be able to resolve metrics service
        from src.monitoring.interfaces import MetricsServiceInterface
        result = container.resolve(MetricsServiceInterface)
        assert result is not None

    def test_di_container_resolve_alert_service(self):
        """Test DIContainer resolve alert service."""
        # Import the module to ensure it's initialized
        import src.monitoring.dependency_injection as di_module
        
        # Ensure container is properly initialized
        if not hasattr(di_module, '_container') or di_module._container is None:
            di_module._container = DIContainer()
        
        # Set up dependencies first
        setup_monitoring_dependencies()

        # Should be able to resolve alert service
        from src.monitoring.interfaces import AlertServiceInterface
        
        # Use the global container that has been set up
        container = get_monitoring_container()
        assert container is not None
        result = container.resolve(AlertServiceInterface)
        assert result is not None
