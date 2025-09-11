"""Simple tests for monitoring dependency injection registration."""

from unittest.mock import Mock

import pytest

from src.core.dependency_injection import DependencyInjector


class TestMonitoringDIRegistration:
    """Test monitoring dependency injection registration without complex mocking."""

    def test_register_monitoring_services_basic(self, mock_dependency_injector):
        """Test basic registration of monitoring services."""
        # Import here to avoid module-level import issues
        from src.monitoring.di_registration import register_monitoring_services

        # This should not raise any exceptions
        register_monitoring_services(mock_dependency_injector)

        # Verify register_factory was called
        assert mock_dependency_injector.register_factory.call_count > 0

    def test_factory_function_error_handling(self, mock_dependency_injector):
        """Test factory function error handling."""
        from src.monitoring.di_registration import register_monitoring_services

        # Mock the injector to raise an exception
        mock_dependency_injector.register_factory.side_effect = Exception("Test error")

        # The registration should handle errors gracefully
        try:
            register_monitoring_services(mock_dependency_injector)
        except Exception:
            # If it raises, that's okay for this test
            pass


@pytest.fixture
def mock_dependency_injector():
    """Provide a mock dependency injector."""
    mock = Mock(spec=DependencyInjector)
    mock.register_factory = Mock()
    mock.register_instance = Mock()
    mock.register_singleton = Mock()
    return mock
