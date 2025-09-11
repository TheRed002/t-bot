"""Simple tests for monitoring dependency injection module."""

from unittest.mock import Mock, patch

from src.monitoring.dependency_injection import (
    DIContainer,
    create_alert_manager,
    create_metrics_collector,
    get_monitoring_container,
    setup_monitoring_dependencies,
)


class TestDIContainer:
    """Test DIContainer basic functionality."""

    def test_di_container_init(self):
        """Test DIContainer initialization."""
        container = DIContainer()
        assert container is not None

    def test_resolve_metrics_service_interface(self):
        """Test resolving MetricsServiceInterface."""
        container = DIContainer()

        with patch("src.monitoring.dependency_injection.create_metrics_service") as mock_create:
            mock_service = Mock()
            mock_create.return_value = mock_service

            result = container.resolve(type("MetricsServiceInterface", (), {}))
            assert result == mock_service


class TestFactoryFunctions:
    """Test factory functions."""

    @patch("src.monitoring.metrics.MetricsCollector")
    def test_create_metrics_collector(self, mock_collector):
        """Test create_metrics_collector."""
        mock_instance = Mock()
        mock_collector.return_value = mock_instance

        result = create_metrics_collector()

        mock_collector.assert_called_once()
        assert result == mock_instance

    @patch("src.monitoring.alerting.AlertManager")
    @patch("src.monitoring.alerting.NotificationConfig")
    def test_create_alert_manager(self, mock_config, mock_manager):
        """Test create_alert_manager."""
        mock_config_instance = Mock()
        mock_manager_instance = Mock()
        mock_config.return_value = mock_config_instance
        mock_manager.return_value = mock_manager_instance

        result = create_alert_manager()

        mock_config.assert_called_once()
        mock_manager.assert_called_once_with(mock_config_instance)
        assert result == mock_manager_instance


class TestGlobalFunctions:
    """Test global functions."""

    def test_get_monitoring_container(self):
        """Test get_monitoring_container."""
        result = get_monitoring_container()
        assert isinstance(result, DIContainer)

    def test_setup_monitoring_dependencies(self):
        """Test setup_monitoring_dependencies."""
        # Should not raise any exceptions
        setup_monitoring_dependencies()
