"""
Tests for simplified monitoring dependency injection.

This module tests the simplified factory functions for creating monitoring services.
"""

from unittest.mock import Mock, patch

import pytest

from src.monitoring.dependency_injection import (
    DIContainer,
    create_alert_manager,
    create_alert_service,
    create_metrics_collector,
    create_metrics_service,
    create_monitoring_service,
    create_performance_profiler,
    create_performance_service,
    get_monitoring_container,
    setup_monitoring_dependencies,
)


class TestSimpleDIContainer:
    """Test simplified DIContainer class."""

    def test_di_container_resolve_string_matching(self):
        """Test DIContainer resolve method with string matching."""
        container = DIContainer()

        # Test resolving different interface types
        with patch("src.monitoring.dependency_injection.create_metrics_service") as mock_create:
            mock_service = Mock()
            mock_create.return_value = mock_service

            # Test with string containing MetricsServiceInterface
            result = container.resolve(type("MetricsServiceInterface", (), {}))
            mock_create.assert_called_once()

    def test_di_container_resolve_unknown_type(self):
        """Test DIContainer resolve method with unknown type."""
        container = DIContainer()

        with pytest.raises(ValueError, match="No factory function"):
            container.resolve(type("UnknownInterface", (), {}))


class TestFactoryFunctions:
    """Test simplified factory functions."""

    @patch("src.monitoring.metrics.MetricsCollector")
    def test_create_metrics_collector(self, mock_metrics_collector):
        """Test create_metrics_collector function."""
        mock_instance = Mock()
        mock_metrics_collector.return_value = mock_instance

        result = create_metrics_collector()

        mock_metrics_collector.assert_called_once()
        assert result == mock_instance

    @patch("src.monitoring.alerting.AlertManager")
    @patch("src.monitoring.alerting.NotificationConfig")
    def test_create_alert_manager(self, mock_notification_config, mock_alert_manager):
        """Test create_alert_manager function."""
        mock_config = Mock()
        mock_notification_config.return_value = mock_config
        mock_instance = Mock()
        mock_alert_manager.return_value = mock_instance

        result = create_alert_manager()

        mock_notification_config.assert_called_once()
        mock_alert_manager.assert_called_once_with(mock_config)
        assert result == mock_instance

    @patch("src.monitoring.dependency_injection.create_metrics_collector")
    @patch("src.monitoring.dependency_injection.create_alert_manager")
    @patch("src.monitoring.performance.PerformanceProfiler")
    def test_create_performance_profiler(self, mock_profiler, mock_create_alert, mock_create_metrics):
        """Test create_performance_profiler function."""
        with patch("src.monitoring.dependency_injection.get_monitoring_container") as mock_get_container:
            mock_container = Mock()
            mock_get_container.return_value = mock_container
            mock_profiler_instance = Mock()
            mock_container.resolve.return_value = mock_profiler_instance

            result = create_performance_profiler()

            mock_container.resolve.assert_called()
            assert result == mock_profiler_instance

    @patch("src.monitoring.dependency_injection.create_alert_manager")
    @patch("src.monitoring.services.DefaultAlertService")
    def test_create_alert_service(self, mock_alert_service, mock_create_alert):
        """Test create_alert_service function."""
        mock_alert_instance = Mock()
        mock_create_alert.return_value = mock_alert_instance
        mock_service_instance = Mock()
        mock_alert_service.return_value = mock_service_instance
        
        result = create_alert_service()
        
        mock_create_alert.assert_called_once()
        mock_alert_service.assert_called_once_with(mock_alert_instance)
        assert result == mock_service_instance

    @patch("src.monitoring.dependency_injection.create_metrics_collector")
    @patch("src.monitoring.services.DefaultMetricsService")
    def test_create_metrics_service(self, mock_metrics_service, mock_create_metrics):
        """Test create_metrics_service function."""
        mock_collector_instance = Mock()
        mock_create_metrics.return_value = mock_collector_instance
        mock_service_instance = Mock()
        mock_metrics_service.return_value = mock_service_instance
        
        result = create_metrics_service()
        
        mock_create_metrics.assert_called_once()
        mock_metrics_service.assert_called_once_with(mock_collector_instance)
        assert result == mock_service_instance

    @patch("src.monitoring.dependency_injection.create_performance_profiler")
    @patch("src.monitoring.services.DefaultPerformanceService")
    def test_create_performance_service(self, mock_performance_service, mock_create_profiler):
        """Test create_performance_service function."""
        mock_profiler = Mock()
        mock_create_profiler.return_value = mock_profiler
        mock_instance = Mock()
        mock_performance_service.return_value = mock_instance

        result = create_performance_service()

        mock_create_profiler.assert_called_once()
        mock_performance_service.assert_called_once_with(mock_profiler)
        assert result == mock_instance

    @patch("src.monitoring.dependency_injection.create_alert_service")
    @patch("src.monitoring.dependency_injection.create_metrics_service")
    @patch("src.monitoring.dependency_injection.create_performance_service")
    @patch("src.monitoring.services.MonitoringService")
    def test_create_monitoring_service(self, mock_monitoring_service, mock_perf, mock_metrics, mock_alerts):
        """Test create_monitoring_service function."""
        mock_alert_service = Mock()
        mock_metrics_service = Mock()
        mock_performance_service = Mock()

        mock_alerts.return_value = mock_alert_service
        mock_metrics.return_value = mock_metrics_service
        mock_perf.return_value = mock_performance_service

        mock_instance = Mock()
        mock_monitoring_service.return_value = mock_instance

        result = create_monitoring_service()

        mock_alerts.assert_called_once()
        mock_metrics.assert_called_once()
        mock_perf.assert_called_once()
        mock_monitoring_service.assert_called_once_with(
            mock_alert_service, mock_metrics_service, mock_performance_service
        )
        assert result == mock_instance


class TestBackwardCompatibility:
    """Test backward compatibility functions."""

    def test_get_monitoring_container(self):
        """Test get_monitoring_container function."""
        result = get_monitoring_container()

        assert isinstance(result, DIContainer)

    def test_setup_monitoring_dependencies(self):
        """Test setup_monitoring_dependencies function."""
        # Should not raise any exceptions
        setup_monitoring_dependencies()


class TestIntegration:
    """Test integration scenarios."""

    def test_full_service_creation_chain(self):
        """Test creating monitoring service through full chain."""
        with patch("src.monitoring.metrics.MetricsCollector") as mock_metrics, \
             patch("src.monitoring.alerting.AlertManager") as mock_alert, \
             patch("src.monitoring.alerting.NotificationConfig") as mock_config, \
             patch("src.monitoring.performance.PerformanceProfiler") as mock_profiler, \
             patch("src.monitoring.services.DefaultAlertService") as mock_alert_service, \
             patch("src.monitoring.services.DefaultMetricsService") as mock_metrics_service, \
             patch("src.monitoring.services.DefaultPerformanceService") as mock_perf_service, \
             patch("src.monitoring.services.MonitoringService") as mock_monitoring:

            # Mock the return values
            mock_config.return_value = Mock()
            mock_metrics.return_value = Mock()
            mock_alert.return_value = Mock()
            mock_profiler.return_value = Mock()
            mock_alert_service.return_value = Mock()
            mock_metrics_service.return_value = Mock()
            mock_perf_service.return_value = Mock()
            mock_monitoring.return_value = Mock()

            # Create monitoring service
            result = create_monitoring_service()

            # Verify all components were created
            mock_config.assert_called()
            mock_metrics.assert_called()
            mock_alert.assert_called()
            mock_profiler.assert_called()
            mock_monitoring.assert_called()

            assert result is not None
