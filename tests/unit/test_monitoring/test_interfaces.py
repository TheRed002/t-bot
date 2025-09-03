"""
Test suite for monitoring interfaces module.

Tests cover monitoring service interfaces and their contracts.
"""

import pytest

from src.monitoring.interfaces import (
    AlertServiceInterface,
    DashboardServiceInterface,
    MetricsServiceInterface,
    MonitoringServiceInterface,
    PerformanceServiceInterface,
)


class TestAlertServiceInterface:
    """Test AlertServiceInterface abstract interface."""
    
    def test_alert_service_interface_is_abstract(self):
        """Test AlertServiceInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AlertServiceInterface()

    def test_alert_service_interface_methods(self):
        """Test AlertServiceInterface has required abstract methods."""
        # Check that the interface defines the expected methods
        assert hasattr(AlertServiceInterface, 'add_rule')
        assert hasattr(AlertServiceInterface, 'add_escalation_policy')


class TestMetricsServiceInterface:
    """Test MetricsServiceInterface abstract interface."""
    
    def test_metrics_service_interface_is_abstract(self):
        """Test MetricsServiceInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MetricsServiceInterface()


class TestPerformanceServiceInterface:
    """Test PerformanceServiceInterface abstract interface."""
    
    def test_performance_service_interface_is_abstract(self):
        """Test PerformanceServiceInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PerformanceServiceInterface()

    def test_performance_service_interface_methods(self):
        """Test PerformanceServiceInterface has required methods."""
        assert hasattr(PerformanceServiceInterface, 'get_latency_stats')
        assert hasattr(PerformanceServiceInterface, 'get_system_resource_stats')


class TestDashboardServiceInterface:
    """Test DashboardServiceInterface abstract interface."""
    
    def test_dashboard_service_interface_is_abstract(self):
        """Test DashboardServiceInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DashboardServiceInterface()


class TestMonitoringServiceInterface:
    """Test MonitoringServiceInterface abstract interface."""
    
    def test_monitoring_service_interface_is_abstract(self):
        """Test MonitoringServiceInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MonitoringServiceInterface()

    def test_monitoring_service_interface_methods(self):
        """Test MonitoringServiceInterface has required methods."""
        assert hasattr(MonitoringServiceInterface, 'get_health_status')