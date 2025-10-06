"""
Integration tests for monitoring module dependencies and API usage.

This test module validates proper integration patterns between monitoring
module and other modules, ensuring correct dependency injection, API usage,
and data contracts are maintained.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from src.core.dependency_injection import DependencyInjector
from src.core.exceptions import ComponentError, ValidationError
from src.core.types import OrderType
from src.monitoring.di_registration import register_monitoring_services
from src.monitoring.services import (
    AlertRequest,
    DefaultAlertService,
    DefaultMetricsService,
    DefaultPerformanceService,
    MetricRequest,
    MonitoringService,
)


class TestMonitoringModuleIntegration:
    """Test monitoring module integration with other modules."""

    @pytest.fixture
    def mock_injector(self):
        """Create mock dependency injector."""
        return Mock(spec=DependencyInjector)

    @pytest.fixture
    def mock_alert_manager(self):
        """Create mock alert manager."""
        mock_manager = Mock()
        mock_manager.fire_alert = AsyncMock()
        mock_manager.resolve_alert = AsyncMock()
        mock_manager.acknowledge_alert = AsyncMock(return_value=True)
        mock_manager.get_active_alerts = Mock(return_value=[])
        mock_manager.get_alert_stats = Mock(return_value={})
        mock_manager.add_rule = Mock()
        mock_manager.add_escalation_policy = Mock()
        return mock_manager

    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector."""
        mock_collector = Mock()
        mock_collector.increment_counter = Mock()
        mock_collector.set_gauge = Mock()
        mock_collector.observe_histogram = Mock()
        mock_collector.export_metrics = Mock(return_value="# metrics")
        return mock_collector

    @pytest.fixture
    def mock_performance_profiler(self):
        """Create mock performance profiler."""
        mock_profiler = Mock()
        mock_profiler.get_performance_summary = Mock(return_value={})
        mock_profiler.record_order_execution = Mock()
        mock_profiler.record_market_data_processing = Mock()
        mock_profiler.get_latency_stats = Mock(return_value={})
        mock_profiler.get_system_resource_stats = Mock(return_value={})
        return mock_profiler

    def test_dependency_injection_registration(self, mock_injector):
        """Test that monitoring services are properly registered with DI container."""
        # Mock dependencies
        mock_metrics_collector = Mock()
        mock_alert_manager = Mock()
        mock_performance_profiler = Mock()
        mock_dashboard_manager = Mock()

        # Mock resolve calls
        mock_injector.resolve = Mock(
            side_effect=lambda key: {
                "MetricsCollector": mock_metrics_collector,
                "AlertManager": mock_alert_manager,
                "PerformanceProfiler": mock_performance_profiler,
                "GrafanaDashboardManager": mock_dashboard_manager,
                "DefaultMetricsService": DefaultMetricsService(mock_metrics_collector),
                "DefaultAlertService": DefaultAlertService(mock_alert_manager),
                "DefaultPerformanceService": DefaultPerformanceService(mock_performance_profiler),
            }.get(key)
        )

        # Mock register_factory
        mock_injector.register_factory = Mock()

        # Register services
        register_monitoring_services(mock_injector)

        # Verify services were registered
        assert mock_injector.register_factory.call_count >= 10

        # Check that main interface factories were registered
        factory_names = [call[0][0] for call in mock_injector.register_factory.call_args_list]
        assert "MetricsServiceInterface" in factory_names
        assert "AlertServiceInterface" in factory_names
        assert "PerformanceServiceInterface" in factory_names
        assert "MonitoringServiceInterface" in factory_names

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_alert_service_integration(self, mock_alert_manager):
        """Test alert service integration with proper data validation."""
        from src.monitoring.alerting import AlertSeverity

        alert_service = DefaultAlertService(mock_alert_manager)

        request = AlertRequest(
            rule_name="test_rule",
            severity=AlertSeverity.HIGH,
            message="Test alert",
            labels={"component": "test"},
            annotations={"details": "test details"},
        )

        # Test create_alert
        result = await alert_service.create_alert(request)

        # Verify alert manager was called
        mock_alert_manager.fire_alert.assert_called_once()
        alert = mock_alert_manager.fire_alert.call_args[0][0]
        assert alert.rule_name == "test_rule"
        assert alert.severity == AlertSeverity.HIGH

    def test_metrics_service_integration(self, mock_metrics_collector):
        """Test metrics service integration with proper data validation."""
        metrics_service = DefaultMetricsService(mock_metrics_collector)

        request = MetricRequest(
            name="test_counter",
            value=Decimal("1.5"),
            labels={"component": "test"},
            namespace="test",
        )

        # Test record_counter
        metrics_service.record_counter(request)

        # Verify metrics collector was called with proper conversion
        mock_metrics_collector.increment_counter.assert_called_once_with(
            "test_counter", {"component": "test"}, 1.5, "test"
        )

    def test_performance_service_integration(self, mock_performance_profiler):
        """Test performance service integration with order type validation."""
        performance_service = DefaultPerformanceService(mock_performance_profiler)

        # Test with string order type
        performance_service.record_order_execution(
            exchange="binance",
            order_type="MARKET",
            symbol="BTCUSDT",
            latency_ms=Decimal("50.0"),
            fill_rate=Decimal("1.0"),
            slippage_bps=Decimal("0.5"),
        )

        # Verify profiler was called with OrderType enum
        mock_performance_profiler.record_order_execution.assert_called_once()
        args = mock_performance_profiler.record_order_execution.call_args[0]
        assert args[0] == "binance"
        assert args[1] == OrderType.MARKET
        assert args[2] == "BTCUSDT"

    def test_monitoring_service_composite_integration(
        self, mock_alert_manager, mock_metrics_collector, mock_performance_profiler
    ):
        """Test MonitoringService as composite service."""
        alert_service = DefaultAlertService(mock_alert_manager)
        metrics_service = DefaultMetricsService(mock_metrics_collector)
        performance_service = DefaultPerformanceService(mock_performance_profiler)

        monitoring_service = MonitoringService(alert_service, metrics_service, performance_service)

        # Test service composition
        assert monitoring_service.alerts == alert_service
        assert monitoring_service.metrics == metrics_service
        assert monitoring_service.performance == performance_service

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_error_handling_to_monitoring_integration(self, mock_alert_manager):
        """Test error_handling -> monitoring boundary validation."""
        alert_service = DefaultAlertService(mock_alert_manager)

        # Test error event from error_handling module
        error_data = {
            "error_id": "test_error_123",
            "component": "execution_service",
            "severity": "high",
            "recovery_success": True,
            "operation": "place_order",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Should not raise validation error
        result = await alert_service.handle_error_event_from_error_handling(error_data)

    def test_boundary_validation_error_handling(self, mock_alert_manager):
        """Test boundary validation catches invalid data."""
        alert_service = DefaultAlertService(mock_alert_manager)

        # Test with invalid request type
        with pytest.raises(ValidationError) as exc_info:
            alert_service._transform_alert_request_data("invalid")

        # Should catch validation error at service boundary
        assert "AlertRequest" in str(exc_info.value)

    def test_financial_data_transformation(self, mock_metrics_collector):
        """Test financial data is properly transformed using Decimal types."""
        metrics_service = DefaultMetricsService(mock_metrics_collector)

        # Test with price metric containing financial data
        request = MetricRequest(
            name="btc_price_usd",
            value=Decimal("65432.50"),  # Financial value as Decimal
            labels={"symbol": "BTC-USD"},
            namespace="trading",
        )

        metrics_service.record_gauge(request)

        # Verify financial data was converted to float for metrics collector
        mock_metrics_collector.set_gauge.assert_called_once()
        args = mock_metrics_collector.set_gauge.call_args
        assert args[0][1] == 65432.50  # Converted to float
        assert isinstance(args[0][1], float)

    def test_service_dependency_validation(self):
        """Test service validates required dependencies at initialization."""
        # Test with None dependencies - should raise ValidationError
        with pytest.raises(ValueError, match="required - use dependency injection"):
            DefaultAlertService(None)

        with pytest.raises(ValueError, match="required - use dependency injection"):
            DefaultMetricsService(None)

        with pytest.raises(ValueError, match="required - use dependency injection"):
            DefaultPerformanceService(None)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_error_propagation_patterns(self, mock_alert_manager):
        """Test error propagation follows consistent patterns."""
        alert_service = DefaultAlertService(mock_alert_manager)

        # Mock alert manager to raise an exception
        mock_alert_manager.fire_alert.side_effect = Exception("Connection failed")

        from src.monitoring.alerting import AlertSeverity

        request = AlertRequest(
            rule_name="test_rule",
            severity=AlertSeverity.HIGH,
            message="Test alert",
            labels={},
            annotations={},
        )

        # Should catch exception and convert to ComponentError
        with pytest.raises(ComponentError) as exc_info:
            await alert_service.create_alert(request)

        error = exc_info.value
        assert "Failed to create alert" in str(error)
        assert error.component == "AlertService"
        assert error.operation == "create_alert"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_health_check_integration(
        self, mock_alert_manager, mock_metrics_collector, mock_performance_profiler
    ):
        """Test health check integration across services."""
        alert_service = DefaultAlertService(mock_alert_manager)
        metrics_service = DefaultMetricsService(mock_metrics_collector)
        performance_service = DefaultPerformanceService(mock_performance_profiler)

        monitoring_service = MonitoringService(alert_service, metrics_service, performance_service)

        health_status = await monitoring_service.health_check()

        assert health_status["monitoring_service"] == "healthy"
        assert "components" in health_status
        assert "timestamp" in health_status
        assert health_status["components"]["alerts"] == "healthy"
        assert health_status["components"]["metrics"] == "healthy"
        assert health_status["components"]["performance"] == "healthy"
