"""
Integration tests for monitoring module dependencies and service usage.

Tests that the monitoring module properly integrates with other modules
using correct dependency injection patterns and service interfaces.
"""

from decimal import Decimal

import pytest

from src.core.types import OrderType
from src.monitoring import (
    get_alert_manager,
    get_metrics_collector,
    get_performance_profiler,
    initialize_monitoring_service,
    setup_monitoring_dependencies,
)
from src.monitoring.dependency_injection import (
    create_alert_service,
    create_metrics_service,
    create_monitoring_service,
    create_performance_service,
    get_monitoring_container,
)
from src.monitoring.interfaces import (
    AlertServiceInterface,
    MetricsServiceInterface,
    MonitoringServiceInterface,
    PerformanceServiceInterface,
)
from src.monitoring.services import (
    AlertRequest,
    MetricRequest,
    MonitoringService,
)


class TestMonitoringDependencyInjection:
    """Test monitoring dependency injection container."""

    def test_di_container_registration(self):
        """Test that DI container properly registers monitoring services."""
        # Setup container with monitoring dependencies
        setup_monitoring_dependencies()
        container = get_monitoring_container()

        # Verify container has required bindings
        from src.monitoring.interfaces import (
            AlertServiceInterface,
            MetricsServiceInterface,
            MonitoringServiceInterface,
            PerformanceServiceInterface,
        )

        assert MonitoringServiceInterface in container._bindings
        assert MetricsServiceInterface in container._bindings
        assert AlertServiceInterface in container._bindings
        assert PerformanceServiceInterface in container._bindings

    def test_service_factory_functions(self):
        """Test that service factory functions work correctly."""
        setup_monitoring_dependencies()

        # Test factory functions create correct instances
        monitoring_service = create_monitoring_service()
        assert isinstance(monitoring_service, MonitoringService)

        metrics_service = create_metrics_service()
        assert hasattr(metrics_service, "record_counter")
        assert hasattr(metrics_service, "record_gauge")
        assert hasattr(metrics_service, "record_histogram")

        alert_service = create_alert_service()
        assert hasattr(alert_service, "create_alert")
        assert hasattr(alert_service, "resolve_alert")

        performance_service = create_performance_service()
        assert hasattr(performance_service, "get_performance_summary")


class TestMonitoringServiceIntegration:
    """Test monitoring service integration patterns."""

    def test_service_interface_compliance(self):
        """Test that services implement required interfaces correctly."""
        setup_monitoring_dependencies()

        # Create services
        metrics_service = create_metrics_service()
        alert_service = create_alert_service()
        performance_service = create_performance_service()
        monitoring_service = create_monitoring_service()

        # Test interface compliance
        assert isinstance(metrics_service, MetricsServiceInterface)
        assert isinstance(alert_service, AlertServiceInterface)
        assert isinstance(performance_service, PerformanceServiceInterface)
        assert isinstance(monitoring_service, MonitoringServiceInterface)

    def test_metrics_service_integration(self):
        """Test metrics service integration with proper request handling."""
        metrics_service = create_metrics_service()

        # Test metric recording with proper request objects
        counter_request = MetricRequest(name="test.counter", value=1, labels={"test": "value"})

        gauge_request = MetricRequest(
            name="test.gauge", value=Decimal("10.5"), labels={"test": "gauge"}
        )

        histogram_request = MetricRequest(
            name="test.histogram", value=0.25, labels={"test": "histogram"}
        )

        # These should not raise exceptions
        metrics_service.record_counter(counter_request)
        metrics_service.record_gauge(gauge_request)
        metrics_service.record_histogram(histogram_request)

        # Test metrics export
        metrics_output = metrics_service.export_metrics()
        assert isinstance(metrics_output, str)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_alert_service_integration(self):
        """Test alert service integration with proper alert handling."""
        setup_monitoring_dependencies()
        alert_service = create_alert_service()

        # Create alert request
        from src.monitoring.alerting import AlertSeverity

        alert_request = AlertRequest(
            rule_name="test_alert",
            severity=AlertSeverity.HIGH,
            message="Test alert message",
            labels={"component": "test"},
            annotations={"description": "Test alert"},
        )

        # Test alert creation
        fingerprint = await alert_service.create_alert(alert_request)
        assert isinstance(fingerprint, str)
        assert len(fingerprint) > 0

        # Test alert resolution
        resolved = await alert_service.resolve_alert(fingerprint)
        assert isinstance(resolved, bool)

        # Test alert stats
        stats = alert_service.get_alert_stats()
        assert isinstance(stats, dict)

    def test_performance_service_integration(self):
        """Test performance service integration."""
        performance_service = create_performance_service()

        # Test performance summary
        summary = performance_service.get_performance_summary()
        assert isinstance(summary, dict)

        # Test order execution recording
        performance_service.record_order_execution(
            exchange="test_exchange",
            order_type=OrderType.LIMIT,
            symbol="BTC/USD",
            latency_ms=Decimal("10.5"),
            fill_rate=Decimal("1.0"),
            slippage_bps=Decimal("0.1"),
        )

        # Test market data processing recording
        performance_service.record_market_data_processing(
            exchange="test_exchange",
            data_type="ticker",
            processing_time_ms=Decimal("2.5"),
            message_count=100,
        )

        # These should complete without errors
        assert True

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_monitoring_service_lifecycle(self):
        """Test monitoring service start/stop lifecycle."""
        monitoring_service = create_monitoring_service()

        # Test service startup
        await monitoring_service.start_monitoring()

        # Test health check
        health_status = await monitoring_service.get_health_status()
        assert isinstance(health_status, dict)
        assert "monitoring_service" in health_status
        assert "components" in health_status
        assert "timestamp" in health_status

        # Test service shutdown
        await monitoring_service.stop_monitoring()


class TestMonitoringModuleUsagePatterns:
    """Test that other modules use monitoring correctly."""

    def test_service_locator_functions(self):
        """Test service locator functions work as backward compatibility."""
        # Test that service locator functions return appropriate instances
        alert_manager = get_alert_manager()
        metrics_collector = get_metrics_collector()
        performance_profiler = get_performance_profiler()

        # These may be None if not initialized, which is acceptable
        if alert_manager:
            assert hasattr(alert_manager, "fire_alert")
            assert hasattr(alert_manager, "get_active_alerts")

        if metrics_collector:
            assert hasattr(metrics_collector, "increment_counter")
            assert hasattr(metrics_collector, "export_metrics")

        if performance_profiler:
            assert hasattr(performance_profiler, "get_performance_summary")

    def test_monitoring_initialization(self):
        """Test monitoring module initialization."""
        from src.monitoring.alerting import NotificationConfig
        from src.monitoring.telemetry import OpenTelemetryConfig

        # Test initialization with configuration
        notification_config = NotificationConfig()
        telemetry_config = OpenTelemetryConfig()

        monitoring_service = initialize_monitoring_service(
            notification_config=notification_config,
            telemetry_config=telemetry_config,
            prometheus_port=8001,
            use_dependency_injection=True,
        )

        assert isinstance(monitoring_service, MonitoringService)
        assert hasattr(monitoring_service, "alerts")
        assert hasattr(monitoring_service, "metrics")
        assert hasattr(monitoring_service, "performance")

    def test_monitoring_service_with_custom_injector(self):
        """Test monitoring service creation with custom injector."""
        from src.core.dependency_injection import DependencyInjector
        from src.monitoring.di_registration import register_monitoring_services

        # Create custom injector
        custom_injector = DependencyInjector()
        register_monitoring_services(custom_injector)

        # Initialize with custom injector
        monitoring_service = initialize_monitoring_service(
            use_dependency_injection=True, injector=custom_injector
        )

        assert isinstance(monitoring_service, MonitoringService)
        assert monitoring_service.alerts is not None
        assert monitoring_service.metrics is not None
        assert monitoring_service.performance is not None


class TestMonitoringIntegrationWithOtherModules:
    """Test integration between monitoring and other system modules."""

    def test_execution_service_integration_pattern(self):
        """Test the pattern used by execution service."""
        # This tests the correct integration pattern
        from src.monitoring.interfaces import MetricsServiceInterface
        from src.monitoring.services import MetricRequest

        # Mock execution service using monitoring correctly
        metrics_service = create_metrics_service()

        # Execution service should use service interface, not direct infrastructure
        assert isinstance(metrics_service, MetricsServiceInterface)

        # Test the pattern execution service should use
        trade_metric = MetricRequest(
            name="trading.orders.total",
            value=1,
            labels={"exchange": "test_exchange", "status": "FILLED", "symbol": "BTC/USD"},
        )

        # This should work without exceptions
        metrics_service.record_counter(trade_metric)

    def test_risk_service_integration_pattern(self):
        """Test the pattern used by risk management service."""
        from src.monitoring.interfaces import MetricsServiceInterface
        from src.monitoring.services import MetricRequest

        # Mock risk service using monitoring correctly
        metrics_service = create_metrics_service()

        # Risk service should use service interface
        assert isinstance(metrics_service, MetricsServiceInterface)

        # Test risk metrics recording pattern
        risk_metric = MetricRequest(
            name="risk.var.1d", value=Decimal("0.05"), labels={"confidence": "0.95"}
        )

        # This should work without exceptions
        metrics_service.record_gauge(risk_metric)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
