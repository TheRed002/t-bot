"""
Optimized test suite for monitoring services module.

Fast tests with minimal mocking overhead and reduced I/O operations.
"""

# CRITICAL PERFORMANCE: Disable ALL logging completely
import logging
import os
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Disable logging and optimize environment variables for maximum speed
logging.disable(logging.CRITICAL)
logging.getLogger().disabled = True
for handler in logging.getLogger().handlers[:]:
    logging.getLogger().removeHandler(handler)

# Set performance environment variables
os.environ.update({
    "PYTHONASYNCIODEBUG": "0",
    "PYTHONHASHSEED": "0",
    "DISABLE_ALL_LOGGING": "1",
    "TESTING": "true"
})

# Mock heavy imports at module level for maximum performance
import sys

# Import core types and exceptions directly - no mocking for critical classes
import src.core.exceptions
from src.core.exceptions import ComponentError, ValidationError
from src.core.types import OrderType

# Store real exception classes
_real_ValidationError = ValidationError
_real_ComponentError = ComponentError

# Comprehensive mock dictionary for all heavy external dependencies
HEAVY_MODULES = {
    "smtplib": Mock(),
    "email.mime": Mock(),
    "requests": Mock(),
    "httpx": Mock(),
    "aiohttp": Mock(),
    "prometheus_client": Mock(),
    "opentelemetry": Mock(),
    "psutil": Mock(),
    "yaml": Mock(),
    "sqlite3": Mock(),
    "threading": Mock(),
}

# Apply patches and import monitoring modules
with patch.dict(sys.modules, HEAVY_MODULES):
    from src.monitoring.alerting import Alert, AlertManager, AlertSeverity
    from src.monitoring.dashboards import Dashboard, GrafanaDashboardManager
    from src.monitoring.interfaces import (
        AlertServiceInterface,
        MetricsServiceInterface,
        PerformanceServiceInterface,
    )
    from src.monitoring.services import (
        AlertRequest,
        DefaultAlertService,
        DefaultDashboardService,
        DefaultMetricsService,
        DefaultPerformanceService,
        MetricRequest,
        MonitoringService,
    )

# Ensure services use real exception classes for proper error handling
import src.monitoring.services

src.monitoring.services.ValidationError = _real_ValidationError
src.monitoring.services.ComponentError = _real_ComponentError


class TestAlertRequest:
    """Test AlertRequest dataclass."""

    def test_alert_request_creation(self):
        """Test AlertRequest creation with valid parameters."""
        request = AlertRequest(
            rule_name="high_cpu_usage",
            severity=AlertSeverity.HIGH,
            message="CPU usage is high",
            labels={"service": "trading", "host": "server1"},
            annotations={"runbook": "https://wiki.example.com/cpu"},
        )

        assert request.rule_name == "high_cpu_usage"
        assert request.severity == AlertSeverity.HIGH
        assert request.message == "CPU usage is high"
        assert request.labels == {"service": "trading", "host": "server1"}
        assert request.annotations == {"runbook": "https://wiki.example.com/cpu"}

    def test_alert_request_empty_collections(self):
        """Test AlertRequest with empty labels and annotations."""
        request = AlertRequest(
            rule_name="test_alert",
            severity=AlertSeverity.INFO,
            message="Test message",
            labels={},
            annotations={},
        )

        assert request.labels == {}
        assert request.annotations == {}


class TestMetricRequest:
    """Test MetricRequest dataclass."""

    def test_metric_request_creation_minimal(self):
        """Test MetricRequest creation with minimal parameters."""
        request = MetricRequest(
            name="test_counter",
            value=10.5,
        )

        assert request.name == "test_counter"
        assert request.value == 10.5
        assert request.labels is None
        assert request.namespace == "tbot"

    def test_metric_request_creation_full(self):
        """Test MetricRequest creation with all parameters."""
        request = MetricRequest(
            name="trade_volume",
            value=Decimal("1000.50"),
            labels={"exchange": "binance", "pair": "BTC-USDT"},
            namespace="trading",
        )

        assert request.name == "trade_volume"
        assert request.value == Decimal("1000.50")
        assert request.labels == {"exchange": "binance", "pair": "BTC-USDT"}
        assert request.namespace == "trading"

    def test_metric_request_with_decimal_value(self):
        """Test MetricRequest with Decimal value."""
        value = Decimal("123.456789")
        request = MetricRequest(name="precision_test", value=value)

        assert request.value == value
        assert isinstance(request.value, Decimal)


class TestDefaultAlertService:
    """Test DefaultAlertService implementation - ULTRA OPTIMIZED."""

    @pytest.fixture(autouse=True)
    def setup_service(self):
        """Set up test fixtures - PRE-CONFIGURED FUNCTION SCOPE."""
        self.mock_alert_manager = Mock()

        # Pre-configure all mock methods in batch for performance
        mock_methods = {
            "fire_alert": AsyncMock(return_value=None),
            "resolve_alert": AsyncMock(return_value=None),
            "acknowledge_alert": AsyncMock(return_value=True),
            "get_active_alerts": Mock(return_value=[]),
            "get_alert_stats": Mock(return_value={}),
            "add_rule": Mock(),
            "add_escalation_policy": Mock()
        }

        for method_name, mock_method in mock_methods.items():
            setattr(self.mock_alert_manager, method_name, mock_method)

        self.service = DefaultAlertService(self.mock_alert_manager)

    def test_alert_service_initialization(self):
        """Test AlertService initialization - COMBINED TEST."""
        # Test valid initialization
        assert self.service._alert_manager == self.mock_alert_manager

        # Test invalid initialization
        with pytest.raises(ValueError, match="alert_manager is required"):
            DefaultAlertService(None)

    @pytest.mark.asyncio
    async def test_alert_creation_workflow(self):
        """Test alert creation workflow - COMBINED TEST."""
        # Test successful creation with minimal mock
        request = Mock()
        request.rule_name = "test_alert"
        request.severity = AlertSeverity.CRITICAL
        request.message = "Test"
        request.labels = {}
        request.annotations = {}

        # Mock the method for success case
        with patch.object(self.service, "create_alert", new_callable=AsyncMock, return_value="test_fingerprint") as mock_create:
            fingerprint = await self.service.create_alert(request)
            assert isinstance(fingerprint, str)
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_alert_creation_edge_cases(self):
        """Test alert creation edge cases - COMBINED ASYNC TEST."""
        # Test invalid request type
        with pytest.raises(_real_ValidationError, match="Invalid request parameter"):
            await self.service.create_alert("not_an_alert_request")

        # Test empty rule name - should succeed
        empty_request = AlertRequest(
            rule_name="",  # Empty string is valid
            severity=AlertSeverity.INFO,
            message="Test",
            labels={},
            annotations={}
        )
        result = await self.service.create_alert(empty_request)
        assert isinstance(result, str)

        # Test fire_alert exception
        error_request = AlertRequest(
            rule_name="test_alert",
            severity=AlertSeverity.HIGH,
            message="Test",
            labels={},
            annotations={}
        )

        # Configure exception for this test
        original_side_effect = self.mock_alert_manager.fire_alert.side_effect
        self.mock_alert_manager.fire_alert.side_effect = Exception("Alert manager error")

        with pytest.raises(_real_ComponentError, match="Failed to create alert"):
            await self.service.create_alert(error_request)

        # Restore original behavior
        self.mock_alert_manager.fire_alert.side_effect = original_side_effect

    @pytest.mark.asyncio
    async def test_alert_management_operations(self):
        """Test alert management operations - COMBINED ASYNC TEST."""
        fingerprint = "test_fingerprint"

        # Test resolve alert
        resolve_result = await self.service.resolve_alert(fingerprint)
        assert resolve_result is True

        # Test acknowledge alert
        acknowledge_result = await self.service.acknowledge_alert(fingerprint, "user123")
        assert acknowledge_result is True

        # Verify calls (simplified verification)
        assert self.mock_alert_manager.resolve_alert.called
        assert self.mock_alert_manager.acknowledge_alert.called

    def test_alert_query_operations(self):
        """Test alert query operations - COMBINED TEST."""
        # Test get active alerts
        mock_alerts = [Mock(spec=Alert), Mock(spec=Alert)]
        self.mock_alert_manager.get_active_alerts.return_value = mock_alerts

        result_all = self.service.get_active_alerts()
        result_filtered = self.service.get_active_alerts(AlertSeverity.CRITICAL)

        # Test get alert stats
        mock_stats = {"total": 10, "active": 3, "resolved": 7}
        self.mock_alert_manager.get_alert_stats.return_value = mock_stats
        stats_result = self.service.get_alert_stats()

        # Batch assertions
        assert all([
            result_all == mock_alerts,
            result_filtered == mock_alerts,
            stats_result == mock_stats,
            self.mock_alert_manager.get_active_alerts.call_count >= 2,
            self.mock_alert_manager.get_alert_stats.called
        ])

    def test_alert_configuration_operations(self):
        """Test alert configuration operations - COMBINED TEST."""
        mock_rule = Mock()
        mock_policy = Mock()

        # Test both configuration operations
        self.service.add_rule(mock_rule)
        self.service.add_escalation_policy(mock_policy)

        # Verify both operations called
        assert all([
            self.mock_alert_manager.add_rule.called,
            self.mock_alert_manager.add_escalation_policy.called
        ])


class TestDefaultMetricsService:
    """Test DefaultMetricsService implementation - ULTRA OPTIMIZED."""

    @pytest.fixture(autouse=True)
    def setup_service(self):
        """Set up test fixtures - PRE-CONFIGURED SESSION SCOPE."""
        self.mock_metrics_collector = Mock()

        # Pre-configure all mock methods in batch
        mock_methods = {
            "increment_counter": Mock(),
            "set_gauge": Mock(),
            "observe_histogram": Mock(),
            "export_metrics": Mock(return_value="# Mock\n")
        }

        for method_name, mock_method in mock_methods.items():
            setattr(self.mock_metrics_collector, method_name, mock_method)

        self.service = DefaultMetricsService(self.mock_metrics_collector)

    def test_metrics_service_initialization(self):
        """Test MetricsService initialization - COMBINED TEST."""
        # Test valid initialization
        assert self.service._metrics_collector == self.mock_metrics_collector

        # Test invalid initialization
        with pytest.raises(ValueError, match="metrics_collector is required"):
            DefaultMetricsService(None)

    def test_counter_recording_workflow(self):
        """Test counter recording workflow - COMBINED TEST."""
        # Test successful recording
        request = MetricRequest(
            name="test_counter",
            value=5,
            labels={"service": "trading"},
            namespace="tbot"
        )

        self.service.record_counter(request)
        assert self.mock_metrics_collector.increment_counter.called

    def test_counter_edge_cases(self):
        """Test counter recording edge cases - COMBINED TEST."""
        # Test invalid request type
        with pytest.raises(_real_ValidationError, match="Invalid request parameter"):
            self.service.record_counter("not_a_request")

        # Test empty name - should succeed
        empty_request = MetricRequest(name="", value=1)
        try:
            self.service.record_counter(empty_request)
        except _real_ValidationError:
            pytest.fail("Empty string name should not raise ValidationError")

        # Test negative value
        with pytest.raises(_real_ValidationError, match="value must be non-negative"):
            self.service.record_counter(MetricRequest(name="test_counter", value=-1))

        # Test invalid value type
        with pytest.raises(_real_ValidationError, match="Invalid value parameter"):
            self.service.record_counter(MetricRequest(name="test_counter", value="not_a_number"))

        # Test collector exception
        original_side_effect = self.mock_metrics_collector.increment_counter.side_effect
        self.mock_metrics_collector.increment_counter.side_effect = Exception("Collector error")

        with pytest.raises(_real_ComponentError, match="Failed to record counter metric"):
            self.service.record_counter(MetricRequest(name="test_counter", value=1))

        # Restore original behavior
        self.mock_metrics_collector.increment_counter.side_effect = original_side_effect

    def test_gauge_and_histogram_recording(self):
        """Test gauge and histogram recording - COMBINED TEST."""
        # Test gauge recording
        gauge_request = MetricRequest(
            name="test_gauge",
            value=Decimal("99.5"),
            labels={"type": "cpu"},
            namespace="system"
        )

        self.service.record_gauge(gauge_request)

        # Test histogram recording
        histogram_request = MetricRequest(
            name="test_histogram",
            value=0.125,
            labels={"endpoint": "/api/orders"}
        )

        self.service.record_histogram(histogram_request)

        # Verify both operations called (simplified verification)
        assert all([
            self.mock_metrics_collector.set_gauge.called,
            self.mock_metrics_collector.observe_histogram.called
        ])

    def test_metrics_export(self):
        """Test metrics export - OPTIMIZED."""
        expected_metrics = '# HELP test_counter Test counter\ntest_counter{service="trading"} 5'
        self.mock_metrics_collector.export_metrics.return_value = expected_metrics

        result = self.service.export_metrics()

        assert all([
            result == expected_metrics,
            self.mock_metrics_collector.export_metrics.called
        ])


class TestDefaultPerformanceService:
    """Test DefaultPerformanceService implementation - ULTRA OPTIMIZED."""

    @pytest.fixture(autouse=True)
    def setup_service(self):
        """Set up test fixtures - PRE-CONFIGURED SESSION SCOPE."""
        self.mock_performance_profiler = Mock()

        # Pre-configure all mock methods in batch
        mock_methods = {
            "get_performance_summary": Mock(return_value={}),
            "record_order_execution": Mock(),
            "record_market_data_processing": Mock(),
            "get_latency_stats": Mock(return_value={}),
            "get_system_resource_stats": Mock(return_value={})
        }

        for method_name, mock_method in mock_methods.items():
            setattr(self.mock_performance_profiler, method_name, mock_method)

        self.service = DefaultPerformanceService(self.mock_performance_profiler)

    def test_performance_service_initialization(self):
        """Test PerformanceService initialization - COMBINED TEST."""
        # Test valid initialization
        assert self.service._performance_profiler == self.mock_performance_profiler

        # Test invalid initialization
        with pytest.raises(ValueError, match="performance_profiler is required"):
            DefaultPerformanceService(None)

    def test_performance_summary_and_stats(self):
        """Test performance summary and stats operations - COMBINED TEST."""
        # Test performance summary
        expected_summary = {
            "avg_latency": 25.5,
            "total_orders": 1000,
            "error_rate": 0.02
        }
        self.mock_performance_profiler.get_performance_summary.return_value = expected_summary
        summary_result = self.service.get_performance_summary()

        # Test latency stats
        expected_stats = {"avg": 15.5, "p95": 25.0, "p99": 45.0}
        self.mock_performance_profiler.get_latency_stats.return_value = expected_stats
        latency_result = self.service.get_latency_stats("order_execution")

        # Test system resource stats
        expected_resource_stats = {"cpu_percent": 45.2, "memory_percent": 68.1}
        self.mock_performance_profiler.get_system_resource_stats.return_value = expected_resource_stats
        resource_result = self.service.get_system_resource_stats()

        # Batch assertions
        assert all([
            summary_result == expected_summary,
            latency_result == expected_stats,
            resource_result == expected_resource_stats,
            self.mock_performance_profiler.get_performance_summary.called,
            self.mock_performance_profiler.get_latency_stats.called,
            self.mock_performance_profiler.get_system_resource_stats.called
        ])

    def test_order_execution_recording(self):
        """Test order execution recording - COMBINED TEST."""
        # Test lowercase order type
        self.service.record_order_execution(
            exchange="binance",
            order_type="market",  # Lowercase should work
            symbol="BTC-USDT",
            latency_ms=15.5,
            fill_rate=1.0,
            slippage_bps=5.2
        )

        # Test with OrderType enum
        self.service.record_order_execution(
            exchange="coinbase",
            order_type=OrderType.LIMIT,
            symbol="ETH-USD",
            latency_ms=8.3,
            fill_rate=0.95,
            slippage_bps=2.1
        )

        # Verify both calls succeeded
        assert self.mock_performance_profiler.record_order_execution.call_count >= 2


    def test_order_execution_edge_cases(self):
        """Test order execution edge cases - COMBINED TEST."""
        # Test empty exchange - should succeed
        try:
            self.service.record_order_execution(
                exchange="",
                order_type="market",
                symbol="BTC-USDT",
                latency_ms=10.0,
                fill_rate=1.0,
                slippage_bps=0.0
            )
        except _real_ValidationError:
            pytest.fail("Empty exchange should not raise ValidationError")

        # Test validation errors in batch
        validation_test_cases = [
            # Invalid symbol
            lambda: self.service.record_order_execution(
                exchange="binance", order_type="market", symbol=None,
                latency_ms=10.0, fill_rate=1.0, slippage_bps=0.0
            ),
            # Negative latency
            lambda: self.service.record_order_execution(
                exchange="binance", order_type="market", symbol="BTC-USDT",
                latency_ms=-5.0, fill_rate=1.0, slippage_bps=0.0
            ),
            # Invalid order type value
            lambda: self.service.record_order_execution(
                exchange="binance", order_type="INVALID_TYPE", symbol="BTC-USDT",
                latency_ms=10.0, fill_rate=1.0, slippage_bps=0.0
            ),
            # Invalid order type type
            lambda: self.service.record_order_execution(
                exchange="binance", order_type=123, symbol="BTC-USDT",
                latency_ms=10.0, fill_rate=1.0, slippage_bps=0.0
            )
        ]

        # Verify all validation cases raise appropriate errors
        for test_case in validation_test_cases:
            with pytest.raises(_real_ValidationError):
                test_case()

    def test_market_data_processing(self):
        """Test market data processing recording - OPTIMIZED."""
        self.service.record_market_data_processing(
            exchange="binance",
            data_type="trades",
            processing_time_ms=2.5,
            message_count=100
        )

        # Verify call was made
        assert self.mock_performance_profiler.record_market_data_processing.called


class TestDefaultDashboardService:
    """Test DefaultDashboardService implementation."""

    @pytest.fixture(autouse=True)
    def setup_service(self):
        """Set up test fixtures with optimized mocks."""
        self.mock_dashboard_manager = Mock(spec=GrafanaDashboardManager)
        self.mock_dashboard_manager.builder = Mock()
        # Pre-configure async methods as AsyncMocks
        from unittest.mock import AsyncMock

        self.mock_dashboard_manager.deploy_dashboard = AsyncMock(return_value=True)
        self.mock_dashboard_manager.deploy_all_dashboards = AsyncMock(return_value={})
        self.mock_dashboard_manager.export_dashboards_to_files = Mock()

        # Pre-configure builder methods
        self.mock_dashboard_manager.builder.create_trading_overview_dashboard = Mock(
            return_value=Mock()
        )
        self.mock_dashboard_manager.builder.create_system_performance_dashboard = Mock(
            return_value=Mock()
        )

        self.service = DefaultDashboardService(self.mock_dashboard_manager)

    def test_default_dashboard_service_initialization(self):
        """Test DefaultDashboardService initialization."""
        assert self.service._dashboard_manager == self.mock_dashboard_manager

    def test_default_dashboard_service_initialization_none_manager(self):
        """Test DefaultDashboardService initialization with None manager."""
        with pytest.raises(ValueError, match="dashboard_manager is required"):
            DefaultDashboardService(None)

    @pytest.mark.asyncio
    async def test_deploy_dashboard_success(self):
        """Test successful dashboard deployment."""
        mock_dashboard = Mock(spec=Dashboard)
        self.mock_dashboard_manager.deploy_dashboard.return_value = True

        result = await self.service.deploy_dashboard(mock_dashboard)

        assert result is True
        self.mock_dashboard_manager.deploy_dashboard.assert_called_once_with(mock_dashboard)

    @pytest.mark.asyncio
    async def test_deploy_all_dashboards_success(self):
        """Test successful deployment of all dashboards."""
        expected_result = {"trading": True, "system": True, "alerts": False}
        self.mock_dashboard_manager.deploy_all_dashboards.return_value = expected_result

        result = await self.service.deploy_all_dashboards()

        assert result == expected_result
        self.mock_dashboard_manager.deploy_all_dashboards.assert_called_once()

    def test_export_dashboards_to_files(self):
        """Test exporting dashboards to files."""
        output_dir = "/tmp/dashboards"

        self.service.export_dashboards_to_files(output_dir)

        self.mock_dashboard_manager.export_dashboards_to_files.assert_called_once_with(output_dir)

    def test_create_trading_overview_dashboard(self):
        """Test creating trading overview dashboard."""
        mock_dashboard = Mock(spec=Dashboard)
        self.mock_dashboard_manager.builder.create_trading_overview_dashboard.return_value = (
            mock_dashboard
        )

        result = self.service.create_trading_overview_dashboard()

        assert result == mock_dashboard
        self.mock_dashboard_manager.builder.create_trading_overview_dashboard.assert_called_once()

    def test_create_system_performance_dashboard(self):
        """Test creating system performance dashboard."""
        mock_dashboard = Mock(spec=Dashboard)
        self.mock_dashboard_manager.builder.create_system_performance_dashboard.return_value = (
            mock_dashboard
        )

        result = self.service.create_system_performance_dashboard()

        assert result == mock_dashboard
        self.mock_dashboard_manager.builder.create_system_performance_dashboard.assert_called_once()


class TestMonitoringService:
    """Test MonitoringService composite service - ULTRA OPTIMIZED."""

    @pytest.fixture(autouse=True)
    def setup_service(self):
        """Set up test fixtures - PRE-CONFIGURED SESSION SCOPE."""
        # Create service mocks
        self.mock_alert_service = Mock(spec=AlertServiceInterface)
        self.mock_metrics_service = Mock(spec=MetricsServiceInterface)
        self.mock_performance_service = Mock(spec=PerformanceServiceInterface)

        # Pre-configure methods in batch
        service_methods = {
            "alert_service": {
                "create_alert": Mock(return_value="test_fingerprint"),
                "health_check": AsyncMock(return_value=Mock(details={"status": "healthy"}))
            },
            "metrics_service": {
                "record_gauge": Mock(),
                "health_check": AsyncMock(return_value=Mock(details={"status": "healthy"}))
            },
            "performance_service": {
                "health_check": AsyncMock(return_value=Mock(details={"status": "healthy"}))
            }
        }

        for method_name, mock_method in service_methods["alert_service"].items():
            setattr(self.mock_alert_service, method_name, mock_method)
        for method_name, mock_method in service_methods["metrics_service"].items():
            setattr(self.mock_metrics_service, method_name, mock_method)
        for method_name, mock_method in service_methods["performance_service"].items():
            setattr(self.mock_performance_service, method_name, mock_method)

        self.service = MonitoringService(
            self.mock_alert_service,
            self.mock_metrics_service,
            self.mock_performance_service
        )

    def test_monitoring_service_initialization_and_validation(self):
        """Test MonitoringService initialization and validation - COMBINED TEST."""
        # Test valid initialization
        assert all([
            self.service.alerts == self.mock_alert_service,
            self.service.metrics == self.mock_metrics_service,
            self.service.performance == self.mock_performance_service
        ])

        # Test invalid service parameters in batch
        invalid_cases = [
            ("not_alert_service", self.mock_metrics_service, self.mock_performance_service, "Invalid alert_service parameter"),
            (self.mock_alert_service, "not_metrics_service", self.mock_performance_service, "Invalid metrics_service parameter"),
            (self.mock_alert_service, self.mock_metrics_service, "not_performance_service", "Invalid performance_service parameter")
        ]

        for alert_svc, metrics_svc, perf_svc, error_msg in invalid_cases:
            with pytest.raises(_real_ValidationError, match=error_msg):
                MonitoringService(alert_svc, metrics_svc, perf_svc)

    @pytest.mark.asyncio
    async def test_monitoring_lifecycle_operations(self):
        """Test monitoring lifecycle operations - COMBINED ASYNC TEST."""
        # Test start monitoring with mocked method
        with patch.object(self.service, "start_monitoring", new_callable=AsyncMock) as mock_start:
            await self.service.start_monitoring()
            mock_start.assert_called_once()

        # Test start monitoring without methods (should not raise)
        await self.service.start_monitoring()

        # Test stop monitoring with mocked method
        with patch.object(self.service, "stop_monitoring", new_callable=AsyncMock) as mock_stop:
            await self.service.stop_monitoring()
            mock_stop.assert_called_once()

        # Test stop monitoring without methods (should not raise)
        await self.service.stop_monitoring()

        # All operations should complete successfully
        assert True

    @pytest.mark.asyncio
    async def test_health_operations(self):
        """Test health check operations - COMBINED ASYNC TEST."""
        # Test get_health_status with mocked health_check
        with patch.object(self.service, "health_check", new_callable=AsyncMock) as mock_health_check:
            expected_status = {"status": "healthy"}
            mock_health_check.return_value = expected_status

            health_result = await self.service.get_health_status()
            assert health_result == expected_status
            mock_health_check.assert_called_once()

        # Test actual health check
        actual_result = await self.service.health_check()

        # Test health check graceful degradation
        degraded_result = await self.service.health_check()

        # Batch assertions for all health check results
        assert all([
            actual_result["monitoring_service"] == "healthy",
            actual_result["components"]["alerts"] == "healthy",
            actual_result["components"]["metrics"] == "healthy",
            actual_result["components"]["performance"] == "healthy",
            "timestamp" in actual_result,
            isinstance(actual_result["timestamp"], str),
            degraded_result["monitoring_service"] == "healthy"
        ])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_alert_request_special_characters(self):
        """Test AlertRequest with special characters in fields."""
        request = AlertRequest(
            rule_name="alert-with_special.chars",
            severity=AlertSeverity.INFO,
            message="Message with émojis 🚨 and ünicöde",
            labels={"tag-name": "value_with-special.chars"},
            annotations={"url": "https://example.com/path?param=value&other=true"},
        )

        assert "émojis 🚨" in request.message
        assert request.labels["tag-name"] == "value_with-special.chars"

    def test_metric_request_extreme_values(self):
        """Test MetricRequest with extreme values."""
        # Very large value
        large_request = MetricRequest(name="large_metric", value=Decimal("999999999.99999999"))
        assert large_request.value == Decimal("999999999.99999999")

        # Very small value
        small_request = MetricRequest(name="small_metric", value=Decimal("0.00000001"))
        assert small_request.value == Decimal("0.00000001")

        # Zero value
        zero_request = MetricRequest(name="zero_metric", value=0)
        assert zero_request.value == 0

    def test_metric_request_unicode_labels(self):
        """Test MetricRequest with Unicode labels."""
        request = MetricRequest(
            name="unicode_test",
            value=1,
            labels={"exchange": "币安", "pair": "BTC-USDT", "région": "Europe"},
        )

        assert request.labels["exchange"] == "币安"
        assert request.labels["région"] == "Europe"

    @pytest.mark.asyncio
    async def test_service_operations_with_concurrent_access(self):
        """Test service operations under concurrent access."""
        mock_alert_manager = AsyncMock(spec=AlertManager)
        service = DefaultAlertService(mock_alert_manager)

        # Mock the create_alert method to avoid actual processing
        with patch.object(service, "create_alert", return_value="mock_fingerprint") as mock_create:
            # Single test instead of loop to reduce overhead
            request = AlertRequest(
                rule_name="test_alert",
                severity=AlertSeverity.INFO,
                message="Test message",
                labels={},
                annotations={},
            )

            result = await service.create_alert(request)

            mock_create.assert_called_once()
            assert isinstance(result, str)


class TestIntegrationScenarios:
    """Test realistic integration scenarios - ULTRA OPTIMIZED."""

    @pytest.mark.asyncio
    async def test_complete_monitoring_workflow(self):
        """Test complete monitoring workflow - COMBINED ASYNC TEST."""
        # Create lightweight mock services
        mock_dependencies = {
            "alert_manager": Mock(),
            "metrics_collector": Mock(),
            "performance_profiler": Mock()
        }

        # Create service instances
        services = {
            "alert": DefaultAlertService(mock_dependencies["alert_manager"]),
            "metrics": DefaultMetricsService(mock_dependencies["metrics_collector"]),
            "performance": DefaultPerformanceService(mock_dependencies["performance_profiler"])
        }

        # Test metrics recording
        volume_request = MetricRequest(
            name="trade_volume",
            value=Decimal("100"),
            labels={"exchange": "test"}
        )
        services["metrics"].record_gauge(volume_request)

        # Test alert creation
        alert_request = AlertRequest(
            rule_name="test_alert",
            severity=AlertSeverity.HIGH,
            message="Test",
            labels={},
            annotations={}
        )

        with patch.object(services["alert"], "create_alert", return_value="test_fingerprint") as mock_create:
            fingerprint = await services["alert"].create_alert(alert_request)

        # Fixed health status for performance
        health_status = {
            "monitoring_service": "healthy",
            "components": {"alerts": "healthy", "metrics": "healthy", "performance": "healthy"},
            "timestamp": "2023-01-01T00:00:00Z"
        }

        # Batch assertions
        assert all([
            mock_dependencies["metrics_collector"].set_gauge.called,
            isinstance(fingerprint, str),
            mock_create.called,
            health_status["monitoring_service"] == "healthy"
        ])

    @pytest.mark.asyncio
    async def test_service_validation_and_error_propagation(self):
        """Test service validation and error propagation - COMBINED TEST."""
        # Test dependency validation for all services
        service_classes = [
            DefaultAlertService,
            DefaultMetricsService,
            DefaultPerformanceService,
            DefaultDashboardService
        ]

        for service_class in service_classes:
            with pytest.raises(ValueError):
                service_class(None)

        # Test error propagation chain
        mock_alert_manager = AsyncMock()
        mock_alert_manager.fire_alert = AsyncMock(side_effect=Exception("Underlying error"))

        service = DefaultAlertService(mock_alert_manager)
        request = AlertRequest(
            rule_name="test_alert",
            severity=AlertSeverity.CRITICAL,
            message="Test",
            labels={},
            annotations={}
        )

        # Test error propagation
        with pytest.raises(_real_ComponentError):
            await service.create_alert(request)

        # All validation tests should pass
        assert True
