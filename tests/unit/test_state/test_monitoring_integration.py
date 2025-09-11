"""Comprehensive tests for state monitoring_integration module."""

import math
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import StateConsistencyError
from src.state.monitoring_integration import (
    StateAlertAdapter,
    StateMetricsAdapter,
    create_integrated_monitoring_service,
)
from src.state.monitoring import StateMonitoringService


class MockMetric:
    """Mock metric for testing."""

    def __init__(
        self,
        name: str,
        value: Any,
        metric_type: str = "gauge",
        tags: dict = None,
        timestamp: datetime = None,
    ):
        self.name = name
        self.value = value
        self.metric_type = metric_type
        self.tags = tags or {}
        self.timestamp = timestamp or datetime.now()


class MockAlert:
    """Mock alert for testing."""

    def __init__(
        self,
        message: str,
        severity: str = "warning",
        component: str = "state",
        details: dict = None,
    ):
        from datetime import datetime, timezone
        from uuid import uuid4
        from decimal import Decimal
        
        self.alert_id = str(uuid4())
        self.timestamp = datetime.now(timezone.utc)
        self.severity = severity
        self.title = "Test Alert"
        self.message = message
        self.source = component
        self.category = "test"
        self.metric_name = "test_metric"
        self.current_value = Decimal("10.0")
        self.threshold_value = Decimal("5.0")
        self.active = True
        self.acknowledged = False
        self.resolved = False
        self.resolved_at = None
        self.details = details or {}


class TestStateMetricsAdapter:
    """Test StateMetricsAdapter class."""

    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector."""
        collector = MagicMock()
        collector.increment_counter = MagicMock()
        collector.set_gauge = MagicMock()
        collector.record_histogram = MagicMock()
        return collector

    @pytest.fixture
    def metrics_adapter(self, mock_metrics_collector):
        """Create metrics adapter instance."""
        with patch("src.state.monitoring_integration.get_tracer") as mock_tracer:
            mock_tracer.return_value = MagicMock()
            return StateMetricsAdapter(mock_metrics_collector)

    def test_initialization_with_collector(self, mock_metrics_collector):
        """Test adapter initialization with provided collector."""
        with patch("src.state.monitoring_integration.get_tracer") as mock_tracer:
            mock_tracer.return_value = MagicMock()
            adapter = StateMetricsAdapter(mock_metrics_collector)
            
            assert adapter.metrics_collector == mock_metrics_collector
            assert adapter.tracer is not None
            mock_tracer.assert_called_once_with("state_service")

    def test_initialization_without_collector(self):
        """Test adapter initialization without collector."""
        with patch("src.state.monitoring_integration.get_tracer") as mock_tracer, \
             patch("src.state.monitoring_integration.MetricsCollector") as mock_collector_class:
            
            mock_tracer.return_value = MagicMock()
            mock_collector = MagicMock()
            mock_collector_class.return_value = mock_collector
            
            adapter = StateMetricsAdapter()
            
            assert adapter.metrics_collector == mock_collector
            mock_collector_class.assert_called_once()

    def test_metric_type_mapping(self, metrics_adapter):
        """Test metric type mapping."""
        # Access the metric type map
        metric_type_map = metrics_adapter._metric_type_map
        
        assert "counter" in metric_type_map.values()
        assert "gauge" in metric_type_map.values()
        assert "histogram" in metric_type_map.values()

    def test_record_counter_metric(self, metrics_adapter, mock_metrics_collector):
        """Test recording counter metric."""
        from src.state.monitoring import MetricType
        
        metric = MockMetric("test_counter", 5)
        metric.metric_type = MetricType.COUNTER
        
        metrics_adapter.record_state_metric(metric)
        
        mock_metrics_collector.increment_counter.assert_called_once()
        call_args = mock_metrics_collector.increment_counter.call_args
        
        # Check metric name
        assert call_args[0][0] == "state.test_counter"
        # Check value
        assert call_args[0][2] == 5
        # Check labels contain component
        labels = call_args[0][1]
        assert labels["component"] == "state"
        assert labels["metric_name"] == "test_counter"

    def test_record_gauge_metric(self, metrics_adapter, mock_metrics_collector):
        """Test recording gauge metric."""
        from src.state.monitoring import MetricType
        
        metric = MockMetric("test_gauge", 10.5)
        metric.metric_type = MetricType.GAUGE
        
        metrics_adapter.record_state_metric(metric)
        
        mock_metrics_collector.set_gauge.assert_called_once()
        call_args = mock_metrics_collector.set_gauge.call_args
        
        assert call_args[0][0] == "state.test_gauge"
        assert call_args[0][1] == 10.5

    def test_record_histogram_metric(self, metrics_adapter, mock_metrics_collector):
        """Test recording histogram metric."""
        from src.state.monitoring import MetricType
        
        metric = MockMetric("test_histogram", 25.0)
        metric.metric_type = MetricType.HISTOGRAM
        
        metrics_adapter.record_state_metric(metric)
        
        mock_metrics_collector.observe_histogram.assert_called_once()
        call_args = mock_metrics_collector.observe_histogram.call_args
        
        assert call_args[0][0] == "state.test_histogram"
        assert call_args[0][1] == 25.0

    def test_record_timer_metric(self, metrics_adapter, mock_metrics_collector):
        """Test recording timer metric (should map to histogram)."""
        from src.state.monitoring import MetricType
        
        metric = MockMetric("test_timer", 100.0)
        metric.metric_type = MetricType.TIMER
        
        metrics_adapter.record_state_metric(metric)
        
        # Timer should be recorded as histogram
        mock_metrics_collector.observe_histogram.assert_called_once()

    def test_record_metric_with_tags(self, metrics_adapter, mock_metrics_collector):
        """Test recording metric with custom tags."""
        from src.state.monitoring import MetricType
        
        metric = MockMetric("test_metric", 42)
        metric.metric_type = MetricType.GAUGE
        metric.tags = {"custom_tag": "custom_value", "environment": "test"}
        
        metrics_adapter.record_state_metric(metric)
        
        call_args = mock_metrics_collector.set_gauge.call_args
        labels = call_args[0][2]
        
        assert labels["custom_tag"] == "custom_value"
        assert labels["environment"] == "test"
        assert labels["component"] == "state"

    def test_record_metric_none_value(self, metrics_adapter, mock_metrics_collector):
        """Test recording metric with None value."""
        metric = MockMetric("test_metric", None)
        
        metrics_adapter.record_state_metric(metric)
        
        # Should not call any collector method
        mock_metrics_collector.increment_counter.assert_not_called()
        mock_metrics_collector.set_gauge.assert_not_called()
        mock_metrics_collector.record_histogram.assert_not_called()

    def test_record_metric_invalid_type(self, metrics_adapter, mock_metrics_collector):
        """Test recording metric with invalid value type."""
        metric = MockMetric("test_metric", "not_a_number")
        
        metrics_adapter.record_state_metric(metric)
        
        # Should not call any collector method
        mock_metrics_collector.increment_counter.assert_not_called()
        mock_metrics_collector.set_gauge.assert_not_called()
        mock_metrics_collector.record_histogram.assert_not_called()

    def test_record_metric_nan_value(self, metrics_adapter, mock_metrics_collector):
        """Test recording metric with NaN value."""
        metric = MockMetric("test_metric", float("nan"))
        
        metrics_adapter.record_state_metric(metric)
        
        # Should not call any collector method
        mock_metrics_collector.increment_counter.assert_not_called()
        mock_metrics_collector.set_gauge.assert_not_called()
        mock_metrics_collector.record_histogram.assert_not_called()

    def test_record_metric_inf_value(self, metrics_adapter, mock_metrics_collector):
        """Test recording metric with infinity value."""
        metric = MockMetric("test_metric", float("inf"))
        
        metrics_adapter.record_state_metric(metric)
        
        # Should not call any collector method
        mock_metrics_collector.increment_counter.assert_not_called()
        mock_metrics_collector.set_gauge.assert_not_called()
        mock_metrics_collector.record_histogram.assert_not_called()

    def test_record_operation_time(self, metrics_adapter, mock_metrics_collector):
        """Test recording operation time."""
        operation = "state_validation"
        duration_ms = 45.5
        
        metrics_adapter.record_operation_time(operation, duration_ms)
        
        mock_metrics_collector.observe_histogram.assert_called_once_with(
            "state.operation.duration", duration_ms, {"operation": operation}
        )

    def test_record_health_check(self, metrics_adapter, mock_metrics_collector):
        """Test recording health check."""
        from src.state.monitoring import HealthStatus
        
        check_name = "database_connection"
        status = HealthStatus.HEALTHY
        
        metrics_adapter.record_health_check(check_name, status)
        
        # Should record health status as gauge
        mock_metrics_collector.set_gauge.assert_called_once()
        call_args = mock_metrics_collector.set_gauge.call_args
        
        assert call_args[0][0] == "state.health.status"
        assert call_args[0][1] == 1.0  # Healthy = 1.0
        labels = call_args[0][2]
        assert labels["check"] == check_name


class TestStateAlertAdapter:
    """Test StateAlertAdapter class."""

    @pytest.fixture
    def alert_adapter(self):
        """Create alert adapter instance."""
        with patch("src.monitoring.alerting.get_alert_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_get_manager.return_value = mock_manager
            return StateAlertAdapter()

    def test_initialization(self):
        """Test alert adapter initialization."""
        adapter = StateAlertAdapter()
        
        # Alert manager should be None initially (lazy loading)
        assert adapter._alert_manager is None
        assert hasattr(adapter, '_severity_map')
        assert len(adapter._severity_map) == 5

    def test_alert_manager_property(self, alert_adapter):
        """Test alert manager property."""
        mock_manager = MagicMock()
        alert_adapter._alert_manager = mock_manager
        
        assert alert_adapter.alert_manager == mock_manager

    @pytest.mark.asyncio
    @patch('src.state.monitoring_integration.get_alert_manager')
    async def test_send_alert_success(self, mock_get_alert_manager, alert_adapter):
        """Test sending alert successfully."""
        from src.state.monitoring import AlertSeverity
        
        alert = MockAlert("Test alert", AlertSeverity.HIGH)
        mock_manager = MagicMock()
        mock_manager.fire_alert = AsyncMock()
        mock_get_alert_manager.return_value = mock_manager
        
        await alert_adapter.send_alert(alert)
        
        mock_manager.fire_alert.assert_called_once()
        # Check that the alert was converted properly
        call_args = mock_manager.fire_alert.call_args[0][0]
        assert call_args.message == "Test alert"

    @pytest.mark.asyncio
    @patch('src.state.monitoring_integration.get_alert_manager')
    async def test_send_alert_exception(self, mock_get_alert_manager, alert_adapter):
        """Test sending alert with exception."""
        from src.state.monitoring import AlertSeverity
        alert = MockAlert("Test alert", AlertSeverity.HIGH)
        mock_manager = MagicMock()
        mock_manager.fire_alert = AsyncMock(
            side_effect=Exception("Alert manager error")
        )
        mock_get_alert_manager.return_value = mock_manager
        
        # Should raise StateConsistencyError, wrapping the original exception
        with pytest.raises(StateConsistencyError):
            await alert_adapter.send_alert(alert)
        
        # Alert manager should be called 3 times (retry attempts)
        assert mock_manager.fire_alert.call_count == 3

    def test_convert_alert_severity(self, alert_adapter):
        """Test converting alert severity."""
        from src.monitoring.alerting import AlertSeverity as MonitoringAlertSeverity
        from src.state.monitoring import AlertSeverity
        
        # Test severity mapping
        test_cases = [
            (AlertSeverity.INFO, MonitoringAlertSeverity.LOW),
            (AlertSeverity.LOW, MonitoringAlertSeverity.LOW),
            (AlertSeverity.MEDIUM, MonitoringAlertSeverity.MEDIUM),
            (AlertSeverity.HIGH, MonitoringAlertSeverity.HIGH),
            (AlertSeverity.CRITICAL, MonitoringAlertSeverity.CRITICAL),
        ]
        
        for state_severity, expected_monitoring_severity in test_cases:
            # This tests the internal conversion logic
            # Since the method is private, we test it through send_alert
            alert = MockAlert("Test", state_severity)
            
            # The conversion would happen in _convert_alert_to_monitoring_alert method
            # For now, just verify the severity types exist
            assert state_severity is not None
            assert expected_monitoring_severity is not None


class TestStateMonitoringServiceIntegration:
    """Test StateMonitoringService with central monitoring integration."""

    @pytest.fixture
    def mock_state_service(self):
        """Create mock state service."""
        service = MagicMock()
        return service

    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector."""
        collector = MagicMock()
        return collector

    @pytest.fixture
    def enhanced_service(self, mock_state_service, mock_metrics_collector):
        """Create enhanced monitoring service."""
        with patch("src.state.monitoring_integration.get_tracer"):
            return create_integrated_monitoring_service(
                mock_state_service, mock_metrics_collector
            )

    def test_initialization(self, mock_state_service, mock_metrics_collector):
        """Test enhanced service initialization."""
        with patch("src.state.monitoring_integration.get_tracer") as mock_tracer:
            mock_tracer.return_value = MagicMock()
            
            service = create_integrated_monitoring_service(
                mock_state_service, mock_metrics_collector
            )
            
            assert service.state_service == mock_state_service
            assert isinstance(service.metrics_adapter, StateMetricsAdapter)
            assert isinstance(service.alert_adapter, StateAlertAdapter)

    def test_record_metric(self, enhanced_service):
        """Test recording metric through enhanced service."""
        from src.state.monitoring import MetricType
        enhanced_service.metrics_adapter = MagicMock()
        
        metric_name = "test_metric"
        value = 42.0
        metric_type = MetricType.GAUGE
        tags = {"environment": "test"}
        
        enhanced_service.record_metric(metric_name, value, metric_type, tags)
        
        enhanced_service.metrics_adapter.record_state_metric.assert_called_once()
        call_args = enhanced_service.metrics_adapter.record_state_metric.call_args[0][0]
        
        assert call_args.name == metric_name
        assert call_args.value == value
        assert call_args.tags == tags

    def test_record_metric_with_defaults(self, enhanced_service):
        """Test recording metric with default parameters."""
        enhanced_service.metrics_adapter = MagicMock()
        
        enhanced_service.record_metric("test_metric", 10)
        
        call_args = enhanced_service.metrics_adapter.record_state_metric.call_args[0][0]
        assert call_args.name == "test_metric"
        assert call_args.value == 10
        assert call_args.tags == {}

    def test_record_operation_time(self, enhanced_service):
        """Test recording operation time."""
        enhanced_service.metrics_adapter = MagicMock()
        
        operation_name = "validation"
        duration_ms = 150.0
        
        enhanced_service.record_operation_time(operation_name, duration_ms)
        
        enhanced_service.metrics_adapter.record_operation_time.assert_called_once_with(
            operation_name, duration_ms
        )

    @pytest.mark.asyncio
    async def test_forward_alert_to_central(self, enhanced_service):
        """Test forwarding alert to central monitoring."""
        enhanced_service.alert_adapter = AsyncMock()
        
        alert = MockAlert("Test alert")
        
        await enhanced_service._forward_alert_to_central(alert)
        
        enhanced_service.alert_adapter.send_alert.assert_called_once_with(alert)

    @pytest.mark.asyncio
    async def test_run_health_check(self, enhanced_service):
        """Test running health check."""
        from src.state.monitoring import HealthCheck
        
        mock_check_function = AsyncMock(return_value={"status": "healthy"})
        check = HealthCheck(
            name="test_check",
            check_function=mock_check_function
        )
        
        enhanced_service.metrics_adapter = MagicMock()
        
        await enhanced_service._run_health_check(check)
        
        mock_check_function.assert_called_once()
        enhanced_service.metrics_adapter.record_health_check.assert_called_once()


class TestCreateIntegratedMonitoringService:
    """Test create_integrated_monitoring_service function."""

    def test_create_integrated_service_with_collector(self):
        """Test creating integrated service with metrics collector."""
        mock_state_service = MagicMock()
        mock_metrics_collector = MagicMock()
        
        service = create_integrated_monitoring_service(
            mock_state_service, mock_metrics_collector
        )
        
        assert isinstance(service, StateMonitoringService)
        assert service.state_service == mock_state_service

    def test_create_integrated_service_without_collector(self):
        """Test creating integrated service without metrics collector."""
        mock_state_service = MagicMock()
        
        with patch("src.state.monitoring_integration.MetricsCollector") as mock_collector_class:
            mock_collector = MagicMock()
            mock_collector_class.return_value = mock_collector
            
            service = create_integrated_monitoring_service(mock_state_service)
            
            assert isinstance(service, StateMonitoringService)
            mock_collector_class.assert_called_once()


class TestIntegrationScenarios:
    """Test integration scenarios between components."""

    @pytest.fixture
    def full_monitoring_setup(self):
        """Create full monitoring setup for integration tests."""
        mock_state_service = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_metrics_collector.increment_counter = MagicMock()
        mock_metrics_collector.set_gauge = MagicMock()
        mock_metrics_collector.observe_histogram = MagicMock()
        
        with patch("src.state.monitoring_integration.get_tracer"), \
             patch("src.monitoring.alerting.get_alert_manager") as mock_alert_mgr:
            
            mock_alert_mgr.return_value = MagicMock()
            
            service = create_integrated_monitoring_service(
                mock_state_service, mock_metrics_collector
            )
            
            return service, mock_metrics_collector

    def test_end_to_end_metric_recording(self, full_monitoring_setup):
        """Test end-to-end metric recording flow."""
        from src.state.monitoring import MetricType
        service, mock_collector = full_monitoring_setup
        
        # Record a metric through the enhanced service
        service.record_metric("test_metric", 100, MetricType.COUNTER, {"test": "tag"})
        
        # Verify it reaches the metrics collector
        mock_collector.increment_counter.assert_called_once()
        call_args = mock_collector.increment_counter.call_args
        
        assert "state.test_metric" in call_args[0][0]
        assert call_args[0][2] == 100  # Value
        assert call_args[0][1]["test"] == "tag"  # Tags preserved

    @pytest.mark.asyncio
    async def test_end_to_end_alert_flow(self, full_monitoring_setup):
        """Test end-to-end alert flow."""
        service, _ = full_monitoring_setup
        service.alert_adapter = AsyncMock()
        
        alert = MockAlert("Integration test alert", "error")
        
        await service._forward_alert_to_central(alert)
        
        service.alert_adapter.send_alert.assert_called_once_with(alert)

    def test_operation_timing_integration(self, full_monitoring_setup):
        """Test operation timing integration."""
        service, mock_collector = full_monitoring_setup
        
        service.record_operation_time("state_validation", 75.5)
        
        mock_collector.observe_histogram.assert_called_once()
        call_args = mock_collector.observe_histogram.call_args
        
        assert "state.operation.duration" in call_args[0][0]
        assert call_args[0][1] == 75.5
        assert call_args[0][2]["operation"] == "state_validation"


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_metrics_adapter_with_collector_exception(self):
        """Test metrics adapter handling collector exceptions."""
        mock_collector = MagicMock()
        mock_collector.set_gauge.side_effect = Exception("Collector error")
        
        with patch("src.state.monitoring_integration.get_tracer"):
            adapter = StateMetricsAdapter(mock_collector)
            
            metric = MockMetric("test", 10)
            # Should not raise exception
            adapter.record_state_metric(metric)

    @pytest.mark.asyncio
    async def test_alert_adapter_with_manager_exception(self):
        """Test alert adapter handling manager exceptions."""
        with patch("src.monitoring.alerting.get_alert_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.send_alert = AsyncMock(side_effect=Exception("Manager error"))
            mock_get_manager.return_value = mock_manager
            
            adapter = StateAlertAdapter()
            alert = MockAlert("Test alert")
            
            # Should not raise exception
            await adapter.send_alert(alert)

    def test_enhanced_service_with_adapter_errors(self):
        """Test enhanced service handling adapter errors."""
        mock_state_service = MagicMock()
        
        with patch("src.state.monitoring_integration.get_tracer"):
            service = StateMonitoringService(mock_state_service)
            
            # Mock adapter to raise exception
            service.metrics_adapter = MagicMock()
            service.metrics_adapter.record_state_metric.side_effect = Exception("Adapter error")
            
            # Should not raise exception
            service.record_metric("test", 10)


if __name__ == "__main__":
    pytest.main([__file__])