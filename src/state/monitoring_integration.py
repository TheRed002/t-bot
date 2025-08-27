"""
Integration layer between State module and central Monitoring module.

This module provides adapters and integration components to connect the
state management system with the central monitoring infrastructure.
"""

from datetime import datetime
from typing import Any

from src.core.base.component import BaseComponent
from src.core.exceptions import StateError
from src.error_handling.context import ErrorContext
from src.error_handling.decorators import with_retry
from src.monitoring import MetricsCollector, get_alert_manager
from src.monitoring.alerting import (
    Alert as MonitoringAlert,
    AlertSeverity as MonitoringAlertSeverity,
    AlertStatus,
)
from src.monitoring.telemetry import get_tracer

from .monitoring import (
    Alert,
    AlertSeverity,
    HealthStatus,
    Metric,
    MetricType,
    StateMonitoringService,
)


class StateMetricsAdapter(BaseComponent):
    """
    Adapter to bridge state monitoring with central metrics collection.

    This adapter translates state-specific metrics to the central monitoring
    format and ensures all state metrics are visible in system dashboards.
    """

    def __init__(self, metrics_collector: MetricsCollector | None = None):
        """Initialize the metrics adapter."""
        super().__init__()
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.tracer = get_tracer("state_service")

        # Map state metric types to prometheus metric types
        self._metric_type_map = {
            MetricType.COUNTER: "counter",
            MetricType.GAUGE: "gauge",
            MetricType.HISTOGRAM: "histogram",
            MetricType.TIMER: "histogram",
        }

        self.logger.info("StateMetricsAdapter initialized")

    def record_state_metric(self, metric: Metric) -> None:
        """
        Record a state metric to central monitoring.

        Args:
            metric: State module metric to record
        """
        try:
            # Validate metric data
            if metric.value is None:
                self.logger.warning(f"Skipping metric {metric.name} with None value")
                return

            # Ensure value is numeric
            if not isinstance(metric.value, int | float):
                self.logger.error(
                    f"Invalid metric value type for {metric.name}: {type(metric.value)}"
                )
                return

            # Check for NaN or Inf
            import math

            if math.isnan(metric.value) or math.isinf(metric.value):
                self.logger.warning(
                    f"Skipping metric {metric.name} with NaN/Inf value: {metric.value}"
                )
                return

            # Map metric type
            metric_type = self._metric_type_map.get(metric.metric_type, "gauge")

            # Add state-specific labels
            labels = {"component": "state", "metric_name": metric.name, **metric.tags}

            # Record to central metrics based on type
            if metric_type == "counter":
                self.metrics_collector.increment_counter(
                    f"state.{metric.name}", labels, metric.value
                )
            elif metric_type == "gauge":
                self.metrics_collector.set_gauge(f"state.{metric.name}", metric.value, labels)
            elif metric_type == "histogram":
                self.metrics_collector.observe_histogram(
                    f"state.{metric.name}", metric.value, labels
                )

        except AttributeError as e:
            self.logger.error(f"Missing required attribute in metric {metric.name}: {e}")
        except ValueError as e:
            self.logger.error(f"Invalid value for metric {metric.name}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to record state metric {metric.name}: {e}")
            # Re-raise critical errors but not data validation errors
            if not isinstance(e, AttributeError | ValueError):
                raise

    def record_operation_time(self, operation: str, duration_ms: float) -> None:
        """Record state operation timing to central metrics."""
        self.metrics_collector.observe_histogram(
            "state.operation.duration", duration_ms, {"operation": operation}
        )

    def record_health_check(self, check_name: str, status: HealthStatus) -> None:
        """Record health check status to central metrics."""
        # Convert health status to numeric value
        health_value = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.DEGRADED: 0.5,
            HealthStatus.UNHEALTHY: 0.0,
            HealthStatus.CRITICAL: 0.0,
            HealthStatus.UNKNOWN: -1.0,
        }.get(status, -1.0)

        self.metrics_collector.set_gauge("state.health.status", health_value, {"check": check_name})


class StateAlertAdapter(BaseComponent):
    """
    Adapter to integrate state alerts with central alerting system.

    This ensures state-specific alerts are properly routed through
    the central alerting infrastructure.
    """

    def __init__(self):
        """Initialize the alert adapter."""
        super().__init__()
        # Get alert manager from global registry
        self._alert_manager = None

        # Map alert severities from state to monitoring module
        # Now properly aligned with monitoring module's severity levels
        self._severity_map = {
            AlertSeverity.INFO: MonitoringAlertSeverity.LOW,
            AlertSeverity.WARNING: MonitoringAlertSeverity.MEDIUM,
            AlertSeverity.ERROR: MonitoringAlertSeverity.HIGH,
            AlertSeverity.CRITICAL: MonitoringAlertSeverity.CRITICAL,
        }

        self.logger.info("StateAlertAdapter initialized")

    @property
    def alert_manager(self):
        """Get alert manager instance lazily."""
        if self._alert_manager is None:
            self._alert_manager = get_alert_manager()
        return self._alert_manager

    @with_retry(max_attempts=3, base_delay=0.5, backoff_factor=2.0)
    async def send_alert(self, alert: Alert) -> None:
        """
        Send a state alert through central alerting.

        Args:
            alert: State module alert to send
        """
        try:
            if not self.alert_manager:
                self.logger.warning("No alert manager available")
                return

            # Validate alert data
            if not alert.message:
                self.logger.error("Alert missing required message field")
                return

            if alert.severity not in AlertSeverity:
                self.logger.error(f"Invalid alert severity: {alert.severity}")
                return

            if not isinstance(alert.timestamp, datetime):
                self.logger.error(f"Invalid alert timestamp type: {type(alert.timestamp)}")
                return

            # Convert state alert to monitoring alert
            monitoring_alert = MonitoringAlert(
                rule_name=f"state.{alert.source}",
                severity=self._severity_map.get(alert.severity, MonitoringAlertSeverity.MEDIUM),
                status=AlertStatus.FIRING,
                message=alert.message,
                labels={
                    "source": "state",
                    "category": alert.category,
                    "alert_id": alert.alert_id,
                },
                annotations={
                    "title": alert.title,
                    "metric_name": alert.metric_name,
                    # Keep numeric values as strings for Prometheus compatibility
                    # but preserve precision and type information
                    "current_value": str(alert.current_value),
                    "threshold_value": str(alert.threshold_value),
                    "current_value_type": type(alert.current_value).__name__,
                    "threshold_value_type": type(alert.threshold_value).__name__,
                },
                starts_at=alert.timestamp,
            )

            # Fire alert through central alerting
            await self.alert_manager.fire_alert(monitoring_alert)

        except Exception as e:
            # Create error context for proper error tracking
            error_context = ErrorContext(
                component="StateAlertAdapter",
                operation="send_alert",
                error_type=type(e).__name__,
                error_message=str(e),
                context_data={
                    "alert_id": alert.alert_id,
                    "alert_source": alert.source,
                    "alert_severity": alert.severity.value if alert.severity else None,
                    "alert_category": alert.category,
                },
            )

            self.logger.error(
                f"Failed to send state alert: {e}", extra={"error_context": error_context.to_dict()}
            )

            # Re-raise critical alerting failures with context
            raise StateError(f"Alert delivery failed for {alert.alert_id}: {e}") from e


class EnhancedStateMonitoringService(StateMonitoringService):
    """
    Enhanced state monitoring service with central monitoring integration.

    This extends the existing StateMonitoringService to integrate with
    the central monitoring infrastructure while maintaining backward compatibility.
    """

    def __init__(self, state_service: Any, metrics_collector: MetricsCollector | None = None):
        """Initialize enhanced monitoring with central integration."""
        super().__init__(state_service)

        # Initialize adapters
        self.metrics_adapter = StateMetricsAdapter(metrics_collector)
        self.alert_adapter = StateAlertAdapter()

        # Register alert handler to forward to central system
        self.register_alert_handler(self._forward_alert_to_central)

        # Enable OpenTelemetry tracing
        self.tracer = get_tracer("state_monitoring")

        self.logger.info("EnhancedStateMonitoringService initialized with central integration")

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: dict[str, str] | None = None,
        unit: str = "",
    ) -> None:
        """
        Record metric to both state and central monitoring.

        Overrides parent method to ensure metrics go to central system.
        """
        # Record to state monitoring (parent)
        super().record_metric(name, value, metric_type, tags, unit)

        # Also record to central monitoring
        metric = Metric(name=name, metric_type=metric_type, value=value, tags=tags or {}, unit=unit)
        self.metrics_adapter.record_state_metric(metric)

    def record_operation_time(self, operation_name: str, duration_ms: float) -> None:
        """
        Record operation time to both systems.

        Overrides parent method to ensure timing goes to central metrics.
        """
        # Record to state monitoring (parent)
        super().record_operation_time(operation_name, duration_ms)

        # Also record to central monitoring
        self.metrics_adapter.record_operation_time(operation_name, duration_ms)

    async def _forward_alert_to_central(self, alert: Alert) -> None:
        """Forward state alerts to central alerting system."""
        await self.alert_adapter.send_alert(alert)

    async def _run_health_check(self, check: Any) -> None:
        """
        Run health check with OpenTelemetry tracing.

        Overrides parent method to add distributed tracing.
        """
        with self.tracer.start_as_current_span(f"health_check.{check.name}"):
            # Run the actual health check
            await super()._run_health_check(check)

            # Record result to central metrics
            self.metrics_adapter.record_health_check(check.name, check.status)


def create_integrated_monitoring_service(
    state_service: Any, metrics_collector: MetricsCollector | None = None
) -> EnhancedStateMonitoringService:
    """
    Factory function to create state monitoring with central integration.

    Args:
        state_service: The state service instance
        metrics_collector: Optional central metrics collector

    Returns:
        Enhanced monitoring service with central integration
    """
    return EnhancedStateMonitoringService(state_service, metrics_collector)
