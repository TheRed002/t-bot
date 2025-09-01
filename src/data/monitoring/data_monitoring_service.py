"""
Data Monitoring Service - Comprehensive Data Infrastructure Monitoring

This module provides enterprise-grade monitoring and alerting for the
data infrastructure, ensuring zero data loss and maintaining SLAs
for mission-critical financial data processing.

Key Features:
- Real-time data quality monitoring
- Performance metrics and alerting
- Data pipeline health monitoring
- SLA tracking and compliance reporting
- Anomaly detection and early warning
- Disaster recovery coordination

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from src.core.base.component import BaseComponent
from src.core.base.interfaces import HealthCheckResult, HealthStatus
from src.core.config import Config

# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler

# Import from P-007A utilities


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Metric type enumeration."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class SLAStatus(Enum):
    """SLA compliance status."""

    MEETING = "meeting"
    AT_RISK = "at_risk"
    BREACHED = "breached"


@dataclass
class Alert:
    """Data infrastructure alert."""

    alert_id: str
    severity: AlertSeverity
    message: str
    component: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: datetime = None


@dataclass
class Metric:
    """Performance metric."""

    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class SLATarget:
    """SLA target definition."""

    name: str
    target_value: float
    current_value: float = 0.0
    status: SLAStatus = SLAStatus.MEETING
    threshold_warning: float = 0.9  # 90% of target
    threshold_critical: float = 0.8  # 80% of target
    measurement_window: int = 3600  # 1 hour in seconds


class DataQualityMonitor:
    """Data quality monitoring component."""

    def __init__(self, config: Config):
        self.config = config
        self.quality_thresholds = {
            "min_quality_score": 85.0,
            "max_error_rate": 0.05,  # 5% error rate
            "max_latency_ms": 1000,
            "min_throughput_per_second": 100,
        }

        # Quality metrics
        self.quality_metrics = {
            "records_processed": 0,
            "records_valid": 0,
            "records_invalid": 0,
            "avg_quality_score": 0.0,
            "quality_degradation_events": 0,
        }

    async def check_data_quality(self, metrics: dict[str, Any]) -> list[Alert]:
        """Check data quality and generate alerts if needed."""
        alerts = []

        try:
            # Check quality score
            quality_score = metrics.get("data_quality_score", 100.0)
            if quality_score < self.quality_thresholds["min_quality_score"]:
                alerts.append(
                    Alert(
                        alert_id=f"quality_score_{datetime.now(timezone.utc).timestamp()}",
                        severity=(
                            AlertSeverity.WARNING if quality_score > 70 else AlertSeverity.ERROR
                        ),
                        message=f"Data quality score below threshold: {quality_score:.1f}%",
                        component="data_quality",
                        metadata={
                            "quality_score": quality_score,
                            "threshold": self.quality_thresholds["min_quality_score"],
                        },
                    )
                )

            # Check error rate
            total_records = metrics.get("successful_records", 0) + metrics.get("failed_records", 0)
            if total_records > 0:
                error_rate = metrics.get("failed_records", 0) / total_records
                if error_rate > self.quality_thresholds["max_error_rate"]:
                    alerts.append(
                        Alert(
                            alert_id=f"error_rate_{datetime.now(timezone.utc).timestamp()}",
                            severity=(
                                AlertSeverity.ERROR if error_rate > 0.1 else AlertSeverity.WARNING
                            ),
                            message=f"Error rate above threshold: {error_rate:.2%}",
                            component="data_quality",
                            metadata={
                                "error_rate": error_rate,
                                "threshold": self.quality_thresholds["max_error_rate"],
                            },
                        )
                    )

            # Check throughput
            throughput = metrics.get("throughput_per_second", 0)
            if throughput < self.quality_thresholds["min_throughput_per_second"]:
                alerts.append(
                    Alert(
                        alert_id=f"throughput_{datetime.now(timezone.utc).timestamp()}",
                        severity=AlertSeverity.WARNING,
                        message=f"Throughput below threshold: {throughput:.1f} records/sec",
                        component="performance",
                        metadata={
                            "throughput": throughput,
                            "threshold": self.quality_thresholds["min_throughput_per_second"],
                        },
                    )
                )

            return alerts

        except Exception as e:
            return [
                Alert(
                    alert_id=f"monitor_error_{datetime.now(timezone.utc).timestamp()}",
                    severity=AlertSeverity.ERROR,
                    message=f"Data quality monitoring failed: {e}",
                    component="monitoring",
                    metadata={"error": str(e)},
                )
            ]


class PerformanceMonitor:
    """Performance monitoring component."""

    def __init__(self, config: Config):
        self.config = config
        self.performance_thresholds = {
            "max_response_time_ms": 500,
            "max_queue_size": 10000,
            "max_memory_usage_mb": 1024,
            "max_cpu_usage_percent": 80,
        }

        # Performance history
        self.response_times: list[float] = []
        self.max_history_size = 1000

    async def check_performance(self, metrics: dict[str, Any]) -> list[Alert]:
        """Check performance metrics and generate alerts."""
        alerts = []

        try:
            # Check response time
            avg_response_time = metrics.get("avg_processing_time_ms", 0)
            self.response_times.append(avg_response_time)

            # Keep only recent history
            if len(self.response_times) > self.max_history_size:
                self.response_times = self.response_times[-self.max_history_size :]

            if avg_response_time > self.performance_thresholds["max_response_time_ms"]:
                alerts.append(
                    Alert(
                        alert_id=f"response_time_{datetime.now(timezone.utc).timestamp()}",
                        severity=(
                            AlertSeverity.WARNING
                            if avg_response_time < 1000
                            else AlertSeverity.ERROR
                        ),
                        message=f"Response time above threshold: {avg_response_time:.1f}ms",
                        component="performance",
                        metadata={
                            "response_time": avg_response_time,
                            "threshold": self.performance_thresholds["max_response_time_ms"],
                        },
                    )
                )

            # Check queue sizes
            queue_sizes = metrics.get("queue_sizes", {})
            for queue_name, size in queue_sizes.items():
                if (
                    isinstance(size, int | float)
                    and size > self.performance_thresholds["max_queue_size"]
                ):
                    alerts.append(
                        Alert(
                            alert_id=f"queue_size_{queue_name}_{datetime.now(timezone.utc).timestamp()}",
                            severity=AlertSeverity.WARNING if size < 15000 else AlertSeverity.ERROR,
                            message=f"Queue size above threshold: {queue_name} = {size}",
                            component="performance",
                            metadata={
                                "queue_name": queue_name,
                                "size": size,
                                "threshold": self.performance_thresholds["max_queue_size"],
                            },
                        )
                    )

            return alerts

        except Exception as e:
            return [
                Alert(
                    alert_id=f"perf_monitor_error_{datetime.now(timezone.utc).timestamp()}",
                    severity=AlertSeverity.ERROR,
                    message=f"Performance monitoring failed: {e}",
                    component="monitoring",
                    metadata={"error": str(e)},
                )
            ]


class SLAMonitor:
    """SLA compliance monitoring component."""

    def __init__(self, config: Config):
        self.config = config
        self.sla_targets = [
            SLATarget(
                name="data_availability",
                target_value=99.9,  # 99.9% uptime
                threshold_warning=99.5,
                threshold_critical=99.0,
            ),
            SLATarget(
                name="data_freshness",
                target_value=60.0,  # Data within 60 seconds
                threshold_warning=120.0,
                threshold_critical=300.0,
            ),
            SLATarget(
                name="processing_latency",
                target_value=100.0,  # 100ms average latency
                threshold_warning=200.0,
                threshold_critical=500.0,
            ),
            SLATarget(
                name="data_quality",
                target_value=95.0,  # 95% quality score
                threshold_warning=90.0,
                threshold_critical=85.0,
            ),
        ]

    async def check_sla_compliance(self, metrics: dict[str, Any]) -> list[Alert]:
        """Check SLA compliance and generate alerts."""
        alerts = []

        try:
            for sla in self.sla_targets:
                # Update current value based on metrics
                if sla.name == "data_availability":
                    total_time = metrics.get("pipeline_uptime", 0)
                    if total_time > 0:
                        # Simplified availability calculation
                        successful_requests = metrics.get("successful_requests", 0)
                        total_requests = successful_requests + metrics.get("failed_requests", 0)
                        if total_requests > 0:
                            sla.current_value = (successful_requests / total_requests) * 100

                elif sla.name == "data_freshness":
                    # Average data age in seconds
                    sla.current_value = metrics.get("avg_data_age_seconds", 0)

                elif sla.name == "processing_latency":
                    # Average processing time in milliseconds
                    sla.current_value = metrics.get("avg_processing_time_ms", 0)

                elif sla.name == "data_quality":
                    # Data quality score
                    sla.current_value = metrics.get("data_quality_score", 100)

                # Check SLA status
                if sla.name in ["data_availability", "data_quality"]:
                    # Higher is better
                    if sla.current_value < sla.threshold_critical:
                        sla.status = SLAStatus.BREACHED
                        severity = AlertSeverity.CRITICAL
                    elif sla.current_value < sla.threshold_warning:
                        sla.status = SLAStatus.AT_RISK
                        severity = AlertSeverity.WARNING
                    else:
                        sla.status = SLAStatus.MEETING
                        continue
                else:
                    # Lower is better
                    if sla.current_value > sla.threshold_critical:
                        sla.status = SLAStatus.BREACHED
                        severity = AlertSeverity.CRITICAL
                    elif sla.current_value > sla.threshold_warning:
                        sla.status = SLAStatus.AT_RISK
                        severity = AlertSeverity.WARNING
                    else:
                        sla.status = SLAStatus.MEETING
                        continue

                # Generate alert for SLA violation
                alerts.append(
                    Alert(
                        alert_id=f"sla_{sla.name}_{datetime.now(timezone.utc).timestamp()}",
                        severity=severity,
                        message=f"SLA {sla.status.value}: {sla.name} = {sla.current_value:.2f} (target: {sla.target_value:.2f})",
                        component="sla",
                        metadata={
                            "sla_name": sla.name,
                            "current_value": sla.current_value,
                            "target_value": sla.target_value,
                            "status": sla.status.value,
                        },
                    )
                )

            return alerts

        except Exception as e:
            return [
                Alert(
                    alert_id=f"sla_monitor_error_{datetime.now(timezone.utc).timestamp()}",
                    severity=AlertSeverity.ERROR,
                    message=f"SLA monitoring failed: {e}",
                    component="monitoring",
                    metadata={"error": str(e)},
                )
            ]


class DataMonitoringService(BaseComponent):
    """
    Enterprise-grade Data Monitoring Service for comprehensive infrastructure monitoring.

    This service provides:
    - Real-time data quality monitoring and alerting
    - Performance metrics tracking and SLA compliance
    - Anomaly detection and early warning systems
    - Comprehensive dashboards and reporting
    - Integration with external monitoring systems
    """

    def __init__(self, config: Config, metrics_provider=None):
        """Initialize the Data Monitoring Service."""
        super().__init__()
        self.config = config
        self.metrics_provider = metrics_provider
        self.error_handler = ErrorHandler(config)

        # Configuration
        self._setup_configuration()

        # Monitoring components
        self.quality_monitor = DataQualityMonitor(config)
        self.performance_monitor = PerformanceMonitor(config)
        self.sla_monitor = SLAMonitor(config)

        # Alert management
        self.active_alerts: dict[str, Alert] = {}
        self.alert_handlers: list[Callable] = []

        # Metrics storage
        self.metrics_history: list[dict[str, Any]] = []
        self.max_history_size = 10000

        # Background task management
        self._background_tasks: list[asyncio.Task] = []

        self._initialized = False
        self._shutdown_requested = False

    def _setup_configuration(self) -> None:
        """Setup monitoring configuration."""
        monitoring_config = getattr(self.config, "data_monitoring", {})

        self.monitoring_config = {
            "check_interval": monitoring_config.get("check_interval", 60),  # Check every minute
            "alert_retention_hours": monitoring_config.get("alert_retention_hours", 24),
            "metrics_retention_hours": monitoring_config.get(
                "metrics_retention_hours", 168
            ),  # 1 week
            "enable_email_alerts": monitoring_config.get("enable_email_alerts", False),
            "enable_slack_alerts": monitoring_config.get("enable_slack_alerts", False),
        }

    async def initialize(self) -> None:
        """Initialize the monitoring service."""
        try:
            if self._initialized:
                return

            self.logger.info("Initializing Data Monitoring Service...")

            # Start monitoring loops
            monitoring_task = asyncio.create_task(self._monitoring_loop())
            alert_cleanup_task = asyncio.create_task(self._alert_cleanup_loop())
            metrics_cleanup_task = asyncio.create_task(self._metrics_cleanup_loop())

            self._background_tasks.extend([monitoring_task, alert_cleanup_task, metrics_cleanup_task])

            self._initialized = True
            self.logger.info("Data Monitoring Service initialized successfully")

        except Exception as e:
            self.logger.error(f"Monitoring service initialization failed: {e}")
            raise

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._shutdown_requested:
            try:
                await asyncio.sleep(self.monitoring_config["check_interval"])

                # Get metrics from metrics provider
                if self.metrics_provider:
                    try:
                        metrics = await self._get_metrics_from_provider()

                        # Store metrics
                        self.metrics_history.append(
                            {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "metrics": metrics,
                            }
                        )

                        # Limit history size
                        if len(self.metrics_history) > self.max_history_size:
                            self.metrics_history = self.metrics_history[-self.max_history_size :]

                        # Run monitoring checks
                        await self._run_monitoring_checks(metrics)

                    except Exception as e:
                        self.logger.error(f"Failed to get metrics from metrics provider: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")

    async def _get_metrics_from_provider(self) -> dict[str, Any]:
        """Get metrics from the configured metrics provider."""
        if hasattr(self.metrics_provider, "get_comprehensive_metrics"):
            return self.metrics_provider.get_comprehensive_metrics()
        elif hasattr(self.metrics_provider, "get_metrics"):
            return await self.metrics_provider.get_metrics()
        else:
            # Fallback to calling it directly if it's callable
            if callable(self.metrics_provider):
                result = self.metrics_provider()
                if hasattr(result, "__await__"):
                    return await result
                return result
            else:
                return {}

    async def _run_monitoring_checks(self, metrics: dict[str, Any]) -> None:
        """Run all monitoring checks and handle alerts."""
        all_alerts = []

        # Flatten metrics for monitoring
        flat_metrics = self._flatten_metrics(metrics)

        # Run quality checks
        quality_alerts = await self.quality_monitor.check_data_quality(flat_metrics)
        all_alerts.extend(quality_alerts)

        # Run performance checks
        perf_alerts = await self.performance_monitor.check_performance(flat_metrics)
        all_alerts.extend(perf_alerts)

        # Run SLA checks
        sla_alerts = await self.sla_monitor.check_sla_compliance(flat_metrics)
        all_alerts.extend(sla_alerts)

        # Process alerts
        for alert in all_alerts:
            await self._handle_alert(alert)

    def _flatten_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Flatten nested metrics dictionary."""
        flat = {}

        def _flatten(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{prefix}_{key}" if prefix else key
                    if isinstance(value, dict):
                        _flatten(value, new_key)
                    else:
                        flat[new_key] = value
            else:
                flat[prefix] = obj

        _flatten(metrics)
        return flat

    async def _handle_alert(self, alert: Alert) -> None:
        """Handle a monitoring alert."""
        try:
            # Check if this is a duplicate alert
            if alert.alert_id in self.active_alerts:
                return

            # Store alert
            self.active_alerts[alert.alert_id] = alert

            # Log alert
            if alert.severity == AlertSeverity.CRITICAL:
                self.logger.critical(f"CRITICAL ALERT: {alert.message}")
            elif alert.severity == AlertSeverity.ERROR:
                self.logger.error(f"ERROR ALERT: {alert.message}")
            elif alert.severity == AlertSeverity.WARNING:
                self.logger.warning(f"WARNING ALERT: {alert.message}")
            else:
                self.logger.info(f"INFO ALERT: {alert.message}")

            # Execute alert handlers
            for handler in self.alert_handlers:
                try:
                    await handler(alert)
                except Exception as e:
                    self.logger.error(f"Alert handler failed: {e}")

        except Exception as e:
            self.logger.error(f"Alert handling failed: {e}")

    async def _alert_cleanup_loop(self) -> None:
        """Cleanup old resolved alerts."""
        while not self._shutdown_requested:
            try:
                await asyncio.sleep(3600)  # Check every hour

                cutoff_time = datetime.now(timezone.utc) - timedelta(
                    hours=self.monitoring_config["alert_retention_hours"]
                )

                expired_alerts = [
                    alert_id
                    for alert_id, alert in self.active_alerts.items()
                    if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff_time
                ]

                for alert_id in expired_alerts:
                    del self.active_alerts[alert_id]

                if expired_alerts:
                    self.logger.debug(f"Cleaned up {len(expired_alerts)} expired alerts")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Alert cleanup error: {e}")

    async def _metrics_cleanup_loop(self) -> None:
        """Cleanup old metrics."""
        while not self._shutdown_requested:
            try:
                await asyncio.sleep(3600)  # Check every hour

                cutoff_time = datetime.now(timezone.utc) - timedelta(
                    hours=self.monitoring_config["metrics_retention_hours"]
                )

                # Remove old metrics
                self.metrics_history = [
                    entry
                    for entry in self.metrics_history
                    if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
                ]

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics cleanup error: {e}")

    def register_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register an alert handler function."""
        self.alert_handlers.append(handler)

    def get_active_alerts(self, severity: AlertSeverity = None) -> list[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())

        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            self.active_alerts[alert_id].resolved_at = datetime.now(timezone.utc)
            return True
        return False

    def get_metrics_summary(self, hours: int = 1) -> dict[str, Any]:
        """Get metrics summary for the specified time period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        recent_metrics = [
            entry
            for entry in self.metrics_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]

        if not recent_metrics:
            return {}

        # Calculate summary statistics
        summary = {
            "time_period_hours": hours,
            "data_points": len(recent_metrics),
            "alerts": {
                "total": len(self.active_alerts),
                "critical": len(
                    [a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]
                ),
                "errors": len(
                    [a for a in self.active_alerts.values() if a.severity == AlertSeverity.ERROR]
                ),
                "warnings": len(
                    [a for a in self.active_alerts.values() if a.severity == AlertSeverity.WARNING]
                ),
            },
            "sla_status": {sla.name: sla.status.value for sla in self.sla_monitor.sla_targets},
        }

        return summary

    async def health_check(self) -> HealthCheckResult:
        """Perform monitoring service health check."""
        critical_alerts = len(
            [a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]
        )

        details = {
            "initialized": self._initialized,
            "active_alerts": len(self.active_alerts),
            "critical_alerts": critical_alerts,
            "metrics_history_size": len(self.metrics_history),
        }

        # Check if there are critical alerts
        if critical_alerts > 0:
            status = HealthStatus.DEGRADED
            message = f"Monitoring service degraded: {critical_alerts} critical alerts"
        else:
            status = HealthStatus.HEALTHY
            message = "Monitoring service healthy"

        return HealthCheckResult(status=status, details=details, message=message)

    async def cleanup(self) -> None:
        """Cleanup monitoring service resources."""
        try:
            self.logger.info("Starting monitoring service cleanup...")
            self._shutdown_requested = True

            # Cancel background tasks
            if self._background_tasks:
                for task in self._background_tasks:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        self.logger.warning(f"Error cancelling monitoring task: {e}")

                self._background_tasks.clear()

            # Clear active alerts
            self.active_alerts.clear()

            # Clear metrics history
            self.metrics_history.clear()

            self._initialized = False
            self.logger.info("Monitoring service cleanup completed")

        except Exception as e:
            self.logger.error(f"Monitoring cleanup error: {e}")
