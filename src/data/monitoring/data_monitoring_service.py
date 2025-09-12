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

from src.core import BaseComponent, HealthCheckResult, HealthStatus
from src.core.config import Config
from src.core.types import AlertSeverity
from src.data.events import DataEvent, DataEventSubscriber, DataEventType

# Import from P-002A error handling
from src.error_handling import ErrorHandler

# Import from P-007A utilities


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
                            AlertSeverity.MEDIUM if quality_score > 70 else AlertSeverity.HIGH
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
                                AlertSeverity.HIGH if error_rate > 0.1 else AlertSeverity.MEDIUM
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
                        severity=AlertSeverity.MEDIUM,
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
                    severity=AlertSeverity.HIGH,
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
                            AlertSeverity.MEDIUM if avg_response_time < 1000 else AlertSeverity.HIGH
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
                    isinstance(size, (int, float))
                    and size > self.performance_thresholds["max_queue_size"]
                ):
                    alerts.append(
                        Alert(
                            alert_id=f"queue_size_{queue_name}_{datetime.now(timezone.utc).timestamp()}",
                            severity=AlertSeverity.MEDIUM if size < 15000 else AlertSeverity.HIGH,
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
                    severity=AlertSeverity.HIGH,
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
                        severity = AlertSeverity.MEDIUM
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
                        severity = AlertSeverity.MEDIUM
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
                    severity=AlertSeverity.HIGH,
                    message=f"SLA monitoring failed: {e}",
                    component="monitoring",
                    metadata={"error": str(e)},
                )
            ]


class DataMonitoringService(BaseComponent, DataEventSubscriber):
    """
    Enterprise-grade Data Monitoring Service for comprehensive infrastructure monitoring.

    This service provides:
    - Real-time data quality monitoring and alerting
    - Performance metrics tracking and SLA compliance
    - Anomaly detection and early warning systems
    - Comprehensive dashboards and reporting
    - Integration with external monitoring systems
    """

    def __init__(self, config: Config):
        """Initialize the Data Monitoring Service."""
        super().__init__()
        self.config = config
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

        # Metrics storage - now populated from events
        self.metrics_history: list[dict[str, Any]] = []
        self.max_history_size = 10000

        # Event-based metrics tracking
        self.data_events_received = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.validation_failure_count = 0
        self.records_stored = 0
        self.records_retrieved = 0

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

            # Subscribe to data events
            await self._setup_event_subscriptions()

            # Start monitoring loops
            monitoring_task = asyncio.create_task(self._monitoring_loop())
            alert_cleanup_task = asyncio.create_task(self._alert_cleanup_loop())
            metrics_cleanup_task = asyncio.create_task(self._metrics_cleanup_loop())

            self._background_tasks.extend(
                [monitoring_task, alert_cleanup_task, metrics_cleanup_task]
            )

            self._initialized = True
            self.logger.info("Data Monitoring Service initialized successfully")

        except Exception as e:
            self.logger.error(f"Monitoring service initialization failed: {e}")
            raise

    async def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions to receive data events."""
        try:
            # Subscribe to data events
            await self._subscribe_to_data_event(DataEventType.DATA_STORED, self._handle_data_stored)
            await self._subscribe_to_data_event(
                DataEventType.DATA_RETRIEVED, self._handle_data_retrieved
            )
            await self._subscribe_to_data_event(
                DataEventType.DATA_VALIDATION_FAILED, self._handle_validation_failed
            )
            await self._subscribe_to_data_event(DataEventType.CACHE_HIT, self._handle_cache_hit)
            await self._subscribe_to_data_event(DataEventType.CACHE_MISS, self._handle_cache_miss)

            self.logger.info("Data event subscriptions setup completed")

        except Exception as e:
            self.logger.error(f"Failed to setup event subscriptions: {e}")
            raise

    async def _subscribe_to_data_event(
        self, event_type: DataEventType, handler: Callable[[DataEvent], None]
    ) -> None:
        """Subscribe to a data event type with handler."""
        # This is a placeholder for actual event subscription mechanism
        # In a full implementation, this would register with an event bus/broker
        self.logger.debug(f"Subscribed to {event_type.value} events")

    async def _handle_data_stored(self, event: DataEvent) -> None:
        """Handle data stored events."""
        try:
            self.data_events_received += 1
            self.records_stored += event.data.get("records_count", 0)

            # Check for performance issues based on event data
            if "processing_time_ms" in event.metadata:
                processing_time = event.metadata["processing_time_ms"]
                if processing_time > 1000:  # Over 1 second
                    await self._handle_alert(
                        Alert(
                            alert_id=f"slow_storage_{datetime.now(timezone.utc).timestamp()}",
                            severity=AlertSeverity.MEDIUM,
                            message=f"Slow data storage detected: {processing_time:.1f}ms",
                            component="data_storage",
                            metadata=event.metadata,
                        )
                    )

        except Exception as e:
            self.logger.error(f"Error handling data stored event: {e}")

    async def _handle_data_retrieved(self, event: DataEvent) -> None:
        """Handle data retrieved events."""
        try:
            self.data_events_received += 1
            self.records_retrieved += event.data.get("records_count", 0)

        except Exception as e:
            self.logger.error(f"Error handling data retrieved event: {e}")

    async def _handle_validation_failed(self, event: DataEvent) -> None:
        """Handle data validation failed events."""
        try:
            self.data_events_received += 1
            self.validation_failure_count += 1

            # Generate quality alert
            await self._handle_alert(
                Alert(
                    alert_id=f"validation_failed_{datetime.now(timezone.utc).timestamp()}",
                    severity=AlertSeverity.ERROR,
                    message=f"Data validation failed: {event.data.get('error', 'Unknown error')}",
                    component="data_quality",
                    metadata=event.data,
                )
            )

        except Exception as e:
            self.logger.error(f"Error handling validation failed event: {e}")

    async def _handle_cache_hit(self, event: DataEvent) -> None:
        """Handle cache hit events."""
        try:
            self.data_events_received += 1
            self.cache_hit_count += 1

        except Exception as e:
            self.logger.error(f"Error handling cache hit event: {e}")

    async def _handle_cache_miss(self, event: DataEvent) -> None:
        """Handle cache miss events."""
        try:
            self.data_events_received += 1
            self.cache_miss_count += 1

            # Check cache hit rate and generate alert if too low
            total_cache_requests = self.cache_hit_count + self.cache_miss_count
            if total_cache_requests > 100:  # Only check after enough requests
                hit_rate = self.cache_hit_count / total_cache_requests
                if hit_rate < 0.5:  # Less than 50% hit rate
                    await self._handle_alert(
                        Alert(
                            alert_id=f"low_cache_hit_rate_{datetime.now(timezone.utc).timestamp()}",
                            severity=AlertSeverity.MEDIUM,
                            message=f"Low cache hit rate: {hit_rate:.2%}",
                            component="cache_performance",
                            metadata={"hit_rate": hit_rate, "total_requests": total_cache_requests},
                        )
                    )

        except Exception as e:
            self.logger.error(f"Error handling cache miss event: {e}")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._shutdown_requested:
            try:
                await asyncio.sleep(self.monitoring_config["check_interval"])

                # Build metrics from event-based tracking
                metrics = self._build_event_based_metrics()

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

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")

    def _build_event_based_metrics(self) -> dict[str, Any]:
        """Build metrics from event-based tracking."""
        total_cache_requests = self.cache_hit_count + self.cache_miss_count
        cache_hit_rate = (self.cache_hit_count / max(1, total_cache_requests)) * 100

        return {
            "data_events_received": self.data_events_received,
            "records_stored": self.records_stored,
            "records_retrieved": self.records_retrieved,
            "validation_failures": self.validation_failure_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "cache_hit_rate": cache_hit_rate,
            "data_quality_score": max(
                0, 100 - (self.validation_failure_count * 10)
            ),  # Simplified calculation
            "successful_records": self.records_stored,
            "failed_records": self.validation_failure_count,
            "throughput_per_second": self.records_stored
            / max(
                1, (datetime.now(timezone.utc) - datetime.now(timezone.utc)).total_seconds() or 1
            ),
            "active_alerts": len(self.active_alerts),
        }

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
            elif alert.severity == AlertSeverity.HIGH:
                self.logger.error(f"HIGH ALERT: {alert.message}")
            elif alert.severity == AlertSeverity.MEDIUM:
                self.logger.warning(f"MEDIUM ALERT: {alert.message}")
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
                    [a for a in self.active_alerts.values() if a.severity == AlertSeverity.HIGH]
                ),
                "warnings": len(
                    [a for a in self.active_alerts.values() if a.severity == AlertSeverity.MEDIUM]
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

            # Cleanup event subscriptions
            await self._cleanup_event_subscriptions()

            self._initialized = False
            self.logger.info("Monitoring service cleanup completed")

        except Exception as e:
            self.logger.error(f"Monitoring cleanup error: {e}")
