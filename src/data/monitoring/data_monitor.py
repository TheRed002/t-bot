"""
Data Monitoring and Alerting System

This module implements comprehensive monitoring and alerting for the data infrastructure,
providing real-time health monitoring, performance tracking, anomaly detection,
and intelligent alerting for mission-critical trading data operations.

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
- DataService components for comprehensive monitoring
"""

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.core.base.component import BaseComponent
from src.core.config import Config


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(Enum):
    """Alert category types."""

    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    DATA_QUALITY = "data_quality"
    CAPACITY = "capacity"
    SECURITY = "security"
    BUSINESS = "business"


class MonitoringStatus(Enum):
    """Monitoring component status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class MetricValue:
    """Metric value with metadata."""

    name: str
    value: float | int | str | bool
    unit: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert with comprehensive details."""

    id: str
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_component: str = ""
    affected_services: list[str] = field(default_factory=list)
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)
    acknowledged: bool = False
    resolved: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Health check result with detailed information."""

    component: str
    status: MonitoringStatus
    response_time_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


class MonitoringConfig(BaseModel):
    """Monitoring system configuration."""

    # Health check intervals
    health_check_interval: int = Field(30, ge=5, le=300)  # seconds
    metric_collection_interval: int = Field(10, ge=1, le=60)  # seconds
    alert_evaluation_interval: int = Field(15, ge=5, le=120)  # seconds

    # Alert thresholds
    response_time_threshold_ms: float = Field(1000.0, ge=10.0, le=30000.0)
    error_rate_threshold: float = Field(0.05, ge=0.0, le=1.0)  # 5%
    cpu_usage_threshold: float = Field(0.80, ge=0.0, le=1.0)  # 80%
    memory_usage_threshold: float = Field(0.85, ge=0.0, le=1.0)  # 85%
    disk_usage_threshold: float = Field(0.90, ge=0.0, le=1.0)  # 90%

    # Data quality thresholds
    min_data_quality_score: float = Field(0.8, ge=0.0, le=1.0)
    max_data_staleness_minutes: int = Field(10, ge=1, le=1440)
    min_data_completeness: float = Field(0.95, ge=0.0, le=1.0)

    # Alerting settings
    enable_email_alerts: bool = False
    enable_slack_alerts: bool = False
    enable_webhook_alerts: bool = False
    alert_cooldown_minutes: int = Field(15, ge=1, le=1440)
    max_alerts_per_hour: int = Field(20, ge=1, le=1000)

    # Retention settings
    metric_retention_days: int = Field(30, ge=1, le=365)
    alert_retention_days: int = Field(90, ge=1, le=365)


class ThresholdRule(BaseModel):
    """Threshold-based alerting rule."""

    name: str = Field(..., min_length=1)
    metric_name: str = Field(..., min_length=1)
    operator: str = Field(..., pattern="^(gt|gte|lt|lte|eq|ne)$")  # greater than, less than, etc.
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    category: AlertCategory = AlertCategory.PERFORMANCE
    enabled: bool = True
    cooldown_minutes: int = Field(15, ge=1, le=1440)
    description: str = ""


class DataMonitor(BaseComponent):
    """
    Comprehensive data monitoring and alerting system.

    Features:
    - Real-time health monitoring of all data components
    - Performance metric collection and analysis
    - Intelligent alerting with configurable thresholds
    - Anomaly detection for data quality issues
    - Dashboard and reporting capabilities
    - Integration with external alerting systems
    """

    def __init__(self, config: Config):
        """Initialize the data monitoring system."""
        super().__init__()
        self.config = config

        # Setup configuration
        self._setup_configuration()

        # Monitoring state
        self._monitored_components: dict[str, Callable] = {}
        self._health_status: dict[str, HealthCheckResult] = {}
        self._metrics: dict[str, list[MetricValue]] = {}
        self._alerts: dict[str, Alert] = {}
        self._threshold_rules: dict[str, ThresholdRule] = {}

        # Alert management
        self._alert_cooldowns: dict[str, datetime] = {}
        self._alert_counts: dict[str, int] = {}  # alerts per hour

        # Background tasks
        self._monitoring_tasks: list[asyncio.Task] = []

        self._initialized = False

    def _setup_configuration(self) -> None:
        """Setup monitoring configuration."""
        monitoring_config = getattr(self.config, "data_monitoring", {})

        self.monitoring_config = MonitoringConfig(
            **monitoring_config if isinstance(monitoring_config, dict) else {}
        )

        self.logger.info(
            f"Data monitoring configured with {self.monitoring_config.health_check_interval}s health check interval"
        )

    async def initialize(self) -> None:
        """Initialize the monitoring system."""
        try:
            if self._initialized:
                return

            self.logger.info("Initializing DataMonitor...")

            # Setup default threshold rules
            await self._setup_default_threshold_rules()

            # Start background monitoring tasks
            await self._start_monitoring_tasks()

            self._initialized = True
            self.logger.info("DataMonitor initialized successfully")

        except Exception as e:
            self.logger.error(f"DataMonitor initialization failed: {e}")
            raise

    async def _setup_default_threshold_rules(self) -> None:
        """Setup default alerting threshold rules."""
        default_rules = [
            ThresholdRule(
                name="high_response_time",
                metric_name="response_time_ms",
                operator="gt",
                threshold=self.monitoring_config.response_time_threshold_ms,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.PERFORMANCE,
                cooldown_minutes=15,
                description="Response time exceeded threshold",
            ),
            ThresholdRule(
                name="high_error_rate",
                metric_name="error_rate",
                operator="gt",
                threshold=self.monitoring_config.error_rate_threshold,
                severity=AlertSeverity.ERROR,
                category=AlertCategory.AVAILABILITY,
                cooldown_minutes=15,
                description="Error rate exceeded threshold",
            ),
            ThresholdRule(
                name="low_data_quality",
                metric_name="data_quality_score",
                operator="lt",
                threshold=self.monitoring_config.min_data_quality_score,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.DATA_QUALITY,
                cooldown_minutes=15,
                description="Data quality score below threshold",
            ),
            ThresholdRule(
                name="high_cpu_usage",
                metric_name="cpu_usage_percent",
                operator="gt",
                threshold=self.monitoring_config.cpu_usage_threshold * 100,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.CAPACITY,
                cooldown_minutes=15,
                description="CPU usage exceeded threshold",
            ),
            ThresholdRule(
                name="high_memory_usage",
                metric_name="memory_usage_percent",
                operator="gt",
                threshold=self.monitoring_config.memory_usage_threshold * 100,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.CAPACITY,
                cooldown_minutes=15,
                description="Memory usage exceeded threshold",
            ),
        ]

        for rule in default_rules:
            self._threshold_rules[rule.name] = rule

        self.logger.info(f"Setup {len(default_rules)} default threshold rules")

    async def _start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks."""
        # Health check task
        health_check_task = asyncio.create_task(self._health_check_loop())
        self._monitoring_tasks.append(health_check_task)

        # Metric collection task
        metric_collection_task = asyncio.create_task(self._metric_collection_loop())
        self._monitoring_tasks.append(metric_collection_task)

        # Alert evaluation task
        alert_evaluation_task = asyncio.create_task(self._alert_evaluation_loop())
        self._monitoring_tasks.append(alert_evaluation_task)

        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._monitoring_tasks.append(cleanup_task)

        self.logger.info(f"Started {len(self._monitoring_tasks)} monitoring background tasks")

    def register_component(self, name: str, health_check_func: Callable) -> None:
        """Register a component for monitoring."""
        self._monitored_components[name] = health_check_func
        self.logger.info(f"Registered component for monitoring: {name}")

    def unregister_component(self, name: str) -> None:
        """Unregister a component from monitoring."""
        if name in self._monitored_components:
            del self._monitored_components[name]
            if name in self._health_status:
                del self._health_status[name]
            self.logger.info(f"Unregistered component from monitoring: {name}")

    async def _health_check_loop(self) -> None:
        """Background task for health checking all components."""
        try:
            while True:
                await self._perform_health_checks()
                await asyncio.sleep(self.monitoring_config.health_check_interval)
        except asyncio.CancelledError:
            self.logger.info("Health check loop cancelled")
        except Exception as e:
            self.logger.error(f"Health check loop error: {e}")

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all registered components."""
        for component_name, health_check_func in self._monitored_components.items():
            try:
                start_time = time.time()

                # Execute health check
                health_data = await health_check_func()

                # Calculate response time
                response_time_ms = (time.time() - start_time) * 1000

                # Process health check result
                status = self._determine_health_status(health_data)

                # Create health check result
                health_result = HealthCheckResult(
                    component=component_name,
                    status=status,
                    response_time_ms=response_time_ms,
                    details=health_data if isinstance(health_data, dict) else {},
                )

                # Add metrics
                health_result.metrics["response_time_ms"] = MetricValue(
                    name="response_time_ms",
                    value=response_time_ms,
                    unit="ms",
                    tags={"component": component_name},
                )

                # Store result
                self._health_status[component_name] = health_result

                # Record metric
                await self._record_metric(
                    f"{component_name}.response_time_ms",
                    response_time_ms,
                    {"component": component_name},
                )

            except Exception as e:
                self.logger.error(f"Health check failed for {component_name}: {e}")

                # Create unhealthy result
                self._health_status[component_name] = HealthCheckResult(
                    component=component_name,
                    status=MonitoringStatus.UNHEALTHY,
                    response_time_ms=0.0,
                    issues=[f"Health check failed: {e}"],
                )

    def _determine_health_status(self, health_data: Any) -> MonitoringStatus:
        """Determine health status from health check data."""
        if isinstance(health_data, dict):
            status_str = health_data.get("status", "").lower()
            if status_str == "healthy":
                return MonitoringStatus.HEALTHY
            elif status_str == "degraded":
                return MonitoringStatus.DEGRADED
            elif status_str in ["unhealthy", "error"]:
                return MonitoringStatus.UNHEALTHY
            else:
                return MonitoringStatus.UNKNOWN
        else:
            # Assume healthy if no specific status
            return MonitoringStatus.HEALTHY

    async def _metric_collection_loop(self) -> None:
        """Background task for metric collection."""
        try:
            while True:
                await self._collect_system_metrics()
                await asyncio.sleep(self.monitoring_config.metric_collection_interval)
        except asyncio.CancelledError:
            self.logger.info("Metric collection loop cancelled")
        except Exception as e:
            self.logger.error(f"Metric collection loop error: {e}")

    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            import psutil

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            await self._record_metric("system.cpu_usage_percent", cpu_percent)

            # Memory metrics
            memory = psutil.virtual_memory()
            await self._record_metric("system.memory_usage_percent", memory.percent)
            await self._record_metric(
                "system.memory_available_mb", memory.available / (1024 * 1024)
            )

            # Disk metrics
            disk = psutil.disk_usage("/")
            disk_usage_percent = (disk.used / disk.total) * 100
            await self._record_metric("system.disk_usage_percent", disk_usage_percent)
            await self._record_metric("system.disk_free_gb", disk.free / (1024 * 1024 * 1024))

        except ImportError:
            # psutil not available, skip system metrics
            pass
        except Exception as e:
            self.logger.error(f"System metric collection failed: {e}")

    async def _record_metric(
        self, name: str, value: float | int, tags: dict[str, str] | None = None
    ) -> None:
        """Record a metric value."""
        metric = MetricValue(
            name=name,
            value=value,
            tags=tags or {},
        )

        if name not in self._metrics:
            self._metrics[name] = []

        self._metrics[name].append(metric)

        # Keep only recent metrics (last hour by default)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        self._metrics[name] = [m for m in self._metrics[name] if m.timestamp > cutoff_time]

    async def _alert_evaluation_loop(self) -> None:
        """Background task for alert evaluation."""
        try:
            while True:
                await self._evaluate_alerts()
                await asyncio.sleep(self.monitoring_config.alert_evaluation_interval)
        except asyncio.CancelledError:
            self.logger.info("Alert evaluation loop cancelled")
        except Exception as e:
            self.logger.error(f"Alert evaluation loop error: {e}")

    async def _evaluate_alerts(self) -> None:
        """Evaluate threshold rules and generate alerts."""
        for rule_name, rule in self._threshold_rules.items():
            if not rule.enabled:
                continue

            try:
                # Check cooldown
                if await self._is_alert_on_cooldown(rule_name):
                    continue

                # Get latest metric value
                latest_metric = await self._get_latest_metric(rule.metric_name)
                if not latest_metric:
                    continue

                # Evaluate threshold
                if await self._evaluate_threshold(latest_metric.value, rule):
                    await self._create_alert(rule, latest_metric)

            except Exception as e:
                self.logger.error(f"Alert evaluation failed for rule {rule_name}: {e}")

    async def _is_alert_on_cooldown(self, rule_name: str) -> bool:
        """Check if alert is on cooldown."""
        if rule_name not in self._alert_cooldowns:
            return False

        cooldown_end = self._alert_cooldowns[rule_name]
        return datetime.now(timezone.utc) < cooldown_end

    async def _get_latest_metric(self, metric_name: str) -> MetricValue | None:
        """Get the latest value for a metric."""
        if metric_name not in self._metrics or not self._metrics[metric_name]:
            return None

        # Return the most recent metric
        return max(self._metrics[metric_name], key=lambda m: m.timestamp)

    async def _evaluate_threshold(self, value: float | int, rule: ThresholdRule) -> bool:
        """Evaluate if a value crosses the threshold."""
        try:
            numeric_value = float(value)
            threshold = rule.threshold

            if rule.operator == "gt":
                return numeric_value > threshold
            elif rule.operator == "gte":
                return numeric_value >= threshold
            elif rule.operator == "lt":
                return numeric_value < threshold
            elif rule.operator == "lte":
                return numeric_value <= threshold
            elif rule.operator == "eq":
                return numeric_value == threshold
            elif rule.operator == "ne":
                return numeric_value != threshold
            else:
                return False

        except (ValueError, TypeError):
            return False

    async def _create_alert(self, rule: ThresholdRule, metric: MetricValue) -> None:
        """Create and process a new alert."""
        alert_id = str(uuid.uuid4())

        alert = Alert(
            id=alert_id,
            severity=rule.severity,
            category=rule.category,
            title=f"{rule.name}: {rule.description}",
            message=f"Metric '{rule.metric_name}' value {metric.value} {rule.operator} {rule.threshold}",
            source_component=metric.tags.get("component", "unknown"),
            metrics={rule.metric_name: metric},
        )

        # Add suggestions based on alert type
        alert.suggestions = await self._generate_alert_suggestions(alert)

        # Store alert
        self._alerts[alert_id] = alert

        # Set cooldown
        cooldown_end = datetime.now(timezone.utc) + timedelta(minutes=rule.cooldown_minutes)
        self._alert_cooldowns[rule.name] = cooldown_end

        # Process alert (send notifications, etc.)
        await self._process_alert(alert)

        self.logger.warning(f"Alert created: {alert.title} - {alert.message}")

    async def _generate_alert_suggestions(self, alert: Alert) -> list[str]:
        """Generate actionable suggestions for an alert."""
        suggestions = []

        if alert.category == AlertCategory.PERFORMANCE:
            suggestions.extend(
                [
                    "Check for high load or resource contention",
                    "Review recent deployments or configuration changes",
                    "Monitor for memory leaks or inefficient queries",
                ]
            )
        elif alert.category == AlertCategory.AVAILABILITY:
            suggestions.extend(
                [
                    "Check service logs for error patterns",
                    "Verify network connectivity and DNS resolution",
                    "Review circuit breaker and retry configurations",
                ]
            )
        elif alert.category == AlertCategory.DATA_QUALITY:
            suggestions.extend(
                [
                    "Validate data sources and feeds",
                    "Check for data pipeline failures",
                    "Review data validation rules and thresholds",
                ]
            )
        elif alert.category == AlertCategory.CAPACITY:
            suggestions.extend(
                [
                    "Consider scaling resources vertically or horizontally",
                    "Review resource allocation and limits",
                    "Check for resource leaks or inefficient usage",
                ]
            )

        return suggestions

    async def _process_alert(self, alert: Alert) -> None:
        """Process a new alert (send notifications, etc.)."""
        # Check alert rate limiting
        hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
        if hour_key not in self._alert_counts:
            self._alert_counts[hour_key] = 0

        if self._alert_counts[hour_key] >= self.monitoring_config.max_alerts_per_hour:
            self.logger.warning(f"Alert rate limit exceeded, skipping alert: {alert.title}")
            return

        self._alert_counts[hour_key] += 1

        # Send notifications based on configuration
        if self.monitoring_config.enable_email_alerts:
            await self._send_email_alert(alert)

        if self.monitoring_config.enable_slack_alerts:
            await self._send_slack_alert(alert)

        if self.monitoring_config.enable_webhook_alerts:
            await self._send_webhook_alert(alert)

    async def _send_email_alert(self, alert: Alert) -> None:
        """Send email alert notification."""
        # Email implementation would go here
        self.logger.info(f"Email alert sent: {alert.title}")

    async def _send_slack_alert(self, alert: Alert) -> None:
        """Send Slack alert notification."""
        # Slack implementation would go here
        self.logger.info(f"Slack alert sent: {alert.title}")

    async def _send_webhook_alert(self, alert: Alert) -> None:
        """Send webhook alert notification."""
        # Webhook implementation would go here
        self.logger.info(f"Webhook alert sent: {alert.title}")

    async def _cleanup_loop(self) -> None:
        """Background task for cleanup of old data."""
        try:
            while True:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Run cleanup every hour
        except asyncio.CancelledError:
            self.logger.info("Cleanup loop cancelled")
        except Exception as e:
            self.logger.error(f"Cleanup loop error: {e}")

    async def _cleanup_old_data(self) -> None:
        """Clean up old metrics and alerts."""
        try:
            current_time = datetime.now(timezone.utc)

            # Cleanup old metrics
            metric_cutoff = current_time - timedelta(
                days=self.monitoring_config.metric_retention_days
            )
            for metric_name in list(self._metrics.keys()):
                self._metrics[metric_name] = [
                    m for m in self._metrics[metric_name] if m.timestamp > metric_cutoff
                ]
                if not self._metrics[metric_name]:
                    del self._metrics[metric_name]

            # Cleanup old alerts
            alert_cutoff = current_time - timedelta(
                days=self.monitoring_config.alert_retention_days
            )
            old_alerts = [
                alert_id
                for alert_id, alert in self._alerts.items()
                if alert.timestamp < alert_cutoff
            ]
            for alert_id in old_alerts:
                del self._alerts[alert_id]

            # Cleanup old cooldowns
            expired_cooldowns = [
                rule_name
                for rule_name, cooldown_end in self._alert_cooldowns.items()
                if current_time > cooldown_end
            ]
            for rule_name in expired_cooldowns:
                del self._alert_cooldowns[rule_name]

            if old_alerts or expired_cooldowns:
                self.logger.debug(
                    f"Cleaned up {len(old_alerts)} old alerts and {len(expired_cooldowns)} expired cooldowns"
                )

        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")

    async def get_system_status(self) -> dict[str, Any]:
        """Get overall system status."""
        # Calculate overall health
        healthy_components = sum(
            1
            for result in self._health_status.values()
            if result.status == MonitoringStatus.HEALTHY
        )
        total_components = len(self._health_status)

        overall_status = MonitoringStatus.HEALTHY
        if total_components == 0:
            overall_status = MonitoringStatus.UNKNOWN
        elif healthy_components == 0:
            overall_status = MonitoringStatus.UNHEALTHY
        elif healthy_components < total_components:
            overall_status = MonitoringStatus.DEGRADED

        # Count active alerts by severity
        active_alerts = [alert for alert in self._alerts.values() if not alert.resolved]
        alert_counts = {
            "critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            "error": len([a for a in active_alerts if a.severity == AlertSeverity.ERROR]),
            "warning": len([a for a in active_alerts if a.severity == AlertSeverity.WARNING]),
            "info": len([a for a in active_alerts if a.severity == AlertSeverity.INFO]),
        }

        return {
            "overall_status": overall_status.value,
            "components": {
                "total": total_components,
                "healthy": healthy_components,
                "degraded": sum(
                    1 for r in self._health_status.values() if r.status == MonitoringStatus.DEGRADED
                ),
                "unhealthy": sum(
                    1
                    for r in self._health_status.values()
                    if r.status == MonitoringStatus.UNHEALTHY
                ),
            },
            "alerts": {
                "total_active": len(active_alerts),
                "by_severity": alert_counts,
            },
            "metrics": {
                "total_metrics": len(self._metrics),
                "total_datapoints": sum(len(values) for values in self._metrics.values()),
            },
            "monitoring": {
                "components_monitored": len(self._monitored_components),
                "threshold_rules": len(self._threshold_rules),
                "monitoring_tasks": len(self._monitoring_tasks),
            },
        }

    async def get_component_health(
        self, component: str | None = None
    ) -> dict[str, HealthCheckResult]:
        """Get health status for components."""
        if component:
            return (
                {component: self._health_status.get(component)}
                if component in self._health_status
                else {}
            )
        return self._health_status.copy()

    async def get_alerts(
        self,
        severity: AlertSeverity | None = None,
        category: AlertCategory | None = None,
        resolved: bool | None = None,
    ) -> list[Alert]:
        """Get alerts with optional filtering."""
        alerts = list(self._alerts.values())

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if category:
            alerts = [a for a in alerts if a.category == category]
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]

        # Sort by timestamp, newest first
        alerts.sort(key=lambda a: a.timestamp, reverse=True)

        return alerts

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert."""
        if alert_id in self._alerts:
            self._alerts[alert_id].acknowledged = True
            self._alerts[alert_id].metadata["acknowledged_by"] = acknowledged_by
            self._alerts[alert_id].metadata["acknowledged_at"] = datetime.now(
                timezone.utc
            ).isoformat()
            self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        return False

    async def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert."""
        if alert_id in self._alerts:
            self._alerts[alert_id].resolved = True
            self._alerts[alert_id].metadata["resolved_by"] = resolved_by
            self._alerts[alert_id].metadata["resolved_at"] = datetime.now(timezone.utc).isoformat()
            self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            return True
        return False

    async def add_threshold_rule(self, rule: ThresholdRule) -> None:
        """Add a new threshold rule."""
        self._threshold_rules[rule.name] = rule
        self.logger.info(f"Added threshold rule: {rule.name}")

    async def remove_threshold_rule(self, rule_name: str) -> bool:
        """Remove a threshold rule."""
        if rule_name in self._threshold_rules:
            del self._threshold_rules[rule_name]
            self.logger.info(f"Removed threshold rule: {rule_name}")
            return True
        return False

    async def health_check(self) -> dict[str, Any]:
        """Perform monitor health check."""
        return {
            "status": "healthy",
            "initialized": self._initialized,
            "monitoring_tasks_running": len([t for t in self._monitoring_tasks if not t.done()]),
            "components_monitored": len(self._monitored_components),
            "active_alerts": len([a for a in self._alerts.values() if not a.resolved]),
            "configuration": {
                "health_check_interval": self.monitoring_config.health_check_interval,
                "metric_collection_interval": self.monitoring_config.metric_collection_interval,
                "alert_evaluation_interval": self.monitoring_config.alert_evaluation_interval,
            },
        }

    async def cleanup(self) -> None:
        """Cleanup monitoring system resources."""
        try:
            # Cancel monitoring tasks
            for task in self._monitoring_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Clear data structures
            self._monitored_components.clear()
            self._health_status.clear()
            self._metrics.clear()
            self._alerts.clear()
            self._threshold_rules.clear()
            self._alert_cooldowns.clear()
            self._alert_counts.clear()

            self._initialized = False
            self.logger.info("DataMonitor cleanup completed")

        except Exception as e:
            self.logger.error(f"DataMonitor cleanup error: {e}")
