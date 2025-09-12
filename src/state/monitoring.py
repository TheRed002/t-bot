"""
Comprehensive Health Monitoring and Metrics for State Management.

This module provides enterprise-grade monitoring capabilities including:
- Real-time health checks and status monitoring
- Performance metrics collection and analysis
- Alert generation and escalation
- SLA compliance monitoring
- Resource utilization tracking
- Predictive health analysis
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

from src.core.base.component import BaseComponent
from src.core.base.interfaces import HealthStatus
from src.core.exceptions import StateConsistencyError

# Import consistent AlertSeverity from core types
from src.core.types import AlertSeverity

from .utils_imports import time_execution

# Import constants through centralized utils import to avoid circular dependencies
try:
    from src.utils.state_constants import (
        ALERT_COOLDOWN_MINUTES,
        DEFAULT_CACHE_TTL,
        HEALTH_CHECK_INTERVAL,
        MEMORY_CRITICAL_THRESHOLD,
        MEMORY_WARNING_THRESHOLD,
        METRICS_COLLECTION_INTERVAL,
        METRICS_RETENTION_HOURS,
    )
except ImportError:
    # Fallback constants in case of circular import
    DEFAULT_CACHE_TTL = 300
    HEALTH_CHECK_INTERVAL = 30
    METRICS_COLLECTION_INTERVAL = 60
    ALERT_COOLDOWN_MINUTES = 10
    METRICS_RETENTION_HOURS = 24
    MEMORY_WARNING_THRESHOLD = 1000  # MB
    MEMORY_CRITICAL_THRESHOLD = 2000  # MB


class MetricType(Enum):
    """Metric type enumeration for state metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class HealthCheck:
    """Health check definition."""

    check_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    check_function: Callable[[], Any] | None = None

    # Configuration
    enabled: bool = True
    interval_seconds: int = 30
    timeout_seconds: int = 10
    failure_threshold: int = 3

    # Status tracking
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check_time: datetime | None = None
    last_success_time: datetime | None = None
    consecutive_failures: int = 0

    # Results
    last_result: dict[str, Any] = field(default_factory=dict)
    last_error: str = ""
    response_time_ms: Decimal = Decimal("0.0")


@dataclass
class Metric:
    """Metric data point."""

    metric_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    metric_type: MetricType = MetricType.GAUGE
    value: Decimal = Decimal("0.0")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Metadata
    tags: dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""


@dataclass
class Alert:
    """Alert notification."""

    alert_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: AlertSeverity = AlertSeverity.MEDIUM

    # Alert details
    title: str = ""
    message: str = ""
    source: str = ""
    category: str = ""

    # Context
    metric_name: str = ""
    current_value: Decimal = Decimal("0.0")
    threshold_value: Decimal = Decimal("0.0")

    # Status
    active: bool = True
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: datetime | None = None


@dataclass
class PerformanceReport:
    """Performance analysis report."""

    report_id: str = field(default_factory=lambda: str(uuid4()))
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_start: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) - timedelta(hours=24)
    )
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Overall metrics
    overall_health: HealthStatus = HealthStatus.UNKNOWN
    uptime_percentage: Decimal = Decimal("100.0")
    availability_sla_met: bool = True

    # Performance metrics
    average_response_time_ms: Decimal = Decimal("0.0")
    peak_response_time_ms: Decimal = Decimal("0.0")
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    error_rate_percentage: Decimal = Decimal("0.0")

    # Resource utilization
    peak_memory_usage_mb: Decimal = Decimal("0.0")
    average_memory_usage_mb: Decimal = Decimal("0.0")
    peak_cpu_usage_percentage: Decimal = Decimal("0.0")
    average_cpu_usage_percentage: Decimal = Decimal("0.0")

    # State-specific metrics
    state_operations_per_second: Decimal = Decimal("0.0")
    cache_hit_rate_percentage: Decimal = Decimal("0.0")
    sync_success_rate_percentage: Decimal = Decimal("0.0")
    recovery_operations: int = 0

    # Alerts and incidents
    total_alerts: int = 0
    critical_alerts: int = 0
    alert_resolution_time_minutes: Decimal = Decimal("0.0")


class StateMonitoringService(BaseComponent):
    """
    Comprehensive monitoring service for state management system with central integration.

    Features:
    - Real-time health monitoring with configurable checks
    - Performance metrics collection and analysis
    - Automated alerting and escalation
    - SLA compliance monitoring
    - Resource utilization tracking
    - Integration with central monitoring infrastructure
    """

    def __init__(self, state_service: Any, metrics_collector: Any = None):  # Type is StateService
        """
        Initialize the monitoring service.

        Args:
            state_service: Reference to the main state service
            metrics_collector: Optional central metrics collector for integration
        """
        super().__init__()
        self.state_service = state_service

        # Central monitoring integration
        self.metrics_collector = metrics_collector
        self._central_alert_manager = None

        # Integration adapters - will be set by monitoring_integration module
        self.metrics_adapter = None
        self.alert_adapter = None

        # Health checks
        self._health_checks: dict[str, HealthCheck] = {}
        self._overall_health: HealthStatus = HealthStatus.UNKNOWN

        # Metrics storage
        self._metrics: dict[str, list[Metric]] = {}
        self._metric_aggregates: dict[str, dict[str, float]] = {}

        # Alerting
        self._alerts: list[Alert] = []
        self._alert_handlers: list[Callable] = []
        self._alert_thresholds: dict[str, dict[str, float | str]] = {}

        # Performance tracking
        self._operation_timers: dict[str, list[float]] = {}
        self._resource_usage: dict[str, list[float]] = {}

        # Configuration
        self.metrics_retention_hours = METRICS_RETENTION_HOURS
        self.health_check_interval_seconds = HEALTH_CHECK_INTERVAL
        self.metrics_collection_interval_seconds = METRICS_COLLECTION_INTERVAL
        self.alert_cooldown_minutes = ALERT_COOLDOWN_MINUTES
        self.sla_availability_target = Decimal("99.9")  # 99.9% uptime SLA target
        self.sla_response_time_target_ms = Decimal("100.0")  # 100ms response time SLA

        # Background tasks
        self._health_check_task: asyncio.Task | None = None
        self._metrics_collection_task: asyncio.Task | None = None
        self._alert_processing_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._alert_tasks: list[asyncio.Task] = []  # Store alert checking tasks
        self._running = False

        # Initialize built-in health checks
        self._initialize_builtin_health_checks()

        # Initialize built-in metrics
        self._initialize_builtin_metrics()

        # Setup central monitoring integration if available
        if self.metrics_collector:
            self._setup_central_integration()

        self.logger.info("StateMonitoringService initialized")

    async def initialize(self) -> None:
        """Initialize the monitoring service."""
        try:
            # Start background tasks with proper connection monitoring
            self._running = True
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())
            self._alert_processing_task = asyncio.create_task(self._alert_processing_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Initial health check
            await self._run_all_health_checks()

            await super().initialize()
            self.logger.info("StateMonitoringService initialization completed")

        except Exception as e:
            self.logger.error(f"StateMonitoringService initialization failed: {e}")
            raise StateConsistencyError(f"Failed to initialize StateMonitoringService: {e}") from e

    async def cleanup(self) -> None:
        """Cleanup monitoring service resources."""
        try:
            self._running = False

            # Cancel and cleanup background tasks
            background_tasks = [
                self._health_check_task,
                self._metrics_collection_task,
                self._alert_processing_task,
                self._cleanup_task,
                self._heartbeat_task,
            ]

            # Clear task references immediately
            self._health_check_task = None
            self._metrics_collection_task = None
            self._alert_processing_task = None
            self._cleanup_task = None
            self._heartbeat_task = None

            for task in background_tasks:
                if task and not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except asyncio.CancelledError:
                        pass
                    except asyncio.TimeoutError:
                        self.logger.warning("Background task cleanup timeout")
                    except Exception as e:
                        self.logger.error(f"Error waiting for background task cleanup: {e}")
                    finally:
                        # Ensure task reference is cleared
                        task = None

            # Cleanup alert tasks
            alert_tasks = self._alert_tasks.copy()
            self._alert_tasks.clear()

            for task in alert_tasks:
                if task and not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except asyncio.CancelledError:
                        pass
                    except asyncio.TimeoutError:
                        self.logger.warning("Alert task cleanup timeout")
                    except Exception as e:
                        self.logger.error(f"Error waiting for alert task cleanup: {e}")
                    finally:
                        # Ensure task reference is cleared
                        task = None

            # Generate final performance report
            await self.generate_performance_report()

            await super().cleanup()
            self.logger.info("StateMonitoringService cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during StateMonitoringService cleanup: {e}")
            raise

    # Health Check Operations

    def register_health_check(
        self,
        name: str,
        check_function: Callable,
        description: str = "",
        interval_seconds: int = 30,
        timeout_seconds: int = 10,
        failure_threshold: int = 3,
    ) -> str:
        """
        Register a custom health check.

        Args:
            name: Health check name
            check_function: Function to execute for health check
            description: Description of the health check
            interval_seconds: Check interval
            timeout_seconds: Check timeout
            failure_threshold: Consecutive failures before marking unhealthy

        Returns:
            Health check ID
        """
        health_check = HealthCheck(
            name=name,
            description=description,
            check_function=check_function,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
            failure_threshold=failure_threshold,
        )

        self._health_checks[health_check.check_id] = health_check
        self.logger.info(f"Registered health check: {name}")

        return health_check.check_id

    async def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status."""
        try:
            # Run health checks if needed
            await self._run_all_health_checks()

            # Aggregate health status
            health_statuses = [
                check.status for check in self._health_checks.values() if check.enabled
            ]

            if not health_statuses:
                overall_status = HealthStatus.UNKNOWN
            elif any(status == HealthStatus.CRITICAL for status in health_statuses):
                overall_status = HealthStatus.CRITICAL
            elif any(status == HealthStatus.UNHEALTHY for status in health_statuses):
                overall_status = HealthStatus.UNHEALTHY
            elif any(status == HealthStatus.DEGRADED for status in health_statuses):
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.HEALTHY

            self._overall_health = overall_status

            # Build detailed status
            status: dict[str, Any] = {
                "overall_status": overall_status.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checks": {},
                "metrics": await self._get_key_metrics(),
                "alerts": {
                    "active_count": len([a for a in self._alerts if a.active]),
                    "critical_count": len(
                        [
                            a
                            for a in self._alerts
                            if a.active and a.severity == AlertSeverity.CRITICAL
                        ]
                    ),
                },
            }

            # Add individual health check results
            for _check_id, check in self._health_checks.items():
                if check.enabled:
                    status["checks"][check.name] = {
                        "status": check.status.value,
                        "last_check": (
                            check.last_check_time.isoformat() if check.last_check_time else None
                        ),
                        "response_time_ms": check.response_time_ms,
                        "consecutive_failures": check.consecutive_failures,
                        "last_error": check.last_error,
                    }

            return status

        except Exception as e:
            self.logger.error(f"Failed to get health status: {e}")
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    # Metrics Operations

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: dict[str, str] | None = None,
        unit: str = "",
    ) -> None:
        """
        Record a metric value to both local and central monitoring.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Metric tags
            unit: Unit of measurement
        """
        try:
            metric = Metric(
                name=name, metric_type=metric_type, value=value, tags=tags or {}, unit=unit
            )

            if name not in self._metrics:
                self._metrics[name] = []

            self._metrics[name].append(metric)

            # Update aggregates
            self._update_metric_aggregates(name, value)

            # Use adapter if available (for integrated services), otherwise use direct central monitoring
            if self.metrics_adapter:
                self.metrics_adapter.record_state_metric(metric)
            else:
                self._record_to_central_monitoring(metric)

            # Check for alerts (schedule task but don't wait for it)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    task = asyncio.create_task(self._check_metric_alerts(name, value))
                    self._alert_tasks.append(task)
            except RuntimeError:
                # No event loop running, skip alert checking
                pass

        except Exception as e:
            self.logger.error(f"Failed to record metric {name}: {e}")

    def record_operation_time(self, operation_name: str, duration_ms: float) -> None:
        """Record operation execution time."""
        if operation_name not in self._operation_timers:
            self._operation_timers[operation_name] = []

        self._operation_timers[operation_name].append(duration_ms)

        # Trim old values
        if len(self._operation_timers[operation_name]) > 1000:
            self._operation_timers[operation_name] = self._operation_timers[operation_name][-500:]

        # Use adapter if available (for integrated services), otherwise record as metric
        if self.metrics_adapter:
            self.metrics_adapter.record_operation_time(operation_name, duration_ms)
        else:
            # Record as metric
            self.record_metric(
                f"operation_time_{operation_name}", duration_ms, MetricType.TIMER, unit="ms"
            )

    def get_metrics(self) -> dict[str, Any]:
        """Get current component metrics (override of BaseComponent method)."""
        return super().get_metrics()

    async def get_filtered_metrics(
        self,
        metric_names: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, list[Metric]]:
        """
        Get metrics with optional filtering.

        Args:
            metric_names: Specific metric names to retrieve
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            Dictionary of metric data
        """
        try:
            filtered_metrics = {}

            metrics_to_check = metric_names or list(self._metrics.keys())

            for name in metrics_to_check:
                if name not in self._metrics:
                    continue

                metrics = self._metrics[name]

                # Apply time filters
                if start_time or end_time:
                    filtered = []
                    for metric in metrics:
                        if start_time and metric.timestamp < start_time:
                            continue
                        if end_time and metric.timestamp > end_time:
                            continue
                        filtered.append(metric)
                    metrics = filtered

                filtered_metrics[name] = metrics

            return filtered_metrics

        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            return {}

    # Alert Operations

    def set_alert_threshold(
        self,
        metric_name: str,
        warning_threshold: float | None = None,
        critical_threshold: float | None = None,
        comparison: str = "greater_than",  # greater_than, less_than, equals
    ) -> None:
        """
        Set alert thresholds for a metric.

        Args:
            metric_name: Name of the metric
            warning_threshold: Warning threshold value
            critical_threshold: Critical threshold value
            comparison: Comparison operator
        """
        if metric_name not in self._alert_thresholds:
            self._alert_thresholds[metric_name] = {}

        thresholds = self._alert_thresholds[metric_name]

        if warning_threshold is not None:
            thresholds["warning"] = warning_threshold
        if critical_threshold is not None:
            thresholds["critical"] = critical_threshold

        thresholds["comparison"] = comparison

        self.logger.info(f"Set alert thresholds for {metric_name}")

    def register_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register a callback for alert notifications."""
        self._alert_handlers.append(handler)

    async def get_active_alerts(self) -> list[Alert]:
        """Get list of active alerts."""
        return [alert for alert in self._alerts if alert.active and not alert.resolved]

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "") -> bool:
        """Acknowledge an alert."""
        try:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                    return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to acknowledge alert: {e}")
            return False

    # Performance Analysis

    @time_execution
    async def generate_performance_report(
        self, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> PerformanceReport:
        """
        Generate comprehensive performance report.

        Args:
            start_time: Report start time (default: 24 hours ago)
            end_time: Report end time (default: now)

        Returns:
            Performance report
        """
        try:
            if not end_time:
                end_time = datetime.now(timezone.utc)
            if not start_time:
                start_time = end_time - timedelta(hours=24)

            report = PerformanceReport(period_start=start_time, period_end=end_time)

            # Calculate overall health and uptime
            report.overall_health = self._overall_health
            report.uptime_percentage = await self._calculate_uptime(start_time, end_time)
            report.availability_sla_met = report.uptime_percentage >= self.sla_availability_target

            # Calculate performance metrics
            await self._calculate_performance_metrics(report, start_time, end_time)

            # Calculate resource utilization
            await self._calculate_resource_metrics(report, start_time, end_time)

            # Calculate state-specific metrics
            await self._calculate_state_metrics(report, start_time, end_time)

            # Calculate alert metrics
            await self._calculate_alert_metrics(report, start_time, end_time)

            self.logger.info(f"Generated performance report: {report.report_id}")
            return report

        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            # Return empty report with error indication
            report = PerformanceReport()
            report.overall_health = HealthStatus.UNKNOWN
            return report

    # Private Helper Methods

    def _initialize_builtin_health_checks(self) -> None:
        """Initialize built-in health checks."""

        # State service connectivity check
        self.register_health_check(
            "state_service_connectivity",
            self._check_state_service_connectivity,
            "Verify state service is responsive",
            interval_seconds=30,
        )

        # Database connectivity check
        self.register_health_check(
            "database_connectivity",
            self._check_database_connectivity,
            "Verify database connectivity",
            interval_seconds=60,
        )

        # Cache connectivity check
        self.register_health_check(
            "cache_connectivity",
            self._check_cache_connectivity,
            "Verify Redis cache connectivity",
            interval_seconds=60,
        )

        # Memory usage check
        self.register_health_check(
            "memory_usage",
            self._check_memory_usage,
            "Monitor memory usage levels",
            interval_seconds=30,
        )

        # Error rate check
        self.register_health_check(
            "error_rate", self._check_error_rate, "Monitor error rate levels", interval_seconds=60
        )

    def _initialize_builtin_metrics(self) -> None:
        """Initialize built-in metrics and thresholds."""

        # Set default alert thresholds
        self.set_alert_threshold(
            "memory_usage_mb",
            warning_threshold=MEMORY_WARNING_THRESHOLD,
            critical_threshold=MEMORY_CRITICAL_THRESHOLD,
        )
        self.set_alert_threshold(
            "error_rate_percentage", warning_threshold=Decimal("5.0"), critical_threshold=Decimal("10.0")
        )
        self.set_alert_threshold("response_time_ms", warning_threshold=200, critical_threshold=500)
        self.set_alert_threshold(
            "cache_hit_rate",
            warning_threshold=Decimal("80.0"),
            critical_threshold=Decimal("50.0"),
            comparison="less_than",
        )

    async def _check_state_service_connectivity(self) -> dict[str, Any]:
        """Check state service connectivity."""
        try:
            # Try to get health status from state service with timeout and backpressure handling
            if hasattr(self.state_service, "get_health_status"):
                try:
                    status = await asyncio.wait_for(
                        self.state_service.get_health_status(), timeout=10.0
                    )
                    return {"status": "healthy", "response": status}
                except asyncio.TimeoutError:
                    return {"status": "degraded", "error": "Health check timeout - possible backpressure"}
                except asyncio.CancelledError:
                    return {"status": "degraded", "error": "Health check cancelled"}
            else:
                return {"status": "healthy", "message": "State service accessible"}

        except asyncio.TimeoutError:
            return {"status": "unhealthy", "error": "State service health check timeout"}
        except asyncio.CancelledError:
            return {"status": "degraded", "error": "State service health check cancelled"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _check_database_connectivity(self) -> dict[str, Any]:
        """Check database connectivity through service layer abstraction."""
        try:
            # Use service layer abstraction - avoid direct access to database_service
            if hasattr(self.state_service, "_persistence_service") and self.state_service._persistence_service:
                try:
                    # Check through persistence service which handles database access
                    if hasattr(self.state_service._persistence_service, "health_check"):
                        health_status = await asyncio.wait_for(
                            self.state_service._persistence_service.health_check(), timeout=15.0
                        )
                        return {"status": "healthy" if health_status == "healthy" else "degraded"}
                    else:
                        # Fallback to checking if service is available
                        return {"status": "healthy", "message": "Persistence service available"}
                except asyncio.TimeoutError:
                    return {"status": "degraded", "error": "Database health check timeout - possible connection pool exhaustion"}
                except asyncio.CancelledError:
                    return {"status": "degraded", "error": "Database health check cancelled"}
                except ConnectionError as e:
                    return {"status": "unhealthy", "error": f"Database connection error: {e}"}
            else:
                return {"status": "degraded", "message": "Persistence service not available"}

        except asyncio.TimeoutError:
            return {"status": "unhealthy", "error": "Database health check timeout"}
        except asyncio.CancelledError:
            return {"status": "degraded", "error": "Database health check cancelled"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _check_cache_connectivity(self) -> dict[str, Any]:
        """Check cache connectivity through service layer abstraction."""
        try:
            # Use service layer abstraction - avoid direct access to redis_client
            if hasattr(self.state_service, "_persistence_service") and self.state_service._persistence_service:
                try:
                    # Check through persistence service which handles cache access
                    if hasattr(self.state_service._persistence_service, "ping_cache"):
                        cache_status = await asyncio.wait_for(
                            self.state_service._persistence_service.ping_cache(), timeout=5.0
                        )
                        return {"status": "healthy" if cache_status else "unhealthy"}
                    else:
                        # Fallback to checking if service has cache capabilities
                        return {"status": "healthy", "message": "Cache service available through persistence"}
                except asyncio.TimeoutError:
                    return {"status": "degraded", "error": "Cache ping timeout - possible network congestion"}
                except asyncio.CancelledError:
                    return {"status": "degraded", "error": "Cache ping cancelled"}
                except ConnectionError as e:
                    return {"status": "unhealthy", "error": f"Cache connection error: {e}"}
            else:
                return {"status": "degraded", "message": "Cache service not available through persistence layer"}

        except asyncio.TimeoutError:
            return {"status": "unhealthy", "error": "Cache ping timeout"}
        except asyncio.CancelledError:
            return {"status": "degraded", "error": "Cache connectivity check cancelled"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _check_memory_usage(self) -> dict[str, Any]:
        """Check memory usage levels."""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            self.record_metric("memory_usage_mb", memory_mb, unit="MB")

            if memory_mb > MEMORY_CRITICAL_THRESHOLD:
                return {"status": "critical", "memory_mb": memory_mb}
            elif memory_mb > MEMORY_WARNING_THRESHOLD:
                return {"status": "degraded", "memory_mb": memory_mb}
            else:
                return {"status": "healthy", "memory_mb": memory_mb}

        except Exception as e:
            return {"status": "unknown", "error": str(e)}

    async def _check_error_rate(self) -> dict[str, Any]:
        """Check error rate levels."""
        try:
            # Get metrics from state service
            metrics = await self.state_service.get_metrics()

            if hasattr(metrics, "error_rate") and metrics.error_rate is not None:
                error_rate = metrics.error_rate * 100  # Convert to percentage

                self.record_metric("error_rate_percentage", error_rate, unit="%")

                if error_rate > 10:
                    return {"status": "critical", "error_rate": error_rate}
                elif error_rate > 5:
                    return {"status": "degraded", "error_rate": error_rate}
                else:
                    return {"status": "healthy", "error_rate": error_rate}
            else:
                return {"status": "unknown", "message": "Error rate not available"}

        except Exception as e:
            return {"status": "unknown", "error": str(e)}

    async def _run_all_health_checks(self) -> None:
        """Run all enabled health checks."""
        current_time = datetime.now(timezone.utc)

        for check in self._health_checks.values():
            if not check.enabled:
                continue

            # Check if it's time to run this check
            if (
                check.last_check_time
                and (current_time - check.last_check_time).total_seconds() < check.interval_seconds
            ):
                continue

            await self._run_health_check(check)

    async def _run_health_check(self, check: HealthCheck) -> None:
        """Run a single health check."""
        try:
            start_time = time.time()
            check.last_check_time = datetime.now(timezone.utc)

            # Execute check with timeout
            try:
                if check.check_function is None:
                    check.status = HealthStatus.UNKNOWN
                    check.last_error = "No check function configured"
                    return

                result = await asyncio.wait_for(
                    check.check_function(), timeout=check.timeout_seconds
                )

                check.response_time_ms = (time.time() - start_time) * 1000
                check.last_result = result

                # Determine status from result
                if isinstance(result, dict):
                    status_str = result.get("status", "unknown").lower()
                    if status_str == "healthy":
                        check.status = HealthStatus.HEALTHY
                        check.consecutive_failures = 0
                        check.last_success_time = check.last_check_time
                        check.last_error = ""
                    elif status_str == "degraded":
                        check.status = HealthStatus.DEGRADED
                        check.consecutive_failures = 0
                        check.last_success_time = check.last_check_time
                        check.last_error = ""
                    elif status_str == "critical":
                        check.status = HealthStatus.CRITICAL
                        check.consecutive_failures += 1
                        check.last_error = result.get("error", "Critical status")
                    else:
                        check.status = HealthStatus.UNHEALTHY
                        check.consecutive_failures += 1
                        check.last_error = result.get("error", "Unhealthy status")
                else:
                    # Boolean result
                    if result:
                        check.status = HealthStatus.HEALTHY
                        check.consecutive_failures = 0
                        check.last_success_time = check.last_check_time
                        check.last_error = ""
                    else:
                        check.status = HealthStatus.UNHEALTHY
                        check.consecutive_failures += 1
                        check.last_error = "Check returned False"

            except asyncio.TimeoutError:
                check.status = HealthStatus.UNHEALTHY
                check.consecutive_failures += 1
                check.last_error = f"Health check timed out after {check.timeout_seconds}s"
                check.response_time_ms = check.timeout_seconds * 1000

            # Check failure threshold
            if check.consecutive_failures >= check.failure_threshold:
                if check.status not in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
                    check.status = HealthStatus.UNHEALTHY

        except Exception as e:
            check.status = HealthStatus.UNHEALTHY
            check.consecutive_failures += 1
            check.last_error = str(e)
            self.logger.error(f"Health check {check.name} failed: {e}")

        # Record health check status via adapter if available
        if self.metrics_adapter:
            self.metrics_adapter.record_health_check(check.name, check.status)

    def _update_metric_aggregates(self, name: str, value: float) -> None:
        """Update metric aggregates for performance."""
        if name not in self._metric_aggregates:
            self._metric_aggregates[name] = {
                "count": 0,
                "sum": Decimal("0.0"),
                "min": Decimal("inf"),
                "max": Decimal("-inf"),
                "avg": Decimal("0.0"),
            }

        agg = self._metric_aggregates[name]
        decimal_value = Decimal(str(value))
        agg["count"] += 1
        agg["sum"] += decimal_value
        agg["min"] = min(agg["min"], decimal_value)
        agg["max"] = max(agg["max"], decimal_value)
        agg["avg"] = agg["sum"] / agg["count"]

    async def _check_metric_alerts(self, metric_name: str, value: float) -> None:
        """Check if metric value triggers any alerts."""
        try:
            if metric_name not in self._alert_thresholds:
                return

            thresholds = self._alert_thresholds[metric_name]
            comparison = str(thresholds.get("comparison", "greater_than"))

            # Check critical threshold
            critical_threshold = thresholds.get("critical")
            if critical_threshold is not None and isinstance(critical_threshold, (int, float)):
                triggered = self._evaluate_threshold(value, Decimal(str(critical_threshold)), comparison)
                if triggered:
                    await self._create_alert(
                        AlertSeverity.CRITICAL,
                        f"Critical threshold exceeded for {metric_name}",
                        f"Metric {metric_name} value {value} exceeded critical "
                        f"threshold {critical_threshold}",
                        metric_name,
                        value,
                        critical_threshold,
                    )
                    return

            # Check warning threshold
            warning_threshold = thresholds.get("warning")
            if warning_threshold is not None and isinstance(warning_threshold, (int, float)):
                triggered = self._evaluate_threshold(value, Decimal(str(warning_threshold)), comparison)
                if triggered:
                    await self._create_alert(
                        AlertSeverity.MEDIUM,
                        f"Warning threshold exceeded for {metric_name}",
                        f"Metric {metric_name} value {value} exceeded warning "
                        f"threshold {warning_threshold}",
                        metric_name,
                        value,
                        warning_threshold,
                    )

        except Exception as e:
            self.logger.error(f"Failed to check alerts for metric {metric_name}: {e}")

    def _evaluate_threshold(self, value: float, threshold: float, comparison: str) -> bool:
        """Evaluate if value meets threshold condition."""
        if comparison == "greater_than":
            return value > threshold
        elif comparison == "less_than":
            return value < threshold
        elif comparison == "equals":
            return abs(value - threshold) < Decimal("0.001")
        else:
            return False

    async def _create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        metric_name: str,
        current_value: float,
        threshold_value: float,
    ) -> None:
        """Create and process a new alert."""
        try:
            # Check for recent duplicate alerts (cooldown)
            recent_alerts = [
                a
                for a in self._alerts
                if (
                    a.metric_name == metric_name
                    and a.severity == severity
                    and (datetime.now(timezone.utc) - a.timestamp).total_seconds()
                    < self.alert_cooldown_minutes * 60
                )
            ]

            if recent_alerts:
                return  # Skip duplicate alert during cooldown

            alert = Alert(
                severity=severity,
                title=title,
                message=message,
                source="StateMonitoringService",
                category="metric_threshold",
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=threshold_value,
            )

            self._alerts.append(alert)

            # Notify handlers
            for handler in self._alert_handlers:
                try:
                    await handler(alert)
                except Exception as e:
                    self.logger.error(f"Alert handler failed: {e}")

            self.logger.warning(f"Alert created: {alert.title}")

        except Exception as e:
            self.logger.error(f"Failed to create alert: {e}")

    async def _get_key_metrics(self) -> dict[str, Any]:
        """Get summary of key metrics."""
        try:
            key_metrics = {}

            # Get latest values for key metrics
            for metric_name in [
                "memory_usage_mb",
                "error_rate_percentage",
                "response_time_ms",
                "cache_hit_rate",
            ]:
                if metric_name in self._metric_aggregates:
                    agg = self._metric_aggregates[metric_name]
                    key_metrics[metric_name] = {
                        "current": agg["avg"],
                        "min": agg["min"],
                        "max": agg["max"],
                        "count": agg["count"],
                    }

            # Add state service metrics if available
            try:
                state_metrics = await self.state_service.get_metrics()
                if state_metrics:
                    key_metrics["state_service"] = {
                        "active_states": getattr(state_metrics, "active_states_count", 0),
                        "cache_hit_rate": getattr(state_metrics, "cache_hit_rate", Decimal("0.0")),
                        "error_rate": getattr(state_metrics, "error_rate", Decimal("0.0")),
                    }
            except Exception as e:
                self.logger.error(f"Failed to get state service metrics: {e}")
                pass

            return key_metrics

        except Exception as e:
            self.logger.error(f"Failed to get key metrics: {e}")
            return {}

    async def _calculate_uptime(self, start_time: datetime, end_time: datetime) -> float:
        """Calculate uptime percentage for the period."""
        try:
            # This is a simplified calculation
            # In a real implementation, you'd track actual downtime periods
            total_seconds = (end_time - start_time).total_seconds()

            # Count unhealthy periods (simplified)
            unhealthy_seconds = 0
            for check in self._health_checks.values():
                if check.status == HealthStatus.UNHEALTHY:
                    # Assume each failure represents 1 minute of downtime
                    unhealthy_seconds += check.consecutive_failures * 60

            uptime_seconds = max(0, total_seconds - unhealthy_seconds)
            return (Decimal(str(uptime_seconds)) / Decimal(str(total_seconds))) * Decimal("100") if total_seconds > 0 else Decimal("100.0")

        except Exception as e:
            self.logger.error(f"Failed to calculate uptime: {e}")
            return Decimal("100.0")

    async def _calculate_performance_metrics(
        self, report: PerformanceReport, start_time: datetime, end_time: datetime
    ) -> None:
        """Calculate performance metrics for the report."""
        try:
            # Calculate operation metrics
            total_ops = 0
            successful_ops = 0
            response_times = []

            for _operation_name, times in self._operation_timers.items():
                total_ops += len(times)
                successful_ops += len(times)  # Assume recorded times are successful
                response_times.extend(times)

            report.total_operations = total_ops
            report.successful_operations = successful_ops
            report.failed_operations = total_ops - successful_ops

            if total_ops > 0:
                report.error_rate_percentage = (report.failed_operations / total_ops) * 100

            if response_times:
                report.average_response_time_ms = sum(response_times) / len(response_times)
                report.peak_response_time_ms = max(response_times)

        except Exception as e:
            self.logger.error(f"Failed to calculate performance metrics: {e}")

    async def _calculate_resource_metrics(
        self, report: PerformanceReport, start_time: datetime, end_time: datetime
    ) -> None:
        """Calculate resource utilization metrics."""
        try:
            # Get memory usage metrics
            if "memory_usage_mb" in self._metric_aggregates:
                agg = self._metric_aggregates["memory_usage_mb"]
                report.average_memory_usage_mb = agg["avg"]
                report.peak_memory_usage_mb = agg["max"]

            # CPU usage would be calculated similarly if tracked

        except Exception as e:
            self.logger.error(f"Failed to calculate resource metrics: {e}")

    async def _calculate_state_metrics(
        self, report: PerformanceReport, start_time: datetime, end_time: datetime
    ) -> None:
        """Calculate state-specific metrics."""
        try:
            # Get state service metrics
            state_metrics = await self.state_service.get_metrics()
            if state_metrics:
                report.cache_hit_rate_percentage = (
                    getattr(state_metrics, "cache_hit_rate", Decimal("0.0")) * Decimal("100")
                )

                # Calculate operations per second
                period_seconds = (end_time - start_time).total_seconds()
                if period_seconds > 0:
                    total_operations = getattr(state_metrics, "total_operations", 0)
                    report.state_operations_per_second = total_operations / period_seconds

        except Exception as e:
            self.logger.error(f"Failed to calculate state metrics: {e}")

    async def _calculate_alert_metrics(
        self, report: PerformanceReport, start_time: datetime, end_time: datetime
    ) -> None:
        """Calculate alert-related metrics."""
        try:
            period_alerts = [a for a in self._alerts if start_time <= a.timestamp <= end_time]

            report.total_alerts = len(period_alerts)
            report.critical_alerts = len(
                [a for a in period_alerts if a.severity == AlertSeverity.CRITICAL]
            )

            # Calculate average resolution time
            resolved_alerts = [a for a in period_alerts if a.resolved and a.resolved_at]
            if resolved_alerts:
                resolution_times = [
                    (a.resolved_at - a.timestamp).total_seconds() / 60
                    for a in resolved_alerts
                    if a.resolved_at is not None
                ]
                report.alert_resolution_time_minutes = sum(resolution_times) / len(resolution_times)

        except Exception as e:
            self.logger.error(f"Failed to calculate alert metrics: {e}")

    # Background Task Loops

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                await self._run_all_health_checks()
                await asyncio.sleep(self.health_check_interval_seconds)

            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.health_check_interval_seconds)

    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop."""
        while self._running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()

                # Collect state service metrics
                await self._collect_state_service_metrics()

                await asyncio.sleep(self.metrics_collection_interval_seconds)

            except Exception as e:
                self.logger.error(f"Metrics collection loop error: {e}")
                await asyncio.sleep(self.metrics_collection_interval_seconds)

    async def _alert_processing_loop(self) -> None:
        """Background alert processing loop."""
        while self._running:
            try:
                # Auto-resolve alerts that are no longer active
                await self._auto_resolve_alerts()

                # Clean up old alerts
                await self._cleanup_old_alerts()

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Alert processing loop error: {e}")
                await asyncio.sleep(60)

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._running:
            try:
                # Clean up old metrics
                await self._cleanup_old_metrics()

                await asyncio.sleep(3600)  # Every hour

            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)

    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            import psutil

            # Memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            self.record_metric("memory_usage_mb", memory_info.rss / 1024 / 1024, unit="MB")

            # CPU usage
            cpu_percent = process.cpu_percent()
            self.record_metric("cpu_usage_percentage", cpu_percent, unit="%")

        except Exception as e:
            self.logger.debug(f"Failed to collect system metrics: {e}")

    async def _collect_state_service_metrics(self) -> None:
        """Collect state service metrics."""
        try:
            metrics = await self.state_service.get_metrics()
            if metrics:
                # Record key state service metrics
                if hasattr(metrics, "cache_hit_rate"):
                    self.record_metric("cache_hit_rate", metrics.cache_hit_rate * 100, unit="%")

                if hasattr(metrics, "error_rate"):
                    self.record_metric("error_rate_percentage", metrics.error_rate * 100, unit="%")

                if hasattr(metrics, "active_states_count"):
                    self.record_metric("active_states_count", metrics.active_states_count)

        except Exception as e:
            self.logger.debug(f"Failed to collect state service metrics: {e}")

    async def _auto_resolve_alerts(self) -> None:
        """Auto-resolve alerts that are no longer active."""
        try:
            current_time = datetime.now(timezone.utc)

            for alert in self._alerts:
                if alert.active and not alert.resolved:
                    # Check if the condition that triggered the alert is still true
                    # This is a simplified implementation
                    if alert.metric_name in self._metric_aggregates:
                        current_value = self._metric_aggregates[alert.metric_name]["avg"]

                        # If current value is now within acceptable range, resolve alert
                        if alert.metric_name in self._alert_thresholds:
                            thresholds = self._alert_thresholds[alert.metric_name]
                            comparison = str(thresholds.get("comparison", "greater_than"))

                            threshold = None
                            if alert.severity == AlertSeverity.CRITICAL:
                                threshold = thresholds.get("critical")
                            elif alert.severity == AlertSeverity.MEDIUM:
                                threshold = thresholds.get("warning")

                            if threshold is not None and isinstance(threshold, (int, float)):
                                triggered = self._evaluate_threshold(
                                    current_value, Decimal(str(threshold)), comparison
                                )
                                if not triggered:
                                    alert.resolved = True
                                    alert.resolved_at = current_time
                                    self.logger.info(f"Auto-resolved alert: {alert.alert_id}")

        except Exception as e:
            self.logger.error(f"Auto-resolve alerts failed: {e}")

    async def _cleanup_old_alerts(self) -> None:
        """Clean up old resolved alerts."""
        try:
            current_time = datetime.now(timezone.utc)
            cutoff_time = current_time - timedelta(days=7)

            self._alerts = [
                alert
                for alert in self._alerts
                if not alert.resolved
                or (alert.resolved_at is not None and alert.resolved_at > cutoff_time)
            ]

        except Exception as e:
            self.logger.error(f"Alert cleanup failed: {e}")

    async def _cleanup_old_metrics(self) -> None:
        """Clean up old metric data."""
        try:
            current_time = datetime.now(timezone.utc)
            cutoff_time = current_time - timedelta(hours=self.metrics_retention_hours)

            for metric_name in list(self._metrics.keys()):
                self._metrics[metric_name] = [
                    metric
                    for metric in self._metrics[metric_name]
                    if metric.timestamp > cutoff_time
                ]

                # Remove empty metric lists
                if not self._metrics[metric_name]:
                    del self._metrics[metric_name]
                    if metric_name in self._metric_aggregates:
                        del self._metric_aggregates[metric_name]

            # Clean up operation timers - keep only last 1000 entries
            for operation_name in list(self._operation_timers.keys()):
                if len(self._operation_timers[operation_name]) > 1000:
                    self._operation_timers[operation_name] = self._operation_timers[operation_name][
                        -1000:
                    ]

            # Clean up resource usage - keep only last 1000 entries
            for resource_name in list(self._resource_usage.keys()):
                if len(self._resource_usage[resource_name]) > 1000:
                    self._resource_usage[resource_name] = self._resource_usage[resource_name][
                        -1000:
                    ]

        except Exception as e:
            self.logger.error(f"Metrics cleanup failed: {e}")

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop for connection monitoring."""
        heartbeat_interval = 30  # 30 seconds heartbeat
        while self._running:
            try:
                # Send heartbeat to all monitored connections
                await self._send_heartbeat()
                await asyncio.sleep(heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(heartbeat_interval)

    async def _send_heartbeat(self) -> None:
        """Send heartbeat signals to monitored connections."""
        try:
            # Heartbeat through service layer - avoid direct infrastructure access
            if hasattr(self.state_service, "_persistence_service") and self.state_service._persistence_service:
                try:
                    # Use persistence service for database heartbeat
                    if hasattr(self.state_service._persistence_service, "health_check"):
                        await asyncio.wait_for(
                            self.state_service._persistence_service.health_check(),
                            timeout=5.0
                        )
                except (asyncio.TimeoutError, Exception) as e:
                    self.logger.debug(f"Persistence service heartbeat failed: {e}")

                try:
                    # Use persistence service for cache heartbeat
                    if hasattr(self.state_service._persistence_service, "ping_cache"):
                        await asyncio.wait_for(
                            self.state_service._persistence_service.ping_cache(),
                            timeout=3.0
                        )
                except (asyncio.TimeoutError, Exception) as e:
                    self.logger.debug(f"Cache service heartbeat failed: {e}")

        except Exception as e:
            self.logger.debug(f"Heartbeat operation failed: {e}")

    # Central Monitoring Integration Methods

    def _record_to_central_monitoring(self, metric: Metric) -> None:
        """Record metric to central monitoring system if available."""
        try:
            if not self.metrics_collector:
                return

            # Map metric type to central monitoring format
            metric_type_map = {
                MetricType.COUNTER: "counter",
                MetricType.GAUGE: "gauge",
                MetricType.HISTOGRAM: "histogram",
                MetricType.TIMER: "histogram",
            }

            # Validate metric data
            if metric.value is None or not isinstance(metric.value, (int, float)):
                return

            import math
            if math.isnan(metric.value) or math.isinf(metric.value):
                return

            # Add state-specific labels
            labels = {"component": "state", "metric_name": metric.name, **metric.tags}
            central_type = metric_type_map.get(metric.metric_type, "gauge")

            # Record to central metrics
            if central_type == "counter":
                self.metrics_collector.increment_counter(f"state.{metric.name}", labels, metric.value)
            elif central_type == "gauge":
                self.metrics_collector.set_gauge(f"state.{metric.name}", metric.value, labels)
            elif central_type == "histogram":
                self.metrics_collector.observe_histogram(f"state.{metric.name}", metric.value, labels)

        except Exception as e:
            self.logger.warning(f"Failed to record to central monitoring: {e}")

    async def _send_alert_to_central(self, alert: Alert) -> None:
        """Send alert to central alerting system."""
        try:
            if not self._central_alert_manager:
                # Try to get alert manager if not already set
                try:
                    from src.monitoring.alerting import get_alert_manager
                    self._central_alert_manager = get_alert_manager()
                except ImportError:
                    return

            if not self._central_alert_manager:
                return

            # Convert to central alert format
            from src.monitoring.alerting import (
                Alert as CentralAlert,
                AlertSeverity as CentralSeverity,
                AlertStatus,
            )

            severity_map = {
                AlertSeverity.LOW: CentralSeverity.LOW,
                AlertSeverity.MEDIUM: CentralSeverity.MEDIUM,
                AlertSeverity.HIGH: CentralSeverity.HIGH,
                AlertSeverity.CRITICAL: CentralSeverity.CRITICAL,
            }

            central_alert = CentralAlert(
                rule_name=f"state.{alert.source}",
                severity=severity_map.get(alert.severity, CentralSeverity.MEDIUM),
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
                    "current_value": str(alert.current_value),
                    "threshold_value": str(alert.threshold_value),
                },
                starts_at=alert.timestamp,
            )

            await self._central_alert_manager.fire_alert(central_alert)

        except Exception as e:
            self.logger.warning(f"Failed to send alert to central system: {e}")

    def _setup_central_integration(self) -> None:
        """Setup integration with central monitoring systems."""
        # Register alert handler to forward to central system
        self.register_alert_handler(self._forward_alert_handler)

    async def _forward_alert_handler(self, alert: Alert) -> None:
        """Forward alert to central system."""
        try:
            await self._send_alert_to_central(alert)
        except Exception as e:
            self.logger.warning(f"Failed to forward alert to central system: {e}")

    async def _forward_alert_to_central(self, alert: Alert) -> None:
        """Forward alert to central system via adapter."""
        try:
            if self.alert_adapter:
                await self.alert_adapter.send_alert(alert)
        except Exception as e:
            self.logger.warning(f"Failed to forward alert via adapter: {e}")
