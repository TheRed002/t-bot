"""
Comprehensive Performance Monitoring System

This module provides real-time performance monitoring and metrics collection
for the T-Bot trading system, with focus on achieving optimal latency targets
for critical trading operations.
"""

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import psutil
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from src.core.base.component import BaseComponent
from src.core.config import Config
from src.core.exceptions import PerformanceError
from src.core.logging import get_logger
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    CACHE_PERFORMANCE = "cache_performance"
    NETWORK_PERFORMANCE = "network_performance"


class OperationType(Enum):
    """Types of operations being monitored."""

    TRADING_ORDER = "trading_order"
    MARKET_DATA_UPDATE = "market_data_update"
    DATABASE_QUERY = "database_query"
    CACHE_ACCESS = "cache_access"
    WEBSOCKET_MESSAGE = "websocket_message"
    API_REQUEST = "api_request"
    ML_PREDICTION = "ml_prediction"
    RISK_CALCULATION = "risk_calculation"


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""

    timestamp: datetime
    operation_type: OperationType
    metric_type: MetricType
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Latency statistics for an operation type."""

    operation_type: OperationType
    count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    p999: float = 0.0
    recent_measurements: deque = field(default_factory=lambda: deque(maxlen=1000))


@dataclass
class ThroughputStats:
    """Throughput statistics for an operation type."""

    operation_type: OperationType
    total_operations: int = 0
    operations_per_second: float = 0.0
    peak_throughput: float = 0.0
    time_window_ops: deque = field(default_factory=lambda: deque(maxlen=60))  # 1 minute window


@dataclass
class ResourceUsageStats:
    """System resource usage statistics."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    open_file_descriptors: int
    active_connections: int


@dataclass
class PerformanceAlert:
    """Performance alert definition."""

    alert_id: str
    level: AlertLevel
    title: str
    description: str
    threshold: float
    current_value: float
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class PerformanceThresholds:
    """Performance thresholds for alerting."""

    # Latency thresholds (milliseconds)
    TRADING_LATENCY_WARNING = 50.0
    TRADING_LATENCY_CRITICAL = 100.0
    DATABASE_LATENCY_WARNING = 25.0
    DATABASE_LATENCY_CRITICAL = 50.0
    CACHE_LATENCY_WARNING = 5.0
    CACHE_LATENCY_CRITICAL = 10.0
    WEBSOCKET_LATENCY_WARNING = 10.0
    WEBSOCKET_LATENCY_CRITICAL = 20.0

    # Resource usage thresholds (percentage)
    CPU_USAGE_WARNING = 70.0
    CPU_USAGE_CRITICAL = 85.0
    MEMORY_USAGE_WARNING = 70.0
    MEMORY_USAGE_CRITICAL = 85.0

    # Error rate thresholds (percentage)
    ERROR_RATE_WARNING = 1.0
    ERROR_RATE_CRITICAL = 5.0

    # Cache performance thresholds
    CACHE_HIT_RATE_WARNING = 80.0
    CACHE_HIT_RATE_CRITICAL = 70.0


class LatencyTracker:
    """High-precision latency tracking for trading operations."""

    def __init__(self, operation_type: OperationType):
        self.operation_type = operation_type
        self.stats = LatencyStats(operation_type=operation_type)
        self._lock = asyncio.Lock()

    async def record_latency(
        self, latency_ms: float, metadata: dict[str, Any] | None = None
    ) -> None:
        """Record a latency measurement."""
        async with self._lock:
            self.stats.count += 1
            self.stats.total_time += latency_ms
            self.stats.min_time = min(self.stats.min_time, latency_ms)
            self.stats.max_time = max(self.stats.max_time, latency_ms)

            # Add to recent measurements for percentile calculation
            self.stats.recent_measurements.append(latency_ms)

            # Update percentiles if we have enough measurements
            if len(self.stats.recent_measurements) >= 10:
                measurements = list(self.stats.recent_measurements)
                measurements.sort()

                self.stats.p50 = self._percentile(measurements, 50)
                self.stats.p95 = self._percentile(measurements, 95)
                self.stats.p99 = self._percentile(measurements, 99)
                self.stats.p999 = self._percentile(measurements, 99.9)

    def _percentile(self, sorted_list: list[float], percentile: float) -> float:
        """Calculate percentile from sorted list."""
        if not sorted_list:
            return 0.0

        index = (percentile / 100.0) * (len(sorted_list) - 1)
        if index.is_integer():
            return sorted_list[int(index)]
        else:
            lower = sorted_list[int(index)]
            upper = sorted_list[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    def get_stats(self) -> LatencyStats:
        """Get current latency statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset statistics (useful for periodic reports)."""
        self.stats = LatencyStats(operation_type=self.operation_type)


class ThroughputTracker:
    """Throughput tracking for operations per second."""

    def __init__(self, operation_type: OperationType):
        self.operation_type = operation_type
        self.stats = ThroughputStats(operation_type=operation_type)
        self._lock = asyncio.Lock()
        self._last_calculation = time.time()

    async def record_operation(self) -> None:
        """Record an operation occurrence."""
        async with self._lock:
            current_time = time.time()
            self.stats.total_operations += 1
            self.stats.time_window_ops.append(current_time)

            # Calculate operations per second
            if current_time - self._last_calculation >= 1.0:  # Update every second
                self._calculate_throughput()
                self._last_calculation = current_time

    def _calculate_throughput(self) -> None:
        """Calculate current throughput."""
        current_time = time.time()
        one_minute_ago = current_time - 60

        # Count operations in the last minute
        recent_ops = sum(1 for op_time in self.stats.time_window_ops if op_time > one_minute_ago)
        self.stats.operations_per_second = recent_ops / 60.0

        # Update peak throughput
        if self.stats.operations_per_second > self.stats.peak_throughput:
            self.stats.peak_throughput = self.stats.operations_per_second

    def get_stats(self) -> ThroughputStats:
        """Get current throughput statistics."""
        return self.stats


class PrometheusMetricsCollector:
    """Prometheus metrics collector for external monitoring."""

    def __init__(self):
        self.registry = CollectorRegistry()

        # Latency histograms
        self.latency_histogram = Histogram(
            "tbot_operation_latency_seconds",
            "Operation latency in seconds",
            ["operation_type", "exchange"],
            registry=self.registry,
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        # Throughput counters
        self.operation_counter = Counter(
            "tbot_operations_total",
            "Total number of operations",
            ["operation_type", "status", "exchange"],
            registry=self.registry,
        )

        # Resource usage gauges
        self.cpu_usage = Gauge(
            "tbot_cpu_usage_percent", "CPU usage percentage", registry=self.registry
        )

        self.memory_usage = Gauge(
            "tbot_memory_usage_bytes", "Memory usage in bytes", registry=self.registry
        )

        self.active_connections = Gauge(
            "tbot_active_connections",
            "Number of active connections",
            ["exchange"],
            registry=self.registry,
        )

        # Cache metrics
        self.cache_hit_rate = Gauge(
            "tbot_cache_hit_rate", "Cache hit rate", ["cache_level"], registry=self.registry
        )

        # Trading specific metrics
        self.order_latency = Histogram(
            "tbot_order_latency_seconds",
            "Order execution latency",
            ["exchange", "order_type"],
            registry=self.registry,
        )

        self.pnl_gauge = Gauge(
            "tbot_total_pnl", "Total profit and loss", ["bot_id"], registry=self.registry
        )

    def record_latency(
        self, operation_type: str, latency_seconds: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record latency measurement."""
        label_values = labels or {}
        self.latency_histogram.labels(
            operation_type=operation_type, exchange=label_values.get("exchange", "unknown")
        ).observe(latency_seconds)

    def increment_operation(
        self, operation_type: str, status: str, labels: dict[str, str] | None = None
    ) -> None:
        """Increment operation counter."""
        label_values = labels or {}
        self.operation_counter.labels(
            operation_type=operation_type,
            status=status,
            exchange=label_values.get("exchange", "unknown"),
        ).inc()

    def update_resource_usage(self, cpu_percent: float, memory_bytes: float) -> None:
        """Update resource usage metrics."""
        self.cpu_usage.set(cpu_percent)
        self.memory_usage.set(memory_bytes)

    def get_metrics(self) -> str:
        """Get Prometheus formatted metrics."""
        return generate_latest(self.registry).decode("utf-8")


class PerformanceMonitor(BaseComponent):
    """
    Comprehensive performance monitoring system for T-Bot trading operations.

    Provides real-time monitoring, alerting, and optimization recommendations
    for achieving sub-100ms trading latency targets.
    """

    def __init__(self, config: Config):
        """Initialize performance monitor."""
        super().__init__()
        self.config = config

        # Latency trackers for different operation types
        self.latency_trackers: dict[OperationType, LatencyTracker] = {}
        for op_type in OperationType:
            self.latency_trackers[op_type] = LatencyTracker(op_type)

        # Throughput trackers
        self.throughput_trackers: dict[OperationType, ThroughputTracker] = {}
        for op_type in OperationType:
            self.throughput_trackers[op_type] = ThroughputTracker(op_type)

        # Resource monitoring
        self.resource_history: deque = deque(maxlen=3600)  # 1 hour at 1-second intervals
        self.last_resource_check = 0.0

        # Error tracking
        self.error_counts: dict[OperationType, int] = defaultdict(int)
        self.total_operations: dict[OperationType, int] = defaultdict(int)

        # Alert system
        self.active_alerts: dict[str, PerformanceAlert] = {}
        self.alert_callbacks: list[Callable] = []
        self.alert_cooldown = 300  # 5 minutes
        self.last_alert_times: dict[str, float] = {}

        # Performance thresholds
        self.thresholds = PerformanceThresholds()

        # Prometheus metrics
        self.prometheus_collector = PrometheusMetricsCollector()

        # Background monitoring
        self._monitoring_task: asyncio.Task | None = None
        self._monitoring_interval = 1.0  # 1 second for high-frequency monitoring

        # Performance regression detection
        self.baseline_metrics: dict[str, float] = {}
        self.regression_threshold = 0.2  # 20% performance degradation threshold

    async def initialize(self) -> None:
        """Initialize performance monitoring system."""
        try:
            self.logger.info("Initializing performance monitoring system...")

            # Start background monitoring
            await self._start_monitoring()

            # Load baseline metrics if available
            await self._load_baseline_metrics()

            self.logger.info("Performance monitoring system initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize performance monitoring: {e}")
            raise PerformanceError(f"Performance monitoring initialization failed: {e}") from e

    async def _start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Performance monitoring started")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self._monitoring_interval)

                # Collect resource usage
                await self._collect_resource_usage()

                # Check for performance alerts
                await self._check_performance_alerts()

                # Update Prometheus metrics
                await self._update_prometheus_metrics()

                # Detect performance regressions
                await self._detect_performance_regressions()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")

    @time_execution
    async def record_operation_start(
        self, operation_type: OperationType, metadata: dict[str, Any] | None = None
    ) -> str:
        """
        Record the start of an operation and return a tracking ID.

        Args:
            operation_type: Type of operation being tracked
            metadata: Additional metadata for the operation

        Returns:
            Tracking ID for the operation
        """
        tracking_id = (
            f"{operation_type.value}_{int(time.time() * 1000000)}"  # Microsecond precision
        )

        # Store start time and metadata
        if not hasattr(self, "_active_operations"):
            self._active_operations = {}

        self._active_operations[tracking_id] = {
            "start_time": time.perf_counter(),
            "operation_type": operation_type,
            "metadata": metadata or {},
        }

        # Record throughput
        await self.throughput_trackers[operation_type].record_operation()

        return tracking_id

    async def record_operation_end(
        self, tracking_id: str, success: bool = True, metadata: dict[str, Any] | None = None
    ) -> float:
        """
        Record the end of an operation and calculate latency.

        Args:
            tracking_id: The tracking ID returned by record_operation_start
            success: Whether the operation was successful
            metadata: Additional metadata for the operation

        Returns:
            Operation latency in milliseconds
        """
        if not hasattr(self, "_active_operations") or tracking_id not in self._active_operations:
            self.logger.warning(f"Unknown tracking ID: {tracking_id}")
            return 0.0

        operation_data = self._active_operations.pop(tracking_id)
        end_time = time.perf_counter()

        # Calculate latency in milliseconds
        latency_ms = (end_time - operation_data["start_time"]) * 1000

        operation_type = operation_data["operation_type"]

        # Record latency
        await self.latency_trackers[operation_type].record_latency(
            latency_ms, {**operation_data["metadata"], **(metadata or {})}
        )

        # Update operation counts
        self.total_operations[operation_type] += 1
        if not success:
            self.error_counts[operation_type] += 1

        # Update Prometheus metrics
        self.prometheus_collector.record_latency(
            operation_type.value,
            latency_ms / 1000,  # Convert to seconds for Prometheus
            operation_data["metadata"],
        )

        self.prometheus_collector.increment_operation(
            operation_type.value, "success" if success else "error", operation_data["metadata"]
        )

        return latency_ms

    async def record_simple_latency(
        self,
        operation_type: OperationType,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a simple latency measurement without start/end tracking.

        Args:
            operation_type: Type of operation
            latency_ms: Latency in milliseconds
            metadata: Additional metadata
        """
        await self.latency_trackers[operation_type].record_latency(latency_ms, metadata)

        # Update Prometheus metrics
        self.prometheus_collector.record_latency(
            operation_type.value, latency_ms / 1000, metadata or {}
        )

    async def _collect_resource_usage(self) -> None:
        """Collect system resource usage statistics."""
        current_time = time.time()

        # Throttle resource collection to avoid overhead
        if current_time - self.last_resource_check < 1.0:
            return

        try:
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()

            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            open_files = process.num_fds() if hasattr(process, "num_fds") else 0
            connections = len(process.connections())

            stats = ResourceUsageStats(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_mb=process_memory.rss / (1024 * 1024),
                disk_io_read_mb=(disk_io.read_bytes / (1024 * 1024)) if disk_io else 0,
                disk_io_write_mb=(disk_io.write_bytes / (1024 * 1024)) if disk_io else 0,
                network_io_sent_mb=(network_io.bytes_sent / (1024 * 1024)) if network_io else 0,
                network_io_recv_mb=(network_io.bytes_recv / (1024 * 1024)) if network_io else 0,
                open_file_descriptors=open_files,
                active_connections=connections,
            )

            self.resource_history.append(stats)

            # Update Prometheus metrics
            self.prometheus_collector.update_resource_usage(cpu_percent, process_memory.rss)

            self.last_resource_check = current_time

        except Exception as e:
            self.logger.error(f"Failed to collect resource usage: {e}")

    async def _check_performance_alerts(self) -> None:
        """Check for performance threshold violations and generate alerts."""
        alerts_to_add: list[PerformanceAlert] = []
        alerts_to_remove: list[str] = []

        # Check different types of alerts
        await self._check_latency_alerts(alerts_to_add, alerts_to_remove)
        self._check_resource_usage_alerts(alerts_to_add)
        self._check_error_rate_alerts(alerts_to_add)

        # Process all collected alerts
        await self._process_alerts(alerts_to_add, alerts_to_remove)

    async def _check_latency_alerts(self, alerts_to_add: list, alerts_to_remove: list) -> None:
        """Check latency alerts for all operation types."""
        for op_type, tracker in self.latency_trackers.items():
            stats = tracker.get_stats()
            if stats.count == 0:
                continue

            alert_id = f"latency_{op_type.value}"
            threshold_warning, threshold_critical = self._get_latency_thresholds(op_type)

            alert = self._create_latency_alert_if_needed(
                alert_id, op_type, stats.p95, threshold_warning, threshold_critical
            )

            if alert:
                alerts_to_add.append(alert)
            elif alert_id in self.active_alerts:
                alerts_to_remove.append(alert_id)

    def _create_latency_alert_if_needed(
        self,
        alert_id: str,
        op_type,
        p95_latency: float,
        threshold_warning: float,
        threshold_critical: float,
    ):
        """Create latency alert if thresholds are exceeded."""
        if p95_latency > threshold_critical:
            return PerformanceAlert(
                alert_id=alert_id,
                level=AlertLevel.CRITICAL,
                title=f"Critical latency for {op_type.value}",
                description=(
                    f"95th percentile latency ({p95_latency:.1f}ms) "
                    f"exceeds critical threshold ({threshold_critical}ms)"
                ),
                threshold=threshold_critical,
                current_value=p95_latency,
                timestamp=datetime.now(timezone.utc),
            )
        elif p95_latency > threshold_warning:
            return PerformanceAlert(
                alert_id=alert_id,
                level=AlertLevel.WARNING,
                title=f"High latency for {op_type.value}",
                description=(
                    f"95th percentile latency ({p95_latency:.1f}ms) "
                    f"exceeds warning threshold ({threshold_warning}ms)"
                ),
                threshold=threshold_warning,
                current_value=p95_latency,
                timestamp=datetime.now(timezone.utc),
            )
        return None

    def _check_resource_usage_alerts(self, alerts_to_add: list) -> None:
        """Check resource usage alerts (CPU and memory)."""
        if not self.resource_history:
            return

        latest_stats = self.resource_history[-1]

        # Check CPU usage
        cpu_alert = self._create_resource_alert(
            "cpu_usage",
            "CPU usage",
            latest_stats.cpu_percent,
            self.thresholds.CPU_USAGE_WARNING,
            self.thresholds.CPU_USAGE_CRITICAL,
        )
        if cpu_alert:
            alerts_to_add.append(cpu_alert)

        # Check memory usage
        memory_alert = self._create_resource_alert(
            "memory_usage",
            "Memory usage",
            latest_stats.memory_percent,
            self.thresholds.MEMORY_USAGE_WARNING,
            self.thresholds.MEMORY_USAGE_CRITICAL,
        )
        if memory_alert:
            alerts_to_add.append(memory_alert)

    def _create_resource_alert(
        self,
        alert_id: str,
        resource_name: str,
        current_value: float,
        warning_threshold: float,
        critical_threshold: float,
    ):
        """Create resource usage alert if thresholds are exceeded."""
        if current_value > critical_threshold:
            return PerformanceAlert(
                alert_id=alert_id,
                level=AlertLevel.CRITICAL,
                title=f"Critical {resource_name.lower()}",
                description=f"{resource_name} ({current_value:.1f}%) exceeds critical threshold",
                threshold=critical_threshold,
                current_value=current_value,
                timestamp=datetime.now(timezone.utc),
            )
        elif current_value > warning_threshold:
            return PerformanceAlert(
                alert_id=alert_id,
                level=AlertLevel.WARNING,
                title=f"High {resource_name.lower()}",
                description=f"{resource_name} ({current_value:.1f}%) exceeds warning threshold",
                threshold=warning_threshold,
                current_value=current_value,
                timestamp=datetime.now(timezone.utc),
            )
        return None

    def _check_error_rate_alerts(self, alerts_to_add: list) -> None:
        """Check error rate alerts for all operation types."""
        for op_type in OperationType:
            if self.total_operations[op_type] <= 0:
                continue

            error_rate = (self.error_counts[op_type] / self.total_operations[op_type]) * 100
            alert_id = f"error_rate_{op_type.value}"

            alert = self._create_error_rate_alert(
                alert_id,
                op_type,
                error_rate,
                self.thresholds.ERROR_RATE_WARNING,
                self.thresholds.ERROR_RATE_CRITICAL,
            )

            if alert:
                alerts_to_add.append(alert)

    def _create_error_rate_alert(
        self,
        alert_id: str,
        op_type,
        error_rate: float,
        warning_threshold: float,
        critical_threshold: float,
    ):
        """Create error rate alert if thresholds are exceeded."""
        if error_rate > critical_threshold:
            return PerformanceAlert(
                alert_id=alert_id,
                level=AlertLevel.CRITICAL,
                title=f"Critical error rate for {op_type.value}",
                description=f"Error rate ({error_rate:.1f}%) exceeds critical threshold",
                threshold=critical_threshold,
                current_value=error_rate,
                timestamp=datetime.now(timezone.utc),
            )
        elif error_rate > warning_threshold:
            return PerformanceAlert(
                alert_id=alert_id,
                level=AlertLevel.WARNING,
                title=f"High error rate for {op_type.value}",
                description=f"Error rate ({error_rate:.1f}%) exceeds warning threshold",
                threshold=warning_threshold,
                current_value=error_rate,
                timestamp=datetime.now(timezone.utc),
            )
        return None

    def _get_latency_thresholds(self, operation_type: OperationType) -> tuple[float, float]:
        """Get latency thresholds for operation type."""
        thresholds_map = {
            OperationType.TRADING_ORDER: (
                self.thresholds.TRADING_LATENCY_WARNING,
                self.thresholds.TRADING_LATENCY_CRITICAL,
            ),
            OperationType.DATABASE_QUERY: (
                self.thresholds.DATABASE_LATENCY_WARNING,
                self.thresholds.DATABASE_LATENCY_CRITICAL,
            ),
            OperationType.CACHE_ACCESS: (
                self.thresholds.CACHE_LATENCY_WARNING,
                self.thresholds.CACHE_LATENCY_CRITICAL,
            ),
            OperationType.WEBSOCKET_MESSAGE: (
                self.thresholds.WEBSOCKET_LATENCY_WARNING,
                self.thresholds.WEBSOCKET_LATENCY_CRITICAL,
            ),
        }
        return thresholds_map.get(operation_type, (50.0, 100.0))  # Default thresholds

    async def _process_alerts(
        self, alerts_to_add: list[PerformanceAlert], alerts_to_remove: list[str]
    ) -> None:
        """Process alerts - add new ones and remove resolved ones."""
        current_time = time.time()

        # Add new alerts
        for alert in alerts_to_add:
            # Check cooldown
            if alert.alert_id in self.last_alert_times:
                if current_time - self.last_alert_times[alert.alert_id] < self.alert_cooldown:
                    continue

            self.active_alerts[alert.alert_id] = alert
            self.last_alert_times[alert.alert_id] = current_time

            # Log alert
            self.logger.warning(
                f"Performance alert: {alert.title}",
                extra={
                    "alert_level": alert.level.value,
                    "alert_id": alert.alert_id,
                    "description": alert.description,
                    "threshold": alert.threshold,
                    "current_value": alert.current_value,
                },
            )

            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback error: {e}")

        # Remove resolved alerts
        for alert_id in alerts_to_remove:
            if alert_id in self.active_alerts:
                resolved_alert = self.active_alerts.pop(alert_id)
                self.logger.info(f"Performance alert resolved: {resolved_alert.title}")

    async def _update_prometheus_metrics(self) -> None:
        """Update Prometheus metrics with current data."""
        # This method updates Prometheus metrics that are already being updated
        # in other methods. Can be used for any additional custom metrics.
        pass

    async def _detect_performance_regressions(self) -> None:
        """Detect performance regressions against baseline metrics."""
        # Placeholder for regression detection logic
        # This would compare current performance against historical baselines
        pass

    async def _load_baseline_metrics(self) -> None:
        """Load baseline performance metrics for regression detection."""
        # Placeholder for loading baseline metrics from storage
        pass

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add callback for performance alerts."""
        self.alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Remove alert callback."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)

    async def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_stats": {},
            "throughput_stats": {},
            "resource_usage": {},
            "error_rates": {},
            "active_alerts": len(self.active_alerts),
            "alerts": [alert.__dict__ for alert in self.active_alerts.values()],
        }

        # Latency statistics
        for op_type, tracker in self.latency_trackers.items():
            stats = tracker.get_stats()
            summary["latency_stats"][op_type.value] = {
                "count": stats.count,
                "p50_ms": stats.p50,
                "p95_ms": stats.p95,
                "p99_ms": stats.p99,
                "p999_ms": stats.p999,
                "min_ms": stats.min_time if stats.min_time != float("inf") else 0,
                "max_ms": stats.max_time,
            }

        # Throughput statistics
        for op_type, tracker in self.throughput_trackers.items():
            stats = tracker.get_stats()
            summary["throughput_stats"][op_type.value] = {
                "total_operations": stats.total_operations,
                "operations_per_second": stats.operations_per_second,
                "peak_throughput": stats.peak_throughput,
            }

        # Resource usage
        if self.resource_history:
            latest = self.resource_history[-1]
            summary["resource_usage"] = {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "memory_mb": latest.memory_mb,
                "active_connections": latest.active_connections,
                "open_file_descriptors": latest.open_file_descriptors,
            }

        # Error rates
        for op_type in OperationType:
            if self.total_operations[op_type] > 0:
                error_rate = (self.error_counts[op_type] / self.total_operations[op_type]) * 100
                summary["error_rates"][op_type.value] = error_rate

        return summary

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics."""
        return self.prometheus_collector.get_metrics()

    async def reset_statistics(self) -> None:
        """Reset all performance statistics."""
        # Reset latency trackers
        for tracker in self.latency_trackers.values():
            tracker.reset_stats()

        # Reset throughput trackers
        for op_type in OperationType:
            self.throughput_trackers[op_type] = ThroughputTracker(op_type)

        # Reset error counts
        self.error_counts.clear()
        self.total_operations.clear()

        # Clear resource history
        self.resource_history.clear()

        # Clear alerts
        self.active_alerts.clear()

        self.logger.info("Performance statistics reset")

    async def cleanup(self) -> None:
        """Cleanup performance monitoring resources."""
        try:
            # Cancel monitoring task
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass

            # Clear all data
            await self.reset_statistics()

            self.logger.info("Performance monitoring cleaned up")

        except Exception as e:
            self.logger.error(f"Performance monitoring cleanup error: {e}")


# Context manager for operation tracking
class OperationTracker:
    """Context manager for automatic operation tracking."""

    def __init__(
        self,
        monitor: PerformanceMonitor,
        operation_type: OperationType,
        metadata: dict[str, Any] | None = None,
    ):
        self.monitor = monitor
        self.operation_type = operation_type
        self.metadata = metadata
        self.tracking_id = None

    async def __aenter__(self):
        self.tracking_id = await self.monitor.record_operation_start(
            self.operation_type, self.metadata
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.tracking_id:
            success = exc_type is None
            await self.monitor.record_operation_end(self.tracking_id, success)


# Decorators for automatic performance tracking
def track_performance(operation_type: OperationType, monitor: PerformanceMonitor | None = None):
    """Decorator for automatic performance tracking."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            if monitor is None:
                # Try to get monitor from first argument (self)
                if args and hasattr(args[0], "performance_monitor"):
                    perf_monitor = args[0].performance_monitor
                else:
                    # No monitor available, just run the function
                    return await func(*args, **kwargs)
            else:
                perf_monitor = monitor

            async with OperationTracker(perf_monitor, operation_type):
                return await func(*args, **kwargs)

        return wrapper

    return decorator
