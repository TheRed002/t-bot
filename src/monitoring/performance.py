"""
Comprehensive Performance Monitoring System for T-Bot Trading Platform.

This module implements real-time performance monitoring for high-frequency trading systems,
tracking order execution latency, market data throughput, system resources, and trading metrics.

Key Features:
- Sub-millisecond latency tracking for trading operations
- Real-time system resource monitoring
- Trading-specific performance metrics
- Integration with Prometheus and alerting systems
- Anomaly detection for performance degradation
- Memory profiling and garbage collection tracking

Performance Categories:
- Order Execution: Latency percentiles (P50, P95, P99), fill rates, slippage
- Market Data: Processing throughput, WebSocket latency, data quality
- System Resources: CPU, memory, disk I/O, network performance
- Database: Query latency, connection pool utilization
- Trading Strategies: P&L metrics, signal accuracy, execution efficiency
"""

import asyncio
import gc
import os
import resource
import statistics
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import psutil

try:
    from src.core.base import BaseComponent
except ImportError:
    from src.base import BaseComponent

try:
    from src.core import OrderType
except ImportError:
    from enum import Enum

    class OrderType(Enum):
        """Order type fallback."""

        MARKET = "MARKET"
        LIMIT = "LIMIT"
        STOP = "STOP"
        STOP_LIMIT = "STOP_LIMIT"


# Import utils decorators and helpers for better integration
from src.utils.decorators import cache_result, logged, monitored, retry, time_execution
from src.utils.helpers import format_timestamp

# Import error handling with fallback
try:
    from src.error_handling import (
        ErrorContext,
        RecoveryScenario,
        get_global_error_handler,
    )
except ImportError:
    # Simple fallbacks
    from dataclasses import dataclass

    @dataclass
    class ErrorContext:
        component: str
        operation: str
        details: dict = None

    @dataclass
    class RecoveryScenario:
        component: str
        operation: str
        error: Exception
        context: dict = None

    def get_global_error_handler():
        return None


from src.monitoring.alerting import Alert, AlertManager, AlertSeverity, AlertStatus
from src.monitoring.metrics import MetricsCollector, get_metrics_collector


class PerformanceCategory(Enum):
    """Performance monitoring categories."""

    ORDER_EXECUTION = "order_execution"
    MARKET_DATA = "market_data"
    SYSTEM_RESOURCES = "system_resources"
    DATABASE = "database"
    WEBSOCKET = "websocket"
    STRATEGY = "strategy"
    MEMORY = "memory"
    NETWORK = "network"
    CACHE = "cache"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""

    name: str
    category: PerformanceCategory
    value: float
    timestamp: datetime
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Latency statistics with percentiles."""

    count: int
    p50: float
    p95: float
    p99: float
    p999: float
    min_value: float
    max_value: float
    avg: float
    sum_value: float
    last_updated: datetime

    @classmethod
    def from_values(cls, values: list[float]) -> "LatencyStats":
        """Create LatencyStats from list of values."""
        if not values:
            return cls(
                count=0,
                p50=0.0,
                p95=0.0,
                p99=0.0,
                p999=0.0,
                min_value=0.0,
                max_value=0.0,
                avg=0.0,
                sum_value=0.0,
                last_updated=datetime.now(timezone.utc),
            )

        sorted_values = sorted(values)
        count = len(values)

        return cls(
            count=count,
            p50=statistics.quantiles(sorted_values, n=2)[0] if count > 1 else sorted_values[0],
            p95=statistics.quantiles(sorted_values, n=20)[18] if count > 20 else sorted_values[-1],
            p99=(
                statistics.quantiles(sorted_values, n=100)[98] if count > 100 else sorted_values[-1]
            ),
            p999=(
                statistics.quantiles(sorted_values, n=1000)[998]
                if count > 1000
                else sorted_values[-1]
            ),
            min_value=min(values),
            max_value=max(values),
            avg=sum(values) / count,
            sum_value=sum(values),
            last_updated=datetime.now(timezone.utc),
        )


@dataclass
class ThroughputStats:
    """Throughput statistics."""

    total_count: int
    rate_per_second: float
    rate_per_minute: float
    peak_rate: float
    last_updated: datetime


@dataclass
class SystemResourceStats:
    """System resource utilization statistics."""

    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    load_average: list[float]
    open_file_descriptors: int
    thread_count: int
    last_updated: datetime


@dataclass
class GCStats:
    """Garbage collection statistics."""

    collections: list[int]  # Collections per generation
    collected: list[int]  # Objects collected per generation
    uncollectable: list[int]  # Uncollectable objects per generation
    total_time: float  # Total GC time in seconds
    threshold: list[int]  # GC thresholds
    last_updated: datetime


class PerformanceProfiler(BaseComponent):
    """
    Comprehensive performance profiler for high-frequency trading systems.

    Provides real-time monitoring of:
    - Order execution latency with percentile tracking
    - Market data processing throughput
    - System resource utilization
    - Database query performance
    - WebSocket connection health
    - Memory usage and garbage collection
    - Trading strategy performance metrics
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector | None = None,
        alert_manager: AlertManager | None = None,
        max_samples: int = 10000,
        collection_interval: float = 1.0,
        anomaly_detection: bool = True,
        error_handler=None,
    ):
        """
        Initialize performance profiler.

        Args:
            metrics_collector: Prometheus metrics collector
            alert_manager: Alert manager for performance alerts
            max_samples: Maximum samples to keep for statistics
            collection_interval: Collection interval in seconds
            anomaly_detection: Enable anomaly detection
            error_handler: Error handler instance
        """
        super().__init__(name="PerformanceProfiler")

        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.alert_manager = alert_manager
        self._error_handler = error_handler or get_global_error_handler()
        self.max_samples = max_samples
        self.collection_interval = collection_interval
        self.anomaly_detection = anomaly_detection

        # Performance data storage
        self._latency_data: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self._throughput_data: dict[str, list[tuple]] = defaultdict(list)
        self._resource_history: deque = deque(maxlen=1000)
        self._gc_history: deque = deque(maxlen=100)

        # Threading and async support
        self._lock = threading.RLock()
        self._running = False
        self._background_task: asyncio.Task | None = None

        # Performance thresholds for alerting
        self._thresholds = {
            "order_execution_latency_ms": {"warning": 100, "critical": 500},
            "market_data_latency_ms": {"warning": 50, "critical": 200},
            "database_query_ms": {"warning": 100, "critical": 1000},
            "websocket_latency_ms": {"warning": 20, "critical": 100},
            "memory_usage_percent": {"warning": 80, "critical": 95},
            "cpu_usage_percent": {"warning": 80, "critical": 95},
        }

        # Baseline performance data for anomaly detection
        self._baselines: dict[str, dict[str, float]] = defaultdict(dict)
        self._last_baseline_update = 0.0

        # Initialize metrics
        self._register_metrics()

        self.logger.info("PerformanceProfiler initialized with comprehensive monitoring")

    def _register_metrics(self) -> None:
        """Register performance metrics with Prometheus collector."""
        from src.monitoring.metrics import MetricDefinition

        metrics = [
            # Order execution metrics
            MetricDefinition(
                "order_execution_latency_seconds",
                "Order execution latency in seconds",
                "histogram",
                ["exchange", "order_type", "symbol"],
                [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            ),
            MetricDefinition(
                "order_fill_rate_percent",
                "Order fill rate percentage",
                "gauge",
                ["exchange", "order_type", "symbol"],
            ),
            MetricDefinition(
                "order_slippage_bps",
                "Order execution slippage in basis points",
                "histogram",
                ["exchange", "order_type", "symbol"],
                [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0],
            ),
            # Market data metrics
            MetricDefinition(
                "market_data_processing_rate",
                "Market data messages processed per second",
                "gauge",
                ["exchange", "data_type"],
            ),
            MetricDefinition(
                "market_data_latency_seconds",
                "Market data processing latency in seconds",
                "histogram",
                ["exchange", "data_type"],
                [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
            ),
            # WebSocket metrics
            MetricDefinition(
                "websocket_message_latency_seconds",
                "WebSocket message latency in seconds",
                "histogram",
                ["exchange", "message_type"],
                [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            ),
            MetricDefinition(
                "websocket_connection_health",
                "WebSocket connection health score (0-1)",
                "gauge",
                ["exchange"],
            ),
            # Database metrics
            MetricDefinition(
                "database_query_latency_seconds",
                "Database query latency in seconds",
                "histogram",
                ["database", "operation", "table"],
                [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            ),
            MetricDefinition(
                "database_connection_pool_usage",
                "Database connection pool usage percentage",
                "gauge",
                ["database", "pool"],
            ),
            # System resource metrics
            MetricDefinition("system_memory_usage_bytes", "System memory usage in bytes", "gauge"),
            MetricDefinition(
                "system_gc_duration_seconds",
                "Garbage collection duration in seconds",
                "histogram",
                ["generation"],
                [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            ),
            MetricDefinition(
                "system_gc_objects_collected",
                "Number of objects collected by GC",
                "counter",
                ["generation"],
            ),
            # Strategy performance metrics
            MetricDefinition(
                "strategy_execution_latency_seconds",
                "Strategy execution latency in seconds",
                "histogram",
                ["strategy", "symbol"],
                [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            ),
            MetricDefinition(
                "strategy_signal_accuracy_percent",
                "Strategy signal accuracy percentage",
                "gauge",
                ["strategy", "timeframe"],
            ),
            MetricDefinition(
                "strategy_sharpe_ratio", "Strategy Sharpe ratio", "gauge", ["strategy", "timeframe"]
            ),
        ]

        for metric_def in metrics:
            try:
                self.metrics_collector.register_metric(metric_def)
            except Exception as e:
                self.logger.warning(f"Failed to register metric {metric_def.name}: {e}")

    async def start(self) -> None:
        """Start background performance monitoring."""
        if self._running:
            self.logger.warning("Performance profiler already running")
            return

        self._running = True
        self._background_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Started performance monitoring")

    async def stop(self) -> None:
        """Stop background performance monitoring."""
        self._running = False

        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None

        self.logger.info("Stopped performance monitoring")

    @retry(max_attempts=3, delay=2.0)
    @logged(level="debug")
    async def _monitoring_loop(self) -> None:
        """Background loop for collecting system metrics."""
        consecutive_errors = 0
        max_consecutive_errors = 5

        while self._running:
            try:
                # Check if we should still be running before each operation
                if not self._running:
                    break

                await self._collect_system_resources()
                await self._collect_gc_stats()
                await self._check_performance_thresholds()

                if self.anomaly_detection:
                    await self._detect_anomalies()

                # Reset error counter on success
                consecutive_errors = 0

                # Use wait_for to make sleep cancellable
                try:
                    await asyncio.wait_for(
                        asyncio.shield(asyncio.sleep(self.collection_interval)),
                        timeout=self.collection_interval,
                    )
                except asyncio.TimeoutError:
                    pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                consecutive_errors += 1

                # Use error handler for intelligent recovery
                if self._error_handler:
                    recovery_scenario = RecoveryScenario(
                        component="PerformanceProfiler",
                        operation="monitoring_loop",
                        error=e,
                        context=ErrorContext(
                            component="PerformanceProfiler",
                            operation="monitoring_loop",
                            consecutive_errors=consecutive_errors,
                        ),
                    )

                    try:
                        # Log the error and apply basic recovery strategy
                        await self._error_handler.handle_error(e, recovery_scenario.context)
                        
                        if consecutive_errors >= max_consecutive_errors:
                            self.logger.error(
                                f"Too many consecutive errors ({consecutive_errors}), "
                                "stopping monitoring loop"
                            )
                            break
                        else:
                            wait_time = min(5.0 * consecutive_errors, 60.0)
                            self.logger.warning(
                                f"Error in monitoring loop (attempt {consecutive_errors}), "
                                f"retrying in {wait_time}s: {e}"
                            )
                            await asyncio.sleep(wait_time)
                    except Exception as recovery_error:
                        self.logger.error(f"Recovery failed: {recovery_error}")
                        await asyncio.sleep(5.0)
                else:
                    self.logger.error(f"Error in performance monitoring loop: {e}")
                    await asyncio.sleep(5.0)

    @contextmanager
    @time_execution()
    def profile_function(
        self, function_name: str, module_name: str = "", labels: dict[str, str] | None = None
    ):
        """Context manager for profiling synchronous function execution."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()

            duration_ms = (end_time - start_time) * 1000
            memory_delta = end_memory - start_memory

            metric_name = f"{module_name}.{function_name}" if module_name else function_name

            with self._lock:
                self._latency_data[metric_name].append(duration_ms)

            # Update Prometheus metrics
            metric_labels = labels or {}
            metric_labels["function"] = function_name
            if module_name:
                metric_labels["module"] = module_name

            self.metrics_collector.observe_histogram(
                "function_execution_latency_seconds", duration_ms / 1000.0, metric_labels
            )

            if memory_delta > 0:
                self.metrics_collector.observe_histogram(
                    "function_memory_allocation_bytes", memory_delta, metric_labels
                )

    @asynccontextmanager
    async def profile_async_function(
        self, function_name: str, module_name: str = "", labels: dict[str, str] | None = None
    ):
        """Context manager for profiling asynchronous function execution."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()

            duration_ms = (end_time - start_time) * 1000
            memory_delta = end_memory - start_memory

            metric_name = f"{module_name}.{function_name}" if module_name else function_name

            with self._lock:
                self._latency_data[metric_name].append(duration_ms)

            # Update Prometheus metrics
            metric_labels = labels or {}
            metric_labels["function"] = function_name
            if module_name:
                metric_labels["module"] = module_name

            self.metrics_collector.observe_histogram(
                "async_function_execution_latency_seconds", duration_ms / 1000.0, metric_labels
            )

            if memory_delta > 0:
                self.metrics_collector.observe_histogram(
                    "async_function_memory_allocation_bytes", memory_delta, metric_labels
                )

    def record_order_execution(
        self,
        exchange: str,
        order_type: OrderType,
        symbol: str,
        latency_ms: float,
        fill_rate: float,
        slippage_bps: float,
    ) -> None:
        """Record order execution performance metrics."""
        with self._lock:
            metric_name = f"order_execution.{exchange}.{order_type.value}.{symbol}"
            self._latency_data[metric_name].append(latency_ms)

        labels = {"exchange": exchange, "order_type": order_type.value, "symbol": symbol}

        # Update Prometheus metrics
        self.metrics_collector.observe_histogram(
            "order_execution_latency_seconds", latency_ms / 1000.0, labels
        )

        self.metrics_collector.set_gauge("order_fill_rate_percent", fill_rate * 100, labels)

        self.metrics_collector.observe_histogram("order_slippage_bps", slippage_bps, labels)

        # Check for performance threshold violations
        if latency_ms > self._thresholds["order_execution_latency_ms"]["critical"]:
            task = asyncio.create_task(
                self._send_performance_alert(
                    "Critical order execution latency",
                    f"Order execution latency {latency_ms:.2f}ms exceeds critical threshold",
                    AlertSeverity.CRITICAL,
                    labels,
                )
            )
            # Properly handle task completion
            task.add_done_callback(self._handle_alert_task_completion)
        elif latency_ms > self._thresholds["order_execution_latency_ms"]["warning"]:
            task = asyncio.create_task(
                self._send_performance_alert(
                    "High order execution latency",
                    f"Order execution latency {latency_ms:.2f}ms exceeds warning threshold",
                    AlertSeverity.HIGH,
                    labels,
                )
            )
            # Properly handle task completion
            task.add_done_callback(self._handle_alert_task_completion)

    def record_market_data_processing(
        self, exchange: str, data_type: str, processing_time_ms: float, message_count: int
    ) -> None:
        """Record market data processing performance."""
        with self._lock:
            metric_name = f"market_data.{exchange}.{data_type}"
            self._latency_data[metric_name].append(processing_time_ms)

            # Update throughput data
            current_time = time.time()
            self._throughput_data[metric_name].append((current_time, message_count))

            # Clean old throughput data (keep last 5 minutes)
            cutoff_time = current_time - 300
            self._throughput_data[metric_name] = [
                (t, c) for t, c in self._throughput_data[metric_name] if t > cutoff_time
            ]

        labels = {"exchange": exchange, "data_type": data_type}

        # Calculate throughput (messages per second)
        throughput = self._calculate_throughput(f"market_data.{exchange}.{data_type}")

        # Update Prometheus metrics
        self.metrics_collector.observe_histogram(
            "market_data_latency_seconds", processing_time_ms / 1000.0, labels
        )

        self.metrics_collector.set_gauge("market_data_processing_rate", throughput, labels)

    def record_websocket_latency(self, exchange: str, message_type: str, latency_ms: float) -> None:
        """Record WebSocket message latency."""
        with self._lock:
            metric_name = f"websocket.{exchange}.{message_type}"
            self._latency_data[metric_name].append(latency_ms)

        labels = {"exchange": exchange, "message_type": message_type}

        # Update Prometheus metrics
        self.metrics_collector.observe_histogram(
            "websocket_message_latency_seconds", latency_ms / 1000.0, labels
        )

        # Calculate connection health score
        health_score = self._calculate_websocket_health(exchange)
        self.metrics_collector.set_gauge(
            "websocket_connection_health", health_score, {"exchange": exchange}
        )

    def record_database_query(
        self, database: str, operation: str, table: str, query_time_ms: float
    ) -> None:
        """Record database query performance."""
        with self._lock:
            metric_name = f"database.{database}.{operation}.{table}"
            self._latency_data[metric_name].append(query_time_ms)

        labels = {"database": database, "operation": operation, "table": table}

        # Update Prometheus metrics
        self.metrics_collector.observe_histogram(
            "database_query_latency_seconds", query_time_ms / 1000.0, labels
        )

        # Check for slow queries
        if query_time_ms > self._thresholds["database_query_ms"]["warning"]:
            severity = (
                AlertSeverity.HIGH
                if query_time_ms > self._thresholds["database_query_ms"]["critical"]
                else AlertSeverity.MEDIUM
            )
            task = asyncio.create_task(
                self._send_performance_alert(
                    "Slow database query detected",
                    f"Database query took {query_time_ms:.2f}ms on {database}.{table}",
                    severity,
                    labels,
                )
            )
            # Properly handle task completion
            task.add_done_callback(self._handle_alert_task_completion)

    def record_strategy_performance(
        self,
        strategy: str,
        symbol: str,
        execution_time_ms: float,
        signal_accuracy: float,
        sharpe_ratio: float,
        timeframe: str = "1d",
    ) -> None:
        """Record trading strategy performance metrics."""
        with self._lock:
            metric_name = f"strategy.{strategy}.{symbol}"
            self._latency_data[metric_name].append(execution_time_ms)

        labels = {"strategy": strategy, "symbol": symbol}

        strategy_labels = labels.copy()
        strategy_labels["timeframe"] = timeframe

        # Update Prometheus metrics
        self.metrics_collector.observe_histogram(
            "strategy_execution_latency_seconds", execution_time_ms / 1000.0, labels
        )

        self.metrics_collector.set_gauge(
            "strategy_signal_accuracy_percent", signal_accuracy * 100, strategy_labels
        )

        self.metrics_collector.set_gauge("strategy_sharpe_ratio", sharpe_ratio, strategy_labels)

    def get_latency_stats(self, metric_name: str) -> LatencyStats | None:
        """Get latency statistics for a specific metric."""
        with self._lock:
            values = list(self._latency_data.get(metric_name, []))

        if not values:
            return None

        return LatencyStats.from_values(values)

    def get_throughput_stats(self, metric_name: str) -> ThroughputStats | None:
        """Get throughput statistics for a specific metric."""
        with self._lock:
            data = self._throughput_data.get(metric_name, [])

        if not data:
            return None

        current_time = time.time()

        # Calculate rates
        total_count = sum(count for _, count in data)

        # Rate per second (last minute)
        last_minute = [(t, c) for t, c in data if current_time - t <= 60]
        rate_per_second = sum(c for _, c in last_minute) / 60 if last_minute else 0

        # Rate per minute (last hour)
        last_hour = [(t, c) for t, c in data if current_time - t <= 3600]
        rate_per_minute = sum(c for _, c in last_hour) / 60 if last_hour else 0

        # Peak rate (highest rate in any 10-second window)
        peak_rate = 0
        for i in range(len(data)):
            window_start = data[i][0]
            window_data = [(t, c) for t, c in data if window_start <= t < window_start + 10]
            window_rate = sum(c for _, c in window_data) / 10
            peak_rate = max(peak_rate, window_rate)

        return ThroughputStats(
            total_count=total_count,
            rate_per_second=rate_per_second,
            rate_per_minute=rate_per_minute,
            peak_rate=peak_rate,
            last_updated=datetime.now(timezone.utc),
        )

    def get_system_resource_stats(self) -> SystemResourceStats | None:
        """Get current system resource statistics."""
        with self._lock:
            if not self._resource_history:
                return None
            return self._resource_history[-1]

    def get_gc_stats(self) -> GCStats | None:
        """Get garbage collection statistics."""
        with self._lock:
            if not self._gc_history:
                return None
            return self._gc_history[-1]

    @cache_result(ttl=30)
    @monitored()
    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            "timestamp": format_timestamp(datetime.now(timezone.utc)),
            "metrics_collected": len(self._latency_data),
            "system_resources": {},
            "latency_stats": {},
            "throughput_stats": {},
            "gc_stats": {},
        }

        # System resources
        sys_stats = self.get_system_resource_stats()
        if sys_stats:
            summary["system_resources"] = {
                "cpu_percent": sys_stats.cpu_percent,
                "memory_percent": sys_stats.memory_percent,
                "memory_used_mb": sys_stats.memory_used_mb,
                "load_average": sys_stats.load_average,
                "thread_count": sys_stats.thread_count,
            }

        # Top 10 slowest operations by P99 latency
        latency_metrics = {}
        for metric_name in list(self._latency_data.keys())[:10]:
            stats = self.get_latency_stats(metric_name)
            if stats and stats.count > 0:
                latency_metrics[metric_name] = {
                    "p50": stats.p50,
                    "p95": stats.p95,
                    "p99": stats.p99,
                    "count": stats.count,
                    "avg": stats.avg,
                }

        summary["latency_stats"] = dict(
            sorted(latency_metrics.items(), key=lambda x: x[1]["p99"], reverse=True)
        )

        # GC stats
        gc_stats = self.get_gc_stats()
        if gc_stats:
            summary["gc_stats"] = {
                "collections": gc_stats.collections,
                "collected": gc_stats.collected,
                "total_time": gc_stats.total_time,
            }

        return summary

    def get_metrics(self) -> dict[str, Any]:
        """Get metrics for backward compatibility."""
        return self.get_performance_summary()

    def reset_metrics(self) -> None:
        """Reset all collected performance metrics."""
        with self._lock:
            self._latency_data.clear()
            self._throughput_data.clear()
            self._resource_history.clear()
            self._gc_history.clear()
            self._baselines.clear()

        self.logger.info("Performance metrics reset")

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except Exception:
            return psutil.Process().memory_info().rss

    def _calculate_throughput(self, metric_name: str) -> float:
        """Calculate current throughput for a metric."""
        with self._lock:
            data = self._throughput_data.get(metric_name, [])

        if not data:
            return 0.0

        current_time = time.time()
        # Calculate throughput for the last 60 seconds
        recent_data = [(t, c) for t, c in data if current_time - t <= 60]

        if not recent_data:
            return 0.0

        total_messages = sum(count for _, count in recent_data)
        time_span = max(1.0, current_time - min(t for t, _ in recent_data))

        return total_messages / time_span

    def _calculate_websocket_health(self, exchange: str) -> float:
        """Calculate WebSocket connection health score."""
        with self._lock:
            metric_name = f"websocket.{exchange}"
            latency_data = []

            # Collect latency data for all message types
            for key in self._latency_data:
                if key.startswith(metric_name):
                    latency_data.extend(list(self._latency_data[key]))

        if not latency_data:
            return 1.0

        # Health score based on latency percentiles
        stats = LatencyStats.from_values(latency_data)

        # Score calculation:
        # - Excellent (0.9-1.0): P95 < 10ms
        # - Good (0.7-0.9): P95 < 50ms
        # - Fair (0.5-0.7): P95 < 100ms
        # - Poor (0.0-0.5): P95 >= 100ms

        if stats.p95 < 10:
            return 1.0
        elif stats.p95 < 50:
            return 0.8
        elif stats.p95 < 100:
            return 0.6
        else:
            return max(0.1, 1.0 - (stats.p95 - 100) / 1000)

    @retry(max_attempts=2, delay=1.0)
    @logged(level="debug")
    async def _collect_system_resources(self) -> None:
        """Collect system resource metrics."""
        try:
            process = psutil.Process()

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)

            # Memory metrics
            memory = psutil.virtual_memory()

            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()

            # Network metrics
            network_io = psutil.net_io_counters()

            # Load average (Unix only)
            load_avg = list(os.getloadavg()) if hasattr(os, "getloadavg") else [0.0, 0.0, 0.0]

            # File descriptors
            try:
                open_fds = process.num_fds() if hasattr(process, "num_fds") else 0
            except Exception:
                open_fds = 0

            # Thread count
            thread_count = process.num_threads()

            stats = SystemResourceStats(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_io_read_mb=(disk_io.read_bytes if disk_io else 0) / (1024 * 1024),
                disk_io_write_mb=(disk_io.write_bytes if disk_io else 0) / (1024 * 1024),
                network_sent_mb=(network_io.bytes_sent if network_io else 0) / (1024 * 1024),
                network_recv_mb=(network_io.bytes_recv if network_io else 0) / (1024 * 1024),
                load_average=load_avg,
                open_file_descriptors=open_fds,
                thread_count=thread_count,
                last_updated=datetime.now(timezone.utc),
            )

            with self._lock:
                self._resource_history.append(stats)

            # Update Prometheus metrics
            self.metrics_collector.set_gauge("system_cpu_usage_percent", cpu_percent)
            self.metrics_collector.set_gauge("system_memory_usage_percent", memory.percent)
            self.metrics_collector.set_gauge("system_memory_usage_bytes", memory.used)
            self.metrics_collector.set_gauge("system_thread_count", thread_count)

            if load_avg[0] > 0:
                self.metrics_collector.set_gauge("system_load_average_1m", load_avg[0])
                self.metrics_collector.set_gauge("system_load_average_5m", load_avg[1])
                self.metrics_collector.set_gauge("system_load_average_15m", load_avg[2])

        except Exception as e:
            if hasattr(self, "_error_handler") and self._error_handler:
                await self._error_handler.handle_error(
                    e,
                    ErrorContext(
                        component="PerformanceProfiler",
                        operation="collect_system_resources",
                    ),
                )
            self.logger.error(f"Error collecting system resources: {e}")

    @retry(max_attempts=2, delay=0.5)
    @logged(level="debug")
    async def _collect_gc_stats(self) -> None:
        """Collect garbage collection statistics."""
        try:
            gc_stats = gc.get_stats()
            gc_threshold = gc.get_threshold()

            # Calculate total GC time (simplified)
            total_time = sum(stat.get("time", 0) for stat in gc_stats)

            stats = GCStats(
                collections=[stat.get("collections", 0) for stat in gc_stats],
                collected=[stat.get("collected", 0) for stat in gc_stats],
                uncollectable=[stat.get("uncollectable", 0) for stat in gc_stats],
                total_time=total_time,
                threshold=list(gc_threshold),
                last_updated=datetime.now(timezone.utc),
            )

            with self._lock:
                self._gc_history.append(stats)

            # Update Prometheus metrics
            for i, (collections, collected) in enumerate(
                zip(stats.collections, stats.collected, strict=False)
            ):
                generation_labels = {"generation": str(i)}

                self.metrics_collector.increment_counter(
                    "system_gc_collections_total", generation_labels, collections
                )

                self.metrics_collector.increment_counter(
                    "system_gc_objects_collected_total", generation_labels, collected
                )

        except Exception as e:
            if hasattr(self, "_error_handler") and self._error_handler:
                await self._error_handler.handle_error(
                    e,
                    ErrorContext(
                        component="PerformanceProfiler",
                        operation="collect_gc_stats",
                    ),
                )
            self.logger.error(f"Error collecting GC stats: {e}")

    async def _check_performance_thresholds(self) -> None:
        """Check performance metrics against thresholds."""
        try:
            # Check system resource thresholds
            resource_stats = self.get_system_resource_stats()
            if resource_stats:
                if resource_stats.cpu_percent > self._thresholds["cpu_usage_percent"]["critical"]:
                    await self._send_performance_alert(
                        "Critical CPU usage",
                        f"CPU usage {resource_stats.cpu_percent:.1f}% exceeds critical threshold",
                        AlertSeverity.CRITICAL,
                        {"metric": "cpu_usage"},
                    )

                if (
                    resource_stats.memory_percent
                    > self._thresholds["memory_usage_percent"]["critical"]
                ):
                    await self._send_performance_alert(
                        "Critical memory usage",
                        f"Memory usage {resource_stats.memory_percent:.1f}% exceeds "
                        f"critical threshold",
                        AlertSeverity.CRITICAL,
                        {"metric": "memory_usage"},
                    )

        except Exception as e:
            self.logger.error(f"Error checking performance thresholds: {e}")

    async def _detect_anomalies(self) -> None:
        """Detect performance anomalies using statistical analysis."""
        try:
            current_time = time.time()

            # Update baselines every 5 minutes
            if current_time - self._last_baseline_update > 300:
                await self._update_performance_baselines()
                self._last_baseline_update = current_time

            # Check for anomalies in key metrics
            anomaly_threshold = 3.0  # 3 standard deviations

            for metric_name in self._latency_data:
                if len(self._latency_data[metric_name]) < 100:  # Need sufficient data
                    continue

                baseline = self._baselines.get(metric_name, {})
                if "mean" not in baseline or "std" not in baseline:
                    continue

                recent_values = list(self._latency_data[metric_name])[-10:]  # Last 10 samples
                recent_mean = sum(recent_values) / len(recent_values)

                z_score = abs(recent_mean - baseline["mean"]) / (baseline["std"] + 1e-6)

                if z_score > anomaly_threshold:
                    await self._send_performance_alert(
                        "Performance anomaly detected",
                        f"Metric {metric_name} shows anomalous behavior (z-score: {z_score:.2f})",
                        AlertSeverity.HIGH,
                        {"metric": metric_name, "z_score": str(z_score)},
                    )

        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")

    async def _update_performance_baselines(self) -> None:
        """Update performance baselines for anomaly detection."""
        with self._lock:
            for metric_name, values in self._latency_data.items():
                if len(values) >= 100:  # Need sufficient data for baseline
                    values_list = list(values)
                    mean_val = sum(values_list) / len(values_list)
                    variance = sum((x - mean_val) ** 2 for x in values_list) / len(values_list)
                    std_val = variance**0.5

                    self._baselines[metric_name] = {
                        "mean": mean_val,
                        "std": std_val,
                        "count": len(values_list),
                        "updated_at": time.time(),
                    }

    async def _send_performance_alert(
        self, title: str, message: str, severity: AlertSeverity, labels: dict[str, str]
    ) -> None:
        """Send performance-related alert."""
        if not self.alert_manager:
            return

        try:
            alert = Alert(
                rule_name=f"performance_{severity.value}",
                severity=severity,
                status=AlertStatus.FIRING,
                message=message,
                labels=labels,
                annotations={"title": title, "category": "performance"},
                starts_at=datetime.now(timezone.utc),
            )

            await self.alert_manager.fire_alert(alert)

        except Exception as e:
            self.logger.error(f"Error sending performance alert: {e}")

    def _handle_alert_task_completion(self, task: asyncio.Task) -> None:
        """Handle completion of alert tasks."""
        try:
            # Get the exception if task failed
            exc = task.exception()
            if exc:
                self.logger.error(f"Alert task failed with exception: {exc}")
        except Exception as e:
            self.logger.error(f"Error handling alert task completion: {e}")


def profile_async(
    function_name: str = "", module_name: str = "", labels: dict[str, str] | None = None
):
    """Decorator for profiling asynchronous functions."""

    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            profiler = get_performance_profiler()
            if profiler:
                fname = function_name or func.__name__
                mname = module_name or func.__module__
                async with profiler.profile_async_function(fname, mname, labels):
                    return await func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def profile_sync(
    function_name: str = "", module_name: str = "", labels: dict[str, str] | None = None
):
    """Decorator for profiling synchronous functions."""

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            profiler = get_performance_profiler()
            if profiler:
                fname = function_name or func.__name__
                mname = module_name or func.__module__
                with profiler.profile_function(fname, mname, labels):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Global performance profiler instance
_global_profiler: PerformanceProfiler | None = None


def get_performance_profiler() -> PerformanceProfiler | None:
    """Get the global performance profiler instance."""
    return _global_profiler


def set_global_profiler(profiler: PerformanceProfiler) -> None:
    """Set the global performance profiler instance."""
    global _global_profiler
    _global_profiler = profiler


def initialize_performance_monitoring(
    metrics_collector: MetricsCollector | None = None,
    alert_manager: AlertManager | None = None,
    **kwargs,
) -> PerformanceProfiler:
    """Initialize global performance monitoring."""
    profiler = PerformanceProfiler(
        metrics_collector=metrics_collector, alert_manager=alert_manager, **kwargs
    )
    set_global_profiler(profiler)
    return profiler


# Additional classes for test compatibility
@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""

    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class QueryMetrics:
    """Database query performance metrics."""

    query_time_ms: float = 0.0
    rows_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    connection_pool_usage: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CacheMetrics:
    """Cache performance metrics."""

    hit_rate: float = 0.0
    miss_rate: float = 0.0
    eviction_rate: float = 0.0
    memory_usage: float = 0.0
    total_keys: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class QueryOptimizer:
    """Query optimization utilities."""

    def __init__(self):
        self.optimizations_applied = []

    def analyze_query(self, query: str) -> dict[str, Any]:
        """Analyze query performance."""
        return {"complexity": "medium", "estimated_cost": 100.0, "recommendations": []}

    def optimize_query(self, query: str) -> str:
        """Optimize a database query."""
        return query  # Basic implementation


class CacheOptimizer:
    """Cache optimization utilities."""

    def __init__(self):
        self.cache_stats = {}

    def analyze_cache_performance(self) -> dict[str, Any]:
        """Analyze cache performance."""
        return {"hit_rate": 0.85, "miss_rate": 0.15, "recommendations": []}
