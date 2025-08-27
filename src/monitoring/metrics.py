"""
Prometheus metrics collection infrastructure for T-Bot Trading System.

This module provides comprehensive metrics collection for all system components
including trading operations, system performance, exchange health, and risk metrics.

Key Features:
- Custom Prometheus metrics for trading operations
- System resource monitoring
- Exchange API performance tracking
- Risk management metrics
- High-performance metric collection with minimal latency impact
"""

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any

import psutil

from src.core import BaseComponent

# Import core types and exceptions directly
from src.core.exceptions import MonitoringError
from src.core.types.trading import OrderStatus, OrderType

# Import utils decorators and helpers for better integration
# Import error handling
from src.error_handling.context import ErrorContext, ErrorSeverity
from src.monitoring.financial_precision import safe_decimal_to_float, validate_financial_range
from src.utils.decorators import cache_result, logged, monitored, retry
from src.utils.validators import (
    validate_null_handling,
    validate_type_conversion,
)

# Try to import Prometheus client, fall back gracefully if not available
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Summary,
        generate_latest,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    # Mock Prometheus classes when not available
    PROMETHEUS_AVAILABLE = False

    class MockMetric:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._name = args[0] if args else "unknown"
            self._description = args[1] if len(args) > 1 else "Mock metric"
            self._fallback_storage = []  # Store metrics when Prometheus unavailable
            self._logger = logging.getLogger(__name__)
            
        def inc(self, value: float = 1) -> None:
            self._store_fallback_metric("counter", "inc", {"value": value})
            
        def set(self, value: float) -> None:
            self._store_fallback_metric("gauge", "set", {"value": value})
            
        def observe(self, value: float) -> None:
            self._store_fallback_metric("histogram", "observe", {"value": value})
            
        def labels(self, **kwargs: Any) -> "MockMetric":
            mock = MockMetric(self._name, self._description)
            mock._fallback_storage = self._fallback_storage
            mock._logger = self._logger
            return mock
            
        def _store_fallback_metric(self, metric_type: str, operation: str, data: dict) -> None:
            """Store metric data when Prometheus is unavailable."""
            metric_data = {
                "timestamp": time.time(),
                "metric_name": self._name,
                "metric_type": metric_type,
                "operation": operation,
                "data": data,
                "correlation_id": str(uuid.uuid4())[:8]
            }
            self._fallback_storage.append(metric_data)
            
            # Log critical metrics that should trigger alerts
            if self._is_critical_metric():
                self._logger.warning(
                    f"PROMETHEUS_UNAVAILABLE: Critical metric {self._name} stored in fallback. "
                    f"Operation: {operation}, Data: {data}, Correlation: {metric_data['correlation_id']}"
                )
            
            # Limit fallback storage to prevent memory issues
            if len(self._fallback_storage) > 10000:
                self._fallback_storage = self._fallback_storage[-5000:]  # Keep last 5000
                self._logger.error(
                    f"Fallback metric storage limit reached for {self._name}. "
                    "Truncated to last 5000 entries. Correlation: {metric_data['correlation_id']}"
                )
                
        def _is_critical_metric(self) -> bool:
            """Determine if this metric is critical and should trigger alerts."""
            critical_patterns = [
                "error", "failure", "exception", "circuit_breaker", "limit_violation",
                "portfolio_value", "risk_", "order_", "trade_", "pnl"
            ]
            return any(pattern in self._name.lower() for pattern in critical_patterns)

    # Use type: ignore to avoid redefinition errors
    Counter = MockMetric  # type: ignore[assignment,misc]
    Gauge = MockMetric  # type: ignore[assignment,misc]
    Histogram = MockMetric  # type: ignore[assignment,misc]
    Summary = MockMetric  # type: ignore[assignment,misc]
    CollectorRegistry = type("CollectorRegistry", (), {})  # type: ignore[assignment,misc]

    def generate_latest(registry: Any = None) -> bytes:  # type: ignore[misc]
        return b"# Mock metrics\n"

    CONTENT_TYPE_LATEST = "text/plain"

    def start_http_server(port: int, addr: str = "0.0.0.0", registry: Any = None) -> None:  # type: ignore[misc]
        pass


class MetricType(Enum):
    """Metric types for different components."""

    TRADING = "trading"
    SYSTEM = "system"
    EXCHANGE = "exchange"
    RISK = "risk"
    PERFORMANCE = "performance"
    ML = "ml"


@dataclass
class MetricDefinition:
    """Definition for a custom metric."""

    name: str
    description: str
    metric_type: str  # counter, gauge, histogram, summary
    labels: list[str] = field(default_factory=list)
    buckets: list[float] | None = None
    namespace: str = "tbot"


class MetricsCollector(BaseComponent):
    """
    Central metrics collector for the T-Bot trading system.

    Provides high-performance metrics collection with minimal overhead
    and comprehensive coverage of all system components.
    """

    def __init__(self, registry: CollectorRegistry | None = None):
        """
        Initialize metrics collector.

        Args:
            registry: Prometheus collector registry. If None, uses default registry.
        """
        super().__init__(name="MetricsCollector")  # Initialize BaseComponent with name
        self.registry = registry or CollectorRegistry()
        self._metrics: dict[str, Any] = {}
        self._metric_definitions: dict[str, MetricDefinition] = {}
        self._lock = threading.RLock()
        self._running = False
        self._collection_interval = 5.0  # seconds
        self._background_task: asyncio.Task[None] | None = None
        self.trading_metrics = TradingMetrics(self)
        self.system_metrics = SystemMetrics(self)
        self.exchange_metrics = ExchangeMetrics(self)
        self.risk_metrics = RiskMetrics(self)

        # Cache for performance optimization
        self._metric_cache: dict[str, Any] = {}
        self._cache_ttl = 1.0  # seconds
        self._last_cache_update = 0.0

        # Error handler for error tracking
        self._error_handler: Any | None = None

        self.logger.debug("MetricsCollector initialized")

        # Register error handling metrics
        self._register_error_handling_metrics()
        
        # Register system monitoring metrics
        self._register_system_monitoring_metrics()

    def register_metric(self, definition: MetricDefinition) -> None:
        """
        Register a new metric definition.

        Args:
            definition: Metric definition to register

        Raises:
            MonitoringError: If metric registration fails
        """
        try:
            with self._lock:
                full_name = f"{definition.namespace}_{definition.name}"

                if full_name in self._metrics:
                    self.logger.warning(f"Metric {full_name} already registered")
                    return

                # Create metric based on type
                if definition.metric_type == "counter":
                    new_metric: Any = Counter(
                        full_name,
                        definition.description,
                        labelnames=definition.labels,
                        registry=self.registry,
                    )
                elif definition.metric_type == "gauge":
                    new_metric: Any = Gauge(
                        full_name,
                        definition.description,
                        labelnames=definition.labels,
                        registry=self.registry,
                    )
                elif definition.metric_type == "histogram":
                    buckets = definition.buckets or [
                        0.001,
                        0.005,
                        0.01,
                        0.025,
                        0.05,
                        0.1,
                        0.25,
                        0.5,
                        1.0,
                        2.5,
                        5.0,
                        10.0,
                    ]
                    new_metric: Any = Histogram(
                        full_name,
                        definition.description,
                        labelnames=definition.labels,
                        buckets=buckets,
                        registry=self.registry,
                    )
                elif definition.metric_type == "summary":
                    new_metric: Any = Summary(
                        full_name,
                        definition.description,
                        labelnames=definition.labels,
                        registry=self.registry,
                    )
                else:
                    raise MonitoringError(
                        f"Unknown metric type: {definition.metric_type}",
                        error_code="MONITORING_002",
                    )

                self._metrics[full_name] = new_metric
                self._metric_definitions[full_name] = definition

                self.logger.debug(f"Registered metric: {full_name}")

        except Exception as e:
            raise MonitoringError(f"Failed to register metric {definition.name}: {e}") from e

    def get_metric(self, name: str, namespace: str = "tbot") -> Any | None:
        """
        Get a registered metric.

        Args:
            name: Metric name
            namespace: Metric namespace

        Returns:
            Prometheus metric object or None if not found
        """
        full_name = f"{namespace}_{name}"
        return self._metrics.get(full_name)

    def increment_counter(
        self,
        name: str,
        labels: dict[str, str] | None = None,
        value: float = 1.0,
        namespace: str = "tbot",
    ) -> None:
        """
        Increment a counter metric with enhanced error handling.

        Args:
            name: Counter name
            labels: Label values
            value: Increment value
            namespace: Metric namespace
        """
        try:
            metric = self.get_metric(name, namespace)
            if metric and hasattr(metric, "inc"):
                if labels:
                    metric.labels(**labels).inc(value)
                else:
                    metric.inc(value)
            elif not PROMETHEUS_AVAILABLE:
                # Fallback metric recording is handled by MockMetric
                pass
            else:
                correlation_id = str(uuid.uuid4())[:8]
                self.logger.warning(
                    f"Counter metric '{namespace}_{name}' not found or invalid. "
                    f"Correlation: {correlation_id}"
                )
        except Exception as e:
            correlation_id = str(uuid.uuid4())[:8]
            self.logger.error(
                f"Failed to increment counter '{namespace}_{name}': {type(e).__name__}: {e}. "
                f"Correlation: {correlation_id}"
            )
            # Track metric operation failures
            if name != "metrics_collection_errors_total":  # Avoid infinite recursion
                try:
                    error_metric = self.get_metric("metrics_collection_errors_total")
                    if error_metric:
                        error_labels = {
                            "component": "MetricsCollector",
                            "operation": "increment_counter",
                            "error_type": type(e).__name__,
                            "correlation_id": correlation_id,
                        }
                        error_metric.labels(**error_labels).inc(1)
                except Exception:
                    # Last resort - just log
                    pass

    def set_gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None, namespace: str = "tbot"
    ) -> None:
        """
        Set a gauge metric value with enhanced error handling.

        Args:
            name: Gauge name
            value: Value to set
            labels: Label values
            namespace: Metric namespace
        """
        try:
            metric = self.get_metric(name, namespace)
            if metric and hasattr(metric, "set"):
                if labels:
                    metric.labels(**labels).set(value)
                else:
                    metric.set(value)
            elif not PROMETHEUS_AVAILABLE:
                # Fallback metric recording is handled by MockMetric
                pass
            else:
                correlation_id = str(uuid.uuid4())[:8]
                self.logger.warning(
                    f"Gauge metric '{namespace}_{name}' not found or invalid. "
                    f"Correlation: {correlation_id}"
                )
        except Exception as e:
            correlation_id = str(uuid.uuid4())[:8]
            self.logger.error(
                f"Failed to set gauge '{namespace}_{name}': {type(e).__name__}: {e}. "
                f"Correlation: {correlation_id}"
            )
            # Track metric operation failures
            if name != "metrics_collection_errors_total":
                try:
                    error_metric = self.get_metric("metrics_collection_errors_total")
                    if error_metric:
                        error_labels = {
                            "component": "MetricsCollector",
                            "operation": "set_gauge",
                            "error_type": type(e).__name__,
                            "correlation_id": correlation_id,
                        }
                        error_metric.labels(**error_labels).inc(1)
                except Exception:
                    pass

    def observe_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None, namespace: str = "tbot"
    ) -> None:
        """
        Observe a histogram metric with enhanced error handling.

        Args:
            name: Histogram name
            value: Value to observe
            labels: Label values
            namespace: Metric namespace
        """
        try:
            metric = self.get_metric(name, namespace)
            if metric and hasattr(metric, "observe"):
                if labels:
                    metric.labels(**labels).observe(value)
                else:
                    metric.observe(value)
            elif not PROMETHEUS_AVAILABLE:
                # Fallback metric recording is handled by MockMetric
                pass
            else:
                correlation_id = str(uuid.uuid4())[:8]
                self.logger.warning(
                    f"Histogram metric '{namespace}_{name}' not found or invalid. "
                    f"Correlation: {correlation_id}"
                )
        except Exception as e:
            correlation_id = str(uuid.uuid4())[:8]
            self.logger.error(
                f"Failed to observe histogram '{namespace}_{name}': {type(e).__name__}: {e}. "
                f"Correlation: {correlation_id}"
            )
            # Track metric operation failures
            if name != "metrics_collection_errors_total":
                try:
                    error_metric = self.get_metric("metrics_collection_errors_total")
                    if error_metric:
                        error_labels = {
                            "component": "MetricsCollector",
                            "operation": "observe_histogram",
                            "error_type": type(e).__name__,
                            "correlation_id": correlation_id,
                        }
                        error_metric.labels(**error_labels).inc(1)
                except Exception:
                    pass

    def time_operation(
        self, name: str, labels: dict[str, str] | None = None, namespace: str = "tbot"
    ):
        """
        Context manager for timing operations.

        Args:
            name: Histogram metric name
            labels: Label values
            namespace: Metric namespace

        Returns:
            Context manager that times the operation
        """

        class Timer:
            def __init__(
                self,
                collector: "MetricsCollector",
                metric_name: str,
                metric_labels: dict[str, str] | None,
                metric_namespace: str,
            ) -> None:
                self.collector = collector
                self.metric_name = metric_name
                self.labels = metric_labels
                self.namespace = metric_namespace
                self.start_time: float | None = None

            def __enter__(self) -> "Timer":
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                if self.start_time:
                    duration = time.time() - self.start_time
                    self.collector.observe_histogram(
                        self.metric_name, duration, self.labels, self.namespace
                    )

        return Timer(self, name, labels, namespace)

    async def start_collection(self) -> None:
        """Start background metrics collection."""
        if self._running:
            self.logger.warning("Metrics collection already running")
            return

        self._running = True
        self._background_task = asyncio.create_task(self._collection_loop())
        self.logger.info("Started metrics collection")

    async def stop_collection(self) -> None:
        """Stop background metrics collection."""
        self._running = False

        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await asyncio.wait_for(self._background_task, timeout=10.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self.logger.warning(
                    "Metrics collection background task did not terminate gracefully"
                )
            except Exception as e:
                self.logger.error(f"Error stopping metrics collection task: {e}")
            finally:
                self._background_task = None

        self.logger.info("Stopped metrics collection")

    async def cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        await self.stop_collection()
        
        # Cleanup Prometheus registry
        if self.registry:
            try:
                # Clear registry collectors
                self.registry._collector_to_names.clear()
                self.registry._names_to_collectors.clear()
            except Exception as e:
                self.logger.warning(f"Error cleaning Prometheus registry: {e}")

        # Clear data structures to prevent memory leaks
        with self._lock:
            self._metrics.clear()
            self._metric_definitions.clear()
            self._metric_cache.clear()
        
        # Clear error handler reference to prevent circular references
        self._error_handler = None

        self.logger.info("Metrics collector cleanup completed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_collection()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    async def _collection_loop(self) -> None:
        """Background loop for collecting system metrics."""
        while self._running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self._collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(1.0)

    @retry(max_attempts=3, delay=1.0)
    @logged(level="debug")
    async def start(self) -> None:
        """Start the metrics collector (BaseComponent interface)."""
        await self.start_collection()

    async def stop(self) -> None:
        """Stop the metrics collector (BaseComponent interface)."""
        await self.stop_collection()

    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics using async-safe operations."""
        correlation_id = str(uuid.uuid4())[:8]
        
        try:
            # Run psutil operations in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()

            # CPU metrics - run in thread pool
            cpu_percent = await loop.run_in_executor(None, psutil.cpu_percent, None)
            self.set_gauge("system_cpu_usage_percent", cpu_percent)

            # Memory metrics - run in thread pool
            memory = await loop.run_in_executor(None, psutil.virtual_memory)
            self.set_gauge("system_memory_usage_bytes", memory.used)
            self.set_gauge("system_memory_total_bytes", memory.total)
            self.set_gauge("system_memory_usage_percent", memory.percent)

            # Disk metrics - run in thread pool
            disk = await loop.run_in_executor(None, psutil.disk_usage, "/")
            self.set_gauge("system_disk_usage_bytes", disk.used)
            self.set_gauge("system_disk_total_bytes", disk.total)
            self.set_gauge("system_disk_usage_percent", disk.percent)

            # Network metrics - run in thread pool
            network = await loop.run_in_executor(None, psutil.net_io_counters)
            self.increment_counter("system_network_bytes_sent_total", value=network.bytes_sent)
            self.increment_counter("system_network_bytes_recv_total", value=network.bytes_recv)

        except Exception as e:
            # Create enhanced error context with correlation ID
            context = ErrorContext(
                error=e,
                component="MetricsCollector",
                operation="collect_system_metrics",
                severity=ErrorSeverity.HIGH,  # System metrics failure is serious
                correlation_id=correlation_id,
                details={
                    "prometheus_available": PROMETHEUS_AVAILABLE,
                    "metrics_count": len(self._metrics),
                    "error_handler_available": self._error_handler is not None
                }
            )
            
            # Record metrics collection failure
            self.increment_counter(
                "system_metrics_collection_errors_total",
                labels={"error_type": type(e).__name__, "correlation_id": correlation_id}
            )
            
            # Enhanced error handling with proper propagation
            handler_success = False
            handler_error = None
            
            if self._error_handler:
                try:
                    if hasattr(self._error_handler, "handle_error"):
                        await self._error_handler.handle_error(e, context)
                        handler_success = True
                    elif hasattr(self._error_handler, "handle_error_sync"):
                        self._error_handler.handle_error_sync(e, context)
                        handler_success = True
                    else:
                        error_msg = f"Error handler has no valid methods. Correlation: {correlation_id}"
                        self.logger.error(error_msg)
                        raise MonitoringError(error_msg, error_code="MONITORING_003")
                        
                except Exception as he:
                    handler_error = he
                    self.logger.error(
                        f"Error handler failed with {type(he).__name__}: {he}. "
                        f"Original error: {type(e).__name__}: {e}. Correlation: {correlation_id}"
                    )
                    
            else:
                error_msg = f"No error handler available for metrics collection error. Correlation: {correlation_id}"
                self.logger.error(error_msg)
                
            # Critical: Don't silently continue if system metrics collection fails repeatedly
            if not handler_success:
                # For critical system monitoring failures, we need to escalate
                if isinstance(e, (MemoryError, OSError)) or "disk" in str(e).lower():
                    critical_error = MonitoringError(
                        f"Critical system metrics collection failure: {e}. Correlation: {correlation_id}",
                        error_code="MONITORING_004"
                    )
                    # Chain original exception and handler error
                    critical_error.__cause__ = e
                    if handler_error:
                        critical_error.__context__ = handler_error
                    raise critical_error
                    
            # Log warning for non-critical failures but don't stop execution
            self.logger.warning(
                f"System metrics collection failed but continuing. "
                f"Error: {type(e).__name__}: {e}. Correlation: {correlation_id}"
            )

    @cache_result(ttl=5)
    @monitored()
    def export_metrics(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus formatted metrics string
        """
        return generate_latest(self.registry).decode("utf-8")

    def get_metrics_content_type(self) -> str:
        """Get content type for metrics response."""
        return CONTENT_TYPE_LATEST

    def _register_error_handling_metrics(self) -> None:
        """Register metrics for error handling system monitoring."""
        error_metrics = [
            MetricDefinition(
                "error_handler_errors_total",
                "Total errors processed by error handler",
                "counter",
                ["component", "operation", "error_type"],
            ),
            MetricDefinition(
                "error_handler_recovery_success_total",
                "Successful error recoveries",
                "counter",
                ["recovery_scenario", "component"],
            ),
            MetricDefinition(
                "error_handler_recovery_failed_total",
                "Failed error recoveries",
                "counter",
                ["recovery_scenario", "component"],
            ),
            MetricDefinition(
                "error_handler_pattern_detections_total",
                "Error pattern detections",
                "counter",
                ["pattern_type", "component"],
            ),
            MetricDefinition(
                "error_handler_circuit_breaker_state",
                "Circuit breaker state (0=closed, 1=open, 2=half-open)",
                "gauge",
                ["component", "operation"],
            ),
        ]

        for metric_def in error_metrics:
            try:
                self.register_metric(metric_def)
            except Exception as e:
                correlation_id = str(uuid.uuid4())[:8]
                self.logger.warning(
                    (
                        f"Failed to register error handling metric '{metric_def.name}': "
                        f"{type(e).__name__}: {e}. Correlation: {correlation_id}"
                    ),
                    extra={
                        "metric_name": metric_def.name,
                        "metric_type": metric_def.metric_type,
                        "error_type": type(e).__name__,
                        "correlation_id": correlation_id,
                    },
                )
                
    def _register_system_monitoring_metrics(self) -> None:
        """Register metrics for monitoring the monitoring system itself."""
        monitoring_metrics = [
            MetricDefinition(
                "prometheus_availability",
                "Prometheus client availability (1=available, 0=mock)",
                "gauge",
            ),
            MetricDefinition(
                "metrics_collection_errors_total",
                "Total metrics collection errors",
                "counter",
                ["component", "operation", "error_type", "correlation_id"],
            ),
            MetricDefinition(
                "system_metrics_collection_errors_total",
                "System metrics collection failures",
                "counter",
                ["error_type", "correlation_id"],
            ),
            MetricDefinition(
                "fallback_metrics_stored_total",
                "Metrics stored in fallback when Prometheus unavailable",
                "counter",
                ["metric_name", "metric_type", "correlation_id"],
            ),
            MetricDefinition(
                "error_handler_failures_total",
                "Error handler execution failures",
                "counter",
                ["handler_type", "correlation_id"],
            ),
        ]
        
        for metric_def in monitoring_metrics:
            try:
                self.register_metric(metric_def)
            except Exception as e:
                correlation_id = str(uuid.uuid4())[:8]
                # Use basic logging since we're setting up the monitoring system
                self.logger.error(
                    f"Failed to register monitoring metric '{metric_def.name}': "
                    f"{type(e).__name__}: {e}. Correlation: {correlation_id}"
                )
                
        # Set Prometheus availability status
        self.set_gauge("prometheus_availability", 1.0 if PROMETHEUS_AVAILABLE else 0.0)


class TradingMetrics(BaseComponent):
    """Trading-specific metrics collection."""

    def __init__(self, collector: MetricsCollector):
        """
        Initialize trading metrics.

        Args:
            collector: Parent metrics collector
        """
        super().__init__(name="TradingMetrics")  # Initialize BaseComponent
        self.collector = collector
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize trading-specific metrics."""
        metrics = [
            # Order metrics
            MetricDefinition(
                "orders_total",
                "Total number of orders by status and exchange",
                "counter",
                ["exchange", "status", "order_type", "symbol"],
            ),
            MetricDefinition(
                "order_execution_duration_seconds",
                "Time taken to execute orders",
                "histogram",
                ["exchange", "order_type", "symbol"],
                [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            ),
            MetricDefinition(
                "order_slippage_bps",
                "Order execution slippage in basis points",
                "histogram",
                ["exchange", "order_type", "symbol"],
                [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0],
            ),
            # Trade metrics
            MetricDefinition(
                "trades_pnl_usd",
                "Profit and loss per trade in USD",
                "histogram",
                ["exchange", "strategy", "symbol"],
                [-1000, -500, -100, -50, -10, 0, 10, 50, 100, 500, 1000],
            ),
            MetricDefinition(
                "trades_volume_usd",
                "Trading volume in USD",
                "histogram",
                ["exchange", "symbol"],
                [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000],
            ),
            # Portfolio metrics
            MetricDefinition(
                "portfolio_value_usd", "Current portfolio value in USD", "gauge", ["exchange"]
            ),
            MetricDefinition(
                "portfolio_pnl_usd", "Portfolio P&L in USD", "gauge", ["exchange", "timeframe"]
            ),
            MetricDefinition(
                "portfolio_exposure_percent",
                "Portfolio exposure percentage",
                "gauge",
                ["exchange", "asset"],
            ),
            # Strategy metrics
            MetricDefinition(
                "strategy_signals_total",
                "Total trading signals generated",
                "counter",
                ["strategy", "signal_type", "symbol"],
            ),
            MetricDefinition(
                "strategy_performance_pnl",
                "Strategy performance P&L",
                "gauge",
                ["strategy", "timeframe"],
            ),
            MetricDefinition(
                "strategy_win_rate_percent",
                "Strategy win rate percentage",
                "gauge",
                ["strategy", "timeframe"],
            ),
        ]

        for metric_def in metrics:
            self.collector.register_metric(metric_def)

    def record_order(
        self,
        exchange: str,
        status: OrderStatus,
        order_type: OrderType,
        symbol: str,
        execution_time: float | None = None,
        slippage_bps: float | None = None,
    ) -> None:
        """
        Record order metrics with trading validation.

        Args:
            exchange: Exchange name
            status: Order status
            order_type: Order type
            symbol: Trading symbol
            execution_time: Order execution time in seconds
            slippage_bps: Slippage in basis points

        Raises:
            ValueError: If metrics contain invalid trading values
        """
        # Validate execution time
        if execution_time is not None:
            if execution_time < 0:
                raise ValueError("Execution time cannot be negative")
            if execution_time > 3600:  # > 1 hour indicates likely error
                self.collector.logger.warning(
                    f"Unusually long execution time: {execution_time:.2f}s for {symbol} order"
                )

        # Validate slippage
        if slippage_bps is not None:
            if abs(slippage_bps) > 10000:  # > 100% slippage indicates error
                raise ValueError(f"Invalid slippage: {slippage_bps} bps (exceeds 100%)")
            if abs(slippage_bps) > 1000:  # > 10% slippage is unusual
                self.collector.logger.warning(
                    f"High slippage recorded: {slippage_bps:.2f} bps for {symbol} on {exchange}"
                )

        # Labels for orders_total counter (includes status)
        order_labels = {
            "exchange": exchange,
            "status": status.value,
            "order_type": order_type.value,
            "symbol": symbol,
        }

        # Labels for histograms (no status)
        execution_labels = {
            "exchange": exchange,
            "order_type": order_type.value,
            "symbol": symbol,
        }

        self.collector.increment_counter("orders_total", order_labels)

        if execution_time is not None:
            # Round to microsecond precision for latency measurements
            execution_time_rounded = round(execution_time, 6)
            self.collector.observe_histogram(
                "order_execution_duration_seconds", execution_time_rounded, execution_labels
            )

        if slippage_bps is not None:
            # Round to 2 decimal places for basis points
            slippage_rounded = round(slippage_bps, 2)
            self.collector.observe_histogram(
                "order_slippage_bps", slippage_rounded, execution_labels
            )

    def record_trade(
        self,
        exchange: str,
        strategy: str,
        symbol: str,
        pnl_usd: Decimal | float,
        volume_usd: Decimal | float,
    ) -> None:
        """
        Record trade metrics with comprehensive financial validation.

        Args:
            exchange: Exchange name
            strategy: Strategy name
            symbol: Trading symbol
            pnl_usd: P&L in USD (Decimal preferred for precision)
            volume_usd: Trade volume in USD (Decimal preferred for precision)

        Raises:
            ValueError: If financial values are invalid
        """
        # Enhanced null handling and validation
        validate_null_handling(pnl_usd, allow_null=False, field_name="pnl_usd")
        validate_null_handling(volume_usd, allow_null=False, field_name="volume_usd")
        validate_null_handling(exchange, allow_null=False, field_name="exchange")
        validate_null_handling(strategy, allow_null=False, field_name="strategy")
        validate_null_handling(symbol, allow_null=False, field_name="symbol")

        # Type conversion with validation
        pnl_decimal = validate_type_conversion(pnl_usd, Decimal, "pnl_usd", strict=False)
        volume_decimal = validate_type_conversion(volume_usd, Decimal, "volume_usd", strict=False)

        # Validate financial ranges before conversion
        validate_financial_range(
            pnl_decimal, "trades_pnl_usd", min_value=-1_000_000, max_value=1_000_000
        )
        validate_financial_range(
            volume_decimal, "trades_volume_usd", min_value=0, max_value=10_000_000
        )

        # Convert to float with precision tracking
        pnl_float = safe_decimal_to_float(pnl_decimal, "trades_pnl_usd", precision_digits=8)
        volume_float = safe_decimal_to_float(
            volume_decimal, "trades_volume_usd", precision_digits=8
        )

        # Log warnings for unusual values
        if abs(pnl_float) > 1_000_000:
            self.collector.logger.warning(
                f"Unusually large P&L recorded: ${pnl_float:,.2f} for {symbol} on {exchange}"
            )

        if volume_float > 10_000_000:  # > $10M per trade
            self.collector.logger.warning(
                f"Unusually large volume recorded: ${volume_float:,.2f} for {symbol} on {exchange}"
            )

        pnl_labels = {"exchange": exchange, "strategy": strategy, "symbol": symbol}
        volume_labels = {"exchange": exchange, "symbol": symbol}

        self.collector.observe_histogram("trades_pnl_usd", pnl_float, pnl_labels)
        self.collector.observe_histogram("trades_volume_usd", volume_float, volume_labels)

    def update_portfolio_metrics(
        self,
        exchange: str,
        value_usd: Decimal | float,
        pnl_usd: Decimal | float,
        timeframe: str = "1d",
    ) -> None:
        """
        Update portfolio metrics with financial validation.

        Args:
            exchange: Exchange name
            value_usd: Portfolio value in USD (Decimal preferred for precision)
            pnl_usd: Portfolio P&L in USD (Decimal preferred for precision)
            timeframe: P&L timeframe

        Raises:
            ValueError: If portfolio values are invalid
        """
        # Validate financial ranges before conversion
        validate_financial_range(
            value_usd, "portfolio_value_usd", min_value=0, max_value=1_000_000_000
        )

        # Convert to float with precision tracking
        value_float = safe_decimal_to_float(value_usd, "portfolio_value_usd", precision_digits=2)
        pnl_float = safe_decimal_to_float(pnl_usd, "portfolio_pnl_usd", precision_digits=2)

        # Validate timeframe
        valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d", "1w", "1M"]
        if timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}")

        # Check for unrealistic portfolio values
        if value_float > 1_000_000_000:  # > $1B portfolio
            self.collector.logger.warning(
                f"Extremely large portfolio value: ${value_float:,.2f} on {exchange}"
            )

        # Calculate P&L percentage for additional validation
        if value_float > 0:
            pnl_percentage = (pnl_float / value_float) * 100
            if abs(pnl_percentage) > 50:  # > 50% P&L change in timeframe
                self.collector.logger.warning(
                    f"Large P&L change: {pnl_percentage:.2f}% over {timeframe} on {exchange}"
                )

        self.collector.set_gauge("portfolio_value_usd", value_float, {"exchange": exchange})
        self.collector.set_gauge(
            "portfolio_pnl_usd", pnl_float, {"exchange": exchange, "timeframe": timeframe}
        )

        # Also record P&L percentage if portfolio has value
        if value_float > 0:
            pnl_pct = safe_decimal_to_float(
                (pnl_float / value_float) * 100, "portfolio_pnl_percent", precision_digits=4
            )
            self.collector.set_gauge(
                "portfolio_pnl_percent",
                pnl_pct,
                {"exchange": exchange, "timeframe": timeframe},
            )

    def record_strategy_signal(self, strategy: str, signal_type: str, symbol: str) -> None:
        """
        Record strategy signal.

        Args:
            strategy: Strategy name
            signal_type: Signal type (buy/sell/hold)
            symbol: Trading symbol
        """
        labels = {"strategy": strategy, "signal_type": signal_type, "symbol": symbol}
        self.collector.increment_counter("strategy_signals_total", labels)

    def record_order_latency(
        self, exchange: str, latency: float, order_type: str | None = None
    ) -> None:
        """
        Record order execution latency.

        Args:
            exchange: Exchange name
            latency: Order execution latency in milliseconds
            order_type: Type of order (optional)
        """
        labels = {"exchange": exchange}
        if order_type:
            labels["order_type"] = order_type

        # Convert milliseconds to seconds for consistency with Prometheus best practices
        latency_seconds = latency / 1000.0

        self.collector.observe_histogram(
            "order_execution_duration_seconds", latency_seconds, labels
        )


class SystemMetrics(BaseComponent):
    """System-level metrics collection."""

    def __init__(self, collector: MetricsCollector):
        """
        Initialize system metrics.

        Args:
            collector: Parent metrics collector
        """
        super().__init__(name="SystemMetrics")  # Initialize BaseComponent
        self.collector = collector
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize system-level metrics."""
        metrics = [
            # Application metrics
            MetricDefinition("app_uptime_seconds", "Application uptime in seconds", "gauge"),
            MetricDefinition(
                "application_info",
                "Application information",
                "gauge",
                ["version", "environment", "build"],
            ),
            # Database metrics
            MetricDefinition(
                "database_connections_active",
                "Active database connections",
                "gauge",
                ["database", "pool"],
            ),
            MetricDefinition(
                "database_query_duration_seconds",
                "Database query duration",
                "histogram",
                ["database", "operation"],
                [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            ),
            # Cache metrics
            MetricDefinition(
                "cache_hit_rate_percent", "Cache hit rate percentage", "gauge", ["cache_type"]
            ),
            MetricDefinition(
                "cache_operations_total",
                "Total cache operations",
                "counter",
                ["cache_type", "operation"],
            ),
        ]

        for metric_def in metrics:
            self.collector.register_metric(metric_def)


class ExchangeMetrics(BaseComponent):
    """Exchange-specific metrics collection."""

    def __init__(self, collector: MetricsCollector):
        """
        Initialize exchange metrics.

        Args:
            collector: Parent metrics collector
        """
        super().__init__(name="ExchangeMetrics")  # Initialize BaseComponent
        self.collector = collector
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize exchange-specific metrics."""
        metrics = [
            # API metrics
            MetricDefinition(
                "exchange_api_requests_total",
                "Total API requests to exchanges",
                "counter",
                ["exchange", "endpoint", "status"],
            ),
            MetricDefinition(
                "exchange_api_response_time_seconds",
                "Exchange API response time",
                "histogram",
                ["exchange", "endpoint"],
                [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            ),
            MetricDefinition(
                "exchange_rate_limit_remaining",
                "Remaining rate limit capacity",
                "gauge",
                ["exchange", "limit_type"],
            ),
            # WebSocket metrics
            MetricDefinition(
                "exchange_websocket_connections",
                "Active WebSocket connections",
                "gauge",
                ["exchange"],
            ),
            MetricDefinition(
                "exchange_websocket_messages_total",
                "Total WebSocket messages",
                "counter",
                ["exchange", "message_type", "direction"],
            ),
            MetricDefinition(
                "exchange_websocket_latency_seconds",
                "WebSocket message latency",
                "histogram",
                ["exchange"],
                [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            ),
            # Health metrics
            MetricDefinition(
                "exchange_health_score", "Exchange health score (0-1)", "gauge", ["exchange"]
            ),
            MetricDefinition(
                "exchange_errors_total",
                "Total exchange errors",
                "counter",
                ["exchange", "error_type"],
            ),
        ]

        for metric_def in metrics:
            self.collector.register_metric(metric_def)

    def record_api_request(
        self, exchange: str, endpoint: str, status: str, response_time: float
    ) -> None:
        """
        Record exchange API request metrics.

        Args:
            exchange: Exchange name
            endpoint: API endpoint
            status: HTTP status code
            response_time: Response time in seconds
        """
        request_labels = {"exchange": exchange, "endpoint": endpoint, "status": status}
        time_labels = {"exchange": exchange, "endpoint": endpoint}

        self.collector.increment_counter("exchange_api_requests_total", request_labels)
        self.collector.observe_histogram(
            "exchange_api_response_time_seconds", response_time, time_labels
        )

    def update_rate_limits(self, exchange: str, limit_type: str, remaining: int) -> None:
        """
        Update rate limit metrics.

        Args:
            exchange: Exchange name
            limit_type: Type of rate limit
            remaining: Remaining capacity
        """
        labels = {"exchange": exchange, "limit_type": limit_type}
        self.collector.set_gauge("exchange_rate_limit_remaining", remaining, labels)

    def record_connection(self, success: bool, exchange: str | None = None) -> None:
        """
        Record exchange connection metrics.

        Args:
            success: Whether the connection was successful
            exchange: Exchange name (optional, for compatibility)
        """
        # Use exchange name from connection event or default
        exchange_name = exchange or "unknown"
        status = "success" if success else "failure"

        # Use connection as endpoint since this tracks connection events
        labels = {"exchange": exchange_name, "endpoint": "connection", "status": status}
        self.collector.increment_counter("exchange_api_requests_total", labels)

    def record_health_check(
        self, success: bool, duration: float | None = None, exchange: str | None = None
    ) -> None:
        """
        Record exchange health check metrics.

        Args:
            success: Whether the health check was successful
            duration: Duration of the health check in seconds
            exchange: Exchange name (optional, for compatibility)
        """
        exchange_name = exchange or "unknown"
        health_score = 1.0 if success else 0.0

        labels = {"exchange": exchange_name}
        self.collector.set_gauge("exchange_health_score", health_score, labels)

        if duration is not None:
            # Record health check duration as API response time
            time_labels = {"exchange": exchange_name, "endpoint": "health_check"}
            self.collector.observe_histogram(
                "exchange_api_response_time_seconds", duration, time_labels
            )

    def record_rate_limit_violation(self, endpoint: str, exchange: str | None = None) -> None:
        """
        Record rate limit violation.

        Args:
            endpoint: API endpoint that hit rate limit
            exchange: Exchange name (optional, for compatibility)
        """
        exchange_name = exchange or "unknown"
        labels = {"exchange": exchange_name, "error_type": "rate_limit"}
        self.collector.increment_counter("exchange_errors_total", labels)

    def record_rate_limit_check(
        self, endpoint: str, weight: int = 1, exchange: str | None = None
    ) -> None:
        """
        Record successful rate limit check.

        Args:
            endpoint: API endpoint being checked
            weight: Request weight
            exchange: Exchange name (optional, for compatibility)
        """
        exchange_name = exchange or "unknown"
        labels = {"exchange": exchange_name, "endpoint": endpoint, "status": "200"}
        self.collector.increment_counter("exchange_api_requests_total", labels)

    def record_order(self, order_type=None, side=None, success=None, **kwargs) -> None:
        """
        Record order metrics - compatibility method that delegates to TradingMetrics.

        This method provides compatibility for existing calls while ensuring proper metrics
        recording. It extracts the necessary parameters and delegates to the
        TradingMetrics.record_order method.

        Args:
            order_type: Order type (required for compatibility)
            side: Order side (required for compatibility)
            success: Whether the order was successful (required for compatibility)
            **kwargs: Additional parameters for future compatibility
        """
        if order_type is None or side is None or success is None:
            self.collector.logger.warning(
                "ExchangeMetrics.record_order called with missing required parameters. "
                "Use TradingMetrics.record_order for full functionality."
            )
            return

        # Extract exchange name from context or use default
        exchange = kwargs.get("exchange", "unknown")
        symbol = kwargs.get("symbol", "unknown")

        # Map success boolean to OrderStatus
        from src.core.types.trading import OrderStatus

        status = OrderStatus.FILLED if success else OrderStatus.REJECTED

        # Get TradingMetrics instance from the collector
        if hasattr(self.collector, "trading_metrics"):
            try:
                self.collector.trading_metrics.record_order(
                    exchange=exchange,
                    status=status,
                    order_type=order_type,
                    symbol=symbol,
                    execution_time=kwargs.get("execution_time"),
                    slippage_bps=kwargs.get("slippage_bps"),
                )
            except Exception as e:
                self.collector.logger.error(
                    f"Failed to record order metrics via TradingMetrics: {e}"
                )
                # Fallback to basic counter increment
                labels = {
                    "exchange": exchange,
                    "status": status.value,
                    "order_type": order_type.value
                    if hasattr(order_type, "value")
                    else str(order_type),
                    "symbol": symbol,
                }
                self.collector.increment_counter("orders_total", labels)
        else:
            self.collector.logger.warning(
                "TradingMetrics not available, using basic order counting"
            )
            # Fallback implementation
            labels = {
                "exchange": exchange,
                "status": status.value,
                "order_type": order_type.value if hasattr(order_type, "value") else str(order_type),
                "symbol": symbol,
            }
            self.collector.increment_counter("orders_total", labels)

    def record_order_latency(
        self, exchange: str, latency: float, order_type: str | None = None
    ) -> None:
        """
        Record order execution latency.

        Args:
            exchange: Exchange name
            latency: Order execution latency in milliseconds
            order_type: Type of order (optional)
        """
        labels = {"exchange": exchange}
        if order_type:
            labels["order_type"] = order_type

        # Convert milliseconds to seconds for consistency with Prometheus best practices
        latency_seconds = latency / 1000.0

        self.collector.observe_histogram("exchange_order_latency_seconds", latency_seconds, labels)


class RiskMetrics(BaseComponent):
    """Risk management metrics collection with financial validation."""

    def __init__(self, collector: MetricsCollector):
        """
        Initialize risk metrics.

        Args:
            collector: Parent metrics collector
        """
        super().__init__(name="RiskMetrics")  # Initialize BaseComponent
        self.collector = collector
        self._initialize_metrics()

    def record_var(self, confidence_level: float, timeframe: str, var_value: float) -> None:
        """
        Record Value at Risk metrics with validation.

        Args:
            confidence_level: VaR confidence level (e.g., 0.95, 0.99)
            timeframe: Time horizon (e.g., '1d', '1w')
            var_value: VaR value in USD

        Raises:
            ValueError: If VaR parameters are invalid
        """
        # Validate confidence level
        if not 0.5 <= confidence_level <= 0.999:
            raise ValueError(
                f"Invalid VaR confidence level: {confidence_level}. Must be between 0.5 and 0.999"
            )

        # Validate VaR value (should be positive for loss potential)
        if var_value < 0:
            self.collector.logger.warning(
                f"Negative VaR value recorded: ${var_value:,.2f}. "
                "VaR should represent potential loss."
            )

        # Check for unrealistic VaR values
        if var_value > 100_000_000:  # > $100M VaR
            self.collector.logger.warning(
                f"Extremely high VaR recorded: ${var_value:,.2f} at {confidence_level} confidence"
            )

        labels = {"confidence_level": str(confidence_level), "timeframe": timeframe}

        var_rounded = round(var_value, 2)
        self.collector.set_gauge("risk_var_usd", var_rounded, labels)

    def record_drawdown(self, timeframe: str, drawdown_pct: float) -> None:
        """
        Record maximum drawdown metrics with validation.

        Args:
            timeframe: Time period for drawdown calculation
            drawdown_pct: Drawdown percentage (positive value)

        Raises:
            ValueError: If drawdown values are invalid
        """
        # Validate drawdown percentage
        if drawdown_pct < 0:
            raise ValueError("Drawdown percentage must be positive (represents loss)")

        if drawdown_pct > 100:
            raise ValueError(f"Invalid drawdown: {drawdown_pct}%. Cannot exceed 100%")

        # Alert on significant drawdown
        if drawdown_pct > 20:  # > 20% drawdown
            self.collector.logger.warning(
                f"High drawdown recorded: {drawdown_pct:.2f}% over {timeframe}"
            )

        drawdown_rounded = round(drawdown_pct, 4)  # Precise to 0.01%
        self.collector.set_gauge(
            "risk_max_drawdown_percent", drawdown_rounded, {"timeframe": timeframe}
        )

    def record_sharpe_ratio(self, timeframe: str, sharpe_ratio: float) -> None:
        """
        Record Sharpe ratio with validation.

        Args:
            timeframe: Time period for Sharpe calculation
            sharpe_ratio: Sharpe ratio value

        Raises:
            ValueError: If Sharpe ratio is unrealistic
        """
        # Validate Sharpe ratio range (practical limits)
        if abs(sharpe_ratio) > 10:
            raise ValueError(f"Unrealistic Sharpe ratio: {sharpe_ratio}. Check calculation.")

        # Log excellent/poor performance
        if sharpe_ratio > 2:
            self.collector.logger.info(
                f"Excellent Sharpe ratio: {sharpe_ratio:.3f} over {timeframe}"
            )
        elif sharpe_ratio < -1:
            self.collector.logger.warning(f"Poor Sharpe ratio: {sharpe_ratio:.3f} over {timeframe}")

        sharpe_rounded = round(sharpe_ratio, 4)
        self.collector.set_gauge("risk_sharpe_ratio", sharpe_rounded, {"timeframe": timeframe})

    def record_position_size(self, exchange: str, symbol: str, size_usd: float) -> None:
        """
        Record position size with risk validation.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            size_usd: Position size in USD

        Raises:
            ValueError: If position size is invalid
        """
        if size_usd < 0:
            raise ValueError("Position size cannot be negative")

        # Alert on very large positions
        if size_usd > 1_000_000:  # > $1M position
            self.collector.logger.warning(
                f"Large position recorded: ${size_usd:,.2f} for {symbol} on {exchange}"
            )

        labels = {"exchange": exchange, "symbol": symbol}
        size_rounded = round(size_usd, 2)
        self.collector.observe_histogram("risk_position_size_usd", size_rounded, labels)

    def _initialize_metrics(self) -> None:
        """Initialize risk management metrics."""
        metrics = [
            # Risk metrics
            MetricDefinition(
                "risk_var_usd", "Value at Risk in USD", "gauge", ["confidence_level", "timeframe"]
            ),
            MetricDefinition(
                "risk_max_drawdown_percent", "Maximum drawdown percentage", "gauge", ["timeframe"]
            ),
            MetricDefinition("risk_sharpe_ratio", "Sharpe ratio", "gauge", ["timeframe"]),
            MetricDefinition(
                "risk_position_size_usd",
                "Position size in USD",
                "histogram",
                ["exchange", "symbol"],
                [100, 500, 1000, 5000, 10000, 25000, 50000, 100000],
            ),
            # Limit violations
            MetricDefinition(
                "risk_limit_violations_total",
                "Total risk limit violations",
                "counter",
                ["limit_type", "severity"],
            ),
            MetricDefinition(
                "risk_circuit_breaker_triggers_total",
                "Circuit breaker triggers",
                "counter",
                ["trigger_type", "exchange"],
            ),
        ]

        for metric_def in metrics:
            self.collector.register_metric(metric_def)


# Global metrics collector instance
_global_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance.

    Returns:
        Global MetricsCollector instance
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def set_metrics_collector(collector: MetricsCollector) -> None:
    """
    Set the global metrics collector instance.

    Args:
        collector: MetricsCollector instance to set globally
    """
    global _global_collector
    _global_collector = collector


def setup_prometheus_server(port: int = 8001, host: str = "0.0.0.0") -> None:
    """
    Setup Prometheus metrics HTTP server.

    Args:
        port: Server port
        host: Server host
    """
    try:
        start_http_server(port, host)
        logger = logging.getLogger(__name__)
        logger.info(f"Prometheus metrics server started on {host}:{port}")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to start Prometheus server: {e}")
        raise MonitoringError(f"Failed to start Prometheus server: {e}") from e
