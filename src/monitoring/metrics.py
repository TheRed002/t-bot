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
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import psutil

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
        def __init__(self, *args, **kwargs):
            pass

        def inc(self, value=1):
            pass

        def set(self, value):
            pass

        def observe(self, value):
            pass

        def labels(self, **kwargs):
            return self

    Counter = Gauge = Histogram = Summary = MockMetric
    CollectorRegistry = type("MockRegistry", (), {})

    def generate_latest(registry):
        return "# Mock metrics\n"

    CONTENT_TYPE_LATEST = "text/plain"

    def start_http_server(port, host="0.0.0.0"):
        pass


from src.core.exceptions import MonitoringError
from src.core.logging import get_logger
from src.core.types import OrderStatus, OrderType

logger = get_logger(__name__)


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


class MetricsCollector:
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
        self.registry = registry or CollectorRegistry()
        self._metrics: dict[str, Any] = {}
        self._metric_definitions: dict[str, MetricDefinition] = {}
        self._lock = threading.RLock()
        self._running = False
        self._collection_interval = 5.0  # seconds
        self._background_task: asyncio.Task | None = None

        # Initialize core metric collections
        self.trading_metrics = TradingMetrics(self)
        self.system_metrics = SystemMetrics(self)
        self.exchange_metrics = ExchangeMetrics(self)
        self.risk_metrics = RiskMetrics(self)

        # Cache for performance optimization
        self._metric_cache: dict[str, Any] = {}
        self._cache_ttl = 1.0  # seconds
        self._last_cache_update = 0.0

        logger.info("MetricsCollector initialized")

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
                    logger.warning(f"Metric {full_name} already registered")
                    return

                # Create metric based on type
                if definition.metric_type == "counter":
                    metric = Counter(
                        full_name,
                        definition.description,
                        labelnames=definition.labels,
                        registry=self.registry,
                    )
                elif definition.metric_type == "gauge":
                    metric = Gauge(
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
                    metric = Histogram(
                        full_name,
                        definition.description,
                        labelnames=definition.labels,
                        buckets=buckets,
                        registry=self.registry,
                    )
                elif definition.metric_type == "summary":
                    metric = Summary(
                        full_name,
                        definition.description,
                        labelnames=definition.labels,
                        registry=self.registry,
                    )
                else:
                    raise MonitoringError(f"Unknown metric type: {definition.metric_type}")

                self._metrics[full_name] = metric
                self._metric_definitions[full_name] = definition

                logger.debug(f"Registered metric: {full_name}")

        except Exception as e:
            raise MonitoringError(f"Failed to register metric {definition.name}: {e}")

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
        Increment a counter metric.

        Args:
            name: Counter name
            labels: Label values
            value: Increment value
            namespace: Metric namespace
        """
        metric = self.get_metric(name, namespace)
        if metric and hasattr(metric, "inc"):
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)

    def set_gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None, namespace: str = "tbot"
    ) -> None:
        """
        Set a gauge metric value.

        Args:
            name: Gauge name
            value: Value to set
            labels: Label values
            namespace: Metric namespace
        """
        metric = self.get_metric(name, namespace)
        if metric and hasattr(metric, "set"):
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)

    def observe_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None, namespace: str = "tbot"
    ) -> None:
        """
        Observe a histogram metric.

        Args:
            name: Histogram name
            value: Value to observe
            labels: Label values
            namespace: Metric namespace
        """
        metric = self.get_metric(name, namespace)
        if metric and hasattr(metric, "observe"):
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)

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
            def __init__(self, collector, metric_name, metric_labels, metric_namespace):
                self.collector = collector
                self.metric_name = metric_name
                self.labels = metric_labels
                self.namespace = metric_namespace
                self.start_time = None

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.start_time:
                    duration = time.time() - self.start_time
                    self.collector.observe_histogram(
                        self.metric_name, duration, self.labels, self.namespace
                    )

        return Timer(self, name, labels, namespace)

    async def start_collection(self) -> None:
        """Start background metrics collection."""
        if self._running:
            logger.warning("Metrics collection already running")
            return

        self._running = True
        self._background_task = asyncio.create_task(self._collection_loop())
        logger.info("Started metrics collection")

    async def stop_collection(self) -> None:
        """Stop background metrics collection."""
        self._running = False

        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None

        logger.info("Stopped metrics collection")

    async def _collection_loop(self) -> None:
        """Background loop for collecting system metrics."""
        while self._running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self._collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(1.0)

    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.set_gauge("system_cpu_usage_percent", cpu_percent)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.set_gauge("system_memory_usage_bytes", memory.used)
            self.set_gauge("system_memory_total_bytes", memory.total)
            self.set_gauge("system_memory_usage_percent", memory.percent)

            # Disk metrics
            disk = psutil.disk_usage("/")
            self.set_gauge("system_disk_usage_bytes", disk.used)
            self.set_gauge("system_disk_total_bytes", disk.total)
            self.set_gauge("system_disk_usage_percent", (disk.used / disk.total) * 100)

            # Network metrics
            network = psutil.net_io_counters()
            self.increment_counter("system_network_bytes_sent_total", value=network.bytes_sent)
            self.increment_counter("system_network_bytes_recv_total", value=network.bytes_recv)

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

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


class TradingMetrics:
    """Trading-specific metrics collection."""

    def __init__(self, collector: MetricsCollector):
        """
        Initialize trading metrics.

        Args:
            collector: Parent metrics collector
        """
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
        Record order metrics.

        Args:
            exchange: Exchange name
            status: Order status
            order_type: Order type
            symbol: Trading symbol
            execution_time: Order execution time in seconds
            slippage_bps: Slippage in basis points
        """
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
            self.collector.observe_histogram(
                "order_execution_duration_seconds", execution_time, execution_labels
            )

        if slippage_bps is not None:
            self.collector.observe_histogram("order_slippage_bps", slippage_bps, execution_labels)

    def record_trade(
        self, exchange: str, strategy: str, symbol: str, pnl_usd: float, volume_usd: float
    ) -> None:
        """
        Record trade metrics.

        Args:
            exchange: Exchange name
            strategy: Strategy name
            symbol: Trading symbol
            pnl_usd: P&L in USD
            volume_usd: Trade volume in USD
        """
        pnl_labels = {"exchange": exchange, "strategy": strategy, "symbol": symbol}
        volume_labels = {"exchange": exchange, "symbol": symbol}

        self.collector.observe_histogram("trades_pnl_usd", pnl_usd, pnl_labels)
        self.collector.observe_histogram("trades_volume_usd", volume_usd, volume_labels)

    def update_portfolio_metrics(
        self, exchange: str, value_usd: float, pnl_usd: float, timeframe: str = "1d"
    ) -> None:
        """
        Update portfolio metrics.

        Args:
            exchange: Exchange name
            value_usd: Portfolio value in USD
            pnl_usd: Portfolio P&L in USD
            timeframe: P&L timeframe
        """
        self.collector.set_gauge("portfolio_value_usd", value_usd, {"exchange": exchange})
        self.collector.set_gauge(
            "portfolio_pnl_usd", pnl_usd, {"exchange": exchange, "timeframe": timeframe}
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


class SystemMetrics:
    """System-level metrics collection."""

    def __init__(self, collector: MetricsCollector):
        """
        Initialize system metrics.

        Args:
            collector: Parent metrics collector
        """
        self.collector = collector
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize system-level metrics."""
        metrics = [
            # Application metrics
            MetricDefinition(
                "application_uptime_seconds", "Application uptime in seconds", "gauge"
            ),
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


class ExchangeMetrics:
    """Exchange-specific metrics collection."""

    def __init__(self, collector: MetricsCollector):
        """
        Initialize exchange metrics.

        Args:
            collector: Parent metrics collector
        """
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


class RiskMetrics:
    """Risk management metrics collection."""

    def __init__(self, collector: MetricsCollector):
        """
        Initialize risk metrics.

        Args:
            collector: Parent metrics collector
        """
        self.collector = collector
        self._initialize_metrics()

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


def setup_prometheus_server(port: int = 8001, host: str = "0.0.0.0") -> None:
    """
    Setup Prometheus metrics HTTP server.

    Args:
        port: Server port
        host: Server host
    """
    try:
        start_http_server(port, host)
        logger.info(f"Prometheus metrics server started on {host}:{port}")
    except Exception as e:
        logger.error(f"Failed to start Prometheus server: {e}")
        raise MonitoringError(f"Failed to start Prometheus server: {e}")
