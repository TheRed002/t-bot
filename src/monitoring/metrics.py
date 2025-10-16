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
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
from enum import Enum
from typing import TYPE_CHECKING, Any

from src.core.base.component import BaseComponent
from src.core.exceptions import MonitoringError, ServiceError
from src.core.logging import get_logger
from src.core.types import OrderStatus, OrderType

# Production-ready imports - only required dependencies
logger = get_logger(__name__)

# Import error handling with service injection pattern - use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from src.core.exceptions import ErrorSeverity
    from src.error_handling.context import ErrorContext
else:
    # Use runtime imports to avoid circular dependencies
    try:
        from src.core.exceptions import ErrorSeverity
    except ImportError:
        # Fallback if error handling module not available
        class _MockErrorSeverity:
            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"
            CRITICAL = "critical"

        ErrorSeverity = _MockErrorSeverity  # type: ignore[assignment,misc]

    try:
        from src.error_handling.context import ErrorContext
    except ImportError:
        # Fallback error context
        class _MockErrorContext:
            def __init__(
                self,
                component: str = "",
                operation: str = "",
                error: Exception | None = None,
                details: dict[str, Any] | None = None,
                severity=None,
                correlation_id: str = "",
            ):
                self.component = component
                self.operation = operation
                self.error = error
                self.details = details
                self.severity = severity or "medium"
                self.correlation_id = correlation_id

            @classmethod
            def from_exception(
                cls,
                error: Exception,
                component: str,
                operation: str,
                severity=None,
                correlation_id: str = "",
                details: dict[str, Any] | None = None,
            ):
                return cls(
                    component=component,
                    operation=operation,
                    error=error,
                    details=details,
                    severity=severity,
                    correlation_id=correlation_id,
                )

        ErrorContext = _MockErrorContext  # type: ignore[assignment,misc]


# Try to import Prometheus client, fall back gracefully if not available
import os

from src.monitoring.config import (
    METRICS_DEFAULT_PROMETHEUS_PORT,
    METRICS_FALLBACK_STORAGE_LIMIT,
)
from src.monitoring.financial_precision import (
    FINANCIAL_CONTEXT,
    validate_financial_range,
)
from src.utils.decimal_utils import decimal_to_float
from src.utils.decorators import cache_result, logged, monitored, retry

# Skip prometheus imports during testing to avoid potential blocking
if os.environ.get("TESTING"):
    # Use mocks during testing
    PROMETHEUS_AVAILABLE = False
else:
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

if not PROMETHEUS_AVAILABLE:
    # Mock Prometheus classes when not available or during testing
    class MockMetric:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._name = args[0] if args else "unknown"
            self._description = args[1] if len(args) > 1 else "Mock metric"
            self._fallback_storage: list[
                dict[str, Any]
            ] = []  # Store metrics when Prometheus unavailable
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
                "correlation_id": str(uuid.uuid4())[:8],
            }
            self._fallback_storage.append(metric_data)

            # Log critical metrics that should trigger alerts
            if self._is_critical_metric():
                self._logger.warning(
                    f"PROMETHEUS_UNAVAILABLE: Critical metric {self._name} stored in fallback. "
                    f"Operation: {operation}, Data: {data}, Correlation: {metric_data['correlation_id']}"
                )

            # Limit fallback storage to prevent memory issues
            if len(self._fallback_storage) > METRICS_FALLBACK_STORAGE_LIMIT:
                self._fallback_storage = self._fallback_storage[
                    -(METRICS_FALLBACK_STORAGE_LIMIT // 2) :
                ]
                self._logger.error(
                    f"Fallback metric storage limit reached for {self._name}. "
                    f"Truncated to last {METRICS_FALLBACK_STORAGE_LIMIT // 2} entries. "
                    f"Correlation: {metric_data['correlation_id']}"
                )

        def _is_critical_metric(self) -> bool:
            """Determine if this metric is critical and should trigger alerts."""
            critical_patterns = [
                "error",
                "failure",
                "exception",
                "circuit_breaker",
                "limit_violation",
                "portfolio_value",
                "risk_",
                "order_",
                "trade_",
                "pnl",
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


# Local utility functions to avoid import dependencies
def validate_null_handling(value: Any, allow_null: bool = False, field_name: str = "value") -> Any:
    """Comprehensive null/None value handling with explicit policies."""
    if value is None:
        if allow_null:
            return None
        else:
            from src.core.exceptions import ValidationError

            raise ValidationError(f"{field_name} cannot be None")

    # Check for other null-like values
    if isinstance(value, str) and value.strip() == "":
        if allow_null:
            return None
        else:
            from src.core.exceptions import ValidationError

            raise ValidationError(f"{field_name} cannot be empty string")

    # Check for NaN values
    if isinstance(value, float) and math.isnan(value):
        if allow_null:
            return None
        else:
            from src.core.exceptions import ValidationError

            raise ValidationError(f"{field_name} cannot be NaN")

    return value


def validate_type_conversion(
    value: Any, target_type: type, field_name: str = "value", strict: bool = True
) -> Any:
    """Validate type conversion with comprehensive error handling."""
    if value is None:
        from src.core.exceptions import ValidationError

        raise ValidationError(f"Cannot convert None {field_name} to {target_type.__name__}")

    try:
        if target_type == Decimal:
            if isinstance(value, Decimal):
                return value
            elif isinstance(value, int | float):
                if isinstance(value, float):
                    if not math.isfinite(value):
                        from src.core.exceptions import ValidationError

                        raise ValidationError(
                            f"Cannot convert non-finite float {field_name} to Decimal"
                        )
                return Decimal(str(value))
            elif isinstance(value, str):
                result = Decimal(value)
                # Check for NaN result which indicates invalid conversion
                if result.is_nan():
                    from src.core.exceptions import ValidationError

                    raise ValidationError(f"Cannot convert invalid string '{value}' to Decimal")
                return result
            else:
                from src.core.exceptions import ValidationError

                raise ValidationError(
                    f"Cannot convert {type(value).__name__} {field_name} to Decimal"
                )
        else:
            result = target_type(value)
            # Check for infinite values when converting to float
            if target_type == float and not math.isfinite(result):
                from src.core.exceptions import ValidationError

                raise ValidationError(
                    f"Conversion of {field_name} to float resulted in non-finite value: {result}"
                )
            return result
    except (ValueError, InvalidOperation, TypeError) as e:
        from src.core.exceptions import ValidationError

        raise ValidationError(f"Type conversion failed for {field_name}: {e}")


class MetricType(Enum):
    """Metric types for different components and Prometheus metrics."""

    # Component types
    TRADING = "trading"
    SYSTEM = "system"
    EXCHANGE = "exchange"
    RISK = "risk"
    PERFORMANCE = "performance"
    ML = "ml"

    # Prometheus metric types
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


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

    def __init__(
        self, registry: CollectorRegistry | None = None, auto_register_metrics: bool = True
    ):
        """
        Initialize metrics collector.

        Args:
            registry: Prometheus collector registry. If None, uses default registry.
            auto_register_metrics: Whether to auto-register default metrics during init.
        """
        super().__init__(name="MetricsCollector")  # Initialize BaseComponent with name
        self.registry = registry or CollectorRegistry()
        self._metrics: dict[str, Any] = {}
        self._metric_definitions: dict[str, MetricDefinition] = {}
        self._lock = threading.RLock()
        self._running = False
        self._collection_interval = 5.0  # seconds
        self._background_task: asyncio.Task[None] | None = None

        # Initialize sub-metrics components based on auto_register_metrics flag
        if auto_register_metrics:
            self.trading_metrics = TradingMetrics(self)
            self.system_metrics = SystemMetrics(self)
            self.exchange_metrics = ExchangeMetrics(self)
            self.risk_metrics = RiskMetrics(self)
        else:
            # Create minimal instances for testing that don't auto-register
            self.trading_metrics = None
            self.system_metrics = None
            self.exchange_metrics = None
            self.risk_metrics = None

        # Cache for performance optimization
        self._metric_cache: dict[str, Any] = {}
        self._cache_ttl = 1.0  # seconds
        self._last_cache_update = 0.0

        # Error handler for error tracking
        self._error_handler: Any | None = None

        self.logger.debug("MetricsCollector initialized")

        # Register default metrics if requested
        if auto_register_metrics:
            # Register error handling metrics
            self._register_error_handling_metrics()

            # Register system monitoring metrics
            self._register_system_monitoring_metrics()

            # Register analytics metrics
            self._register_analytics_metrics()

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
                    self.logger.debug(f"Metric {full_name} already registered, reusing existing metric")
                    return self._metrics[full_name]

                # Normalize metric type (handle both string and enum values)
                metric_type_str = definition.metric_type
                if hasattr(definition.metric_type, "value"):
                    metric_type_str = definition.metric_type.value

                # Create metric based on type
                new_metric: Any
                if metric_type_str == "counter":
                    new_metric = Counter(
                        full_name,
                        definition.description,
                        labelnames=definition.labels,
                        registry=self.registry,
                    )
                elif metric_type_str == "gauge":
                    new_metric = Gauge(
                        full_name,
                        definition.description,
                        labelnames=definition.labels,
                        registry=self.registry,
                    )
                elif metric_type_str == "histogram":
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
                    new_metric = Histogram(
                        full_name,
                        definition.description,
                        labelnames=definition.labels,
                        buckets=buckets,
                        registry=self.registry,
                    )
                elif metric_type_str == "summary":
                    new_metric = Summary(
                        full_name,
                        definition.description,
                        labelnames=definition.labels,
                        registry=self.registry,
                    )
                else:
                    raise MonitoringError(
                        f"Unknown metric type: {metric_type_str}",
                        error_code="MON_1001",
                    )

                self._metrics[full_name] = new_metric
                self._metric_definitions[full_name] = definition

                self.logger.debug(f"Registered metric: {full_name}")
                return new_metric

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

    def get_all_metrics(self) -> dict[str, Any]:
        """
        Get all registered metrics.

        Returns:
            Dictionary of metric names to metric objects
        """
        with self._lock:
            return dict(self._metrics)

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

        Raises:
            ValidationError: If parameters are invalid
            MonitoringError: If metric operation fails
        """
        # Validate inputs using core validation patterns
        from src.core.exceptions import ValidationError

        if not isinstance(name, str) or not name:
            raise ValidationError(
                "Invalid metric name",
                field_name="name",
                field_value=name,
                expected_type="non-empty str",
            )

        if not isinstance(value, (int, float)) or value < 0:
            raise ValidationError(
                "Invalid counter value",
                field_name="value",
                field_value=value,
                validation_rule="must be non-negative number",
            )

        if labels is not None and not isinstance(labels, dict):
            raise ValidationError(
                "Invalid labels parameter",
                field_name="labels",
                field_value=type(labels).__name__,
                expected_type="dict or None",
            )

        try:
            metric = self.get_metric(name, namespace)
            if metric and hasattr(metric, "inc"):
                if labels:
                    # Validate label values
                    for key, val in labels.items():
                        if not isinstance(val, str):
                            raise ValidationError(
                                f"Invalid label value for key '{key}'",
                                field_name=f"labels.{key}",
                                field_value=val,
                                expected_type="str",
                            )
                    metric.labels(**labels).inc(value)
                else:
                    metric.inc(value)
            elif not PROMETHEUS_AVAILABLE:
                # Fallback metric recording is handled by MockMetric
                pass
            else:
                correlation_id = str(uuid.uuid4())[:8]
                from src.core.exceptions import MonitoringError

                raise MonitoringError(
                    f"Counter metric '{namespace}_{name}' not found or invalid",
                    component_name="MetricsCollector",
                    details={
                        "metric_name": name,
                        "namespace": namespace,
                        "correlation_id": correlation_id,
                    },
                )
        except ValidationError:
            # Re-raise validation errors without modification
            raise
        except Exception as e:
            correlation_id = str(uuid.uuid4())[:8]
            from src.core.exceptions import ComponentError

            # Track metric operation failures using consistent error patterns
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
                except Exception as fallback_error:
                    # Last resort - just log
                    self.logger.debug(f"Failed to record metrics error: {fallback_error}")

            raise ComponentError(
                f"Failed to increment counter '{namespace}_{name}': {e}",
                component="MetricsCollector",
                operation="increment_counter",
                context={
                    "metric_name": name,
                    "namespace": namespace,
                    "value": value,
                    "labels": labels,
                    "correlation_id": correlation_id,
                    "original_error": str(e),
                },
            ) from e

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        namespace: str = "tbot",
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
                except Exception as fallback_error:
                    self.logger.debug(f"Failed to record metrics error: {fallback_error}")

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        namespace: str = "tbot",
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
                except Exception as fallback_error:
                    self.logger.debug(f"Failed to record metrics error: {fallback_error}")

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
                    self._collector.observe_histogram(
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
        # Check if in mock mode - skip I/O operations
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            self.logger.info("Metrics collector running in MOCK_MODE - skipping actual collection")
            # Just keep the loop alive but don't do any real work
            try:
                while self._running:
                    await asyncio.sleep(self._collection_interval)
            except asyncio.CancelledError:
                self.logger.debug("Collection loop cancelled in MOCK_MODE")
            return

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
        """
        Infrastructure method that delegates system metrics collection to service layer.
        This method should only handle infrastructure concerns, not business logic.
        """
        # This is infrastructure code - delegate to service layer for business logic
        if hasattr(self, "system_metrics") and self.system_metrics:
            try:
                # Delegate to SystemMetrics service for actual collection logic
                await self.system_metrics.collect_and_record_system_metrics()
            except Exception as e:
                self.logger.error(f"Failed to collect system metrics via service layer: {e}")
                # Record collection failure
                self.increment_counter(
                    "system_metrics_collection_errors_total",
                    labels={"error_type": type(e).__name__}
                )

    @cache_result(ttl=5)
    @monitored()
    def export_metrics(self) -> bytes:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus formatted metrics as bytes
        """
        return generate_latest(self.registry)

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

    def _register_analytics_metrics(self) -> None:
        """Register metrics for analytics components."""
        analytics_metrics = [
            # Operational analytics metrics
            MetricDefinition(
                "operational_order_events",
                "Order events tracked by operational analytics",
                "counter",
                ["exchange", "event_type", "success"],
            ),
            MetricDefinition(
                "operational_order_execution_time",
                "Order execution time in milliseconds",
                "histogram",
                ["exchange", "event_type"],
                [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
            ),
            MetricDefinition(
                "operational_strategy_events",
                "Strategy events tracked by operational analytics",
                "counter",
                ["strategy", "event_type", "success"],
            ),
            MetricDefinition(
                "operational_market_data_events",
                "Market data events tracked by operational analytics",
                "counter",
                ["exchange", "event_type", "success"],
            ),
            MetricDefinition(
                "operational_market_data_latency",
                "Market data latency in milliseconds",
                "histogram",
                ["exchange", "symbol"],
                [1, 5, 10, 25, 50, 100, 250, 500, 1000],
            ),
            MetricDefinition(
                "operational_errors",
                "Operational errors by component and type",
                "counter",
                ["component", "error_type", "severity"],
            ),
            MetricDefinition(
                "operational_api_calls",
                "API calls tracked by operational analytics",
                "counter",
                ["service", "endpoint", "success"],
            ),
            MetricDefinition(
                "operational_api_response_time",
                "API response time in milliseconds",
                "histogram",
                ["service", "endpoint"],
                [10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
            ),
            MetricDefinition(
                "operational_cpu_usage",
                "CPU usage percentage tracked by operational analytics",
                "gauge",
            ),
            MetricDefinition(
                "operational_memory_usage",
                "Memory usage percentage tracked by operational analytics",
                "gauge",
            ),
            MetricDefinition(
                "operational_disk_usage",
                "Disk usage percentage tracked by operational analytics",
                "gauge",
            ),
            MetricDefinition(
                "operational_database_health",
                "Database health status (1=healthy, 0=unhealthy)",
                "gauge",
            ),
            MetricDefinition(
                "operational_api_health",
                "API health status (1=healthy, 0=unhealthy)",
                "gauge",
            ),
            MetricDefinition(
                "operational_cache_health",
                "Cache health status (1=healthy, 0=unhealthy)",
                "gauge",
            ),
            # Analytics reporting metrics
            MetricDefinition(
                "reports_generated",
                "Total reports generated",
                "counter",
                ["report_type"],
            ),
            # Analytics export metrics
            MetricDefinition(
                "analytics_exports_total",
                "Total analytics data exports",
                "counter",
                ["data_type", "format"],
            ),
            MetricDefinition(
                "analytics_export_size_bytes",
                "Size of analytics exports in bytes",
                "histogram",
                ["data_type", "format"],
                [1024, 10240, 102400, 1048576, 10485760, 104857600],
            ),
            # Analytics alert metrics
            MetricDefinition(
                "analytics_alerts_generated",
                "Alerts generated by analytics",
                "counter",
                ["severity", "metric"],
            ),
            MetricDefinition(
                "analytics_active_alerts_total",
                "Total active analytics alerts",
                "gauge",
            ),
            MetricDefinition(
                "analytics_active_alerts_by_severity",
                "Active alerts by severity level",
                "gauge",
                ["severity"],
            ),
            # Portfolio analytics metrics
            MetricDefinition(
                "portfolio_positions_count",
                "Number of positions in portfolio",
                "gauge",
            ),
            MetricDefinition(
                "portfolio_max_position_weight",
                "Maximum position weight in portfolio",
                "gauge",
            ),
            MetricDefinition(
                "portfolio_concentration_hhi",
                "Portfolio concentration using HHI",
                "gauge",
            ),
        ]

        for metric_def in analytics_metrics:
            try:
                self.register_metric(metric_def)
            except Exception as e:
                correlation_id = str(uuid.uuid4())[:8]
                self.logger.warning(
                    (
                        f"Failed to register analytics metric '{metric_def.name}': "
                        f"{type(e).__name__}: {e}. Correlation: {correlation_id}"
                    ),
                    extra={
                        "metric_name": metric_def.name,
                        "metric_type": metric_def.metric_type,
                        "error_type": type(e).__name__,
                        "correlation_id": correlation_id,
                    },
                )


class TradingMetrics(BaseComponent):
    """Trading-specific metrics collection."""

    def __init__(self, collector: MetricsCollector):
        """
        Initialize trading metrics.

        Args:
            collector: Parent metrics collector
        """
        super().__init__(name="TradingMetrics")  # Initialize BaseComponent
        self._collector = collector
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
                "portfolio_pnl_percent",
                "Portfolio P&L percentage",
                "gauge",
                ["exchange", "timeframe"],
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
            self._collector.register_metric(metric_def)

    def record_order(
        self,
        exchange: str,
        symbol: str,
        order_type: OrderType,
        status: OrderStatus | None = None,
        side: str | None = None,
        quantity: Decimal | None = None,
        price: Decimal | None = None,
        execution_time: float | None = None,
        slippage_bps: float | None = None,
        **kwargs,
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
                self._collector.logger.warning(
                    f"Unusually long execution time: {execution_time:.2f}s for {symbol} order"
                )

        # Validate slippage
        if slippage_bps is not None:
            if abs(slippage_bps) > 10000:  # > 100% slippage indicates error
                raise ValueError(f"Invalid slippage: {slippage_bps} bps (exceeds 100%)")
            if abs(slippage_bps) > 1000:  # > 10% slippage is unusual
                self._collector.logger.warning(
                    f"High slippage recorded: {slippage_bps:.2f} bps for {symbol} on {exchange}"
                )

        # Labels for orders_total counter (includes status)
        order_labels = {
            "exchange": exchange,
            "status": status.value if status else "unknown",
            "order_type": order_type.value,
            "symbol": symbol,
        }

        # Labels for histograms (no status)
        execution_labels = {
            "exchange": exchange,
            "order_type": order_type.value,
            "symbol": symbol,
        }

        self._collector.increment_counter("orders_total", order_labels)

        if execution_time is not None:
            # Round to microsecond precision for latency measurements
            execution_time_rounded = round(execution_time, 6)
            self._collector.observe_histogram(
                "order_execution_duration_seconds", execution_time_rounded, execution_labels
            )

        if slippage_bps is not None:
            # Round to 2 decimal places for basis points
            slippage_rounded = round(slippage_bps, 2)
            self._collector.observe_histogram(
                "order_slippage_bps", slippage_rounded, execution_labels
            )

    def record_trade(
        self,
        exchange: str,
        symbol: str,
        strategy: str | None = None,
        side: str | None = None,
        quantity: Decimal | None = None,
        price: Decimal | None = None,
        fee: Decimal | None = None,
        pnl_usd: Decimal | float | None = None,
        volume_usd: Decimal | float | None = None,
        **kwargs,
    ) -> None:
        """
        Record trade metrics with comprehensive financial validation.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            strategy: Strategy name (optional)
            side: Trade side (optional)
            quantity: Trade quantity (optional)
            price: Trade price (optional)
            fee: Trade fee (optional)
            pnl_usd: P&L in USD (optional)
            volume_usd: Trade volume in USD (optional)

        Raises:
            ValueError: If financial values are invalid
        """
        # Calculate derived values if basic parameters are provided
        if pnl_usd is None and quantity is not None and price is not None:
            # Calculate volume from quantity and price
            volume_usd = quantity * price
            # For testing purposes, assume break-even trade if no P&L specified
            pnl_usd = Decimal("0.0")

        if volume_usd is None and quantity is not None and price is not None:
            volume_usd = quantity * price

        # Set defaults for optional parameters
        if strategy is None:
            strategy = "default_strategy"

        # Enhanced null handling and validation - only validate if values are provided
        if pnl_usd is not None:
            validate_null_handling(pnl_usd, allow_null=False, field_name="pnl_usd")
        if volume_usd is not None:
            validate_null_handling(volume_usd, allow_null=False, field_name="volume_usd")

        validate_null_handling(exchange, allow_null=False, field_name="exchange")
        validate_null_handling(strategy, allow_null=False, field_name="strategy")
        validate_null_handling(symbol, allow_null=False, field_name="symbol")

        # Only process if we have the required financial data
        if pnl_usd is None or volume_usd is None:
            # Just log the trade without financial metrics for test compatibility
            return

        # Type conversion with validation
        pnl_decimal = validate_type_conversion(pnl_usd, Decimal, "pnl_usd", strict=False)
        volume_decimal = validate_type_conversion(volume_usd, Decimal, "volume_usd", strict=False)

        # Validate financial ranges before conversion
        validate_financial_range(
            pnl_decimal, min_value=-1_000_000, max_value=1_000_000, metric_name="trades_pnl_usd"
        )
        validate_financial_range(
            volume_decimal, min_value=0, max_value=10_000_000, metric_name="trades_volume_usd"
        )

        # Convert to float with precision tracking
        pnl_float = decimal_to_float(pnl_decimal)
        volume_float = decimal_to_float(
            volume_decimal, "trades_volume_usd", precision_digits=8
        )

        # Log warnings for unusual values
        if abs(pnl_float) > 1_000_000:
            self._collector.logger.warning(
                f"Unusually large P&L recorded: ${pnl_float:,.2f} for {symbol} on {exchange}"
            )

        if volume_float > 10_000_000:  # > $10M per trade
            self._collector.logger.warning(
                f"Unusually large volume recorded: ${volume_float:,.2f} for {symbol} on {exchange}"
            )

        pnl_labels = {"exchange": exchange, "strategy": strategy, "symbol": symbol}
        volume_labels = {"exchange": exchange, "symbol": symbol}

        self._collector.observe_histogram("trades_pnl_usd", pnl_float, pnl_labels)
        self._collector.observe_histogram("trades_volume_usd", volume_float, volume_labels)

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
            value_usd, min_value=0, max_value=1_000_000_000, metric_name="portfolio_value_usd"
        )

        # Convert to float with precision tracking
        value_float = decimal_to_float(value_usd)
        pnl_float = decimal_to_float(pnl_usd)

        # Validate timeframe
        valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d", "1w", "1M"]
        if timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}")

        # Check for unrealistic portfolio values
        if value_float > 1_000_000_000:  # > $1B portfolio
            self.collector.logger.warning(
                f"Extremely large portfolio value: ${value_float:,.2f} on {exchange}"
            )

        # Calculate P&L percentage for additional validation using Decimal arithmetic
        if value_float > 0:
            # Use Decimal arithmetic for financial calculations to maintain precision
            with localcontext(FINANCIAL_CONTEXT):
                value_decimal = Decimal(str(value_float))
                pnl_decimal_calc = Decimal(str(pnl_float))
                pnl_percentage_decimal = (pnl_decimal_calc / value_decimal) * Decimal("100")
                pnl_percentage = float(pnl_percentage_decimal)

            if abs(pnl_percentage) > 50:  # > 50% P&L change in timeframe
                self._collector.logger.warning(
                    f"Large P&L change: {pnl_percentage:.2f}% over {timeframe} on {exchange}"
                )

        self._collector.set_gauge("portfolio_value_usd", value_float, {"exchange": exchange})
        self._collector.set_gauge(
            "portfolio_pnl_usd", pnl_float, {"exchange": exchange, "timeframe": timeframe}
        )

        # Also record P&L percentage if portfolio has value
        if value_float > 0:
            # Use Decimal arithmetic for precise percentage calculation
            with localcontext(FINANCIAL_CONTEXT):
                value_decimal = Decimal(str(value_float))
                pnl_decimal_calc = Decimal(str(pnl_float))
                pnl_percentage_decimal = (pnl_decimal_calc / value_decimal) * Decimal("100")

            pnl_pct = decimal_to_float(
                pnl_percentage_decimal, "portfolio_pnl_percent", precision_digits=4
            )
            self._collector.set_gauge(
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
        self._collector.increment_counter("strategy_signals_total", labels)

    def record_pnl(
        self,
        strategy: str,
        symbol: str,
        realized_pnl: Decimal | None = None,
        unrealized_pnl: Decimal | None = None,
        **kwargs,
    ) -> None:
        """
        Record P&L metrics.

        Args:
            strategy: Strategy name
            symbol: Trading symbol
            realized_pnl: Realized profit/loss
            unrealized_pnl: Unrealized profit/loss
        """
        if realized_pnl is not None:
            pnl_labels = {"strategy": strategy, "symbol": symbol, "type": "realized"}
            pnl_float = float(realized_pnl)
            self._collector.observe_histogram("trades_pnl_usd", pnl_float, pnl_labels)

        if unrealized_pnl is not None:
            pnl_labels = {"strategy": strategy, "symbol": symbol, "type": "unrealized"}
            pnl_float = float(unrealized_pnl)
            self._collector.observe_histogram("trades_pnl_usd", pnl_float, pnl_labels)

    def record_latency(self, operation: str, exchange: str, latency_ms: float, **kwargs) -> None:
        """
        Record operation latency metrics.

        Args:
            operation: Operation type (e.g., 'order_placement')
            exchange: Exchange name
            latency_ms: Latency in milliseconds
        """
        labels = {"operation": operation, "exchange": exchange}
        latency_seconds = latency_ms / 1000.0
        self._collector.observe_histogram(
            "order_execution_duration_seconds", latency_seconds, labels
        )

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

        # Convert milliseconds to seconds using Decimal precision
        with localcontext(FINANCIAL_CONTEXT):
            latency_decimal = Decimal(str(latency))
            latency_seconds_decimal = latency_decimal / Decimal("1000")
            latency_seconds = decimal_to_float(
                latency_seconds_decimal, "order_execution_latency", precision_digits=6
            )

        self._collector.observe_histogram(
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
        self._collector = collector
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
            self._collector.register_metric(metric_def)

    def record_cpu_usage(self, cpu_percent: float) -> None:
        """
        Record CPU usage metrics.

        Args:
            cpu_percent: CPU usage percentage
        """
        self._collector.set_gauge("system_cpu_usage_percent", cpu_percent)

    def record_memory_usage(self, used_mb: float, total_mb: float) -> None:
        """
        Record memory usage metrics.

        Args:
            used_mb: Memory used in MB
            total_mb: Total memory in MB
        """
        self._collector.set_gauge("system_memory_usage_mb", used_mb)
        self._collector.set_gauge("system_memory_total_mb", total_mb)

        # Calculate usage percentage
        if total_mb > 0:
            usage_percent = (used_mb / total_mb) * 100
            self._collector.set_gauge("system_memory_usage_percent", usage_percent)

    def record_network_io(self, bytes_sent: float, bytes_received: float) -> None:
        """
        Record network I/O metrics.

        Args:
            bytes_sent: Bytes sent
            bytes_received: Bytes received
        """
        self._collector.increment_counter("system_network_bytes_sent_total", value=bytes_sent)
        self._collector.increment_counter(
            "system_network_bytes_received_total", value=bytes_received
        )

    def record_disk_usage(self, mount_point: str, usage_percent: float) -> None:
        """
        Record disk usage metrics.

        Args:
            mount_point: Disk mount point
            usage_percent: Disk usage percentage
        """
        labels = {"mount_point": mount_point}
        self._collector.set_gauge("system_disk_usage_percent", usage_percent, labels)

    async def collect_and_record_system_metrics(self) -> None:
        """
        Business logic for collecting system metrics - moved from MetricsCollector.
        This is where the business logic belongs in the service layer.
        """
        from src.utils.monitoring_helpers import SystemMetricsCollector, generate_correlation_id

        correlation_id = generate_correlation_id()

        try:
            # Use shared system metrics collector to eliminate duplication
            metrics = await SystemMetricsCollector.collect_system_metrics()

            if not metrics:
                self.logger.warning(f"No system metrics collected [correlation: {correlation_id}]")
                return

            # Update metrics using shared utilities for consistent precision handling
            from decimal import Decimal


            # Record CPU usage
            if "cpu_percent" in metrics:
                self.record_cpu_usage(
                    decimal_to_float(
                        Decimal(str(metrics.get("cpu_percent", 0))),
                        "system_cpu_usage_percent",
                        precision_digits=2,
                    )
                )

            # Record memory usage
            if "memory_used" in metrics and "memory_total" in metrics:
                self.record_memory_usage(
                    decimal_to_float(
                        Decimal(str(metrics["memory_used"])),
                        "system_memory_usage_mb",
                        precision_digits=0,
                    ) / (1024 * 1024),  # Convert bytes to MB
                    decimal_to_float(
                        Decimal(str(metrics["memory_total"])),
                        "system_memory_total_mb",
                        precision_digits=0,
                    ) / (1024 * 1024),  # Convert bytes to MB
                )

            # Record network I/O
            if "network_bytes_sent" in metrics and "network_bytes_recv" in metrics:
                self.record_network_io(
                    metrics["network_bytes_sent"],
                    metrics["network_bytes_recv"]
                )

            self.logger.debug(
                f"System metrics collected successfully [correlation: {correlation_id}]"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to collect system metrics: {type(e).__name__}: {e}. "
                f"Correlation: {correlation_id}"
            )
            # Record collection failure
            self._collector.increment_counter(
                "system_metrics_collection_errors_total",
                labels={"error_type": type(e).__name__, "correlation_id": correlation_id},
            )
            raise


class ExchangeMetrics(BaseComponent):
    """Exchange-specific metrics collection."""

    def __init__(self, collector: MetricsCollector):
        """
        Initialize exchange metrics.

        Args:
            collector: Parent metrics collector
        """
        super().__init__(name="ExchangeMetrics")  # Initialize BaseComponent
        self._collector = collector
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
            # WebSocket metrics with enhanced monitoring
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
            MetricDefinition(
                "exchange_websocket_connection_errors_total",
                "WebSocket connection errors",
                "counter",
                ["exchange", "error_type"],
            ),
            MetricDefinition(
                "exchange_websocket_reconnections_total",
                "WebSocket reconnection attempts",
                "counter",
                ["exchange", "reason"],
            ),
            MetricDefinition(
                "exchange_websocket_heartbeat_latency_seconds",
                "WebSocket heartbeat response latency",
                "histogram",
                ["exchange"],
                [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            ),
            MetricDefinition(
                "exchange_websocket_backpressure_events_total",
                "WebSocket backpressure events",
                "counter",
                ["exchange", "severity"],
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
            self._collector.register_metric(metric_def)

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

        self._collector.increment_counter("exchange_api_requests_total", request_labels)
        self._collector.observe_histogram(
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
        self._collector.set_gauge("exchange_rate_limit_remaining", remaining, labels)

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
        self._collector.increment_counter("exchange_api_requests_total", labels)

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
        self._collector.set_gauge("exchange_health_score", health_score, labels)

        if duration is not None:
            # Record health check duration as API response time
            time_labels = {"exchange": exchange_name, "endpoint": "health_check"}
            self._collector.observe_histogram(
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
        self._collector.increment_counter("exchange_errors_total", labels)

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
        self._collector.increment_counter("exchange_api_requests_total", labels)

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
        from src.core.types import OrderStatus

        status = OrderStatus.FILLED if success else OrderStatus.REJECTED

        # Delegate to TradingMetrics service
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
                self._collector.logger.error(
                    f"Failed to record order metrics via TradingMetrics: {e}"
                )
                # Fallback to basic counter increment
                self._record_basic_order_metrics(exchange, status, order_type, symbol)
        else:
            self.collector.logger.warning(
                "TradingMetrics service not available, using basic order counting"
            )
            self._record_basic_order_metrics(exchange, status, order_type, symbol)

    def _record_basic_order_metrics(self, exchange: str, status, order_type, symbol: str) -> None:
        """Record basic order metrics as fallback."""
        labels = {
            "exchange": exchange,
            "status": status.value,
            "order_type": order_type.value if hasattr(order_type, "value") else str(order_type),
            "symbol": symbol,
        }
        self._collector.increment_counter("orders_total", labels)

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

        # Convert milliseconds to seconds using Decimal precision
        with localcontext(FINANCIAL_CONTEXT):
            latency_decimal = Decimal(str(latency))
            latency_seconds_decimal = latency_decimal / Decimal("1000")
            latency_seconds = decimal_to_float(
                latency_seconds_decimal, "exchange_order_latency", precision_digits=6
            )

        self._collector.observe_histogram("exchange_order_latency_seconds", latency_seconds, labels)

    async def record_websocket_connection(
        self, exchange: str, connected: bool, error_type: str | None = None
    ) -> None:
        """
        Record WebSocket connection events with async-safe operations.

        Args:
            exchange: Exchange name
            connected: True if connection successful, False if failed
            error_type: Type of error if connection failed
        """
        if connected:
            self._collector.increment_counter(
                "exchange_websocket_connections", {"exchange": exchange}, 1
            )
        else:
            # Record connection error
            error_labels = {"exchange": exchange, "error_type": error_type or "unknown"}
            self._collector.increment_counter(
                "exchange_websocket_connection_errors_total", error_labels
            )

            # Update connection gauge to 0
            self._collector.set_gauge("exchange_websocket_connections", 0, {"exchange": exchange})

    async def record_websocket_message(
        self, exchange: str, message_type: str, direction: str, latency_seconds: float | None = None
    ) -> None:
        """
        Record WebSocket message with backpressure handling.

        Args:
            exchange: Exchange name
            message_type: Type of message (e.g., 'orderbook', 'trade', 'heartbeat')
            direction: Message direction ('incoming' or 'outgoing')
            latency_seconds: Optional latency in seconds
        """
        labels = {"exchange": exchange, "message_type": message_type, "direction": direction}

        # Increment message counter with timeout protection
        try:
            await asyncio.wait_for(
                self._safe_increment_counter("exchange_websocket_messages_total", labels),
                timeout=0.1,
            )
        except asyncio.TimeoutError:
            # Record backpressure event if metric recording is slow
            backpressure_labels = {"exchange": exchange, "severity": "warning"}
            self._collector.increment_counter(
                "exchange_websocket_backpressure_events_total", backpressure_labels
            )

        # Record latency if provided
        if latency_seconds is not None:
            latency_labels = {"exchange": exchange}
            try:
                await asyncio.wait_for(
                    self._safe_observe_histogram(
                        "exchange_websocket_latency_seconds", latency_seconds, latency_labels
                    ),
                    timeout=0.1,
                )
            except asyncio.TimeoutError:
                # Record severe backpressure for latency metrics
                backpressure_labels = {"exchange": exchange, "severity": "critical"}
                self._collector.increment_counter(
                    "exchange_websocket_backpressure_events_total", backpressure_labels
                )

    async def record_websocket_heartbeat(self, exchange: str, latency_seconds: float) -> None:
        """
        Record WebSocket heartbeat latency for connection health monitoring.

        Args:
            exchange: Exchange name
            latency_seconds: Heartbeat round-trip time in seconds
        """
        labels = {"exchange": exchange}

        # Record heartbeat latency with timeout protection
        try:
            await asyncio.wait_for(
                self._safe_observe_histogram(
                    "exchange_websocket_heartbeat_latency_seconds", latency_seconds, labels
                ),
                timeout=0.1,
            )

            # Update connection health based on heartbeat
            health_score = 1.0 if latency_seconds < 0.1 else max(0.1, 1.0 - (latency_seconds / 1.0))
            self._collector.set_gauge("exchange_health_score", health_score, labels)

        except asyncio.TimeoutError:
            # Heartbeat timeout indicates connection issues
            self._collector.set_gauge("exchange_health_score", 0.1, labels)
            backpressure_labels = {"exchange": exchange, "severity": "critical"}
            self._collector.increment_counter(
                "exchange_websocket_backpressure_events_total", backpressure_labels
            )

    async def record_websocket_reconnection(self, exchange: str, reason: str) -> None:
        """
        Record WebSocket reconnection attempt.

        Args:
            exchange: Exchange name
            reason: Reason for reconnection (e.g., 'timeout', 'error', 'disconnect')
        """
        labels = {"exchange": exchange, "reason": reason}
        self._collector.increment_counter("exchange_websocket_reconnections_total", labels)

    async def _safe_increment_counter(self, metric_name: str, labels: dict) -> None:
        """Safely increment counter in thread pool to prevent blocking."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.collector.increment_counter, metric_name, labels)

    async def _safe_observe_histogram(self, metric_name: str, value: float, labels: dict) -> None:
        """Safely observe histogram in thread pool to prevent blocking."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.collector.observe_histogram, metric_name, value, labels
        )


class RiskMetrics(BaseComponent):
    """Risk management metrics collection with financial validation."""

    def __init__(self, collector: MetricsCollector):
        """
        Initialize risk metrics.

        Args:
            collector: Parent metrics collector
        """
        super().__init__(name="RiskMetrics")  # Initialize BaseComponent
        self._collector = collector
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
        self._collector.set_gauge("risk_var_usd", var_rounded, labels)

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
        self._collector.set_gauge(
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
        self._collector.set_gauge("risk_sharpe_ratio", sharpe_rounded, {"timeframe": timeframe})

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
        self._collector.observe_histogram("risk_position_size_usd", size_rounded, labels)

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
            self._collector.register_metric(metric_def)


# Global metrics collector instance with thread-safe singleton
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    _global_collector: "MetricsCollector" | None = None
else:
    _global_collector = None

# Thread lock for thread-safe singleton creation
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """
    Get metrics collector instance using factory pattern with thread-safe singleton fallback.

    Returns:
        MetricsCollector instance from global singleton or DI container
    """
    # First check if we have a global collector set via set_metrics_collector()
    global _global_collector
    if _global_collector is not None:
        return _global_collector

    try:
        from src.monitoring.dependency_injection import get_monitoring_container

        container = get_monitoring_container()
        return container.resolve(MetricsCollector)
    except (ServiceError, MonitoringError, ImportError, KeyError, ValueError) as e:
        logger.warning(f"Failed to resolve metrics collector from DI container: {e}")
        # Thread-safe fallback to global singleton instance
        if _global_collector is None:
            with _collector_lock:
                # Double-check locking pattern to ensure singleton
                if _global_collector is None:
                    _global_collector = MetricsCollector()
        return _global_collector


def set_metrics_collector(collector: MetricsCollector) -> None:
    """
    Set the global metrics collector instance.

    Note: This is for backward compatibility. Prefer using dependency injection.

    Args:
        collector: MetricsCollector instance to set globally
    """
    global _global_collector
    _global_collector = collector


def setup_prometheus_server(
    port: int = METRICS_DEFAULT_PROMETHEUS_PORT, host: str = "0.0.0.0", addr: str | None = None
) -> None:
    """
    Setup Prometheus metrics HTTP server using factory pattern.

    Args:
        port: Server port
        host: Server host (default parameter)
        addr: Server address (alternative parameter name for compatibility)

    Raises:
        MonitoringError: If server setup fails
    """
    # Support both 'host' and 'addr' parameter names for compatibility
    server_host = addr if addr is not None else host

    try:
        start_http_server(port, server_host)
        logger = logging.getLogger(__name__)
        logger.info(f"Prometheus metrics server started on {server_host}:{port}")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to start Prometheus server: {e}")
        raise MonitoringError(f"Failed to start Prometheus server: {e}") from e
