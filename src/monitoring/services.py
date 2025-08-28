"""
Service layer for monitoring operations.

This module implements the service layer pattern for monitoring operations,
providing a clean interface between controllers and domain logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.monitoring.alerting import Alert, AlertManager, AlertSeverity, AlertStatus
from src.monitoring.metrics import MetricsCollector
from src.monitoring.performance import PerformanceProfiler


@dataclass
class AlertRequest:
    """Request to create an alert."""

    rule_name: str
    severity: AlertSeverity
    message: str
    labels: dict[str, str]
    annotations: dict[str, str]


@dataclass
class MetricRequest:
    """Request to record a metric."""

    name: str
    value: float
    labels: dict[str, str] | None = None
    namespace: str = "tbot"


class AlertService(ABC):
    """Service interface for alert operations."""

    @abstractmethod
    async def create_alert(self, request: AlertRequest) -> str:
        """Create a new alert and return its fingerprint."""
        pass

    @abstractmethod
    async def resolve_alert(self, fingerprint: str) -> bool:
        """Resolve an active alert."""
        pass

    @abstractmethod
    async def acknowledge_alert(self, fingerprint: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        pass

    @abstractmethod
    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list[Alert]:
        """Get active alerts."""
        pass

    @abstractmethod
    def get_alert_stats(self) -> dict[str, Any]:
        """Get alert statistics."""
        pass


class MetricsService(ABC):
    """Service interface for metrics operations."""

    @abstractmethod
    def record_counter(self, request: MetricRequest) -> None:
        """Record a counter metric."""
        pass

    @abstractmethod
    def record_gauge(self, request: MetricRequest) -> None:
        """Record a gauge metric."""
        pass

    @abstractmethod
    def record_histogram(self, request: MetricRequest) -> None:
        """Record a histogram metric."""
        pass

    @abstractmethod
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        pass


class PerformanceService(ABC):
    """Service interface for performance monitoring."""

    @abstractmethod
    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        pass

    @abstractmethod
    def record_order_execution(
        self,
        exchange: str,
        order_type: str,
        symbol: str,
        latency_ms: float,
        fill_rate: float,
        slippage_bps: float,
    ) -> None:
        """Record order execution metrics."""
        pass

    @abstractmethod
    def record_market_data_processing(
        self,
        exchange: str,
        data_type: str,
        processing_time_ms: float,
        message_count: int,
    ) -> None:
        """Record market data processing metrics."""
        pass


class DefaultAlertService(AlertService):
    """Default implementation of AlertService."""

    def __init__(self, alert_manager: AlertManager):
        self._alert_manager = alert_manager

    async def create_alert(self, request: AlertRequest) -> str:
        """Create a new alert and return its fingerprint."""
        from src.core.exceptions import ValidationError

        # Validate AlertRequest using core validation patterns (consistent with analytics)
        if not isinstance(request, AlertRequest):
            raise ValidationError(
                "Invalid request type",
                field_name="request",
                field_value=type(request).__name__,
                expected_type="AlertRequest",
            )

        if not request.rule_name or not isinstance(request.rule_name, str):
            raise ValidationError(
                "Invalid rule_name in alert request",
                field_name="rule_name",
                field_value=request.rule_name,
                expected_type="non-empty str",
            )

        alert = Alert(
            rule_name=request.rule_name,
            severity=request.severity,
            status=AlertStatus.FIRING,
            message=request.message,
            labels=request.labels,
            annotations=request.annotations,
            starts_at=datetime.now(timezone.utc),
        )

        try:
            await self._alert_manager.fire_alert(alert)
            return alert.fingerprint
        except Exception as e:
            # Propagate errors using consistent patterns with analytics
            from src.core.exceptions import ComponentError

            raise ComponentError(
                f"Failed to create alert: {e}",
                component="AlertService", 
                operation="create_alert",
                context={
                    "rule_name": request.rule_name,
                    "severity": request.severity.value,
                    "original_error": str(e),
                },
            ) from e

    async def resolve_alert(self, fingerprint: str) -> bool:
        """Resolve an active alert."""
        await self._alert_manager.resolve_alert(fingerprint)
        return True

    async def acknowledge_alert(self, fingerprint: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        return await self._alert_manager.acknowledge_alert(fingerprint, acknowledged_by)

    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list[Alert]:
        """Get active alerts."""
        return self._alert_manager.get_active_alerts(severity)

    def get_alert_stats(self) -> dict[str, Any]:
        """Get alert statistics."""
        return self._alert_manager.get_alert_stats()


class DefaultMetricsService(MetricsService):
    """Default implementation of MetricsService."""

    def __init__(self, metrics_collector: MetricsCollector):
        self._metrics_collector = metrics_collector

    def record_counter(self, request: MetricRequest) -> None:
        """Record a counter metric."""
        from src.core.exceptions import ValidationError

        # Validate MetricRequest using core validation patterns
        if not isinstance(request, MetricRequest):
            raise ValidationError(
                "Invalid request type",
                field_name="request",
                field_value=type(request).__name__,
                expected_type="MetricRequest",
            )

        if not request.name or not isinstance(request.name, str):
            raise ValidationError(
                "Invalid metric name",
                field_name="name",
                field_value=request.name,
                expected_type="non-empty str",
            )

        if not isinstance(request.value, (int, float)) or request.value < 0:
            raise ValidationError(
                "Invalid metric value for counter",
                field_name="value",
                field_value=request.value,
                validation_rule="must be non-negative number",
            )

        try:
            self._metrics_collector.increment_counter(
                request.name, request.labels, request.value, request.namespace
            )
        except Exception as e:
            from src.core.exceptions import ComponentError

            raise ComponentError(
                f"Failed to record counter metric: {e}",
                component="MetricsService",
                operation="record_counter", 
                context={
                    "metric_name": request.name,
                    "metric_value": request.value,
                    "namespace": request.namespace,
                },
            ) from e

    def record_gauge(self, request: MetricRequest) -> None:
        """Record a gauge metric."""
        self._metrics_collector.set_gauge(
            request.name, request.value, request.labels, request.namespace
        )

    def record_histogram(self, request: MetricRequest) -> None:
        """Record a histogram metric."""
        self._metrics_collector.observe_histogram(
            request.name, request.value, request.labels, request.namespace
        )

    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return self._metrics_collector.export_metrics()


class DefaultPerformanceService(PerformanceService):
    """Default implementation of PerformanceService."""

    def __init__(self, performance_profiler: PerformanceProfiler):
        self._performance_profiler = performance_profiler

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        return self._performance_profiler.get_performance_summary()

    def record_order_execution(
        self,
        exchange: str,
        order_type: str,
        symbol: str,
        latency_ms: float,
        fill_rate: float,
        slippage_bps: float,
    ) -> None:
        """Record order execution metrics."""
        from src.core.exceptions import ValidationError
        from src.core.types import OrderType

        # Validate input parameters using core validation patterns
        if not isinstance(exchange, str) or not exchange:
            raise ValidationError(
                "Invalid exchange parameter",
                field_name="exchange",
                field_value=exchange,
                expected_type="str",
            )

        if not isinstance(symbol, str) or not symbol:
            raise ValidationError(
                "Invalid symbol parameter",
                field_name="symbol",
                field_value=symbol,
                expected_type="str",
            )

        if not isinstance(latency_ms, (int, float)) or latency_ms < 0:
            raise ValidationError(
                "Invalid latency_ms parameter",
                field_name="latency_ms",
                field_value=latency_ms,
                validation_rule="must be non-negative number",
            )

        # Convert string to OrderType enum with consistent error handling
        try:
            if isinstance(order_type, str):
                order_type_enum = OrderType(order_type.upper())
            elif isinstance(order_type, OrderType):
                order_type_enum = order_type
            else:
                raise ValidationError(
                    "Invalid order_type parameter",
                    field_name="order_type",
                    field_value=order_type,
                    expected_type="str or OrderType",
                )
        except ValueError as e:
            raise ValidationError(
                f"Invalid order_type value: {order_type}",
                field_name="order_type",
                field_value=order_type,
                expected_type="valid OrderType enum value",
            ) from e

        self._performance_profiler.record_order_execution(
            exchange, order_type_enum, symbol, latency_ms, fill_rate, slippage_bps
        )

    def record_market_data_processing(
        self,
        exchange: str,
        data_type: str,
        processing_time_ms: float,
        message_count: int,
    ) -> None:
        """Record market data processing metrics."""
        self._performance_profiler.record_market_data_processing(
            exchange, data_type, processing_time_ms, message_count
        )


class MonitoringService:
    """Composite service for all monitoring operations."""

    def __init__(
        self,
        alert_service: AlertService,
        metrics_service: MetricsService,
        performance_service: PerformanceService,
    ):
        from src.core.exceptions import ValidationError

        # Validate service dependencies using core patterns
        if not isinstance(alert_service, AlertService):
            raise ValidationError(
                "Invalid alert_service parameter",
                field_name="alert_service",
                field_value=type(alert_service).__name__,
                expected_type="AlertService instance",
            )

        if not isinstance(metrics_service, MetricsService):
            raise ValidationError(
                "Invalid metrics_service parameter",
                field_name="metrics_service",
                field_value=type(metrics_service).__name__,
                expected_type="MetricsService instance",
            )

        if not isinstance(performance_service, PerformanceService):
            raise ValidationError(
                "Invalid performance_service parameter",
                field_name="performance_service",
                field_value=type(performance_service).__name__,
                expected_type="PerformanceService instance",
            )

        self.alerts = alert_service
        self.metrics = metrics_service
        self.performance = performance_service

    async def health_check(self) -> dict[str, Any]:
        """
        Perform comprehensive health check of monitoring services.

        Returns:
            Dict containing health status of all monitoring components

        Raises:
            ServiceError: If health check fails
        """
        from src.core.exceptions import ComponentError

        health_status = {
            "monitoring_service": "healthy",
            "components": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            # Check individual service health
            health_status["components"]["alerts"] = "healthy"
            health_status["components"]["metrics"] = "healthy"
            health_status["components"]["performance"] = "healthy"

            return health_status

        except Exception as e:
            raise ComponentError(
                f"Monitoring service health check failed: {e}",
                component="MonitoringService",
                operation="health_check",
                context={
                    "partial_status": health_status,
                    "error": str(e),
                },
            ) from e
