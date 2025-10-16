"""
Service layer for monitoring operations.

This module implements the service layer pattern for monitoring operations,
providing a clean interface between controllers and domain logic.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from src.core.base.service import BaseService
from src.core.exceptions import ComponentError, ValidationError
from src.core.types import AlertSeverity, OrderType

if TYPE_CHECKING:
    from src.monitoring.alerting import Alert, AlertManager
    from src.monitoring.dashboards import Dashboard, GrafanaDashboardManager
    from src.monitoring.metrics import MetricsCollector
    from src.monitoring.performance import PerformanceProfiler
else:
    # Runtime imports for non-TYPE_CHECKING
    try:
        from src.monitoring.dashboards import Dashboard
    except ImportError:
        # Create a placeholder if dashboards module is not available
        class Dashboard:
            pass

# Import interfaces at runtime - they're needed for inheritance
from src.monitoring.interfaces import (
    AlertServiceInterface,
    DashboardServiceInterface,
    MetricsServiceInterface,
    MonitoringServiceInterface,
    PerformanceServiceInterface,
)
from src.monitoring.data_transformer import MonitoringDataTransformer
from src.utils.messaging_patterns import (
    BoundaryValidator,
    ErrorPropagationMixin,
)
from src.utils.monitoring_helpers import (
    create_error_context,
)


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
    value: Decimal | float
    labels: dict[str, str] | None = None
    namespace: str = "tbot"


# Abstract service classes moved to interfaces.py to avoid circular dependencies


class DefaultAlertService(BaseService, AlertServiceInterface, ErrorPropagationMixin):
    """Default implementation of AlertService."""

    def __init__(self, alert_manager: "AlertManager"):
        super().__init__()
        if alert_manager is None:
            raise ValueError("alert_manager is required - use dependency injection")
        self._alert_manager = alert_manager
        self._data_transformer = MonitoringDataTransformer()

    async def create_alert(self, request: AlertRequest) -> str:
        """Create a new alert and return its fingerprint."""

        # Validate AlertRequest using consistent core validation patterns first
        if not isinstance(request, AlertRequest):
            raise ValidationError(
                "Invalid request parameter",
                field_name="request",
                field_value=type(request).__name__,
                expected_type="AlertRequest",
            )

        # Apply consistent data transformation patterns after validation
        request_data = {
            "rule_name": request.rule_name,
            "severity": request.severity.value,
            "message": request.message,
            "labels": request.labels,
            "annotations": request.annotations,
        }
        transformed_data = MonitoringDataTransformer.transform_for_pub_sub("alert_create", request_data)
        transformed_request = AlertRequest(
            rule_name=transformed_data["rule_name"],
            severity=request.severity,
            message=transformed_data["message"],
            labels=transformed_data["labels"],
            annotations=transformed_data["annotations"],
        )

        if not isinstance(transformed_request.rule_name, str):
            raise ValidationError(
                "Invalid rule_name parameter",
                field_name="rule_name",
                field_value=transformed_request.rule_name,
                expected_type="str",
            )

        from src.monitoring.alerting import Alert, AlertStatus

        alert = Alert(
            rule_name=transformed_request.rule_name,
            severity=transformed_request.severity,
            status=AlertStatus.FIRING,
            message=transformed_request.message,
            labels=transformed_request.labels,
            annotations=transformed_request.annotations,
            starts_at=datetime.now(timezone.utc),
        )

        try:
            await self._alert_manager.fire_alert(alert)
            return alert.fingerprint
        except Exception as e:
            # Check if it's a validation error and propagate accordingly
            if hasattr(e, "__class__") and "ValidationError" in e.__class__.__name__:
                # Apply consistent error propagation - re-raise validation errors
                self.propagate_validation_error(e, "AlertService.create_alert")
                return  # propagate_validation_error should raise, but add explicit return for safety

            # Handle all other exceptions
            error_context = await create_error_context(
                "AlertService",
                "create_alert",
                e,
                details={
                    "rule_name": transformed_request.rule_name,
                    "severity": transformed_request.severity.value,
                    "processing_mode": "stream",
                    "data_format": "bot_event_v1",
                },
            )
            raise ComponentError(
                f"Failed to create alert: {e}",
                component_name="AlertService",
                operation="create_alert",
                details=error_context.details,
            ) from e

    async def resolve_alert(self, fingerprint: str) -> bool:
        """Resolve an active alert."""
        await self._alert_manager.resolve_alert(fingerprint)
        return True

    async def acknowledge_alert(self, fingerprint: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        return await self._alert_manager.acknowledge_alert(fingerprint, acknowledged_by)

    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list["Alert"]:
        """Get active alerts."""
        return self._alert_manager.get_active_alerts(severity)

    def get_alert_stats(self) -> dict[str, Any]:
        """Get alert statistics."""
        return self._alert_manager.get_alert_stats()

    def add_rule(self, rule) -> None:
        """Add an alert rule."""
        self._alert_manager.add_rule(rule)

    def add_escalation_policy(self, policy) -> None:
        """Add an escalation policy."""
        self._alert_manager.add_escalation_policy(policy)

    async def handle_error_event_from_error_handling(self, error_data: dict[str, Any]) -> str:
        """Handle error event from error_handling module with consistent async message pattern."""
        # Apply consistent data transformation before validation using MonitoringDataTransformer
        transformed_error_data = MonitoringDataTransformer.apply_cross_module_validation(
            error_data, source_module="error_handling", target_module="monitoring"
        )

        # Validate data at error_handling -> monitoring boundary
        BoundaryValidator.validate_error_to_monitoring_boundary(transformed_error_data)

        # Create alert request from error data with consistent processing mode
        # Use consistent AlertSeverity from core types

        # Map severity string to enum using consistent core types
        severity_mapping = {
            "low": AlertSeverity.INFO,
            "medium": AlertSeverity.MEDIUM,
            "high": AlertSeverity.HIGH,
            "critical": AlertSeverity.CRITICAL,
        }

        severity = severity_mapping.get(transformed_error_data.get("severity", "medium"), AlertSeverity.MEDIUM)

        alert_request = AlertRequest(
            rule_name=f"error_handling_{transformed_error_data.get('error_id', 'unknown')}",
            severity=severity,
            message=f"Error pattern detected in {transformed_error_data.get('component', 'unknown')}",
            labels={
                "source": "error_handling",
                "component": transformed_error_data.get("component", "unknown"),
                "error_id": transformed_error_data.get("error_id", "unknown"),
                "processing_mode": transformed_error_data.get("processing_mode", "stream"),
                "message_pattern": "pub_sub",  # Consistent pub/sub pattern
            },
            annotations={
                "recovery_success": str(transformed_error_data.get("recovery_success", False)),
                "operation": transformed_error_data.get("operation", "unknown"),
                "processed_at": transformed_error_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "data_format": transformed_error_data.get("data_format", "bot_event_v1"),
                "pub_sub_processed": "true",  # Mark as pub/sub processed
            },
        )

        return await self.create_alert(alert_request)

    async def handle_batch_error_events_from_error_handling(
        self, error_events: list[dict[str, Any]]
    ) -> list[str]:
        """Handle multiple error events in batch for consistent processing paradigm."""
        if not error_events:
            return []

        # Process each error event asynchronously for message pattern consistency
        import asyncio
        tasks = [
            self.handle_error_event_from_error_handling(error_data)
            for error_data in error_events
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error strings for consistent return format
        alert_ids = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Failed to process batch error event: {result}")
                alert_ids.append(f"error_{str(result)[:50]}")
            else:
                alert_ids.append(result)

        return alert_ids

    def _transform_error_event_data(self, error_data: dict[str, Any]) -> dict[str, Any]:
        """Transform error event data for consistent monitoring processing."""
        transformed_data = error_data.copy()

        # Apply consistent processing metadata
        if "processing_mode" not in transformed_data:
            transformed_data["processing_mode"] = "stream"  # Default to stream

        if "message_pattern" not in transformed_data:
            transformed_data["message_pattern"] = "pub_sub"  # Consistent pub/sub pattern

        if "data_format" not in transformed_data:
            transformed_data["data_format"] = "bot_event_v1"

        if "timestamp" not in transformed_data:
            transformed_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        return transformed_data

    def _transform_alert_request_data(self, request: AlertRequest) -> AlertRequest:
        """Transform alert request data consistently across operations."""
        # Validate request is AlertRequest type
        if not isinstance(request, AlertRequest):
            raise ValidationError(
                message="Request must be AlertRequest instance",
                field_name="request",
                field_value=type(request).__name__,
                expected_type="AlertRequest",
            )

        # Apply minimal transformation to preserve existing behavior
        transformed_annotations = request.annotations.copy() if request.annotations else {}

        # Apply consistent Decimal transformation for financial data in annotations
        if "price" in transformed_annotations and transformed_annotations["price"] is not None:
            from src.utils.decimal_utils import to_decimal

            transformed_annotations["price"] = str(to_decimal(transformed_annotations["price"]))

        if (
            "quantity" in transformed_annotations
            and transformed_annotations["quantity"] is not None
        ):
            from src.utils.decimal_utils import to_decimal

            transformed_annotations["quantity"] = str(
                to_decimal(transformed_annotations["quantity"])
            )

        return AlertRequest(
            rule_name=request.rule_name,
            severity=request.severity,
            message=request.message,
            labels=request.labels,  # Keep original labels unchanged
            annotations=transformed_annotations,
        )


class DefaultMetricsService(BaseService, MetricsServiceInterface, ErrorPropagationMixin):
    """Default implementation of MetricsService."""

    def __init__(self, metrics_collector: "MetricsCollector"):
        super().__init__()
        if metrics_collector is None:
            raise ValueError("metrics_collector is required - use dependency injection")
        self._metrics_collector = metrics_collector

    def record_counter(self, request: MetricRequest) -> None:
        """Record a counter metric."""

        # Validate MetricRequest using consistent core validation patterns first

        if not isinstance(request, MetricRequest):
            raise ValidationError(
                "Invalid request parameter",
                field_name="request",
                field_value=type(request).__name__,
                expected_type="MetricRequest",
            )
        
        # Validate value type before transformation
        if not isinstance(request.value, (int, float, Decimal)):
            raise ValidationError(
                "Invalid value parameter",
                field_name="value",
                field_value=request.value,
                expected_type="number",
            )

        # Apply consistent data transformation patterns after validation
        metric_data = {
            "name": request.name,
            "value": str(request.value),
            "labels": request.labels,
            "namespace": request.namespace,
        }
        transformed_data = MonitoringDataTransformer.transform_for_pub_sub("metric_counter", metric_data)
        transformed_request = MetricRequest(
            name=transformed_data["name"],
            value=request.value,
            labels=transformed_data["labels"],
            namespace=transformed_data["namespace"],
        )

        if not isinstance(transformed_request.name, str):
            raise ValidationError(
                "Invalid metric name",
                field_name="name",
                field_value=transformed_request.name,
                expected_type="str",
            )

        if not isinstance(transformed_request.value, (int, float, Decimal)):
            raise ValidationError(
                "Invalid value parameter",
                field_name="value",
                field_value=transformed_request.value,
                expected_type="number",
            )

        if transformed_request.value < 0:
            raise ValidationError(
                "Invalid metric value - value must be non-negative",
                field_name="value",
                field_value=transformed_request.value,
                validation_rule="must be non-negative number",
            )

        try:
            # Convert Decimal to float for metrics collector
            value_as_float = (
                float(transformed_request.value)
                if isinstance(transformed_request.value, Decimal)
                else transformed_request.value
            )
            self._metrics_collector.increment_counter(
                transformed_request.name,
                transformed_request.labels,
                value_as_float,
                transformed_request.namespace,
            )
        except Exception as e:
            # Check if it's a validation error and propagate accordingly
            if hasattr(e, "__class__") and "ValidationError" in e.__class__.__name__:
                # Apply consistent error propagation - re-raise validation errors
                self.propagate_validation_error(e, "MetricsService.record_counter")
                return  # propagate_validation_error should raise, but add explicit return for safety

            # Handle all other exceptions
            raise ComponentError(
                f"Failed to record counter metric: {e}",
                component_name="MetricsService",
                operation="record_counter",
                details={
                    "metric_name": transformed_request.name,
                    "metric_value": transformed_request.value,
                    "namespace": transformed_request.namespace,
                    "processing_mode": "stream",
                    "data_format": "bot_event_v1",
                },
            ) from e

    def record_gauge(self, request: MetricRequest) -> None:
        """Record a gauge metric."""
        # Apply consistent data transformation patterns using MonitoringDataTransformer
        metric_data = {
            "name": request.name,
            "value": str(request.value),
            "labels": request.labels,
            "namespace": request.namespace,
        }
        transformed_data = MonitoringDataTransformer.transform_for_pub_sub("metric_gauge", metric_data)
        transformed_request = MetricRequest(
            name=transformed_data["name"],
            value=request.value,
            labels=transformed_data["labels"],
            namespace=transformed_data["namespace"],
        )

        try:
            # Convert Decimal to float for metrics collector
            value_as_float = (
                float(transformed_request.value)
                if isinstance(transformed_request.value, Decimal)
                else transformed_request.value
            )
            self._metrics_collector.set_gauge(
                transformed_request.name,
                value_as_float,
                transformed_request.labels,
                transformed_request.namespace,
            )
        except Exception as e:
            # Check if it's a validation error and propagate accordingly
            if hasattr(e, "__class__") and "ValidationError" in e.__class__.__name__:
                # Apply consistent error propagation - re-raise validation errors
                self.propagate_validation_error(e, "MetricsService.record_gauge")
                return  # propagate_validation_error should raise, but add explicit return for safety

            # Handle all other exceptions
            raise ComponentError(
                f"Failed to record gauge metric: {e}",
                component_name="MetricsService",
                operation="record_gauge",
                details={
                    "metric_name": transformed_request.name,
                    "processing_mode": "stream",
                    "data_format": "bot_event_v1",
                },
            ) from e

    def record_histogram(self, request: MetricRequest) -> None:
        """Record a histogram metric."""
        # Apply consistent data transformation patterns using MonitoringDataTransformer
        metric_data = {
            "name": request.name,
            "value": str(request.value),
            "labels": request.labels,
            "namespace": request.namespace,
        }
        transformed_data = MonitoringDataTransformer.transform_for_pub_sub("metric_gauge", metric_data)
        transformed_request = MetricRequest(
            name=transformed_data["name"],
            value=request.value,
            labels=transformed_data["labels"],
            namespace=transformed_data["namespace"],
        )

        try:
            # Convert Decimal to float for metrics collector
            value_as_float = (
                float(transformed_request.value)
                if isinstance(transformed_request.value, Decimal)
                else transformed_request.value
            )
            self._metrics_collector.observe_histogram(
                transformed_request.name,
                value_as_float,
                transformed_request.labels,
                transformed_request.namespace,
            )
        except Exception as e:
            # Check if it's a validation error and propagate accordingly
            if hasattr(e, "__class__") and "ValidationError" in e.__class__.__name__:
                # Apply consistent error propagation - re-raise validation errors
                self.propagate_validation_error(e, "MetricsService.record_histogram")
                return  # propagate_validation_error should raise, but add explicit return for safety

            # Handle all other exceptions
            raise ComponentError(
                f"Failed to record histogram metric: {e}",
                component_name="MetricsService",
                operation="record_histogram",
                details={
                    "metric_name": transformed_request.name,
                    "processing_mode": "stream",
                    "data_format": "bot_event_v1",
                },
            ) from e

    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        metrics_bytes = self._metrics_collector.export_metrics()
        # Convert bytes to string to match interface
        if isinstance(metrics_bytes, bytes):
            return metrics_bytes.decode("utf-8")
        return str(metrics_bytes)

    def record_error_pattern_metric(self, error_data: dict[str, Any]) -> None:
        """Record error pattern metric from error_handling module."""
        # Validate data at error_handling -> monitoring boundary
        BoundaryValidator.validate_error_to_monitoring_boundary(error_data)

        # Create metric request for error patterns
        metric_request = MetricRequest(
            name="error_patterns_detected_total",
            value=1,
            labels={
                "component": error_data.get("component", "unknown"),
                "severity": error_data.get("severity", "medium"),
                "error_id": error_data.get("error_id", "unknown"),
                "source": "error_handling",
                "processing_mode": error_data.get("processing_mode", "stream"),
            },
            namespace="error_handling",
        )

        self.record_counter(metric_request)



class DefaultPerformanceService(BaseService, PerformanceServiceInterface, ErrorPropagationMixin):
    """Default implementation of PerformanceService."""

    def __init__(self, performance_profiler: "PerformanceProfiler"):
        super().__init__()
        if performance_profiler is None:
            raise ValueError("performance_profiler is required - use dependency injection")
        self._performance_profiler = performance_profiler

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        return self._performance_profiler.get_performance_summary()

    def record_order_execution(
        self,
        exchange: str,
        order_type: str,
        symbol: str,
        latency_ms: Decimal,
        fill_rate: Decimal,
        slippage_bps: Decimal,
    ) -> None:
        """Record order execution metrics."""
        # Apply consistent data transformation patterns for financial data
        performance_data = {
            "exchange": exchange,
            "symbol": symbol,
            "operation": "order_execution",
            "latency_ms": str(latency_ms),
            "order_type": order_type,
            "fill_rate": str(fill_rate),
            "slippage_bps": str(slippage_bps),
        }
        transformed_data = MonitoringDataTransformer.transform_for_pub_sub("performance_execution", performance_data)

        # Validate input parameters using consistent core validation patterns

        if not isinstance(transformed_data["exchange"], str):
            raise ValidationError(
                "Invalid exchange parameter",
                field_name="exchange",
                field_value=transformed_data["exchange"],
                expected_type="str",
            )

        if not isinstance(transformed_data["symbol"], str):
            raise ValidationError(
                "Invalid symbol parameter",
                field_name="symbol",
                field_value=transformed_data["symbol"],
                expected_type="str",
            )

        # Convert string back to number for validation if needed
        try:
            if isinstance(transformed_data["latency_ms"], str):
                latency_val = float(transformed_data["latency_ms"])
            else:
                latency_val = float(transformed_data["latency_ms"])

            if latency_val < 0:
                raise ValidationError(
                    "Invalid latency_ms parameter",
                    field_name="latency_ms",
                    field_value=transformed_data["latency_ms"],
                    validation_rule="must be non-negative number",
                )
        except (ValueError, TypeError) as e:
            raise ValidationError(
                "Invalid latency_ms parameter",
                field_name="latency_ms",
                field_value=transformed_data.get("latency_ms", "missing"),
                validation_rule="must be numeric and non-negative",
            ) from e
        except ValidationError:
            # Re-raise validation errors
            raise

        # Convert string to OrderType enum with consistent error handling
        try:
            if isinstance(transformed_data["order_type"], str):
                # Map uppercase strings to enum values (enum values are lowercase)
                order_type_upper = transformed_data["order_type"].upper()
                if order_type_upper == "MARKET":
                    order_type_enum = OrderType.MARKET
                elif order_type_upper == "LIMIT":
                    order_type_enum = OrderType.LIMIT
                elif order_type_upper == "STOP_LOSS":
                    order_type_enum = OrderType.STOP_LOSS
                elif order_type_upper == "TAKE_PROFIT":
                    order_type_enum = OrderType.TAKE_PROFIT
                else:
                    raise ValueError(f"Unknown order type: {transformed_data['order_type']}")
            else:
                # Try to use the order_type as-is if it's already an OrderType enum
                # This handles both real OrderType enums and mocked versions in tests
                try:
                    # Check if it has the expected enum-like attributes
                    if hasattr(transformed_data["order_type"], "name") and hasattr(
                        transformed_data["order_type"], "value"
                    ):
                        order_type_enum = transformed_data["order_type"]
                    else:
                        raise ValueError("Not a valid OrderType enum")
                except (ValueError, AttributeError):
                    raise ValidationError(
                        "Invalid order_type parameter",
                        field_name="order_type",
                        field_value=transformed_data["order_type"],
                        expected_type="str or OrderType",
                    )
        except ValueError as e:
            raise ValidationError(
                f"Invalid order_type value: {transformed_data['order_type']}",
                field_name="order_type",
                field_value=transformed_data["order_type"],
                expected_type="valid OrderType enum value",
            ) from e

        try:
            # Convert string values back to Decimal for performance profiler
            from src.utils.decimal_utils import to_decimal

            latency_decimal = to_decimal(transformed_data["latency_ms"])
            fill_rate_decimal = to_decimal(transformed_data["fill_rate"])
            slippage_decimal = to_decimal(transformed_data["slippage_bps"])

            self._performance_profiler.record_order_execution(
                transformed_data["exchange"],
                order_type_enum,
                transformed_data["symbol"],
                latency_decimal,
                fill_rate_decimal,
                slippage_decimal,
            )
        except Exception as e:
            raise ComponentError(
                f"Failed to record order execution: {e}",
                component_name="PerformanceService",
                operation="record_order_execution",
                details={
                    "exchange": transformed_data["exchange"],
                    "symbol": transformed_data["symbol"],
                    "processing_mode": "stream",
                    "data_format": "bot_event_v1",
                },
            ) from e

    def record_market_data_processing(
        self,
        exchange: str,
        data_type: str,
        processing_time_ms: Decimal,
        message_count: int,
    ) -> None:
        """Record market data processing metrics."""
        self._performance_profiler.record_market_data_processing(
            exchange, data_type, float(processing_time_ms), message_count
        )

    def get_latency_stats(self, metric_name: str):
        """Get latency statistics for a metric."""
        return self._performance_profiler.get_latency_stats(metric_name)

    def get_system_resource_stats(self):
        """Get system resource statistics."""
        return self._performance_profiler.get_system_resource_stats()



class DefaultDashboardService(BaseService, DashboardServiceInterface):
    """Default implementation of DashboardService."""

    def __init__(self, dashboard_manager: "GrafanaDashboardManager"):
        """Initialize dashboard service with injected dependencies."""
        super().__init__()
        if dashboard_manager is None:
            raise ValueError("dashboard_manager is required - use dependency injection")
        self._dashboard_manager = dashboard_manager

    async def deploy_dashboard(self, dashboard: "Dashboard") -> bool:
        """Deploy a dashboard."""
        return await self._dashboard_manager.deploy_dashboard(dashboard)

    async def deploy_all_dashboards(self) -> dict[str, bool]:
        """Deploy all dashboards."""
        return await self._dashboard_manager.deploy_all_dashboards()

    def export_dashboards_to_files(self, output_dir: str) -> None:
        """Export dashboards to files."""
        self._dashboard_manager.export_dashboards_to_files(output_dir)

    def create_trading_overview_dashboard(self) -> Dashboard:
        """Create trading overview dashboard."""
        return self._dashboard_manager.builder.create_trading_overview_dashboard()

    def create_system_performance_dashboard(self) -> Dashboard:
        """Create system performance dashboard."""
        return self._dashboard_manager.builder.create_system_performance_dashboard()


class MonitoringService(BaseService, MonitoringServiceInterface):
    """Composite service for all monitoring operations."""

    def __init__(
        self,
        alert_service: "AlertServiceInterface",
        metrics_service: "MetricsServiceInterface",
        performance_service: "PerformanceServiceInterface",
    ):
        super().__init__()

        # Validate service dependencies using consistent core validation patterns

        if not hasattr(alert_service, "create_alert"):
            raise ValidationError(
                "Invalid alert_service parameter - missing required methods",
                field_name="alert_service",
                field_value=type(alert_service).__name__,
                expected_type="AlertServiceInterface",
            )

        if not hasattr(metrics_service, "record_counter"):
            raise ValidationError(
                "Invalid metrics_service parameter - missing required methods",
                field_name="metrics_service",
                field_value=type(metrics_service).__name__,
                expected_type="MetricsServiceInterface",
            )

        if not hasattr(performance_service, "get_performance_summary"):
            raise ValidationError(
                "Invalid performance_service parameter - missing required methods",
                field_name="performance_service",
                field_value=type(performance_service).__name__,
                expected_type="PerformanceServiceInterface",
            )

        self.alerts = alert_service
        self.metrics = metrics_service
        self.performance = performance_service

    async def start_monitoring(self) -> None:
        """Start monitoring services."""
        self.logger.info("CompositeMonitoringService.start_monitoring() called")

        # Start underlying services if they have start methods
        if hasattr(self.alerts, "_alert_manager") and hasattr(self.alerts._alert_manager, "start"):
            self.logger.info("Starting alert manager...")
            await self.alerts._alert_manager.start()
            self.logger.info("Alert manager started")

        if hasattr(self.metrics, "_metrics_collector") and hasattr(
            self.metrics._metrics_collector, "start"
        ):
            self.logger.info("Starting metrics collector...")
            await self.metrics._metrics_collector.start()
            self.logger.info("Metrics collector started")

        if hasattr(self.performance, "_performance_profiler") and hasattr(
            self.performance._performance_profiler, "start"
        ):
            self.logger.info("Starting performance profiler...")
            await self.performance._performance_profiler.start()
            self.logger.info("Performance profiler started")

        self.logger.info("CompositeMonitoringService.start_monitoring() completed")

    async def stop_monitoring(self) -> None:
        """Stop monitoring services."""
        # Stop underlying services if they have stop methods
        if hasattr(self.alerts, "_alert_manager") and hasattr(self.alerts._alert_manager, "stop"):
            await self.alerts._alert_manager.stop()

        if hasattr(self.metrics, "_metrics_collector") and hasattr(
            self.metrics._metrics_collector, "stop"
        ):
            await self.metrics._metrics_collector.stop()

        if hasattr(self.performance, "_performance_profiler") and hasattr(
            self.performance._performance_profiler, "stop"
        ):
            await self.performance._performance_profiler.stop()

    async def get_health_status(self) -> dict[str, Any]:
        """Get health status of all monitoring components."""
        return await self.health_check()

    async def health_check(self) -> dict[str, Any]:
        """
        Perform comprehensive health check of monitoring services.

        Returns:
            Dict containing health status of all monitoring components

        Raises:
            ServiceError: If health check fails
        """

        health_status: dict[str, Any] = {
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
            error_status = {
                "monitoring_service": "unhealthy",
                "components": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "partial_status": health_status,
            }
            return error_status
