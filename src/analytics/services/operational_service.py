"""
Operational Service.

This service provides a proper service layer implementation for operational analytics,
following service layer patterns and using dependency injection.
"""

from typing import Any

from src.analytics.interfaces import OperationalServiceProtocol
from src.analytics.operational.operational_analytics import OperationalAnalyticsEngine
from src.analytics.types import AnalyticsConfiguration, OperationalMetrics
from src.core.base.service import BaseService
from src.core.exceptions import ComponentError, ValidationError
from src.core.types import Order


class OperationalService(BaseService, OperationalServiceProtocol):
    """
    Service layer implementation for operational analytics.

    This service acts as a facade over the OperationalAnalyticsEngine,
    providing proper service layer abstraction and dependency injection.
    """

    def __init__(
        self,
        config: AnalyticsConfiguration,
        operational_engine: OperationalAnalyticsEngine | None = None,
    ):
        """
        Initialize the operational service.

        Args:
            config: Analytics configuration
            operational_engine: Injected operational analytics engine (optional)
        """
        super().__init__()
        self.config = config

        # Use dependency injection - operational_engine must be injected
        if operational_engine is None:
            raise ComponentError(
                "operational_engine must be injected via dependency injection",
                component="OperationalService",
                operation="__init__",
                context={"missing_dependency": "operational_engine"},
            )

        self._engine = operational_engine

        self.logger.info("OperationalService initialized")

    async def start(self) -> None:
        """Start the operational service."""
        try:
            await self._engine.start()
            self.logger.info("Operational service started")
        except Exception as e:
            raise ComponentError(
                f"Failed to start operational service: {e}",
                component="OperationalService",
                operation="start",
            ) from e

    async def stop(self) -> None:
        """Stop the operational service."""
        try:
            await self._engine.stop()
            self.logger.info("Operational service stopped")
        except Exception as e:
            self.logger.error(f"Error stopping operational service: {e}")

    async def get_operational_metrics(self) -> OperationalMetrics:
        """
        Get operational metrics.

        Returns:
            Current operational metrics

        Raises:
            ComponentError: If retrieval fails
        """
        try:
            return await self._engine.get_operational_metrics()
        except Exception as e:
            raise ComponentError(
                f"Failed to get operational metrics: {e}",
                component="OperationalService",
                operation="get_operational_metrics",
            ) from e

    def record_order_update(self, order: Order) -> None:
        """
        Record order update for operational tracking.

        Args:
            order: Order to record

        Raises:
            ValidationError: If order is invalid
            ComponentError: If recording fails
        """
        if not isinstance(order, Order):
            raise ValidationError(
                "Invalid order parameter",
                field_name="order",
                field_value=type(order),
                expected_type="Order",
            )

        try:
            self._engine.record_order_update(order)
            self.logger.debug(f"Order update recorded: {order.order_id}")
        except Exception as e:
            raise ComponentError(
                f"Failed to record order update: {e}",
                component="OperationalService",
                operation="record_order_update",
                context={"order_id": order.order_id},
            ) from e

    def record_strategy_event(
        self,
        strategy_name: str,
        event_type: str,
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        """
        Record strategy event for operational tracking.

        Args:
            strategy_name: Strategy name
            event_type: Event type
            success: Whether event was successful
            error_message: Error message if not successful

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If recording fails
        """
        if not isinstance(strategy_name, str) or not strategy_name:
            raise ValidationError(
                "Invalid strategy_name parameter",
                field_name="strategy_name",
                field_value=strategy_name,
                expected_type="non-empty str",
            )

        if not isinstance(event_type, str) or not event_type:
            raise ValidationError(
                "Invalid event_type parameter",
                field_name="event_type",
                field_value=event_type,
                expected_type="non-empty str",
            )

        try:
            self._engine.record_strategy_event(strategy_name, event_type, success, error_message)
            self.logger.debug(f"Strategy event recorded: {strategy_name}/{event_type}")
        except Exception as e:
            raise ComponentError(
                f"Failed to record strategy event: {e}",
                component="OperationalService",
                operation="record_strategy_event",
                context={
                    "strategy_name": strategy_name,
                    "event_type": event_type,
                    "success": success,
                },
            ) from e

    def record_system_error(
        self, component: str, error_type: str, error_message: str, severity: str = "low"
    ) -> None:
        """
        Record system error for tracking.

        Args:
            component: Component where error occurred
            error_type: Type of error
            error_message: Error message
            severity: Error severity level

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If recording fails
        """
        if not isinstance(component, str) or not component:
            raise ValidationError(
                "Invalid component parameter",
                field_name="component",
                field_value=component,
                expected_type="non-empty str",
            )

        valid_severities = ["low", "medium", "high", "critical", "info"]
        if severity not in valid_severities:
            raise ValidationError(
                "Invalid severity parameter",
                field_name="severity",
                field_value=severity,
                validation_rule=f"must be one of {valid_severities}",
            )

        try:
            self._engine.record_system_error(component, error_type, error_message, severity)
            self.logger.debug(f"System error recorded: {component}/{error_type}")
        except Exception as e:
            raise ComponentError(
                f"Failed to record system error: {e}",
                component="OperationalService",
                operation="record_system_error",
                context={
                    "component": component,
                    "error_type": error_type,
                    "severity": severity,
                },
            ) from e

    async def record_api_call(
        self,
        service: str,
        endpoint: str,
        response_time_ms: float,
        status_code: int,
        success: bool = True,
    ) -> None:
        """
        Record API call metrics.

        Args:
            service: Service name
            endpoint: API endpoint
            response_time_ms: Response time in milliseconds
            status_code: HTTP status code
            success: Whether call was successful

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If recording fails
        """
        if not isinstance(service, str) or not service:
            raise ValidationError(
                "Invalid service parameter",
                field_name="service",
                field_value=service,
                expected_type="non-empty str",
            )

        if response_time_ms < 0:
            raise ValidationError(
                "Invalid response_time_ms parameter",
                field_name="response_time_ms",
                field_value=response_time_ms,
                validation_rule="must be non-negative",
            )

        try:
            await self._engine.record_api_call(
                service, endpoint, response_time_ms, status_code, success
            )
            self.logger.debug(f"API call recorded: {service}/{endpoint}")
        except Exception as e:
            raise ComponentError(
                f"Failed to record API call: {e}",
                component="OperationalService",
                operation="record_api_call",
                context={
                    "service": service,
                    "endpoint": endpoint,
                    "status_code": status_code,
                },
            ) from e

    async def generate_system_health_dashboard(self) -> dict[str, Any]:
        """
        Generate system health dashboard.

        Returns:
            System health dashboard data

        Raises:
            ComponentError: If dashboard generation fails
        """
        try:
            return await self._engine.generate_system_health_dashboard()
        except Exception as e:
            raise ComponentError(
                f"Failed to generate system health dashboard: {e}",
                component="OperationalService",
                operation="generate_system_health_dashboard",
            ) from e
