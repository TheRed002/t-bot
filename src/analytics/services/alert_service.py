"""
Alert Service.

This service provides a proper service layer implementation for alert management,
following service layer patterns and using dependency injection.
"""

from typing import Any

from src.analytics.alerts.alert_manager import AlertManager
from src.analytics.interfaces import AlertServiceProtocol
from src.analytics.types import AnalyticsAlert, AnalyticsConfiguration
from src.core.base.service import BaseService
from src.core.exceptions import ComponentError, ValidationError


class AlertService(BaseService, AlertServiceProtocol):
    """
    Service layer implementation for alert management.

    This service acts as a facade over the AlertManager,
    providing proper service layer abstraction and dependency injection.
    """

    def __init__(
        self,
        config: AnalyticsConfiguration,
        alert_manager: AlertManager | None = None,
    ):
        """
        Initialize the alert service.

        Args:
            config: Analytics configuration
            alert_manager: Injected alert manager engine (optional)
        """
        super().__init__()
        self.config = config

        # Use dependency injection - alert_manager must be injected
        if alert_manager is None:
            raise ComponentError(
                "alert_manager must be injected via dependency injection",
                component="AlertService",
                operation="__init__",
                context={"missing_dependency": "alert_manager"},
            )

        self._manager = alert_manager

        self.logger.info("AlertService initialized")

    async def start(self) -> None:
        """Start the alert service."""
        try:
            await self._manager.start()
            self.logger.info("Alert service started")
        except Exception as e:
            raise ComponentError(
                f"Failed to start alert service: {e}",
                component="AlertService",
                operation="start",
            ) from e

    async def stop(self) -> None:
        """Stop the alert service."""
        try:
            await self._manager.stop()
            self.logger.info("Alert service stopped")
        except Exception as e:
            self.logger.error(f"Error stopping alert service: {e}")

    def get_active_alerts(self) -> list[AnalyticsAlert]:
        """
        Get all active alerts.

        Returns:
            List of active alerts

        Raises:
            ComponentError: If retrieval fails
        """
        try:
            return self._manager.get_active_alerts()
        except Exception as e:
            raise ComponentError(
                f"Failed to get active alerts: {e}",
                component="AlertService",
                operation="get_active_alerts",
            ) from e

    def add_alert_rule(self, rule: Any) -> None:
        """
        Add custom alert rule.

        Args:
            rule: Alert rule to add

        Raises:
            ValidationError: If rule is invalid
            ComponentError: If addition fails
        """
        if not rule:
            raise ValidationError(
                "Invalid rule parameter",
                field_name="rule",
                field_value=rule,
                expected_type="AlertRule",
            )

        try:
            self._manager.add_alert_rule(rule)
            self.logger.debug(f"Alert rule added: {getattr(rule, 'rule_id', 'unknown')}")
        except Exception as e:
            raise ComponentError(
                f"Failed to add alert rule: {e}",
                component="AlertService",
                operation="add_alert_rule",
                context={"rule_id": getattr(rule, "rule_id", "unknown")},
            ) from e

    def remove_alert_rule(self, rule_id: str) -> None:
        """
        Remove alert rule.

        Args:
            rule_id: ID of rule to remove

        Raises:
            ValidationError: If rule_id is invalid
            ComponentError: If removal fails
        """
        if not isinstance(rule_id, str) or not rule_id:
            raise ValidationError(
                "Invalid rule_id parameter",
                field_name="rule_id",
                field_value=rule_id,
                expected_type="non-empty str",
            )

        try:
            self._manager.remove_alert_rule(rule_id)
            self.logger.debug(f"Alert rule removed: {rule_id}")
        except Exception as e:
            raise ComponentError(
                f"Failed to remove alert rule: {e}",
                component="AlertService",
                operation="remove_alert_rule",
                context={"rule_id": rule_id},
            ) from e

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: ID of alert to acknowledge
            acknowledged_by: Who is acknowledging the alert

        Returns:
            True if successfully acknowledged

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If acknowledgment fails
        """
        if not isinstance(alert_id, str) or not alert_id:
            raise ValidationError(
                "Invalid alert_id parameter",
                field_name="alert_id",
                field_value=alert_id,
                expected_type="non-empty str",
            )

        if not isinstance(acknowledged_by, str) or not acknowledged_by:
            raise ValidationError(
                "Invalid acknowledged_by parameter",
                field_name="acknowledged_by",
                field_value=acknowledged_by,
                expected_type="non-empty str",
            )

        try:
            result = await self._manager.acknowledge_alert(alert_id, acknowledged_by)
            self.logger.debug(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return result
        except Exception as e:
            raise ComponentError(
                f"Failed to acknowledge alert: {e}",
                component="AlertService",
                operation="acknowledge_alert",
                context={"alert_id": alert_id, "acknowledged_by": acknowledged_by},
            ) from e

    async def resolve_alert(
        self, alert_id: str, resolved_by: str, resolution_note: str | None = None
    ) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: ID of alert to resolve
            resolved_by: Who is resolving the alert
            resolution_note: Optional resolution note

        Returns:
            True if successfully resolved

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If resolution fails
        """
        if not isinstance(alert_id, str) or not alert_id:
            raise ValidationError(
                "Invalid alert_id parameter",
                field_name="alert_id",
                field_value=alert_id,
                expected_type="non-empty str",
            )

        if not isinstance(resolved_by, str) or not resolved_by:
            raise ValidationError(
                "Invalid resolved_by parameter",
                field_name="resolved_by",
                field_value=resolved_by,
                expected_type="non-empty str",
            )

        try:
            result = await self._manager.resolve_alert(alert_id, resolved_by, resolution_note)
            self.logger.debug(f"Alert resolved: {alert_id} by {resolved_by}")
            return result
        except Exception as e:
            raise ComponentError(
                f"Failed to resolve alert: {e}",
                component="AlertService",
                operation="resolve_alert",
                context={"alert_id": alert_id, "resolved_by": resolved_by},
            ) from e

    def get_alert_statistics(self, period_hours: int = 24) -> dict[str, Any]:
        """
        Get alert statistics.

        Args:
            period_hours: Period in hours for statistics

        Returns:
            Alert statistics

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If retrieval fails
        """
        if period_hours <= 0:
            raise ValidationError(
                "Invalid period_hours parameter",
                field_name="period_hours",
                field_value=period_hours,
                validation_rule="must be positive",
            )

        try:
            return self._manager.get_alert_statistics(period_hours)
        except Exception as e:
            raise ComponentError(
                f"Failed to get alert statistics: {e}",
                component="AlertService",
                operation="get_alert_statistics",
                context={"period_hours": period_hours},
            ) from e

    def store_risk_alert(self, alert) -> None:
        """
        Store risk alert for analytics.

        Args:
            alert: Risk alert to store

        Raises:
            ValidationError: If alert data is invalid
            ComponentError: If storage fails
        """
        if not alert:
            raise ValidationError(
                "Invalid alert parameter",
                field_name="alert",
                field_value=alert,
                expected_type="RiskAlert",
            )

        try:
            self._manager.store_risk_alert(alert)
            self.logger.debug("Risk alert stored for analytics")
        except Exception as e:
            raise ComponentError(
                f"Failed to store risk alert: {e}",
                component="AlertService",
                operation="store_risk_alert",
            ) from e
