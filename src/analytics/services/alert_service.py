"""
Alert Service - Simplified implementation.

Provides alert management without complex orchestration.
"""

from typing import Any
from uuid import uuid4

from src.analytics.base_analytics_service import BaseAnalyticsService
from src.analytics.common import ServiceInitializationHelper
from src.analytics.interfaces import AlertServiceProtocol
from src.analytics.types import AnalyticsAlert, AnalyticsConfiguration
from src.core.types import AlertSeverity
from src.utils.datetime_utils import get_current_utc_timestamp


class AlertService(BaseAnalyticsService, AlertServiceProtocol):
    """Simple alert service."""

    def __init__(
        self,
        config: AnalyticsConfiguration | None = None,
        metrics_collector=None,
    ):
        """Initialize the alert service."""
        super().__init__(
            name="AlertService",
            config=ServiceInitializationHelper.prepare_service_config(config),
            metrics_collector=metrics_collector,
        )
        self.config = config or AnalyticsConfiguration()
        self._alerts: list[AnalyticsAlert] = []

    async def generate_alert(
        self,
        rule_name: str,
        severity: str,
        message: str,
        labels: dict[str, str],
        annotations: dict[str, str],
        **kwargs,
    ) -> AnalyticsAlert:
        """Generate a new alert with parameters aligned to monitoring AlertRequest."""
        alert = AnalyticsAlert(
            id=str(uuid4()),
            timestamp=get_current_utc_timestamp(),
            severity=AlertSeverity(severity),
            title=rule_name,
            message=message,
            metric_name=annotations.get("metric_name", rule_name),
            metadata={"labels": labels, "annotations": annotations, **kwargs},
        )
        self._alerts.append(alert)
        return alert

    def get_active_alerts(self) -> list[AnalyticsAlert]:
        """Get active alerts."""
        return [alert for alert in self._alerts if not alert.resolved]

    async def acknowledge_alert(self, fingerprint: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert using consistent fingerprint parameter."""
        for alert in self._alerts:
            if alert.id == fingerprint:
                alert.acknowledged = True
                alert.metadata["acknowledged_by"] = acknowledged_by
                return True
        return False

    # Required abstract method implementations
    async def calculate_metrics(self, *args, **kwargs) -> dict[str, Any]:
        """Calculate service-specific metrics."""
        return {"active_alerts": len(self._alerts)}

    async def validate_data(self, data: Any) -> bool:
        """Validate service-specific data."""
        return data is not None
