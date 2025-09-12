"""
Operational Service - Simplified implementation.

Provides operational analytics without complex orchestration.
"""

from typing import Any

from src.analytics.base_analytics_service import BaseAnalyticsService
from src.analytics.common import ServiceInitializationHelper
from src.analytics.interfaces import OperationalServiceProtocol
from src.analytics.types import AnalyticsConfiguration, OperationalMetrics
from src.utils.datetime_utils import get_current_utc_timestamp


class OperationalService(BaseAnalyticsService, OperationalServiceProtocol):
    """Simple operational analytics service."""

    def __init__(
        self,
        config: AnalyticsConfiguration | None = None,
        metrics_collector=None,
    ):
        """Initialize the operational service."""
        super().__init__(
            name="OperationalService",
            config=ServiceInitializationHelper.prepare_service_config(config),
            metrics_collector=metrics_collector,
        )
        self.config = config or AnalyticsConfiguration()

    async def get_operational_metrics(self) -> OperationalMetrics:
        """Get operational metrics."""
        return OperationalMetrics(timestamp=get_current_utc_timestamp())

    def record_strategy_event(
        self, strategy_name: str, event_type: str, success: bool = True, **kwargs
    ) -> None:
        """Record strategy event."""
        self.logger.info(
            f"Strategy event recorded: {strategy_name} - {event_type} "
            f"(success: {success})",
            extra={"strategy": strategy_name, "event_type": event_type, "success": success, **kwargs}
        )

    def record_system_error(
        self, component: str, error_type: str, error_message: str, **kwargs
    ) -> None:
        """Record system error."""
        self.logger.error(
            f"System error recorded: {component} - {error_type}: {error_message}",
            extra={"component": component, "error_type": error_type, "error_message": error_message, **kwargs}
        )

    # Required abstract method implementations
    async def calculate_metrics(self, *args, **kwargs) -> dict[str, Any]:
        """Calculate service-specific metrics."""
        return {"operational": await self.get_operational_metrics()}

    async def validate_data(self, data: Any) -> bool:
        """Validate service-specific data."""
        return data is not None
