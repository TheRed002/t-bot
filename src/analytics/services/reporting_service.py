"""
Reporting Service - Simplified implementation.

Provides reporting functionality without complex engine orchestration.
"""

from datetime import datetime
from typing import Any

from src.analytics.base_analytics_service import BaseAnalyticsService
from src.analytics.common import AnalyticsErrorHandler, ServiceInitializationHelper
from src.analytics.interfaces import ReportingServiceProtocol
from src.analytics.types import AnalyticsConfiguration, AnalyticsReport, ReportType
from src.utils.datetime_utils import get_current_utc_timestamp


class ReportingService(BaseAnalyticsService, ReportingServiceProtocol):
    """Simple reporting service."""

    def __init__(
        self,
        config: AnalyticsConfiguration | None = None,
        metrics_collector=None,
    ):
        """Initialize the reporting service."""
        super().__init__(
            name="ReportingService",
            config=ServiceInitializationHelper.prepare_service_config(config),
            metrics_collector=metrics_collector,
        )
        self.config = config or AnalyticsConfiguration()

    async def generate_performance_report(
        self,
        report_type: ReportType,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> AnalyticsReport:
        """Generate performance report."""
        try:
            return {
                "report_type": report_type.value,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "generated_at": get_current_utc_timestamp().isoformat(),
                "data": {"summary": "Simple performance report", "metrics": {}},
            }
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            raise AnalyticsErrorHandler.create_operation_error(
                "ReportingService", "generate_performance_report", None, e
            ) from e

    # Required abstract method implementations
    async def calculate_metrics(self, *args, **kwargs) -> dict[str, Any]:
        """Calculate service-specific metrics."""
        return {"reports_generated": 0}

    async def validate_data(self, data: Any) -> bool:
        """Validate service-specific data."""
        return data is not None
