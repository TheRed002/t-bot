"""
Analytics Service Factory - Simplified implementation.

Provides simple factory functions for creating analytics services.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.analytics.service import AnalyticsService

from src.analytics.interfaces import AnalyticsServiceFactoryProtocol
from src.analytics.types import AnalyticsConfiguration


class AnalyticsServiceFactory(AnalyticsServiceFactoryProtocol):
    """Simple factory for creating analytics services."""

    def __init__(self, injector=None):
        """Initialize the factory with dependency injector."""
        self._injector = injector

    def create_analytics_service(
        self, config: AnalyticsConfiguration | None = None, **kwargs
    ) -> "AnalyticsService":
        """Create analytics service with dependencies."""
        from src.analytics.service import AnalyticsService

        # Get dependencies from injector if available
        dependencies = {}
        if self._injector:
            # Try to get each service from injector using proper service names
            service_mappings = {
                "realtime_analytics": "RealtimeAnalyticsService",
                "portfolio_service": "PortfolioService",
                "reporting_service": "ReportingService",
                "risk_service": "RiskService",
                "operational_service": "OperationalService",
                "alert_service": "AlertService",
                "export_service": "ExportService",
                "dashboard_service": "DashboardService",
                "metrics_collector": "MetricsCollector",
            }
            for param_name, service_name in service_mappings.items():
                try:
                    dependencies[param_name] = self._injector.resolve(service_name)
                except Exception:
                    # Service not available - that's okay
                    dependencies[param_name] = None

        # Override with any provided kwargs
        dependencies.update(kwargs)

        return AnalyticsService(config=config, **dependencies)

    def create_realtime_analytics_service(self, config: AnalyticsConfiguration | None = None):
        """Create realtime analytics service."""
        from src.analytics.services.realtime_analytics_service import RealtimeAnalyticsService

        return RealtimeAnalyticsService(config=config)

    def create_portfolio_service(self, config: AnalyticsConfiguration | None = None):
        """Create portfolio analytics service."""
        from src.analytics.services.portfolio_analytics_service import PortfolioAnalyticsService

        return PortfolioAnalyticsService(config=config)

    def create_risk_service(self, config: AnalyticsConfiguration | None = None):
        """Create risk analytics service."""
        from src.analytics.services.risk_service import RiskService

        return RiskService(config=config)

    def create_reporting_service(self, config: AnalyticsConfiguration | None = None):
        """Create reporting service."""
        from src.analytics.services.reporting_service import ReportingService

        return ReportingService(config=config)

    def create_operational_service(self, config: AnalyticsConfiguration | None = None):
        """Create operational analytics service."""
        from src.analytics.services.operational_service import OperationalService

        return OperationalService(config=config)

    def create_alert_service(self, config: AnalyticsConfiguration | None = None):
        """Create alert service."""
        from src.analytics.services.alert_service import AlertService

        return AlertService(config=config)

    def create_export_service(self, config: AnalyticsConfiguration | None = None):
        """Create export service."""
        from src.analytics.services.export_service import ExportService

        return ExportService(config=config)

    def create_dashboard_service(self, config: AnalyticsConfiguration | None = None):
        """Create dashboard service."""
        from src.analytics.services.dashboard_service import DashboardService

        return DashboardService(config=config)


def create_default_analytics_service(
    config: AnalyticsConfiguration | None = None, injector=None
) -> "AnalyticsService":
    """Create analytics service with default configuration."""
    factory = AnalyticsServiceFactory(injector)
    return factory.create_analytics_service(config)
