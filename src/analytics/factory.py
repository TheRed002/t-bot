"""
Analytics Service Factory.

This module provides factory functions for creating analytics services
with proper dependency injection and configuration.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.analytics.service import AnalyticsService

from src.analytics.interfaces import (
    AlertServiceProtocol,
    ExportServiceProtocol,
    OperationalServiceProtocol,
    PortfolioServiceProtocol,
    RealtimeAnalyticsServiceProtocol,
    ReportingServiceProtocol,
    RiskServiceProtocol,
)
from src.analytics.types import AnalyticsConfiguration


class AnalyticsServiceFactory:
    """Factory for creating analytics services with dependency injection."""

    def __init__(self, injector=None):
        """
        Initialize the factory with dependency injector.

        Args:
            injector: Dependency injection container
        """
        # Don't create default injector - require injection
        if injector is None:
            from src.core.exceptions import ComponentError

            raise ComponentError(
                "Injector must be provided to factory",
                component="AnalyticsServiceFactory",
                operation="__init__",
                context={"missing_dependency": "injector"},
            )
        self._injector = injector

    def create_analytics_service(
        self,
        config: AnalyticsConfiguration | None = None,
    ) -> "AnalyticsService":
        """
        Create a complete analytics service with all dependencies using DI.

        Args:
            config: Analytics configuration

        Returns:
            Configured analytics service
        """
        # Use dependency injection to create service with all dependencies
        return self._injector.resolve("AnalyticsService")

    def create_portfolio_service(self) -> PortfolioServiceProtocol:
        """Create portfolio analytics service using DI (returns service layer)."""
        return self._injector.resolve("PortfolioServiceProtocol")

    def create_risk_service(self) -> RiskServiceProtocol:
        """Create risk monitoring service using DI (returns service layer)."""
        return self._injector.resolve("RiskServiceProtocol")

    def create_reporting_service(self) -> ReportingServiceProtocol:
        """Create performance reporting service using DI (returns service layer)."""
        return self._injector.resolve("ReportingServiceProtocol")

    def create_operational_service(self) -> OperationalServiceProtocol:
        """Create operational analytics service using DI (returns service layer)."""
        return self._injector.resolve("OperationalServiceProtocol")

    def create_alert_service(self) -> AlertServiceProtocol:
        """Create alert management service using DI (returns service layer)."""
        return self._injector.resolve("AlertServiceProtocol")

    def create_export_service(self) -> ExportServiceProtocol:
        """Create data export service using DI (returns service layer)."""
        return self._injector.resolve("ExportServiceProtocol")

    def create_realtime_analytics_service(self) -> "RealtimeAnalyticsServiceProtocol":
        """Create realtime analytics service using DI (returns service layer)."""
        return self._injector.resolve("RealtimeAnalyticsServiceProtocol")


def create_default_analytics_service(
    config: AnalyticsConfiguration | None = None,
    injector=None,
) -> "AnalyticsService":
    """
    Create analytics service with default dependencies.

    Args:
        config: Analytics configuration
        injector: Required dependency injector

    Returns:
        Configured analytics service
    """
    if injector is None:
        from src.analytics.di_registration import configure_analytics_dependencies

        injector = configure_analytics_dependencies()

    factory = AnalyticsServiceFactory(injector=injector)
    return factory.create_analytics_service(config)
