"""Analytics Module Integration Tests.

Tests to verify analytics module properly integrates with other modules
and follows architectural patterns.

NO MOCKS - All operations use real service implementations.
Simplified version that works without Docker infrastructure.
"""

import pytest

from src.analytics import (
    AnalyticsService,
    AnalyticsServiceFactory,
    get_analytics_factory,
    get_analytics_service,
)
from src.analytics.types import AnalyticsConfiguration
from src.core.dependency_injection import DependencyInjector


@pytest.fixture(scope="module")
def analytics_injector():
    """Create analytics dependencies with REAL services."""
    from src.analytics.services.data_transformation_service import DataTransformationService
    from src.analytics.services.portfolio_analytics_service import PortfolioAnalyticsService
    from src.analytics.services.risk_service import RiskService
    from src.analytics.services.reporting_service import ReportingService
    from src.analytics.services.operational_service import OperationalService
    from src.analytics.services.alert_service import AlertService
    from src.analytics.services.export_service import ExportService

    # Create DependencyInjector
    injector = DependencyInjector()

    # Create analytics config
    config = AnalyticsConfiguration()

    # Create and register transformation service
    transformation_service = DataTransformationService()
    injector.register_service("DataTransformationServiceProtocol", transformation_service, singleton=True)

    # Create and register portfolio service
    portfolio_service = PortfolioAnalyticsService(config=config)
    injector.register_service("PortfolioServiceProtocol", portfolio_service, singleton=True)

    # Create and register risk service
    risk_service = RiskService(config=config)
    injector.register_service("RiskServiceProtocol", risk_service, singleton=True)

    # Create and register reporting service
    reporting_service = ReportingService(config=config)
    injector.register_service("ReportingServiceProtocol", reporting_service, singleton=True)

    # Create and register operational service
    operational_service = OperationalService(config=config)
    injector.register_service("OperationalServiceProtocol", operational_service, singleton=True)

    # Create and register alert service
    alert_service = AlertService(config=config)
    injector.register_service("AlertServiceProtocol", alert_service, singleton=True)

    # Create and register export service
    export_service = ExportService(config=config)
    injector.register_service("ExportServiceProtocol", export_service, singleton=True)

    # Register factory
    factory = AnalyticsServiceFactory(injector=injector)
    injector.register_service("AnalyticsServiceFactory", factory, singleton=True)

    return injector


class TestAnalyticsDependencyInjection:
    """Test analytics dependency injection patterns with real services."""

    def test_analytics_services_registration(self, analytics_injector):
        """Test that all analytics services can be registered with DI using real services."""
        injector = analytics_injector

        assert injector is not None
        assert isinstance(injector, DependencyInjector)

    def test_analytics_factory_resolution(self, analytics_injector):
        """Test that AnalyticsServiceFactory can be resolved with real dependencies."""
        injector = analytics_injector

        factory = get_analytics_factory(injector)

        assert isinstance(factory, AnalyticsServiceFactory)

    def test_individual_service_resolution(self, analytics_injector):
        """Test that individual analytics services can be resolved with real dependencies."""
        injector = analytics_injector

        portfolio_service = injector.resolve("PortfolioServiceProtocol")
        risk_service = injector.resolve("RiskServiceProtocol")
        reporting_service = injector.resolve("ReportingServiceProtocol")

        assert portfolio_service is not None
        assert risk_service is not None
        assert reporting_service is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
