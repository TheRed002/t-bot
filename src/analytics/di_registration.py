"""Analytics services dependency injection registration."""

from typing import TYPE_CHECKING

from src.core.dependency_injection import DependencyInjector
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.analytics.factory import AnalyticsServiceFactory
    from src.analytics.interfaces import (
        AlertServiceProtocol,
        ExportServiceProtocol,
        OperationalServiceProtocol,
        PortfolioServiceProtocol,
        RealtimeAnalyticsServiceProtocol,
        ReportingServiceProtocol,
        RiskServiceProtocol,
    )
    from src.analytics.repository import AnalyticsRepository
    from src.analytics.service import AnalyticsService

logger = get_logger(__name__)


def register_analytics_services(injector: DependencyInjector) -> None:
    """
    Register analytics services with the dependency injector.

    Args:
        injector: Dependency injector instance
    """

    # Register AnalyticsServiceFactory as singleton with proper DI
    def analytics_factory_factory() -> "AnalyticsServiceFactory":
        from src.analytics.factory import AnalyticsServiceFactory

        return AnalyticsServiceFactory(injector=injector)

    injector.register_factory("AnalyticsServiceFactory", analytics_factory_factory, singleton=True)

    # Register factory protocol interface
    injector.register_service(
        "AnalyticsServiceFactoryProtocol",
        lambda: injector.resolve("AnalyticsServiceFactory"),
        singleton=True,
    )

    # Register individual service factories using direct creation to avoid circular dependencies
    def portfolio_service_factory() -> "PortfolioServiceProtocol":
        from src.analytics.services.portfolio_analytics_service import PortfolioAnalyticsService

        config = (
            injector.resolve("AnalyticsConfiguration")
            if injector.is_registered("AnalyticsConfiguration")
            else None
        )
        return PortfolioAnalyticsService(config=config)

    def risk_service_factory() -> "RiskServiceProtocol":
        from src.analytics.services.risk_service import RiskService

        config = (
            injector.resolve("AnalyticsConfiguration")
            if injector.is_registered("AnalyticsConfiguration")
            else None
        )
        return RiskService(config=config)

    def reporting_service_factory() -> "ReportingServiceProtocol":
        from src.analytics.services.reporting_service import ReportingService

        config = (
            injector.resolve("AnalyticsConfiguration")
            if injector.is_registered("AnalyticsConfiguration")
            else None
        )
        return ReportingService(config=config)

    def operational_service_factory() -> "OperationalServiceProtocol":
        from src.analytics.services.operational_service import OperationalService

        config = (
            injector.resolve("AnalyticsConfiguration")
            if injector.is_registered("AnalyticsConfiguration")
            else None
        )
        return OperationalService(config=config)

    def alert_service_factory() -> "AlertServiceProtocol":
        from src.analytics.services.alert_service import AlertService

        config = (
            injector.resolve("AnalyticsConfiguration")
            if injector.is_registered("AnalyticsConfiguration")
            else None
        )
        return AlertService(config=config)

    def export_service_factory() -> "ExportServiceProtocol":
        from src.analytics.services.export_service import ExportService

        config = (
            injector.resolve("AnalyticsConfiguration")
            if injector.is_registered("AnalyticsConfiguration")
            else None
        )
        return ExportService(config=config)

    def realtime_analytics_service_factory() -> "RealtimeAnalyticsServiceProtocol":
        from src.analytics.services.realtime_analytics_service import RealtimeAnalyticsService

        config = (
            injector.resolve("AnalyticsConfiguration")
            if injector.is_registered("AnalyticsConfiguration")
            else None
        )
        return RealtimeAnalyticsService(config=config)

    # Register service factories
    injector.register_factory("PortfolioService", portfolio_service_factory, singleton=True)
    injector.register_factory("RiskService", risk_service_factory, singleton=True)
    injector.register_factory("ReportingService", reporting_service_factory, singleton=True)
    injector.register_factory("OperationalService", operational_service_factory, singleton=True)
    injector.register_factory("AlertService", alert_service_factory, singleton=True)
    injector.register_factory("ExportService", export_service_factory, singleton=True)
    injector.register_factory(
        "RealtimeAnalyticsService", realtime_analytics_service_factory, singleton=True
    )

    # Register interface implementations
    injector.register_service(
        "PortfolioServiceProtocol", lambda: injector.resolve("PortfolioService"), singleton=True
    )
    injector.register_service(
        "RiskServiceProtocol", lambda: injector.resolve("RiskService"), singleton=True
    )
    injector.register_service(
        "ReportingServiceProtocol", lambda: injector.resolve("ReportingService"), singleton=True
    )
    injector.register_service(
        "OperationalServiceProtocol", lambda: injector.resolve("OperationalService"), singleton=True
    )
    injector.register_service(
        "AlertServiceProtocol", lambda: injector.resolve("AlertService"), singleton=True
    )
    injector.register_service(
        "ExportServiceProtocol", lambda: injector.resolve("ExportService"), singleton=True
    )
    injector.register_service(
        "RealtimeAnalyticsServiceProtocol",
        lambda: injector.resolve("RealtimeAnalyticsService"),
        singleton=True,
    )

    # Register MetricsCollector as singleton - this is a common dependency
    def metrics_collector_factory():
        """Create metrics collector instance."""
        from src.monitoring.metrics import get_metrics_collector

        return get_metrics_collector()

    # Register MetricsCollector as singleton if not already registered
    if not injector.is_registered("MetricsCollector"):
        injector.register_factory("MetricsCollector", metrics_collector_factory, singleton=True)

    # Register AnalyticsConfiguration as singleton
    def analytics_config_factory():
        """Create analytics configuration instance."""
        from src.analytics.types import AnalyticsConfiguration

        return AnalyticsConfiguration()

    injector.register_factory("AnalyticsConfiguration", analytics_config_factory, singleton=True)

    # Register DataTransformationService
    def data_transformation_service_factory():
        """Create data transformation service instance."""
        from src.analytics.services.data_transformation_service import DataTransformationService

        return DataTransformationService()

    injector.register_factory(
        "DataTransformationService", data_transformation_service_factory, singleton=True
    )

    # Register AnalyticsRepository with proper database integration and service injection
    def analytics_repository_factory() -> "AnalyticsRepository":
        from src.analytics.repository import AnalyticsRepository

        # Resolve dependencies from injector
        try:
            session = injector.resolve("AsyncSession")
        except Exception:
            # Fallback if database session not available
            session = None
        transformation_service = injector.resolve("DataTransformationService")
        return AnalyticsRepository(session=session, transformation_service=transformation_service)

    injector.register_factory("AnalyticsRepository", analytics_repository_factory, singleton=True)
    injector.register_service(
        "AnalyticsDataRepository", lambda: injector.resolve("AnalyticsRepository"), singleton=True
    )

    # Register main AnalyticsService using direct creation to avoid circular dependencies
    def analytics_service_factory() -> "AnalyticsService":
        from src.analytics.service import AnalyticsService

        # Get configuration
        config = (
            injector.resolve("AnalyticsConfiguration")
            if injector.is_registered("AnalyticsConfiguration")
            else None
        )

        # Resolve service dependencies safely
        dependencies = {}
        service_mappings = {
            "realtime_analytics": "RealtimeAnalyticsService",
            "portfolio_service": "PortfolioService",
            "reporting_service": "ReportingService",
            "risk_service": "RiskService",
            "operational_service": "OperationalService",
            "alert_service": "AlertService",
            "export_service": "ExportService",
            "metrics_collector": "MetricsCollector",
        }

        for param_name, service_name in service_mappings.items():
            try:
                if injector.is_registered(service_name):
                    dependencies[param_name] = injector.resolve(service_name)
                else:
                    dependencies[param_name] = None
            except Exception:
                dependencies[param_name] = None

        return AnalyticsService(config=config, **dependencies)

    injector.register_factory("AnalyticsService", analytics_service_factory, singleton=True)

    # Register interface implementation
    injector.register_service(
        "AnalyticsServiceProtocol", lambda: injector.resolve("AnalyticsService"), singleton=True
    )

    logger.info("Analytics services registered with dependency injector")


def configure_analytics_dependencies(
    injector: DependencyInjector | None = None,
) -> DependencyInjector:
    """
    Configure analytics dependencies with proper service lifetimes.

    Args:
        injector: Optional existing injector instance

    Returns:
        Configured dependency injector
    """
    if injector is None:
        injector = DependencyInjector()

    register_analytics_services(injector)

    return injector


# Service locator convenience functions
def get_analytics_service(injector: DependencyInjector) -> "AnalyticsService":
    """Get AnalyticsService from DI container using service locator pattern."""
    return injector.resolve("AnalyticsService")


def get_analytics_factory(injector: DependencyInjector) -> "AnalyticsServiceFactory":
    """Get AnalyticsServiceFactory from DI container."""
    return injector.resolve("AnalyticsServiceFactory")
