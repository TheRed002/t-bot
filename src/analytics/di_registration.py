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

    # Register individual service factories using DI resolution pattern
    def portfolio_service_factory() -> "PortfolioServiceProtocol":
        from src.analytics.portfolio.portfolio_analytics import PortfolioAnalyticsEngine
        from src.analytics.services.portfolio_analytics_service import PortfolioAnalyticsService

        config = injector.resolve("AnalyticsConfiguration")
        metrics_collector = injector.resolve("MetricsCollector")
        analytics_repository = injector.resolve("AnalyticsDataRepository")

        # Create engine and wrap in service layer with proper DI
        engine = PortfolioAnalyticsEngine(
            config=config, metrics_collector=metrics_collector, repository=analytics_repository
        )

        return PortfolioAnalyticsService(config=config, analytics_engine=engine)

    def risk_service_factory() -> "RiskServiceProtocol":
        from src.analytics.risk.risk_monitor import RiskMonitor
        from src.analytics.services.risk_service import RiskService

        config = injector.resolve("AnalyticsConfiguration")
        metrics_collector = injector.resolve("MetricsCollector")

        # Create engine and wrap in service layer with proper DI
        engine = RiskMonitor(config=config, metrics_collector=metrics_collector)
        return RiskService(config=config, risk_monitor=engine)

    def reporting_service_factory() -> "ReportingServiceProtocol":
        from src.analytics.reporting.performance_reporter import PerformanceReporter
        from src.analytics.services.reporting_service import ReportingService

        config = injector.resolve("AnalyticsConfiguration")
        metrics_collector = injector.resolve("MetricsCollector")

        # Create engine and wrap in service layer with proper DI
        engine = PerformanceReporter(config=config, metrics_collector=metrics_collector)
        return ReportingService(config=config, performance_reporter=engine)

    def operational_service_factory() -> "OperationalServiceProtocol":
        from src.analytics.operational.operational_analytics import OperationalAnalyticsEngine
        from src.analytics.services.operational_service import OperationalService

        config = injector.resolve("AnalyticsConfiguration")
        metrics_collector = injector.resolve("MetricsCollector")

        # Create engine and wrap in service layer with proper DI
        engine = OperationalAnalyticsEngine(config=config, metrics_collector=metrics_collector)
        return OperationalService(config=config, operational_engine=engine)

    def alert_service_factory() -> "AlertServiceProtocol":
        from src.analytics.alerts.alert_manager import AlertManager
        from src.analytics.services.alert_service import AlertService

        config = injector.resolve("AnalyticsConfiguration")
        metrics_collector = injector.resolve("MetricsCollector")

        # Create engine and wrap in service layer with proper DI
        engine = AlertManager(config=config, metrics_collector=metrics_collector)
        return AlertService(config=config, alert_manager=engine)

    def export_service_factory() -> "ExportServiceProtocol":
        from src.analytics.export.data_exporter import DataExporter
        from src.analytics.services.export_service import ExportService

        metrics_collector = injector.resolve("MetricsCollector")

        # Create engine and wrap in service layer
        engine = DataExporter(metrics_collector=metrics_collector)
        return ExportService(data_exporter=engine)

    def realtime_analytics_service_factory() -> "RealtimeAnalyticsServiceProtocol":
        from src.analytics.services.realtime_analytics_service import RealtimeAnalyticsService
        from src.analytics.trading.realtime_analytics import RealtimeAnalyticsEngine

        config = injector.resolve("AnalyticsConfiguration")
        metrics_collector = injector.resolve("MetricsCollector")

        # Create the underlying engine with proper DI
        engine = RealtimeAnalyticsEngine(config=config, metrics_collector=metrics_collector)

        # Wrap in service layer
        return RealtimeAnalyticsService(
            config=config, analytics_engine=engine, metrics_collector=metrics_collector
        )

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
        from src.monitoring.metrics import get_metrics_collector

        return get_metrics_collector()

    injector.register_factory("MetricsCollector", metrics_collector_factory, singleton=True)

    # Register AnalyticsConfiguration as singleton
    def analytics_config_factory():
        from src.analytics.types import AnalyticsConfiguration

        return AnalyticsConfiguration()

    injector.register_factory("AnalyticsConfiguration", analytics_config_factory, singleton=True)

    # Register AnalyticsRepository with proper database integration
    def analytics_repository_factory() -> "AnalyticsRepository":
        from src.analytics.repository import AnalyticsRepository

        # Resolve UnitOfWork from database module
        uow = injector.resolve("UnitOfWork")
        return AnalyticsRepository(uow=uow)

    injector.register_factory("AnalyticsRepository", analytics_repository_factory, singleton=True)
    injector.register_service(
        "AnalyticsDataRepository", lambda: injector.resolve("AnalyticsRepository"), singleton=True
    )

    # Register main AnalyticsService using factory pattern
    def analytics_service_factory() -> "AnalyticsService":
        from src.analytics.service import AnalyticsService

        config = injector.resolve("AnalyticsConfiguration")
        metrics_collector = injector.resolve("MetricsCollector")

        # Resolve all service dependencies through DI
        realtime_analytics = injector.resolve("RealtimeAnalyticsServiceProtocol")
        portfolio_service = injector.resolve("PortfolioServiceProtocol")
        reporting_service = injector.resolve("ReportingServiceProtocol")
        risk_service = injector.resolve("RiskServiceProtocol")
        operational_service = injector.resolve("OperationalServiceProtocol")
        alert_service = injector.resolve("AlertServiceProtocol")
        export_service = injector.resolve("ExportServiceProtocol")

        return AnalyticsService(
            config=config,
            realtime_analytics=realtime_analytics,
            portfolio_service=portfolio_service,
            reporting_service=reporting_service,
            risk_service=risk_service,
            operational_service=operational_service,
            alert_service=alert_service,
            export_service=export_service,
            metrics_collector=metrics_collector,
        )

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
