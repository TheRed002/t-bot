"""Monitoring services dependency injection registration."""

from src.core.dependency_injection import DependencyInjector
from src.core.logging import get_logger
from src.monitoring.alerting import AlertManager, NotificationConfig
from src.monitoring.interfaces import (
    AlertServiceInterface,
    DashboardServiceInterface,
    MetricsServiceInterface,
    MonitoringServiceInterface,
    PerformanceServiceInterface,
)
from src.monitoring.metrics import MetricsCollector
from src.monitoring.performance import PerformanceProfiler
from src.monitoring.services import (
    DefaultAlertService,
    DefaultDashboardService,
    DefaultMetricsService,
    DefaultPerformanceService,
    MonitoringService,
)

logger = get_logger(__name__)


def register_monitoring_services(injector: DependencyInjector) -> None:
    """
    Register monitoring services with the dependency injector.

    Args:
        injector: Dependency injector instance
    """

    # Register MetricsCollector as singleton
    def metrics_collector_factory() -> MetricsCollector:
        return MetricsCollector()

    injector.register_factory(
        "MetricsCollector",
        metrics_collector_factory,
        singleton=True
    )

    # Register AlertManager as singleton
    def alert_manager_factory() -> AlertManager:
        return AlertManager(NotificationConfig())

    injector.register_factory(
        "AlertManager",
        alert_manager_factory,
        singleton=True
    )

    # Register PerformanceProfiler as singleton with dependency injection
    def performance_profiler_factory() -> PerformanceProfiler:
        metrics_collector = injector.resolve("MetricsCollector")
        alert_manager = injector.resolve("AlertManager")
        return PerformanceProfiler(
            metrics_collector=metrics_collector,
            alert_manager=alert_manager
        )

    injector.register_factory(
        "PerformanceProfiler",
        performance_profiler_factory,
        singleton=True
    )

    # Register GrafanaDashboardManager as singleton
    def dashboard_manager_factory():
        import os

        from src.monitoring.dashboards import GrafanaDashboardManager

        grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3000")
        api_key = os.getenv("GRAFANA_API_KEY", "")

        # Try to inject error handler if available
        error_handler = None
        try:
            error_handler = injector.resolve("ErrorHandler")
        except (KeyError, ValueError):
            # Error handler is optional dependency
            pass

        return GrafanaDashboardManager(grafana_url, api_key, error_handler)

    injector.register_factory(
        "GrafanaDashboardManager",
        dashboard_manager_factory,
        singleton=True
    )

    # Register service implementations with proper dependency injection

    # DefaultMetricsService
    def metrics_service_factory() -> DefaultMetricsService:
        metrics_collector = injector.resolve("MetricsCollector")
        return DefaultMetricsService(metrics_collector)

    injector.register_factory(
        "DefaultMetricsService",
        metrics_service_factory,
        singleton=True
    )

    # DefaultAlertService
    def alert_service_factory() -> DefaultAlertService:
        alert_manager = injector.resolve("AlertManager")
        return DefaultAlertService(alert_manager)

    injector.register_factory(
        "DefaultAlertService",
        alert_service_factory,
        singleton=True
    )

    # DefaultPerformanceService
    def performance_service_factory() -> DefaultPerformanceService:
        performance_profiler = injector.resolve("PerformanceProfiler")
        return DefaultPerformanceService(performance_profiler)

    injector.register_factory(
        "DefaultPerformanceService",
        performance_service_factory,
        singleton=True
    )

    # DefaultDashboardService
    def dashboard_service_factory() -> DefaultDashboardService:
        dashboard_manager = injector.resolve("GrafanaDashboardManager")
        return DefaultDashboardService(dashboard_manager)

    injector.register_factory(
        "DefaultDashboardService",
        dashboard_service_factory,
        singleton=True
    )

    # Register service interfaces bound to implementations
    def metrics_service_interface_factory() -> MetricsServiceInterface:
        return injector.resolve("DefaultMetricsService")

    def alert_service_interface_factory() -> AlertServiceInterface:
        return injector.resolve("DefaultAlertService")

    def performance_service_interface_factory() -> PerformanceServiceInterface:
        return injector.resolve("DefaultPerformanceService")

    def dashboard_service_interface_factory() -> DashboardServiceInterface:
        return injector.resolve("DefaultDashboardService")

    injector.register_factory("MetricsServiceInterface", metrics_service_interface_factory, singleton=True)
    injector.register_factory("AlertServiceInterface", alert_service_interface_factory, singleton=True)
    injector.register_factory("PerformanceServiceInterface", performance_service_interface_factory, singleton=True)
    injector.register_factory("DashboardServiceInterface", dashboard_service_interface_factory, singleton=True)

    # Register composite MonitoringService
    def monitoring_service_factory() -> MonitoringService:
        alert_service = injector.resolve("AlertServiceInterface")
        metrics_service = injector.resolve("MetricsServiceInterface")
        performance_service = injector.resolve("PerformanceServiceInterface")
        return MonitoringService(alert_service, metrics_service, performance_service)

    injector.register_factory(
        "MonitoringService",
        monitoring_service_factory,
        singleton=True
    )

    # Register main interface
    def monitoring_service_interface_factory() -> MonitoringServiceInterface:
        return injector.resolve("MonitoringService")

    injector.register_factory("MonitoringServiceInterface", monitoring_service_interface_factory, singleton=True)

    logger.info("Monitoring services registered with dependency injector")
