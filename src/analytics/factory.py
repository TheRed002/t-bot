"""
Analytics Service Factory.

This module provides factory functions for creating analytics services
with proper dependency injection and configuration.
"""

from typing import Any

from src.analytics.alerts.alert_manager import AlertManager
from src.analytics.export.data_exporter import DataExporter
from src.analytics.interfaces import (
    AlertServiceProtocol,
    AnalyticsDataRepository,
    ExportServiceProtocol,
    MetricsCalculationService,
    OperationalServiceProtocol,
    PortfolioServiceProtocol,
    ReportingServiceProtocol,
    RiskCalculationService,
    RiskServiceProtocol,
)
from src.analytics.operational.operational_analytics import OperationalAnalyticsEngine
from src.analytics.portfolio.portfolio_analytics import PortfolioAnalyticsEngine
from src.analytics.reporting.performance_reporter import PerformanceReporter
from src.analytics.risk.risk_monitor import RiskMonitor
from src.analytics.service import AnalyticsService
from src.analytics.trading.realtime_analytics import RealtimeAnalyticsEngine
from src.analytics.types import AnalyticsConfiguration


class AnalyticsServiceFactory:
    """Factory for creating analytics services with dependency injection."""

    def __init__(
        self,
        metrics_collector=None,
        data_repository: AnalyticsDataRepository | None = None,
        metrics_calculation_service: MetricsCalculationService | None = None,
        risk_calculation_service: RiskCalculationService | None = None,
    ):
        """
        Initialize the factory with optional dependencies.

        Args:
            metrics_collector: Metrics collection service
            data_repository: Data repository implementation
            metrics_calculation_service: Metrics calculation service
            risk_calculation_service: Risk calculation service
        """
        self.metrics_collector = metrics_collector
        self.data_repository = data_repository
        self.metrics_calculation_service = metrics_calculation_service
        self.risk_calculation_service = risk_calculation_service

    def create_analytics_service(
        self,
        config: AnalyticsConfiguration | None = None,
        custom_services: dict[str, Any] | None = None,
    ) -> AnalyticsService:
        """
        Create a complete analytics service with all dependencies.

        Args:
            config: Analytics configuration
            custom_services: Dictionary of custom service implementations

        Returns:
            Configured analytics service
        """
        config = config or AnalyticsConfiguration()
        custom_services = custom_services or {}

        # Create individual services
        portfolio_service = custom_services.get("portfolio") or self._create_portfolio_service(
            config
        )
        risk_service = custom_services.get("risk") or self._create_risk_service(config)
        reporting_service = custom_services.get("reporting") or self._create_reporting_service(
            config
        )
        operational_service = custom_services.get(
            "operational"
        ) or self._create_operational_service(config)
        alert_service = custom_services.get("alert") or self._create_alert_service(config)
        export_service = custom_services.get("export") or self._create_export_service()

        # Create realtime analytics engine
        realtime_analytics = self._create_realtime_analytics(config)

        # Create main analytics service with injected dependencies
        return AnalyticsService(
            config=config,
            realtime_analytics=realtime_analytics,
            portfolio_service=portfolio_service,
            reporting_service=reporting_service,
            risk_service=risk_service,
            operational_service=operational_service,
            alert_service=alert_service,
            export_service=export_service,
            metrics_collector=self.metrics_collector,
        )

    def create_portfolio_service(self, config: AnalyticsConfiguration) -> PortfolioServiceProtocol:
        """Create portfolio analytics service."""
        return self._create_portfolio_service(config)

    def create_risk_service(self, config: AnalyticsConfiguration) -> RiskServiceProtocol:
        """Create risk monitoring service."""
        return self._create_risk_service(config)

    def create_reporting_service(self, config: AnalyticsConfiguration) -> ReportingServiceProtocol:
        """Create performance reporting service."""
        return self._create_reporting_service(config)

    def create_operational_service(
        self, config: AnalyticsConfiguration
    ) -> OperationalServiceProtocol:
        """Create operational analytics service."""
        return self._create_operational_service(config)

    def create_alert_service(self, config: AnalyticsConfiguration) -> AlertServiceProtocol:
        """Create alert management service."""
        return self._create_alert_service(config)

    def create_export_service(self) -> ExportServiceProtocol:
        """Create data export service."""
        return self._create_export_service()

    def _create_portfolio_service(self, config: AnalyticsConfiguration) -> PortfolioAnalyticsEngine:
        """Create portfolio analytics service with dependencies."""
        service = PortfolioAnalyticsEngine(config)
        # Inject additional dependencies if available
        if self.data_repository:
            service._data_repository = self.data_repository
        if self.metrics_calculation_service:
            service._metrics_service = self.metrics_calculation_service
        return service

    def _create_risk_service(self, config: AnalyticsConfiguration) -> RiskMonitor:
        """Create risk monitoring service with dependencies."""
        service = RiskMonitor(config)
        # Inject additional dependencies if available
        if self.risk_calculation_service:
            service._risk_calc_service = self.risk_calculation_service
        if self.data_repository:
            service._data_repository = self.data_repository
        return service

    def _create_reporting_service(self, config: AnalyticsConfiguration) -> PerformanceReporter:
        """Create performance reporting service with dependencies."""
        service = PerformanceReporter(config)
        # Inject additional dependencies if available
        if self.data_repository:
            service._data_repository = self.data_repository
        return service

    def _create_operational_service(
        self, config: AnalyticsConfiguration
    ) -> OperationalAnalyticsEngine:
        """Create operational analytics service with dependencies."""
        service = OperationalAnalyticsEngine(config)
        # Inject additional dependencies if available
        if self.metrics_collector:
            service.metrics_collector = self.metrics_collector
        return service

    def _create_alert_service(self, config: AnalyticsConfiguration) -> AlertManager:
        """Create alert management service with dependencies."""
        service = AlertManager(config)
        # Inject additional dependencies if available
        if self.metrics_collector:
            service.metrics_collector = self.metrics_collector
        return service

    def _create_export_service(self) -> DataExporter:
        """Create data export service with dependencies."""
        service = DataExporter()
        # Inject additional dependencies if available
        if self.metrics_collector:
            service.metrics_collector = self.metrics_collector
        return service

    def _create_realtime_analytics(self, config: AnalyticsConfiguration) -> RealtimeAnalyticsEngine:
        """Create realtime analytics engine with dependencies."""
        service = RealtimeAnalyticsEngine(config)
        # Inject additional dependencies if available
        if self.metrics_collector:
            service.metrics_collector = self.metrics_collector
        return service


def create_default_analytics_service(
    config: AnalyticsConfiguration | None = None,
) -> AnalyticsService:
    """
    Create analytics service with default dependencies.

    Args:
        config: Analytics configuration

    Returns:
        Configured analytics service
    """
    factory = AnalyticsServiceFactory()
    return factory.create_analytics_service(config)


def create_analytics_service_with_custom_dependencies(
    config: AnalyticsConfiguration,
    custom_services: dict[str, Any],
    **kwargs,
) -> AnalyticsService:
    """
    Create analytics service with custom dependencies.

    Args:
        config: Analytics configuration
        custom_services: Custom service implementations
        **kwargs: Additional factory dependencies

    Returns:
        Configured analytics service with custom dependencies
    """
    factory = AnalyticsServiceFactory(**kwargs)
    return factory.create_analytics_service(config, custom_services)
