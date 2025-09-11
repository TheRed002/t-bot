"""
Monitoring Service Factory.

This module provides factory functions for creating monitoring services
with proper dependency injection and configuration.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.monitoring.services import MonitoringService

from src.monitoring.interfaces import (
    AlertServiceInterface,
    DashboardServiceInterface,
    MetricsServiceInterface,
    PerformanceServiceInterface,
)


class MonitoringServiceFactory:
    """Factory for creating monitoring services with dependency injection."""

    def __init__(self, injector=None):
        """
        Initialize the factory with dependency injector.

        Args:
            injector: Dependency injection container
        """
        if injector is None:
            from src.core.exceptions import ComponentError

            raise ComponentError(
                "Injector must be provided to factory",
                component="MonitoringServiceFactory",
                operation="__init__",
                context={"missing_dependency": "injector"},
            )
        self._injector = injector

    def create_monitoring_service(self) -> "MonitoringService":
        """
        Create a complete monitoring service with all dependencies using DI.

        Returns:
            Configured monitoring service
        """
        return self._injector.resolve("MonitoringServiceInterface")

    def create_metrics_service(self) -> MetricsServiceInterface:
        """Create metrics service using DI (returns service layer)."""
        return self._injector.resolve("MetricsServiceInterface")

    def create_alert_service(self) -> AlertServiceInterface:
        """Create alert service using DI (returns service layer)."""
        return self._injector.resolve("AlertServiceInterface")

    def create_performance_service(self) -> PerformanceServiceInterface:
        """Create performance service using DI (returns service layer)."""
        return self._injector.resolve("PerformanceServiceInterface")

    def create_dashboard_service(self) -> DashboardServiceInterface:
        """Create dashboard service using DI (returns service layer)."""
        return self._injector.resolve("DashboardServiceInterface")


def create_default_monitoring_service(injector=None) -> "MonitoringService":
    """
    Create monitoring service with default dependencies.

    Args:
        injector: Required dependency injector

    Returns:
        Configured monitoring service
    """
    if injector is None:
        from src.core.dependency_injection import DependencyInjector
        from src.monitoring.di_registration import register_monitoring_services

        injector = DependencyInjector()
        register_monitoring_services(injector)

    factory = MonitoringServiceFactory(injector=injector)
    return factory.create_monitoring_service()
