"""
Risk Management Dependency Injection Registration.

This module registers all risk management services with the dependency injection
container following proper service layer patterns.
"""

from typing import TYPE_CHECKING

from src.core.dependency_injection import DependencyInjector
from src.core.logging import get_logger

if TYPE_CHECKING:
    pass

from .interfaces import (
    RiskManagementFactoryInterface,
)
from .service import RiskService
from .services import (
    PositionSizingService,
    RiskMetricsService,
    RiskMonitoringService,
    RiskValidationService,
)

logger = get_logger(__name__)


def register_risk_management_services(injector: DependencyInjector) -> None:
    """Register all risk management services with the dependency injector."""

    # Register service implementations using proper factory pattern
    def position_sizing_service_factory() -> "PositionSizingService":
        database_service = injector.resolve("DatabaseService")
        state_service = injector.resolve("StateService")
        config = injector.resolve("Config") if injector.has_service("Config") else None

        return PositionSizingService(
            database_service=database_service,
            state_service=state_service,
            config=config,
        )

    def risk_validation_service_factory() -> "RiskValidationService":
        database_service = injector.resolve("DatabaseService")
        state_service = injector.resolve("StateService")
        config = injector.resolve("Config") if injector.has_service("Config") else None

        return RiskValidationService(
            database_service=database_service,
            state_service=state_service,
            config=config,
        )

    def risk_metrics_service_factory() -> "RiskMetricsService":
        database_service = injector.resolve("DatabaseService")
        state_service = injector.resolve("StateService")
        config = injector.resolve("Config") if injector.has_service("Config") else None

        return RiskMetricsService(
            database_service=database_service,
            state_service=state_service,
            config=config,
        )

    def risk_monitoring_service_factory() -> "RiskMonitoringService":
        database_service = injector.resolve("DatabaseService")
        state_service = injector.resolve("StateService")
        config = injector.resolve("Config") if injector.has_service("Config") else None

        return RiskMonitoringService(
            database_service=database_service,
            state_service=state_service,
            config=config,
        )

    def risk_service_factory() -> "RiskService":
        database_service = injector.resolve("DatabaseService")
        state_service = injector.resolve("StateService")
        analytics_service = injector.resolve("AnalyticsService") if injector.has_service("AnalyticsService") else None
        metrics_service = injector.resolve("MetricsService") if injector.has_service("MetricsService") else None
        config = injector.resolve("Config") if injector.has_service("Config") else None

        return RiskService(
            database_service=database_service,
            state_service=state_service,
            analytics_service=analytics_service,
            config=config,
            metrics_service=metrics_service,
        )

    # Register service factories as singletons
    injector.register_factory("PositionSizingService", position_sizing_service_factory, singleton=True)
    injector.register_factory("RiskValidationService", risk_validation_service_factory, singleton=True)
    injector.register_factory("RiskMetricsService", risk_metrics_service_factory, singleton=True)
    injector.register_factory("RiskMonitoringService", risk_monitoring_service_factory, singleton=True)
    injector.register_factory("RiskService", risk_service_factory, singleton=True)

    # Register factory
    def risk_management_factory() -> "RiskManagementFactoryInterface":
        from .factory import RiskManagementFactory

        return RiskManagementFactory(injector=injector)

    injector.register_factory("RiskManagementFactory", risk_management_factory, singleton=True)

    # Register interface implementations
    injector.register_service(
        "PositionSizingServiceInterface",
        lambda: injector.resolve("PositionSizingService"),
        singleton=True,
    )

    injector.register_service(
        "RiskValidationServiceInterface",
        lambda: injector.resolve("RiskValidationService"),
        singleton=True,
    )

    injector.register_service(
        "RiskMetricsServiceInterface",
        lambda: injector.resolve("RiskMetricsService"),
        singleton=True,
    )

    injector.register_service(
        "RiskMonitoringServiceInterface",
        lambda: injector.resolve("RiskMonitoringService"),
        singleton=True,
    )

    injector.register_service("RiskServiceInterface", lambda: injector.resolve("RiskService"), singleton=True)

    injector.register_service(
        "RiskManagementFactoryInterface",
        lambda: injector.resolve("RiskManagementFactory"),
        singleton=True,
    )


def configure_risk_management_dependencies(
    injector: DependencyInjector | None = None,
) -> DependencyInjector:
    """
    Configure risk management dependencies with proper service lifetimes.

    Args:
        injector: Optional existing injector instance

    Returns:
        Configured dependency injector
    """
    if injector is None:
        injector = DependencyInjector()

    register_risk_management_services(injector)

    logger.info("Risk management services registered with dependency injector")

    return injector


# Service locator convenience functions
def get_risk_service(injector: DependencyInjector) -> "RiskService":
    """Get RiskService from DI container using service locator pattern."""
    return injector.resolve("RiskService")


def get_position_sizing_service(injector: DependencyInjector) -> "PositionSizingService":
    """Get PositionSizingService from DI container."""
    return injector.resolve("PositionSizingService")


def get_risk_validation_service(injector: DependencyInjector) -> "RiskValidationService":
    """Get RiskValidationService from DI container."""
    return injector.resolve("RiskValidationService")


def get_risk_metrics_service(injector: DependencyInjector) -> "RiskMetricsService":
    """Get RiskMetricsService from DI container."""
    return injector.resolve("RiskMetricsService")


def get_risk_monitoring_service(injector: DependencyInjector) -> "RiskMonitoringService":
    """Get RiskMonitoringService from DI container."""
    return injector.resolve("RiskMonitoringService")


def get_risk_management_factory(injector: DependencyInjector) -> "RiskManagementFactoryInterface":
    """Get RiskManagementFactory from DI container using service locator pattern."""
    return injector.resolve("RiskManagementFactory")
