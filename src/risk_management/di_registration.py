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

# Repository implementations
from src.database.repository.risk import (
    PortfolioRepository,
    PortfolioRepositoryImpl,
    RiskMetricsRepository,
    RiskMetricsRepositoryImpl,
)

from .interfaces import (
    PortfolioRepositoryInterface,
    RiskManagementFactoryInterface,
    RiskMetricsRepositoryInterface,
)
from .service import RiskService
from .services import (
    PortfolioLimitsService,
    PositionSizingService,
    RiskMetricsService,
    RiskMonitoringService,
    RiskValidationService,
)

logger = get_logger(__name__)


def register_risk_management_services(injector: DependencyInjector) -> None:
    """Register all risk management services with the dependency injector."""

    # Register repository factory functions first
    def risk_metrics_repository_factory() -> RiskMetricsRepositoryInterface:
        try:
            database_service = injector.resolve("DatabaseService")
            # Create session directly from database service
            session = getattr(database_service, "_session", None) or database_service.session
            repository = RiskMetricsRepository(session)
            return RiskMetricsRepositoryImpl(repository)
        except Exception as e:
            logger.error(f"Failed to create RiskMetricsRepository: {e}")
            # Return placeholder implementation for now
            class PlaceholderRiskMetricsRepository:
                async def get_historical_returns(self, symbol: str, days: int) -> list:
                    return []
                async def get_price_history(self, symbol: str, days: int) -> list:
                    return []
                async def get_portfolio_positions(self) -> list:
                    return []
                async def save_risk_metrics(self, metrics) -> None:
                    pass
                async def get_correlation_data(self, symbols: list[str], days: int) -> dict:
                    return {symbol: [] for symbol in symbols}
            return PlaceholderRiskMetricsRepository()

    def portfolio_repository_factory() -> PortfolioRepositoryInterface:
        try:
            database_service = injector.resolve("DatabaseService")
            # Create session directly from database service
            session = getattr(database_service, "_session", None) or database_service.session
            repository = PortfolioRepository(session)
            return PortfolioRepositoryImpl(repository)
        except Exception as e:
            logger.error(f"Failed to create PortfolioRepository: {e}")
            # Return placeholder implementation for now
            class PlaceholderPortfolioRepository:
                async def get_current_positions(self) -> list:
                    return []
                async def get_portfolio_value(self):
                    from decimal import Decimal
                    return Decimal("0")
                async def get_position_history(self, symbol: str, days: int) -> list:
                    return []
                async def update_portfolio_limits(self, limits: dict) -> None:
                    pass
            return PlaceholderPortfolioRepository()

    # Register repositories
    injector.register_factory("RiskMetricsRepository", risk_metrics_repository_factory)
    injector.register_factory("PortfolioRepository", portfolio_repository_factory)

    # Register service implementations using proper factory pattern with dependency injection
    def position_sizing_service_factory() -> "PositionSizingService":
        try:
            risk_metrics_repository = injector.resolve("RiskMetricsRepository")
            state_service = injector.resolve("StateService")
            config = injector.resolve("Config") if injector.has_service("Config") else None

            return PositionSizingService(
                risk_metrics_repository=risk_metrics_repository,
                state_service=state_service,
                config=config,
            )
        except Exception as e:
            logger.error(f"Failed to create PositionSizingService: {e}")
            raise

    def portfolio_limits_service_factory() -> "PortfolioLimitsService":
        try:
            config = injector.resolve("Config") if injector.has_service("Config") else None

            return PortfolioLimitsService(config=config)
        except Exception as e:
            logger.error(f"Failed to create PortfolioLimitsService: {e}")
            raise

    def risk_validation_service_factory() -> "RiskValidationService":
        try:
            portfolio_repository = injector.resolve("PortfolioRepository")
            state_service = injector.resolve("StateService")
            config = injector.resolve("Config") if injector.has_service("Config") else None

            return RiskValidationService(
                portfolio_repository=portfolio_repository,
                state_service=state_service,
                config=config,
            )
        except Exception as e:
            logger.error(f"Failed to create RiskValidationService: {e}")
            raise

    def risk_metrics_service_factory() -> "RiskMetricsService":
        try:
            risk_metrics_repository = injector.resolve("RiskMetricsRepository")
            state_service = injector.resolve("StateService")
            config = injector.resolve("Config") if injector.has_service("Config") else None

            return RiskMetricsService(
                risk_metrics_repository=risk_metrics_repository,
                state_service=state_service,
                config=config,
            )
        except Exception as e:
            logger.error(f"Failed to create RiskMetricsService: {e}")
            raise

    def risk_monitoring_service_factory() -> "RiskMonitoringService":
        try:
            portfolio_repository = injector.resolve("PortfolioRepository")
            state_service = injector.resolve("StateService")
            config = injector.resolve("Config") if injector.has_service("Config") else None

            return RiskMonitoringService(
                portfolio_repository=portfolio_repository,
                state_service=state_service,
                config=config,
            )
        except Exception as e:
            logger.error(f"Failed to create RiskMonitoringService: {e}")
            raise

    def risk_service_factory() -> "RiskService":
        try:
            risk_metrics_repository = injector.resolve("RiskMetricsRepository")
            portfolio_repository = injector.resolve("PortfolioRepository")
            state_service = injector.resolve("StateService")
            # Use interface to avoid circular dependency
            analytics_service = (
                injector.resolve("AnalyticsService")
                if injector.has_service("AnalyticsService")
                else None
            )
            metrics_service = (
                injector.resolve("MetricsService")
                if injector.has_service("MetricsService")
                else None
            )
            cache_service = (
                injector.resolve("CacheService") if injector.has_service("CacheService") else None
            )
            config = injector.resolve("Config") if injector.has_service("Config") else None

            return RiskService(
                risk_metrics_repository=risk_metrics_repository,
                portfolio_repository=portfolio_repository,
                state_service=state_service,
                analytics_service=analytics_service,
                config=config,
                metrics_service=metrics_service,
                cache_service=cache_service,
            )
        except Exception as e:
            logger.error(f"Failed to create RiskService: {e}")
            raise

    # Register service factories as singletons
    injector.register_factory(
        "PositionSizingService", position_sizing_service_factory, singleton=True
    )
    injector.register_factory(
        "PortfolioLimitsService", portfolio_limits_service_factory, singleton=True
    )
    injector.register_factory(
        "RiskValidationService", risk_validation_service_factory, singleton=True
    )
    injector.register_factory("RiskMetricsService", risk_metrics_service_factory, singleton=True)
    injector.register_factory(
        "RiskMonitoringService", risk_monitoring_service_factory, singleton=True
    )
    injector.register_factory("RiskService", risk_service_factory, singleton=True)

    # Register factory using proper dependency injection pattern
    def risk_management_factory() -> "RiskManagementFactoryInterface":
        from .factory import RiskManagementFactory

        try:
            # Pass the injector instance - circular dependency is resolved by lazy loading
            return RiskManagementFactory(injector=injector)
        except Exception as e:
            logger.error(f"Failed to create RiskManagementFactory: {e}")
            raise

    injector.register_factory("RiskManagementFactory", risk_management_factory, singleton=True)

    # Register interface implementations using factory pattern
    injector.register_factory(
        "PositionSizingServiceInterface",
        lambda: injector.resolve("PositionSizingService"),
        singleton=True,
    )

    injector.register_factory(
        "RiskValidationServiceInterface",
        lambda: injector.resolve("RiskValidationService"),
        singleton=True,
    )

    injector.register_factory(
        "RiskMetricsServiceInterface",
        lambda: injector.resolve("RiskMetricsService"),
        singleton=True,
    )

    injector.register_factory(
        "RiskMonitoringServiceInterface",
        lambda: injector.resolve("RiskMonitoringService"),
        singleton=True,
    )

    injector.register_factory(
        "RiskServiceInterface", lambda: injector.resolve("RiskService"), singleton=True
    )

    injector.register_factory(
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
