"""Dependency injection registration for strategies module."""

from typing import Any

from src.core.dependency_injection import DependencyInjector
from src.core.logging import get_logger
from src.strategies.factory import StrategyFactory
from src.strategies.dynamic.strategy_factory import DynamicStrategyFactory
from src.strategies.repository import StrategyRepository
from src.strategies.service import StrategyService

logger = get_logger(__name__)


def register_strategies_dependencies(injector: DependencyInjector) -> None:
    """Register all strategies module dependencies using DependencyInjector API.

    Args:
        injector: DependencyInjector instance (not DependencyContainer)

    Note:
        This follows the same pattern as risk_management module which uses
        injector.register_factory() instead of container.register()
    """
    try:
        logger.info("Registering strategies module dependencies")

        # Register factory for strategy service with dependencies
        def strategy_service_factory() -> StrategyService:
            """Factory function for StrategyService with injected dependencies."""
            # Dependencies will be resolved by DI container when available
            repository = None
            risk_manager = None
            exchange_factory = None
            data_service = None
            service_manager = None

            # Try to resolve dependencies if available
            if injector.has_service("StrategyRepository"):
                try:
                    repository = injector.resolve("StrategyRepository")
                except Exception:
                    pass

            if injector.has_service("RiskService"):
                try:
                    risk_manager = injector.resolve("RiskService")
                except Exception:
                    pass

            if injector.has_service("ExchangeFactory"):
                try:
                    exchange_factory = injector.resolve("ExchangeFactory")
                except Exception:
                    pass

            if injector.has_service("DataService"):
                try:
                    data_service = injector.resolve("DataService")
                except Exception:
                    pass

            if injector.has_service("ServiceManager"):
                try:
                    service_manager = injector.resolve("ServiceManager")
                except Exception:
                    pass

            return StrategyService(
                name="StrategyService",
                config={},
                repository=repository,
                risk_manager=risk_manager,
                exchange_factory=exchange_factory,
                data_service=data_service,
                service_manager=service_manager,
            )

        # Register using DependencyInjector.register_factory() API
        injector.register_factory("StrategyService", strategy_service_factory, singleton=True)

        # Register factory for repository with database session
        def strategy_repository_factory() -> StrategyRepository:
            """Factory function for StrategyRepository with database session."""
            # Try to get database service
            session = None
            if injector.has_service("DatabaseService"):
                try:
                    db_service = injector.resolve("DatabaseService")
                    if hasattr(db_service, "get_session"):
                        session = db_service.get_session()
                except Exception:
                    pass

            return StrategyRepository(session=session)

        # Register using DependencyInjector.register_factory() API (not singleton)
        injector.register_factory("StrategyRepository", strategy_repository_factory, singleton=False)

        # Register factory for strategy factory with dependencies
        def strategy_factory_factory() -> StrategyFactory:
            """Factory function for StrategyFactory with injected dependencies."""
            # Dependencies will be resolved by DI container when available
            validation_framework = None
            repository = None
            risk_manager = None
            exchange_factory = None
            data_service = None
            service_manager = None

            # Try to resolve dependencies if available
            if injector.has_service("ValidationFramework"):
                try:
                    validation_framework = injector.resolve("ValidationFramework")
                except Exception:
                    pass

            if injector.has_service("StrategyRepository"):
                try:
                    repository = injector.resolve("StrategyRepository")
                except Exception:
                    pass

            if injector.has_service("RiskService"):
                try:
                    risk_manager = injector.resolve("RiskService")
                except Exception:
                    pass

            if injector.has_service("ExchangeFactory"):
                try:
                    exchange_factory = injector.resolve("ExchangeFactory")
                except Exception:
                    pass

            if injector.has_service("DataService"):
                try:
                    data_service = injector.resolve("DataService")
                except Exception:
                    pass

            if injector.has_service("ServiceManager"):
                try:
                    service_manager = injector.resolve("ServiceManager")
                except Exception:
                    pass

            return StrategyFactory(
                validation_framework=validation_framework,
                repository=repository,
                risk_manager=risk_manager,
                exchange_factory=exchange_factory,
                data_service=data_service,
                service_manager=service_manager,
            )

        # Register using DependencyInjector.register_factory() API
        injector.register_factory("StrategyFactory", strategy_factory_factory, singleton=True)

        # Register factory for enhanced dynamic strategy factory
        def dynamic_strategy_factory_factory() -> DynamicStrategyFactory:
            """Factory function for DynamicStrategyFactory with injected dependencies."""
            # Dependencies will be resolved by DI container when available
            service_container = None
            technical_indicators = None
            strategy_service = None
            regime_detector = None
            adaptive_risk_manager = None

            # Try to resolve dependencies if available
            if injector.has_service("StrategyServiceContainer"):
                try:
                    service_container = injector.resolve("StrategyServiceContainer")
                except Exception:
                    pass

            if injector.has_service("TechnicalIndicators"):
                try:
                    technical_indicators = injector.resolve("TechnicalIndicators")
                except Exception:
                    pass

            if injector.has_service("StrategyService"):
                try:
                    strategy_service = injector.resolve("StrategyService")
                except Exception:
                    pass

            if injector.has_service("RegimeDetector"):
                try:
                    regime_detector = injector.resolve("RegimeDetector")
                except Exception:
                    pass

            if injector.has_service("AdaptiveRiskManager"):
                try:
                    adaptive_risk_manager = injector.resolve("AdaptiveRiskManager")
                except Exception:
                    pass

            return DynamicStrategyFactory(
                service_container=service_container,
                technical_indicators=technical_indicators,
                strategy_service=strategy_service,
                regime_detector=regime_detector,
                adaptive_risk_manager=adaptive_risk_manager,
            )

        # Register using DependencyInjector.register_factory() API
        injector.register_factory("DynamicStrategyFactory", dynamic_strategy_factory_factory, singleton=True)

        # Register the interface types for lookup using register_factory
        # These are class registrations, not instance registrations
        injector.register_factory(
            "StrategyServiceInterface",
            lambda: StrategyService(name="StrategyService", config={}),
            singleton=True,
        )

        injector.register_factory(
            "StrategyRepositoryInterface",
            lambda: StrategyRepository(session=None),
            singleton=False,
        )

        injector.register_factory(
            "StrategyFactoryInterface",
            lambda: StrategyFactory(),
            singleton=True,
        )

        logger.info("Strategies module dependencies registered successfully")

    except Exception as e:
        logger.error(f"Failed to register strategies dependencies: {e}")
        raise
