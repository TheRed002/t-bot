"""Dependency injection registration for strategies module."""

from typing import Any

from src.core.dependency_injection import DependencyContainer
from src.core.logging import get_logger
from src.strategies.factory import StrategyFactory
from src.strategies.dynamic.strategy_factory import DynamicStrategyFactory
from src.strategies.repository import StrategyRepository
from src.strategies.service import StrategyService

logger = get_logger(__name__)


def register_strategies_dependencies(container: DependencyContainer) -> None:
    """Register all strategies module dependencies."""
    try:
        logger.info("Registering strategies module dependencies")

        # Register factory for strategy service with dependencies
        def strategy_service_factory(
            repository=None,
            risk_manager=None,
            exchange_factory=None,
            data_service=None,
            service_manager=None,
            config: dict[str, Any] | None = None,
        ) -> StrategyService:
            """Factory function for StrategyService with injected dependencies."""
            return StrategyService(
                name="StrategyService",
                config=config or {},
                repository=repository,
                risk_manager=risk_manager,
                exchange_factory=exchange_factory,
                data_service=data_service,
                service_manager=service_manager,
            )

        # Register the service factory as singleton
        container.register(
            name="StrategyService",
            service=strategy_service_factory,
            singleton=True,
        )

        # Register factory for repository with database session
        def strategy_repository_factory(session) -> StrategyRepository:
            """Factory function for StrategyRepository with database session."""
            return StrategyRepository(session=session)

        # Register the repository factory (not singleton since it needs fresh sessions)
        container.register(
            name="StrategyRepository",
            service=strategy_repository_factory,
            singleton=False,
        )

        # Register factory for strategy factory with dependencies
        def strategy_factory_factory(
            validation_framework=None,
            repository=None,
            risk_manager=None,
            exchange_factory=None,
            data_service=None,
            service_manager=None,
        ) -> StrategyFactory:
            """Factory function for StrategyFactory with injected dependencies."""
            return StrategyFactory(
                validation_framework=validation_framework,
                repository=repository,
                risk_manager=risk_manager,
                exchange_factory=exchange_factory,
                data_service=data_service,
                service_manager=service_manager,
            )

        # Register the factory as singleton
        container.register(
            name="StrategyFactory",
            service=strategy_factory_factory,
            singleton=True,
        )

        # Register factory for enhanced dynamic strategy factory
        def dynamic_strategy_factory_factory(
            service_container=None,
            technical_indicators=None,
            strategy_service=None,
            regime_detector=None,
            adaptive_risk_manager=None,
        ) -> DynamicStrategyFactory:
            """Factory function for DynamicStrategyFactory with injected dependencies."""
            return DynamicStrategyFactory(
                service_container=service_container,
                technical_indicators=technical_indicators,
                strategy_service=strategy_service,
                regime_detector=regime_detector,
                adaptive_risk_manager=adaptive_risk_manager,
            )

        # Register the enhanced dynamic factory as singleton
        container.register(
            name="DynamicStrategyFactory",
            service=dynamic_strategy_factory_factory,
            singleton=True,
        )

        # Register the interface types for lookup
        container.register(
            name="StrategyServiceInterface",
            service=StrategyService,
            singleton=True,
        )

        container.register(
            name="StrategyRepositoryInterface",
            service=StrategyRepository,
            singleton=False,
        )

        container.register(
            name="StrategyFactoryInterface",
            service=StrategyFactory,
            singleton=True,
        )

        logger.info("Strategies module dependencies registered successfully")

    except Exception as e:
        logger.error(f"Failed to register strategies dependencies: {e}")
        raise
