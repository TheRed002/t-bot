"""
Factory for creating optimization components with proper dependency injection.

This module provides factory functions for creating optimization components
following the established service layer patterns and dependency injection.
"""

from typing import TYPE_CHECKING, Any

from src.core.base.factory import BaseFactory
from src.core.dependency_injection import DependencyInjector
from src.core.exceptions import CreationError
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.optimization.controller import OptimizationController
    from src.optimization.interfaces import (
        BacktestIntegrationProtocol,
        IOptimizationService,
        OptimizationRepositoryProtocol,
    )

logger = get_logger(__name__)


class OptimizationFactory(BaseFactory["IOptimizationService"]):
    """
    Factory for creating optimization components.

    Provides centralized creation of optimization services, repositories,
    and controllers with proper dependency injection.
    """

    def __init__(
        self, injector: DependencyInjector | None = None, correlation_id: str | None = None
    ):
        """
        Initialize optimization factory.

        Args:
            injector: Dependency injector for service resolution
            correlation_id: Request correlation ID
        """
        # Import at runtime to avoid circular imports
        from src.optimization.interfaces import IOptimizationService

        super().__init__(
            product_type=IOptimizationService,
            name="OptimizationFactory",
            correlation_id=correlation_id,
        )

        if injector:
            self.configure_dependencies(injector)

        # Register default creators
        self.register("service", self._create_optimization_service, singleton=True)
        self.register("controller", self._create_optimization_controller, singleton=True)
        self.register("repository", self._create_optimization_repository, singleton=False)
        self.register("backtest_integration", self._create_backtest_integration, singleton=False)

        logger.info("OptimizationFactory initialized")

    def _create_optimization_repository(self, **kwargs) -> "OptimizationRepositoryProtocol":
        """
        Create optimization repository with dependency injection.

        Returns:
            Configured OptimizationRepository instance
        """
        try:
            from src.optimization.repository import OptimizationRepository

            # Resolve database dependency
            database_session = None
            if self._dependency_container:
                try:
                    database_session = self._dependency_container.resolve("AsyncSession")
                except Exception:
                    try:
                        database_session = self._dependency_container.resolve("DatabaseSession")
                    except Exception:
                        pass  # Optional dependency

            if not database_session:
                logger.warning("No database session available for optimization repository")

            repository = OptimizationRepository(database_session)
            logger.info("OptimizationRepository created via factory")

            return repository

        except Exception as e:
            logger.error(f"Failed to create OptimizationRepository: {e}")
            raise CreationError(f"Factory error creating repository: {e}") from e

    def _create_backtest_integration(self, **kwargs) -> "BacktestIntegrationProtocol":
        """
        Create backtesting integration service with dependency injection.

        Returns:
            Configured BacktestIntegrationService instance
        """
        try:
            from src.optimization.backtesting_integration import BacktestIntegrationService

            # Resolve backtest service dependency
            backtest_service = None
            if self._dependency_container:
                try:
                    backtest_service = self._dependency_container.resolve("BacktestService")
                except Exception:
                    pass  # Optional dependency

            integration = BacktestIntegrationService(backtest_service)
            logger.info("BacktestIntegrationService created via factory")

            return integration

        except Exception as e:
            logger.error(f"Failed to create BacktestIntegrationService: {e}")
            raise CreationError(f"Factory error creating backtest integration: {e}") from e

    def _create_optimization_service(self, **kwargs) -> "IOptimizationService":
        """
        Create optimization service with all required dependencies.

        Returns:
            Configured OptimizationService instance
        """
        try:
            from src.optimization.service import OptimizationService

            # Resolve dependencies
            backtest_integration = None
            repository = None

            if self._dependency_container:
                try:
                    backtest_integration = self._dependency_container.resolve("BacktestIntegration")
                except Exception:
                    # Create via factory method
                    backtest_integration = self._create_backtest_integration()

                try:
                    repository = self._dependency_container.resolve("OptimizationRepository")
                except Exception:
                    # Repository is optional
                    pass

            if not backtest_integration:
                backtest_integration = self._create_backtest_integration()

            service = OptimizationService(
                backtest_integration=backtest_integration,
                optimization_repository=repository,
            )
            logger.info("OptimizationService created via factory")

            return service

        except Exception as e:
            logger.error(f"Failed to create OptimizationService: {e}")
            raise CreationError(f"Factory error creating service: {e}") from e

    def _create_optimization_controller(self, **kwargs) -> "OptimizationController":
        """
        Create optimization controller with proper service dependency.

        Returns:
            Configured OptimizationController instance
        """
        try:
            from src.optimization.controller import OptimizationController

            # Resolve service dependency
            optimization_service = None
            if self._dependency_container:
                try:
                    optimization_service = self._dependency_container.resolve("OptimizationService")
                except Exception:
                    pass

            if not optimization_service:
                optimization_service = self._create_optimization_service()

            controller = OptimizationController(optimization_service=optimization_service)
            logger.info("OptimizationController created via factory")

            return controller

        except Exception as e:
            logger.error(f"Failed to create OptimizationController: {e}")
            raise CreationError(f"Factory error creating controller: {e}") from e

    def create_complete_optimization_stack(self) -> dict[str, Any]:
        """
        Create complete optimization stack with all components.

        Returns:
            Dictionary containing all optimization components
        """
        return {
            "service": self.create("service"),
            "controller": self.create("controller"),
            "repository": self.create("repository"),
            "backtest_integration": self.create("backtest_integration"),
        }


class OptimizationComponentFactory:
    """
    Composite factory for all optimization components.

    Provides a single entry point for creating all optimization
    components with proper dependency resolution.
    """

    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None):
        """
        Initialize optimization component factory.

        Args:
            dependency_container: Dependency injection container
            correlation_id: Request correlation ID
        """
        self.dependency_container = dependency_container
        self.correlation_id = correlation_id

        # Initialize main factory
        self.optimization_factory = OptimizationFactory(dependency_container, correlation_id)

    def create_service(self, **kwargs) -> "IOptimizationService":
        """Create OptimizationService instance."""
        return self.optimization_factory.create("service", **kwargs)

    def create_controller(self, **kwargs) -> "OptimizationController":
        """Create OptimizationController instance."""
        return self.optimization_factory.create("controller", **kwargs)

    def create_repository(self, **kwargs) -> "OptimizationRepositoryProtocol":
        """Create OptimizationRepository instance."""
        return self.optimization_factory.create("repository", **kwargs)

    def create_backtest_integration(self, **kwargs) -> "BacktestIntegrationProtocol":
        """Create BacktestIntegrationService instance."""
        return self.optimization_factory.create("backtest_integration", **kwargs)

    def register_factories(self, container: Any) -> None:
        """
        Register all factories with the dependency injection container.

        Args:
            container: Dependency injection container
        """
        container.register(
            "OptimizationFactory", lambda: self.optimization_factory, singleton=True
        )
        container.register("OptimizationComponentFactory", lambda: self, singleton=True)


def create_optimization_service(
    injector: DependencyInjector | None = None,
) -> "IOptimizationService":
    """
    Convenience function to create optimization service using service locator pattern.

    Args:
        injector: Optional dependency injector

    Returns:
        Configured optimization service
    """
    if injector and injector.has_service("OptimizationService"):
        return injector.resolve("OptimizationService")

    factory = OptimizationFactory(injector)
    return factory.create("service")


def create_optimization_controller(
    injector: DependencyInjector | None = None,
) -> "OptimizationController":
    """
    Convenience function to create optimization controller using service locator pattern.

    Args:
        injector: Optional dependency injector

    Returns:
        Configured optimization controller
    """
    if injector and injector.has_service("OptimizationController"):
        return injector.resolve("OptimizationController")

    factory = OptimizationFactory(injector)
    return factory.create("controller")


def create_optimization_stack(injector: DependencyInjector | None = None) -> dict[str, Any]:
    """
    Convenience function to create complete optimization stack using service locator pattern.

    Args:
        injector: Optional dependency injector

    Returns:
        Dictionary containing all optimization components
    """
    if injector:
        # Use DI container if available
        service = (
            injector.resolve("OptimizationService")
            if injector.has_service("OptimizationService")
            else None
        )
        controller = (
            injector.resolve("OptimizationController")
            if injector.has_service("OptimizationController")
            else None
        )
        repository = (
            injector.resolve("OptimizationRepository")
            if injector.has_service("OptimizationRepository")
            else None
        )
        backtest_integration = (
            injector.resolve("BacktestIntegration")
            if injector.has_service("BacktestIntegration")
            else None
        )

        return {
            "service": service,
            "controller": controller,
            "repository": repository,
            "backtest_integration": backtest_integration,
        }

    factory = OptimizationFactory(injector)
    return factory.create_complete_optimization_stack()
