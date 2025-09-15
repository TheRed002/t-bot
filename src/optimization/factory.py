"""
Factory for creating optimization components with proper dependency injection.

This module provides factory functions for creating optimization components
following the established service layer patterns and dependency injection.
"""

from typing import TYPE_CHECKING, Any

from src.core.base.factory import BaseFactory
from src.core.dependency_injection import DependencyInjector
from src.core.exceptions import FactoryError
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.optimization.controller import OptimizationController
    from src.optimization.interfaces import (
        IAnalysisService,
        IBacktestIntegrationService,
        IOptimizationService,
        OptimizationRepositoryProtocol,
    )

logger = get_logger(__name__)


class OptimizationFactory(BaseFactory[Any]):
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
        # Import interfaces at runtime to avoid circular imports

        super().__init__(
            product_type=object,
            name="OptimizationFactory",
            correlation_id=correlation_id,
        )

        if injector:
            self.configure_dependencies(injector)

        # Disable product validation since we create multiple types
        self.configure_validation(validate_creators=True, validate_products=False)

        # Register default creators
        self.register("service", self._create_optimization_service, singleton=True)
        self.register("controller", self._create_optimization_controller, singleton=True)
        self.register("repository", self._create_optimization_repository, singleton=True)
        self.register("backtest_integration", self._create_backtest_integration, singleton=False)
        self.register("analysis_service", self._create_analysis_service, singleton=True)

        logger.info("OptimizationFactory initialized")

    def _create_optimization_repository(
        self, database_session: Any = None, **kwargs
    ) -> "OptimizationRepositoryProtocol":
        """
        Create optimization repository with dependency injection.

        Args:
            database_session: Database session instance (optional)

        Returns:
            Configured OptimizationRepository instance
        """
        try:
            from src.optimization.repository import OptimizationRepository

            # Use dependency injection for database session
            if database_session is None and self._dependency_container:
                try:
                    database_session = self._dependency_container.resolve("AsyncSession")
                except Exception as e:
                    logger.debug(f"Failed to resolve AsyncSession: {e}")
                    try:
                        database_session = self._dependency_container.resolve("DatabaseSession")
                    except Exception as e2:
                        logger.debug(f"Failed to resolve DatabaseSession: {e2}")
                        pass  # Optional dependency

            repository = OptimizationRepository(database_session)
            logger.info("OptimizationRepository created via factory")

            return repository

        except Exception as e:
            logger.error(f"Failed to create OptimizationRepository: {e}")
            raise FactoryError(
                f"Factory error creating repository: {e}",
                error_code="FACT_003",
                component="OptimizationRepository",
            ) from e

    def _create_backtest_integration(
        self, backtest_service: Any = None, **kwargs
    ) -> "IBacktestIntegrationService":
        """
        Create backtesting integration service with dependency injection.

        Args:
            backtest_service: Backtest service instance (optional)

        Returns:
            Configured BacktestIntegrationService instance
        """
        try:
            from src.optimization.backtesting_integration import BacktestIntegrationService

            # Use dependency injection for backtest service
            if backtest_service is None and self._dependency_container:
                try:
                    backtest_service = self._dependency_container.resolve("BacktestService")
                except Exception as e:
                    logger.debug(f"Failed to resolve BacktestService: {e}")
                    pass  # Optional dependency

            integration = BacktestIntegrationService(backtest_service)
            logger.info("BacktestIntegrationService created via factory")

            return integration

        except Exception as e:
            logger.error(f"Failed to create BacktestIntegrationService: {e}")
            raise FactoryError(
                f"Factory error creating backtest integration: {e}",
                error_code="FACT_004",
                component="BacktestIntegrationService",
            ) from e

    def _create_analysis_service(
        self, results_analyzer: Any = None, **kwargs
    ) -> "IAnalysisService":
        """
        Create analysis service with dependency injection.

        Args:
            results_analyzer: Results analyzer instance (optional)

        Returns:
            Configured AnalysisService instance
        """
        try:
            from src.optimization.analysis_service import AnalysisService

            # Use dependency injection for results analyzer
            if results_analyzer is None and self._dependency_container:
                try:
                    results_analyzer = self._dependency_container.resolve("ResultsAnalyzer")
                except Exception as e:
                    logger.debug(f"ResultsAnalyzer not available: {e}")
                    pass  # Optional dependency

            service = AnalysisService(results_analyzer=results_analyzer)
            logger.info("AnalysisService created via factory")

            return service

        except Exception as e:
            logger.error(f"Failed to create AnalysisService: {e}")
            raise FactoryError(
                f"Factory error creating analysis service: {e}",
                error_code="FACT_007",
                component="AnalysisService",
            ) from e

    def _create_optimization_service(
        self,
        backtest_integration: Any = None,
        optimization_repository: Any = None,
        analysis_service: Any = None,
        websocket_manager: Any = None,
        **kwargs,
    ) -> "IOptimizationService":
        """
        Create optimization service with all required dependencies.

        Args:
            backtest_integration: Backtest integration service (optional)
            optimization_repository: Optimization repository (optional)
            analysis_service: Analysis service (optional)
            websocket_manager: WebSocket manager (optional)

        Returns:
            Configured OptimizationService instance
        """
        try:
            from src.optimization.service import OptimizationService

            # Use dependency injection for service dependencies
            if self._dependency_container:
                if backtest_integration is None:
                    try:
                        backtest_integration = self._dependency_container.resolve(
                            "OptimizationBacktestIntegration"
                        )
                    except Exception as e:
                        logger.debug(f"Failed to resolve OptimizationBacktestIntegration: {e}")

                if optimization_repository is None:
                    try:
                        optimization_repository = self._dependency_container.resolve(
                            "OptimizationRepository"
                        )
                    except Exception as e:
                        logger.debug(f"OptimizationRepository not available in DI container: {e}")

                if analysis_service is None:
                    try:
                        analysis_service = self._dependency_container.resolve(
                            "OptimizationAnalysisService"
                        )
                    except Exception as e:
                        logger.debug(f"Failed to resolve OptimizationAnalysisService: {e}")

                if websocket_manager is None:
                    try:
                        websocket_manager = self._dependency_container.resolve("WebSocketManager")
                    except Exception as e:
                        logger.debug(f"WebSocketManager not available: {e}")

            # Create service with resolved dependencies
            service = OptimizationService(
                backtest_integration=backtest_integration,
                optimization_repository=optimization_repository,
                analysis_service=analysis_service,
                websocket_manager=websocket_manager,
            )
            logger.info("OptimizationService created via factory")

            return service

        except Exception as e:
            logger.error(f"Failed to create OptimizationService: {e}")
            raise FactoryError(
                f"Factory error creating service: {e}",
                error_code="FACT_005",
                component="OptimizationService",
            ) from e

    def _create_optimization_controller(
        self, optimization_service: Any = None, **kwargs
    ) -> "OptimizationController":
        """
        Create optimization controller with proper service dependency.

        Args:
            optimization_service: Optimization service instance (optional)

        Returns:
            Configured OptimizationController instance
        """
        try:
            from src.optimization.controller import OptimizationController

            # Use dependency injection for service dependency
            if optimization_service is None and self._dependency_container:
                try:
                    optimization_service = self._dependency_container.resolve("OptimizationService")
                except Exception as e:
                    logger.debug("OptimizationService not available in DI container: %s", str(e))

            if not optimization_service:
                optimization_service = self._create_optimization_service()

            controller = OptimizationController(optimization_service=optimization_service)
            logger.info("OptimizationController created via factory")

            return controller

        except Exception as e:
            logger.error(f"Failed to create OptimizationController: {e}")
            raise FactoryError(
                f"Factory error creating controller: {e}",
                error_code="FACT_006",
                component="OptimizationController",
            ) from e

    def create_complete_optimization_stack(self) -> dict[str, Any]:
        """
        Create complete optimization stack with all components.

        Returns:
            Dictionary containing all optimization components
        """
        try:
            return {
                "service": self.create("service"),
                "controller": self.create("controller"),
                "repository": self.create("repository"),
                "backtest_integration": self.create("backtest_integration"),
                "analysis_service": self.create("analysis_service"),
            }
        except Exception as e:
            logger.error(f"Failed to create complete optimization stack: {e}")
            raise FactoryError(
                f"Stack creation failed: {e}", error_code="FACT_008", component="OptimizationStack"
            ) from e


class OptimizationComponentFactory:
    """
    Composite factory for all optimization components.

    Uses factory pattern with proper service locator integration.
    """

    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None):
        """Initialize component factory."""
        self._factory = OptimizationFactory(dependency_container, correlation_id)

    def create_service(self, **kwargs) -> "IOptimizationService":
        """Create optimization service using factory pattern."""
        return self._factory.create("service", **kwargs)

    def create_controller(self, **kwargs) -> "OptimizationController":
        """Create optimization controller using factory pattern."""
        return self._factory.create("controller", **kwargs)

    def create_repository(self, **kwargs) -> "OptimizationRepositoryProtocol":
        """Create optimization repository using factory pattern."""
        return self._factory.create("repository", **kwargs)

    def create_backtest_integration(self, **kwargs) -> "IBacktestIntegrationService":
        """Create backtest integration using factory pattern."""
        return self._factory.create("backtest_integration", **kwargs)

    def create_analysis_service(self, **kwargs) -> "IAnalysisService":
        """Create analysis service using factory pattern."""
        return self._factory.create("analysis_service", **kwargs)

    def register_factories(self, container: Any) -> None:
        """Register factory in service container."""
        if hasattr(container, "register_factory"):
            container.register_factory("OptimizationFactory", lambda: self._factory, singleton=True)


def create_optimization_service(
    injector: DependencyInjector | None = None,
) -> "IOptimizationService":
    """Create optimization service using factory pattern."""
    factory = OptimizationFactory(injector)
    return factory.create("service")


def create_optimization_controller(
    injector: DependencyInjector | None = None,
) -> "OptimizationController":
    """Create optimization controller using factory pattern."""
    factory = OptimizationFactory(injector)
    return factory.create("controller")


def create_optimization_stack(injector: DependencyInjector | None = None) -> dict[str, Any]:
    """Create complete optimization stack using factory pattern."""
    factory = OptimizationFactory(injector)
    return factory.create_complete_optimization_stack()
