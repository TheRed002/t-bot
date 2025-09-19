"""
Dependency injection registration for optimization module.

This module handles registration of optimization components with the
dependency injection container following established patterns.
"""

from typing import TYPE_CHECKING, Any

from src.core.dependency_injection import DependencyInjector
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.optimization.factory import OptimizationFactory
    from src.optimization.interfaces import (
        IAnalysisService,
        IBacktestIntegrationService,
        IOptimizationService,
        OptimizationRepositoryProtocol,
    )

logger = get_logger(__name__)


def register_optimization_dependencies(injector: DependencyInjector) -> None:
    """
    Register optimization dependencies with the DI container following proper patterns.

    Args:
        injector: Dependency injector instance
    """

    # Register repository factory first (critical for data layer integration)
    def optimization_repository_factory() -> "OptimizationRepositoryProtocol":
        """Factory for OptimizationRepository with proper session injection."""
        from src.optimization.repository import OptimizationRepository

        session = injector.resolve("AsyncSession")
        return OptimizationRepository(session)

    injector.register_factory(
        "OptimizationRepository", optimization_repository_factory, singleton=True
    )

    # Register results analyzer factory first (dependency for analysis service)
    def results_analyzer_factory():
        """Factory for ResultsAnalyzer component."""
        from src.optimization.analysis import ResultsAnalyzer

        return ResultsAnalyzer()

    injector.register_factory("ResultsAnalyzer", results_analyzer_factory, singleton=True)

    # Register parameter space service factory
    def parameter_space_service_factory():
        """Factory for ParameterSpaceService."""
        from src.optimization.parameter_space_service import ParameterSpaceService
        return ParameterSpaceService()

    injector.register_factory("ParameterSpaceService", parameter_space_service_factory, singleton=True)

    # Register result transformation service factory
    def result_transformation_service_factory():
        """Factory for ResultTransformationService."""
        from src.optimization.result_transformation_service import ResultTransformationService
        return ResultTransformationService()

    injector.register_factory("ResultTransformationService", result_transformation_service_factory, singleton=True)

    # WebSocket manager uses global instance from core module

    # Register analysis service factory
    def analysis_service_factory() -> "IAnalysisService":
        """Factory for AnalysisService with proper dependency injection."""
        from src.optimization.analysis_service import AnalysisService

        results_analyzer = injector.resolve("ResultsAnalyzer")
        return AnalysisService(results_analyzer=results_analyzer)

    injector.register_factory(
        "OptimizationAnalysisService", analysis_service_factory, singleton=True
    )

    # Register backtesting integration factory
    def backtest_integration_factory() -> "IBacktestIntegrationService":
        """Factory for BacktestIntegrationService with proper dependency injection."""
        from src.optimization.backtesting_integration import BacktestIntegrationService

        # Try to resolve BacktestService dependency
        backtest_service = None
        try:
            backtest_service = injector.resolve("BacktestService")
        except Exception as e:
            logger.debug(f"BacktestService not available: {e}")

        return BacktestIntegrationService(backtest_service)

    injector.register_factory(
        "OptimizationBacktestIntegration", backtest_integration_factory, singleton=True
    )

    # Register main factory first
    def optimization_factory_factory() -> "OptimizationFactory":
        """Factory for OptimizationFactory with DI integration."""
        from src.optimization.factory import OptimizationFactory

        return OptimizationFactory(injector)

    injector.register_factory("OptimizationFactory", optimization_factory_factory, singleton=True)

    # Register component factory
    def component_factory_factory():
        """Factory for OptimizationComponentFactory with DI integration."""
        from src.optimization.factory import OptimizationComponentFactory

        return OptimizationComponentFactory(injector)

    injector.register_factory(
        "OptimizationComponentFactory", component_factory_factory, singleton=True
    )

    # Register services through factory pattern
    def optimization_service_factory() -> "IOptimizationService":
        """Factory for OptimizationService with proper dependency injection."""
        from src.optimization.service import OptimizationService

        # Resolve dependencies
        backtest_integration = None
        try:
            backtest_integration = injector.resolve("OptimizationBacktestIntegration")
        except Exception:
            logger.debug("BacktestIntegration not available")

        optimization_repository = None
        try:
            optimization_repository = injector.resolve("OptimizationRepository")
        except Exception:
            logger.debug("OptimizationRepository not available")

        analysis_service = None
        try:
            analysis_service = injector.resolve("OptimizationAnalysisService")
        except Exception:
            logger.debug("AnalysisService not available")

        parameter_space_service = injector.resolve("ParameterSpaceService")
        result_transformation_service = injector.resolve("ResultTransformationService")

        return OptimizationService(
            backtest_integration=backtest_integration,
            optimization_repository=optimization_repository,
            analysis_service=analysis_service,
            parameter_space_service=parameter_space_service,
            result_transformation_service=result_transformation_service,
        )

    injector.register_factory("OptimizationService", optimization_service_factory, singleton=True)

    # Register controller using factory pattern
    def optimization_controller_factory():
        """Factory for OptimizationController with proper dependency injection."""
        from src.optimization.controller import OptimizationController

        # Resolve optimization service dependency
        optimization_service = injector.resolve("OptimizationService")
        return OptimizationController(optimization_service=optimization_service)

    injector.register_factory(
        "OptimizationController", optimization_controller_factory, singleton=True
    )

    # Register interface implementations for proper service layer architecture
    injector.register_factory(
        "IOptimizationService", lambda: injector.resolve("OptimizationService"), singleton=True
    )
    injector.register_factory(
        "IBacktestIntegrationService",
        lambda: injector.resolve("OptimizationBacktestIntegration"),
        singleton=True,
    )
    injector.register_factory(
        "OptimizationRepositoryProtocol",
        lambda: injector.resolve("OptimizationRepository"),
        singleton=True,
    )
    injector.register_factory(
        "IAnalysisService", lambda: injector.resolve("OptimizationAnalysisService"), singleton=True
    )

    logger.info("Optimization dependencies registered with service container")


def configure_optimization_module(
    injector: DependencyInjector,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Configure optimization module with custom settings.

    Args:
        injector: Dependency injector instance
        config: Optional configuration overrides
    """
    # Register base dependencies
    register_optimization_dependencies(injector)

    # Apply any custom configuration
    if config:
        # Handle any module-specific configuration when needed
        logger.debug("Custom optimization configuration applied: %s", config)


def get_optimization_service(injector: DependencyInjector) -> "IOptimizationService":
    """Get OptimizationService from DI container using service locator pattern."""
    return injector.resolve("OptimizationService")


def get_optimization_controller(injector: DependencyInjector):
    """Get OptimizationController from DI container using service locator pattern."""
    return injector.resolve("OptimizationController")


def get_optimization_repository(injector: DependencyInjector):
    """Get OptimizationRepository from DI container using service locator pattern."""
    return injector.resolve("OptimizationRepository")
