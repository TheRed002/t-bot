"""Backtesting services dependency injection registration."""

from typing import TYPE_CHECKING

from src.core.dependency_injection import DependencyInjector
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.backtesting.analysis import MonteCarloAnalyzer, WalkForwardAnalyzer
    from src.backtesting.attribution import PerformanceAttributor
    from src.backtesting.controller import BacktestController
    from src.backtesting.data_replay import DataReplayManager
    from src.backtesting.factory import BacktestFactory
    from src.backtesting.metrics import MetricsCalculator
    from src.backtesting.repository import BacktestRepository
    from src.backtesting.service import BacktestService
    from src.backtesting.simulator import TradeSimulator

logger = get_logger(__name__)


def register_backtesting_services(injector: DependencyInjector) -> None:
    """
    Register backtesting services with the dependency injector.

    Args:
        injector: Dependency injector instance
    """

    # Register BacktestFactory first as it coordinates other registrations
    def factory_factory() -> "BacktestFactory":
        from src.backtesting.factory import BacktestFactory

        return BacktestFactory(injector=injector)

    injector.register_factory("BacktestFactory", factory_factory, singleton=True)

    # Register factory interface
    injector.register_factory(
        "BacktestFactoryInterface", lambda: injector.resolve("BacktestFactory"), singleton=True
    )

    # Register MetricsCalculator using factory pattern
    def metrics_calculator_factory() -> "MetricsCalculator":
        factory = injector.resolve("BacktestFactory")
        return factory.create_metrics_calculator()

    injector.register_factory("MetricsCalculator", metrics_calculator_factory, singleton=False)

    # Register MonteCarloAnalyzer using factory pattern
    def monte_carlo_analyzer_factory() -> "MonteCarloAnalyzer":
        factory = injector.resolve("BacktestFactory")
        return factory.create_analyzer("monte_carlo")

    injector.register_factory("MonteCarloAnalyzer", monte_carlo_analyzer_factory, singleton=False)

    # Register WalkForwardAnalyzer using factory pattern
    def walk_forward_analyzer_factory() -> "WalkForwardAnalyzer":
        factory = injector.resolve("BacktestFactory")
        return factory.create_analyzer("walk_forward")

    injector.register_factory("WalkForwardAnalyzer", walk_forward_analyzer_factory, singleton=False)

    # Register PerformanceAttributor using factory pattern
    def performance_attributor_factory() -> "PerformanceAttributor":
        factory = injector.resolve("BacktestFactory")
        return factory.create_analyzer("performance_attribution")

    injector.register_factory(
        "PerformanceAttributor", performance_attributor_factory, singleton=False
    )

    # Register DataReplayManager factory
    def data_replay_manager_factory() -> "DataReplayManager":
        from src.backtesting.data_replay import DataReplayManager

        config = injector.resolve("Config")
        return DataReplayManager(config=config)

    injector.register_factory("DataReplayManager", data_replay_manager_factory, singleton=False)

    # Register TradeSimulator using factory pattern
    def trade_simulator_factory() -> "TradeSimulator":
        from src.backtesting.simulator import SimulationConfig

        factory = injector.resolve("BacktestFactory")
        sim_config = SimulationConfig()
        return factory.create_simulator(sim_config)

    injector.register_factory("TradeSimulator", trade_simulator_factory, singleton=False)

    # Register BacktestEngine factory using factory pattern
    def backtest_engine_factory():
        """Factory function that returns a BacktestEngine creator with dependency injection."""

        def create_engine(config, strategy, **kwargs):
            factory = injector.resolve("BacktestFactory")
            return factory.create_engine(config=config, strategy=strategy, **kwargs)

        return create_engine

    injector.register_factory("BacktestEngineFactory", backtest_engine_factory, singleton=True)

    # Register BacktestRepository using factory pattern
    def repository_factory() -> "BacktestRepository":
        factory = injector.resolve("BacktestFactory")
        return factory.create_repository()

    injector.register_factory("BacktestRepository", repository_factory, singleton=True)

    # Register interface implementation
    injector.register_factory(
        "BacktestRepositoryInterface",
        lambda: injector.resolve("BacktestRepository"),
        singleton=True,
    )

    # Register main BacktestService using factory pattern
    def backtest_service_factory() -> "BacktestService":
        factory = injector.resolve("BacktestFactory")
        return factory.create_service()

    injector.register_factory("BacktestService", backtest_service_factory, singleton=True)

    # Register BacktestController using factory pattern
    def controller_factory() -> "BacktestController":
        factory = injector.resolve("BacktestFactory")
        return factory.create_controller()

    injector.register_factory("BacktestController", controller_factory, singleton=True)

    # Register interface implementations for proper service layer architecture
    injector.register_factory(
        "BacktestServiceInterface", lambda: injector.resolve("BacktestService"), singleton=True
    )
    injector.register_factory(
        "BacktestControllerInterface",
        lambda: injector.resolve("BacktestController"),
        singleton=True,
    )

    # Register factory interfaces
    injector.register_factory(
        "BacktestEngineFactoryInterface",
        lambda: injector.resolve("BacktestEngineFactory"),
        singleton=True,
    )
    injector.register_factory(
        "TradeSimulatorInterface", lambda: injector.resolve("TradeSimulator"), singleton=False
    )

    logger.info("Backtesting services registered with dependency injector")


def configure_backtesting_dependencies(
    injector: DependencyInjector | None = None,
) -> DependencyInjector:
    """
    Configure backtesting dependencies with proper service lifetimes.

    Args:
        injector: Optional existing injector instance

    Returns:
        Configured dependency injector
    """
    if injector is None:
        injector = DependencyInjector()

    register_backtesting_services(injector)

    return injector


# Service locator convenience function
def get_backtest_service(injector: DependencyInjector) -> "BacktestService":
    """Get BacktestService from DI container using service locator pattern."""
    return injector.resolve("BacktestService")
