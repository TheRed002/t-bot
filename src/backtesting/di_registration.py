"""Backtesting services dependency injection registration."""

from typing import TYPE_CHECKING

from src.core.dependency_injection import DependencyInjector
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.backtesting.analysis import MonteCarloAnalyzer, WalkForwardAnalyzer
    from src.backtesting.attribution import PerformanceAttributor
    from src.backtesting.data_replay import DataReplayManager
    from src.backtesting.metrics import MetricsCalculator
    from src.backtesting.service import BacktestService
    from src.backtesting.simulator import TradeSimulator

logger = get_logger(__name__)


def register_backtesting_services(injector: DependencyInjector) -> None:
    """
    Register backtesting services with the dependency injector.

    Args:
        injector: Dependency injector instance
    """

    # Register MetricsCalculator as singleton
    def metrics_calculator_factory() -> "MetricsCalculator":
        from src.backtesting.metrics import MetricsCalculator

        return MetricsCalculator()

    injector.register_factory("MetricsCalculator", metrics_calculator_factory, singleton=True)

    # Register MonteCarloAnalyzer factory
    def monte_carlo_analyzer_factory() -> "MonteCarloAnalyzer":
        from src.backtesting.analysis import MonteCarloAnalyzer

        config = injector.resolve("Config")
        engine_factory = injector.resolve("BacktestEngineFactory")
        return MonteCarloAnalyzer(config=config, engine_factory=engine_factory)

    injector.register_factory("MonteCarloAnalyzer", monte_carlo_analyzer_factory, singleton=False)

    # Register WalkForwardAnalyzer factory
    def walk_forward_analyzer_factory() -> "WalkForwardAnalyzer":
        from src.backtesting.analysis import WalkForwardAnalyzer

        config = injector.resolve("Config")
        engine_factory = injector.resolve("BacktestEngineFactory")
        return WalkForwardAnalyzer(config=config, engine_factory=engine_factory)

    injector.register_factory("WalkForwardAnalyzer", walk_forward_analyzer_factory, singleton=False)

    # Register PerformanceAttributor factory
    def performance_attributor_factory() -> "PerformanceAttributor":
        from src.backtesting.attribution import PerformanceAttributor

        config = injector.resolve("Config")
        return PerformanceAttributor(config=config)

    injector.register_factory("PerformanceAttributor", performance_attributor_factory, singleton=False)

    # Register DataReplayManager factory
    def data_replay_manager_factory() -> "DataReplayManager":
        from src.backtesting.data_replay import DataReplayManager

        config = injector.resolve("Config")
        return DataReplayManager(config=config)

    injector.register_factory("DataReplayManager", data_replay_manager_factory, singleton=False)

    # Register TradeSimulator factory with dependency injection
    def trade_simulator_factory() -> "TradeSimulator":
        from src.backtesting.simulator import SimulationConfig, TradeSimulator

        sim_config = SimulationConfig()
        return TradeSimulator(sim_config)

    injector.register_factory("TradeSimulator", trade_simulator_factory, singleton=False)

    # Register BacktestEngine factory using interface pattern
    def backtest_engine_factory():
        """Factory function that returns a BacktestEngine creator with dependency injection."""

        def create_engine(config, strategy, **kwargs):
            from src.backtesting.engine import BacktestEngine

            # Resolve dependencies from injector
            metrics_calculator = injector.resolve("MetricsCalculator")

            # Optionally resolve other services
            try:
                data_service = injector.resolve("DataService")
            except Exception:
                data_service = kwargs.get("data_service")

            try:
                execution_service = injector.resolve("ExecutionService")
            except Exception:
                execution_service = kwargs.get("execution_engine_service")

            return BacktestEngine(
                config=config,
                strategy=strategy,
                metrics_calculator=metrics_calculator,
                data_service=data_service,
                execution_engine_service=execution_service,
                **{k: v for k, v in kwargs.items() if k not in ["data_service", "execution_engine_service"]},
            )

        return create_engine

    injector.register_factory("BacktestEngineFactory", backtest_engine_factory, singleton=True)

    # Register main BacktestService using dependency injection
    def backtest_service_factory() -> "BacktestService":
        from src.backtesting.service import BacktestService

        config = injector.resolve("Config")

        # Validate injector is available
        if injector is None:
            raise ValueError("BacktestService requires dependency injector")

        # Resolve service dependencies - these may be None if services aren't registered
        services = {"injector": injector}
        optional_services = [
            "DataService",
            "ExecutionService",
            "RiskService",
            "StrategyService",
            "CapitalService",
            "MLService",
        ]

        for service_name in optional_services:
            try:
                services[service_name] = injector.resolve(service_name)
            except Exception:
                services[service_name] = None

        return BacktestService(config=config, **services)

    injector.register_factory("BacktestService", backtest_service_factory, singleton=True)

    # Register interface implementations
    injector.register_service("BacktestServiceInterface", lambda: injector.resolve("BacktestService"), singleton=True)

    # Register factory interfaces
    injector.register_service(
        "BacktestEngineFactoryInterface", lambda: injector.resolve("BacktestEngineFactory"), singleton=True
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
