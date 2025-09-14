"""
Backtesting Factory - Component creation and dependency injection.

This factory handles creation of backtesting components with proper
dependency injection and service layer wiring.
"""

from typing import TYPE_CHECKING, Any

from src.core.base.component import BaseComponent
from src.core.dependency_injection import DependencyInjector
from src.core.exceptions import FactoryError
from src.core.logging import get_logger
from src.utils.initialization_helpers import log_factory_initialization

if TYPE_CHECKING:
    from src.backtesting.controller import BacktestController
    from src.backtesting.engine import BacktestEngine
    from src.backtesting.interfaces import BacktestServiceInterface
    from src.backtesting.metrics import MetricsCalculator
    from src.backtesting.repository import BacktestRepository
    from src.backtesting.service import BacktestService
    from src.backtesting.simulator import TradeSimulator

logger = get_logger(__name__)


class BacktestFactory(BaseComponent):
    """
    Factory for creating backtesting components with dependency injection.

    This factory ensures proper service layer architecture by:
    - Creating components with required dependencies
    - Wiring service interfaces correctly
    - Managing component lifecycles
    - Following interface patterns
    - Using service locator pattern
    """

    def __init__(
        self, injector: DependencyInjector | None = None, correlation_id: str | None = None
    ):
        """
        Initialize factory with dependency injector.

        Args:
            injector: Dependency injector for service resolution
            correlation_id: Request correlation ID
        """
        super().__init__(name="BacktestFactory", correlation_id=correlation_id)
        self._injector = injector
        log_factory_initialization(
            "BacktestFactory", injection_enabled=True, logger_instance=logger
        )

    def get_injector(self) -> DependencyInjector:
        """Get dependency injector instance."""
        if self._injector is None:
            from src.core.dependency_injection import get_global_injector

            self._injector = get_global_injector()
        return self._injector

    def create_controller(self) -> "BacktestController":
        """
        Create BacktestController with proper service dependency.

        Returns:
            Configured BacktestController instance
        """
        try:
            from src.backtesting.controller import BacktestController

            # Use dependency injection to resolve service
            injector = self.get_injector()
            backtest_service = injector.resolve("BacktestServiceInterface")

            controller = BacktestController(backtest_service=backtest_service)
            logger.info("BacktestController created via factory")

            return controller

        except Exception as e:
            logger.error(f"Failed to create BacktestController: {e}")
            raise FactoryError(
                f"Factory error creating controller: {e}", error_code="FACTORY_001"
            ) from e

    def create_service(self, config: Any = None) -> "BacktestService":
        """
        Create BacktestService with all required dependencies.

        Args:
            config: Configuration object

        Returns:
            Configured BacktestService instance
        """
        try:
            from src.backtesting.service import BacktestService

            # Use dependency injection to resolve dependencies
            injector = self.get_injector()

            # Get config from injector if not provided
            if config is None:
                config = injector.resolve("Config")

            # Resolve service dependencies via injector
            services = {"injector": injector}
            optional_services = [
                "DataService",
                "ExecutionService",
                "RiskService",
                "StrategyService",
                "CapitalService",
                "MLService",
                "CacheService",
            ]

            for service_name in optional_services:
                try:
                    resolved_service = injector.resolve(service_name)
                    services[service_name] = resolved_service
                except Exception:
                    # Set to None for optional services that can't be resolved
                    services[service_name] = None

            # Create repository directly to avoid circular dependency
            try:
                repository = self.create_repository()
                services["BacktestRepositoryInterface"] = repository
            except Exception:
                services["BacktestRepositoryInterface"] = None

            service = BacktestService(config=config, **services)
            logger.info("BacktestService created via factory")

            return service

        except Exception as e:
            logger.error(f"Failed to create BacktestService: {e}")
            raise FactoryError(
                f"Factory error creating service: {e}", error_code="FACTORY_002"
            ) from e

    def create_repository(self) -> "BacktestRepository":
        """
        Create BacktestRepository with database dependency.

        Returns:
            Configured BacktestRepository instance
        """
        try:
            from src.backtesting.repository import BacktestRepository

            # Use dependency injection to resolve database service
            injector = self.get_injector()
            db_manager = injector.resolve("DatabaseServiceInterface")

            repository = BacktestRepository(db_manager=db_manager)
            logger.info("BacktestRepository created via factory")

            return repository

        except Exception as e:
            logger.error(f"Failed to create BacktestRepository: {e}")
            raise FactoryError(
                f"Factory error creating repository: {e}", error_code="FACTORY_003"
            ) from e

    def create_engine(self, config: Any, strategy: Any, **kwargs) -> "BacktestEngine":
        """
        Create BacktestEngine with proper service dependencies.

        Args:
            config: Backtest configuration
            strategy: Strategy instance
            **kwargs: Additional configuration

        Returns:
            Configured BacktestEngine instance
        """
        try:
            from src.backtesting.engine import BacktestEngine

            # Use dependency injection to resolve service dependencies
            injector = self.get_injector()
            services = {}

            # Resolve services with fallback to kwargs
            service_mappings = {
                "data_service": "DataService",
                "execution_engine_service": "ExecutionService",
                "risk_manager": "RiskService",
                "metrics_calculator": "MetricsCalculator",
            }

            for param_name, service_name in service_mappings.items():
                try:
                    services[param_name] = injector.resolve(service_name)
                except Exception:
                    services[param_name] = kwargs.get(param_name)

            # Filter out None values and merge with kwargs
            engine_kwargs = {k: v for k, v in services.items() if v is not None}
            engine_kwargs.update({k: v for k, v in kwargs.items() if k not in services})

            engine = BacktestEngine(config=config, strategy=strategy, **engine_kwargs)

            logger.info("BacktestEngine created via factory")
            return engine

        except Exception as e:
            logger.error(f"Failed to create BacktestEngine: {e}")
            raise FactoryError(
                f"Factory error creating engine: {e}", error_code="FACTORY_004"
            ) from e

    def create_simulator(self, config: Any) -> "TradeSimulator":
        """
        Create TradeSimulator with configuration.

        Args:
            config: Simulation configuration

        Returns:
            Configured TradeSimulator instance
        """
        try:
            from src.backtesting.simulator import TradeSimulator

            # Use dependency injection to resolve optional dependencies
            injector = self.get_injector()
            slippage_model = None
            try:
                slippage_model = injector.resolve("SlippageModel")
            except Exception:
                # Slippage model is optional
                pass

            simulator = TradeSimulator(config=config, slippage_model=slippage_model)

            logger.info("TradeSimulator created via factory")
            return simulator

        except Exception as e:
            logger.error(f"Failed to create TradeSimulator: {e}")
            raise FactoryError(
                f"Factory error creating simulator: {e}", error_code="FACTORY_005"
            ) from e

    def create_metrics_calculator(self, risk_free_rate: float | None = None) -> "MetricsCalculator":
        """
        Create MetricsCalculator with configuration.

        Args:
            risk_free_rate: Risk-free rate for metrics calculation

        Returns:
            Configured MetricsCalculator instance
        """
        try:
            from src.backtesting.metrics import MetricsCalculator

            calculator = MetricsCalculator(risk_free_rate=risk_free_rate)
            logger.info("MetricsCalculator created via factory")

            return calculator

        except Exception as e:
            logger.error(f"Failed to create MetricsCalculator: {e}")
            raise FactoryError(
                f"Factory error creating calculator: {e}", error_code="FACTORY_006"
            ) from e

    def create_analyzer(self, analyzer_type: str, config: dict[str, Any] | None = None) -> Any:
        """
        Create analyzer component (Monte Carlo, Walk Forward, etc.).

        Args:
            analyzer_type: Type of analyzer to create
            config: Analyzer configuration

        Returns:
            Configured analyzer instance
        """
        try:
            config = config or {}
            injector = self.get_injector()

            if analyzer_type == "monte_carlo":
                from src.backtesting.analysis import MonteCarloAnalyzer

                # Use dependency injection to get engine factory
                try:
                    engine_factory = injector.resolve("BacktestEngineFactory")
                except Exception:
                    engine_factory = None

                return MonteCarloAnalyzer(
                    config=config.get("config"),
                    num_simulations=config.get("num_simulations", 1000),
                    confidence_level=config.get("confidence_level", 0.95),
                    seed=config.get("seed"),
                    engine_factory=engine_factory,
                )

            elif analyzer_type == "walk_forward":
                from src.backtesting.analysis import WalkForwardAnalyzer

                try:
                    engine_factory = injector.resolve("BacktestEngineFactory")
                except Exception:
                    engine_factory = None

                return WalkForwardAnalyzer(
                    config=config.get("config"),
                    engine_factory=engine_factory,
                )

            elif analyzer_type == "performance_attribution":
                from src.backtesting.attribution import PerformanceAttributor

                return PerformanceAttributor(config=config.get("config"))

            else:
                raise FactoryError(
                    f"Unknown analyzer type: {analyzer_type}", error_code="FACTORY_007"
                )

        except Exception as e:
            logger.error(f"Failed to create {analyzer_type} analyzer: {e}")
            raise FactoryError(
                f"Factory error creating analyzer: {e}", error_code="FACTORY_008"
            ) from e

    def wire_dependencies(self) -> None:
        """
        Wire all backtesting component dependencies.

        This method is deprecated - use register_backtesting_services()
        from di_registration.py instead to avoid duplicate registrations.
        """
        logger.warning(
            "wire_dependencies() is deprecated - use register_backtesting_services() instead"
        )

    # Interface compliance methods
    def get_interface(self) -> "BacktestServiceInterface":
        """Return interface instance for dependency injection."""
        return self.create_service()
