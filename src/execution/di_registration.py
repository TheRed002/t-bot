"""
Dependency Injection Registration for Execution Module.

This module registers all execution module dependencies with the DI container,
properly configuring the service layer architecture.
"""

from typing import TYPE_CHECKING, Any

from src.core.config import Config
from src.core.exceptions import ServiceError
from src.core.logging import get_logger
from src.execution.algorithm_factory import (
    ExecutionAlgorithmFactory,
)
from src.execution.algorithms.iceberg import IcebergAlgorithm
from src.execution.algorithms.smart_router import SmartOrderRouter
from src.execution.algorithms.twap import TWAPAlgorithm
from src.execution.algorithms.vwap import VWAPAlgorithm

# Controller import commented out until BaseController is available
# from src.execution.controller import ExecutionController
from src.execution.execution_engine import ExecutionEngine
from src.execution.execution_orchestration_service import ExecutionOrchestrationService
from src.execution.interfaces import (
    ExecutionAlgorithmFactoryInterface,
    ExecutionEngineServiceInterface,
)
from src.execution.order_manager import OrderManager
from src.execution.repository import (
    DatabaseExecutionRepository,
    DatabaseOrderRepository,
    ExecutionRepositoryInterface,
    OrderRepositoryInterface,
)
from src.execution.service import ExecutionService
from src.execution.slippage.cost_analyzer import CostAnalyzer
from src.execution.slippage.slippage_model import SlippageModel


class ExecutionModuleDIRegistration:
    """Handles dependency injection registration for execution module."""

    def __init__(self, container, config: Config):
        """
        Initialize DI registration.

        Args:
            container: DI container instance
            config: Application configuration
        """
        self.container = container
        self.config = config
        self.logger = get_logger(__name__)

    def _register_dependency(
        self, name_or_interface, service_factory, singleton: bool = True
    ) -> None:
        """Helper method to register dependencies with different container types."""
        if hasattr(self.container, "register_service"):
            # DependencyInjector style
            self.container.register_service(name_or_interface, service_factory, singleton=singleton)
        elif hasattr(self.container, "register"):
            # DependencyContainer style
            self.container.register(name_or_interface, service_factory, singleton=singleton)
        else:
            raise ServiceError(f"Unsupported container type: {type(self.container)}")

    def _is_registered(self, name_or_interface) -> bool:
        """Helper method to check if dependency is registered with different container types."""
        if hasattr(self.container, "is_registered"):
            return self.container.is_registered(name_or_interface)
        elif hasattr(self.container, "has"):
            return self.container.has(name_or_interface)
        else:
            return False

    def register_all(self) -> None:
        """Register all execution module dependencies."""
        try:
            # Register in dependency order
            self._register_repositories()
            self._register_components()
            self._register_services()
            self._register_service_adapters()
            self._register_orchestration_services()
            self._register_controllers()

            self.logger.info("Execution module DI registration completed")

        except Exception as e:
            self.logger.error("Failed to register execution module dependencies", error=str(e))
            raise

    def _register_repositories(self) -> None:
        """Register repository interfaces and implementations."""
        # Register repository interfaces
        self._register_dependency(
            ExecutionRepositoryInterface,
            lambda c: DatabaseExecutionRepository(database_service=c.get("DatabaseService")),
            singleton=True,
        )

        self._register_dependency(
            OrderRepositoryInterface,
            lambda c: DatabaseOrderRepository(database_service=c.get("DatabaseService")),
            singleton=True,
        )

        # Register concrete implementations
        self._register_dependency(
            "ExecutionRepository", lambda c: c.get(ExecutionRepositoryInterface), singleton=True
        )

        self._register_dependency(
            "OrderRepository", lambda c: c.get(OrderRepositoryInterface), singleton=True
        )

    def _register_components(self) -> None:
        """Register core execution components."""
        # Register OrderManager with ExchangeService dependency
        self._register_dependency(
            OrderManager,
            lambda c: OrderManager(
                config=self.config,
                exchange_service=c.get_optional("ExchangeService"),  # Inject ExchangeService
                redis_client=c.get_optional("RedisClient"),
                state_service=c.get_optional("StateService"),
                metrics_collector=c.get_optional("MetricsCollector"),
            ),
            singleton=True,
        )

        # Register SlippageModel
        self._register_dependency(
            "SlippageModel", lambda c: SlippageModel(config=self.config), singleton=True
        )

        # Register CostAnalyzer using dependency injection
        self._register_dependency(
            "CostAnalyzer",
            lambda c: CostAnalyzer(
                execution_service=c.get_optional("ExecutionService"), config=self.config
            ),
            singleton=True,
        )

        # Register algorithm factory interface using dependency injection
        self._register_dependency(
            ExecutionAlgorithmFactoryInterface,
            lambda c: ExecutionAlgorithmFactory(injector=c),
            singleton=True,
        )

        # Register concrete algorithm factory
        self._register_dependency(
            "ExecutionAlgorithmFactory",
            lambda c: c.get(ExecutionAlgorithmFactoryInterface),
            singleton=True,
        )

        # Register individual execution algorithms for direct resolution
        self._register_dependency(
            "TWAPAlgorithm",
            lambda c: TWAPAlgorithm(config=c.get("Config")),
            singleton=False,  # Create new instances as needed
        )

        self._register_dependency(
            "VWAPAlgorithm",
            lambda c: VWAPAlgorithm(config=c.get("Config")),
            singleton=False,
        )

        self._register_dependency(
            "IcebergAlgorithm",
            lambda c: IcebergAlgorithm(config=c.get("Config")),
            singleton=False,
        )

        self._register_dependency(
            "SmartOrderRouter",
            lambda c: SmartOrderRouter(config=c.get("Config")),
            singleton=False,
        )

        # Register execution algorithms using factory pattern
        self._register_dependency(
            "ExecutionAlgorithms",
            lambda c: c.get(ExecutionAlgorithmFactoryInterface).create_all_algorithms(),
            singleton=True,
        )

    def _register_services(self) -> None:
        """Register service layer components."""
        # Register ExecutionService with proper interface-based dependencies
        if not self._is_registered("ExecutionService"):
            self._register_dependency(
                "ExecutionService",
                lambda c: ExecutionService(
                    database_service=c.get("DatabaseService"),
                    risk_service=c.get_optional("RiskService"),
                    metrics_service=c.get_optional("MetricsService"),
                    validation_service=c.get_optional("ValidationService"),
                    analytics_service=c.get_optional("AnalyticsService"),
                ),
                singleton=True,
            )

        # Register ExecutionService interface
        self._register_dependency(
            "ExecutionServiceInterface",
            lambda c: c.get("ExecutionService"),
            singleton=True,
        )

    def _register_service_adapters(self) -> None:
        """Register service adapters that wrap existing components."""
        # Simplified after removing unnecessary service adapters
        pass

    def _register_orchestration_services(self) -> None:
        """Register orchestration services."""
        # First register ExecutionEngine without orchestration service to break circular dependency
        self._register_dependency(
            ExecutionEngine,
            lambda c: ExecutionEngine(
                execution_service=c.get_optional("ExecutionService"),
                risk_service=c.get_optional("RiskService"),
                config=self.config,
                orchestration_service=None,  # Will be set later via property injection
                exchange_factory=c.get_optional("ExchangeFactory"),
                state_service=c.get_optional("StateService"),
                trade_lifecycle_manager=c.get_optional("TradeLifecycleManager"),
                metrics_collector=c.get_optional("MetricsCollector"),
                order_manager=c.get(OrderManager),
                slippage_model=c.get("SlippageModel"),
                cost_analyzer=c.get("CostAnalyzer"),
                algorithms=c.get("ExecutionAlgorithms"),
            ),
            singleton=True,
        )

        # Register ExecutionOrchestrationService - simplified to use direct components
        self._register_dependency(
            ExecutionOrchestrationService,
            lambda c: ExecutionOrchestrationService(
                execution_service=c.get("ExecutionServiceInterface"),
                order_manager=c.get(OrderManager),
                execution_engine=c.get(ExecutionEngine),
                risk_service=c.get_optional("RiskService"),
            ),
            singleton=True,
        )

        # Register ExecutionOrchestrationService interface
        self._register_dependency(
            "ExecutionOrchestrationServiceInterface",
            lambda c: c.get(ExecutionOrchestrationService),
            singleton=True,
        )

        # Register ExecutionEngineServiceInterface
        self._register_dependency(
            ExecutionEngineServiceInterface,
            lambda c: c.get(ExecutionEngine),
            singleton=True,
        )

        # Set up lazy resolution for circular dependency
        def _get_execution_engine_with_orchestration(container):
            # Check if ExecutionEngine already exists
            if hasattr(container, "_execution_engine_instance"):
                return container._execution_engine_instance

            # Create ExecutionEngine without orchestration service first
            execution_engine = ExecutionEngine(
                execution_service=container.get_optional("ExecutionService"),
                risk_service=container.get_optional("RiskService"),
                config=self.config,
                orchestration_service=None,
                exchange_factory=container.get_optional("ExchangeFactory"),
                state_service=container.get_optional("StateService"),
                trade_lifecycle_manager=container.get_optional("TradeLifecycleManager"),
                metrics_collector=container.get_optional("MetricsCollector"),
                order_manager=container.get(OrderManager),
                slippage_model=container.get("SlippageModel"),
                cost_analyzer=container.get("CostAnalyzer"),
                algorithms=container.get("ExecutionAlgorithms"),
            )

            # Cache the instance
            container._execution_engine_instance = execution_engine

            # Try to set orchestration service if available (lazy loading)
            try:
                orchestration_service = container.get_optional(ExecutionOrchestrationService)
                if orchestration_service:
                    execution_engine.orchestration_service = orchestration_service
            except Exception:
                # Orchestration service not available yet, will be set later
                pass

            return execution_engine

        # Override ExecutionEngine registration with lazy setup
        self._register_dependency(
            ExecutionEngine, _get_execution_engine_with_orchestration, singleton=True
        )

    def _register_controllers(self) -> None:
        """Register controller layer components."""
        # Import ExecutionController locally to avoid circular dependencies
        try:
            from src.execution.controller import ExecutionController

            self._register_dependency(
                ExecutionController,
                lambda c: ExecutionController(
                    orchestration_service=c.get("ExecutionOrchestrationServiceInterface"),
                    execution_service=c.get("ExecutionServiceInterface"),
                ),
                singleton=True,
            )

            # Register controller interface
            self._register_dependency(
                "ExecutionControllerInterface",
                lambda c: c.get(ExecutionController),
                singleton=True,
            )

            self.logger.info("ExecutionController registered successfully")

        except ImportError as e:
            self.logger.warning(f"ExecutionController not available: {e}")
        except Exception as e:
            self.logger.error(f"Failed to register ExecutionController: {e}")

    def register_for_testing(self) -> None:
        """Register components specifically for testing scenarios."""
        try:
            # Register mock implementations for testing - import locally to avoid circular dependencies
            if TYPE_CHECKING:
                from src.execution.mocks import MockExecutionRepository, MockOrderRepository
            else:
                from src.execution.mocks import MockExecutionRepository, MockOrderRepository

            self._register_dependency(
                "MockExecutionRepository", lambda c: MockExecutionRepository(), singleton=True
            )

            self._register_dependency(
                "MockOrderRepository", lambda c: MockOrderRepository(), singleton=True
            )

            self.logger.info("Execution module test dependencies registered")

        except ImportError:
            self.logger.warning("Mock implementations not available for testing")
        except Exception as e:
            self.logger.error("Failed to register test dependencies", error=str(e))

    def validate_registrations(self) -> bool:
        """Validate that all required dependencies are registered."""
        required_components = [
            "ExecutionService",
            ExecutionOrchestrationService,
            ExecutionEngine,
            # ExecutionController,  # Commented out until BaseController available
            OrderManager,
            ExecutionAlgorithmFactoryInterface,
            "ExecutionAlgorithmFactory",
        ]

        try:
            for component in required_components:
                if not self._is_registered(component):
                    self.logger.error(f"Required component not registered: {component}")
                    return False

            self.logger.info("All execution module dependencies validated")
            return True

        except Exception as e:
            self.logger.error("Failed to validate registrations", error=str(e))
            return False

    def get_registration_info(self) -> dict[str, Any]:
        """Get information about registered components."""
        return {
            "module": "execution",
            "repositories": [
                "ExecutionRepositoryInterface",
                "OrderRepositoryInterface",
            ],
            "services": [
                "ExecutionService",
                "ExecutionOrchestrationService",
            ],
            "factories": [
                "ExecutionAlgorithmFactory",
            ],
            "components": [
                "ExecutionEngine",
                "OrderManager",
            ],
            "controllers": [
                # "ExecutionController",  # Commented out until BaseController available
            ],
            "service_adapters": [
                # Removed unnecessary service adapters for simplification
            ],
            "configuration": {
                "follows_service_patterns": True,
                "has_repository_interfaces": True,
                "has_service_orchestration": True,
                "controller_service_only": True,
            },
        }


def register_execution_module(container, config: Config) -> ExecutionModuleDIRegistration:
    """
    Register execution module with DI container.

    Args:
        container: DI container instance
        config: Application configuration

    Returns:
        ExecutionModuleDIRegistration instance
    """
    registration = ExecutionModuleDIRegistration(container, config)
    registration.register_all()

    if not registration.validate_registrations():
        raise ServiceError("Execution module DI registration validation failed")

    return registration
