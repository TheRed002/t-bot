"""
Dependency Injection Registration for Execution Module.

This module registers all execution module dependencies with the DI container,
properly configuring the service layer architecture.
"""

from typing import TYPE_CHECKING, Any

from src.core.config import Config
from src.core.logging import get_logger
from src.core.types import ExecutionAlgorithm
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
from src.execution.high_performance_executor import HighPerformanceExecutor
from src.execution.interfaces import ExecutionAlgorithmFactoryInterface
from src.execution.order_management_service import OrderManagementService
from src.execution.order_manager import OrderManager
from src.execution.repository import (
    DatabaseExecutionRepository,
    DatabaseOrderRepository,
    ExecutionRepositoryInterface,
    OrderRepositoryInterface,
)
from src.execution.service import ExecutionService
from src.execution.service_adapters import (
    ExecutionEngineServiceAdapter,
    OrderManagementServiceAdapter,
    RiskValidationServiceAdapter,
)
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
        self.container.register(
            ExecutionRepositoryInterface,
            lambda c: DatabaseExecutionRepository(
                database_service=c.get("DatabaseService")
            ),
            singleton=True
        )

        self.container.register(
            OrderRepositoryInterface,
            lambda c: DatabaseOrderRepository(
                database_service=c.get("DatabaseService")
            ),
            singleton=True
        )

        # Register concrete implementations
        self.container.register(
            "ExecutionRepository",
            lambda c: c.get(ExecutionRepositoryInterface),
            singleton=True
        )

        self.container.register(
            "OrderRepository",
            lambda c: c.get(OrderRepositoryInterface),
            singleton=True
        )

    def _register_components(self) -> None:
        """Register core execution components."""
        # Register OrderManager
        self.container.register(
            OrderManager,
            lambda c: OrderManager(
                config=self.config,
                redis_client=c.get_optional("RedisClient"),
                state_service=c.get_optional("StateService"),
                metrics_collector=c.get_optional("MetricsCollector"),
            ),
            singleton=True
        )

        # Register SlippageModel
        self.container.register(
            "SlippageModel",
            lambda c: SlippageModel(config=self.config),
            singleton=True
        )

        # Register CostAnalyzer using dependency injection
        self.container.register(
            "CostAnalyzer",
            lambda c: CostAnalyzer(
                execution_service=c.get_optional("ExecutionService"),
                config=self.config
            ),
            singleton=True
        )

        # Register algorithm factory interface
        self.container.register(
            ExecutionAlgorithmFactoryInterface,
            lambda c: ExecutionAlgorithmFactory(self.config),
            singleton=True
        )

        # Register concrete algorithm factory
        self.container.register(
            "ExecutionAlgorithmFactory",
            lambda c: c.get(ExecutionAlgorithmFactoryInterface),
            singleton=True
        )

        # Register execution algorithms using factory pattern
        self.container.register(
            "ExecutionAlgorithms",
            lambda c: c.get(ExecutionAlgorithmFactoryInterface).create_all_algorithms(),
            singleton=True
        )

        # Register HighPerformanceExecutor
        self.container.register(
            HighPerformanceExecutor,
            lambda c: HighPerformanceExecutor(
                config=self.config
            ),
            singleton=True
        )

    def _register_services(self) -> None:
        """Register service layer components."""
        # Register ExecutionService (should already be registered by database module)
        if not self.container.is_registered("ExecutionService"):
            self.container.register(
                "ExecutionService",
                lambda c: ExecutionService(
                    database_service=c.get("DatabaseService"),
                    risk_service=c.get_optional("RiskService"),
                    metrics_service=c.get_optional("MetricsService"),
                    validation_service=c.get_optional("ValidationService"),
                    analytics_service=c.get_optional("AnalyticsService"),
                ),
                singleton=True
            )

        # Register OrderManagementService
        self.container.register(
            OrderManagementService,
            lambda c: OrderManagementService(
                order_manager=c.get(OrderManager)
            ),
            singleton=True
        )

    def _register_service_adapters(self) -> None:
        """Register service adapters that wrap existing components."""
        # Register ExecutionEngineServiceAdapter
        self.container.register(
            "ExecutionEngineServiceAdapter",
            lambda c: ExecutionEngineServiceAdapter(
                execution_engine=c.get_optional("ExecutionEngine")  # Will be None initially
            ),
            singleton=True
        )

        # Register OrderManagementServiceAdapter
        self.container.register(
            "OrderManagementServiceAdapter",
            lambda c: OrderManagementServiceAdapter(
                order_manager=c.get(OrderManager)
            ),
            singleton=True
        )

        # Register RiskValidationServiceAdapter
        self.container.register(
            "RiskValidationServiceAdapter",
            lambda c: RiskValidationServiceAdapter(
                risk_service=c.get_optional("RiskService")
            ),
            singleton=True
        )

    def _register_orchestration_services(self) -> None:
        """Register orchestration services."""
        # First register ExecutionEngine without orchestration service to break circular dependency
        self.container.register(
            ExecutionEngine,
            lambda c: ExecutionEngine(
                orchestration_service=None,  # Will be set later via property injection
                execution_service=c.get_optional("ExecutionService"),
                risk_service=c.get_optional("RiskService"),
                config=self.config,
                exchange_factory=c.get_optional("ExchangeFactory"),
                state_service=c.get_optional("StateService"),
                trade_lifecycle_manager=c.get_optional("TradeLifecycleManager"),
                metrics_collector=c.get_optional("MetricsCollector"),
                order_manager=c.get(OrderManager),
                slippage_model=c.get("SlippageModel"),
                cost_analyzer=c.get_optional("CostAnalyzer"),
                algorithms=c.get("ExecutionAlgorithms"),
            ),
            singleton=True
        )

        # Register ExecutionEngineServiceAdapter with the actual engine
        self.container.register(
            "ExecutionEngineServiceAdapter",
            lambda c: ExecutionEngineServiceAdapter(
                execution_engine=c.get(ExecutionEngine)
            ),
            singleton=True
        )

        # Register ExecutionOrchestrationService
        self.container.register(
            ExecutionOrchestrationService,
            lambda c: ExecutionOrchestrationService(
                execution_service=c.get("ExecutionService"),
                order_management_service=c.get(OrderManagementService),
                execution_engine_service=c.get("ExecutionEngineServiceAdapter"),
                risk_validation_service=c.get_optional("RiskValidationServiceAdapter"),
            ),
            singleton=True
        )

        # Set up lazy resolution for circular dependency
        def _get_execution_engine_with_orchestration(container):
            # Check if ExecutionEngine already exists
            if hasattr(container, "_execution_engine_instance"):
                return container._execution_engine_instance

            # Create ExecutionEngine without orchestration service first
            execution_engine = ExecutionEngine(
                orchestration_service=None,
                execution_service=container.get_optional("ExecutionService"),
                risk_service=container.get_optional("RiskService"),
                config=self.config,
                exchange_factory=container.get_optional("ExchangeFactory"),
                state_service=container.get_optional("StateService"),
                trade_lifecycle_manager=container.get_optional("TradeLifecycleManager"),
                metrics_collector=container.get_optional("MetricsCollector"),
                order_manager=container.get(OrderManager),
                slippage_model=container.get("SlippageModel"),
                cost_analyzer=container.get_optional("CostAnalyzer"),
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
        self.container.register(
            ExecutionEngine,
            _get_execution_engine_with_orchestration,
            singleton=True,
            force=True
        )

    def _register_controllers(self) -> None:
        """Register controller layer components."""
        # ExecutionController registration commented out until BaseController is available
        # self.container.register(
        #     ExecutionController,
        #     lambda c: ExecutionController(
        #         orchestration_service=c.get(ExecutionOrchestrationService),
        #         execution_service=c.get('ExecutionService'),
        #     ),
        #     singleton=True
        # )
        pass

    def register_for_testing(self) -> None:
        """Register components specifically for testing scenarios."""
        try:
            # Register mock implementations for testing - import locally to avoid circular dependencies
            if TYPE_CHECKING:
                from src.execution.mocks import MockExecutionRepository, MockOrderRepository
            else:
                from src.execution.mocks import MockExecutionRepository, MockOrderRepository

            self.container.register(
                "MockExecutionRepository",
                lambda c: MockExecutionRepository(),
                singleton=True
            )

            self.container.register(
                "MockOrderRepository",
                lambda c: MockOrderRepository(),
                singleton=True
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
            OrderManagementService,
            ExecutionOrchestrationService,
            ExecutionEngine,
            # ExecutionController,  # Commented out until BaseController available
            OrderManager,
            ExecutionAlgorithmFactoryInterface,
            "ExecutionAlgorithmFactory",
        ]

        try:
            for component in required_components:
                if not self.container.is_registered(component):
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
                "OrderManagementService",
                "ExecutionOrchestrationService",
            ],
            "factories": [
                "ExecutionAlgorithmFactory",
            ],
            "components": [
                "ExecutionEngine",
                "OrderManager",
                "HighPerformanceExecutor",
            ],
            "controllers": [
                # "ExecutionController",  # Commented out until BaseController available
            ],
            "service_adapters": [
                "ExecutionEngineServiceAdapter",
                "OrderManagementServiceAdapter",
                "RiskValidationServiceAdapter",
            ],
            "configuration": {
                "follows_service_patterns": True,
                "has_repository_interfaces": True,
                "has_service_orchestration": True,
                "controller_service_only": True,
            }
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
        raise RuntimeError("Execution module DI registration validation failed")

    return registration


    def _create_execution_algorithms_factory(self, container) -> dict:
        """Factory method to create execution algorithms using dependency injection."""
        try:
            return {
                ExecutionAlgorithm.TWAP: TWAPAlgorithm(self.config),
                ExecutionAlgorithm.VWAP: VWAPAlgorithm(self.config),
                ExecutionAlgorithm.ICEBERG: IcebergAlgorithm(self.config),
                ExecutionAlgorithm.SMART_ROUTER: SmartOrderRouter(self.config),
            }
        except Exception as e:
            self.logger.error(f"Failed to create execution algorithms: {e}")
            # Fallback to minimal algorithms if config issues - import locally to avoid circular dependencies
            if TYPE_CHECKING:
                from src.execution.algorithms.base_algorithm import BaseAlgorithm
            else:
                from src.execution.algorithms.base_algorithm import BaseAlgorithm

            class MockAlgorithm(BaseAlgorithm):
                def __init__(self, algorithm_type):
                    super().__init__(self.config)
                    self.algorithm_type = algorithm_type

                def get_algorithm_type(self):
                    return self.algorithm_type

                async def execute(self, instruction, exchange_factory=None, risk_manager=None):
                    # Local import to avoid circular dependency
                    if TYPE_CHECKING:
                        from src.execution.execution_result_wrapper import ExecutionResultWrapper
                    else:
                        from src.execution.execution_result_wrapper import ExecutionResultWrapper
                    return ExecutionResultWrapper(None)

                async def cancel_execution(self, execution_id):
                    return True

                async def _validate_algorithm_parameters(self, instruction):
                    pass

            return {
                ExecutionAlgorithm.TWAP: MockAlgorithm(ExecutionAlgorithm.TWAP),
                ExecutionAlgorithm.VWAP: MockAlgorithm(ExecutionAlgorithm.VWAP),
                ExecutionAlgorithm.ICEBERG: MockAlgorithm(ExecutionAlgorithm.ICEBERG),
                ExecutionAlgorithm.SMART_ROUTER: MockAlgorithm(ExecutionAlgorithm.SMART_ROUTER),
            }


def _create_execution_algorithms(config: Config) -> dict:
    """Legacy factory method for backward compatibility."""
    try:
        return {
            ExecutionAlgorithm.TWAP: TWAPAlgorithm(config),
            ExecutionAlgorithm.VWAP: VWAPAlgorithm(config),
            ExecutionAlgorithm.ICEBERG: IcebergAlgorithm(config),
            ExecutionAlgorithm.SMART_ROUTER: SmartOrderRouter(config),
        }
    except Exception:
        # Fallback to minimal algorithms if config issues - import locally to avoid circular dependencies
        if TYPE_CHECKING:
            from src.execution.algorithms.base_algorithm import BaseAlgorithm
        else:
            from src.execution.algorithms.base_algorithm import BaseAlgorithm

        class MockAlgorithm(BaseAlgorithm):
            def __init__(self, algorithm_type):
                super().__init__(config)
                self.algorithm_type = algorithm_type

            def get_algorithm_type(self):
                return self.algorithm_type

            async def execute(self, instruction, exchange_factory=None, risk_manager=None):
                # Local import to avoid circular dependency
                if TYPE_CHECKING:
                    from src.execution.execution_result_wrapper import ExecutionResultWrapper
                else:
                    from src.execution.execution_result_wrapper import ExecutionResultWrapper
                return ExecutionResultWrapper(None)

            async def cancel_execution(self, execution_id):
                return True

            async def _validate_algorithm_parameters(self, instruction):
                pass

        return {
            ExecutionAlgorithm.TWAP: MockAlgorithm(ExecutionAlgorithm.TWAP),
            ExecutionAlgorithm.VWAP: MockAlgorithm(ExecutionAlgorithm.VWAP),
            ExecutionAlgorithm.ICEBERG: MockAlgorithm(ExecutionAlgorithm.ICEBERG),
            ExecutionAlgorithm.SMART_ROUTER: MockAlgorithm(ExecutionAlgorithm.SMART_ROUTER),
        }
