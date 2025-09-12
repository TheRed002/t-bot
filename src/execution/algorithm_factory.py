"""
Execution Algorithm Factory - Following Proper Factory Pattern.

This module implements the factory pattern for creating execution algorithms
using dependency injection and proper interface contracts.
"""

from src.core.dependency_injection import DependencyInjector
from src.core.exceptions import DependencyError, ServiceError, ValidationError
from src.core.logging import get_logger
from src.core.types import ExecutionAlgorithm
from src.execution.algorithms.base_algorithm import BaseAlgorithm
from src.execution.interfaces import ExecutionAlgorithmFactoryInterface

# Algorithms are imported locally in methods to avoid circular dependencies


class ExecutionAlgorithmFactory(ExecutionAlgorithmFactoryInterface):
    """Factory for creating execution algorithm instances using dependency injection."""

    def __init__(self, injector: DependencyInjector):
        """
        Initialize algorithm factory using dependency injection.

        Args:
            injector: Dependency injector instance (required)

        Raises:
            DependencyError: If injector not provided or invalid
        """
        if injector is None:
            raise DependencyError(
                "Injector must be provided to factory",
                dependency_name="DependencyInjector",
                error_code="FAC_ALG_001",
                suggested_action="Provide configured dependency injector to factory constructor",
            )

        self.injector = injector
        self.logger = get_logger(__name__)

        # Registry of algorithm creators - lazy loaded to avoid circular imports
        self._algorithm_registry: dict[ExecutionAlgorithm, str] = {
            ExecutionAlgorithm.TWAP: "TWAPAlgorithm",
            ExecutionAlgorithm.VWAP: "VWAPAlgorithm",
            ExecutionAlgorithm.ICEBERG: "IcebergAlgorithm",
            ExecutionAlgorithm.SMART_ROUTER: "SmartOrderRouter",
        }

        self.logger.info(
            "ExecutionAlgorithmFactory initialized with dependency injection",
            available_algorithms=[algo.value for algo in self._algorithm_registry.keys()],
            has_config=self.injector.has_service("Config"),
        )

    def create_algorithm(self, algorithm_type: ExecutionAlgorithm) -> BaseAlgorithm:
        """
        Create an execution algorithm instance using dependency injection.

        Args:
            algorithm_type: Type of algorithm to create

        Returns:
            BaseAlgorithm: Algorithm instance

        Raises:
            ValidationError: If algorithm type not supported
            ServiceError: If algorithm creation fails
            DependencyError: If required dependencies not available
        """
        try:
            if algorithm_type not in self._algorithm_registry:
                # Handle both enum and string types for error messages
                algo_str = (
                    algorithm_type.value
                    if hasattr(algorithm_type, "value")
                    else str(algorithm_type)
                )
                raise ValidationError(f"Unsupported algorithm type: {algo_str}")

            # Use service locator pattern to resolve algorithm from DI container
            algorithm_service_name = self._algorithm_registry[algorithm_type]

            try:
                algorithm = self.injector.resolve(algorithm_service_name)
            except Exception as di_error:
                # Fallback to direct creation with config if service not registered
                self.logger.warning(
                    f"Algorithm service {algorithm_service_name} not registered in DI container, "
                    f"falling back to direct creation: {di_error}"
                )
                algorithm = self._create_algorithm_direct(algorithm_type)

            # Handle both enum and string types for logging
            algo_str = (
                algorithm_type.value if hasattr(algorithm_type, "value") else str(algorithm_type)
            )
            self.logger.debug(
                "Created execution algorithm via dependency injection",
                algorithm_type=algo_str,
                algorithm_class=type(algorithm).__name__,
            )

            return algorithm

        except ValidationError:
            raise
        except Exception as e:
            # Handle both enum and string types for error logging
            algo_str = (
                algorithm_type.value if hasattr(algorithm_type, "value") else str(algorithm_type)
            )
            self.logger.error(
                "Failed to create execution algorithm", algorithm_type=algo_str, error=str(e)
            )
            raise ServiceError(f"Algorithm creation failed: {e}")

    def _create_algorithm_direct(self, algorithm_type: ExecutionAlgorithm) -> BaseAlgorithm:
        """
        Direct algorithm creation fallback when DI resolution fails.

        Args:
            algorithm_type: Type of algorithm to create

        Returns:
            BaseAlgorithm: Algorithm instance

        Raises:
            DependencyError: If config service not available
            ServiceError: If algorithm creation fails
        """
        try:
            # Get config from DI container
            config = self.injector.resolve("Config")

            # Import and create algorithm directly
            if algorithm_type == ExecutionAlgorithm.TWAP:
                from src.execution.algorithms.twap import TWAPAlgorithm

                return TWAPAlgorithm(config)
            elif algorithm_type == ExecutionAlgorithm.VWAP:
                from src.execution.algorithms.vwap import VWAPAlgorithm

                return VWAPAlgorithm(config)
            elif algorithm_type == ExecutionAlgorithm.ICEBERG:
                from src.execution.algorithms.iceberg import IcebergAlgorithm

                return IcebergAlgorithm(config)
            elif algorithm_type == ExecutionAlgorithm.SMART_ROUTER:
                from src.execution.algorithms.smart_router import SmartOrderRouter

                return SmartOrderRouter(config)
            else:
                raise ValidationError(f"Unknown algorithm type: {algorithm_type}")

        except Exception as e:
            raise DependencyError(
                f"Failed to create algorithm {algorithm_type} directly: {e}",
                dependency_name="Config",
                error_code="FAC_ALG_002",
                suggested_action="Ensure Config service is registered in DI container",
            )

    def get_available_algorithms(self) -> list[ExecutionAlgorithm]:
        """
        Get list of available algorithm types.

        Returns:
            List[ExecutionAlgorithm]: Available algorithm types
        """
        return list(self._algorithm_registry.keys())

    def is_algorithm_available(self, algorithm_type: ExecutionAlgorithm) -> bool:
        """
        Check if algorithm type is available.

        Args:
            algorithm_type: Algorithm type to check

        Returns:
            bool: True if available
        """
        return algorithm_type in self._algorithm_registry

    def create_all_algorithms(self) -> dict[ExecutionAlgorithm, BaseAlgorithm]:
        """
        Create instances of all available algorithms using dependency injection.

        Returns:
            Dict[ExecutionAlgorithm, BaseAlgorithm]: All algorithm instances

        Raises:
            ServiceError: If any algorithm creation fails
        """
        try:
            algorithms = {}
            failures = []

            for algorithm_type in self._algorithm_registry:
                try:
                    algorithms[algorithm_type] = self.create_algorithm(algorithm_type)
                except Exception as e:
                    failures.append((algorithm_type, str(e)))
                    self.logger.error(
                        f"Failed to create algorithm {algorithm_type}: {e}",
                        algorithm_type=algorithm_type.value,
                        error=str(e),
                    )

            if failures:
                failure_summary = ", ".join(f"{algo.value}: {err}" for algo, err in failures)
                raise ServiceError(f"Failed to create some algorithms: {failure_summary}")

            self.logger.info(
                "Created all algorithm instances via dependency injection",
                count=len(algorithms),
                algorithms=[algo.value for algo in algorithms.keys()],
            )

            return algorithms

        except Exception as e:
            self.logger.error(f"Failed to create all algorithms: {e}")
            raise ServiceError(f"Algorithm factory initialization failed: {e}")

    def register_algorithm(self, algorithm_type: ExecutionAlgorithm, algorithm_class: type) -> None:
        """
        Register a new algorithm type.

        Args:
            algorithm_type: Algorithm type enum
            algorithm_class: Algorithm implementation class

        Raises:
            ValidationError: If algorithm class invalid
        """
        try:
            if not issubclass(algorithm_class, BaseAlgorithm):
                raise ValidationError("Algorithm class must inherit from BaseAlgorithm")

            self._algorithm_registry[algorithm_type] = algorithm_class

            self.logger.info(
                "Registered new algorithm type",
                algorithm_type=algorithm_type.value,
                algorithm_class=algorithm_class.__name__,
            )

        except Exception as e:
            self.logger.error(
                "Failed to register algorithm", algorithm_type=algorithm_type.value, error=str(e)
            )
            raise ValidationError(f"Algorithm registration failed: {e}")


def create_execution_algorithm_factory(injector: DependencyInjector) -> ExecutionAlgorithmFactory:
    """
    Factory function to create ExecutionAlgorithmFactory instance using dependency injection.

    Args:
        injector: Dependency injector instance (required)

    Returns:
        ExecutionAlgorithmFactory: Factory instance

    Raises:
        DependencyError: If injector not provided
    """
    if injector is None:
        raise DependencyError(
            "Dependency injector must be provided to create factory",
            dependency_name="DependencyInjector",
            error_code="FAC_ALG_003",
            suggested_action="Provide configured dependency injector",
        )

    return ExecutionAlgorithmFactory(injector)


def get_algorithm_factory(injector: DependencyInjector) -> ExecutionAlgorithmFactory:
    """
    Get execution algorithm factory using dependency injection.

    Args:
        injector: Dependency injector instance (required)

    Returns:
        ExecutionAlgorithmFactory instance

    Raises:
        DependencyError: If injector not provided
    """
    return create_execution_algorithm_factory(injector)


def create_algorithm(
    injector: DependencyInjector,
    algorithm_type: ExecutionAlgorithm,
) -> BaseAlgorithm:
    """
    Convenience function to create algorithm using dependency injection.

    Args:
        injector: Dependency injector instance (required)
        algorithm_type: Type of algorithm to create

    Returns:
        BaseAlgorithm: Algorithm instance

    Raises:
        DependencyError: If injector not provided
        ValidationError: If algorithm type not supported
        ServiceError: If algorithm creation fails
    """
    factory = get_algorithm_factory(injector)
    return factory.create_algorithm(algorithm_type)
