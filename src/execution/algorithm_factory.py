"""
Execution Algorithm Factory - Following Proper Factory Pattern.

This module implements the factory pattern for creating execution algorithms
using dependency injection and proper interface contracts.
"""


from src.core.config import Config
from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.core.types import ExecutionAlgorithm
from src.execution.algorithms.base_algorithm import BaseAlgorithm
from src.execution.algorithms.iceberg import IcebergAlgorithm
from src.execution.algorithms.smart_router import SmartOrderRouter
from src.execution.algorithms.twap import TWAPAlgorithm
from src.execution.algorithms.vwap import VWAPAlgorithm
from src.execution.interfaces import ExecutionAlgorithmFactoryInterface


class ExecutionAlgorithmFactory(ExecutionAlgorithmFactoryInterface):
    """Factory for creating execution algorithm instances using dependency injection."""

    def __init__(self, config: Config):
        """
        Initialize algorithm factory.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Registry of algorithm creators
        self._algorithm_registry: dict[ExecutionAlgorithm, type] = {
            ExecutionAlgorithm.TWAP: TWAPAlgorithm,
            ExecutionAlgorithm.VWAP: VWAPAlgorithm,
            ExecutionAlgorithm.ICEBERG: IcebergAlgorithm,
            ExecutionAlgorithm.SMART_ROUTER: SmartOrderRouter,
        }

        self.logger.info(
            "ExecutionAlgorithmFactory initialized",
            available_algorithms=[algo.value for algo in self._algorithm_registry.keys()]
        )

    def create_algorithm(self, algorithm_type: ExecutionAlgorithm) -> BaseAlgorithm:
        """
        Create an execution algorithm instance.
        
        Args:
            algorithm_type: Type of algorithm to create
            
        Returns:
            BaseAlgorithm: Algorithm instance
            
        Raises:
            ValidationError: If algorithm type not supported
            ServiceError: If algorithm creation fails
        """
        try:
            if algorithm_type not in self._algorithm_registry:
                # Handle both enum and string types for error messages
                algo_str = algorithm_type.value if hasattr(algorithm_type, 'value') else str(algorithm_type)
                raise ValidationError(f"Unsupported algorithm type: {algo_str}")

            algorithm_class = self._algorithm_registry[algorithm_type]
            algorithm = algorithm_class(self.config)

            # Handle both enum and string types for logging
            algo_str = algorithm_type.value if hasattr(algorithm_type, 'value') else str(algorithm_type)
            self.logger.debug(
                "Created execution algorithm",
                algorithm_type=algo_str,
                algorithm_class=algorithm_class.__name__
            )

            return algorithm

        except ValidationError:
            raise
        except Exception as e:
            # Handle both enum and string types for error logging
            algo_str = algorithm_type.value if hasattr(algorithm_type, 'value') else str(algorithm_type)
            self.logger.error(
                "Failed to create execution algorithm",
                algorithm_type=algo_str,
                error=str(e)
            )
            raise ServiceError(f"Algorithm creation failed: {e}")

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
        Create instances of all available algorithms.
        
        Returns:
            Dict[ExecutionAlgorithm, BaseAlgorithm]: All algorithm instances
            
        Raises:
            ServiceError: If any algorithm creation fails
        """
        try:
            algorithms = {}
            for algorithm_type in self._algorithm_registry:
                algorithms[algorithm_type] = self.create_algorithm(algorithm_type)

            self.logger.info(
                "Created all algorithm instances",
                count=len(algorithms)
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
                algorithm_class=algorithm_class.__name__
            )

        except Exception as e:
            self.logger.error(
                "Failed to register algorithm",
                algorithm_type=algorithm_type.value,
                error=str(e)
            )
            raise ValidationError(f"Algorithm registration failed: {e}")


def create_execution_algorithm_factory(config: Config) -> ExecutionAlgorithmFactory:
    """
    Factory function to create ExecutionAlgorithmFactory instance.
    
    Args:
        config: Application configuration
        
    Returns:
        ExecutionAlgorithmFactory: Factory instance
    """
    return ExecutionAlgorithmFactory(config)
