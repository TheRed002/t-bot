"""Unit tests for AlgorithmFactory."""

import logging
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

# Disable logging for performance
logging.disable(logging.CRITICAL)

# Pre-defined constants for faster test data creation
TEST_DECIMALS = {
    "ONE": Decimal("1.0"),
    "PRICE_50K": Decimal("50000"),
    "PORTFOLIO_100K": "100000"
}

from src.core.config import Config
from src.core.dependency_injection import DependencyInjector
from src.core.types import ExecutionAlgorithm
from src.execution.algorithm_factory import ExecutionAlgorithmFactory
from src.execution.types import ExecutionInstruction

# Cache common mock configurations for better performance
COMMON_CONFIG_ATTRS = {
    "execution_get_return": TEST_DECIMALS["PORTFOLIO_100K"],
    "symbol": "BTC/USDT",
    "strategy_name": "test_strategy",
    "time_horizon": 60,
    "max_slices": 10
}


class TestAlgorithmFactory:
    """Test cases for AlgorithmFactory."""

    @pytest.fixture(scope="session")
    def config(self):
        """Create test configuration."""
        config = MagicMock()
        config.execution = MagicMock()
        config.execution.get = MagicMock(
            return_value=COMMON_CONFIG_ATTRS["execution_get_return"]
        )  # Return string for Decimal conversion
        # Add additional required attributes for performance
        config.database = MagicMock()
        config.monitoring = MagicMock()
        config.redis = MagicMock()
        config.error_handling = MagicMock()
        return config

    @pytest.fixture(scope="session")
    def mock_injector(self, config):
        """Create mock dependency injector."""
        injector = MagicMock(spec=DependencyInjector)
        def resolve_side_effect(service):
            if service == "Config":
                return config
            # Simulate algorithm services not being registered - raise exception
            raise Exception(f"Service {service} not found")
            
        injector.resolve.side_effect = resolve_side_effect
        injector.has_service.side_effect = lambda service: service == "Config"
        return injector

    @pytest.fixture(scope="session")
    def algorithm_factory(self, mock_injector):
        """Create ExecutionAlgorithmFactory instance."""
        return ExecutionAlgorithmFactory(mock_injector)

    @pytest.fixture(scope="session")
    def sample_execution_instruction(self):
        """Create sample execution instruction using cached constants."""
        from src.core.types import OrderRequest, OrderSide, OrderType

        order = OrderRequest(
            symbol=COMMON_CONFIG_ATTRS["symbol"],
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=TEST_DECIMALS["ONE"],
            price=TEST_DECIMALS["PRICE_50K"],
        )
        return ExecutionInstruction(
            order=order,
            algorithm=ExecutionAlgorithm.TWAP,
            strategy_name=COMMON_CONFIG_ATTRS["strategy_name"],
            time_horizon_minutes=COMMON_CONFIG_ATTRS["time_horizon"],
            max_slices=COMMON_CONFIG_ATTRS["max_slices"],
        )

    def test_initialization(self, algorithm_factory, mock_injector):
        """Test ExecutionAlgorithmFactory initialization."""
        assert algorithm_factory.injector == mock_injector
        assert hasattr(algorithm_factory, "_algorithm_registry")
        assert isinstance(algorithm_factory._algorithm_registry, dict)

    def test_create_all_algorithms(self, algorithm_factory):
        """Test creating all algorithm types efficiently."""
        # Batch test all algorithm creations
        algorithm_types = [
            (ExecutionAlgorithm.TWAP, "TWAPAlgorithm"),
            (ExecutionAlgorithm.VWAP, "VWAPAlgorithm"),
            (ExecutionAlgorithm.ICEBERG, "IcebergAlgorithm"),
            (ExecutionAlgorithm.SMART_ROUTER, "SmartOrderRouter")
        ]
        
        algorithms = {}
        for algo_type, expected_name in algorithm_types:
            algorithms[algo_type] = algorithm_factory.create_algorithm(algo_type)
        
        # Batch verify all algorithms
        for algo_type, expected_name in algorithm_types:
            algorithm = algorithms[algo_type]
            assert algorithm is not None
            assert algorithm.__class__.__name__ == expected_name

    def test_create_algorithm_invalid(self, algorithm_factory):
        """Test creating invalid algorithm raises error."""
        from src.core.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Unsupported algorithm"):
            algorithm_factory.create_algorithm("invalid_algorithm")

    def test_create_algorithm_with_config(self, algorithm_factory, config):
        """Test algorithm creation uses config settings."""
        # Test that factory passes config to algorithms
        algorithm = algorithm_factory.create_algorithm(ExecutionAlgorithm.TWAP)
        assert algorithm.config == config

    def test_algorithm_creation_new_instances(self, algorithm_factory):
        """Test that each algorithm creation returns a new instance."""
        algorithm1 = algorithm_factory.create_algorithm(ExecutionAlgorithm.TWAP)
        algorithm2 = algorithm_factory.create_algorithm(ExecutionAlgorithm.TWAP)
        # Each call creates a new instance
        assert algorithm1 is not algorithm2
        assert type(algorithm1) == type(algorithm2)

    def test_get_available_algorithms(self, algorithm_factory):
        """Test getting available algorithms."""
        algorithms = algorithm_factory.get_available_algorithms()
        assert ExecutionAlgorithm.TWAP in algorithms
        assert ExecutionAlgorithm.VWAP in algorithms
        assert ExecutionAlgorithm.ICEBERG in algorithms
        assert ExecutionAlgorithm.SMART_ROUTER in algorithms

    def test_is_algorithm_available(self, algorithm_factory):
        """Test checking if algorithm is available."""
        assert algorithm_factory.is_algorithm_available(ExecutionAlgorithm.TWAP)
        assert algorithm_factory.is_algorithm_available(ExecutionAlgorithm.VWAP)
        # Test with invalid enum should create a dummy enum value for testing
        from enum import Enum

        class TestEnum(Enum):
            INVALID = "invalid"

        assert not algorithm_factory.is_algorithm_available(TestEnum.INVALID)

    def test_factory_requires_injector(self):
        """Test that factory initialization requires dependency injector."""
        from src.core.exceptions import DependencyError
        
        with pytest.raises(DependencyError, match="Injector must be provided"):
            ExecutionAlgorithmFactory(None)

    def test_factory_fallback_creation(self, config):
        """Test factory fallback to direct creation when DI resolution fails."""
        # Create injector that only has Config, no algorithm services
        injector = MagicMock(spec=DependencyInjector)
        def resolve_side_effect(service):
            if service == "Config":
                return config
            # For algorithm services, raise an exception to trigger fallback
            raise Exception(f"Service {service} not found")
            
        injector.resolve.side_effect = resolve_side_effect
        injector.has_service.side_effect = lambda service: service == "Config"
        
        factory = ExecutionAlgorithmFactory(injector)
        
        # Should still work via fallback mechanism
        algorithm = factory.create_algorithm(ExecutionAlgorithm.TWAP)
        assert algorithm is not None
        assert algorithm.__class__.__name__ == "TWAPAlgorithm"
