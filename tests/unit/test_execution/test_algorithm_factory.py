"""Unit tests for AlgorithmFactory."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from decimal import Decimal
from datetime import datetime, timezone

from src.core.config import Config
from src.core.types import ExecutionAlgorithm
from src.execution.algorithm_factory import ExecutionAlgorithmFactory
from src.execution.types import ExecutionInstruction


class TestAlgorithmFactory:
    """Test cases for AlgorithmFactory."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = MagicMock(spec=Config)
        config.execution = MagicMock()
        config.execution.get = MagicMock(return_value="100000")  # Return string for Decimal conversion
        return config

    @pytest.fixture
    def algorithm_factory(self, config):
        """Create ExecutionAlgorithmFactory instance."""
        return ExecutionAlgorithmFactory(config)

    @pytest.fixture 
    def sample_execution_instruction(self):
        """Create sample execution instruction."""
        from src.core.types import OrderRequest, OrderSide, OrderType
        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=Decimal("50000")
        )
        return ExecutionInstruction(
            order=order,
            algorithm=ExecutionAlgorithm.TWAP,
            strategy_name="test_strategy",
            time_horizon_minutes=60,
            max_slices=10
        )

    def test_initialization(self, algorithm_factory, config):
        """Test ExecutionAlgorithmFactory initialization."""
        assert algorithm_factory.config == config
        assert hasattr(algorithm_factory, '_algorithm_registry')
        assert isinstance(algorithm_factory._algorithm_registry, dict)

    def test_create_algorithm_twap(self, algorithm_factory, sample_execution_instruction):
        """Test creating TWAP algorithm."""
        algorithm = algorithm_factory.create_algorithm(ExecutionAlgorithm.TWAP)
        assert algorithm is not None
        assert algorithm.__class__.__name__ == "TWAPAlgorithm"

    def test_create_algorithm_vwap(self, algorithm_factory, sample_execution_instruction):
        """Test creating VWAP algorithm.""" 
        algorithm = algorithm_factory.create_algorithm(ExecutionAlgorithm.VWAP)
        assert algorithm is not None
        assert algorithm.__class__.__name__ == "VWAPAlgorithm"

    def test_create_algorithm_iceberg(self, algorithm_factory, sample_execution_instruction):
        """Test creating Iceberg algorithm."""
        algorithm = algorithm_factory.create_algorithm(ExecutionAlgorithm.ICEBERG) 
        assert algorithm is not None
        assert algorithm.__class__.__name__ == "IcebergAlgorithm"

    def test_create_algorithm_smart_router(self, algorithm_factory, sample_execution_instruction):
        """Test creating Smart Router algorithm."""
        algorithm = algorithm_factory.create_algorithm(ExecutionAlgorithm.SMART_ROUTER)
        assert algorithm is not None
        assert algorithm.__class__.__name__ == "SmartOrderRouter"

    def test_create_algorithm_invalid(self, algorithm_factory):
        """Test creating invalid algorithm raises error."""
        from src.core.exceptions import ValidationError
        with pytest.raises(ValidationError, match="Unsupported algorithm"):
            algorithm_factory.create_algorithm("invalid_algorithm")

    def test_create_algorithm_with_config(self, algorithm_factory):
        """Test algorithm creation uses config settings."""
        # Test that factory passes config to algorithms
        algorithm = algorithm_factory.create_algorithm(ExecutionAlgorithm.TWAP)
        assert algorithm.config == algorithm_factory.config

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