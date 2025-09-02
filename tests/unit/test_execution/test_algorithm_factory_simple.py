"""Simple unit tests for ExecutionAlgorithmFactory."""

import pytest
from unittest.mock import MagicMock
from src.core.config import Config
from src.core.types import ExecutionAlgorithm
from src.execution.algorithm_factory import ExecutionAlgorithmFactory


class TestExecutionAlgorithmFactory:
    """Test cases for ExecutionAlgorithmFactory."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = MagicMock(spec=Config)
        config.execution = MagicMock()
        # Return proper default values instead of empty dict
        def mock_get(key, default=None):
            defaults = {
                "default_portfolio_value": "100000",
                "default_time_horizon_minutes": 60,
                "default_participation_rate": 0.2,
                "min_slice_size_pct": 0.01,
                "max_slices": 100
            }
            return defaults.get(key, default)
        config.execution.get = MagicMock(side_effect=mock_get)
        return config

    @pytest.fixture
    def algorithm_factory(self, config):
        """Create ExecutionAlgorithmFactory instance."""
        return ExecutionAlgorithmFactory(config)

    def test_initialization(self, algorithm_factory, config):
        """Test ExecutionAlgorithmFactory initialization."""
        assert algorithm_factory.config == config

    def test_create_algorithm_twap(self, algorithm_factory):
        """Test creating TWAP algorithm."""
        algorithm = algorithm_factory.create_algorithm(ExecutionAlgorithm.TWAP)
        assert algorithm is not None
        assert algorithm.__class__.__name__ == "TWAPAlgorithm"

    def test_create_algorithm_vwap(self, algorithm_factory):
        """Test creating VWAP algorithm."""
        algorithm = algorithm_factory.create_algorithm(ExecutionAlgorithm.VWAP)
        assert algorithm is not None
        assert algorithm.__class__.__name__ == "VWAPAlgorithm"

    def test_create_algorithm_iceberg(self, algorithm_factory):
        """Test creating Iceberg algorithm."""
        algorithm = algorithm_factory.create_algorithm(ExecutionAlgorithm.ICEBERG)
        assert algorithm is not None
        assert algorithm.__class__.__name__ == "IcebergAlgorithm"

    def test_create_algorithm_smart_router(self, algorithm_factory):
        """Test creating Smart Router algorithm."""
        algorithm = algorithm_factory.create_algorithm(ExecutionAlgorithm.SMART_ROUTER)
        assert algorithm is not None
        assert algorithm.__class__.__name__ == "SmartOrderRouter"

    def test_create_algorithm_invalid(self, algorithm_factory):
        """Test creating invalid algorithm raises error."""
        from src.core.exceptions import ValidationError
        with pytest.raises(ValidationError):
            algorithm_factory.create_algorithm("invalid_algorithm")

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
        assert not algorithm_factory.is_algorithm_available("invalid_algorithm")

    def test_algorithm_creation_uses_config(self, algorithm_factory):
        """Test algorithm creation uses config."""
        algorithm = algorithm_factory.create_algorithm(ExecutionAlgorithm.TWAP)
        assert algorithm.config == algorithm_factory.config