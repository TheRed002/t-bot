"""
Unit tests for StrategyFactory.

Tests the strategy factory pattern for dynamic strategy instantiation and management.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.exceptions import ValidationError

# Import from P-001
from src.core.types import MarketData, Signal, StrategyStatus
from src.strategies.base import BaseStrategy

# Import from P-011
from src.strategies.factory import StrategyFactory


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""

    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
        return []

    async def validate_signal(self, signal: Signal) -> bool:
        return True

    def get_position_size(self, signal: Signal) -> Decimal:
        return Decimal("0.02")

    def should_exit(self, position, data: MarketData) -> bool:
        return False


class TestStrategyFactory:
    """Test cases for StrategyFactory."""

    @pytest.fixture
    def factory(self):
        """Create strategy factory instance."""
        return StrategyFactory()

    @pytest.fixture
    def mock_config(self):
        """Create mock strategy configuration."""
        return {
            "name": "test_strategy",
            "strategy_type": "static",
            "enabled": True,
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "timeframe": "1h",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {"test_param": "test_value"},
        }

    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager."""
        risk_manager = Mock()
        risk_manager.validate_signal = AsyncMock(return_value=True)
        return risk_manager

    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange."""
        return Mock()

    def test_factory_initialization(self, factory):
        """Test factory initialization."""
        assert isinstance(factory._strategies, dict)
        assert isinstance(factory._strategy_classes, dict)
        assert factory._risk_manager is None
        assert factory._exchange is None

    def test_register_strategy_class(self, factory):
        """Test registering strategy class."""
        factory._register_strategy_class("mock_strategy", MockStrategy)

        assert "mock_strategy" in factory._strategy_classes
        assert factory._strategy_classes["mock_strategy"] == MockStrategy

    def test_register_invalid_strategy_class(self, factory):
        """Test registering invalid strategy class."""

        class InvalidStrategy:
            pass

        with pytest.raises(ValidationError, match="must inherit from BaseStrategy"):
            factory._register_strategy_class("invalid", InvalidStrategy)

    def test_get_strategy_class_registered(self, factory):
        """Test getting registered strategy class."""
        factory._register_strategy_class("mock_strategy", MockStrategy)

        strategy_class = factory._get_strategy_class("mock_strategy")
        assert strategy_class == MockStrategy

    def test_get_strategy_class_not_found(self, factory):
        """Test getting non-existent strategy class."""
        strategy_class = factory._get_strategy_class("non_existent")
        assert strategy_class is None

    @patch("importlib.import_module")
    def test_get_strategy_class_dynamic_import(self, mock_import, factory):
        """Test dynamic import of strategy class."""
        # Mock the import
        mock_module = Mock()
        mock_module.MockStrategy = MockStrategy
        mock_import.return_value = mock_module

        strategy_class = factory._get_strategy_class("mock")

        assert strategy_class == MockStrategy
        mock_import.assert_called_once_with("src.strategies.static.mock")

    def test_create_strategy_success(self, factory, mock_config):
        """Test successful strategy creation."""
        # Register strategy class
        factory._register_strategy_class("test_strategy", MockStrategy)

        # Create strategy
        strategy = factory.create_strategy("test_strategy", mock_config)

        assert isinstance(strategy, MockStrategy)
        assert strategy.name == "test_strategy"
        assert "test_strategy" in factory._strategies

    def test_create_strategy_unknown(self, factory, mock_config):
        """Test creating unknown strategy."""
        with pytest.raises(ValidationError, match="Unknown strategy"):
            factory.create_strategy("unknown_strategy", mock_config)

    def test_create_strategy_invalid_config(self, factory):
        """Test creating strategy with invalid config."""
        factory._register_strategy_class("test_strategy", MockStrategy)

        invalid_config = {"invalid": "config"}

        with pytest.raises(ValidationError):
            factory.create_strategy("test_strategy", invalid_config)

    def test_create_strategy_with_dependencies(
        self, factory, mock_config, mock_risk_manager, mock_exchange
    ):
        """Test creating strategy with dependencies."""
        # Register strategy class
        factory._register_strategy_class("test_strategy", MockStrategy)

        # Set dependencies
        factory.set_risk_manager(mock_risk_manager)
        factory.set_exchange(mock_exchange)

        # Create strategy
        strategy = factory.create_strategy("test_strategy", mock_config)

        assert strategy._risk_manager == mock_risk_manager
        assert strategy._exchange == mock_exchange

    @pytest.mark.asyncio
    async def test_start_strategy_success(self, factory, mock_config):
        """Test successful strategy start."""
        # Register and create strategy
        factory._register_strategy_class("test_strategy", MockStrategy)
        strategy = factory.create_strategy("test_strategy", mock_config)

        # Start strategy
        result = await factory.start_strategy("test_strategy")

        assert result is True
        assert strategy.status == StrategyStatus.RUNNING

    @pytest.mark.asyncio
    async def test_start_strategy_not_found(self, factory):
        """Test starting non-existent strategy."""
        result = await factory.start_strategy("non_existent")
        assert result is False

    @pytest.mark.asyncio
    async def test_start_strategy_error(self, factory, mock_config):
        """Test strategy start with error."""

        # Create strategy that raises error on start
        class ErrorStrategy(MockStrategy):
            async def start(self):
                raise Exception("Start error")

        factory._register_strategy_class("error_strategy", ErrorStrategy)
        strategy = factory.create_strategy("error_strategy", mock_config)

        result = await factory.start_strategy("ErrorStrategy")
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_strategy_success(self, factory, mock_config):
        """Test successful strategy stop."""
        # Register and create strategy
        factory._register_strategy_class("test_strategy", MockStrategy)
        strategy = factory.create_strategy("test_strategy", mock_config)

        # Start then stop strategy
        await factory.start_strategy("test_strategy")
        result = await factory.stop_strategy("test_strategy")

        assert result is True
        assert strategy.status == StrategyStatus.STOPPED

    @pytest.mark.asyncio
    async def test_stop_strategy_not_found(self, factory):
        """Test stopping non-existent strategy."""
        result = await factory.stop_strategy("non_existent")
        assert result is False

    @pytest.mark.asyncio
    async def test_restart_strategy(self, factory, mock_config):
        """Test strategy restart."""
        # Register and create strategy
        factory._register_strategy_class("test_strategy", MockStrategy)
        strategy = factory.create_strategy("test_strategy", mock_config)

        # Start strategy
        await factory.start_strategy("test_strategy")
        assert strategy.status == StrategyStatus.RUNNING

        # Restart strategy
        result = await factory.restart_strategy("test_strategy")

        assert result is True
        assert strategy.status == StrategyStatus.RUNNING

    def test_get_strategy(self, factory, mock_config):
        """Test getting strategy instance."""
        # Register and create strategy
        factory._register_strategy_class("test_strategy", MockStrategy)
        strategy = factory.create_strategy("test_strategy", mock_config)

        # Get strategy
        retrieved_strategy = factory.get_strategy("test_strategy")
        assert retrieved_strategy == strategy

    def test_get_all_strategies(self, factory, mock_config):
        """Test getting all strategies."""
        # Register and create multiple strategies
        factory._register_strategy_class("test_strategy", MockStrategy)
        factory._register_strategy_class("test_strategy2", MockStrategy)
        strategy1 = factory.create_strategy("test_strategy", mock_config)

        mock_config2 = mock_config.copy()
        mock_config2["name"] = "test_strategy2"
        strategy2 = factory.create_strategy("test_strategy2", mock_config2)

        all_strategies = factory.get_all_strategies()

        assert len(all_strategies) == 2
        assert "test_strategy" in all_strategies
        assert "test_strategy2" in all_strategies

    def test_get_strategy_status(self, factory, mock_config):
        """Test getting strategy status."""
        # Register and create strategy
        factory._register_strategy_class("test_strategy", MockStrategy)
        strategy = factory.create_strategy("test_strategy", mock_config)

        status = factory.get_strategy_status("test_strategy")
        assert status == StrategyStatus.STOPPED

    def test_get_strategy_performance(self, factory, mock_config):
        """Test getting strategy performance."""
        # Register and create strategy
        factory._register_strategy_class("test_strategy", MockStrategy)
        strategy = factory.create_strategy("test_strategy", mock_config)

        performance = factory.get_strategy_performance("test_strategy")
        assert performance is not None
        assert "strategy_name" in performance
        assert "status" in performance

    def test_set_risk_manager(self, factory, mock_risk_manager):
        """Test setting risk manager."""
        factory.set_risk_manager(mock_risk_manager)
        assert factory._risk_manager == mock_risk_manager

    def test_set_exchange(self, factory, mock_exchange):
        """Test setting exchange."""
        factory.set_exchange(mock_exchange)
        assert factory._exchange == mock_exchange

    @pytest.mark.asyncio
    async def test_hot_swap_strategy(self, factory, mock_config):
        """Test hot swapping strategy."""
        # Register and create strategy
        factory._register_strategy_class("test_strategy", MockStrategy)
        strategy = factory.create_strategy("test_strategy", mock_config)

        # Start strategy
        await factory.start_strategy("test_strategy")
        assert strategy.status == StrategyStatus.RUNNING

        # Hot swap with new config
        new_config = mock_config.copy()
        new_config["min_confidence"] = 0.8

        result = await factory.hot_swap_strategy("test_strategy", new_config)

        assert result is True
        assert strategy.config.min_confidence == 0.8
        assert strategy.status == StrategyStatus.RUNNING

    @pytest.mark.asyncio
    async def test_hot_swap_strategy_not_found(self, factory):
        """Test hot swapping non-existent strategy."""
        result = await factory.hot_swap_strategy("non_existent", {})
        assert result is False

    def test_remove_strategy(self, factory, mock_config):
        """Test removing strategy."""
        # Register and create strategy
        factory._register_strategy_class("test_strategy", MockStrategy)
        strategy = factory.create_strategy("test_strategy", mock_config)

        # Remove strategy
        result = factory.remove_strategy("test_strategy")

        assert result is True
        assert "test_strategy" not in factory._strategies

    def test_remove_strategy_not_found(self, factory):
        """Test removing non-existent strategy."""
        result = factory.remove_strategy("non_existent")
        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown_all_strategies(self, factory, mock_config):
        """Test shutting down all strategies."""
        # Register and create multiple strategies
        factory._register_strategy_class("test_strategy", MockStrategy)
        strategy1 = factory.create_strategy("test_strategy", mock_config)

        mock_config2 = mock_config.copy()
        mock_config2["name"] = "test_strategy2"
        strategy2 = factory.create_strategy("test_strategy", mock_config2)

        # Start strategies
        await factory.start_strategy("test_strategy")

        # Shutdown all strategies
        await factory.shutdown_all_strategies()

        assert strategy1.status == StrategyStatus.STOPPED
        assert strategy2.status == StrategyStatus.STOPPED

    def test_get_strategy_summary(self, factory, mock_config):
        """Test getting strategy summary."""
        # Register and create strategy
        factory._register_strategy_class("test_strategy", MockStrategy)
        strategy = factory.create_strategy("test_strategy", mock_config)

        summary = factory.get_strategy_summary()

        assert "total_strategies" in summary
        assert "running_strategies" in summary
        assert "stopped_strategies" in summary
        assert "error_strategies" in summary
        assert "strategies" in summary

        assert summary["total_strategies"] == 1
        assert summary["stopped_strategies"] == 1
        assert "test_strategy" in summary["strategies"]

    def test_get_strategy_class_dynamic_import_error(self, factory):
        """Test dynamic import with import error."""
        with patch("importlib.import_module", side_effect=ImportError("Module not found")):
            strategy_class = factory._get_strategy_class("non_existent")
            assert strategy_class is None
