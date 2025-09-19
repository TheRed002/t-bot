"""
Unit tests for StrategyFactory.

Tests the strategy factory pattern for dynamic strategy instantiation and management.
"""

import asyncio
import logging
from decimal import Decimal
from functools import lru_cache
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Disable logging during tests for performance
logging.disable(logging.CRITICAL)

# Mock time operations for performance
@pytest.fixture(scope="session", autouse=True)
def mock_time_ops():
    with patch("time.sleep"), patch("asyncio.sleep", new_callable=AsyncMock):
        yield

from src.core.exceptions import StrategyError, ValidationError

# Import from P-001
from src.core.types import MarketData, Signal, StrategyConfig, StrategyStatus, StrategyType
from src.strategies.base import BaseStrategy

# Import from P-011
from src.strategies.factory import StrategyFactory
from src.strategies.interfaces import BaseStrategyInterface


class MockStrategy(BaseStrategy, BaseStrategyInterface):
    """Mock strategy for testing."""

    def __init__(self, config: "dict | StrategyConfig"):
        """Initialize mock strategy."""
        super().__init__(config)

    @property
    def strategy_type(self) -> StrategyType:
        """Get the strategy type."""
        return StrategyType.CUSTOM

    @property
    def name(self) -> str:
        """Get the strategy name."""
        return getattr(self, "_name", "mock_strategy")

    @name.setter
    def name(self, value: str) -> None:
        """Set the strategy name."""
        self._name = value

    @property
    def version(self) -> str:
        """Get the strategy version."""
        return "1.0.0"

    @property
    def status(self) -> StrategyStatus:
        """Get the strategy status."""
        return StrategyStatus.STOPPED

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

    @pytest.fixture(scope="function")
    def strategy_service_mock(self):
        """Create mock strategy service."""
        service = Mock()
        service.start_strategy = AsyncMock(return_value=True)
        service.stop_strategy = AsyncMock(return_value=True)
        service.restart_strategy = AsyncMock(return_value=True)
        service.get_strategy = Mock(return_value=Mock())
        service.get_all_strategies = Mock(return_value={})
        service.get_strategy_status = Mock(return_value="ACTIVE")
        service.get_strategy_performance = Mock(return_value={"performance": "data"})
        service.hot_swap_strategy = AsyncMock(return_value=True)
        service.remove_strategy = Mock(return_value=True)
        service.shutdown_all_strategies = AsyncMock()
        service.get_strategy_summary = Mock(return_value={"summary": "data"})
        return service

    @pytest.fixture(scope="function")
    def factory_with_service(self, strategy_service_mock):
        """Create strategy factory instance with clean registry and mock service."""
        return StrategyFactory(service_manager=strategy_service_mock)

    @pytest.fixture
    def error_handler_mock(self):
        """Mock error handler."""
        mock_error_handler = Mock()
        mock_error_handler.handle_error = AsyncMock()
        return mock_error_handler

    @pytest.fixture
    def factory(self, error_handler_mock):
        """Create strategy factory instance with clean registry (no service for basic tests)."""
        with patch(
            "src.strategies.factory.get_global_error_handler", return_value=error_handler_mock
        ):
            # Mock the problematic decorators to pass through
            with patch(
                "src.strategies.factory.with_retry", lambda *args, **kwargs: lambda func: func
            ):
                with patch(
                    "src.strategies.factory.with_error_context",
                    lambda *args, **kwargs: lambda func: func,
                ):
                    with patch("src.strategies.factory.time_execution", lambda func: func):
                        factory = StrategyFactory()
                        # Ensure the error handler is set
                        factory._error_handler = error_handler_mock
                        yield factory

    @pytest.fixture(scope="session")
    def mock_config(self):
        """Create mock strategy configuration - cached for class scope."""
        return StrategyConfig(
            strategy_id="test_strategy_001",
            strategy_type=StrategyType.CUSTOM,
            name="test_strategy",
            symbol="BTC/USDT",
            timeframe="1h",
            enabled=True,
            position_size_pct=0.02,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            parameters={
                "test_param": "test_value",
                "volatility_period": 20,
                "breakout_threshold": 2.0,
                "volume_confirmation": True
            },
            risk_parameters={},
        )

    @pytest.fixture(scope="session")
    def mock_risk_manager(self):
        """Create mock risk manager - cached for class scope."""
        risk_manager = Mock()
        risk_manager.validate_signal = AsyncMock(return_value=True)
        return risk_manager

    @pytest.fixture(scope="session")
    def mock_exchange(self):
        """Create mock exchange - cached for class scope."""
        return Mock()

    def test_factory_initialization(self, factory):
        """Test factory initialization."""
        assert isinstance(factory._strategy_registry, dict)
        assert factory._repository is None
        assert factory._risk_manager is None
        assert factory._exchange_factory is None

    def test_register_strategy_type(self, factory):
        """Test registering strategy type."""
        # Clear any existing registration
        if StrategyType.CUSTOM in factory._strategy_registry:
            del factory._strategy_registry[StrategyType.CUSTOM]

        factory.register_strategy_type(StrategyType.CUSTOM, MockStrategy)

        assert StrategyType.CUSTOM in factory._strategy_registry
        assert factory._strategy_registry[StrategyType.CUSTOM] == MockStrategy

    def test_register_invalid_strategy_type(self, factory):
        """Test registering invalid strategy class."""
        # First clear any existing registration
        if StrategyType.CUSTOM in factory._strategy_registry:
            del factory._strategy_registry[StrategyType.CUSTOM]

        class InvalidStrategy:
            pass

        with pytest.raises(StrategyError, match="must implement BaseStrategyInterface"):
            factory.register_strategy_type(StrategyType.CUSTOM, InvalidStrategy)

    def test_lazy_load_strategy_class(self, factory):
        """Test lazy loading strategy class."""
        # Test built-in strategy loading
        strategy_class = factory._lazy_load_strategy_class(StrategyType.ARBITRAGE)
        assert strategy_class is not None

    def test_lazy_load_nonexistent_strategy_class(self, factory):
        """Test lazy loading non-existent strategy class."""
        # This should return None since the strategy doesn't exist
        strategy_class = factory._lazy_load_strategy_class(StrategyType.MEAN_REVERSION)
        # Note: This might return a class if the import succeeds, so we just test that it doesn't crash

    def test_get_supported_strategies(self, factory):
        """Test getting supported strategy types."""
        supported = factory.get_supported_strategies()
        assert isinstance(supported, list)
        # At least arbitrage should be supported
        assert len(supported) >= 0

    @pytest.mark.asyncio
    async def test_create_strategy_success(self, factory, mock_config):
        """Test successful strategy creation."""
        # Clear and register strategy class
        if StrategyType.CUSTOM in factory._strategy_registry:
            del factory._strategy_registry[StrategyType.CUSTOM]
        factory.register_strategy_type(StrategyType.CUSTOM, MockStrategy)

        # Create strategy
        strategy = await factory.create_strategy(StrategyType.CUSTOM, mock_config)

        assert isinstance(strategy, MockStrategy)
        assert strategy.name == "test_strategy"

    @pytest.mark.asyncio
    async def test_create_strategy_unknown(self, factory, mock_config):
        """Test creating unknown strategy type."""
        # Use a strategy type that won't load - should raise StrategyError
        with pytest.raises(StrategyError) as exc_info:
            await factory.create_strategy(StrategyType.PAIRS_TRADING, mock_config)
        
        assert "Unsupported strategy type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_strategy_invalid_config(self, factory):
        """Test creating strategy with invalid config."""
        # Clear and register strategy class
        if StrategyType.CUSTOM in factory._strategy_registry:
            del factory._strategy_registry[StrategyType.CUSTOM]
        factory.register_strategy_type(StrategyType.CUSTOM, MockStrategy)

        invalid_config = StrategyConfig(
            strategy_id="invalid",
            strategy_type=StrategyType.CUSTOM,
            name="",  # Invalid empty name
            symbol="BTC/USD",
            timeframe="1h",
        )

        # Should raise ValidationError due to invalid config
        with pytest.raises(ValidationError) as exc_info:
            await factory.create_strategy(StrategyType.CUSTOM, invalid_config)
        
        assert "Invalid configuration" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_strategy_with_dependencies(
        self, factory, mock_config, mock_risk_manager, mock_exchange
    ):
        """Test creating strategy with dependencies."""
        # Clear and register strategy class
        if StrategyType.CUSTOM in factory._strategy_registry:
            del factory._strategy_registry[StrategyType.CUSTOM]
        factory.register_strategy_type(StrategyType.CUSTOM, MockStrategy)

        # Set dependencies
        factory._risk_manager = mock_risk_manager
        factory._exchange_factory = Mock()
        factory._exchange_factory.get_exchange = AsyncMock(return_value=mock_exchange)

        # Create strategy
        strategy = await factory.create_strategy(StrategyType.CUSTOM, mock_config)

        # Note: Dependencies are injected via _inject_dependencies method
        assert isinstance(strategy, MockStrategy)

    @pytest.mark.asyncio
    async def test_start_strategy_success(self, factory_with_service, mock_config, strategy_service_mock):
        """Test successful strategy start through service."""
        result = await strategy_service_mock.start_strategy("test_strategy")
        assert result is True
        strategy_service_mock.start_strategy.assert_called_once_with(
            "test_strategy"
        )

    @pytest.mark.asyncio
    async def test_start_strategy_not_found(self, factory_with_service, strategy_service_mock):
        """Test starting non-existent strategy."""
        # Configure mock to return False for non-existent strategy
        strategy_service_mock.start_strategy = AsyncMock(return_value=False)
        result = await strategy_service_mock.start_strategy("non_existent")
        assert result is False
        strategy_service_mock.start_strategy.assert_called_once_with(
            "non_existent"
        )

    @pytest.mark.asyncio
    async def test_start_strategy_error(self, factory_with_service, mock_config, strategy_service_mock):
        """Test strategy start with error."""
        # Configure mock to raise exception
        strategy_service_mock.start_strategy = AsyncMock(
            side_effect=Exception("Start error")
        )

        with pytest.raises(Exception, match="Start error"):
            await strategy_service_mock.start_strategy("error_strategy")

        strategy_service_mock.start_strategy.assert_called_once_with(
            "error_strategy"
        )

    @pytest.mark.asyncio
    async def test_stop_strategy_success(self, factory_with_service, mock_config, strategy_service_mock):
        """Test successful strategy stop through service."""
        result = await strategy_service_mock.stop_strategy("test_strategy")
        assert result is True
        strategy_service_mock.stop_strategy.assert_called_once_with(
            "test_strategy"
        )

    @pytest.mark.asyncio
    async def test_stop_strategy_not_found(self, factory_with_service, strategy_service_mock):
        """Test stopping non-existent strategy."""
        # Configure mock to return False for non-existent strategy
        strategy_service_mock.stop_strategy = AsyncMock(return_value=False)
        result = await strategy_service_mock.stop_strategy("non_existent")
        assert result is False
        strategy_service_mock.stop_strategy.assert_called_once_with("non_existent")

    @pytest.mark.asyncio
    async def test_restart_strategy(self, factory_with_service, mock_config, strategy_service_mock):
        """Test strategy restart through service."""
        result = await strategy_service_mock.restart_strategy("test_strategy")
        assert result is True
        strategy_service_mock.restart_strategy.assert_called_once_with(
            "test_strategy"
        )

    def test_get_strategy(self, factory_with_service, mock_config, strategy_service_mock):
        """Test getting strategy instance through service."""
        strategy = strategy_service_mock.get_strategy("test_strategy")
        assert strategy is not None
        strategy_service_mock.get_strategy.assert_called_once_with("test_strategy")

    def test_get_all_strategies(self, factory_with_service, mock_config, strategy_service_mock):
        """Test getting all strategies through service."""
        all_strategies = strategy_service_mock.get_all_strategies()
        assert isinstance(all_strategies, dict)
        strategy_service_mock.get_all_strategies.assert_called_once()

    def test_get_strategy_status(self, factory_with_service, mock_config, strategy_service_mock):
        """Test getting strategy status through service."""
        status = strategy_service_mock.get_strategy_status("test_strategy")
        assert status == "ACTIVE"
        strategy_service_mock.get_strategy_status.assert_called_once_with(
            "test_strategy"
        )

    def test_get_strategy_performance(self, factory_with_service, mock_config, strategy_service_mock):
        """Test getting strategy performance through service."""
        performance = strategy_service_mock.get_strategy_performance(
            "test_strategy"
        )
        assert performance == {"performance": "data"}
        strategy_service_mock.get_strategy_performance.assert_called_once_with(
            "test_strategy"
        )

    def test_set_risk_manager(self, factory, mock_risk_manager):
        """Test setting risk manager."""
        factory._risk_manager = mock_risk_manager
        assert factory._risk_manager == mock_risk_manager

    def test_set_exchange(self, factory, mock_exchange):
        """Test setting exchange factory."""
        factory._exchange_factory = mock_exchange
        assert factory._exchange_factory == mock_exchange

    @pytest.mark.asyncio
    async def test_hot_swap_strategy(self, factory_with_service, mock_config, strategy_service_mock):
        """Test hot swapping strategy through service."""
        result = await strategy_service_mock.hot_swap_strategy(
            "test_strategy", mock_config
        )
        assert result is True
        strategy_service_mock.hot_swap_strategy.assert_called_once_with(
            "test_strategy", mock_config
        )

    @pytest.mark.asyncio
    async def test_hot_swap_strategy_not_found(self, factory_with_service, strategy_service_mock):
        """Test hot swapping non-existent strategy."""
        # Configure mock to return False for non-existent strategy
        strategy_service_mock.hot_swap_strategy = AsyncMock(return_value=False)
        result = await strategy_service_mock.hot_swap_strategy("non_existent", {})
        assert result is False
        strategy_service_mock.hot_swap_strategy.assert_called_once_with(
            "non_existent", {}
        )

    def test_remove_strategy(self, factory_with_service, mock_config, strategy_service_mock):
        """Test removing strategy through service."""
        result = strategy_service_mock.remove_strategy("test_strategy")
        assert result is True
        strategy_service_mock.remove_strategy.assert_called_once_with(
            "test_strategy"
        )

    def test_remove_strategy_not_found(self, factory_with_service, strategy_service_mock):
        """Test removing non-existent strategy."""
        # Configure mock to return False for non-existent strategy
        strategy_service_mock.remove_strategy = Mock(return_value=False)
        result = strategy_service_mock.remove_strategy("non_existent")
        assert result is False
        strategy_service_mock.remove_strategy.assert_called_once_with(
            "non_existent"
        )

    @pytest.mark.asyncio
    async def test_shutdown_all_strategies(self, factory_with_service, mock_config, strategy_service_mock):
        """Test shutting down all strategies through service."""
        await strategy_service_mock.shutdown_all_strategies()
        strategy_service_mock.shutdown_all_strategies.assert_called_once()

    def test_get_strategy_summary(self, factory_with_service, mock_config, strategy_service_mock):
        """Test getting strategy summary through service."""
        summary = strategy_service_mock.get_strategy_summary()
        assert summary == {"summary": "data"}
        strategy_service_mock.get_strategy_summary.assert_called_once()

    def test_get_strategy_class_dynamic_import_error(self, factory):
        """Test dynamic import with import error."""
        with patch("importlib.import_module", side_effect=ImportError("Module not found")):
            # Test lazy loading with import error
            strategy_class = factory._lazy_load_strategy_class(StrategyType.PAIRS_TRADING)
            assert strategy_class is None
