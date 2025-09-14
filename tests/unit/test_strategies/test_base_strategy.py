"""
Unit tests for BaseStrategy interface.

Tests the abstract base strategy interface and its core functionality.
"""

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from functools import lru_cache

import pytest

# Disable logging during tests for performance
logging.disable(logging.CRITICAL)

# Fast mock time for deterministic tests
FIXED_TIME = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# Import from P-001
from src.core.types import (
    MarketData,
    Position,
    PositionSide,
    PositionStatus,
    Signal,
    SignalDirection,
    StrategyMetrics,
    StrategyStatus,
    StrategyType,
)

# Pre-computed signal objects for performance
VALID_SIGNAL = Signal(
    direction=SignalDirection.BUY,
    strength=Decimal("0.8"),
    timestamp=FIXED_TIME,
    symbol="BTC/USDT",
    source="test",
    metadata={},
)

INVALID_SIGNAL = Signal(
    direction=SignalDirection.BUY,
    strength=Decimal("0.5"),
    timestamp=FIXED_TIME,
    symbol="BTC/USDT",
    source="test",
    metadata={},
)

# Pre-computed mock objects for performance
MOCK_TRADE_RESULT_POSITIVE = Mock(pnl=Decimal("10.50"))
MOCK_TRADE_RESULT_NEGATIVE = Mock(pnl=Decimal("-5.25"))


# Cached mock factory functions for performance
@lru_cache(maxsize=2)
def create_passing_risk_manager():
    return Mock(validate_signal=AsyncMock(return_value=True))

@lru_cache(maxsize=2)
def create_failing_risk_manager():
    return Mock(validate_signal=AsyncMock(return_value=False))


MOCK_EXCHANGE = Mock()

# Import from P-011
from src.strategies.base import BaseStrategy


class MockStrategy(BaseStrategy):
    """Mock strategy implementation for testing."""

    def __init__(self, config: dict):
        """Initialize mock strategy."""
        super().__init__(config)

    @property
    def strategy_type(self) -> StrategyType:
        """Return the strategy type."""
        return StrategyType.CUSTOM

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return self._name

    @property
    def version(self) -> str:
        """Return the strategy version."""
        return "1.0.0"

    @property
    def status(self) -> StrategyStatus:
        """Return the strategy status."""
        return self._status if hasattr(self, "_status") else StrategyStatus.STOPPED

    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
        """Generate mock signals."""
        try:
            if not data or data.close <= 0:
                return []

            # Create a mock signal
            signal = Signal(
                symbol=data.symbol,
                direction=SignalDirection.BUY,
                strength=Decimal("0.8"),
                timestamp=FIXED_TIME,
                source=self.name,
                metadata={"test": True},
            )

            return [signal]
        except Exception:
            # Return empty list on any error to avoid None return
            return []

    async def validate_signal(self, signal: Signal) -> bool:
        """Validate mock signal."""
        min_confidence = self.config.parameters.get("min_confidence", 0.6)
        return signal.strength >= min_confidence

    def get_position_size(self, signal: Signal) -> Decimal:
        """Get mock position size."""
        # StrategyConfig doesn't have position_size_pct attribute
        # Get from parameters or use default
        position_size = self.config.parameters.get("position_size_pct", 0.02)
        return Decimal(str(position_size))

    def should_exit(self, position: Position, data: MarketData) -> bool:
        """Mock exit condition."""
        return position.unrealized_pnl < -Decimal("0.01")


class TestBaseStrategy:
    """Test cases for BaseStrategy interface."""

    @pytest.fixture(scope="session", autouse=True)
    def setup_performance_optimizations(self):
        """Setup performance optimizations for all tests."""
        # Pre-create common objects for reuse
        self._cached_market_data = MarketData(
            symbol="BTC/USDT",
            timestamp=FIXED_TIME,
            open=Decimal("49000.0"),
            high=Decimal("51000.0"),
            low=Decimal("48000.0"),
            close=Decimal("50000.0"),
            volume=Decimal("1000.0"),
            exchange="binance"
        )
        
    @pytest.fixture(scope="session")
    def mock_config(self):
        """Create mock strategy configuration - cached for session scope."""
        return {
            "name": "test_strategy",
            "strategy_id": "test_strategy_001",
            "strategy_type": StrategyType.MEAN_REVERSION,  # Use actual enum
            "symbol": "BTC/USDT",
            "enabled": True,
            "timeframe": "1h",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {
                "test_param": "test_value",
            },
        }

    @pytest.fixture(scope="session")
    def mock_strategy(self, mock_config):
        """Create mock strategy instance - cached for session scope."""
        return MockStrategy(mock_config)

    @pytest.fixture(scope="function")
    def fresh_strategy(self, mock_config):
        """Create fresh strategy instance for each test."""
        return MockStrategy(mock_config)

    @pytest.fixture(scope="session")
    def mock_market_data(self):
        """Create mock market data - cached for session scope with fixed time."""
        return MarketData(
            symbol="BTC/USDT",  # Use proper symbol format
            open=Decimal("49900"),
            high=Decimal("50100"),
            low=Decimal("49800"),
            close=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=FIXED_TIME,  # Use fixed time for performance
            exchange="binance",
            bid_price=Decimal("49999"),
            ask_price=Decimal("50001"),
        )

    @pytest.fixture(scope="session")
    def mock_position(self):
        """Create mock position - cached for session scope with fixed time."""
        return Position(
            symbol="BTC/USDT",  # Use proper symbol format
            side=PositionSide.LONG,  # Use PositionSide, not OrderSide
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50100"),
            unrealized_pnl=Decimal("10"),
            status=PositionStatus.OPEN,  # Required field
            opened_at=FIXED_TIME,  # Use fixed time for performance
            exchange="binance",  # Required field
        )

    def test_strategy_initialization(self, mock_strategy, mock_config):
        """Test strategy initialization."""
        assert mock_strategy.name == "test_strategy"  # Now uses config name
        assert mock_strategy.version == "1.0.0"
        assert mock_strategy.status == StrategyStatus.STOPPED
        assert isinstance(mock_strategy.metrics, StrategyMetrics)
        assert mock_strategy.config.name == mock_config["name"]
        assert mock_strategy.config.strategy_type == mock_config["strategy_type"]

    def test_strategy_info(self, mock_strategy):
        """Test getting strategy information."""
        info = mock_strategy.get_strategy_info()

        assert "name" in info
        assert "version" in info
        assert "status" in info
        assert "config" in info
        assert "metrics" in info

        assert info["name"] == "test_strategy"  # Now uses config name
        assert info["status"] == StrategyStatus.STOPPED.value

    @pytest.mark.asyncio
    async def test_generate_signals(self, fresh_strategy, mock_market_data):
        """Test signal generation."""
        signals = await fresh_strategy._generate_signals_impl(mock_market_data)

        # Batch assertions for better performance
        assert isinstance(signals, list) and len(signals) == 1
        signal = signals[0]
        assert isinstance(signal, Signal) and signal.direction == SignalDirection.BUY
        assert signal.strength == Decimal("0.8") and signal.symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_generate_signals_empty_data(self, fresh_strategy):
        """Test signal generation with empty data."""
        # Test both None and zero data cases
        none_signals, zero_signals = await asyncio.gather(
            fresh_strategy.generate_signals(None),
            fresh_strategy.generate_signals(MarketData(
                symbol="BTC/USDT", open=Decimal("0"), high=Decimal("0"),
                low=Decimal("0"), close=Decimal("0"), volume=Decimal("0"),
                timestamp=FIXED_TIME, exchange="binance",
                bid_price=Decimal("0"), ask_price=Decimal("0")
            ))
        )
        assert none_signals == [] and zero_signals == []

    @pytest.mark.asyncio
    async def test_validate_signal(self, fresh_strategy):
        """Test signal validation."""

        # Use pre-computed signals for performance
        assert await fresh_strategy.validate_signal(VALID_SIGNAL) is True
        assert await fresh_strategy.validate_signal(INVALID_SIGNAL) is False

    def test_get_position_size(self, mock_strategy):
        """Test position size calculation."""
        # Use pre-computed signal for performance
        position_size = mock_strategy.get_position_size(VALID_SIGNAL)
        assert isinstance(position_size, Decimal)
        assert position_size == Decimal("0.02")

    def test_should_exit(self, mock_strategy, mock_position, mock_market_data):
        """Test exit condition checking."""
        # Position with positive P&L
        assert mock_strategy.should_exit(mock_position, mock_market_data) is False

        # Position with negative P&L
        negative_position = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("49900"),
            unrealized_pnl=Decimal("-10"),
            status=PositionStatus.OPEN,
            opened_at=FIXED_TIME,
            exchange="binance",
        )
        assert mock_strategy.should_exit(negative_position, mock_market_data) is True

    @pytest.mark.asyncio
    async def test_pre_trade_validation(self, fresh_strategy):
        """Test pre-trade validation."""

        # Use pre-computed signals for performance
        assert await fresh_strategy.pre_trade_validation(VALID_SIGNAL) is True
        assert await fresh_strategy.pre_trade_validation(INVALID_SIGNAL) is False

    @pytest.mark.asyncio
    async def test_pre_trade_validation_with_risk_manager(self, fresh_strategy):
        """Test pre-trade validation with risk manager."""

        # Use fresh mock for performance
        mock_risk_manager = create_passing_risk_manager()
        fresh_strategy.set_risk_manager(mock_risk_manager)

        assert await fresh_strategy.pre_trade_validation(VALID_SIGNAL) is True
        mock_risk_manager.validate_signal.assert_called_once_with(VALID_SIGNAL)

    @pytest.mark.asyncio
    async def test_pre_trade_validation_with_risk_manager_failure(self, fresh_strategy):
        """Test pre-trade validation with risk manager failure."""

        # Use fresh mock for performance
        mock_risk_manager = create_failing_risk_manager()
        fresh_strategy.set_risk_manager(mock_risk_manager)

        assert await fresh_strategy.pre_trade_validation(VALID_SIGNAL) is False
        mock_risk_manager.validate_signal.assert_called_once_with(VALID_SIGNAL)

    def test_set_risk_manager(self, fresh_strategy):
        """Test setting risk manager."""
        mock_risk_manager = create_passing_risk_manager()
        fresh_strategy.set_risk_manager(mock_risk_manager)
        # Risk manager is now stored in services container
        assert fresh_strategy.services.risk_service == mock_risk_manager

    def test_set_exchange(self, fresh_strategy):
        """Test setting exchange."""
        fresh_strategy.set_exchange(MOCK_EXCHANGE)
        # Exchange is stored for backward compatibility
        assert fresh_strategy._exchange == MOCK_EXCHANGE

    @pytest.mark.asyncio
    async def test_strategy_lifecycle(self, fresh_strategy):
        """Test strategy lifecycle methods."""

        # Test start
        await fresh_strategy.start()
        assert fresh_strategy.status == StrategyStatus.ACTIVE

        # Test pause
        await fresh_strategy.pause()
        assert fresh_strategy.status == StrategyStatus.PAUSED

        # Test resume
        await fresh_strategy.resume()
        assert fresh_strategy.status == StrategyStatus.ACTIVE

        # Test stop
        await fresh_strategy.stop()
        assert fresh_strategy.status == StrategyStatus.STOPPED

    @pytest.mark.asyncio
    async def test_strategy_start_error(self, fresh_strategy):
        """Test strategy start with error."""

        # Mock _on_start to raise exception
        async def mock_on_start():
            raise Exception("Start error")

        fresh_strategy._on_start = mock_on_start

        result = await fresh_strategy.start()
        
        assert result is False  # Should return False on error
        assert fresh_strategy.status == StrategyStatus.ERROR

    @pytest.mark.asyncio
    async def test_strategy_stop_error(self, fresh_strategy):
        """Test strategy stop with error."""

        # Mock _on_stop to raise exception
        async def mock_on_stop():
            raise Exception("Stop error")

        fresh_strategy._on_stop = mock_on_stop

        result = await fresh_strategy.stop()
        
        assert result is False  # Should return False on error
        # Status should remain STOPPED even with error
        assert fresh_strategy.status == StrategyStatus.STOPPED

    def test_update_config(self, fresh_strategy):
        """Test configuration update."""

        # Pre-built config for performance
        new_config = {
            "name": "updated_strategy",
            "strategy_id": "updated_strategy_001",
            "strategy_type": StrategyType.CUSTOM,  # Use valid enum
            "symbol": "BTC/USDT",  # Required field
            "enabled": True,
            "timeframe": "5m",
            "min_confidence": 0.7,
            "max_positions": 3,
            "position_size_pct": 0.03,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06,
            "parameters": {
                "updated_param": "updated_value",
            },
        }

        fresh_strategy.update_config(new_config)

        assert fresh_strategy.config.name == "updated_strategy"
        assert fresh_strategy.config.min_confidence == 0.7
        assert fresh_strategy.config.max_positions == 3

    def test_get_performance_summary(self, fresh_strategy):
        """Test performance summary generation."""

        # Update metrics directly for performance
        fresh_strategy.metrics.total_trades = 10
        fresh_strategy.metrics.winning_trades = 7
        fresh_strategy.metrics.losing_trades = 3
        fresh_strategy.metrics.total_pnl = Decimal("100.50")
        fresh_strategy.metrics.win_rate = 0.7
        fresh_strategy.metrics.sharpe_ratio = 1.2
        fresh_strategy.metrics.max_drawdown = 0.05

        summary = fresh_strategy.get_performance_summary()

        # Now uses config name
        assert summary["strategy_name"] == "test_strategy"
        assert summary["status"] == StrategyStatus.STOPPED.value
        assert summary["total_trades"] == 10
        assert summary["winning_trades"] == 7
        assert summary["losing_trades"] == 3
        assert summary["win_rate"] == 0.7
        assert summary["total_pnl"] == 100.50
        assert summary["sharpe_ratio"] == 1.2
        assert summary["max_drawdown"] == 0.05

    @pytest.mark.asyncio
    async def test_post_trade_processing(self, fresh_strategy):
        """Test post-trade processing."""
        initial_trades = fresh_strategy.metrics.total_trades

        # Use pre-computed mock for performance
        await fresh_strategy.post_trade_processing(MOCK_TRADE_RESULT_POSITIVE)

        assert fresh_strategy.metrics.total_trades == initial_trades + 1
        assert fresh_strategy.metrics.winning_trades == 1
        assert fresh_strategy.metrics.total_pnl == Decimal("10.50")
        assert fresh_strategy.metrics.win_rate == 1.0

    @pytest.mark.asyncio
    async def test_post_trade_processing_negative_pnl(self, fresh_strategy):
        """Test post-trade processing with negative PnL."""
        initial_trades = fresh_strategy.metrics.total_trades

        # Use pre-computed mock for performance
        await fresh_strategy.post_trade_processing(MOCK_TRADE_RESULT_NEGATIVE)

        assert fresh_strategy.metrics.total_trades == initial_trades + 1
        assert fresh_strategy.metrics.losing_trades == 1
        assert fresh_strategy.metrics.total_pnl == Decimal("-5.25")
        assert fresh_strategy.metrics.win_rate == 0.0

    def test_abstract_methods_required(self):
        """Test that concrete strategies must implement abstract methods."""

        class IncompleteStrategy(BaseStrategy):
            """Incomplete strategy implementation."""

            pass

        with pytest.raises(TypeError):
            IncompleteStrategy({})
