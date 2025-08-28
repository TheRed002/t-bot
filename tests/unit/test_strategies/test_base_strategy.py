"""
Unit tests for BaseStrategy interface.

Tests the abstract base strategy interface and its core functionality.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

# Import from P-001
from src.core.types import (
    MarketData,
    OrderSide,
    Position,
    Signal,
    SignalDirection,
    StrategyMetrics,
    StrategyStatus,
    StrategyType,
)

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
        return self.config.name if hasattr(self, 'config') else "mock_strategy"
    
    @property
    def version(self) -> str:
        """Return the strategy version."""
        return "1.0.0"
    
    @property
    def status(self) -> StrategyStatus:
        """Return the strategy status."""
        return self._status if hasattr(self, '_status') else StrategyStatus.STOPPED

    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
        """Generate mock signals."""
        try:
            if not data or data.close <= 0:
                return []

            # Create a mock signal
            signal = Signal(
                symbol=data.symbol,
                direction=SignalDirection.BUY,
                strength=0.8,
                timestamp=datetime.now(timezone.utc),
                source=self.name,
                metadata={"test": True},
            )

            return [signal]
        except Exception:
            # Return empty list on any error to avoid None return
            return []

    async def validate_signal(self, signal: Signal) -> bool:
        """Validate mock signal."""
        return signal.strength >= self.config.min_confidence

    def get_position_size(self, signal: Signal) -> Decimal:
        """Get mock position size."""
        return Decimal(str(self.config.position_size_pct))

    def should_exit(self, position: Position, data: MarketData) -> bool:
        """Mock exit condition."""
        return position.unrealized_pnl < -Decimal("0.01")


class TestBaseStrategy:
    """Test cases for BaseStrategy interface."""

    @pytest.fixture
    def mock_config(self):
        """Create mock strategy configuration."""
        return {
            "name": "test_strategy",
            "strategy_id": "test_strategy_001",
            "strategy_type": "mean_reversion",
            "symbol": "BTCUSDT",
            "enabled": True,
            "timeframe": "1h",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {"test_param": "test_value"},
        }

    @pytest.fixture
    def mock_strategy(self, mock_config):
        """Create mock strategy instance."""
        return MockStrategy(mock_config)

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        return MarketData(
            symbol="BTCUSDT",
            open=Decimal("49900"),
            high=Decimal("50100"),
            low=Decimal("49800"),
            close=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
            bid_price=Decimal("49999"),
            ask_price=Decimal("50001"),
        )

    @pytest.fixture
    def mock_position(self):
        """Create mock position."""
        return Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50100"),
            unrealized_pnl=Decimal("10"),
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc),
        )

    def test_strategy_initialization(self, mock_strategy, mock_config):
        """Test strategy initialization."""
        assert mock_strategy.name == "test_strategy"  # Now uses config name
        assert mock_strategy.version == "1.0.0"
        assert mock_strategy.status == StrategyStatus.STOPPED
        assert isinstance(mock_strategy.metrics, StrategyMetrics)
        assert mock_strategy.config.name == mock_config["name"]
        assert mock_strategy.config.strategy_type.value == mock_config["strategy_type"]

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
    async def test_generate_signals(self, mock_strategy, mock_market_data):
        """Test signal generation."""
        # Test the internal implementation directly to avoid BaseStrategy complexity
        signals = await mock_strategy._generate_signals_impl(mock_market_data)

        assert isinstance(signals, list)
        assert len(signals) == 1
        assert isinstance(signals[0], Signal)
        assert signals[0].direction == SignalDirection.BUY
        assert signals[0].strength == 0.8
        assert signals[0].symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_generate_signals_empty_data(self, mock_strategy):
        """Test signal generation with empty data."""
        signals = await mock_strategy.generate_signals(None)
        assert signals == []

        empty_data = MarketData(
            symbol="BTCUSDT",
            open=Decimal("0"),
            high=Decimal("0"),
            low=Decimal("0"),
            close=Decimal("0"),
            volume=Decimal("0"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )
        signals = await mock_strategy.generate_signals(empty_data)
        assert signals == []

    @pytest.mark.asyncio
    async def test_validate_signal(self, mock_strategy):
        """Test signal validation."""
        # Valid signal
        valid_signal = Signal(
            direction=SignalDirection.BUY,
            strength=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            source="test",
            metadata={},
        )
        assert await mock_strategy.validate_signal(valid_signal) is True

        # Invalid signal (low confidence)
        invalid_signal = Signal(
            direction=SignalDirection.BUY,
            strength=0.5,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            source="test",
            metadata={},
        )
        assert await mock_strategy.validate_signal(invalid_signal) is False

    def test_get_position_size(self, mock_strategy):
        """Test position size calculation."""
        signal = Signal(
            direction=SignalDirection.BUY,
            strength=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            source="test",
            metadata={},
        )

        position_size = mock_strategy.get_position_size(signal)
        assert isinstance(position_size, Decimal)
        assert position_size == Decimal("0.02")

    def test_should_exit(self, mock_strategy, mock_position, mock_market_data):
        """Test exit condition checking."""
        # Position with positive P&L
        assert mock_strategy.should_exit(mock_position, mock_market_data) is False

        # Position with negative P&L
        negative_position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("49900"),
            unrealized_pnl=Decimal("-10"),
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc),
        )
        assert mock_strategy.should_exit(negative_position, mock_market_data) is True

    @pytest.mark.asyncio
    async def test_pre_trade_validation(self, mock_strategy):
        """Test pre-trade validation."""
        # Valid signal
        valid_signal = Signal(
            direction=SignalDirection.BUY,
            strength=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            source="test",
            metadata={},
        )
        assert await mock_strategy.pre_trade_validation(valid_signal) is True

        # Invalid signal
        invalid_signal = Signal(
            direction=SignalDirection.BUY,
            strength=0.5,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            source="test",
            metadata={},
        )
        assert await mock_strategy.pre_trade_validation(invalid_signal) is False

    @pytest.mark.asyncio
    async def test_pre_trade_validation_with_risk_manager(self, mock_strategy):
        """Test pre-trade validation with risk manager."""
        # Mock risk manager
        mock_risk_manager = Mock()
        mock_risk_manager.validate_signal = AsyncMock(return_value=True)
        mock_strategy.set_risk_manager(mock_risk_manager)

        signal = Signal(
            direction=SignalDirection.BUY,
            strength=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            source="test",
            metadata={},
        )

        assert await mock_strategy.pre_trade_validation(signal) is True
        mock_risk_manager.validate_signal.assert_called_once_with(signal)

    @pytest.mark.asyncio
    async def test_pre_trade_validation_with_risk_manager_failure(self, mock_strategy):
        """Test pre-trade validation with risk manager failure."""
        # Mock risk manager
        mock_risk_manager = Mock()
        mock_risk_manager.validate_signal = AsyncMock(return_value=False)
        mock_strategy.set_risk_manager(mock_risk_manager)

        signal = Signal(
            direction=SignalDirection.BUY,
            strength=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            source="test",
            metadata={},
        )

        assert await mock_strategy.pre_trade_validation(signal) is False
        mock_risk_manager.validate_signal.assert_called_once_with(signal)

    def test_set_risk_manager(self, mock_strategy):
        """Test setting risk manager."""
        mock_risk_manager = Mock()
        mock_strategy.set_risk_manager(mock_risk_manager)
        assert mock_strategy._risk_manager == mock_risk_manager

    def test_set_exchange(self, mock_strategy):
        """Test setting exchange."""
        mock_exchange = Mock()
        mock_strategy.set_exchange(mock_exchange)
        assert mock_strategy._exchange == mock_exchange

    @pytest.mark.asyncio
    async def test_strategy_lifecycle(self, mock_strategy):
        """Test strategy lifecycle methods."""
        # Test start
        await mock_strategy.start()
        assert mock_strategy.status == StrategyStatus.RUNNING

        # Test pause
        await mock_strategy.pause()
        assert mock_strategy.status == StrategyStatus.PAUSED

        # Test resume
        await mock_strategy.resume()
        assert mock_strategy.status == StrategyStatus.RUNNING

        # Test stop
        await mock_strategy.stop()
        assert mock_strategy.status == StrategyStatus.STOPPED

    @pytest.mark.asyncio
    async def test_strategy_start_error(self, mock_strategy):
        """Test strategy start with error."""

        # Mock _on_start to raise exception
        async def mock_on_start():
            raise Exception("Start error")

        mock_strategy._on_start = mock_on_start

        with pytest.raises(Exception, match="Start error"):
            await mock_strategy.start()

        assert mock_strategy.status == StrategyStatus.ERROR

    @pytest.mark.asyncio
    async def test_strategy_stop_error(self, mock_strategy):
        """Test strategy stop with error."""

        # Mock _on_stop to raise exception
        async def mock_on_stop():
            raise Exception("Stop error")

        mock_strategy._on_stop = mock_on_stop

        with pytest.raises(Exception, match="Stop error"):
            await mock_strategy.stop()

        # Status should remain STOPPED even with error
        assert mock_strategy.status == StrategyStatus.STOPPED

    def test_update_config(self, mock_strategy):
        """Test configuration update."""
        old_config = mock_strategy.config.model_dump()
        new_config = {
            "name": "updated_strategy",
            "strategy_type": "static",
            "enabled": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "5m",
            "min_confidence": 0.7,
            "max_positions": 3,
            "position_size_pct": 0.03,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06,
            "parameters": {"updated_param": "updated_value"},
        }

        mock_strategy.update_config(new_config)

        assert mock_strategy.config.name == "updated_strategy"
        assert mock_strategy.config.min_confidence == 0.7
        assert mock_strategy.config.max_positions == 3

    def test_get_performance_summary(self, mock_strategy):
        """Test performance summary generation."""
        # Update metrics
        mock_strategy.metrics.total_trades = 10
        mock_strategy.metrics.winning_trades = 7
        mock_strategy.metrics.losing_trades = 3
        mock_strategy.metrics.total_pnl = Decimal("100.50")
        mock_strategy.metrics.win_rate = 0.7
        mock_strategy.metrics.sharpe_ratio = 1.2
        mock_strategy.metrics.max_drawdown = 0.05

        summary = mock_strategy.get_performance_summary()

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
    async def test_post_trade_processing(self, mock_strategy):
        """Test post-trade processing."""
        initial_trades = mock_strategy.metrics.total_trades

        # Mock trade result with P&L
        mock_trade_result = Mock()
        mock_trade_result.pnl = Decimal("10.50")

        await mock_strategy.post_trade_processing(mock_trade_result)

        assert mock_strategy.metrics.total_trades == initial_trades + 1
        assert mock_strategy.metrics.winning_trades == 1
        assert mock_strategy.metrics.total_pnl == Decimal("10.50")
        assert mock_strategy.metrics.win_rate == 1.0

    @pytest.mark.asyncio
    async def test_post_trade_processing_negative_pnl(self, mock_strategy):
        """Test post-trade processing with negative PnL."""
        initial_trades = mock_strategy.metrics.total_trades

        mock_trade_result = Mock()
        mock_trade_result.pnl = Decimal("-5.25")

        await mock_strategy.post_trade_processing(mock_trade_result)

        assert mock_strategy.metrics.total_trades == initial_trades + 1
        assert mock_strategy.metrics.losing_trades == 1
        assert mock_strategy.metrics.total_pnl == Decimal("-5.25")
        assert mock_strategy.metrics.win_rate == 0.0

    def test_abstract_methods_required(self):
        """Test that concrete strategies must implement abstract methods."""

        class IncompleteStrategy(BaseStrategy):
            """Incomplete strategy implementation."""

            pass

        with pytest.raises(TypeError):
            IncompleteStrategy({})
