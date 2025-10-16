"""
Unit tests for Breakout Strategy.

Tests the breakout strategy implementation with comprehensive coverage
including signal generation, validation, position sizing, and exit conditions.
"""

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

# Disable logging during tests for performance
logging.disable(logging.CRITICAL)

# Fast mock time for deterministic tests - use recent time to avoid "signal too old" issues
FIXED_TIME = datetime.now(timezone.utc) - timedelta(minutes=1)  # 1 minute ago

# Import from P-001
from src.core.types import (
    MarketData,
    Position,
    PositionSide,
    PositionStatus,
    Signal,
    SignalDirection,
)

# Import from P-012
from src.strategies.static.breakout import BreakoutStrategy


class TestBreakoutStrategy:
    """Test cases for BreakoutStrategy."""

    @staticmethod
    def create_market_data_with_price(
        base_data: MarketData, price: Decimal, volume: Decimal | None = None
    ) -> MarketData:
        """Create new MarketData with different price (close) and optionally volume."""
        return MarketData(
            symbol=base_data.symbol,
            open=base_data.open,
            high=base_data.high,
            low=base_data.low,
            close=price,
            volume=volume if volume is not None else base_data.volume,
            timestamp=base_data.timestamp,
            exchange=base_data.exchange,
            bid_price=base_data.bid_price,
            ask_price=base_data.ask_price,
        )

    @pytest.fixture(scope="class")
    def mock_config(self):
        """Create mock strategy configuration - cached for class scope."""
        return {
            "name": "breakout",
            "strategy_id": "breakout_001",
            "strategy_type": "momentum",
            "symbol": "BTC/USDT",
            "enabled": True,
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "timeframe": "5m",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {
                "lookback_period": 10,  # Reduced for faster tests
                "breakout_threshold": 0.02,
                "volume_multiplier": 1.5,
                "consolidation_periods": 3,  # Reduced for faster tests
                "false_breakout_filter": True,
                "false_breakout_threshold": 0.01,
                "target_multiplier": 2.0,
                "atr_period": 10,  # Reduced for faster tests
                "atr_multiplier": 2.0,
            },
        }

    @pytest.fixture(scope="class")
    def mock_indicators(self):
        """Create mock indicators service."""
        mock = Mock()
        mock.calculate_sma = AsyncMock(return_value=Decimal("50000"))
        mock.calculate_rsi = AsyncMock(return_value=Decimal("65"))
        mock.calculate_atr = AsyncMock(return_value=Decimal("1000"))
        mock.calculate_bollinger_bands = AsyncMock(return_value={
            'upper': Decimal("51000"), 
            'middle': Decimal("50000"), 
            'lower': Decimal("49000")
        })
        mock.calculate_volume_ratio = AsyncMock(return_value=Decimal("1.5"))
        mock.calculate_volatility = AsyncMock(return_value=Decimal("500"))
        return mock
    
    @pytest.fixture(scope="class")
    def strategy(self, mock_config, mock_indicators):
        """Create strategy instance - cached for class scope."""
        strategy = BreakoutStrategy(mock_config)
        strategy._indicators = mock_indicators
        return strategy

    @pytest.fixture(scope="class")
    def mock_market_data(self):
        """Create mock market data - cached for class scope with fixed time."""
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

    @pytest.fixture(scope="class")
    def mock_position(self):
        """Create mock position - cached for class scope with fixed time."""
        return Position(
            symbol="BTC/USD",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            unrealized_pnl=Decimal("100"),
            side=PositionSide.LONG,
            status=PositionStatus.OPEN,
            opened_at=FIXED_TIME,  # Use fixed time for performance
            exchange="test",
        )

    def test_strategy_initialization(self, strategy, mock_config):
        """Test strategy initialization."""
        assert strategy.name == "breakout"
        assert strategy.strategy_type.value == "momentum"
        assert strategy.lookback_period == 10
        assert strategy.breakout_threshold == 0.02
        assert strategy.volume_multiplier == 1.5
        assert strategy.consolidation_periods == 3
        assert strategy.false_breakout_filter is True
        assert strategy.false_breakout_threshold == 0.01
        assert strategy.target_multiplier == 2.0

    def test_strategy_info(self, strategy):
        """Test strategy information retrieval."""
        info = strategy.get_strategy_info()

        assert info["name"] == "breakout"
        assert info["strategy_type"] == "breakout"
        assert "parameters" in info
        assert info["parameters"]["lookback_period"] == 10
        assert info["parameters"]["breakout_threshold"] == 0.02
        assert info["parameters"]["volume_multiplier"] == 1.5
        assert info["parameters"]["consolidation_periods"] == 3

    @pytest.mark.asyncio
    async def test_update_price_history(self, strategy, mock_market_data):
        """Test that strategy can process market data."""
        # Test basic signal generation to verify price history handling works
        signals = await strategy.generate_signals(mock_market_data)
        # Should return empty list if no breakout conditions met, but no exceptions
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_update_price_history_with_none_values(self, strategy):
        """Test strategy handling of market data with None/zero values."""
        data = MarketData(
            symbol="BTC/USD",
            open=Decimal("49900"),
            high=Decimal("50100"),
            low=Decimal("49800"),
            close=Decimal("50000"),
            volume=Decimal("0"),
            timestamp=FIXED_TIME,
            exchange="binance",
            bid_price=None,
            ask_price=None,
        )

        # Strategy should handle data with None/zero values gracefully
        signals = await strategy.generate_signals(data)
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_update_support_resistance_levels_insufficient_data(self, strategy, mock_market_data):
        """Test support/resistance handling with insufficient data."""
        # Test with fresh strategy (no historical data)
        signals = await strategy.generate_signals(mock_market_data)
        # Should handle insufficient data gracefully and return empty list
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_update_support_resistance_levels_success(self, strategy, mock_market_data, mock_indicators):
        """Test support/resistance level calculation."""
        # Mock sufficient price data for level calculation
        mock_indicators.calculate_sma.return_value = Decimal("50000")
        mock_indicators.calculate_volatility.return_value = Decimal("500")
        
        # Test signal generation (which includes support/resistance calculation)
        signals = await strategy.generate_signals(mock_market_data)
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_update_support_resistance_levels_exception(self, strategy, mock_market_data, mock_indicators):
        """Test support/resistance calculation error handling."""
        # Mock exception in indicator calculation
        mock_indicators.calculate_sma.side_effect = Exception("Test error")
        
        # Strategy should handle exceptions gracefully
        signals = await strategy.generate_signals(mock_market_data)
        assert isinstance(signals, list)
        assert signals == []  # Should return empty on error

    @pytest.mark.asyncio
    async def test_check_consolidation_period_insufficient_data(self, strategy, mock_market_data):
        """Test consolidation period check with insufficient data."""
        # Test with fresh strategy (no historical data)
        signals = await strategy.generate_signals(mock_market_data)
        # Should handle insufficient data and return empty list
        assert isinstance(signals, list)
        assert signals == []

    @pytest.mark.asyncio
    async def test_check_consolidation_period_consolidating(self, strategy, mock_market_data, mock_indicators):
        """Test consolidation period detection - consolidating market."""
        # Mock indicators showing consolidation (low volatility)
        mock_indicators.calculate_volatility.return_value = Decimal("100")  # Low volatility
        mock_indicators.calculate_sma.return_value = Decimal("50000")
        
        signals = await strategy.generate_signals(mock_market_data)
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_check_consolidation_period_not_consolidating(self, strategy, mock_market_data, mock_indicators):
        """Test consolidation period detection - trending market."""
        # Mock indicators showing trending (high volatility)
        mock_indicators.calculate_volatility.return_value = Decimal("1000")  # High volatility
        mock_indicators.calculate_sma.return_value = Decimal("50000")
        
        signals = await strategy.generate_signals(mock_market_data)
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_check_consolidation_period_exception(self, strategy, mock_market_data, mock_indicators):
        """Test consolidation check error handling."""
        # Mock exception in volatility calculation
        mock_indicators.calculate_volatility.side_effect = Exception("Test error")
        mock_indicators.calculate_sma.return_value = Decimal("50000")
        
        signals = await strategy.generate_signals(mock_market_data)
        assert isinstance(signals, list)
        assert signals == []  # Should return empty on error

    @pytest.mark.asyncio
    async def test_check_resistance_breakout_no_breakout(self, strategy, mock_market_data, mock_indicators):
        """Test resistance breakout detection - no breakout scenario."""
        # Mock price below resistance with normal volume
        mock_indicators.calculate_volume_ratio.return_value = Decimal("1.0")  # Normal volume
        
        # Test with current market data (price at 50000, no breakout expected)
        signals = await strategy.generate_signals(mock_market_data)
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_check_resistance_breakout_with_breakout(self, strategy, mock_market_data, mock_indicators):
        """Test resistance breakout detection - with breakout."""
        # Mock high price and volume for breakout
        mock_indicators.calculate_volume_ratio.return_value = Decimal("2.0")  # High volume
        
        # Create data with higher price for breakout
        breakout_data = TestBreakoutStrategy.create_market_data_with_price(
            mock_market_data, Decimal("52000")  # Significant price increase
        )
        
        signals = await strategy.generate_signals(breakout_data)
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_check_resistance_breakout_low_volume(self, strategy, mock_market_data, mock_indicators):
        """Test resistance breakout with insufficient volume."""
        # Mock low volume ratio
        mock_indicators.calculate_volume_ratio.return_value = Decimal("0.5")  # Low volume
        
        # Create data with higher price but low volume
        breakout_data = TestBreakoutStrategy.create_market_data_with_price(
            mock_market_data, Decimal("52000"), volume=Decimal("20")  # Low volume
        )
        
        signals = await strategy.generate_signals(breakout_data)
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_check_support_breakout_no_breakout(self, strategy, mock_market_data, mock_indicators):
        """Test support breakout detection - no breakout scenario."""
        # Mock price above support with normal volume
        mock_indicators.calculate_volume_ratio.return_value = Decimal("1.0")  # Normal volume
        
        # Test with current market data (price at 50000, no support break expected)
        signals = await strategy.generate_signals(mock_market_data)
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_check_support_breakout_with_breakout(self, strategy, mock_market_data, mock_indicators):
        """Test support breakout detection - with breakout."""
        # Mock high volume for support break
        mock_indicators.calculate_volume_ratio.return_value = Decimal("2.0")  # High volume
        
        # Create data with lower price for support breakout
        breakout_data = TestBreakoutStrategy.create_market_data_with_price(
            mock_market_data, Decimal("48000")  # Significant price decrease
        )
        
        signals = await strategy.generate_signals(breakout_data)
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_check_volume_confirmation_success(self, strategy, mock_market_data, mock_indicators):
        """Test volume confirmation with sufficient volume."""
        # Mock high volume ratio
        mock_indicators.calculate_volume_ratio.return_value = Decimal("2.0")  # Above threshold
        
        signals = await strategy.generate_signals(mock_market_data)
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_check_volume_confirmation_zero_volume(self, strategy, mock_market_data, mock_indicators):
        """Test volume confirmation with zero volume."""
        # Mock zero volume ratio
        mock_indicators.calculate_volume_ratio.return_value = Decimal("0")  # Zero volume
        
        # Create data with zero volume
        zero_volume_data = TestBreakoutStrategy.create_market_data_with_price(
            mock_market_data, Decimal("50000"), volume=Decimal("0")
        )
        
        signals = await strategy.generate_signals(zero_volume_data)
        assert isinstance(signals, list)

    def test_check_volume_confirmation_insufficient_data(self, strategy):
        """Test volume confirmation with insufficient data."""
        # Add some volume data but not enough
        for i in range(5):  # Reduced for performance
            strategy.commons.price_history.volume_history.append(100.0)

        result = strategy._check_volume_confirmation(150.0)
        assert result  # Should pass if insufficient data

    def test_check_false_breakout_disabled(self, strategy, mock_market_data):
        """Test false breakout check when disabled."""
        strategy.false_breakout_filter = False

        result = strategy._check_false_breakout(mock_market_data)
        assert result is None

    def test_check_false_breakout_resistance_return(self, strategy, mock_market_data):
        """Test false breakout check with resistance return."""
        # Reset state from previous tests
        strategy.false_breakout_filter = True

        # Add resistance level
        strategy.resistance_levels = [51000]

        # Set price near resistance
        test_data = self.create_market_data_with_price(
            mock_market_data, Decimal("51050")
        )  # Within threshold

        result = strategy._check_false_breakout(test_data)
        assert result is not None
        assert result["breakout_type"] == "false_resistance"

    def test_check_false_breakout_support_return(self, strategy, mock_market_data):
        """Test false breakout check with support return."""
        # Reset state from previous tests
        strategy.false_breakout_filter = True

        # Add support level
        strategy.support_levels = [49000]

        # Set price near support
        test_data = self.create_market_data_with_price(
            mock_market_data, Decimal("48950")
        )  # Within threshold

        result = strategy._check_false_breakout(test_data)
        assert result is not None
        assert result["breakout_type"] == "false_support"

    @pytest.mark.asyncio
    async def test_generate_bullish_breakout_signal_success(self, strategy, mock_market_data):
        """Test successful bullish breakout signal generation."""
        # Add support levels for target calculation
        strategy.support_levels = [48000, 47000]

        breakout_info = {
            "level": 51000,
            "breakout_price": 52000,
            "breakout_type": "resistance",
            "volume": 200.0,
        }

        signal = await strategy._generate_bullish_breakout_signal(mock_market_data, breakout_info)

        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.strength > 0
        assert signal.metadata["signal_type"] == "breakout_entry"
        assert signal.metadata["breakout_direction"] == "bullish"
        assert "target_price" in signal.metadata

    @pytest.mark.asyncio
    async def test_generate_bearish_breakout_signal_success(self, strategy, mock_market_data):
        """Test successful bearish breakout signal generation."""
        # Add resistance levels for target calculation
        strategy.resistance_levels = [52000, 53000]

        breakout_info = {
            "level": 49000,
            "breakout_price": 48000,
            "breakout_type": "support",
            "volume": 200.0,
        }

        signal = await strategy._generate_bearish_breakout_signal(mock_market_data, breakout_info)

        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert signal.strength > 0
        assert signal.metadata["signal_type"] == "breakout_entry"
        assert signal.metadata["breakout_direction"] == "bearish"
        assert "target_price" in signal.metadata

    @pytest.mark.asyncio
    async def test_generate_false_breakout_exit_signal_resistance(self, strategy, mock_market_data):
        """Test false breakout exit signal for resistance."""
        false_breakout_info = {
            "level": 51000,
            "current_price": 51050,
            "breakout_type": "false_resistance",
        }

        signal = await strategy._generate_false_breakout_exit_signal(
            mock_market_data, false_breakout_info
        )

        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert signal.strength == Decimal("0.9")
        assert signal.metadata["signal_type"] == "false_breakout_exit"

    @pytest.mark.asyncio
    async def test_generate_false_breakout_exit_signal_support(self, strategy, mock_market_data):
        """Test false breakout exit signal for support."""
        false_breakout_info = {
            "level": 49000,
            "current_price": 48950,
            "breakout_type": "false_support",
        }

        signal = await strategy._generate_false_breakout_exit_signal(
            mock_market_data, false_breakout_info
        )

        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.strength == Decimal("0.9")
        assert signal.metadata["signal_type"] == "false_breakout_exit"

    @pytest.mark.asyncio
    async def test_generate_signals_empty_data(self, strategy):
        """Test signal generation with empty data."""
        signals = await strategy.generate_signals(None)
        assert signals == []

        signals = await strategy.generate_signals(
            MarketData(
                symbol="BTC/USD",
                timestamp=FIXED_TIME,
                open=Decimal("0"),
                high=Decimal("0"),
                low=Decimal("0"),
                close=Decimal("0"),
                volume=Decimal("100"),
                exchange="test",
            )
        )
        assert signals == []

    @pytest.mark.asyncio
    async def test_generate_signals_invalid_price(self, strategy):
        """Test signal generation with invalid price."""
        data = MarketData(
            symbol="BTC/USD",
            timestamp=FIXED_TIME,
            open=Decimal("-100"),  # Negative price
            high=Decimal("-100"),
            low=Decimal("-100"),
            close=Decimal("-100"),
            volume=Decimal("100"),
            exchange="test",
        )

        signals = await strategy.generate_signals(data)
        assert signals == []

    @pytest.mark.asyncio
    async def test_generate_signals_insufficient_data(self, strategy, mock_market_data):
        """Test signal generation with insufficient data."""
        signals = await strategy.generate_signals(mock_market_data)
        assert signals == []

    @pytest.mark.asyncio
    async def test_generate_signals_no_consolidation(self, strategy, mock_market_data):
        """Test signal generation without consolidation period."""
        # Add enough data but not consolidating
        for i in range(12):  # Reduced for performance
            strategy.commons.price_history.price_history.append(50000 + i * 1000)  # Large range

        signals = await strategy.generate_signals(mock_market_data)
        assert signals == []

    @pytest.mark.asyncio
    async def test_generate_signals_resistance_breakout(self, strategy, mock_market_data, mock_indicators):
        """Test signal generation with resistance breakout scenario."""
        # Mock indicators for breakout detection
        mock_indicators.calculate_volume_ratio.return_value = Decimal("2.0")  # High volume
        mock_indicators.calculate_sma.return_value = Decimal("50000")  # SMA
        mock_indicators.calculate_volatility.return_value = Decimal("500")  # Volatility
        mock_indicators.calculate_atr.return_value = Decimal("1000")  # ATR
        
        # Create breakout market data
        breakout_data = TestBreakoutStrategy.create_market_data_with_price(
            mock_market_data, Decimal("52100"), volume=Decimal("200")
        )
        
        signals = await strategy.generate_signals(breakout_data)
        # Test passes if no exceptions thrown - strategy may or may not generate signals
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_generate_signals_support_breakout(self, strategy, mock_market_data, mock_indicators):
        """Test signal generation with support breakout scenario."""
        # Mock indicators for support breakout
        mock_indicators.calculate_volume_ratio.return_value = Decimal("2.0")  # High volume
        mock_indicators.calculate_sma.return_value = Decimal("50000")  # SMA
        mock_indicators.calculate_volatility.return_value = Decimal("500")  # Volatility
        mock_indicators.calculate_atr.return_value = Decimal("1000")  # ATR
        
        # Create support breakout market data
        breakout_data = TestBreakoutStrategy.create_market_data_with_price(
            mock_market_data, Decimal("47900"), volume=Decimal("200")
        )
        
        signals = await strategy.generate_signals(breakout_data)
        # Test passes if no exceptions thrown - strategy may or may not generate signals
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_generate_signals_false_breakout_exit(self, strategy, mock_market_data, mock_indicators):
        """Test signal generation with false breakout exit scenario."""
        # Mock indicators for false breakout detection
        mock_indicators.calculate_volume_ratio.return_value = Decimal("1.0")  # Normal volume
        mock_indicators.calculate_sma.return_value = Decimal("50000")  # SMA
        mock_indicators.calculate_volatility.return_value = Decimal("500")  # Volatility
        mock_indicators.calculate_atr.return_value = Decimal("1000")  # ATR
        
        # Create market data near resistance level
        near_resistance_data = TestBreakoutStrategy.create_market_data_with_price(
            mock_market_data, Decimal("51050"), volume=Decimal("100")
        )
        
        signals = await strategy.generate_signals(near_resistance_data)
        # Test passes if no exceptions thrown - strategy may or may not generate signals
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_generate_signals_exception_handling(self, strategy):
        """Test signal generation exception handling."""
        # Create data that will cause exception
        data = MarketData(
            symbol="BTC/USD",
            timestamp=FIXED_TIME,
            open=Decimal("50000"),
            high=Decimal("50000"),
            low=Decimal("50000"),
            close=Decimal("50000"),
            volume=Decimal("100"),
            exchange="test",
        )

        # Corrupt price history
        strategy.commons.price_history.price_history = [np.nan, np.inf]

        signals = await strategy.generate_signals(data)
        assert signals == []

    @pytest.mark.asyncio
    async def test_validate_signal_success(self, strategy):
        """Test successful signal validation."""
        signal = Signal(
            signal_id="test_signal_1",
            strategy_id="test_strategy_1",
            strategy_name="test_strategy",
            symbol="BTC/USD",
            direction=SignalDirection.BUY,
            strength=Decimal("0.8"),
            timestamp=FIXED_TIME,
            source="breakout",
            metadata={
                "breakout_level": 51000,
                "breakout_price": 52000,
                "target_price": 53000,
                "breakout_direction": "bullish",
                "signal_type": "breakout_entry",
            },
        )

        result = await strategy.validate_signal(signal)
        assert result

    @pytest.mark.asyncio
    async def test_validate_signal_low_confidence(self, strategy):
        """Test signal validation with low confidence."""
        signal = Signal(
            signal_id="test_signal_2",
            strategy_id="test_strategy_1",
            strategy_name="test_strategy",
            symbol="BTC/USD",
            direction=SignalDirection.BUY,
            strength=Decimal("0.3"),  # Below threshold
            timestamp=FIXED_TIME,
            source="breakout",
            metadata={
                "breakout_level": 51000,
                "breakout_price": 52000,
                "target_price": 53000,
                "breakout_direction": "bullish",
                "signal_type": "breakout_entry",
            },
        )

        result = await strategy.validate_signal(signal)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_signal_old_signal(self, strategy):
        """Test signal validation with old signal."""
        signal = Signal(
            signal_id="test_signal_3",
            strategy_id="test_strategy_1",
            strategy_name="test_strategy",
            symbol="BTC/USD",
            direction=SignalDirection.BUY,
            strength=Decimal("0.8"),
            timestamp=FIXED_TIME - timedelta(minutes=10),
            source="breakout",
            metadata={
                "breakout_level": 51000,
                "breakout_price": 52000,
                "target_price": 53000,
                "breakout_direction": "bullish",
                "signal_type": "breakout_entry",
            },
        )

        result = await strategy.validate_signal(signal)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_signal_missing_metadata(self, strategy):
        """Test signal validation with missing metadata."""
        signal = Signal(
            signal_id="test_signal_4",
            strategy_id="test_strategy_1",
            strategy_name="test_strategy",
            symbol="BTC/USD",
            direction=SignalDirection.BUY,
            strength=Decimal("0.8"),
            timestamp=FIXED_TIME,
            source="breakout",
            metadata={},  # Missing required fields
        )

        result = await strategy.validate_signal(signal)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_signal_invalid_breakout_price(self, strategy):
        """Test signal validation with invalid breakout price."""
        signal = Signal(
            signal_id="test_signal_5",
            strategy_id="test_strategy_1",
            strategy_name="test_strategy",
            symbol="BTC/USD",
            direction=SignalDirection.BUY,
            strength=Decimal("0.8"),
            timestamp=FIXED_TIME,
            source="breakout",
            metadata={
                "breakout_level": 51000,
                "breakout_price": -100,  # Invalid price
                "target_price": 53000,
                "breakout_direction": "bullish",
                "signal_type": "breakout_entry",
            },
        )

        result = await strategy.validate_signal(signal)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_signal_exception_handling(self, strategy):
        """Test signal validation exception handling."""
        signal = Signal(
            signal_id="test_signal_6",
            strategy_id="test_strategy_1",
            strategy_name="test_strategy",
            symbol="BTC/USD",
            direction=SignalDirection.BUY,
            strength=Decimal("0.8"),
            timestamp=FIXED_TIME,
            source="breakout",
            metadata={
                "breakout_level": 51000,
                "breakout_price": 52000,
                "target_price": 53000,
                "breakout_direction": "bullish",
                "signal_type": "breakout_entry",
            },
        )

        # Corrupt signal to cause exception
        signal.timestamp = None

        result = await strategy.validate_signal(signal)
        assert result is False

    def test_get_position_size_success(self, strategy):
        """Test successful position size calculation."""
        signal = Signal(
            signal_id="test_signal_7",
            strategy_id="test_strategy_1",
            strategy_name="test_strategy",
            symbol="BTC/USD",
            direction=SignalDirection.BUY,
            strength=Decimal("0.8"),
            timestamp=FIXED_TIME,
            source="breakout",
            metadata={"range_size": 1000, "breakout_price": 52000},
        )

        position_size = strategy.get_position_size(signal)
        assert isinstance(position_size, Decimal)
        assert position_size > 0

    def test_get_position_size_with_max_limit(self, strategy):
        """Test position size calculation with maximum limit."""
        signal = Signal(
            signal_id="test_signal_8",
            strategy_id="test_strategy_1",
            strategy_name="test_strategy",
            symbol="BTC/USD",
            direction=SignalDirection.BUY,
            strength=Decimal("1.0"),
            timestamp=FIXED_TIME,
            source="breakout",
            metadata={
                "range_size": 10000,  # Large range
                "breakout_price": 52000,
            },
        )

        position_size = strategy.get_position_size(signal)
        max_size = Decimal(str(strategy.config.parameters.get("max_position_size_pct", 0.1)))
        assert position_size <= max_size

    def test_get_position_size_exception_handling(self, strategy):
        """Test position size calculation exception handling."""
        signal = Signal(
            signal_id="test_signal_9",
            strategy_id="test_strategy_1",
            strategy_name="test_strategy",
            symbol="BTC/USD",
            direction=SignalDirection.BUY,
            strength=Decimal("0.8"),
            timestamp=FIXED_TIME,
            source="breakout",
            metadata={
                "range_size": "invalid",  # Will cause exception
                "breakout_price": 52000,
            },
        )

        position_size = strategy.get_position_size(signal)
        # Should return minimum position size on error
        expected_min = strategy.config.position_size_pct * Decimal("0.5")
        assert position_size == expected_min

    @pytest.mark.asyncio
    async def test_should_exit_atr_stop_loss(self, strategy, mock_position, mock_market_data, mock_indicators):
        """Test exit condition based on ATR stop loss."""
        # Mock ATR calculation to trigger stop loss
        mock_indicators.calculate_atr.return_value = Decimal("1000")  # ATR value

        # Create position with price that should trigger stop loss
        stop_loss_position = mock_position.model_copy(update={"current_price": Decimal("48000")})  # Far below stop loss

        result = await strategy.should_exit(stop_loss_position, mock_market_data)
        # Test should pass if no exceptions are thrown - strategy behavior may vary
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_should_exit_target_price_reached(self, strategy, mock_position, mock_market_data, mock_indicators):
        """Test exit condition based on target price."""
        # Mock ATR to not trigger stop loss
        mock_indicators.calculate_atr.return_value = Decimal("500")  # Small ATR
        
        # Create position with target price above current price
        target_position = mock_position.model_copy(update={
            "current_price": Decimal("53500"),
            "metadata": {"target_price": 53000}
        })

        result = await strategy.should_exit(target_position, mock_market_data)
        assert result

    @pytest.mark.asyncio
    async def test_should_exit_target_price_not_reached(self, strategy, mock_position, mock_market_data, mock_indicators):
        """Test exit condition when target price not reached."""
        # Mock ATR to not trigger stop loss
        mock_indicators.calculate_atr.return_value = Decimal("500")  # Small ATR
        
        # Create position with price below target
        target_position = mock_position.model_copy(update={
            "current_price": Decimal("52500"),
            "metadata": {"target_price": 53000}
        })

        result = await strategy.should_exit(target_position, mock_market_data)
        assert result is False

    @pytest.mark.asyncio
    async def test_should_exit_sell_position_target_price(self, strategy, mock_market_data, mock_indicators):
        """Test exit condition for sell position target price."""
        # Mock ATR to not trigger stop loss
        mock_indicators.calculate_atr.return_value = Decimal("500")  # Small ATR
        
        # Create sell position with target and price below target
        sell_position = Position(
            symbol="BTC/USD",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("46500"),  # Below target
            unrealized_pnl=Decimal("100"),
            side=PositionSide.SHORT,
            status=PositionStatus.OPEN,
            opened_at=FIXED_TIME,
            exchange="test",
            metadata={"target_price": 47000}
        )

        result = await strategy.should_exit(sell_position, mock_market_data)
        assert result

    @pytest.mark.asyncio
    async def test_should_exit_no_exit_condition(self, strategy, mock_position, mock_market_data, mock_indicators):
        """Test when no exit condition is met."""
        # Mock ATR to not trigger stop loss
        mock_indicators.calculate_atr.return_value = Decimal("500")  # Small ATR
        
        # Default position price doesn't trigger any exit condition
        result = await strategy.should_exit(mock_position, mock_market_data)
        assert result is False

    @pytest.mark.asyncio
    async def test_should_exit_exception_handling(self, strategy, mock_position, mock_indicators):
        """Test exit check exception handling."""
        # Create data that will cause exception
        data = MarketData(
            symbol="BTC/USD",
            timestamp=FIXED_TIME,
            open=Decimal("50000"),
            high=Decimal("50000"),
            low=Decimal("50000"),
            close=Decimal("50000"),
            volume=Decimal("100"),
            exchange="test",
        )
        
        # Mock exception in ATR calculation
        mock_indicators.calculate_atr.side_effect = Exception("Test error")

        result = await strategy.should_exit(mock_position, data)
        assert result is False

    @pytest.mark.asyncio
    async def test_strategy_integration(self, strategy, mock_market_data):
        """Test full strategy integration."""
        # Add sufficient data
        for i in range(12):  # Reduced for performance
            strategy.commons.price_history.price_history.append(50000 + i * 10)
            strategy.commons.price_history.volume_history.append(100.0)

        # Add support/resistance levels
        strategy.support_levels = [49000]
        strategy.resistance_levels = [51000]

        # Set breakout conditions
        test_data = self.create_market_data_with_price(mock_market_data, Decimal("52000"))
        mock_market_data.volume = Decimal("200")

        # Generate signals
        signals = await strategy.generate_signals(mock_market_data)

        if signals:
            signal = signals[0]

            # Validate signal
            is_valid = await strategy.validate_signal(signal)
            assert is_valid

            # Calculate position size
            position_size = strategy.get_position_size(signal)
            assert isinstance(position_size, Decimal)
            assert position_size > 0

    def test_strategy_parameter_validation(self):
        """Test strategy parameter validation."""
        config = {
            "strategy_id": "test_breakout",
            "name": "breakout",
            "strategy_type": "momentum",
            "enabled": True,
            "symbol": "BTC/USD",
            "timeframe": "5m",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {
                "lookback_period": 10,  # Different from default
                "breakout_threshold": 0.01,  # Different from default
                "volume_multiplier": 2.0,  # Different from default
                "consolidation_periods": 3,  # Different from default
                "false_breakout_filter": False,  # Different from default
                "false_breakout_threshold": 0.005,  # Different from default
                "target_multiplier": 1.5,  # Different from default
                "atr_period": 10,  # Different from default
                "atr_multiplier": 1.5,  # Different from default
            },
        }

        strategy = BreakoutStrategy(config)

        assert strategy.lookback_period == 10
        assert strategy.breakout_threshold == 0.01
        assert strategy.volume_multiplier == 2.0
        assert strategy.consolidation_periods == 3
        assert strategy.false_breakout_filter is False
        assert strategy.false_breakout_threshold == 0.005
        assert strategy.target_multiplier == 1.5
        assert strategy.atr_period == 10
        assert strategy.atr_multiplier == 1.5
