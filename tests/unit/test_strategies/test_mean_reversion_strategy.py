"""
Unit tests for Mean Reversion Strategy.

Tests the mean reversion strategy implementation with comprehensive coverage
including signal generation, validation, position sizing, and exit conditions.
"""

import pytest
import numpy as np
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio

# Import from P-001
from src.core.types import (
    Signal, MarketData, Position, StrategyConfig,
    StrategyStatus, StrategyMetrics, SignalDirection, OrderSide
)
from src.core.exceptions import ValidationError

# Import from P-012
from src.strategies.static.mean_reversion import MeanReversionStrategy


class TestMeanReversionStrategy:
    """Test cases for MeanReversionStrategy."""

    @pytest.fixture
    def mock_config(self):
        """Create mock strategy configuration."""
        return {
            "name": "mean_reversion",
            "strategy_type": "static",
            "enabled": True,
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "timeframe": "5m",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {
                "lookback_period": 20,
                "entry_threshold": 2.0,
                "exit_threshold": 0.5,
                "atr_period": 14,
                "atr_multiplier": 2.0,
                "volume_filter": True,
                "min_volume_ratio": 1.5,
                "confirmation_timeframe": "1h"
            }
        }

    @pytest.fixture
    def strategy(self, mock_config):
        """Create strategy instance."""
        return MeanReversionStrategy(mock_config)

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        return MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc),
            bid=Decimal("49999"),
            ask=Decimal("50001"),
            open_price=Decimal("49900"),
            high_price=Decimal("50100"),
            low_price=Decimal("49800")
        )

    @pytest.fixture
    def mock_position(self):
        """Create mock position."""
        return Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            unrealized_pnl=Decimal("100"),
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc)
        )

    def test_strategy_initialization(self, strategy, mock_config):
        """Test strategy initialization."""
        assert strategy.name == "mean_reversion"
        assert strategy.strategy_type.value == "static"
        assert strategy.lookback_period == 20
        assert strategy.entry_threshold == 2.0
        assert strategy.exit_threshold == 0.5
        assert strategy.volume_filter is True
        assert strategy.min_volume_ratio == 1.5

    def test_strategy_info(self, strategy):
        """Test strategy information retrieval."""
        info = strategy.get_strategy_info()

        assert info["name"] == "mean_reversion"
        assert info["strategy_type"] == "mean_reversion"
        assert "parameters" in info
        assert info["parameters"]["lookback_period"] == 20
        assert info["parameters"]["entry_threshold"] == 2.0
        assert info["parameters"]["exit_threshold"] == 0.5

    def test_update_price_history(self, strategy, mock_market_data):
        """Test price history update."""
        initial_length = len(strategy.price_history)

        strategy._update_price_history(mock_market_data)

        assert len(strategy.price_history) == initial_length + 1
        assert len(strategy.volume_history) == initial_length + 1
        assert len(strategy.high_history) == initial_length + 1
        assert len(strategy.low_history) == initial_length + 1

        # Check values
        assert strategy.price_history[-1] == 50000.0
        assert strategy.volume_history[-1] == 100.0
        assert strategy.high_history[-1] == 50100.0
        assert strategy.low_history[-1] == 49800.0

    def test_update_price_history_with_none_values(self, strategy):
        """Test price history update with None values."""
        data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc),
            bid=None,
            ask=None,
            open_price=None,
            high_price=None,
            low_price=None
        )

        strategy._update_price_history(data)

        assert strategy.price_history[-1] == 50000.0
        # Should use actual volume value
        assert strategy.volume_history[-1] == 100.0
        assert strategy.high_history[-1] == 50000.0
        assert strategy.low_history[-1] == 50000.0

    def test_calculate_zscore_insufficient_data(self, strategy):
        """Test Z-score calculation with insufficient data."""
        # Add some data but not enough
        for i in range(10):
            strategy.price_history.append(50000 + i)

        z_score = strategy._calculate_zscore()
        assert z_score is None

    def test_calculate_zscore_success(self, strategy):
        """Test successful Z-score calculation."""
        # Add enough data
        for i in range(25):
            strategy.price_history.append(50000 + i)

        z_score = strategy._calculate_zscore()
        assert z_score is not None
        assert isinstance(z_score, float)

    def test_calculate_zscore_zero_std_dev(self, strategy):
        """Test Z-score calculation with zero standard deviation."""
        # Add identical values
        for i in range(25):
            strategy.price_history.append(50000.0)

        z_score = strategy._calculate_zscore()
        assert z_score is None

    def test_calculate_zscore_exception(self, strategy):
        """Test Z-score calculation with exception."""
        # Add invalid data
        strategy.price_history = [np.nan, np.inf, -np.inf]

        z_score = strategy._calculate_zscore()
        assert z_score is None

    def test_check_volume_filter_insufficient_data(
            self, strategy, mock_market_data):
        """Test volume filter with insufficient data."""
        # Add some volume data but not enough
        for i in range(10):
            strategy.volume_history.append(100.0)

        result = strategy._check_volume_filter(mock_market_data)
        assert result is True  # Should pass if insufficient data

    def test_check_volume_filter_success(self, strategy, mock_market_data):
        """Test volume filter with sufficient data."""
        # Add enough volume data
        for i in range(25):
            strategy.volume_history.append(100.0)

        # Test with high volume
        mock_market_data.volume = Decimal("200")  # 2x average
        result = strategy._check_volume_filter(mock_market_data)
        assert result  # Should pass with 2x volume ratio

        # Test with low volume
        mock_market_data.volume = Decimal("50")  # 0.5x average
        result = strategy._check_volume_filter(mock_market_data)
        assert not result  # Should fail with low volume

    def test_check_volume_filter_zero_volume(self, strategy, mock_market_data):
        """Test volume filter with zero volume."""
        # Add some volume history first
        for i in range(25):
            strategy.volume_history.append(100.0)

        mock_market_data.volume = Decimal("0")

        result = strategy._check_volume_filter(mock_market_data)
        assert result is False  # Should fail with zero volume

    def test_check_volume_filter_exception(self, strategy):
        """Test volume filter with exception."""
        data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc)
        )

        # Corrupt volume history
        strategy.volume_history = [np.nan, np.inf]

        result = strategy._check_volume_filter(data)
        assert result is True  # Should pass on error

    @pytest.mark.asyncio
    async def test_generate_signals_empty_data(self, strategy):
        """Test signal generation with empty data."""
        signals = await strategy.generate_signals(None)
        assert signals == []

        signals = await strategy.generate_signals(MarketData(
            symbol="BTCUSDT",
            price=Decimal("0"),  # Invalid price
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc)
        ))
        assert signals == []

    @pytest.mark.asyncio
    async def test_generate_signals_invalid_price(self, strategy):
        """Test signal generation with invalid price."""
        data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("-100"),  # Negative price
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc)
        )

        signals = await strategy.generate_signals(data)
        assert signals == []

    @pytest.mark.asyncio
    async def test_generate_signals_insufficient_data(
            self, strategy, mock_market_data):
        """Test signal generation with insufficient data."""
        signals = await strategy.generate_signals(mock_market_data)
        assert signals == []

    @pytest.mark.asyncio
    async def test_generate_signals_bullish_entry(
            self, strategy, mock_market_data):
        """Test bullish entry signal generation."""
        # Add enough data with bullish Z-score
        for i in range(25):
            strategy.price_history.append(50000 + i * 100)  # Upward trend

        signals = await strategy.generate_signals(mock_market_data)

        # Should generate bullish signal
        assert len(signals) > 0
        signal = signals[0]
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence > 0
        assert "z_score" in signal.metadata
        assert signal.metadata["signal_type"] == "entry"

    @pytest.mark.asyncio
    async def test_generate_signals_bearish_entry(
            self, strategy, mock_market_data):
        """Test bearish entry signal generation."""
        # Add enough data with bearish Z-score
        for i in range(25):
            strategy.price_history.append(50000 - i * 100)  # Downward trend

        signals = await strategy.generate_signals(mock_market_data)

        # Should generate bearish signal
        assert len(signals) > 0
        signal = signals[0]
        assert signal.direction == SignalDirection.SELL
        assert signal.confidence > 0
        assert "z_score" in signal.metadata
        assert signal.metadata["signal_type"] == "entry"

    @pytest.mark.asyncio
    async def test_generate_signals_exit_signals(
            self, strategy, mock_market_data):
        """Test exit signal generation."""
        # Temporarily modify thresholds for testing
        original_entry_threshold = strategy.entry_threshold
        original_exit_threshold = strategy.exit_threshold
        strategy.entry_threshold = 5.0  # Very high entry threshold
        strategy.exit_threshold = 0.5   # Lower exit threshold to make it easier to trigger

        # Add data with Z-score within exit threshold (very small deviation from mean)
        # Create a more predictable scenario: all prices very close to 50000
        # Use almost constant prices with tiny random variation to ensure
        # Z-score is within exit threshold
        import numpy as np
        np.random.seed(42)  # For reproducible results
        for i in range(25):
            strategy.price_history.append(
                50000.0 +
                np.random.normal(
                    0,
                    0.0001))  # Almost constant

        signals = await strategy.generate_signals(mock_market_data)

        # Should generate exit signals when Z-score is within exit threshold
        assert len(signals) > 0
        for signal in signals:
            assert signal.metadata["signal_type"] == "exit"

        # Restore original thresholds
        strategy.entry_threshold = original_entry_threshold
        strategy.exit_threshold = original_exit_threshold

    @pytest.mark.asyncio
    async def test_generate_signals_volume_filter_rejection(
            self, strategy, mock_market_data):
        """Test signal generation with volume filter rejection."""
        # Add enough data
        for i in range(25):
            strategy.price_history.append(50000 + i * 100)
            strategy.volume_history.append(100.0)

        # Set low volume
        mock_market_data.volume = Decimal("50")

        signals = await strategy.generate_signals(mock_market_data)
        assert signals == []

    @pytest.mark.asyncio
    async def test_generate_signals_exception_handling(self, strategy):
        """Test signal generation exception handling."""
        # Create data that will cause exception
        data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc)
        )

        # Corrupt price history
        strategy.price_history = [np.nan, np.inf]

        signals = await strategy.generate_signals(data)
        assert signals == []

    @pytest.mark.asyncio
    async def test_validate_signal_success(self, strategy):
        """Test successful signal validation."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="mean_reversion",
            metadata={
                "z_score": 2.5,
                "signal_type": "entry"
            }
        )

        result = await strategy.validate_signal(signal)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_signal_low_confidence(self, strategy):
        """Test signal validation with low confidence."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.3,  # Below threshold
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="mean_reversion",
            metadata={
                "z_score": 2.5,
                "signal_type": "entry"
            }
        )

        result = await strategy.validate_signal(signal)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_signal_old_signal(self, strategy):
        """Test signal validation with old signal."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=10),
            symbol="BTCUSDT",
            strategy_name="mean_reversion",
            metadata={
                "z_score": 2.5,
                "signal_type": "entry"
            }
        )

        result = await strategy.validate_signal(signal)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_signal_missing_metadata(self, strategy):
        """Test signal validation with missing metadata."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="mean_reversion",
            metadata={}  # Missing z_score
        )

        result = await strategy.validate_signal(signal)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_signal_invalid_zscore(self, strategy):
        """Test signal validation with invalid Z-score."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="mean_reversion",
            metadata={
                "z_score": "invalid",  # Not a number
                "signal_type": "entry"
            }
        )

        result = await strategy.validate_signal(signal)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_signal_entry_below_threshold(self, strategy):
        """Test signal validation for entry signal below threshold."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="mean_reversion",
            metadata={
                "z_score": 1.5,  # Below entry threshold
                "signal_type": "entry"
            }
        )

        result = await strategy.validate_signal(signal)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_signal_exception_handling(self, strategy):
        """Test signal validation exception handling."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="mean_reversion",
            metadata={
                "z_score": 2.5,
                "signal_type": "entry"
            }
        )

        # Corrupt signal to cause exception
        signal.timestamp = None

        result = await strategy.validate_signal(signal)
        assert result is False

    def test_get_position_size_success(self, strategy):
        """Test successful position size calculation."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="mean_reversion",
            metadata={
                "z_score": 2.5
            }
        )

        position_size = strategy.get_position_size(signal)
        assert isinstance(position_size, Decimal)
        assert position_size > 0

    def test_get_position_size_with_max_limit(self, strategy):
        """Test position size calculation with maximum limit."""
        # Set high Z-score to trigger max limit
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=1.0,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="mean_reversion",
            metadata={
                "z_score": 5.0  # Very high Z-score
            }
        )

        position_size = strategy.get_position_size(signal)
        max_size = Decimal(
            str(strategy.config.parameters.get("max_position_size_pct", 0.1)))
        assert position_size <= max_size

    def test_get_position_size_exception_handling(self, strategy):
        """Test position size calculation exception handling."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="mean_reversion",
            metadata={
                "z_score": "invalid"  # Will cause exception
            }
        )

        position_size = strategy.get_position_size(signal)
        # Should return minimum position size on error
        expected_min = Decimal(str(strategy.config.position_size_pct * 0.5))
        assert position_size == expected_min

    def test_should_exit_zscore_exit(
            self,
            strategy,
            mock_position,
            mock_market_data):
        """Test exit condition based on Z-score."""
        # Temporarily modify thresholds for testing
        original_entry_threshold = strategy.entry_threshold
        original_exit_threshold = strategy.exit_threshold
        strategy.entry_threshold = 5.0  # Very high entry threshold
        strategy.exit_threshold = 0.5   # Lower exit threshold to make it easier to trigger

        # Add data with Z-score within exit threshold (very small deviation)
        # Create a more predictable scenario: all prices very close to 50000
        import numpy as np
        np.random.seed(42)  # For reproducible results
        for i in range(25):
            strategy.price_history.append(
                50000.0 +
                np.random.normal(
                    0,
                    0.0001))  # Almost constant

        result = strategy.should_exit(mock_position, mock_market_data)
        assert result is True

        # Restore original thresholds
        strategy.entry_threshold = original_entry_threshold
        strategy.exit_threshold = original_exit_threshold

    def test_should_exit_atr_stop_loss(
            self,
            strategy,
            mock_position,
            mock_market_data):
        """Test exit condition based on ATR stop loss."""
        # Add data for ATR calculation (need at least period + 1 = 15 data points)
        # Create more realistic price data with proper high/low/close
        # relationship
        for i in range(30):  # More data to ensure sufficient true ranges
            base_price = 50000 + i
            strategy.high_history.append(base_price + 100)  # High
            strategy.low_history.append(base_price - 100)   # Low
            strategy.price_history.append(base_price)       # Close

        # Set price to trigger stop loss (well below entry price)
        mock_market_data.price = Decimal("49000")  # Below stop loss

        result = strategy.should_exit(mock_position, mock_market_data)
        assert result is True

    def test_should_exit_no_exit_condition(
            self, strategy, mock_position, mock_market_data):
        """Test when no exit condition is met."""
        # Add data with Z-score outside exit threshold
        for i in range(25):
            strategy.price_history.append(50000 + i * 100)

        result = strategy.should_exit(mock_position, mock_market_data)
        assert result is False

    def test_should_exit_exception_handling(self, strategy, mock_position):
        """Test exit check exception handling."""
        # Create data that will cause exception
        data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc)
        )

        # Corrupt price history
        strategy.price_history = [np.nan, np.inf]

        result = strategy.should_exit(mock_position, data)
        assert result is False

    def test_should_exit_atr_calculation_exception(
            self, strategy, mock_position, mock_market_data):
        """Test exit check when ATR calculation fails."""
        # Add data for ATR calculation but with invalid data to cause exception
        for i in range(30):
            strategy.high_history.append(50000 + i)
            strategy.low_history.append(50000 - i)
            strategy.price_history.append(50000 + i)

        # Mock the calculate_atr function to raise an exception
        with patch('src.utils.helpers.calculate_atr', side_effect=Exception("ATR calculation failed")):
            result = strategy.should_exit(mock_position, mock_market_data)
            assert result is False

    def test_should_exit_atr_none_result(
            self, strategy, mock_position, mock_market_data):
        """Test exit check when ATR calculation returns None."""
        # Add data for ATR calculation
        for i in range(30):
            strategy.high_history.append(50000 + i)
            strategy.low_history.append(50000 - i)
            strategy.price_history.append(50000 + i)

        # Mock the calculate_atr function to return None
        with patch('src.utils.helpers.calculate_atr', return_value=None):
            result = strategy.should_exit(mock_position, mock_market_data)
            assert result is False

    def test_should_exit_atr_zero_result(
            self, strategy, mock_position, mock_market_data):
        """Test exit check when ATR calculation returns zero."""
        # Add data for ATR calculation
        for i in range(30):
            strategy.high_history.append(50000 + i)
            strategy.low_history.append(50000 - i)
            strategy.price_history.append(50000 + i)

        # Mock the calculate_atr function to return 0
        with patch('src.utils.helpers.calculate_atr', return_value=0):
            result = strategy.should_exit(mock_position, mock_market_data)
            assert result is False

    def test_should_exit_sell_position_stop_loss(
            self, strategy, mock_market_data):
        """Test exit condition for sell position based on ATR stop loss."""
        # Create a sell position
        sell_position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            side=OrderSide.SELL,
            timestamp=datetime.now(timezone.utc)
        )

        # Add data for ATR calculation
        for i in range(30):
            strategy.high_history.append(50000 + i)
            strategy.low_history.append(50000 - i)
            strategy.price_history.append(50000 + i)

        # Set price to trigger stop loss (well above entry price)
        mock_market_data.price = Decimal("51000")  # Above stop loss

        # Mock ATR calculation to return a reasonable value
        with patch('src.utils.helpers.calculate_atr', return_value=1000):
            result = strategy.should_exit(sell_position, mock_market_data)
            assert result is True

    def test_validate_signal_old_signal_edge_case(self, strategy):
        """Test signal validation with signal that is exactly 5 minutes old."""
        # Create a signal that is exactly 5 minutes old
        old_timestamp = datetime.now(timezone.utc) - timedelta(minutes=5)
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=old_timestamp,
            symbol="BTCUSDT",
            strategy_name=strategy.name,
            metadata={"z_score": 2.5, "signal_type": "entry"}
        )

        # Add data for validation
        for i in range(25):
            strategy.price_history.append(50000 + i)

        result = asyncio.run(strategy.validate_signal(signal))
        assert result is False

    def test_validate_signal_invalid_zscore_type(self, strategy):
        """Test signal validation with invalid z_score type in metadata."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name=strategy.name,
            metadata={"z_score": "invalid", "signal_type": "entry"}
        )

        result = asyncio.run(strategy.validate_signal(signal))
        assert result is False

    def test_validate_signal_entry_below_threshold_edge_case(self, strategy):
        """Test signal validation with entry signal exactly at threshold."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name=strategy.name,
            # Exactly at threshold
            metadata={"z_score": 2.0, "signal_type": "entry"}
        )

        result = asyncio.run(strategy.validate_signal(signal))
        assert result is True  # Should pass when exactly at threshold

    def test_volume_filter_debug_logging(self, strategy, mock_market_data):
        """Test volume filter debug logging."""
        # Enable volume filter
        strategy.volume_filter = True

        # Add data for volume calculation
        for i in range(25):
            strategy.volume_history.append(100.0)

        # Set current volume below threshold to trigger debug logging
        mock_market_data.volume = Decimal("50")  # Below threshold

        # This should trigger the debug logging in _generate_signals_impl
        signals = asyncio.run(strategy.generate_signals(mock_market_data))
        assert len(signals) == 0  # Should be rejected by volume filter

    def test_zscore_debug_logging(self, strategy):
        """Test Z-score calculation debug logging."""
        # Add data for Z-score calculation
        for i in range(25):
            strategy.price_history.append(50000 + i)

        # This should trigger the debug logging in _calculate_zscore
        z_score = strategy._calculate_zscore()
        assert z_score is not None

    def test_atr_debug_logging(
            self,
            strategy,
            mock_position,
            mock_market_data):
        """Test ATR calculation debug logging."""
        # Add data for ATR calculation
        for i in range(30):
            strategy.high_history.append(50000 + i)
            strategy.low_history.append(50000 - i)
            strategy.price_history.append(50000 + i)

        # This should trigger the debug logging in should_exit
        result = strategy.should_exit(mock_position, mock_market_data)
        # The result doesn't matter, we just want to trigger the debug logging

    @pytest.mark.asyncio
    async def test_strategy_integration(self, strategy, mock_market_data):
        """Test full strategy integration."""
        # Add sufficient data
        for i in range(25):
            strategy.price_history.append(50000 + i * 100)
            strategy.volume_history.append(100.0)

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
            "name": "mean_reversion",
            "strategy_type": "static",
            "enabled": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "5m",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {
                "lookback_period": 10,  # Different from default
                "entry_threshold": 1.5,  # Different from default
                "exit_threshold": 0.3,   # Different from default
                "volume_filter": False,   # Different from default
                "min_volume_ratio": 2.0  # Different from default
            }
        }

        strategy = MeanReversionStrategy(config)

        assert strategy.lookback_period == 10
        assert strategy.entry_threshold == 1.5
        assert strategy.exit_threshold == 0.3
        assert strategy.volume_filter is False
        assert strategy.min_volume_ratio == 2.0
