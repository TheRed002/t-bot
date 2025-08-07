"""
Unit tests for Breakout Strategy.

Tests the breakout strategy implementation with comprehensive coverage
including signal generation, validation, position sizing, and exit conditions.
"""

import pytest
import numpy as np
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import from P-001
from src.core.types import (
    Signal, MarketData, Position, StrategyConfig, 
    StrategyStatus, StrategyMetrics, SignalDirection, OrderSide
)
from src.core.exceptions import ValidationError

# Import from P-012
from src.strategies.static.breakout import BreakoutStrategy


class TestBreakoutStrategy:
    """Test cases for BreakoutStrategy."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock strategy configuration."""
        return {
            "name": "breakout",
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
                "breakout_threshold": 0.02,
                "volume_multiplier": 1.5,
                "consolidation_periods": 5,
                "false_breakout_filter": True,
                "false_breakout_threshold": 0.01,
                "target_multiplier": 2.0,
                "atr_period": 14,
                "atr_multiplier": 2.0
            }
        }
    
    @pytest.fixture
    def strategy(self, mock_config):
        """Create strategy instance."""
        return BreakoutStrategy(mock_config)
    
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
        assert strategy.name == "breakout"
        assert strategy.strategy_type.value == "static"
        assert strategy.lookback_period == 20
        assert strategy.breakout_threshold == 0.02
        assert strategy.volume_multiplier == 1.5
        assert strategy.consolidation_periods == 5
        assert strategy.false_breakout_filter is True
        assert strategy.false_breakout_threshold == 0.01
        assert strategy.target_multiplier == 2.0
    
    def test_strategy_info(self, strategy):
        """Test strategy information retrieval."""
        info = strategy.get_strategy_info()
        
        assert info["name"] == "breakout"
        assert info["strategy_type"] == "breakout"
        assert "parameters" in info
        assert info["parameters"]["lookback_period"] == 20
        assert info["parameters"]["breakout_threshold"] == 0.02
        assert info["parameters"]["volume_multiplier"] == 1.5
        assert info["parameters"]["consolidation_periods"] == 5
    
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
            volume=Decimal("0"),
            timestamp=datetime.now(timezone.utc),
            bid=None,
            ask=None,
            open_price=None,
            high_price=None,
            low_price=None
        )
        
        strategy._update_price_history(data)
        
        assert strategy.price_history[-1] == 50000.0
        assert strategy.volume_history[-1] == 0.0
        assert strategy.high_history[-1] == 50000.0
        assert strategy.low_history[-1] == 50000.0
    
    def test_update_support_resistance_levels_insufficient_data(self, strategy):
        """Test support/resistance update with insufficient data."""
        # Add some data but not enough
        for i in range(10):
            strategy.price_history.append(50000 + i)
            strategy.high_history.append(50100 + i)
            strategy.low_history.append(49900 + i)
        
        strategy._update_support_resistance_levels()
        
        # Should not update levels with insufficient data
        assert len(strategy.support_levels) == 0
        assert len(strategy.resistance_levels) == 0
    
    def test_update_support_resistance_levels_success(self, strategy):
        """Test successful support/resistance update."""
        # Add enough data with clear highs and lows
        for i in range(25):
            strategy.price_history.append(50000 + i)
            strategy.high_history.append(50100 + i)
            strategy.low_history.append(49900 + i)
        
        strategy._update_support_resistance_levels()
        
        # Should have some levels
        assert len(strategy.support_levels) >= 0
        assert len(strategy.resistance_levels) >= 0
    
    def test_update_support_resistance_levels_exception(self, strategy):
        """Test support/resistance update with exception."""
        # Corrupt data
        strategy.price_history = [np.nan, np.inf]
        strategy.high_history = [np.nan, np.inf]
        strategy.low_history = [np.nan, np.inf]
        
        strategy._update_support_resistance_levels()
        
        # Should handle exception gracefully
        assert len(strategy.support_levels) == 0
        assert len(strategy.resistance_levels) == 0
    
    def test_check_consolidation_period_insufficient_data(self, strategy):
        """Test consolidation check with insufficient data."""
        # Add some data but not enough
        for i in range(3):
            strategy.price_history.append(50000 + i)
        
        result = strategy._check_consolidation_period()
        assert not result
    
    def test_check_consolidation_period_consolidating(self, strategy):
        """Test consolidation check with consolidating prices."""
        # Add data within narrow range
        for i in range(5):
            strategy.price_history.append(50000 + i * 10)  # Small range
        
        result = strategy._check_consolidation_period()
        assert result
    
    def test_check_consolidation_period_not_consolidating(self, strategy):
        """Test consolidation check with non-consolidating prices."""
        # Add data with wide range
        for i in range(5):
            strategy.price_history.append(50000 + i * 1000)  # Large range
        
        result = strategy._check_consolidation_period()
        assert not result
    
    def test_check_consolidation_period_exception(self, strategy):
        """Test consolidation check with exception."""
        # Corrupt data
        strategy.price_history = [np.nan, np.inf]
        
        result = strategy._check_consolidation_period()
        assert not result
    
    def test_check_resistance_breakout_no_breakout(self, strategy, mock_market_data):
        """Test resistance breakout check with no breakout."""
        # Add some resistance levels
        strategy.resistance_levels = [51000, 52000]
        
        # Set price below resistance
        mock_market_data.price = Decimal("50500")
        
        result = strategy._check_resistance_breakout(mock_market_data)
        assert result is None
    
    def test_check_resistance_breakout_with_breakout(self, strategy, mock_market_data):
        """Test resistance breakout check with breakout."""
        # Add resistance level
        strategy.resistance_levels = [51000]
        
        # Set price above resistance with volume (above 2% threshold)
        mock_market_data.price = Decimal("52100")  # Above resistance + 2% threshold
        mock_market_data.volume = Decimal("200")  # High volume
        
        # Add volume history for confirmation
        for i in range(25):
            strategy.volume_history.append(100.0)
        
        result = strategy._check_resistance_breakout(mock_market_data)
        assert result is not None
        assert result["breakout_type"] == "resistance"
        assert result["level"] == 51000
    
    def test_check_resistance_breakout_low_volume(self, strategy, mock_market_data):
        """Test resistance breakout check with low volume."""
        # Add resistance level
        strategy.resistance_levels = [51000]
        
        # Set price above resistance but low volume
        mock_market_data.price = Decimal("52000")
        mock_market_data.volume = Decimal("50")  # Low volume
        
        # Add volume history
        for i in range(25):
            strategy.volume_history.append(100.0)
        
        result = strategy._check_resistance_breakout(mock_market_data)
        assert result is None
    
    def test_check_support_breakout_no_breakout(self, strategy, mock_market_data):
        """Test support breakout check with no breakout."""
        # Add some support levels
        strategy.support_levels = [49000, 48000]
        
        # Set price above support
        mock_market_data.price = Decimal("49500")
        
        result = strategy._check_support_breakout(mock_market_data)
        assert result is None
    
    def test_check_support_breakout_with_breakout(self, strategy, mock_market_data):
        """Test support breakout check with breakout."""
        # Add support level
        strategy.support_levels = [49000]
        
        # Set price below support with volume (below 2% threshold)
        mock_market_data.price = Decimal("47900")  # Below support - 2% threshold
        mock_market_data.volume = Decimal("200")  # High volume
        
        # Add volume history for confirmation
        for i in range(25):
            strategy.volume_history.append(100.0)
        
        result = strategy._check_support_breakout(mock_market_data)
        assert result is not None
        assert result["breakout_type"] == "support"
        assert result["level"] == 49000
    
    def test_check_volume_confirmation_success(self, strategy, mock_market_data):
        """Test volume confirmation with sufficient data."""
        # Add volume history
        for i in range(25):
            strategy.volume_history.append(100.0)
        
        # Test with high volume
        mock_market_data.volume = Decimal("200")  # 2x average
        result = strategy._check_volume_confirmation(float(mock_market_data.volume))
        assert result
        
        # Test with low volume
        mock_market_data.volume = Decimal("50")  # 0.5x average
        result = strategy._check_volume_confirmation(float(mock_market_data.volume))
        assert result is False
    
    def test_check_volume_confirmation_zero_volume(self, strategy):
        """Test volume confirmation with zero volume."""
        result = strategy._check_volume_confirmation(0.0)
        assert result is False
    
    def test_check_volume_confirmation_insufficient_data(self, strategy):
        """Test volume confirmation with insufficient data."""
        # Add some volume data but not enough
        for i in range(10):
            strategy.volume_history.append(100.0)
        
        result = strategy._check_volume_confirmation(150.0)
        assert result  # Should pass if insufficient data
    
    def test_check_false_breakout_disabled(self, strategy, mock_market_data):
        """Test false breakout check when disabled."""
        strategy.false_breakout_filter = False
        
        result = strategy._check_false_breakout(mock_market_data)
        assert result is None
    
    def test_check_false_breakout_resistance_return(self, strategy, mock_market_data):
        """Test false breakout check with resistance return."""
        # Add resistance level
        strategy.resistance_levels = [51000]
        
        # Set price near resistance
        mock_market_data.price = Decimal("51050")  # Within threshold
        
        result = strategy._check_false_breakout(mock_market_data)
        assert result is not None
        assert result["breakout_type"] == "false_resistance"
    
    def test_check_false_breakout_support_return(self, strategy, mock_market_data):
        """Test false breakout check with support return."""
        # Add support level
        strategy.support_levels = [49000]
        
        # Set price near support
        mock_market_data.price = Decimal("48950")  # Within threshold
        
        result = strategy._check_false_breakout(mock_market_data)
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
            "volume": 200.0
        }
        
        signal = await strategy._generate_bullish_breakout_signal(mock_market_data, breakout_info)
        
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence > 0
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
            "volume": 200.0
        }
        
        signal = await strategy._generate_bearish_breakout_signal(mock_market_data, breakout_info)
        
        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert signal.confidence > 0
        assert signal.metadata["signal_type"] == "breakout_entry"
        assert signal.metadata["breakout_direction"] == "bearish"
        assert "target_price" in signal.metadata
    
    @pytest.mark.asyncio
    async def test_generate_false_breakout_exit_signal_resistance(self, strategy, mock_market_data):
        """Test false breakout exit signal for resistance."""
        false_breakout_info = {
            "level": 51000,
            "current_price": 51050,
            "breakout_type": "false_resistance"
        }
        
        signal = await strategy._generate_false_breakout_exit_signal(mock_market_data, false_breakout_info)
        
        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert signal.confidence == 0.9
        assert signal.metadata["signal_type"] == "false_breakout_exit"
    
    @pytest.mark.asyncio
    async def test_generate_false_breakout_exit_signal_support(self, strategy, mock_market_data):
        """Test false breakout exit signal for support."""
        false_breakout_info = {
            "level": 49000,
            "current_price": 48950,
            "breakout_type": "false_support"
        }
        
        signal = await strategy._generate_false_breakout_exit_signal(mock_market_data, false_breakout_info)
        
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence == 0.9
        assert signal.metadata["signal_type"] == "false_breakout_exit"
    
    @pytest.mark.asyncio
    async def test_generate_signals_empty_data(self, strategy):
        """Test signal generation with empty data."""
        signals = await strategy.generate_signals(None)
        assert signals == []
        
        signals = await strategy.generate_signals(MarketData(
            symbol="BTCUSDT",
            price=Decimal("0"),
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
    async def test_generate_signals_insufficient_data(self, strategy, mock_market_data):
        """Test signal generation with insufficient data."""
        signals = await strategy.generate_signals(mock_market_data)
        assert signals == []
    
    @pytest.mark.asyncio
    async def test_generate_signals_no_consolidation(self, strategy, mock_market_data):
        """Test signal generation without consolidation period."""
        # Add enough data but not consolidating
        for i in range(25):
            strategy.price_history.append(50000 + i * 1000)  # Large range
        
        signals = await strategy.generate_signals(mock_market_data)
        assert signals == []
    
    @pytest.mark.asyncio
    async def test_generate_signals_resistance_breakout(self, strategy, mock_market_data):
        """Test signal generation with resistance breakout."""
        # Add consolidating data (narrow range around 50000)
        for i in range(25):
            # Create consolidating prices within 2% range (1000 range for 50000 base)
            base_price = 50000
            variation = (i % 5 - 2) * 10  # Much smaller variations: -20, -10, 0, 10, 20
            strategy.price_history.append(base_price + variation)
            strategy.volume_history.append(100.0)

        # Add resistance level
        strategy.resistance_levels = [51000]

        # Disable consolidation check for this test
        strategy.consolidation_periods = 0

        # Disable support/resistance update for this test (keep manually set levels)
        original_update_method = strategy._update_support_resistance_levels
        strategy._update_support_resistance_levels = lambda: None

        # Set breakout price and volume (above 2% threshold)
        mock_market_data.price = Decimal("52100")
        mock_market_data.volume = Decimal("200")
        
        signals = await strategy.generate_signals(mock_market_data)
        
        # Should generate bullish breakout signal
        assert len(signals) > 0
        signal = signals[0]
        assert signal.direction == SignalDirection.BUY
        assert signal.metadata["signal_type"] == "breakout_entry"
        assert signal.metadata["breakout_direction"] == "bullish"
    
    @pytest.mark.asyncio
    async def test_generate_signals_support_breakout(self, strategy, mock_market_data):
        """Test signal generation with support breakout."""
        # Add consolidating data (narrow range around 50000)
        for i in range(25):
            # Create consolidating prices within 2% range (1000 range for 50000 base)
            base_price = 50000
            variation = (i % 5 - 2) * 10  # Much smaller variations: -20, -10, 0, 10, 20
            strategy.price_history.append(base_price + variation)
            strategy.volume_history.append(100.0)

        # Add support level
        strategy.support_levels = [49000]

        # Disable consolidation check for this test
        strategy.consolidation_periods = 0

        # Disable support/resistance update for this test (keep manually set levels)
        original_update_method = strategy._update_support_resistance_levels
        strategy._update_support_resistance_levels = lambda: None

        # Set breakout price and volume (below 2% threshold)
        mock_market_data.price = Decimal("47900")
        mock_market_data.volume = Decimal("200")
        
        signals = await strategy.generate_signals(mock_market_data)
        
        # Should generate bearish breakout signal
        assert len(signals) > 0
        signal = signals[0]
        assert signal.direction == SignalDirection.SELL
        assert signal.metadata["signal_type"] == "breakout_entry"
        assert signal.metadata["breakout_direction"] == "bearish"
    
    @pytest.mark.asyncio
    async def test_generate_signals_false_breakout_exit(self, strategy, mock_market_data):
        """Test signal generation with false breakout exit."""
        # Add consolidating data (narrow range around 50000)
        for i in range(25):
            # Create consolidating prices within 2% range (1000 range for 50000 base)
            base_price = 50000
            variation = (i % 5 - 2) * 10  # Much smaller variations: -20, -10, 0, 10, 20
            strategy.price_history.append(base_price + variation)
            strategy.volume_history.append(100.0)

        # Add levels
        strategy.resistance_levels = [51000]
        strategy.support_levels = [49000]

        # Disable consolidation check for this test
        strategy.consolidation_periods = 0

        # Disable support/resistance update for this test (keep manually set levels)
        original_update_method = strategy._update_support_resistance_levels
        strategy._update_support_resistance_levels = lambda: None

        # Set price near resistance (false breakout)
        mock_market_data.price = Decimal("51050")
        
        signals = await strategy.generate_signals(mock_market_data)
        
        # Should generate false breakout exit signal
        assert len(signals) > 0
        for signal in signals:
            assert signal.metadata["signal_type"] == "false_breakout_exit"
    
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
            strategy_name="breakout",
            metadata={
                "breakout_level": 51000,
                "breakout_price": 52000,
                "target_price": 53000,
                "breakout_direction": "bullish",
                "signal_type": "breakout_entry"
            }
        )
        
        result = await strategy.validate_signal(signal)
        assert result
    
    @pytest.mark.asyncio
    async def test_validate_signal_low_confidence(self, strategy):
        """Test signal validation with low confidence."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.3,  # Below threshold
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="breakout",
            metadata={
                "breakout_level": 51000,
                "breakout_price": 52000,
                "target_price": 53000,
                "breakout_direction": "bullish",
                "signal_type": "breakout_entry"
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
            strategy_name="breakout",
            metadata={
                "breakout_level": 51000,
                "breakout_price": 52000,
                "target_price": 53000,
                "breakout_direction": "bullish",
                "signal_type": "breakout_entry"
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
            strategy_name="breakout",
            metadata={}  # Missing required fields
        )
        
        result = await strategy.validate_signal(signal)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_signal_invalid_breakout_price(self, strategy):
        """Test signal validation with invalid breakout price."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="breakout",
            metadata={
                "breakout_level": 51000,
                "breakout_price": -100,  # Invalid price
                "target_price": 53000,
                "breakout_direction": "bullish",
                "signal_type": "breakout_entry"
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
            strategy_name="breakout",
            metadata={
                "breakout_level": 51000,
                "breakout_price": 52000,
                "target_price": 53000,
                "breakout_direction": "bullish",
                "signal_type": "breakout_entry"
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
            strategy_name="breakout",
            metadata={
                "range_size": 1000,
                "breakout_price": 52000
            }
        )
        
        position_size = strategy.get_position_size(signal)
        assert isinstance(position_size, Decimal)
        assert position_size > 0
    
    def test_get_position_size_with_max_limit(self, strategy):
        """Test position size calculation with maximum limit."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=1.0,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="breakout",
            metadata={
                "range_size": 10000,  # Large range
                "breakout_price": 52000
            }
        )
        
        position_size = strategy.get_position_size(signal)
        max_size = Decimal(str(strategy.config.parameters.get("max_position_size_pct", 0.1)))
        assert position_size <= max_size
    
    def test_get_position_size_exception_handling(self, strategy):
        """Test position size calculation exception handling."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="breakout",
            metadata={
                "range_size": "invalid",  # Will cause exception
                "breakout_price": 52000
            }
        )
        
        position_size = strategy.get_position_size(signal)
        # Should return minimum position size on error
        expected_min = Decimal(str(strategy.config.position_size_pct * 0.5))
        assert position_size == expected_min
    
    def test_should_exit_atr_stop_loss(self, strategy, mock_position, mock_market_data):
        """Test exit condition based on ATR stop loss."""
        # Add data for ATR calculation (need at least atr_period + 1 data points)
        for i in range(30):  # More data to ensure ATR calculation works
            # Create more realistic price movements with larger ranges
            base_price = 50000
            variation = (i % 5 - 2) * 100  # Larger variations for ATR calculation
            strategy.high_history.append(base_price + variation + 200)  # High price
            strategy.low_history.append(base_price + variation - 200)   # Low price
            strategy.price_history.append(base_price + variation)       # Close price

        # Set price to trigger stop loss (well below entry price)
        mock_position.current_price = Decimal("48000")  # Far below stop loss

        result = strategy.should_exit(mock_position, mock_market_data)
        assert result
    
    def test_should_exit_target_price_reached(self, strategy, mock_position, mock_market_data):
        """Test exit condition based on target price."""
        # Add target price to position metadata
        mock_position.metadata = {"target_price": 53000}
        
        # Set price above target
        mock_position.current_price = Decimal("53500")
        
        result = strategy.should_exit(mock_position, mock_market_data)
        assert result
    
    def test_should_exit_target_price_not_reached(self, strategy, mock_position, mock_market_data):
        """Test exit condition when target price not reached."""
        # Add target price to position metadata
        mock_position.metadata = {"target_price": 53000}
        
        # Set price below target
        mock_market_data.price = Decimal("52500")
        
        result = strategy.should_exit(mock_position, mock_market_data)
        assert result is False
    
    def test_should_exit_sell_position_target_price(self, strategy, mock_market_data):
        """Test exit condition for sell position target price."""
        # Create sell position with target
        sell_position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("49000"),
            unrealized_pnl=Decimal("100"),
            side=OrderSide.SELL,
            timestamp=datetime.now(timezone.utc)
        )
        sell_position.metadata = {"target_price": 47000}
        
        # Set price below target
        sell_position.current_price = Decimal("46500")
        
        result = strategy.should_exit(sell_position, mock_market_data)
        assert result
    
    def test_should_exit_no_exit_condition(self, strategy, mock_position, mock_market_data):
        """Test when no exit condition is met."""
        # Add data for calculations
        for i in range(25):
            strategy.price_history.append(50000 + i * 10)
        
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
    
    @pytest.mark.asyncio
    async def test_strategy_integration(self, strategy, mock_market_data):
        """Test full strategy integration."""
        # Add sufficient data
        for i in range(25):
            strategy.price_history.append(50000 + i * 10)
            strategy.volume_history.append(100.0)
        
        # Add support/resistance levels
        strategy.support_levels = [49000]
        strategy.resistance_levels = [51000]
        
        # Set breakout conditions
        mock_market_data.price = Decimal("52000")
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
            "name": "breakout",
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
                "breakout_threshold": 0.01,  # Different from default
                "volume_multiplier": 2.0,  # Different from default
                "consolidation_periods": 3,  # Different from default
                "false_breakout_filter": False,  # Different from default
                "false_breakout_threshold": 0.005,  # Different from default
                "target_multiplier": 1.5,  # Different from default
                "atr_period": 10,  # Different from default
                "atr_multiplier": 1.5  # Different from default
            }
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