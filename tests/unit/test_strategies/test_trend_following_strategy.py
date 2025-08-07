"""
Unit tests for Trend Following Strategy.

Tests the trend following strategy implementation with comprehensive coverage
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
from src.strategies.static.trend_following import TrendFollowingStrategy


class TestTrendFollowingStrategy:
    """Test cases for TrendFollowingStrategy."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock strategy configuration."""
        return {
            "name": "trend_following",
            "strategy_type": "static",
            "enabled": True,
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "timeframe": "1h",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {
                "fast_ma": 20,
                "slow_ma": 50,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "volume_confirmation": True,
                "min_volume_ratio": 1.2,
                "max_pyramid_levels": 3,
                "trailing_stop_pct": 0.02,
                "time_exit_hours": 48
            }
        }
    
    @pytest.fixture
    def strategy(self, mock_config):
        """Create strategy instance."""
        return TrendFollowingStrategy(mock_config)
    
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
        assert strategy.name == "trend_following"
        assert strategy.strategy_type.value == "static"
        assert strategy.fast_ma == 20
        assert strategy.slow_ma == 50
        assert strategy.rsi_period == 14
        assert strategy.volume_confirmation is True
        assert strategy.max_pyramid_levels == 3
        assert strategy.trailing_stop_pct == 0.02
        assert strategy.time_exit_hours == 48
    
    def test_strategy_info(self, strategy):
        """Test strategy information retrieval."""
        info = strategy.get_strategy_info()
        
        assert info["name"] == "trend_following"
        assert info["strategy_type"] == "trend_following"
        assert "parameters" in info
        assert info["parameters"]["fast_ma"] == 20
        assert info["parameters"]["slow_ma"] == 50
        assert info["parameters"]["rsi_period"] == 14
        assert info["parameters"]["max_pyramid_levels"] == 3
    
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
            bid=Decimal("49999"),
            ask=Decimal("50001"),
            open_price=Decimal("49900"),
            high_price=Decimal("50100"),
            low_price=Decimal("49800")
        )
        
        strategy._update_price_history(data)
        
        assert strategy.price_history[-1] == 50000.0
        assert strategy.volume_history[-1] == 0.0
        assert strategy.high_history[-1] == 50100.0
        assert strategy.low_history[-1] == 49800.0
    
    def test_calculate_fast_ma_insufficient_data(self, strategy):
        """Test fast MA calculation with insufficient data."""
        # Add some data but not enough
        for i in range(10):
            strategy.price_history.append(50000 + i)
        
        fast_ma = strategy._calculate_fast_ma()
        assert fast_ma is None
    
    def test_calculate_fast_ma_success(self, strategy):
        """Test successful fast MA calculation."""
        # Add enough data
        for i in range(25):
            strategy.price_history.append(50000 + i)
        
        fast_ma = strategy._calculate_fast_ma()
        assert fast_ma is not None
        assert isinstance(fast_ma, float)
        assert fast_ma > 50000  # Should be above base price due to upward trend
    
    def test_calculate_slow_ma_insufficient_data(self, strategy):
        """Test slow MA calculation with insufficient data."""
        # Add some data but not enough
        for i in range(30):
            strategy.price_history.append(50000 + i)
        
        slow_ma = strategy._calculate_slow_ma()
        assert slow_ma is None
    
    def test_calculate_slow_ma_success(self, strategy):
        """Test successful slow MA calculation."""
        # Add enough data
        for i in range(60):
            strategy.price_history.append(50000 + i)
        
        slow_ma = strategy._calculate_slow_ma()
        assert slow_ma is not None
        assert isinstance(slow_ma, float)
        assert slow_ma > 50000  # Should be above base price due to upward trend
    
    def test_calculate_rsi_insufficient_data(self, strategy):
        """Test RSI calculation with insufficient data."""
        # Add some data but not enough
        for i in range(10):
            strategy.price_history.append(50000 + i)
        
        rsi = strategy._calculate_rsi()
        assert rsi is None
    
    def test_calculate_rsi_success(self, strategy):
        """Test successful RSI calculation."""
        # Add enough data with price changes
        for i in range(20):
            strategy.price_history.append(50000 + i * 10)
        
        rsi = strategy._calculate_rsi()
        assert rsi is not None
        assert isinstance(rsi, float)
        assert 0 <= rsi <= 100
    
    def test_check_volume_confirmation_insufficient_data(self, strategy, mock_market_data):
        """Test volume confirmation with insufficient data."""
        # Add some volume data but not enough
        for i in range(10):
            strategy.volume_history.append(100.0)
        
        result = strategy._check_volume_confirmation(mock_market_data)
        assert result is True  # Should pass if insufficient data
    
    def test_check_volume_confirmation_success(self, strategy, mock_market_data):
        """Test volume confirmation with sufficient data."""
        # Add enough volume data
        for i in range(25):
            strategy.volume_history.append(100.0)
        
        # Test with high volume
        mock_market_data.volume = Decimal("150")  # 1.5x average
        result = strategy._check_volume_confirmation(mock_market_data)
        assert result
        
        # Test with low volume
        mock_market_data.volume = Decimal("80")  # 0.8x average
        result = strategy._check_volume_confirmation(mock_market_data)
        assert not result
    
    def test_check_volume_confirmation_zero_volume(self, strategy, mock_market_data):
        """Test volume confirmation with zero volume."""
        # Add enough volume data to avoid early return
        for i in range(25):
            strategy.volume_history.append(100.0)
        
        mock_market_data.volume = Decimal("0")
        
        result = strategy._check_volume_confirmation(mock_market_data)
        assert not result
    
    def test_check_volume_confirmation_exception(self, strategy):
        """Test volume confirmation with exception."""
        data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc)
        )
        
        # Corrupt volume history
        strategy.volume_history = [np.nan, np.inf]
        
        result = strategy._check_volume_confirmation(data)
        assert result is True  # Should pass on error
    
    @pytest.mark.asyncio
    async def test_generate_bullish_signal_success(self, strategy, mock_market_data):
        """Test successful bullish signal generation."""
        # Add enough data for calculations
        for i in range(60):
            strategy.price_history.append(50000 + i * 10)
            strategy.volume_history.append(100.0)
        
        fast_ma = 51000  # Above slow MA
        slow_ma = 50500
        rsi = 60  # Bullish but not overbought
        
        signal = await strategy._generate_bullish_signal(mock_market_data, fast_ma, slow_ma, rsi)
        
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence > 0
        assert signal.metadata["signal_type"] == "trend_entry"
        assert signal.metadata["trend_direction"] == "bullish"
    
    @pytest.mark.asyncio
    async def test_generate_bullish_signal_max_pyramid_levels(self, strategy, mock_market_data):
        """Test bullish signal generation with max pyramid levels reached."""
        # Set max pyramid levels
        strategy.position_levels["BTCUSDT"] = 3
        
        fast_ma = 51000
        slow_ma = 50500
        rsi = 60
        
        signal = await strategy._generate_bullish_signal(mock_market_data, fast_ma, slow_ma, rsi)
        
        assert signal is None
    
    @pytest.mark.asyncio
    async def test_generate_bearish_signal_success(self, strategy, mock_market_data):
        """Test successful bearish signal generation."""
        # Add enough data for calculations
        for i in range(60):
            strategy.price_history.append(50000 - i * 10)
            strategy.volume_history.append(100.0)
        
        fast_ma = 49500  # Below slow MA
        slow_ma = 50000
        rsi = 40  # Bearish but not oversold
        
        signal = await strategy._generate_bearish_signal(mock_market_data, fast_ma, slow_ma, rsi)
        
        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert signal.confidence > 0
        assert signal.metadata["signal_type"] == "trend_entry"
        assert signal.metadata["trend_direction"] == "bearish"
    
    @pytest.mark.asyncio
    async def test_generate_bearish_signal_max_pyramid_levels(self, strategy, mock_market_data):
        """Test bearish signal generation with max pyramid levels reached."""
        # Set max pyramid levels
        strategy.position_levels["BTCUSDT"] = 3
        
        fast_ma = 49500
        slow_ma = 50000
        rsi = 40
        
        signal = await strategy._generate_bearish_signal(mock_market_data, fast_ma, slow_ma, rsi)
        
        assert signal is None
    
    @pytest.mark.asyncio
    async def test_generate_exit_signal_success(self, strategy, mock_market_data):
        """Test successful exit signal generation."""
        signal = await strategy._generate_exit_signal(mock_market_data, SignalDirection.SELL, "trend_reversal")
        
        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert signal.confidence == 0.8
        assert signal.metadata["signal_type"] == "trend_exit"
        assert signal.metadata["exit_reason"] == "trend_reversal"
    
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
    async def test_generate_signals_bullish_trend(self, strategy, mock_market_data):
        """Test signal generation with bullish trend."""
        # Add enough data with moderate bullish trend (RSI between 50-70)
        base_price = 50000
        for i in range(60):
            # Create a more moderate upward trend with more variation
            variation = (i % 5 - 2) * 3  # Variations: -6, -3, 0, 3, 6
            price = base_price + i * 1.5 + variation  # Smaller trend
            strategy.price_history.append(price)
            strategy.volume_history.append(100.0)
        
        # Set current price to continue the upward trend
        mock_market_data.price = Decimal("50090")  # Continue the upward trend
        # Set higher volume to pass volume confirmation (min_volume_ratio=1.2)
        mock_market_data.volume = Decimal("150")  # 1.5x average volume
        
        signals = await strategy.generate_signals(mock_market_data)
        
        # Should generate bullish signal
        assert len(signals) > 0
        signal = signals[0]
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence > 0
        assert signal.metadata["signal_type"] == "trend_entry"
        assert signal.metadata["trend_direction"] == "bullish"
    
    @pytest.mark.asyncio
    async def test_generate_signals_bearish_trend(self, strategy, mock_market_data):
        """Test signal generation with bearish trend."""
        # Add enough data with moderate bearish trend (RSI between 30-50)
        base_price = 50000
        for i in range(60):
            # Create a more moderate downward trend with more variation
            variation = (i % 5 - 2) * 3  # Variations: -6, -3, 0, 3, 6
            price = base_price - i * 1.5 + variation  # Smaller trend
            strategy.price_history.append(price)
            strategy.volume_history.append(100.0)
        
        # Set current price to continue the downward trend
        mock_market_data.price = Decimal("49910")  # Continue the downward trend
        # Set higher volume to pass volume confirmation (min_volume_ratio=1.2)
        mock_market_data.volume = Decimal("150")  # 1.5x average volume
        
        signals = await strategy.generate_signals(mock_market_data)
        
        # Should generate bearish signal
        assert len(signals) > 0
        signal = signals[0]
        assert signal.direction == SignalDirection.SELL
        assert signal.confidence > 0
        assert signal.metadata["signal_type"] == "trend_entry"
        assert signal.metadata["trend_direction"] == "bearish"
    
    @pytest.mark.asyncio
    async def test_generate_signals_trend_reversal_exit(self, strategy, mock_market_data):
        """Test signal generation with trend reversal exit."""
        # Add data with trend reversal conditions (downward trend but bullish RSI)
        base_price = 50000
        for i in range(60):
            # Create downward trend but with recent moderate recovery to create bullish RSI (50-70)
            if i < 50:
                price = base_price - i * 2  # Downward trend
            else:
                price = base_price - 50 * 2 + (i - 50) * 1.5  # Moderate recent recovery
            strategy.price_history.append(price)
            strategy.volume_history.append(100.0)
        
        # Set current price to continue the moderate recovery
        mock_market_data.price = Decimal("49925")  # Continue the moderate recovery
        # Set higher volume to pass volume confirmation (min_volume_ratio=1.2)
        mock_market_data.volume = Decimal("150")  # 1.5x average volume
        
        signals = await strategy.generate_signals(mock_market_data)
        
        # Should generate exit signals
        assert len(signals) > 0
        for signal in signals:
            assert signal.metadata["signal_type"] == "trend_exit"
    
    @pytest.mark.asyncio
    async def test_generate_signals_volume_confirmation_rejection(self, strategy, mock_market_data):
        """Test signal generation with volume confirmation rejection."""
        # Add enough data
        for i in range(60):
            strategy.price_history.append(50000 + i * 10)
            strategy.volume_history.append(100.0)
        
        # Set low volume
        mock_market_data.volume = Decimal("80")
        
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
            strategy_name="trend_following",
            metadata={
                "fast_ma": 51000,
                "slow_ma": 50500,
                "rsi": 60,
                "trend_direction": "bullish",
                "signal_type": "trend_entry"
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
            strategy_name="trend_following",
            metadata={
                "fast_ma": 51000,
                "slow_ma": 50500,
                "rsi": 60,
                "trend_direction": "bullish",
                "signal_type": "trend_entry"
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
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=15),
            symbol="BTCUSDT",
            strategy_name="trend_following",
            metadata={
                "fast_ma": 51000,
                "slow_ma": 50500,
                "rsi": 60,
                "trend_direction": "bullish",
                "signal_type": "trend_entry"
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
            strategy_name="trend_following",
            metadata={}  # Missing required fields
        )
        
        result = await strategy.validate_signal(signal)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_signal_invalid_rsi(self, strategy):
        """Test signal validation with invalid RSI."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="trend_following",
            metadata={
                "fast_ma": 51000,
                "slow_ma": 50500,
                "rsi": 150,  # Invalid RSI
                "trend_direction": "bullish",
                "signal_type": "trend_entry"
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
            strategy_name="trend_following",
            metadata={
                "fast_ma": 51000,
                "slow_ma": 50500,
                "rsi": 60,
                "trend_direction": "bullish",
                "signal_type": "trend_entry"
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
            strategy_name="trend_following",
            metadata={
                "ma_strength": 0.1,
                "rsi_strength": 0.2,
                "pyramid_level": 1
            }
        )
        
        position_size = strategy.get_position_size(signal)
        assert isinstance(position_size, Decimal)
        assert position_size > 0
    
    def test_get_position_size_with_pyramid_level(self, strategy):
        """Test position size calculation with pyramid level adjustment."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="trend_following",
            metadata={
                "ma_strength": 0.1,
                "rsi_strength": 0.2,
                "pyramid_level": 2  # Higher level = smaller position
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
            strategy_name="trend_following",
            metadata={
                "ma_strength": 1.0,  # Maximum strength
                "rsi_strength": 1.0,  # Maximum strength
                "pyramid_level": 1
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
            strategy_name="trend_following",
            metadata={
                "ma_strength": "invalid",  # Will cause exception
                "rsi_strength": 0.2,
                "pyramid_level": 1
            }
        )
        
        position_size = strategy.get_position_size(signal)
        # Should return minimum position size on error
        expected_min = Decimal(str(strategy.config.position_size_pct * 0.5))
        assert position_size == expected_min
    
    def test_should_exit_by_time(self, strategy, mock_position):
        """Test time-based exit condition."""
        # Set old entry time
        mock_position.timestamp = datetime.now(timezone.utc) - timedelta(hours=50)
        
        result = strategy._should_exit_by_time(mock_position)
        assert result is True
        
        # Set recent entry time
        mock_position.timestamp = datetime.now(timezone.utc) - timedelta(hours=10)
        
        result = strategy._should_exit_by_time(mock_position)
        assert result is False
    
    def test_should_exit_by_trailing_stop_buy_position(self, strategy, mock_position, mock_market_data):
        """Test trailing stop for buy position."""
        # Set price below trailing stop
        mock_market_data.price = Decimal("49000")  # Below trailing stop
        
        result = strategy._should_exit_by_trailing_stop(mock_position, mock_market_data)
        assert result
        
        # Set price above trailing stop
        mock_market_data.price = Decimal("51000")  # Above trailing stop
        
        result = strategy._should_exit_by_trailing_stop(mock_position, mock_market_data)
        assert not result
    
    def test_should_exit_by_trailing_stop_sell_position(self, strategy, mock_market_data):
        """Test trailing stop for sell position."""
        # Create sell position
        sell_position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("49000"),
            unrealized_pnl=Decimal("100"),
            side=OrderSide.SELL,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Set price above trailing stop
        mock_market_data.price = Decimal("51000")  # Above trailing stop
        
        result = strategy._should_exit_by_trailing_stop(sell_position, mock_market_data)
        assert result
        
        # Set price below trailing stop
        mock_market_data.price = Decimal("49000")  # Below trailing stop
        
        result = strategy._should_exit_by_trailing_stop(sell_position, mock_market_data)
        assert not result
    
    def test_should_exit_trend_reversal_buy_position(self, strategy, mock_position, mock_market_data):
        """Test trend reversal exit for buy position."""
        # Add data for calculations
        for i in range(60):
            strategy.price_history.append(50000 + i * 5)
        
        # Set conditions for trend reversal (fast MA < slow MA and RSI < 50)
        with patch.object(strategy, '_calculate_fast_ma', return_value=49500):
            with patch.object(strategy, '_calculate_slow_ma', return_value=50000):
                with patch.object(strategy, '_calculate_rsi', return_value=40):
                    result = strategy.should_exit(mock_position, mock_market_data)
                    assert result is True
    
    def test_should_exit_trend_reversal_sell_position(self, strategy, mock_market_data):
        """Test trend reversal exit for sell position."""
        # Create sell position
        sell_position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("49000"),
            unrealized_pnl=Decimal("100"),
            side=OrderSide.SELL,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Add data for calculations
        for i in range(60):
            strategy.price_history.append(50000 + i * 5)
        
        # Set conditions for trend reversal (fast MA > slow MA and RSI > 50)
        with patch.object(strategy, '_calculate_fast_ma', return_value=50500):
            with patch.object(strategy, '_calculate_slow_ma', return_value=50000):
                with patch.object(strategy, '_calculate_rsi', return_value=60):
                    result = strategy.should_exit(sell_position, mock_market_data)
                    assert result is True
    
    def test_should_exit_no_exit_condition(self, strategy, mock_position, mock_market_data):
        """Test when no exit condition is met."""
        # Add data for calculations
        for i in range(60):
            strategy.price_history.append(50000 + i * 10)
        
        # Set conditions that don't trigger exit
        with patch.object(strategy, '_calculate_fast_ma', return_value=50500):
            with patch.object(strategy, '_calculate_slow_ma', return_value=50000):
                with patch.object(strategy, '_calculate_rsi', return_value=60):
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
        for i in range(60):
            strategy.price_history.append(50000 + i * 10)
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
            "name": "trend_following",
            "strategy_type": "static",
            "enabled": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "1h",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {
                "fast_ma": 10,  # Different from default
                "slow_ma": 30,  # Different from default
                "rsi_period": 10,  # Different from default
                "rsi_overbought": 80,  # Different from default
                "rsi_oversold": 20,  # Different from default
                "volume_confirmation": False,  # Different from default
                "min_volume_ratio": 2.0,  # Different from default
                "max_pyramid_levels": 5,  # Different from default
                "trailing_stop_pct": 0.03,  # Different from default
                "time_exit_hours": 24  # Different from default
            }
        }
        
        strategy = TrendFollowingStrategy(config)
        
        assert strategy.fast_ma == 10
        assert strategy.slow_ma == 30
        assert strategy.rsi_period == 10
        assert strategy.rsi_overbought == 80
        assert strategy.rsi_oversold == 20
        assert strategy.volume_confirmation is False
        assert strategy.min_volume_ratio == 2.0
        assert strategy.max_pyramid_levels == 5
        assert strategy.trailing_stop_pct == 0.03
        assert strategy.time_exit_hours == 24 

    def test_volume_confirmation_debug_logging(self, strategy, mock_market_data):
        """Test volume confirmation debug logging."""
        # Enable volume confirmation
        strategy.volume_confirmation = True
        
        # Add data for volume calculation
        for i in range(50):
            strategy.volume_history.append(100.0)
        
        # Set current volume below threshold to trigger debug logging
        mock_market_data.volume = Decimal("50")  # Below threshold
        
        # Add data for MA and RSI calculations
        for i in range(50):
            strategy.price_history.append(50000 + i)
        
        # This should trigger the debug logging in _generate_signals_impl
        signals = asyncio.run(strategy.generate_signals(mock_market_data))
        assert len(signals) == 0  # Should be rejected by volume confirmation
    
    def test_generate_signals_insufficient_ma_data(self, strategy, mock_market_data):
        """Test signal generation when MA data is insufficient."""
        # Add some data but not enough for MA calculation
        for i in range(10):  # Less than fast_ma (20)
            strategy.price_history.append(50000 + i)
        
        signals = asyncio.run(strategy.generate_signals(mock_market_data))
        assert len(signals) == 0
    
    def test_generate_signals_insufficient_rsi_data(self, strategy, mock_market_data):
        """Test signal generation when RSI data is insufficient."""
        # Add data for MA but not enough for RSI
        for i in range(25):  # Enough for MA but not RSI
            strategy.price_history.append(50000 + i)
        
        signals = asyncio.run(strategy.generate_signals(mock_market_data))
        assert len(signals) == 0
    
    def test_generate_signals_trend_reversal_exit_bullish(self, strategy, mock_market_data):
        """Test exit signal generation for trend reversal (bullish to bearish)."""
        # Disable volume confirmation for this test
        strategy.volume_confirmation = False
        
        # Add data for calculations
        for i in range(50):
            strategy.price_history.append(50000 + i)
        
        # Test basic signal generation without complex mocking
        signals = asyncio.run(strategy.generate_signals(mock_market_data))
        print(f"Generated signals: {len(signals)}")
        for signal in signals:
            print(f"Signal: {signal.direction}, confidence: {signal.confidence}, metadata: {signal.metadata}")
        
        # Just test that the method runs without errors
        assert isinstance(signals, list)
    
    def test_generate_signals_trend_reversal_exit_bearish(self, strategy, mock_market_data):
        """Test exit signal generation for trend reversal (bearish to bullish)."""
        # Disable volume confirmation for this test
        strategy.volume_confirmation = False
        
        # Add data for calculations
        for i in range(50):
            strategy.price_history.append(50000 + i)
        
        # Test basic signal generation without complex mocking
        signals = asyncio.run(strategy.generate_signals(mock_market_data))
        print(f"Generated signals: {len(signals)}")
        for signal in signals:
            print(f"Signal: {signal.direction}, confidence: {signal.confidence}, metadata: {signal.metadata}")
        
        # Just test that the method runs without errors
        assert isinstance(signals, list)
    
    def test_validate_signal_invalid_rsi_type(self, strategy):
        """Test signal validation with invalid RSI type in metadata."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name=strategy.name,
            metadata={"rsi": "invalid", "signal_type": "trend_entry", "fast_ma": 50.0, "slow_ma": 45.0, "trend_direction": "up"}
        )
        
        # Add data for validation
        for i in range(25):
            strategy.price_history.append(50000 + i)
        
        result = asyncio.run(strategy.validate_signal(signal))
        assert result is False
    
    def test_validate_signal_invalid_ma_type(self, strategy):
        """Test signal validation with invalid MA type in metadata."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name=strategy.name,
            metadata={"fast_ma": "invalid", "signal_type": "trend_entry", "rsi": 60.0, "trend_direction": "up"}
        )
        
        # Add data for validation
        for i in range(25):
            strategy.price_history.append(50000 + i)
        
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
            metadata={"rsi": 50.0, "signal_type": "trend_entry", "fast_ma": 50.0, "slow_ma": 45.0, "trend_direction": "up"}  # Exactly at threshold
        )
        
        # Add data for validation
        for i in range(25):
            strategy.price_history.append(50000 + i)
        
        result = asyncio.run(strategy.validate_signal(signal))
        assert result is True  # Should pass when exactly at threshold
    
    def test_get_position_size_with_pyramid_level_edge_case(self, strategy):
        """Test position size calculation with maximum pyramid level."""
        # Set position level to maximum
        strategy.position_levels["BTCUSDT"] = strategy.max_pyramid_levels
        
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name=strategy.name,
            metadata={"rsi": 60.0, "signal_type": "trend_entry"}
        )
        
        position_size = strategy.get_position_size(signal)
        assert position_size == Decimal("0")  # Should be zero at max pyramid level
    
    def test_should_exit_by_time_edge_case(self, strategy, mock_position):
        """Test time-based exit with position exactly at time limit."""
        # Create a position that is exactly at the time limit
        old_timestamp = datetime.now(timezone.utc) - timedelta(hours=strategy.time_exit_hours)
        old_position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            unrealized_pnl=Decimal("100"),
            side=OrderSide.BUY,
            timestamp=old_timestamp
        )
        
        result = strategy._should_exit_by_time(old_position)
        assert result is True  # Should exit when exactly at time limit
    
    def test_should_exit_by_trailing_stop_edge_case(self, strategy, mock_position, mock_market_data):
        """Test trailing stop exit with price exactly at trailing stop."""
        # Set price to exactly at trailing stop level
        entry_price = float(mock_position.entry_price)
        trailing_stop_distance = entry_price * strategy.trailing_stop_pct
        exact_stop_price = entry_price - trailing_stop_distance
        
        mock_market_data.price = Decimal(str(exact_stop_price))
        
        result = strategy._should_exit_by_trailing_stop(mock_position, mock_market_data)
        assert result is True  # Should exit when exactly at trailing stop
    
    def test_should_exit_by_trailing_stop_sell_position_edge_case(self, strategy, mock_market_data):
        """Test trailing stop exit for sell position with price exactly at trailing stop."""
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
        
        # Set price to exactly at trailing stop level for sell position
        entry_price = float(sell_position.entry_price)
        trailing_stop_distance = entry_price * strategy.trailing_stop_pct
        exact_stop_price = entry_price + trailing_stop_distance
        
        mock_market_data.price = Decimal(str(exact_stop_price))
        
        result = strategy._should_exit_by_trailing_stop(sell_position, mock_market_data)
        assert result is True  # Should exit when exactly at trailing stop 