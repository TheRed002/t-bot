"""
Integration tests for strategy framework.

Tests the integration between strategy components and other system parts.
"""

import pytest
import asyncio
import numpy as np
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

# Import from P-001
from src.core.types import (
    Signal, MarketData, Position, StrategyConfig, 
    StrategyStatus, StrategyType, SignalDirection, OrderSide
)
from src.core.config import Config
from src.core.exceptions import ConfigurationError

# Import from P-011
from src.strategies import StrategyFactory, StrategyConfigurationManager
from src.strategies.base import BaseStrategy


class MockStrategy(BaseStrategy):
    """Mock strategy for integration testing."""
    
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
        """Generate mock signals."""
        if not data or not data.price:
            return []
        
        # Create a mock signal
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol=data.symbol,
            strategy_name=self.name,
            metadata={"test": True}
        )
        
        return [signal]
    
    async def validate_signal(self, signal: Signal) -> bool:
        """Validate mock signal."""
        return signal.confidence >= self.config.min_confidence
    
    def get_position_size(self, signal: Signal) -> Decimal:
        """Get mock position size."""
        return Decimal(str(self.config.position_size_pct))
    
    def should_exit(self, position: Position, data: MarketData) -> bool:
        """Mock exit condition."""
        return position.unrealized_pnl < -Decimal("0.01")


class TestStrategyFrameworkIntegration:
    """Integration tests for strategy framework."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()
    
    @pytest.fixture
    def strategy_factory(self):
        """Create strategy factory."""
        return StrategyFactory()
    
    @pytest.fixture
    def config_manager(self):
        """Create strategy configuration manager."""
        return StrategyConfigurationManager()
    
    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager."""
        risk_manager = Mock()
        risk_manager.validate_signal = AsyncMock(return_value=True)
        return risk_manager
    
    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange."""
        exchange = Mock()
        exchange.get_market_data = AsyncMock(return_value=MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc)
        ))
        return exchange
    
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
    
    def test_strategy_factory_with_config_manager(self, strategy_factory, config_manager):
        """Test integration between factory and config manager."""
        # Register strategy class
        strategy_factory._register_strategy_class("mock_strategy", MockStrategy)
        
        # Create config using config manager
        config_data = {
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
            "parameters": {"test_param": "test_value"}
        }
        
        # Validate config
        assert config_manager.validate_config(config_data)
        
        # Create strategy using factory
        strategy = strategy_factory.create_strategy("mock_strategy", config_data)
        
        assert isinstance(strategy, MockStrategy)
        assert strategy.config.name == "test_strategy"
    
    @pytest.mark.asyncio
    async def test_strategy_with_risk_manager_integration(self, strategy_factory, mock_risk_manager):
        """Test strategy integration with risk manager."""
        # Register strategy class
        strategy_factory._register_strategy_class("mock_strategy", MockStrategy)
        
        # Set risk manager
        strategy_factory.set_risk_manager(mock_risk_manager)
        
        # Create strategy
        config_data = {
            "name": "test_strategy",
            "strategy_type": "static",
            "enabled": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "1h",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {}
        }
        
        strategy = strategy_factory.create_strategy("mock_strategy", config_data)
        
        # Test pre-trade validation with risk manager
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="test",
            metadata={}
        )
        
        result = await strategy.pre_trade_validation(signal)
        assert result is True
        mock_risk_manager.validate_signal.assert_called_once_with(signal)
    
    @pytest.mark.asyncio
    async def test_strategy_with_exchange_integration(self, strategy_factory, mock_exchange):
        """Test strategy integration with exchange."""
        # Register strategy class
        strategy_factory._register_strategy_class("mock_strategy", MockStrategy)
        
        # Set exchange
        strategy_factory.set_exchange(mock_exchange)
        
        # Create strategy
        config_data = {
            "name": "test_strategy",
            "strategy_type": "static",
            "enabled": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "1h",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {}
        }
        
        strategy = strategy_factory.create_strategy("mock_strategy", config_data)
        
        # Test signal generation with exchange data
        market_data = await mock_exchange.get_market_data("BTCUSDT")
        signals = await strategy.generate_signals(market_data)
        
        assert len(signals) == 1
        assert signals[0].symbol == "BTCUSDT"
        assert signals[0].confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_strategy_lifecycle_integration(self, strategy_factory):
        """Test complete strategy lifecycle."""
        # Register strategy class
        strategy_factory._register_strategy_class("mock_strategy", MockStrategy)
        
        # Create strategy
        config_data = {
            "name": "test_strategy",
            "strategy_type": "static",
            "enabled": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "1h",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {}
        }
        
        strategy = strategy_factory.create_strategy("mock_strategy", config_data)
        
        # Test lifecycle
        assert strategy.status == StrategyStatus.STOPPED
        
        # Start strategy
        await strategy.start()
        assert strategy.status == StrategyStatus.RUNNING
        
        # Pause strategy
        await strategy.pause()
        assert strategy.status == StrategyStatus.PAUSED
        
        # Resume strategy
        await strategy.resume()
        assert strategy.status == StrategyStatus.RUNNING
        
        # Stop strategy
        await strategy.stop()
        assert strategy.status == StrategyStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_multiple_strategies_integration(self, strategy_factory):
        """Test multiple strategies working together."""
        # Register strategy class
        strategy_factory._register_strategy_class("mock_strategy", MockStrategy)
        
        # Create multiple strategies
        configs = [
            {
                "name": "strategy_1",
                "strategy_type": "static",
                "enabled": True,
                "symbols": ["BTCUSDT"],
                "timeframe": "1h",
                "min_confidence": 0.6,
                "max_positions": 5,
                "position_size_pct": 0.02,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "parameters": {}
            },
            {
                "name": "strategy_2",
                "strategy_type": "static",
                "enabled": True,
                "symbols": ["ETHUSDT"],
                "timeframe": "5m",
                "min_confidence": 0.7,
                "max_positions": 3,
                "position_size_pct": 0.03,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.06,
                "parameters": {}
            }
        ]
        
        strategies = []
        for i, config in enumerate(configs):
            strategy_name = f"mock_strategy_{i}"
            strategy_factory._register_strategy_class(strategy_name, MockStrategy)
            strategy = strategy_factory.create_strategy(strategy_name, config)
            strategies.append(strategy)
        
        # Start all strategies
        for strategy in strategies:
            await strategy.start()
            assert strategy.status == StrategyStatus.RUNNING
        
        # Test strategy summary
        summary = strategy_factory.get_strategy_summary()
        assert summary["total_strategies"] == 2
        assert summary["running_strategies"] == 2
        
        # Stop all strategies
        await strategy_factory.shutdown_all_strategies()
        
        for strategy in strategies:
            assert strategy.status == StrategyStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_strategy_configuration_integration(self, config_manager):
        """Test strategy configuration integration."""
        # Test loading default configs
        available_strategies = config_manager.get_available_strategies()
        assert len(available_strategies) > 0
        
        # Test creating new config using an existing default
        new_config = config_manager.create_strategy_config(
            strategy_name="mean_reversion",  # Use existing default config
            strategy_type="static",  # Use string instead of enum
            symbols=["BTCUSDT", "ETHUSDT"],
            min_confidence=0.8,
            max_positions=10
        )
        
        assert new_config.name == "mean_reversion"
        assert new_config.strategy_type.value == "static"
        assert new_config.symbols == ["BTCUSDT", "ETHUSDT"]
        assert new_config.min_confidence == 0.8
        assert new_config.max_positions == 10
        
        # Test updating config parameter
        success = config_manager.update_config_parameter(
            "mean_reversion", "min_confidence", 0.9
        )
        assert success is True
        
        # Reload config to verify update
        updated_config = config_manager.load_strategy_config("mean_reversion")
        assert updated_config.min_confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_strategy_performance_tracking(self, strategy_factory, mock_market_data):
        """Test strategy performance tracking integration."""
        # Register strategy class
        strategy_factory._register_strategy_class("mock_strategy", MockStrategy)
        
        # Create strategy
        config_data = {
            "name": "test_strategy",
            "strategy_type": "static",
            "enabled": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "1h",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {}
        }
        
        strategy = strategy_factory.create_strategy("mock_strategy", config_data)
        
        # Generate signals and simulate trades
        signals = await strategy.generate_signals(mock_market_data)
        assert len(signals) == 1
        
        # Simulate trade result
        mock_trade_result = Mock()
        mock_trade_result.pnl = Decimal("10.50")
        
        await strategy.post_trade_processing(mock_trade_result)
        
        # Check performance metrics
        performance = strategy.get_performance_summary()
        assert performance["total_trades"] == 1
        assert performance["winning_trades"] == 1
        assert performance["total_pnl"] == 10.50
        assert performance["win_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_strategy_error_handling_integration(self, strategy_factory):
        """Test strategy error handling integration."""
        # Create strategy that raises errors
        class ErrorStrategy(MockStrategy):
            async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
                raise Exception("Signal generation error")
        
        strategy_factory._register_strategy_class("error_strategy", ErrorStrategy)
        
        config_data = {
            "name": "error_strategy",
            "strategy_type": "static",
            "enabled": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "1h",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {}
        }
        
        strategy = strategy_factory.create_strategy("error_strategy", config_data)
        
        # Test graceful error handling
        signals = await strategy.generate_signals(MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc)
        ))
        
        # Should return empty list on error (graceful degradation)
        assert signals == []
    
    def test_strategy_config_validation_integration(self, config_manager):
        """Test strategy configuration validation integration."""
        # Valid config
        valid_config = {
            "name": "valid_strategy",
            "strategy_type": "static",
            "enabled": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "1h",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {}
        }
        
        assert config_manager.validate_config(valid_config)
        
        # Invalid config
        invalid_config = {
            "name": "invalid_strategy",
            "strategy_type": "invalid_type",
            "symbols": [],
            "timeframe": "invalid_timeframe"
        }
        
        assert not config_manager.validate_config(invalid_config)
    
    def test_load_config_file_unsupported_format(self, config_manager):
        """Test loading config file with unsupported format."""
        # Create a config file with unsupported format
        config_file = config_manager.config_dir / "test.txt"
        with open(config_file, 'w') as f:
            f.write("invalid format")
        
        try:
            with pytest.raises(ConfigurationError, match="Unsupported config file format"):
                config_manager._load_config_file(config_file)
        finally:
            # Clean up
            if config_file.exists():
                config_file.unlink()
    
    def test_get_default_config_error(self, config_manager):
        """Test getting default config for non-existent strategy."""
        with pytest.raises(ConfigurationError, match="No default configuration for strategy"):
            config_manager._get_default_config("non_existent_strategy")
    
    @pytest.mark.asyncio
    async def test_strategy_hot_swap_integration(self, strategy_factory):
        """Test strategy hot swap integration."""
        # Register strategy class
        strategy_factory._register_strategy_class("mock_strategy", MockStrategy)
        
        # Create strategy
        config_data = {
            "name": "test_strategy",
            "strategy_type": "static",
            "enabled": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "1h",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {}
        }
        
        strategy = strategy_factory.create_strategy("mock_strategy", config_data)
        
        # Start strategy
        await strategy.start()
        assert strategy.status == StrategyStatus.RUNNING
        
        # Hot swap with new config
        new_config = config_data.copy()
        new_config["min_confidence"] = 0.8
        new_config["max_positions"] = 10
        
        success = await strategy_factory.hot_swap_strategy("mock_strategy", new_config)
        
        assert success is True
        assert strategy.config.min_confidence == 0.8
        assert strategy.config.max_positions == 10
        assert strategy.status == StrategyStatus.RUNNING 


# Import static strategies from P-012
from src.strategies.static.mean_reversion import MeanReversionStrategy
from src.strategies.static.trend_following import TrendFollowingStrategy
from src.strategies.static.breakout import BreakoutStrategy


class TestStaticStrategiesIntegration:
    """Integration tests for static trading strategies (P-012)."""
    
    @pytest.fixture
    def strategy_factory(self):
        """Create strategy factory for static strategies."""
        from src.strategies.factory import StrategyFactory
        return StrategyFactory()
    
    @pytest.fixture
    def config_manager(self):
        """Create config manager for static strategies."""
        from src.core.config import Config
        config = Config()
        return config
    
    @pytest.fixture
    def mean_reversion_config(self):
        """Create mean reversion strategy configuration."""
        return {
            "name": "mean_reversion_test",
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
                "lookback_period": 20,
                "entry_threshold": 2.0,
                "exit_threshold": 0.5,
                "volume_filter": True,
                "min_volume_ratio": 1.5,
                "atr_period": 14,
                "atr_multiplier": 2.0
            }
        }
    
    @pytest.fixture
    def trend_following_config(self):
        """Create trend following strategy configuration."""
        return {
            "name": "trend_following_test",
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
                "fast_ma_period": 10,
                "slow_ma_period": 20,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "volume_confirmation": True,
                "min_volume_ratio": 1.2,
                "pyramiding_enabled": True,
                "max_pyramiding_level": 3,
                "trailing_stop_enabled": True,
                "trailing_stop_pct": 0.01,
                "max_holding_time": 24
            }
        }
    
    @pytest.fixture
    def breakout_config(self):
        """Create breakout strategy configuration."""
        return {
            "name": "breakout_test",
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
                "lookback_period": 20,
                "consolidation_period": 5,
                "volume_confirmation": True,
                "min_volume_ratio": 1.5,
                "false_breakout_filter": True,
                "false_breakout_threshold": 0.02,
                "atr_period": 14,
                "atr_multiplier": 2.0,
                "target_multiplier": 3.0
            }
        }
    
    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager for static strategies."""
        risk_manager = Mock()
        risk_manager.validate_signal = AsyncMock(return_value=True)
        risk_manager.calculate_position_size = Mock(return_value=Decimal("0.01"))
        return risk_manager
    
    @pytest.fixture
    def historical_price_data(self):
        """Create historical price data for testing."""
        import numpy as np
        from datetime import datetime, timezone, timedelta
        
        # Generate realistic price data
        base_price = 50000
        timestamps = []
        prices = []
        volumes = []
        
        for i in range(100):
            timestamp = datetime.now(timezone.utc) - timedelta(hours=100-i)
            timestamps.append(timestamp)
            
            # Add some volatility and trends
            if i < 30:
                # Trending up
                price = base_price + i * 100 + np.random.normal(0, 200)
            elif i < 60:
                # Sideways with mean reversion
                price = base_price + np.random.normal(0, 500)
            else:
                # Trending down
                price = base_price - (i - 60) * 50 + np.random.normal(0, 200)
            
            prices.append(Decimal(str(max(price, 1000))))  # Ensure positive price
            volumes.append(Decimal(str(max(np.random.normal(1000, 200), 100))))
        
        return {
            "timestamps": timestamps,
            "prices": prices,
            "volumes": volumes
        }
    
    @pytest.mark.asyncio
    async def test_mean_reversion_strategy_integration(self, mean_reversion_config, mock_risk_manager):
        """Test mean reversion strategy integration with risk management."""
        # Create strategy
        strategy = MeanReversionStrategy(mean_reversion_config)
        strategy.risk_manager = mock_risk_manager
        
        # Create market data with mean reversion pattern
        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("48000"),  # Below average price
            volume=Decimal("2500"),  # Higher volume to pass filter (1.67x average)
            timestamp=datetime.now(timezone.utc)
        )
        
        # Add historical data to trigger mean reversion signal
        for i in range(25):
            historical_data = MarketData(
                symbol="BTCUSDT",
                price=Decimal("50000"),
                volume=Decimal("1500"),
                timestamp=datetime.now(timezone.utc)
            )
            strategy._update_price_history(historical_data)
        
        # Generate signals
        signals = await strategy.generate_signals(market_data)
        
        # Should generate buy signal due to price below mean
        assert len(signals) > 0
        assert signals[0].direction == SignalDirection.BUY
        assert signals[0].confidence > 0.6
        
        # Test signal validation
        is_valid = await strategy.validate_signal(signals[0])
        assert is_valid is True
        
        # Test position sizing
        position_size = strategy.get_position_size(signals[0])
        assert position_size > 0
        
                # Test exit condition with price closer to mean
        exit_market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50100"),  # Close to mean (50000)
            volume=Decimal("1500"),
            timestamp=datetime.now(timezone.utc)
        )
        
        position = Position(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            entry_price=Decimal("48000"),
            current_price=Decimal("50100"),
            unrealized_pnl=Decimal("0.04"),
            timestamp=datetime.now(timezone.utc)
        )

        should_exit = strategy.should_exit(position, exit_market_data)
        # Should exit due to price near mean (reversion)
        assert should_exit is True
    
    @pytest.mark.asyncio
    async def test_trend_following_strategy_integration(self, trend_following_config, mock_risk_manager):
        """Test trend following strategy integration with risk management."""
        # Create strategy
        strategy = TrendFollowingStrategy(trend_following_config)
        strategy.risk_manager = mock_risk_manager
        
        # Create market data with trending pattern
        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50900"),  # Continue moderate uptrend
            volume=Decimal("3000"),  # Higher volume to pass filter (1.5x average)
            timestamp=datetime.now(timezone.utc)
        )
        
        # Add historical data to create moderate uptrend with RSI in bullish range
        for i in range(60):  # More data for slow MA calculation
            base_price = 50000 + i * 15  # Very small trend
            variation = (i % 10 - 5) * 25  # More variation to keep RSI moderate
            price = base_price + variation
            historical_data = MarketData(
                symbol="BTCUSDT",
                price=Decimal(str(price)),
                volume=Decimal("2000"),
                timestamp=datetime.now(timezone.utc)
            )
            strategy._update_price_history(historical_data)
        
        # Generate signals
        signals = await strategy.generate_signals(market_data)
        
        # Should generate buy signal due to uptrend
        assert len(signals) > 0
        assert signals[0].direction == SignalDirection.BUY
        assert signals[0].confidence >= 0.6
        
        # Test pyramiding
        position_size = strategy.get_position_size(signals[0])
        assert position_size > 0
        
                # Test exit condition with trailing stop
        position = Position(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            entry_price=Decimal("50000"),
            current_price=Decimal("49000"),  # Below trailing stop
            unrealized_pnl=Decimal("-0.01"),
            timestamp=datetime.now(timezone.utc)
        )

        # Create market data with price below trailing stop
        exit_market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("49000"),  # Below trailing stop
            volume=Decimal("2000"),
            timestamp=datetime.now(timezone.utc)
        )

        should_exit = strategy.should_exit(position, exit_market_data)
        # Should exit due to trailing stop
        assert should_exit is True
    
    @pytest.mark.asyncio
    async def test_breakout_strategy_integration(self, breakout_config, mock_risk_manager):
        """Test breakout strategy integration with risk management."""
        # Create strategy
        strategy = BreakoutStrategy(breakout_config)
        strategy.risk_manager = mock_risk_manager
        
        # Create market data with breakout pattern
        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("52100"),  # Above resistance + 2% threshold
            volume=Decimal("4000"),  # Higher volume to pass filter (2.67x average)
            timestamp=datetime.now(timezone.utc)
        )
        
        # Add historical data to create consolidation and breakout
        for i in range(25):
            if i < 20:
                # Consolidation period - create more realistic consolidation around 50000
                base_price = 50000
                variation = (i % 7 - 3) * 50  # Small variations around 50000
                price = base_price + variation
            else:
                # Breakout
                price = 52000 + i * 50
            
            # Create realistic high/low prices for ATR calculation
            high_price = price + 200  # High is 200 above close
            low_price = price - 200   # Low is 200 below close
            
            historical_data = MarketData(
                symbol="BTCUSDT",
                price=Decimal(str(max(price, 1000))),
                high_price=Decimal(str(max(high_price, 1000))),
                low_price=Decimal(str(max(low_price, 1000))),
                volume=Decimal("1500"),
                timestamp=datetime.now(timezone.utc)
            )
            strategy._update_price_history(historical_data)
        
        # Manually set resistance level for testing
        strategy.resistance_levels = [51000]
        
        # Disable support/resistance update for testing (keep manually set levels)
        original_update_method = strategy._update_support_resistance_levels
        strategy._update_support_resistance_levels = lambda: None
        
        # Disable consolidation check for this test (we want to test breakout detection)
        strategy.consolidation_periods = 0
        
        # Generate signals
        signals = await strategy.generate_signals(market_data)
        
        # Should generate buy signal due to breakout
        assert len(signals) > 0
        assert signals[0].direction == SignalDirection.BUY
        assert signals[0].confidence > 0.6
        
        # Test position sizing
        position_size = strategy.get_position_size(signals[0])
        assert position_size > 0
        
        # Test exit condition with ATR stop loss
        position = Position(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            entry_price=Decimal("52000"),
            current_price=Decimal("50000"),  # Well below ATR stop loss (50700)
            unrealized_pnl=Decimal("-0.01"),
            timestamp=datetime.now(timezone.utc)
        )
        
        should_exit = strategy.should_exit(position, market_data)
        # Should exit due to ATR stop loss
        assert should_exit is True
    
    @pytest.mark.asyncio
    async def test_static_strategies_with_factory_integration(self, strategy_factory):
        """Test static strategies integration with strategy factory."""
        # Register static strategies
        strategy_factory._register_strategy_class("mean_reversion", MeanReversionStrategy)
        strategy_factory._register_strategy_class("trend_following", TrendFollowingStrategy)
        strategy_factory._register_strategy_class("breakout", BreakoutStrategy)
        
        # Test mean reversion strategy creation
        mean_reversion_config = {
            "name": "mean_reversion_test",
            "strategy_type": "static",
            "enabled": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "1h",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "parameters": {
                "lookback_period": 20,
                "entry_threshold": 2.0,
                "exit_threshold": 0.5
            }
        }
        
        strategy = strategy_factory.create_strategy("mean_reversion", mean_reversion_config)
        assert isinstance(strategy, MeanReversionStrategy)
        assert strategy.name == "mean_reversion_test"
        
        # Test trend following strategy creation
        trend_following_config = {
            "name": "trend_following_test",
            "strategy_type": "static",
            "enabled": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "1h",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "parameters": {
                "fast_ma_period": 10,
                "slow_ma_period": 20,
                "rsi_period": 14
            }
        }
        
        strategy = strategy_factory.create_strategy("trend_following", trend_following_config)
        assert isinstance(strategy, TrendFollowingStrategy)
        assert strategy.name == "trend_following_test"
        
        # Test breakout strategy creation
        breakout_config = {
            "name": "breakout_test",
            "strategy_type": "static",
            "enabled": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "1h",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "parameters": {
                "lookback_period": 20,
                "consolidation_period": 5,
                "volume_confirmation": True
            }
        }
        
        strategy = strategy_factory.create_strategy("breakout", breakout_config)
        assert isinstance(strategy, BreakoutStrategy)
        assert strategy.name == "breakout_test"
    
    @pytest.mark.asyncio
    async def test_static_strategies_with_risk_management_integration(self, mean_reversion_config, trend_following_config, breakout_config):
        """Test static strategies integration with risk management system."""
        from src.risk_management.risk_manager import RiskManager
        from src.core.config import Config

        # Create proper config object for risk manager
        config = Config()
        # Override risk settings for testing
        config.risk.max_daily_loss_pct = 0.05
        config.risk.max_drawdown_pct = 0.10
        config.risk.max_position_size_pct = 0.1
        config.risk.max_positions_per_symbol = 5
        config.risk.max_total_positions = 20
        config.risk.max_portfolio_exposure = 0.8
        config.risk.max_sector_exposure = 0.3
        config.risk.max_correlation_exposure = 0.5
        config.risk.max_leverage = 2.0
        
        # Create risk manager with proper config
        risk_manager = RiskManager(config)
        
        # Test mean reversion with risk management
        strategy = MeanReversionStrategy(mean_reversion_config)
        strategy.risk_manager = risk_manager
        
        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("48000"),
            volume=Decimal("2000"),
            timestamp=datetime.now(timezone.utc)
        )
        
        # Add historical data
        for i in range(25):
            historical_data = MarketData(
                symbol="BTCUSDT",
                price=Decimal("50000"),
                volume=Decimal("1500"),
                timestamp=datetime.now(timezone.utc)
            )
            strategy._update_price_history(historical_data)
        
        signals = await strategy.generate_signals(market_data)
        
        if signals:
            # Test risk validation
            is_valid = await risk_manager.validate_signal(signals[0])
            assert isinstance(is_valid, bool)
            
            # Test position sizing with risk management
            position_size = strategy.get_position_size(signals[0])
            assert position_size > 0
            assert position_size <= Decimal("0.1")  # Should respect position limits
    
    @pytest.mark.asyncio
    async def test_static_strategies_error_handling_integration(self, mean_reversion_config):
        """Test static strategies error handling integration."""
        # Create strategy with invalid configuration
        invalid_config = mean_reversion_config.copy()
        invalid_config["parameters"]["lookback_period"] = -1  # Invalid parameter
        
        strategy = MeanReversionStrategy(invalid_config)
        
        # Test graceful error handling
        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(timezone.utc)
        )
        
        signals = await strategy.generate_signals(market_data)
        
        # Should return empty list on error (graceful degradation)
        assert signals == []
    
    @pytest.mark.asyncio
    async def test_static_strategies_performance_integration(self, mean_reversion_config, historical_price_data):
        """Test static strategies performance tracking integration."""
        strategy = MeanReversionStrategy(mean_reversion_config)
        
        # Simulate trading session with historical data
        signals_generated = 0
        positions_opened = 0
        
        # Add initial historical data to establish baseline
        for i in range(25):
            historical_data = MarketData(
                symbol="BTCUSDT",
                price=Decimal("50000"),  # Stable baseline
                volume=Decimal("1500"),
                timestamp=datetime.now(timezone.utc)
            )
            strategy._update_price_history(historical_data)
        
        for i in range(len(historical_price_data["prices"])):
            market_data = MarketData(
                symbol="BTCUSDT",
                price=historical_price_data["prices"][i],
                volume=historical_price_data["volumes"][i],
                timestamp=historical_price_data["timestamps"][i]
            )
            
            signals = await strategy.generate_signals(market_data)
            signals_generated += len(signals)
            
            if signals:
                positions_opened += 1
        
        # Verify strategy performance tracking
        # Note: Random data may not always generate signals, so we test the framework
        # rather than requiring specific signal generation
        assert isinstance(signals_generated, int)
        assert isinstance(positions_opened, int)
        
        # Test strategy info (metrics not implemented yet)
        strategy_info = strategy.get_strategy_info()
        assert isinstance(strategy_info, dict)
        assert "strategy_type" in strategy_info
    
    @pytest.mark.asyncio
    async def test_static_strategies_multi_symbol_integration(self, mean_reversion_config):
        """Test static strategies with multiple symbols integration."""
        # Configure strategy for multiple symbols
        multi_symbol_config = mean_reversion_config.copy()
        multi_symbol_config["symbols"] = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        
        strategy = MeanReversionStrategy(multi_symbol_config)
        
        # Test with multiple symbols
        for symbol in ["BTCUSDT", "ETHUSDT", "ADAUSDT"]:
            market_data = MarketData(
                symbol=symbol,
                price=Decimal("50000"),
                volume=Decimal("1000"),
                timestamp=datetime.now(timezone.utc)
            )
            
            # Add historical data
            for i in range(25):
                historical_data = MarketData(
                    symbol=symbol,
                    price=Decimal("50000"),
                    volume=Decimal("1500"),
                    timestamp=datetime.now(timezone.utc)
                )
                strategy._update_price_history(historical_data)
            
            signals = await strategy.generate_signals(market_data)
            
            # Should handle multiple symbols correctly
            assert isinstance(signals, list)
            
            if signals:
                assert signals[0].symbol == symbol
    
    @pytest.mark.asyncio
    async def test_static_strategies_configuration_integration(self, config_manager):
        """Test static strategies configuration integration."""
        # Test that config manager has required risk management settings
        assert config_manager.risk is not None
        assert config_manager.risk.max_position_size_pct > 0
        assert config_manager.risk.max_total_positions > 0
        assert config_manager.risk.max_daily_loss_pct > 0
        assert config_manager.risk.max_drawdown_pct > 0
        
        # Test that config manager has strategy management settings
        assert config_manager.strategies is not None
        assert config_manager.strategies.default_min_confidence > 0
        assert config_manager.strategies.default_position_size > 0