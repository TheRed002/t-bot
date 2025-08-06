"""
Integration tests for strategy framework.

Tests the integration between strategy components and other system parts.
"""

import pytest
import asyncio
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
        for config in configs:
            strategy = strategy_factory.create_strategy("mock_strategy", config)
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
        
        success = await strategy_factory.hot_swap_strategy("test_strategy", new_config)
        
        assert success is True
        assert strategy.config.min_confidence == 0.8
        assert strategy.config.max_positions == 10
        assert strategy.status == StrategyStatus.RUNNING 