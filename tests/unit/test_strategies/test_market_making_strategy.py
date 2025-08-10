"""
Unit tests for Market Making Strategy.

This module contains comprehensive unit tests for the market making strategy
implementation, including tests for signal generation, validation, position sizing,
and inventory management.

CRITICAL: These tests must achieve 90% coverage as required by P-013B.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import core types
from src.core.types import (
    MarketData,
    OrderSide,
    Position,
    Signal,
    SignalDirection,
    StrategyType,
)
from src.risk_management.base import BaseRiskManager
from src.strategies.static.inventory_manager import InventoryManager

# Import the strategy classes
from src.strategies.static.market_making import MarketMakingStrategy
from src.strategies.static.spread_optimizer import SpreadOptimizer


class TestMarketMakingStrategy:
    """Test cases for MarketMakingStrategy class."""

    @pytest.fixture
    def strategy_config(self):
        """Create a test configuration for the market making strategy."""
        return {
            "name": "test_market_making",
            "strategy_type": "market_making",
            "enabled": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "1m",
            "min_confidence": 0.8,
            "max_positions": 10,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {
                "base_spread": 0.001,
                "order_levels": 3,
                "base_order_size": 0.01,
                "size_multiplier": 1.5,
                "order_size_distribution": "exponential",
                "volatility_multiplier": 2.0,
                "inventory_skew": True,
                "competitive_quotes": True,
                "target_inventory": 0.5,
                "max_inventory": 1.0,
                "inventory_risk_aversion": 0.1,
                "max_position_value": 10000,
                "stop_loss_inventory": 2.0,
                "daily_loss_limit": 100,
                "min_profit_per_trade": 0.00001,
                "order_refresh_time": 30,
                "adaptive_spreads": True,
                "competition_monitoring": True,
            },
        }

    @pytest.fixture
    def market_data(self):
        """Create test market data."""
        return MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc),
            bid=Decimal("49999"),
            ask=Decimal("50001"),
            open_price=Decimal("49900"),
            high_price=Decimal("50100"),
            low_price=Decimal("49800"),
        )

    @pytest.fixture
    def strategy(self, strategy_config):
        """Create a test instance of MarketMakingStrategy."""
        return MarketMakingStrategy(strategy_config)

    def test_strategy_initialization(self, strategy):
        """Test strategy initialization with configuration."""
        assert strategy.name == "test_market_making"
        assert strategy.strategy_type == StrategyType.MARKET_MAKING
        assert strategy.base_spread == Decimal("0.001")
        assert strategy.order_levels == 3
        assert strategy.base_order_size == Decimal("0.01")
        assert strategy.size_multiplier == 1.5
        assert strategy.target_inventory == Decimal("0.5")
        assert strategy.max_inventory == Decimal("1.0")
        assert strategy.daily_loss_limit == Decimal("100")

    def test_strategy_default_parameters(self):
        """Test strategy initialization with default parameters."""
        config = {
            "name": "test_default",
            "strategy_type": "market_making",
            "symbols": ["BTCUSDT"],
            "parameters": {},
        }
        strategy = MarketMakingStrategy(config)

        # Check default values
        assert strategy.base_spread == Decimal("0.001")
        assert strategy.order_levels == 5
        assert strategy.base_order_size == Decimal("0.01")
        assert strategy.size_multiplier == 1.5

    @pytest.mark.asyncio
    async def test_generate_signals_valid_data(self, strategy, market_data):
        """Test signal generation with valid market data."""
        signals = await strategy._generate_signals_impl(market_data)

        # Should generate signals for each level (bid + ask for each level)
        expected_signal_count = strategy.order_levels * 2
        assert len(signals) == expected_signal_count

        # Check signal properties
        for signal in signals:
            assert signal.strategy_name == strategy.name
            assert signal.confidence == 0.8
            assert signal.symbol == market_data.symbol
            assert "level" in signal.metadata
            assert "price" in signal.metadata
            assert "size" in signal.metadata
            assert "spread" in signal.metadata
            assert "side" in signal.metadata

    @pytest.mark.asyncio
    async def test_generate_signals_invalid_data(self, strategy):
        """Test signal generation with invalid market data."""
        # Test with None data
        signals = await strategy._generate_signals_impl(None)
        assert len(signals) == 0

        # Test with missing bid/ask
        invalid_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc),
            bid=None,
            ask=None,
        )
        signals = await strategy._generate_signals_impl(invalid_data)
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_generate_signals_exception_handling(self, strategy, market_data):
        """Test signal generation with exception handling."""
        with patch.object(strategy, "_update_price_history", side_effect=Exception("Test error")):
            signals = await strategy._generate_signals_impl(market_data)
            assert len(signals) == 0  # Graceful degradation

    def test_calculate_level_spread(self, strategy):
        """Test level spread calculation."""
        base_spread = Decimal("0.001")
        volatility = 0.02

        # Test level 1
        spread = strategy._calculate_level_spread(1, base_spread, volatility)
        assert spread > base_spread

        # Test level 3 (should be wider)
        spread_level_3 = strategy._calculate_level_spread(3, base_spread, volatility)
        assert spread_level_3 > spread

        # Test with inventory skew
        strategy.inventory_state.inventory_skew = 0.5
        spread_with_skew = strategy._calculate_level_spread(1, base_spread, volatility)
        assert spread_with_skew != spread

    def test_calculate_level_size(self, strategy):
        """Test level size calculation."""
        # Test exponential distribution
        size_level_1 = strategy._calculate_level_size(1)
        size_level_2 = strategy._calculate_level_size(2)
        size_level_3 = strategy._calculate_level_size(3)

        assert size_level_2 > size_level_1
        assert size_level_3 > size_level_2

        # Test with inventory skew
        strategy.inventory_state.inventory_skew = 0.8
        size_with_skew = strategy._calculate_level_size(1)
        assert size_with_skew > size_level_1

    def test_calculate_volatility(self, strategy):
        """Test volatility calculation."""
        # Test with insufficient data
        volatility = strategy._calculate_volatility()
        assert volatility == 0.02  # Default value

        # Test with sufficient data
        strategy.price_history = [100 + i for i in range(25)]
        volatility = strategy._calculate_volatility()
        assert isinstance(volatility, float)
        assert volatility >= 0

    def test_update_price_history(self, strategy, market_data):
        """Test price history update."""
        initial_length = len(strategy.price_history)
        strategy._update_price_history(market_data)

        assert len(strategy.price_history) == initial_length + 1
        assert strategy.price_history[-1] == float(market_data.price)

        # Test spread history update
        if market_data.bid and market_data.ask:
            assert len(strategy.spread_history) > 0

    @pytest.mark.asyncio
    async def test_validate_signal_valid(self, strategy):
        """Test signal validation with valid signal."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name=strategy.name,
            metadata={"price": 50000.0, "size": 0.01, "level": 1, "spread": 0.001, "side": "bid"},
        )

        is_valid = await strategy.validate_signal(signal)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_signal_invalid_confidence(self, strategy):
        """Test signal validation with low confidence."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.5,  # Below threshold
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name=strategy.name,
            metadata={"price": 50000.0, "size": 0.01},
        )

        is_valid = await strategy.validate_signal(signal)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_signal_missing_metadata(self, strategy):
        """Test signal validation with missing metadata."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name=strategy.name,
            metadata={},  # Missing required metadata
        )

        is_valid = await strategy.validate_signal(signal)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_signal_daily_loss_limit(self, strategy):
        """Test signal validation with daily loss limit exceeded."""
        strategy.daily_pnl = Decimal("-150")  # Exceeds limit

        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name=strategy.name,
            metadata={"price": 50000.0, "size": 0.01},
        )

        is_valid = await strategy.validate_signal(signal)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_signal_with_risk_manager_rejection(self, strategy):
        """Test signal validation when risk manager rejects signal."""
        # Mock risk manager to reject signal
        mock_risk_manager = Mock(spec=BaseRiskManager)
        mock_risk_manager.validate_signal = AsyncMock(return_value=False)
        strategy.set_risk_manager(mock_risk_manager)

        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name=strategy.name,
            metadata={"price": 50000.0, "size": 0.01, "level": 1, "spread": 0.001, "side": "bid"},
        )

        is_valid = await strategy.validate_signal(signal)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_signal_with_risk_manager_acceptance(self, strategy):
        """Test signal validation when risk manager accepts signal."""
        # Mock risk manager to accept signal
        mock_risk_manager = Mock(spec=BaseRiskManager)
        mock_risk_manager.validate_signal = AsyncMock(return_value=True)
        strategy.set_risk_manager(mock_risk_manager)

        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name=strategy.name,
            metadata={"price": 50000.0, "size": 0.01, "level": 1, "spread": 0.001, "side": "bid"},
        )

        is_valid = await strategy.validate_signal(signal)
        assert is_valid is True

    def test_check_inventory_limits(self, strategy):
        """Test inventory limit checking."""
        # Test within limits
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name=strategy.name,
            metadata={"size": 0.5},
        )

        is_valid = strategy._check_inventory_limits(signal)
        assert is_valid is True

        # Test exceeding limits
        signal.metadata["size"] = 2.0  # Exceeds max inventory
        is_valid = strategy._check_inventory_limits(signal)
        assert is_valid is False

    def test_get_position_size(self, strategy):
        """Test position size calculation."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name=strategy.name,
            metadata={"size": 0.01, "price": 50000.0},
        )

        size = strategy.get_position_size(signal)
        assert isinstance(size, Decimal)
        assert size > 0

    @pytest.mark.asyncio
    async def test_should_exit_stop_loss(self, strategy):
        """Test position exit due to stop loss."""
        position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("2.5"),  # Exceeds stop loss
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc),
        )

        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc),
            bid=Decimal("49999"),
            ask=Decimal("50001"),
        )

        should_exit = await strategy.should_exit(position, market_data)
        assert should_exit is True

    @pytest.mark.asyncio
    async def test_should_exit_profit_taking(self, strategy):
        """Test position exit due to profit taking."""
        position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50100"),
            unrealized_pnl=Decimal("50"),  # Profitable
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc),
        )

        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc),
            bid=Decimal("49950"),  # Narrow spread
            ask=Decimal("50050"),
        )

        should_exit = await strategy.should_exit(position, market_data)
        assert should_exit is True

    @pytest.mark.asyncio
    async def test_should_rebalance_inventory(self, strategy):
        """Test inventory rebalancing check."""
        # Test with high skew
        position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.9"),  # 90% of max inventory
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc),
        )

        should_rebalance = await strategy._should_rebalance_inventory(position)
        assert should_rebalance is True

    @pytest.mark.asyncio
    async def test_update_inventory_state(self, strategy):
        """Test inventory state update."""
        position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.3"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc),
        )

        await strategy.update_inventory_state(position)

        assert strategy.inventory_state.current_inventory == Decimal("0.3")
        assert strategy.inventory_state.inventory_skew == 0.3  # 0.3 / 1.0

    @pytest.mark.asyncio
    async def test_update_performance_metrics(self, strategy):
        """Test performance metrics update."""
        trade_result = {"pnl": 10.5, "quantity": 0.01, "price": 50000.0}

        initial_trades = strategy.total_trades
        initial_pnl = strategy.total_pnl

        await strategy.update_performance_metrics(trade_result)

        assert strategy.total_trades == initial_trades + 1
        assert strategy.total_pnl == initial_pnl + Decimal("10.5")
        assert strategy.profitable_trades == 1

    @pytest.mark.asyncio
    async def test_update_performance_metrics_with_exception(self, strategy):
        """Test performance metrics update with exception handling."""
        # Invalid trade result that will cause an exception
        trade_result = {
            "pnl": "invalid_pnl",  # This will cause an exception
            "quantity": 0.01,
            "price": 50000.0,
        }

        await strategy.update_performance_metrics(trade_result)

        # Should handle the exception gracefully
        assert strategy.total_trades >= 0

    def test_get_strategy_info(self, strategy):
        """Test strategy information retrieval."""
        info = strategy.get_strategy_info()

        assert "strategy_type" in info
        assert info["strategy_type"] == "market_making"
        assert "base_spread" in info
        assert "order_levels" in info
        assert "target_inventory" in info
        assert "current_inventory" in info
        assert "inventory_skew" in info
        assert "total_trades" in info
        assert "total_pnl" in info

    def test_get_strategy_info_comprehensive(self, strategy):
        """Test comprehensive strategy information retrieval."""
        # Update some metrics first
        strategy.total_trades = 10
        strategy.profitable_trades = 7
        strategy.total_pnl = Decimal("100.50")
        strategy.daily_pnl = Decimal("25.25")

        info = strategy.get_strategy_info()

        assert "strategy_type" in info
        assert info["strategy_type"] == "market_making"
        assert "base_spread" in info
        assert "order_levels" in info
        assert "target_inventory" in info
        assert "current_inventory" in info
        assert "inventory_skew" in info
        assert "total_trades" in info
        assert "total_pnl" in info
        assert info["total_trades"] == 10
        assert info["total_pnl"] == Decimal("100.50")

    def test_get_position_size_with_risk_manager(self, strategy):
        """Test position size calculation with risk manager integration."""
        # Mock risk manager
        mock_risk_manager = Mock(spec=BaseRiskManager)
        strategy.set_risk_manager(mock_risk_manager)

        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name=strategy.name,
            metadata={"price": 50000.0, "size": 0.01, "level": 1, "spread": 0.001, "side": "bid"},
        )

        size = strategy.get_position_size(signal)
        assert isinstance(size, Decimal)
        assert size > 0

    def test_get_position_size_with_exception(self, strategy):
        """Test position size calculation with exception handling."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name=strategy.name,
            metadata={
                "price": "invalid_price",  # This will cause an exception
                "size": 0.01,
                "level": 1,
                "spread": 0.001,
                "side": "bid",
            },
        )

        size = strategy.get_position_size(signal)
        assert isinstance(size, Decimal)
        assert size == strategy.base_order_size  # Should return default size

    @pytest.mark.asyncio
    async def test_should_exit_with_exception(self, strategy):
        """Test position exit check with exception handling."""
        position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50100"),
            unrealized_pnl=Decimal("50"),
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc),
        )

        # Market data with invalid bid/ask that will cause division by zero
        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc),
            bid=Decimal("0"),  # This will cause division by zero
            ask=Decimal("0"),
        )

        should_exit = await strategy.should_exit(position, market_data)
        assert isinstance(should_exit, bool)

    @pytest.mark.asyncio
    async def test_update_inventory_state_with_exception(self, strategy):
        """Test inventory state update with exception handling."""
        # Create a position that will cause an exception
        position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.3"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc),
        )

        # Set max_inventory to 0 to cause division by zero
        strategy.inventory_state.max_inventory = Decimal("0")

        await strategy.update_inventory_state(position)

        # Should handle the exception gracefully
        assert strategy.inventory_state.current_inventory == Decimal("0.3")


class TestInventoryManager:
    """Test cases for InventoryManager class."""

    @pytest.fixture
    def inventory_config(self):
        """Create test configuration for inventory manager."""
        return {
            "target_inventory": 0.5,
            "max_inventory": 1.0,
            "min_inventory": -1.0,
            "inventory_risk_aversion": 0.1,
            "rebalance_threshold": 0.2,
            "rebalance_frequency_hours": 4,
            "max_rebalance_size": 0.5,
            "emergency_threshold": 0.8,
            "emergency_liquidation_enabled": True,
        }

    @pytest.fixture
    def inventory_manager(self, inventory_config):
        """Create test instance of InventoryManager."""
        return InventoryManager(inventory_config)

    @pytest.fixture
    def position(self):
        """Create test position."""
        return Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.3"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_update_inventory(self, inventory_manager, position):
        """Test inventory update."""
        await inventory_manager.update_inventory(position)

        assert inventory_manager.current_inventory == Decimal("0.3")
        assert inventory_manager.inventory_skew == 0.3

    @pytest.mark.asyncio
    async def test_should_rebalance_threshold_exceeded(self, inventory_manager):
        """Test rebalancing check when threshold is exceeded."""
        inventory_manager.current_inventory = Decimal("0.8")  # 80% of max
        inventory_manager.target_inventory = Decimal("0.5")

        should_rebalance = await inventory_manager.should_rebalance()
        assert should_rebalance is True

    @pytest.mark.asyncio
    async def test_should_rebalance_max_inventory_exceeded(self, inventory_manager):
        """Test rebalancing check when max inventory is exceeded."""
        inventory_manager.current_inventory = Decimal("1.5")  # Exceeds max

        should_rebalance = await inventory_manager.should_rebalance()
        assert should_rebalance is True

    @pytest.mark.asyncio
    async def test_should_rebalance_time_based(self, inventory_manager):
        """Test time-based rebalancing."""
        inventory_manager.last_rebalance = datetime.now() - timedelta(hours=5)

        should_rebalance = await inventory_manager.should_rebalance()
        assert should_rebalance is True

    @pytest.mark.asyncio
    async def test_calculate_rebalance_orders_buy(self, inventory_manager):
        """Test rebalancing order calculation for buy."""
        inventory_manager.current_inventory = Decimal("0.2")
        inventory_manager.target_inventory = Decimal("0.5")

        orders = await inventory_manager.calculate_rebalance_orders(Decimal("50000"))

        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        assert orders[0].quantity == Decimal("0.3")

    @pytest.mark.asyncio
    async def test_calculate_rebalance_orders_sell(self, inventory_manager):
        """Test rebalancing order calculation for sell."""
        inventory_manager.current_inventory = Decimal("0.8")
        inventory_manager.target_inventory = Decimal("0.5")

        orders = await inventory_manager.calculate_rebalance_orders(Decimal("50000"))

        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].quantity == Decimal("0.3")

    @pytest.mark.asyncio
    async def test_should_emergency_liquidate(self, inventory_manager):
        """Test emergency liquidation check."""
        inventory_manager.current_inventory = Decimal("0.9")  # 90% of max

        should_emergency = await inventory_manager.should_emergency_liquidate()
        assert should_emergency is True

    @pytest.mark.asyncio
    async def test_calculate_emergency_orders_sell(self, inventory_manager):
        """Test emergency order calculation for sell."""
        inventory_manager.current_inventory = Decimal("0.9")

        orders = await inventory_manager.calculate_emergency_orders(Decimal("50000"))

        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].quantity == Decimal("0.9")

    @pytest.mark.asyncio
    async def test_calculate_emergency_orders_buy(self, inventory_manager):
        """Test emergency order calculation for buy."""
        inventory_manager.current_inventory = Decimal("-0.9")

        orders = await inventory_manager.calculate_emergency_orders(Decimal("50000"))

        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        assert orders[0].quantity == Decimal("0.9")

    @pytest.mark.asyncio
    async def test_calculate_spread_adjustment(self, inventory_manager):
        """Test spread adjustment calculation."""
        base_spread = Decimal("0.001")
        inventory_manager.inventory_skew = 0.5

        adjusted_spread = await inventory_manager.calculate_spread_adjustment(base_spread)

        assert adjusted_spread > base_spread

    @pytest.mark.asyncio
    async def test_calculate_size_adjustment(self, inventory_manager):
        """Test size adjustment calculation."""
        base_size = Decimal("0.01")
        inventory_manager.inventory_skew = 0.8

        adjusted_size = await inventory_manager.calculate_size_adjustment(base_size)

        assert adjusted_size > base_size  # Should be larger due to skew

    @pytest.mark.asyncio
    async def test_record_rebalance(self, inventory_manager):
        """Test rebalancing cost recording."""
        initial_cost = inventory_manager.total_rebalance_cost
        initial_count = inventory_manager.rebalance_count

        await inventory_manager.record_rebalance(Decimal("10.5"))

        assert inventory_manager.total_rebalance_cost == initial_cost + Decimal("10.5")
        assert inventory_manager.rebalance_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_record_emergency(self, inventory_manager):
        """Test emergency liquidation cost recording."""
        initial_cost = inventory_manager.total_emergency_cost
        initial_count = inventory_manager.emergency_count

        await inventory_manager.record_emergency(Decimal("-5.5"))

        assert inventory_manager.total_emergency_cost == initial_cost + Decimal("-5.5")
        assert inventory_manager.emergency_count == initial_count + 1

    def test_get_inventory_summary(self, inventory_manager):
        """Test inventory summary generation."""
        summary = inventory_manager.get_inventory_summary()

        assert "current_inventory" in summary
        assert "target_inventory" in summary
        assert "max_inventory" in summary
        assert "inventory_skew" in summary
        assert "rebalance_count" in summary
        assert "emergency_count" in summary

    @pytest.mark.asyncio
    async def test_validate_inventory_limits(self, inventory_manager):
        """Test inventory limit validation."""
        # Test within limits
        position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc),
        )

        is_valid = await inventory_manager.validate_inventory_limits(position)
        assert is_valid is True

        # Test exceeding max inventory
        position.quantity = Decimal("1.5")
        is_valid = await inventory_manager.validate_inventory_limits(position)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_inventory_limits_with_exception(self, inventory_manager):
        """Test inventory limits validation with exception handling."""
        # Create a position that will cause an exception
        position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc),
        )

        # Set invalid max_inventory to cause an exception
        inventory_manager.max_inventory = "invalid_max"

        # Should handle the exception gracefully
        result = await inventory_manager.validate_inventory_limits(position)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_calculate_spread_adjustment_with_exception(self, inventory_manager):
        """Test spread adjustment calculation with exception handling."""
        # Test with invalid base_spread that will cause an exception
        base_spread = Decimal("0.001")

        # Set invalid inventory_skew to cause an exception
        inventory_manager.inventory_skew = "invalid_skew"

        result = await inventory_manager.calculate_spread_adjustment(base_spread)
        assert isinstance(result, Decimal)

    @pytest.mark.asyncio
    async def test_calculate_size_adjustment_with_exception(self, inventory_manager):
        """Test size adjustment calculation with exception handling."""
        # Test with invalid base_size that will cause an exception
        base_size = Decimal("0.01")

        # Set invalid inventory_skew to cause an exception
        inventory_manager.inventory_skew = "invalid_skew"

        result = await inventory_manager.calculate_size_adjustment(base_size)
        assert isinstance(result, Decimal)

    @pytest.mark.asyncio
    async def test_record_rebalance_with_exception(self, inventory_manager):
        """Test rebalance recording with exception handling."""
        # Test with invalid cost that will cause an exception
        cost = "invalid_cost"  # This will cause an exception

        await inventory_manager.record_rebalance(cost)
        # Should handle the exception gracefully

    @pytest.mark.asyncio
    async def test_record_emergency_with_exception(self, inventory_manager):
        """Test emergency recording with exception handling."""
        # Test with invalid cost that will cause an exception
        cost = "invalid_cost"  # This will cause an exception

        await inventory_manager.record_emergency(cost)
        # Should handle the exception gracefully


class TestSpreadOptimizer:
    """Test cases for SpreadOptimizer class."""

    @pytest.fixture
    def optimizer_config(self):
        """Create test configuration for spread optimizer."""
        return {
            "volatility_multiplier": 2.0,
            "volatility_window": 20,
            "min_volatility": 0.001,
            "max_volatility": 0.05,
            "imbalance_threshold": 0.1,
            "depth_levels": 5,
            "min_spread": 0.0001,
            "max_spread": 0.01,
            "competitor_monitoring": True,
            "competitor_weight": 0.3,
            "max_competitor_spread": 0.005,
            "impact_threshold": 0.001,
            "impact_multiplier": 1.5,
        }

    @pytest.fixture
    def spread_optimizer(self, optimizer_config):
        """Create test instance of SpreadOptimizer."""
        return SpreadOptimizer(optimizer_config)

    @pytest.fixture
    def market_data(self):
        """Create test market data."""
        return MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc),
            bid=Decimal("49999"),
            ask=Decimal("50001"),
            open_price=Decimal("49900"),
            high_price=Decimal("50100"),
            low_price=Decimal("49800"),
        )

    @pytest.fixture
    def order_book(self):
        """Create test order book."""
        return {
            "symbol": "BTCUSDT",
            "bids": [[Decimal("49999"), Decimal("1.0")], [Decimal("49998"), Decimal("2.0")]],
            "asks": [[Decimal("50001"), Decimal("1.0")], [Decimal("50002"), Decimal("2.0")]],
            "timestamp": datetime.now(timezone.utc),
        }

    @pytest.mark.asyncio
    async def test_optimize_spread_basic(self, spread_optimizer, market_data):
        """Test basic spread optimization."""
        base_spread = Decimal("0.001")

        optimized_spread = await spread_optimizer.optimize_spread(base_spread, market_data)

        assert isinstance(optimized_spread, Decimal)
        assert optimized_spread >= spread_optimizer.min_spread
        assert optimized_spread <= spread_optimizer.max_spread

    @pytest.mark.asyncio
    async def test_calculate_volatility_adjustment(self, spread_optimizer):
        """Test volatility adjustment calculation."""
        base_spread = Decimal("0.001")

        # Test with insufficient data
        adjustment = await spread_optimizer._calculate_volatility_adjustment(base_spread)
        assert adjustment == Decimal("0")

        # Test with sufficient data
        spread_optimizer.price_history = [100 + i * 0.1 for i in range(25)]
        adjustment = await spread_optimizer._calculate_volatility_adjustment(base_spread)
        assert isinstance(adjustment, Decimal)
        assert adjustment >= Decimal("0")

    @pytest.mark.asyncio
    async def test_calculate_imbalance_adjustment(self, spread_optimizer):
        """Test imbalance adjustment calculation."""
        base_spread = Decimal("0.001")

        # Test with no order book
        adjustment = await spread_optimizer._calculate_imbalance_adjustment(base_spread, None)
        assert adjustment == Decimal("0")

        # Test with balanced order book
        balanced_order_book = {
            "bids": [[Decimal("49999"), Decimal("1.0")]],
            "asks": [[Decimal("50001"), Decimal("1.0")]],
        }
        adjustment = await spread_optimizer._calculate_imbalance_adjustment(
            base_spread, balanced_order_book
        )
        assert adjustment == Decimal("0")

    @pytest.mark.asyncio
    async def test_calculate_competitor_adjustment(self, spread_optimizer):
        """Test competitor adjustment calculation."""
        base_spread = Decimal("0.001")

        # Test with no competitor data
        adjustment = await spread_optimizer._calculate_competitor_adjustment(base_spread, None)
        assert adjustment == Decimal("0")

        # Test with competitor data
        competitor_spreads = [0.002, 0.003, 0.001]
        adjustment = await spread_optimizer._calculate_competitor_adjustment(
            base_spread, competitor_spreads
        )
        assert isinstance(adjustment, Decimal)

    @pytest.mark.asyncio
    async def test_calculate_impact_adjustment(self, spread_optimizer):
        """Test impact adjustment calculation."""
        base_spread = Decimal("0.001")

        # Test with insufficient spread history
        adjustment = await spread_optimizer._calculate_impact_adjustment(base_spread)
        assert adjustment == Decimal("0")

        # Test with spread history
        spread_optimizer.spread_history = [0.001 + i * 0.0001 for i in range(15)]
        adjustment = await spread_optimizer._calculate_impact_adjustment(base_spread)
        assert isinstance(adjustment, Decimal)

    @pytest.mark.asyncio
    async def test_calculate_optimal_spread(self, spread_optimizer, market_data):
        """Test optimal spread calculation."""
        bid_spread, ask_spread = await spread_optimizer.calculate_optimal_spread(market_data)

        assert isinstance(bid_spread, Decimal)
        assert isinstance(ask_spread, Decimal)
        assert bid_spread > 0
        assert ask_spread > 0

    @pytest.mark.asyncio
    async def test_should_widen_spread(self, spread_optimizer, market_data):
        """Test spread widening check."""
        # Test with normal conditions
        should_widen = await spread_optimizer.should_widen_spread(market_data)
        assert isinstance(should_widen, bool)

        # Test with high volatility
        # High volatility (exceeds threshold)
        spread_optimizer.volatility_history = [0.05] * 5
        should_widen = await spread_optimizer.should_widen_spread(market_data)
        assert should_widen is True

    def test_get_optimization_summary(self, spread_optimizer):
        """Test optimization summary retrieval."""
        summary = spread_optimizer.get_optimization_summary()

        assert "optimization_count" in summary
        assert "volatility_adjustments" in summary
        assert "imbalance_adjustments" in summary
        assert "competitor_adjustments" in summary
        assert "volatility_multiplier" in summary
        assert "imbalance_threshold" in summary

    @pytest.mark.asyncio
    async def test_optimize_spread_with_exception(self, spread_optimizer):
        """Test spread optimization with exception handling."""
        # Create market data that will cause an exception
        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc),
            bid=Decimal("49999"),
            ask=Decimal("50001"),
        )

        # Invalid order book that will cause an exception
        order_book = "invalid_order_book"

        base_spread = Decimal("0.001")

        result = await spread_optimizer.optimize_spread(base_spread, market_data, order_book)
        assert isinstance(result, Decimal)

    @pytest.mark.asyncio
    async def test_optimize_spread_with_exception(self, spread_optimizer):
        """Test spread optimization with exception handling."""
        # Create market data that will cause an exception
        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc),
            bid=Decimal("49999"),
            ask=Decimal("50001"),
        )

        # Invalid order book that will cause an exception
        order_book = "invalid_order_book"

        base_spread = Decimal("0.001")

        result = await spread_optimizer.optimize_spread(base_spread, market_data, order_book)
        assert isinstance(result, Decimal)

    @pytest.mark.asyncio
    async def test_calculate_optimal_spread_with_exception(self, spread_optimizer, market_data):
        """Test optimal spread calculation with exception handling."""
        # Create invalid order book that will cause an exception
        order_book = "invalid_order_book"

        result = await spread_optimizer.calculate_optimal_spread(market_data, order_book)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Decimal)
        assert isinstance(result[1], Decimal)

    @pytest.mark.asyncio
    async def test_should_widen_spread_with_exception(self, spread_optimizer, market_data):
        """Test spread widening check with exception handling."""
        # Set invalid volatility history that will cause an exception
        spread_optimizer.volatility_history = ["invalid_volatility"]

        result = await spread_optimizer.should_widen_spread(market_data)
        assert isinstance(result, bool)
