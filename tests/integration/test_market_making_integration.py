"""
Integration tests for Market Making Strategy.

This module contains integration tests for the market making strategy
that test the complete workflow including signal generation, validation,
order placement, and inventory management.

CRITICAL: These tests verify the integration between all market making components.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

# Import core types
from src.core.types import (
    MarketData,
    OrderRequest,
    OrderResponse,
    OrderSide,
    Position,
    Signal,
    SignalDirection,
)

# Import exchange interface
from src.exchanges.base import BaseExchange

# Import risk management
from src.risk_management.base import BaseRiskManager
from src.strategies.static.inventory_manager import InventoryManager

# Import the strategy classes
from src.strategies.static.market_making import MarketMakingStrategy
from src.strategies.static.spread_optimizer import SpreadOptimizer


class TestMarketMakingIntegration:
    """Integration tests for Market Making Strategy."""

    @pytest.fixture
    def strategy_config(self):
        """Create a comprehensive test configuration."""
        return {
            "name": "test_market_making_integration",
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
                "rebalance_threshold": 0.2,
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
    def market_data_sequence(self):
        """Create a sequence of market data for testing."""
        base_price = 50000
        return [
            MarketData(
                symbol="BTCUSDT",
                price=Decimal(str(base_price + i * 10)),
                volume=Decimal("100"),
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=i),
                bid=Decimal(str(base_price + i * 10 - 1)),
                ask=Decimal(str(base_price + i * 10 + 1)),
                open_price=Decimal(str(base_price + i * 10 - 50)),
                high_price=Decimal(str(base_price + i * 10 + 50)),
                low_price=Decimal(str(base_price + i * 10 - 50)),
            )
            for i in range(10)
        ]

    @pytest.fixture
    def mock_exchange(self):
        """Create a mock exchange for testing."""
        exchange = Mock(spec=BaseExchange)
        exchange.place_order = AsyncMock(
            return_value=OrderResponse(
                id="test_order_123",
                client_order_id="test_client_123",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type="limit",
                quantity=Decimal("0.01"),
                price=Decimal("50000"),
                filled_quantity=Decimal("0.01"),
                status="filled",
                timestamp=datetime.now(timezone.utc),
            )
        )
        exchange.get_market_data = AsyncMock()
        exchange.get_order_book = AsyncMock()
        exchange.cancel_order = AsyncMock(return_value=True)
        return exchange

    @pytest.fixture
    def mock_risk_manager(self):
        """Create a mock risk manager for testing."""
        risk_manager = Mock(spec=BaseRiskManager)
        risk_manager.calculate_position_size = AsyncMock(return_value=Decimal("0.01"))
        risk_manager.validate_signal = AsyncMock(return_value=True)
        risk_manager.validate_order = AsyncMock(return_value=True)
        return risk_manager

    @pytest.fixture
    def strategy(self, strategy_config, mock_exchange, mock_risk_manager):
        """Create a test strategy instance with mocked dependencies."""
        strategy = MarketMakingStrategy(strategy_config)
        strategy.set_exchange(mock_exchange)
        strategy.set_risk_manager(mock_risk_manager)
        return strategy

    @pytest.mark.asyncio
    async def test_complete_market_making_workflow(self, strategy, market_data_sequence):
        """Test the complete market making workflow."""
        # Step 1: Generate signals from market data
        signals = await strategy._generate_signals_impl(market_data_sequence[0])

        assert len(signals) == 6  # 3 levels * 2 (bid + ask)

        # Step 2: Validate signals
        valid_signals = []
        for signal in signals:
            if await strategy.validate_signal(signal):
                valid_signals.append(signal)

        assert len(valid_signals) > 0

        # Step 3: Calculate position sizes
        for signal in valid_signals:
            size = strategy.get_position_size(signal)
            assert isinstance(size, Decimal)
            assert size > 0

        # Step 4: Update inventory state
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

        # Step 5: Check if rebalancing is needed
        should_rebalance = await strategy._should_rebalance_inventory(position)
        # This should be True since we're at 30% but target is 50%
        assert should_rebalance is True

    @pytest.mark.asyncio
    async def test_signal_generation_with_volatility(self, strategy):
        """Test signal generation with varying volatility."""
        # Create market data with increasing volatility
        base_price = 50000
        market_data_list = []

        for i in range(25):
            # Simulate increasing volatility
            price_change = i * 2  # Increasing price changes
            market_data = MarketData(
                symbol="BTCUSDT",
                price=Decimal(str(base_price + price_change)),
                volume=Decimal("100"),
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=i),
                bid=Decimal(str(base_price + price_change - 1)),
                ask=Decimal(str(base_price + price_change + 1)),
            )
            market_data_list.append(market_data)

        # Update strategy with volatile data
        for data in market_data_list:
            strategy._update_price_history(data)

        # Generate signals with high volatility
        signals = await strategy._generate_signals_impl(market_data_list[-1])

        # Should have wider spreads due to volatility
        for signal in signals:
            if "spread" in signal.metadata:
                spread = signal.metadata["spread"]
                assert spread > 0.001  # Should be wider than base spread

    @pytest.mark.asyncio
    async def test_inventory_management_integration(self, strategy):
        """Test inventory management integration."""
        # Create inventory manager
        inventory_config = {
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

        inventory_manager = InventoryManager(inventory_config)

        # Test inventory update
        position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.7"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc),
        )

        await inventory_manager.update_inventory(position)
        assert inventory_manager.current_inventory == Decimal("0.7")
        assert inventory_manager.inventory_skew == 0.7

        # Test rebalancing check
        should_rebalance = await inventory_manager.should_rebalance()
        assert should_rebalance is True  # 70% vs 50% target

        # Test rebalancing order calculation
        orders = await inventory_manager.calculate_rebalance_orders(Decimal("50000"))
        assert len(orders) == 1
        # Need to sell to reduce inventory
        assert orders[0].side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_spread_optimization_integration(self, strategy):
        """Test spread optimization integration."""
        # Create spread optimizer
        optimizer_config = {
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

        spread_optimizer = SpreadOptimizer(optimizer_config)

        # Create market data with order book
        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc),
            bid=Decimal("49999"),
            ask=Decimal("50001"),
        )

        order_book = {
            "symbol": "BTCUSDT",
            "bids": [[Decimal("49999"), Decimal("2.0")], [Decimal("49998"), Decimal("1.0")]],
            "asks": [[Decimal("50001"), Decimal("1.0")], [Decimal("50002"), Decimal("2.0")]],
            "timestamp": datetime.now(timezone.utc),
        }

        # Test spread optimization
        base_spread = Decimal("0.001")
        optimized_spread = await spread_optimizer.optimize_spread(
            base_spread, market_data, order_book
        )

        assert isinstance(optimized_spread, Decimal)
        assert optimized_spread >= spread_optimizer.min_spread
        assert optimized_spread <= spread_optimizer.max_spread

        # Test optimal spread calculation
        bid_spread, ask_spread = await spread_optimizer.calculate_optimal_spread(
            market_data, order_book
        )

        assert isinstance(bid_spread, Decimal)
        assert isinstance(ask_spread, Decimal)
        assert bid_spread > 0
        assert ask_spread > 0

    @pytest.mark.asyncio
    async def test_risk_management_integration(self, strategy, mock_risk_manager):
        """Test risk management integration."""
        # Test signal validation with risk manager
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name=strategy.name,
            metadata={"price": 50000.0, "size": 0.01, "level": 1, "spread": 0.001, "side": "bid"},
        )

        # Mock risk manager to reject signal
        mock_risk_manager.validate_signal.return_value = False

        is_valid = await strategy.validate_signal(signal)
        assert is_valid is False

        # Mock risk manager to accept signal
        mock_risk_manager.validate_signal.return_value = True

        is_valid = await strategy.validate_signal(signal)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_exchange_integration(self, strategy, mock_exchange):
        """Test exchange integration."""
        # Test order placement
        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type="limit",
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
            client_order_id="test_order",
        )

        # Mock successful order placement
        mock_exchange.place_order.return_value = OrderResponse(
            id="test_order_123",
            client_order_id="test_order",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type="limit",
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
            filled_quantity=Decimal("0.01"),
            status="filled",
            timestamp=datetime.now(timezone.utc),
        )

        response = await mock_exchange.place_order(order_request)
        assert response.status == "filled"
        assert response.filled_quantity == Decimal("0.01")

        # Test order cancellation
        cancelled = await mock_exchange.cancel_order("test_order_123")
        assert cancelled is True

    @pytest.mark.asyncio
    async def test_performance_tracking_integration(self, strategy):
        """Test performance tracking integration."""
        # Simulate multiple trades
        trade_results = [
            {"pnl": 10.5, "quantity": 0.01, "price": 50000.0},
            {"pnl": -5.2, "quantity": 0.01, "price": 50000.0},
            {"pnl": 15.8, "quantity": 0.01, "price": 50000.0},
            {"pnl": 8.3, "quantity": 0.01, "price": 50000.0},
        ]

        for trade_result in trade_results:
            await strategy.update_performance_metrics(trade_result)

        # Check performance metrics
        assert strategy.total_trades == 4
        assert strategy.profitable_trades == 3
        assert strategy.total_pnl == Decimal("29.4")  # 10.5 - 5.2 + 15.8 + 8.3
        assert strategy.metrics.win_rate == 0.75  # 3/4

        # Check strategy info
        info = strategy.get_strategy_info()
        assert info["total_trades"] == 4
        assert info["profitable_trades"] == 3
        assert float(info["total_pnl"]) == 29.4

    @pytest.mark.asyncio
    async def test_emergency_liquidation_integration(self, strategy):
        """Test emergency liquidation integration."""
        # Create inventory manager
        inventory_config = {
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

        inventory_manager = InventoryManager(inventory_config)

        # Set inventory to emergency level
        position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.9"),  # 90% of max inventory
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc),
        )

        await inventory_manager.update_inventory(position)

        # Check if emergency liquidation is needed
        should_emergency = await inventory_manager.should_emergency_liquidate()
        assert should_emergency is True

        # Calculate emergency orders
        emergency_orders = await inventory_manager.calculate_emergency_orders(Decimal("50000"))
        assert len(emergency_orders) == 1
        assert emergency_orders[0].side == OrderSide.SELL
        assert emergency_orders[0].quantity == Decimal("0.9")

    @pytest.mark.asyncio
    async def test_market_conditions_adaptation(self, strategy):
        """Test adaptation to changing market conditions."""
        # Create market data with changing conditions
        market_conditions = [
            # Normal conditions
            {"price": 50000, "bid": 49999, "ask": 50001, "volatility": 0.01},
            # High volatility
            {"price": 50100, "bid": 50099, "ask": 50101, "volatility": 0.05},
            # Wide spread
            {"price": 50200, "bid": 50150, "ask": 50250, "volatility": 0.02},
            # Narrow spread
            {"price": 50300, "bid": 50299, "ask": 50301, "volatility": 0.01},
        ]

        for i, condition in enumerate(market_conditions):
            market_data = MarketData(
                symbol="BTCUSDT",
                price=Decimal(str(condition["price"])),
                volume=Decimal("100"),
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=i),
                bid=Decimal(str(condition["bid"])),
                ask=Decimal(str(condition["ask"])),
            )

            # Update strategy with new market data
            strategy._update_price_history(market_data)

            # Generate signals
            signals = await strategy._generate_signals_impl(market_data)

            # Check that signals adapt to market conditions
            for signal in signals:
                if "spread" in signal.metadata:
                    spread = signal.metadata["spread"]

                    if condition["volatility"] > 0.03:  # High volatility
                        # Spreads should be wider in high volatility
                        assert spread > 0.001

                    if condition["ask"] - condition["bid"] > 50:  # Wide spread
                        # Should adapt to wide market spread
                        assert spread > 0.001

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, strategy, mock_exchange):
        """Test error handling integration."""
        # Test exchange error handling
        mock_exchange.place_order.side_effect = Exception("Exchange error")

        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type="limit",
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
        )

        # Should handle exchange errors gracefully
        with pytest.raises(Exception):
            await mock_exchange.place_order(order_request)

        # Test signal generation with invalid data
        invalid_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc),
            bid=None,  # Invalid data
            ask=None,
        )

        signals = await strategy._generate_signals_impl(invalid_data)
        assert len(signals) == 0  # Graceful degradation

        # Test validation with invalid signal
        invalid_signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.1,  # Too low
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name=strategy.name,
            metadata={},  # Missing required metadata
        )

        is_valid = await strategy.validate_signal(invalid_signal)
        assert is_valid is False
