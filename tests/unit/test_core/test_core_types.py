"""
Unit tests for core type definitions.

These tests verify the core data structures used throughout the system.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Any

from src.core.types import (
    TradingMode, SignalDirection, OrderSide, OrderType,
    Signal, MarketData, OrderRequest, OrderResponse, Position
)


class TestTradingMode:
    """Test TradingMode enum values."""

    def test_trading_mode_enum(self):
        """Test TradingMode enum values."""
        assert TradingMode.LIVE.value == "live"
        assert TradingMode.PAPER.value == "paper"
        assert TradingMode.BACKTEST.value == "backtest"

    def test_trading_mode_comparison(self):
        """Test TradingMode enum comparison."""
        assert TradingMode.LIVE == TradingMode.LIVE
        assert TradingMode.LIVE != TradingMode.PAPER


class TestSignalDirection:
    """Test SignalDirection enum values."""

    def test_signal_direction_enum(self):
        """Test SignalDirection enum values."""
        assert SignalDirection.BUY.value == "buy"
        assert SignalDirection.SELL.value == "sell"
        assert SignalDirection.HOLD.value == "hold"

    def test_signal_direction_comparison(self):
        """Test SignalDirection enum comparison."""
        assert SignalDirection.BUY == SignalDirection.BUY
        assert SignalDirection.BUY != SignalDirection.SELL


class TestOrderSide:
    """Test OrderSide enum values."""

    def test_order_side_enum(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_side_comparison(self):
        """Test OrderSide enum comparison."""
        assert OrderSide.BUY == OrderSide.BUY
        assert OrderSide.BUY != OrderSide.SELL


class TestOrderType:
    """Test OrderType enum values."""

    def test_order_type_enum(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP_LOSS.value == "stop_loss"
        assert OrderType.TAKE_PROFIT.value == "take_profit"

    def test_order_type_comparison(self):
        """Test OrderType enum comparison."""
        assert OrderType.MARKET == OrderType.MARKET
        assert OrderType.MARKET != OrderType.LIMIT


class TestSignal:
    """Test Signal model creation and validation."""

    def test_signal_creation(self):
        """Test Signal model creation and validation."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            strategy_name="test_strategy",
            metadata={"test": "data"}
        )

        assert signal.direction == SignalDirection.BUY
        assert signal.confidence == 0.8
        assert signal.symbol == "BTC/USDT"
        assert signal.strategy_name == "test_strategy"
        assert signal.metadata["test"] == "data"

    def test_signal_confidence_validation(self):
        """Test Signal confidence validation."""
        # Valid confidence
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.5,
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            strategy_name="test_strategy"
        )
        assert signal.confidence == 0.5

        # Invalid confidence should raise validation error
        with pytest.raises(ValueError):
            Signal(
                direction=SignalDirection.BUY,
                confidence=1.5,  # Invalid: > 1.0
                timestamp=datetime.now(timezone.utc),
                symbol="BTC/USDT",
                strategy_name="test_strategy"
            )

    def test_signal_confidence_boundaries(self):
        """Test Signal confidence boundary values."""
        # Test minimum confidence
        signal_min = Signal(
            direction=SignalDirection.BUY,
            confidence=0.0,
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            strategy_name="test_strategy"
        )
        assert signal_min.confidence == 0.0

        # Test maximum confidence
        signal_max = Signal(
            direction=SignalDirection.BUY,
            confidence=1.0,
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            strategy_name="test_strategy"
        )
        assert signal_max.confidence == 1.0


class TestMarketData:
    """Test MarketData model creation."""

    def test_market_data_creation(self):
        """Test MarketData model creation."""
        market_data = MarketData(
            symbol="BTC/USDT",
            price=Decimal("50000.00"),
            volume=Decimal("100.5"),
            timestamp=datetime.now(timezone.utc),
            bid=Decimal("49999.00"),
            ask=Decimal("50001.00"),
            open_price=Decimal("49900.00"),
            high_price=Decimal("50100.00"),
            low_price=Decimal("49800.00")
        )

        assert market_data.symbol == "BTC/USDT"
        assert market_data.price == Decimal("50000.00")
        assert market_data.volume == Decimal("100.5")
        assert market_data.bid == Decimal("49999.00")
        assert market_data.ask == Decimal("50001.00")

    def test_market_data_optional_fields(self):
        """Test MarketData with optional fields."""
        market_data = MarketData(
            symbol="BTC/USDT",
            price=Decimal("50000.00"),
            volume=Decimal("100.5"),
            timestamp=datetime.now(timezone.utc)
        )

        assert market_data.symbol == "BTC/USDT"
        assert market_data.price == Decimal("50000.00")
        assert market_data.bid is None
        assert market_data.ask is None


class TestOrderRequest:
    """Test OrderRequest model creation."""

    def test_order_request_creation(self):
        """Test OrderRequest model creation."""
        order_request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            time_in_force="GTC",
            client_order_id="test_order_123"
        )

        assert order_request.symbol == "BTC/USDT"
        assert order_request.side == OrderSide.BUY
        assert order_request.order_type == OrderType.LIMIT
        assert order_request.quantity == Decimal("1.0")
        assert order_request.price == Decimal("50000.00")
        assert order_request.client_order_id == "test_order_123"

    def test_order_request_market_order(self):
        """Test OrderRequest for market order (no price required)."""
        order_request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0")
        )

        assert order_request.symbol == "BTC/USDT"
        assert order_request.side == OrderSide.BUY
        assert order_request.order_type == OrderType.MARKET
        assert order_request.price is None


class TestOrderResponse:
    """Test OrderResponse model creation."""

    def test_order_response_creation(self):
        """Test OrderResponse model creation."""
        order_response = OrderResponse(
            id="test_order_id",
            client_order_id="test_client_id",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            filled_quantity=Decimal("0.5"),
            status="partially_filled",
            timestamp=datetime.now(timezone.utc)
        )

        assert order_response.id == "test_order_id"
        assert order_response.client_order_id == "test_client_id"
        assert order_response.symbol == "BTC/USDT"
        assert order_response.side == OrderSide.BUY
        assert order_response.filled_quantity == Decimal("0.5")
        assert order_response.status == "partially_filled"


class TestPosition:
    """Test Position model creation."""

    def test_position_creation(self):
        """Test Position model creation."""
        position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("2.0"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("51000.00"),
            unrealized_pnl=Decimal("2000.00"),
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc)
        )

        assert position.symbol == "BTC/USDT"
        assert position.quantity == Decimal("2.0")
        assert position.entry_price == Decimal("50000.00")
        assert position.current_price == Decimal("51000.00")
        assert position.unrealized_pnl == Decimal("2000.00")
        assert position.side == OrderSide.BUY

    def test_position_short_side(self):
        """Test Position with short side."""
        position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("49000.00"),
            unrealized_pnl=Decimal("1000.00"),
            side=OrderSide.SELL,
            timestamp=datetime.now(timezone.utc)
        )

        assert position.side == OrderSide.SELL
        assert position.unrealized_pnl == Decimal("1000.00")
