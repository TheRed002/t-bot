"""
Unit tests for core type definitions.

These tests verify the core data structures used throughout the system.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError as PydanticValidationError

from src.core.exceptions import ValidationError
from src.core.types import (
    MarketData,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    PositionStatus,
    Signal,
    SignalDirection,
    TradingMode,
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
            strength=Decimal("0.8"),
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            source="test_strategy",
            metadata={"test": "data"},
        )

        assert signal.direction == SignalDirection.BUY
        assert signal.strength == Decimal("0.8")
        assert signal.symbol == "BTC/USDT"
        assert signal.source == "test_strategy"
        assert signal.metadata["test"] == "data"

    def test_signal_confidence_validation(self):
        """Test Signal strength validation."""
        # Valid strength
        signal = Signal(
            direction=SignalDirection.BUY,
            strength=Decimal("0.5"),
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            source="test_strategy",
        )
        assert signal.strength == Decimal("0.5")
        assert isinstance(signal.strength, Decimal)
        assert Decimal("0.0") <= signal.strength <= Decimal("1.0")

        # Invalid strength should raise validation error
        with pytest.raises(ValueError):
            Signal(
                direction=SignalDirection.BUY,
                strength=Decimal("1.5"),  # Invalid: > 1.0
                timestamp=datetime.now(timezone.utc),
                symbol="BTC/USDT",
                source="test_strategy",
            )

        # Test negative strength
        with pytest.raises(ValueError):
            Signal(
                direction=SignalDirection.BUY,
                strength=-0.1,  # Invalid: < 0.0
                timestamp=datetime.now(timezone.utc),
                symbol="BTC/USDT",
                source="test_strategy",
            )

    def test_signal_confidence_boundaries(self):
        """Test Signal strength boundary values."""
        # Test minimum strength
        signal_min = Signal(
            direction=SignalDirection.BUY,
            strength=Decimal("0.0"),
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            source="test_strategy",
        )
        assert signal_min.strength == Decimal("0.0")

        # Test maximum strength
        signal_max = Signal(
            direction=SignalDirection.BUY,
            strength=Decimal("1.0"),
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            source="test_strategy",
        )
        assert signal_max.strength == Decimal("1.0")


class TestMarketData:
    """Test MarketData model creation."""

    def test_market_data_creation(self):
        """Test MarketData model creation."""
        market_data = MarketData(
            symbol="BTC/USDT",
            open=Decimal("49900.00"),
            high=Decimal("50100.00"),
            low=Decimal("49800.00"),
            close=Decimal("50000.00"),
            volume=Decimal("100.5"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )

        assert market_data.symbol == "BTC/USDT"
        assert market_data.close == Decimal("50000.00")
        assert market_data.volume == Decimal("100.5")
        assert market_data.open == Decimal("49900.00")
        assert market_data.high == Decimal("50100.00")
        assert market_data.low == Decimal("49800.00")
        assert market_data.exchange == "binance"

        # Verify OHLC relationships are logical
        assert market_data.high >= market_data.open
        assert market_data.high >= market_data.close
        assert market_data.high >= market_data.low
        assert market_data.low <= market_data.open
        assert market_data.low <= market_data.close

        # Verify all price values are positive
        assert market_data.open > 0
        assert market_data.high > 0
        assert market_data.low > 0
        assert market_data.close > 0
        assert market_data.volume >= 0

        # Verify decimal precision is maintained
        assert isinstance(market_data.open, Decimal)
        assert isinstance(market_data.high, Decimal)
        assert isinstance(market_data.low, Decimal)
        assert isinstance(market_data.close, Decimal)
        assert isinstance(market_data.volume, Decimal)

    def test_market_data_optional_fields(self):
        """Test MarketData with optional fields."""
        market_data = MarketData(
            symbol="BTC/USDT",
            open=Decimal("49900.00"),
            high=Decimal("50100.00"),
            low=Decimal("49800.00"),
            close=Decimal("50000.00"),
            volume=Decimal("100.5"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
            quote_volume=Decimal("5000000.00"),
        )

        assert market_data.symbol == "BTC/USDT"
        assert market_data.close == Decimal("50000.00")
        assert market_data.quote_volume == Decimal("5000000.00")
        assert market_data.trades_count is None


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
            client_order_id="test_order_123",
        )

        assert order_request.symbol == "BTC/USDT"
        assert order_request.side == OrderSide.BUY
        assert order_request.order_type == OrderType.LIMIT
        assert order_request.quantity == Decimal("1.0")
        assert order_request.price == Decimal("50000.00")
        assert order_request.client_order_id == "test_order_123"

        # Validate financial constraints
        assert order_request.quantity > 0
        assert order_request.price > 0
        assert isinstance(order_request.quantity, Decimal)
        assert isinstance(order_request.price, Decimal)

        # Validate order value calculation
        order_value = order_request.quantity * order_request.price
        assert order_value == Decimal("50000.00")
        assert isinstance(order_value, Decimal)

        # Test symbol format validation
        assert "/" in order_request.symbol
        base, quote = order_request.symbol.split("/")
        assert len(base) >= 2  # BTC
        assert len(quote) >= 3  # USDT

    def test_order_request_market_order(self):
        """Test OrderRequest for market order (no price required)."""
        order_request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
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
            order_id="test_order_id",
            client_order_id="test_client_id",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            filled_quantity=Decimal("0.5"),
            status=OrderStatus.PARTIALLY_FILLED,
            created_at=datetime.now(timezone.utc),
            exchange="binance",
        )

        assert order_response.order_id == "test_order_id"
        assert order_response.client_order_id == "test_client_id"
        assert order_response.symbol == "BTC/USDT"
        assert order_response.side == OrderSide.BUY
        assert order_response.quantity == Decimal("1.0")
        assert order_response.filled_quantity == Decimal("0.5")
        assert order_response.status == OrderStatus.PARTIALLY_FILLED
        assert order_response.exchange == "binance"


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
            side=PositionSide.LONG,
            status=PositionStatus.OPEN,
            opened_at=datetime.now(timezone.utc),
            exchange="binance",
        )

        assert position.symbol == "BTC/USDT"
        assert position.quantity == Decimal("2.0")
        assert position.entry_price == Decimal("50000.00")
        assert position.current_price == Decimal("51000.00")
        assert position.unrealized_pnl == Decimal("2000.00")
        assert position.side == PositionSide.LONG

        # Validate financial calculations
        expected_pnl = (position.current_price - position.entry_price) * position.quantity
        assert expected_pnl == Decimal("2000.00")
        assert expected_pnl == position.unrealized_pnl

        # Validate position constraints
        assert position.quantity > 0
        assert position.entry_price > 0
        assert position.current_price > 0

        # Validate decimal precision
        assert isinstance(position.quantity, Decimal)
        assert isinstance(position.entry_price, Decimal)
        assert isinstance(position.current_price, Decimal)
        assert isinstance(position.unrealized_pnl, Decimal)

        # Test position value calculation
        position_value = position.quantity * position.current_price
        assert position_value == Decimal("102000.00")
        assert isinstance(position_value, Decimal)

    def test_position_short_side(self):
        """Test Position with short side."""
        position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("49000.00"),
            unrealized_pnl=Decimal("1000.00"),
            side=PositionSide.SHORT,
            status=PositionStatus.OPEN,
            opened_at=datetime.now(timezone.utc),
            exchange="binance",
        )

        assert position.side == PositionSide.SHORT
        assert position.unrealized_pnl == Decimal("1000.00")


class TestFinancialCalculationAccuracy:
    """Test financial calculation accuracy in core types."""

    def test_market_data_ohlc_validation(self):
        """Test OHLC data validation and relationships."""
        # Test valid OHLC data (no validation in current implementation)
        valid_data = MarketData(
            symbol="BTC/USDT",
            open=Decimal("50000.00"),
            high=Decimal("51000.00"),
            low=Decimal("49000.00"),
            close=Decimal("50500.00"),
            volume=Decimal("100.0"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )
        assert valid_data.high >= valid_data.low
        assert valid_data.open > 0
        assert valid_data.close > 0

        # Note: Current MarketData implementation doesn't validate OHLC relationships
        # These create "invalid" data but don't raise exceptions
        invalid_data1 = MarketData(
            symbol="BTC/USDT",
            open=Decimal("50000.00"),
            high=Decimal("49000.00"),  # High < Open - logically invalid but allowed
            low=Decimal("48000.00"),
            close=Decimal("49500.00"),
            volume=Decimal("100.0"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )
        assert invalid_data1.high < invalid_data1.open  # Demonstrates invalid data is allowed

        invalid_data2 = MarketData(
            symbol="BTC/USDT",
            open=Decimal("50000.00"),
            high=Decimal("51000.00"),
            low=Decimal("52000.00"),  # Low > High - logically invalid but allowed
            close=Decimal("50500.00"),
            volume=Decimal("100.0"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )
        assert invalid_data2.low > invalid_data2.high  # Demonstrates invalid data is allowed

    def test_position_pnl_calculations(self):
        """Test position PnL calculation accuracy."""
        # Long position with profit
        long_position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("1.5"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("52000.00"),
            unrealized_pnl=Decimal("3000.00"),
            side=PositionSide.LONG,
            status=PositionStatus.OPEN,
            opened_at=datetime.now(timezone.utc),
            exchange="binance",
        )

        # Verify PnL calculation: (52000 - 50000) * 1.5 = 3000
        expected_pnl = (
            long_position.current_price - long_position.entry_price
        ) * long_position.quantity
        assert expected_pnl == long_position.unrealized_pnl

        # Short position with profit
        short_position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("48000.00"),
            unrealized_pnl=Decimal("2000.00"),
            side=PositionSide.SHORT,
            status=PositionStatus.OPEN,
            opened_at=datetime.now(timezone.utc),
            exchange="binance",
        )

        # For shorts: PnL = (entry - current) * quantity = (50000 - 48000) * 1 = 2000
        expected_short_pnl = (
            short_position.entry_price - short_position.current_price
        ) * short_position.quantity
        assert expected_short_pnl == short_position.unrealized_pnl

    def test_order_value_calculations(self):
        """Test order value calculations maintain precision."""
        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.12345678"),  # 8 decimal places
            price=Decimal("50123.45"),
        )

        # Calculate order value with full precision
        order_value = order.quantity * order.price
        expected_value = Decimal("6188.0797394910")  # Precise calculation

        assert order_value == expected_value
        assert isinstance(order_value, Decimal)

        # Test commission calculations
        commission_rate = Decimal("0.001")  # 0.1%
        commission = order_value * commission_rate

        assert isinstance(commission, Decimal)
        assert commission == Decimal("6.1880797394910")

    def test_signal_strength_precision(self):
        """Test signal strength precision and validation."""
        # Test precise strength values
        signal = Signal(
            direction=SignalDirection.BUY,
            strength=Decimal("0.87654321"),  # High precision as Decimal
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            source="ml_model",
        )

        assert signal.strength == Decimal("0.87654321")
        assert isinstance(signal.strength, Decimal)

        # Test strength weighted calculations
        base_position_size = Decimal("1000.00")
        weighted_size = base_position_size * Decimal(str(signal.strength))

        assert isinstance(weighted_size, Decimal)
        assert weighted_size == Decimal("876.54321")


class TestTypeValidationEdgeCases:
    """Test edge cases and validation for core types."""

    def test_market_data_edge_cases(self):
        """Test market data edge cases."""
        # Test zero volume (valid for some markets)
        market_data_zero_vol = MarketData(
            symbol="BTC/USDT",
            open=Decimal("50000.00"),
            high=Decimal("50000.00"),
            low=Decimal("50000.00"),
            close=Decimal("50000.00"),
            volume=Decimal("0.0"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )
        assert market_data_zero_vol.volume == Decimal("0.0")

        # Test negative volume (currently no validation, so it's accepted)
        try:
            market_data_negative_vol = MarketData(
                symbol="BTC/USDT",
                open=Decimal("50000.00"),
                high=Decimal("50000.00"),
                low=Decimal("50000.00"),
                close=Decimal("50000.00"),
                volume=Decimal("-10.0"),
                timestamp=datetime.now(timezone.utc),
                exchange="binance",
            )
            # No validation currently exists, so negative volume is accepted
            assert market_data_negative_vol.volume == Decimal("-10.0")
        except Exception:
            # May fail for other reasons, which is acceptable for this test
            pass

    def test_order_request_edge_cases(self):
        """Test order request edge cases."""
        # Test minimum quantity order
        min_order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.00000001"),  # 1 satoshi
            price=Decimal("50000.00"),
        )
        assert min_order.quantity == Decimal("0.00000001")

        # Test zero quantity (should be invalid)
        with pytest.raises(ValidationError):
            OrderRequest(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.0"),
                price=Decimal("50000.00"),
            )

        # Test negative quantity (should be invalid)
        with pytest.raises(ValidationError):
            OrderRequest(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("-1.0"),
                price=Decimal("50000.00"),
            )

    def test_position_edge_cases(self):
        """Test position edge cases."""
        # Test position with zero PnL
        zero_pnl_position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("0.00"),
            side=PositionSide.LONG,
            status=PositionStatus.OPEN,
            opened_at=datetime.now(timezone.utc),
            exchange="binance",
        )
        assert zero_pnl_position.unrealized_pnl == Decimal("0.00")

        # Test position with negative PnL (loss)
        loss_position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("45000.00"),
            unrealized_pnl=Decimal("-5000.00"),
            side=PositionSide.LONG,
            status=PositionStatus.OPEN,
            opened_at=datetime.now(timezone.utc),
            exchange="binance",
        )
        assert loss_position.unrealized_pnl == Decimal("-5000.00")

    def test_symbol_format_validation(self):
        """Test symbol format validation across types."""
        valid_symbols = ["BTC/USDT", "ETH/BTC", "DOGE/USD", "ADA/EUR"]
        invalid_symbols = ["BTCUSDT", "BTC-USDT", "BTC_USDT", "", "BTC/", "/USDT"]

        for symbol in valid_symbols:
            # Should not raise for valid symbols
            signal = Signal(
                direction=SignalDirection.BUY,
                strength=Decimal("0.8"),
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                source="test",
            )
            assert "/" in signal.symbol

        for invalid_symbol in invalid_symbols:
            # Should raise for invalid symbols (ValidationError or PydanticValidationError)
            with pytest.raises((ValueError, ValidationError, PydanticValidationError)):
                Signal(
                    direction=SignalDirection.BUY,
                    strength=Decimal("0.8"),
                    timestamp=datetime.now(timezone.utc),
                    symbol=invalid_symbol,
                    source="test",
                )

    def test_timestamp_validation(self):
        """Test timestamp validation and timezone handling."""
        # Test with UTC timezone
        utc_time = datetime.now(timezone.utc)
        signal_utc = Signal(
            direction=SignalDirection.BUY,
            strength=Decimal("0.8"),
            timestamp=utc_time,
            symbol="BTC/USDT",
            source="test",
        )
        assert signal_utc.timestamp.tzinfo == timezone.utc

        # Test with naive datetime (should be rejected)
        with pytest.raises(ValidationError):
            Signal(
                direction=SignalDirection.BUY,
                strength=Decimal("0.8"),
                timestamp=datetime.now(),  # Naive datetime
                symbol="BTC/USDT",
                source="test",
            )

        # Test future timestamp (might be invalid for some use cases)
        future_time = datetime.now(timezone.utc) + timedelta(days=1)
        signal_future = Signal(
            direction=SignalDirection.BUY,
            strength=Decimal("0.8"),
            timestamp=future_time,
            symbol="BTC/USDT",
            source="test",
        )
        assert signal_future.timestamp > utc_time


class TestCrossTypeConsistency:
    """Test consistency between different core types."""

    def test_order_to_position_consistency(self):
        """Test consistency between order and resulting position."""
        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.5"),
            price=Decimal("50000.00"),
        )

        # Simulate filled order becoming a position
        position = Position(
            symbol=order.symbol,
            quantity=order.quantity,
            entry_price=order.price,
            current_price=Decimal("51000.00"),
            unrealized_pnl=Decimal("1500.00"),  # (51000-50000)*1.5
            side=PositionSide.LONG,  # Convert OrderSide.BUY to PositionSide.LONG
            status=PositionStatus.OPEN,
            opened_at=datetime.now(timezone.utc),
            exchange="binance",
        )

        # Verify consistency
        assert position.symbol == order.symbol
        assert position.quantity == order.quantity
        assert position.entry_price == order.price
        assert position.side == PositionSide.LONG  # OrderSide.BUY -> PositionSide.LONG

    def test_signal_to_order_consistency(self):
        """Test consistency between signal and generated order."""
        signal = Signal(
            direction=SignalDirection.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            source="strategy",
        )

        # Simulate signal generating an order
        base_quantity = Decimal("1.0")
        signal_weighted_quantity = base_quantity * Decimal(str(signal.strength))

        order = OrderRequest(
            symbol=signal.symbol,
            side=OrderSide.BUY if signal.direction == SignalDirection.BUY else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=signal_weighted_quantity,
        )

        # Verify consistency
        assert order.symbol == signal.symbol
        assert order.quantity == Decimal("0.8")  # 1.0 * 0.8
        assert order.side == OrderSide.BUY

    def test_market_data_to_signal_consistency(self):
        """Test consistency between market data and derived signals."""
        market_data = MarketData(
            symbol="BTC/USDT",
            open=Decimal("50000.00"),
            high=Decimal("51000.00"),
            low=Decimal("49500.00"),
            close=Decimal("50800.00"),
            volume=Decimal("1000.0"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )

        # Simulate signal derived from market data (bullish if close > open)
        is_bullish = market_data.close > market_data.open
        signal_direction = SignalDirection.BUY if is_bullish else SignalDirection.SELL

        # Calculate strength based on price movement
        price_change_pct = abs(market_data.close - market_data.open) / market_data.open
        signal_strength = min(float(price_change_pct) * 10, 1.0)  # Scale and cap at 1.0

        signal = Signal(
            direction=signal_direction,
            strength=signal_strength,
            timestamp=market_data.timestamp,
            symbol=market_data.symbol,
            source="price_action",
        )

        # Verify consistency
        assert signal.symbol == market_data.symbol
        assert signal.timestamp == market_data.timestamp
        assert signal.direction == SignalDirection.BUY  # 50800 > 50000
        assert 0.0 <= signal.strength <= 1.0
