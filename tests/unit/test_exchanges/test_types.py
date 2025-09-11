"""
Tests for exchanges types module.

This module tests exchange-specific types and data structures used
across different exchange implementations.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError as PydanticValidationError

from src.exchanges.types import (
    ExchangeBalance,
    ExchangeCapability,
    ExchangeConnectionConfig,
    ExchangeErrorResponse,
    ExchangeFee,
    ExchangeHealthStatus,
    ExchangeOrder,
    ExchangeOrderBookLevel,
    ExchangeOrderBookSnapshot,
    ExchangePosition,
    ExchangeRateLimit,
    ExchangeTrade,
    ExchangeTradingPair,
    ExchangeTypes,
    ExchangeWebSocketMessage,
)


class TestExchangeTypes:
    """Test ExchangeTypes utility class."""

    @patch("src.utils.ValidationFramework.validate_symbol")
    def test_validate_symbol_success_with_alphanumeric(self, mock_validate):
        """Test symbol validation succeeds with valid alphanumeric symbol."""
        mock_validate.return_value = True

        result = ExchangeTypes.validate_symbol("BTCUSDT")

        assert result is True
        mock_validate.assert_called_once_with("BTCUSDT")

    @patch("src.utils.ValidationFramework.validate_symbol")
    def test_validate_symbol_success_with_numbers(self, mock_validate):
        """Test symbol validation succeeds with alphanumeric including numbers."""
        mock_validate.return_value = True

        result = ExchangeTypes.validate_symbol("BTC2USDT")

        assert result is True
        mock_validate.assert_called_once_with("BTC2USDT")

    @patch("src.utils.ValidationFramework.validate_symbol")
    def test_validate_symbol_success_lowercase_converted(self, mock_validate):
        """Test symbol validation succeeds and converts lowercase to uppercase."""
        mock_validate.return_value = True

        result = ExchangeTypes.validate_symbol("btcusdt")

        assert result is True
        mock_validate.assert_called_once_with("btcusdt")

    @patch("src.utils.ValidationFramework.validate_symbol")
    def test_validate_symbol_fails_core_validation(self, mock_validate):
        """Test symbol validation fails when core validation fails."""
        mock_validate.return_value = False

        result = ExchangeTypes.validate_symbol("BTCUSDT")

        assert result is False
        mock_validate.assert_called_once_with("BTCUSDT")

    @patch("src.utils.ValidationFramework.validate_symbol")
    def test_validate_symbol_fails_with_separators(self, mock_validate):
        """Test symbol validation fails with separators (exchange-specific rule)."""
        mock_validate.return_value = True

        result = ExchangeTypes.validate_symbol("BTC-USDT")

        assert result is False  # Fails exchange-specific alphanumeric-only rule

    @patch("src.utils.ValidationFramework.validate_symbol")
    def test_validate_symbol_fails_with_slash(self, mock_validate):
        """Test symbol validation fails with slash separator."""
        mock_validate.return_value = True

        result = ExchangeTypes.validate_symbol("BTC/USDT")

        assert result is False  # Fails exchange-specific alphanumeric-only rule

    @patch("src.utils.ValidationFramework.validate_symbol")
    def test_validate_symbol_fails_with_special_chars(self, mock_validate):
        """Test symbol validation fails with special characters."""
        mock_validate.return_value = True

        result = ExchangeTypes.validate_symbol("BTC@USDT")

        assert result is False  # Fails exchange-specific alphanumeric-only rule

    @patch("src.utils.ValidationFramework.validate_symbol")
    def test_validate_symbol_exception_handling(self, mock_validate):
        """Test symbol validation handles exceptions gracefully."""
        mock_validate.side_effect = Exception("Validation framework error")

        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            result = ExchangeTypes.validate_symbol("BTCUSDT")

            assert result is False
            mock_logger.error.assert_called_once()
            # Verify error message contains the symbol and exception
            error_call = mock_logger.error.call_args[0][0]
            assert "BTCUSDT" in error_call
            assert "Validation framework error" in error_call

    def test_validate_symbol_empty_string(self):
        """Test symbol validation with empty string."""
        with patch("src.utils.ValidationFramework.validate_symbol") as mock_validate:
            mock_validate.return_value = True

            result = ExchangeTypes.validate_symbol("")

            assert result is False  # Empty string fails regex


class TestExchangeCapability:
    """Test ExchangeCapability enum."""

    def test_all_capabilities_exist(self):
        """Test all expected capabilities are defined."""
        expected_capabilities = {
            "spot_trading",
            "futures_trading",
            "margin_trading",
            "staking",
            "lending",
            "derivatives",
        }

        actual_capabilities = {cap.value for cap in ExchangeCapability}
        assert actual_capabilities == expected_capabilities

    def test_capability_values(self):
        """Test individual capability values."""
        assert ExchangeCapability.SPOT_TRADING.value == "spot_trading"
        assert ExchangeCapability.FUTURES_TRADING.value == "futures_trading"
        assert ExchangeCapability.MARGIN_TRADING.value == "margin_trading"
        assert ExchangeCapability.STAKING.value == "staking"
        assert ExchangeCapability.LENDING.value == "lending"
        assert ExchangeCapability.DERIVATIVES.value == "derivatives"


class TestExchangeTradingPair:
    """Test ExchangeTradingPair model."""

    def test_trading_pair_creation_success(self):
        """Test successful trading pair creation."""
        pair = ExchangeTradingPair(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            min_quantity=Decimal("0.00001"),
            max_quantity=Decimal("9000"),
            step_size=Decimal("0.00001"),
            min_price=Decimal("0.01"),
            max_price=Decimal("1000000"),
            tick_size=Decimal("0.01"),
        )

        assert pair.symbol == "BTCUSDT"
        assert pair.base_asset == "BTC"
        assert pair.quote_asset == "USDT"
        assert pair.is_active is True  # Default value

    def test_trading_pair_inactive(self):
        """Test trading pair with inactive status."""
        pair = ExchangeTradingPair(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            min_quantity=Decimal("0.00001"),
            max_quantity=Decimal("9000"),
            step_size=Decimal("0.00001"),
            min_price=Decimal("0.01"),
            max_price=Decimal("1000000"),
            tick_size=Decimal("0.01"),
            is_active=False,
        )

        assert pair.is_active is False

    def test_trading_pair_validation_errors(self):
        """Test trading pair validation with missing fields."""
        with pytest.raises(PydanticValidationError):
            ExchangeTradingPair(
                symbol="BTCUSDT"
                # Missing required fields
            )


class TestExchangeFee:
    """Test ExchangeFee model."""

    def test_exchange_fee_defaults(self):
        """Test exchange fee with default values."""
        fee = ExchangeFee()

        assert fee.maker_fee == Decimal("0.001")
        assert fee.taker_fee == Decimal("0.001")
        assert fee.min_fee == Decimal("0")
        assert fee.max_fee == Decimal("0.01")

    def test_exchange_fee_custom_values(self):
        """Test exchange fee with custom values."""
        fee = ExchangeFee(
            maker_fee=Decimal("0.0005"),
            taker_fee=Decimal("0.0007"),
            min_fee=Decimal("0.5"),
            max_fee=Decimal("10.0"),
        )

        assert fee.maker_fee == Decimal("0.0005")
        assert fee.taker_fee == Decimal("0.0007")
        assert fee.min_fee == Decimal("0.5")
        assert fee.max_fee == Decimal("10.0")


class TestExchangeRateLimit:
    """Test ExchangeRateLimit model."""

    def test_rate_limit_creation(self):
        """Test rate limit creation."""
        rate_limit = ExchangeRateLimit(
            requests_per_minute=1200,
            orders_per_second=10,
            websocket_connections=5,
            weight_per_request=2,
        )

        assert rate_limit.requests_per_minute == 1200
        assert rate_limit.orders_per_second == 10
        assert rate_limit.websocket_connections == 5
        assert rate_limit.weight_per_request == 2

    def test_rate_limit_default_weight(self):
        """Test rate limit with default weight."""
        rate_limit = ExchangeRateLimit(
            requests_per_minute=1200, orders_per_second=10, websocket_connections=5
        )

        assert rate_limit.weight_per_request == 1  # Default value


class TestExchangeConnectionConfig:
    """Test ExchangeConnectionConfig model."""

    def test_connection_config_full(self):
        """Test connection config with all fields."""
        config = ExchangeConnectionConfig(
            base_url="https://api.exchange.com",
            websocket_url="wss://stream.exchange.com",
            api_key="test_key",
            api_secret="test_secret",
            passphrase="test_passphrase",
            timeout=60,
            max_retries=5,
            testnet=False,
        )

        assert config.base_url == "https://api.exchange.com"
        assert config.websocket_url == "wss://stream.exchange.com"
        assert config.api_key == "test_key"
        assert config.api_secret == "test_secret"
        assert config.passphrase == "test_passphrase"
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.testnet is False

    def test_connection_config_defaults(self):
        """Test connection config with default values."""
        config = ExchangeConnectionConfig(
            base_url="https://api.exchange.com",
            websocket_url="wss://stream.exchange.com",
            api_key="test_key",
            api_secret="test_secret",
        )

        assert config.passphrase is None
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.testnet is True


class TestExchangeOrderBookLevel:
    """Test ExchangeOrderBookLevel model."""

    def test_order_book_level_creation(self):
        """Test order book level creation."""
        level = ExchangeOrderBookLevel(
            price=Decimal("50000.00"), quantity=Decimal("1.5"), total_quantity=Decimal("5.0")
        )

        assert level.price == Decimal("50000.00")
        assert level.quantity == Decimal("1.5")
        assert level.total_quantity == Decimal("5.0")

    def test_order_book_level_default_total(self):
        """Test order book level with default total quantity."""
        level = ExchangeOrderBookLevel(price=Decimal("50000.00"), quantity=Decimal("1.5"))

        assert level.total_quantity == Decimal("0")


class TestExchangeOrderBookSnapshot:
    """Test ExchangeOrderBookSnapshot model."""

    def test_order_book_snapshot_creation(self):
        """Test order book snapshot creation."""
        now = datetime.now(timezone.utc)
        bids = [
            ExchangeOrderBookLevel(price=Decimal("49900"), quantity=Decimal("1.0")),
            ExchangeOrderBookLevel(price=Decimal("49800"), quantity=Decimal("2.0")),
        ]
        asks = [
            ExchangeOrderBookLevel(price=Decimal("50100"), quantity=Decimal("1.5")),
            ExchangeOrderBookLevel(price=Decimal("50200"), quantity=Decimal("0.5")),
        ]

        snapshot = ExchangeOrderBookSnapshot(
            symbol="BTCUSDT", bids=bids, asks=asks, timestamp=now, sequence_number=12345
        )

        assert snapshot.symbol == "BTCUSDT"
        assert len(snapshot.bids) == 2
        assert len(snapshot.asks) == 2
        assert snapshot.timestamp == now
        assert snapshot.sequence_number == 12345

    def test_order_book_snapshot_no_sequence(self):
        """Test order book snapshot without sequence number."""
        now = datetime.now(timezone.utc)

        snapshot = ExchangeOrderBookSnapshot(symbol="BTCUSDT", bids=[], asks=[], timestamp=now)

        assert snapshot.sequence_number is None


class TestExchangeTrade:
    """Test ExchangeTrade model."""

    def test_exchange_trade_creation(self):
        """Test exchange trade creation."""
        now = datetime.now(timezone.utc)

        trade = ExchangeTrade(
            id="12345",
            symbol="BTCUSDT",
            side="buy",
            quantity=Decimal("1.5"),
            price=Decimal("50000.00"),
            timestamp=now,
            fee=Decimal("1.0"),
            fee_currency="BTC",
            is_maker=True,
        )

        assert trade.id == "12345"
        assert trade.symbol == "BTCUSDT"
        assert trade.side == "buy"
        assert trade.quantity == Decimal("1.5")
        assert trade.price == Decimal("50000.00")
        assert trade.timestamp == now
        assert trade.fee == Decimal("1.0")
        assert trade.fee_currency == "BTC"
        assert trade.is_maker is True

    def test_exchange_trade_defaults(self):
        """Test exchange trade with default values."""
        now = datetime.now(timezone.utc)

        trade = ExchangeTrade(
            id="12345",
            symbol="BTCUSDT",
            side="sell",
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            timestamp=now,
        )

        assert trade.fee == Decimal("0")
        assert trade.fee_currency == "USDT"
        assert trade.is_maker is False


class TestExchangeBalance:
    """Test ExchangeBalance model."""

    def test_exchange_balance_creation(self):
        """Test exchange balance creation."""
        now = datetime.now(timezone.utc)

        balance = ExchangeBalance(
            asset="BTC",
            free_balance=Decimal("1.5"),
            locked_balance=Decimal("0.5"),
            total_balance=Decimal("2.0"),
            usd_value=Decimal("100000.00"),
            last_updated=now,
        )

        assert balance.asset == "BTC"
        assert balance.free_balance == Decimal("1.5")
        assert balance.locked_balance == Decimal("0.5")
        assert balance.total_balance == Decimal("2.0")
        assert balance.usd_value == Decimal("100000.00")
        assert balance.last_updated == now

    def test_exchange_balance_no_usd_value(self):
        """Test exchange balance without USD value."""
        now = datetime.now(timezone.utc)

        balance = ExchangeBalance(
            asset="BTC",
            free_balance=Decimal("1.5"),
            locked_balance=Decimal("0.5"),
            total_balance=Decimal("2.0"),
            last_updated=now,
        )

        assert balance.usd_value is None


class TestExchangePosition:
    """Test ExchangePosition model."""

    def test_exchange_position_creation(self):
        """Test exchange position creation."""
        position = ExchangePosition(
            symbol="BTCUSDT",
            side="long",
            quantity=Decimal("2.0"),
            entry_price=Decimal("48000.00"),
            mark_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("4000.00"),
            margin_type="cross",
            leverage=Decimal("10"),
            liquidation_price=Decimal("45000.00"),
        )

        assert position.symbol == "BTCUSDT"
        assert position.side == "long"
        assert position.quantity == Decimal("2.0")
        assert position.entry_price == Decimal("48000.00")
        assert position.mark_price == Decimal("50000.00")
        assert position.unrealized_pnl == Decimal("4000.00")
        assert position.margin_type == "cross"
        assert position.leverage == Decimal("10")
        assert position.liquidation_price == Decimal("45000.00")

    def test_exchange_position_defaults(self):
        """Test exchange position with default values."""
        position = ExchangePosition(
            symbol="BTCUSDT",
            side="short",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            mark_price=Decimal("48000.00"),
            unrealized_pnl=Decimal("2000.00"),
        )

        assert position.margin_type == "isolated"
        assert position.leverage == Decimal("1")
        assert position.liquidation_price is None


class TestExchangeOrder:
    """Test ExchangeOrder model."""

    def test_exchange_order_creation(self):
        """Test exchange order creation."""
        created_at = datetime.now(timezone.utc)
        updated_at = datetime.now(timezone.utc)

        order = ExchangeOrder(
            id="order_123",
            client_order_id="client_456",
            symbol="BTCUSDT",
            side="buy",
            order_type="limit",
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            stop_price=Decimal("49000.00"),
            filled_quantity=Decimal("0.5"),
            remaining_quantity=Decimal("0.5"),
            status="partially_filled",
            time_in_force="IOC",
            created_at=created_at,
            updated_at=updated_at,
            fees=Decimal("25.0"),
        )

        assert order.id == "order_123"
        assert order.client_order_id == "client_456"
        assert order.symbol == "BTCUSDT"
        assert order.side == "buy"
        assert order.order_type == "limit"
        assert order.quantity == Decimal("1.0")
        assert order.price == Decimal("50000.00")
        assert order.stop_price == Decimal("49000.00")
        assert order.filled_quantity == Decimal("0.5")
        assert order.remaining_quantity == Decimal("0.5")
        assert order.status == "partially_filled"
        assert order.time_in_force == "IOC"
        assert order.created_at == created_at
        assert order.updated_at == updated_at
        assert order.fees == Decimal("25.0")

    def test_exchange_order_defaults(self):
        """Test exchange order with default values."""
        created_at = datetime.now(timezone.utc)
        updated_at = datetime.now(timezone.utc)

        order = ExchangeOrder(
            id="order_123",
            client_order_id=None,
            symbol="BTCUSDT",
            side="sell",
            order_type="market",
            quantity=Decimal("1.0"),
            status="new",
            created_at=created_at,
            updated_at=updated_at,
        )

        assert order.client_order_id is None
        assert order.price is None
        assert order.stop_price is None
        assert order.filled_quantity == Decimal("0")
        assert order.remaining_quantity == Decimal("0")
        assert order.time_in_force == "GTC"
        assert order.fees == Decimal("0")


class TestExchangeWebSocketMessage:
    """Test ExchangeWebSocketMessage model."""

    def test_websocket_message_creation(self):
        """Test WebSocket message creation."""
        now = datetime.now(timezone.utc)
        data = {"price": "50000.00", "volume": "1.5"}

        message = ExchangeWebSocketMessage(
            channel="ticker", symbol="BTCUSDT", data=data, timestamp=now
        )

        assert message.channel == "ticker"
        assert message.symbol == "BTCUSDT"
        assert message.data == data
        assert message.timestamp == now

    def test_websocket_message_no_symbol(self):
        """Test WebSocket message without symbol."""
        now = datetime.now(timezone.utc)
        data = {"status": "connected"}

        message = ExchangeWebSocketMessage(channel="system", data=data, timestamp=now)

        assert message.symbol is None


class TestExchangeErrorResponse:
    """Test ExchangeErrorResponse model."""

    def test_error_response_creation(self):
        """Test error response creation."""
        now = datetime.now(timezone.utc)
        details = {"field": "quantity", "issue": "too_small"}

        error = ExchangeErrorResponse(
            code=400, message="Invalid quantity", details=details, timestamp=now
        )

        assert error.code == 400
        assert error.message == "Invalid quantity"
        assert error.details == details
        assert error.timestamp == now

    def test_error_response_defaults(self):
        """Test error response with default timestamp."""
        error = ExchangeErrorResponse(code=500, message="Internal server error")

        assert error.code == 500
        assert error.message == "Internal server error"
        assert error.details is None
        assert isinstance(error.timestamp, datetime)
        assert error.timestamp.tzinfo == timezone.utc


class TestExchangeHealthStatus:
    """Test ExchangeHealthStatus model."""

    def test_health_status_creation(self):
        """Test health status creation."""
        now = datetime.now(timezone.utc)

        health = ExchangeHealthStatus(
            exchange_name="binance",
            status="online",
            latency_ms=150.5,
            last_heartbeat=now,
            error_count=2,
            success_rate=0.98,
        )

        assert health.exchange_name == "binance"
        assert health.status == "online"
        assert health.latency_ms == 150.5
        assert health.last_heartbeat == now
        assert health.error_count == 2
        assert health.success_rate == 0.98

    def test_health_status_defaults(self):
        """Test health status with default values."""
        health = ExchangeHealthStatus(exchange_name="coinbase", status="maintenance")

        assert health.latency_ms is None
        assert health.last_heartbeat is None
        assert health.error_count == 0
        assert health.success_rate == 1.0


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_trading_pair_dict_conversion(self):
        """Test trading pair to dict conversion."""
        pair = ExchangeTradingPair(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            min_quantity=Decimal("0.00001"),
            max_quantity=Decimal("9000"),
            step_size=Decimal("0.00001"),
            min_price=Decimal("0.01"),
            max_price=Decimal("1000000"),
            tick_size=Decimal("0.01"),
        )

        pair_dict = pair.model_dump()

        assert pair_dict["symbol"] == "BTCUSDT"
        assert pair_dict["is_active"] is True

    def test_order_json_conversion(self):
        """Test order JSON serialization."""
        created_at = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        updated_at = datetime(2023, 1, 1, 12, 5, 0, tzinfo=timezone.utc)

        order = ExchangeOrder(
            id="order_123",
            client_order_id="client_123",
            symbol="BTCUSDT",
            side="buy",
            order_type="limit",
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            status="new",
            created_at=created_at,
            updated_at=updated_at,
        )

        # Test JSON serialization works
        json_str = order.model_dump_json()
        assert "order_123" in json_str
        assert "50000.00" in json_str

        # Test deserialization
        order_data = order.model_dump()
        recreated_order = ExchangeOrder(**order_data)
        assert recreated_order.id == "order_123"
        assert recreated_order.price == Decimal("50000.00")


class TestDecimalPrecision:
    """Test decimal precision handling in models."""

    def test_high_precision_decimals(self):
        """Test models handle high precision decimals correctly."""
        # Test with 8 decimal places (crypto precision)
        level = ExchangeOrderBookLevel(
            price=Decimal("50000.12345678"), quantity=Decimal("0.00000001")
        )

        assert level.price == Decimal("50000.12345678")
        assert level.quantity == Decimal("0.00000001")

    def test_decimal_arithmetic_preservation(self):
        """Test decimal arithmetic is preserved in models."""
        balance = ExchangeBalance(
            asset="BTC",
            free_balance=Decimal("1.12345678"),
            locked_balance=Decimal("0.87654322"),
            total_balance=Decimal("1.12345678") + Decimal("0.87654322"),
            last_updated=datetime.now(timezone.utc),
        )

        # Verify exact decimal arithmetic
        expected_total = Decimal("2.00000000")
        assert balance.total_balance == expected_total

    def test_fee_precision(self):
        """Test fee models maintain precision."""
        fee = ExchangeFee(
            maker_fee=Decimal("0.00075"),  # 0.075%
            taker_fee=Decimal("0.00100"),  # 0.1%
        )

        assert fee.maker_fee == Decimal("0.00075")
        assert fee.taker_fee == Decimal("0.00100")

        # Test fee calculation maintains precision
        order_value = Decimal("10000.00")
        calculated_fee = order_value * fee.maker_fee
        assert calculated_fee == Decimal("7.50000")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_order_book_snapshot(self):
        """Test order book snapshot with empty bids/asks."""
        now = datetime.now(timezone.utc)

        snapshot = ExchangeOrderBookSnapshot(symbol="BTCUSDT", bids=[], asks=[], timestamp=now)

        assert len(snapshot.bids) == 0
        assert len(snapshot.asks) == 0

    def test_zero_quantities_and_prices(self):
        """Test models with zero values."""
        balance = ExchangeBalance(
            asset="DUST",
            free_balance=Decimal("0"),
            locked_balance=Decimal("0"),
            total_balance=Decimal("0"),
            last_updated=datetime.now(timezone.utc),
        )

        assert balance.free_balance == Decimal("0")
        assert balance.total_balance == Decimal("0")

    def test_very_large_numbers(self):
        """Test models with very large numbers."""
        order = ExchangeOrder(
            id="large_order",
            client_order_id="large_client",
            symbol="TRILLION",
            side="buy",
            order_type="market",
            quantity=Decimal("999999999999.99999999"),
            status="new",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        assert order.quantity == Decimal("999999999999.99999999")

    def test_unicode_symbols(self):
        """Test models with unicode characters in strings."""
        trade = ExchangeTrade(
            id="unicode_test",
            symbol="BTC_USDT",  # Underscore separator
            side="buy",
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            timestamp=datetime.now(timezone.utc),
            fee_currency="₿TC",  # Unicode character
        )

        assert trade.fee_currency == "₿TC"
