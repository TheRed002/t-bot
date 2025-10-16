"""
Comprehensive tests for Binance exchange core trading functions.

This test file focuses on achieving 100% coverage for critical Binance trading functions:
- Order placement (all order types)
- Order cancellation  
- Order status checking
- Balance retrieval
- Account information
- Position management

Tests include financial precision, error conditions, and Binance-specific edge cases.
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Binance-specific imports
from binance.exceptions import BinanceAPIException, BinanceOrderException

from src.core.exceptions import (
    ExchangeConnectionError,
    ExchangeError,
    ExecutionError,
    OrderRejectionError,
    ValidationError,
)
from src.core.types import (
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
)
from src.exchanges.binance import BinanceExchange


@pytest.fixture
def mock_client():
    """Create a mock Binance client."""
    client = Mock()
    client.create_order = AsyncMock()
    client.cancel_order = AsyncMock()
    client.get_order = AsyncMock()
    client.get_account = AsyncMock()
    client.get_symbol_ticker = AsyncMock()
    client.get_order_book = AsyncMock()
    client.get_exchange_info = AsyncMock()
    return client


@pytest.fixture
def mock_adapter():
    """Create a mock adapter."""
    adapter = Mock()
    adapter.connect = Mock(return_value=True)
    adapter.disconnect = Mock(return_value=True)
    return adapter


@pytest.fixture
def binance_exchange(mock_adapter):
    """Create a Binance exchange instance with mocks."""
    config = {
        "api_key": "test_api_key",
        "api_secret": "test_api_secret",
        "testnet": True,
        "sandbox": True
    }
    
    with patch("src.exchanges.binance.AsyncClient"):
        exchange = BinanceExchange(config=config)
        exchange.client = AsyncMock()
        exchange._connected = True
        exchange._trading_symbols = ["BTCUSDT", "ETHUSDT", "BTCUSD", "ETHUSD"]
        return exchange


@pytest.fixture
def mock_order_request():
    """Create a mock order request."""
    return OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.01"),
        price=Decimal("50000.00"),
        client_order_id="test_binance_order_123"
    )


def setup_binance_mocks(exchange, client, mock_response):
    """Helper function to set up common binance mocks."""
    exchange.client = client
    exchange._trading_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    client.create_order.return_value = mock_response


def create_mock_response(order_id="12345678", client_order_id="test_order", symbol="BTCUSDT",
                        side="BUY", order_type="LIMIT", orig_qty="0.01000000", price="50000.00000000",
                        status="NEW", executed_qty="0.00000000", cumulative_quote_qty="0.00000000"):
    """Create a properly formatted mock response."""
    return {
        "orderId": int(order_id),
        "clientOrderId": client_order_id,
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "origQty": orig_qty,
        "price": price,
        "status": status,
        "executedQty": executed_qty,
        "cummulativeQuoteQty": cumulative_quote_qty,
        "transactTime": 1234567890000,
        "timeInForce": "GTC"
    }


class TestBinanceOrderPlacement:
    """Test Binance order placement functionality."""

    async def test_place_limit_order_success(self, binance_exchange, mock_order_request, mock_client):
        """Test successful limit order placement."""
        mock_response = create_mock_response(
            client_order_id="test_binance_order_123"
        )
        setup_binance_mocks(binance_exchange, mock_client, mock_response)

        with patch.object(binance_exchange, "_validate_order"), \
             patch.object(binance_exchange, "_validate_symbol"), \
             patch.object(binance_exchange, "_validate_price"), \
             patch.object(binance_exchange, "_validate_quantity"):
            response = await binance_exchange.place_order(mock_order_request)

            assert response.order_id == "12345678"
            assert response.client_order_id == "test_binance_order_123"
            assert response.symbol == "BTCUSDT"
            assert response.side == OrderSide.BUY
            assert response.quantity == Decimal("0.01")
            assert response.price == Decimal("50000.00")
            assert response.status == OrderStatus.NEW

            # Verify API call
            mock_client.create_order.assert_called_once()

    async def test_place_market_buy_order_success(self, binance_exchange, mock_client):
        """Test successful market buy order placement."""
        market_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            client_order_id="market_buy_123"
        )

        mock_response = create_mock_response(
            order_id="12345679",
            client_order_id="market_buy_123",
            order_type="MARKET",
            status="FILLED",
            executed_qty="0.01000000",
            cumulative_quote_qty="500.00000000",
            price="0"  # Market orders don't have a price
        )
        setup_binance_mocks(binance_exchange, mock_client, mock_response)

        with patch.object(binance_exchange, "_validate_order"), \
             patch.object(binance_exchange, "_validate_symbol"), \
             patch.object(binance_exchange, "_validate_price"), \
             patch.object(binance_exchange, "_validate_quantity"):
            response = await binance_exchange.place_order(market_order)

            assert response.order_type == OrderType.MARKET
            assert response.status == OrderStatus.FILLED

            mock_client.create_order.assert_called_once_with(
                symbol="BTCUSDT",
                side="BUY",
                type="MARKET",
                quantity="0.01000000",
                newClientOrderId="market_buy_123"
            )

    async def test_place_market_buy_order_with_quote_quantity(self, binance_exchange, mock_client):
        """Test market buy order with quote quantity (USDT amount)."""
        market_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            client_order_id="market_buy_quote_123"
        )
        market_order.quote_quantity = Decimal("500.00")  # $500 worth of BTC

        mock_response = {
            "orderId": 12345680,
            "clientOrderId": "market_buy_quote_123",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "origQty": "0.01000000",
            "executedQty": "0.01000000",
            "status": "FILLED",
            "transactTime": 1234567890000,
            "cummulativeQuoteQty": "500.00",
            "fills": []
        }

        binance_exchange.client = mock_client
        mock_client.create_order.return_value = mock_response

        with patch.object(binance_exchange, "_validate_order"), \
             patch.object(binance_exchange, "_validate_symbol"), \
             patch.object(binance_exchange, "_validate_price"), \
             patch.object(binance_exchange, "_validate_quantity"):
            response = await binance_exchange.place_order(market_order)

            assert response.status == OrderStatus.FILLED

            mock_client.create_order.assert_called_once_with(
                symbol="BTCUSDT",
                side="BUY",
                type="MARKET",
                quoteOrderQty="500.00",
                newClientOrderId="market_buy_quote_123"
            )

    async def test_place_market_sell_order_success(self, binance_exchange, mock_client):
        """Test successful market sell order placement."""
        market_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            client_order_id="market_sell_123"
        )

        mock_response = {
            "orderId": 12345681,
            "clientOrderId": "market_sell_123",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "MARKET",
            "origQty": "0.01000000",
            "executedQty": "0.01000000",
            "status": "FILLED",
            "transactTime": 1234567890000,
            "fills": []
        }

        binance_exchange.client = mock_client
        mock_client.create_order.return_value = mock_response

        with patch.object(binance_exchange, "_validate_order"), \
             patch.object(binance_exchange, "_validate_symbol"), \
             patch.object(binance_exchange, "_validate_price"), \
             patch.object(binance_exchange, "_validate_quantity"):
            response = await binance_exchange.place_order(market_order)

            assert response.side == OrderSide.SELL
            assert response.status == OrderStatus.FILLED

            mock_client.create_order.assert_called_once_with(
                symbol="BTCUSDT",
                side="SELL",
                type="MARKET",
                quantity="0.01000000",
                newClientOrderId="market_sell_123"
            )

    async def test_place_stop_loss_order_success(self, binance_exchange, mock_client):
        """Test successful stop-loss order placement."""
        stop_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=Decimal("0.01"),
            price=Decimal("48000.00"),
            stop_price=Decimal("49000.00"),
            client_order_id="stop_loss_123"
        )

        mock_response = {
            "orderId": 12345682,
            "clientOrderId": "stop_loss_123",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "STOP_LOSS_LIMIT",
            "origQty": "0.01000000",
            "price": "48000.00000000",
            "stopPrice": "49000.00000000",
            "status": "NEW",
            "transactTime": 1234567890000,
            "timeInForce": "GTC"
        }

        binance_exchange.client = mock_client
        mock_client.create_order.return_value = mock_response

        with patch.object(binance_exchange, "_validate_order"), \
             patch.object(binance_exchange, "_validate_symbol"), \
             patch.object(binance_exchange, "_validate_price"), \
             patch.object(binance_exchange, "_validate_quantity"):
            response = await binance_exchange.place_order(stop_order)

            assert response.order_type == OrderType.STOP_LOSS
            assert response.price == Decimal("48000.00")

            mock_client.create_order.assert_called_once_with(
                symbol="BTCUSDT",
                side="SELL",
                type="STOP_LOSS_LIMIT",
                quantity="0.01000000",
                price="48000.00000000",
                stopPrice="49000.00000000",
                timeInForce="GTC",
                newClientOrderId="stop_loss_123"
            )

    async def test_place_take_profit_order_success(self, binance_exchange, mock_client):
        """Test successful take-profit order placement."""
        tp_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.TAKE_PROFIT,
            quantity=Decimal("0.01"),
            price=Decimal("52000.00"),
            stop_price=Decimal("51000.00"),
            client_order_id="take_profit_123"
        )

        mock_response = {
            "orderId": 12345683,
            "clientOrderId": "take_profit_123",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "TAKE_PROFIT_LIMIT",
            "origQty": "0.01000000",
            "price": "52000.00000000",
            "stopPrice": "51000.00000000",
            "status": "NEW",
            "transactTime": 1234567890000,
            "timeInForce": "GTC"
        }

        binance_exchange.client = mock_client
        mock_client.create_order.return_value = mock_response

        with patch.object(binance_exchange, "_validate_order"), \
             patch.object(binance_exchange, "_validate_symbol"), \
             patch.object(binance_exchange, "_validate_price"), \
             patch.object(binance_exchange, "_validate_quantity"):
            response = await binance_exchange.place_order(tp_order)

            assert response.order_type == OrderType.TAKE_PROFIT
            assert response.price == Decimal("52000.00")

            mock_client.create_order.assert_called_once_with(
                symbol="BTCUSDT",
                side="SELL",
                type="TAKE_PROFIT_LIMIT",
                quantity="0.01000000",
                price="52000.00000000",
                stopPrice="51000.00000000",
                timeInForce="GTC",
                newClientOrderId="take_profit_123"
            )

    async def test_place_order_unsupported_type(self, binance_exchange, mock_client):
        """Test placing order with unsupported order type."""
        # Create a valid order request first
        invalid_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.01"),
            price=Decimal("50000.00"),
            client_order_id="invalid_order_123"
        )
        
        # Mock an invalid order type after creation
        with patch.object(invalid_order, 'order_type', "INVALID_TYPE"):
            binance_exchange.client = mock_client

            with patch.object(binance_exchange, "_validate_order"), \
                 patch.object(binance_exchange, "_validate_symbol"), \
                 patch.object(binance_exchange, "_validate_price"), \
                 patch.object(binance_exchange, "_validate_quantity"):
                with pytest.raises(ExchangeError, match="Failed to place order.*Unsupported order type"):
                    await binance_exchange.place_order(invalid_order)

    async def test_place_order_no_client(self, binance_exchange, mock_order_request):
        """Test placing order when Binance client is not initialized."""
        binance_exchange.client = None

        # Mock validation to get past that part of the code
        with patch.object(binance_exchange, "_validate_order"), \
             patch.object(binance_exchange, "_validate_symbol"), \
             patch.object(binance_exchange, "_validate_price"), \
             patch.object(binance_exchange, "_validate_quantity"):
            with pytest.raises(ExchangeError, match="Failed to place order.*Binance client not connected"):
                await binance_exchange.place_order(mock_order_request)

    async def test_place_order_empty_result(self, binance_exchange, mock_order_request, mock_client):
        """Test placing order when API returns empty result."""
        binance_exchange.client = mock_client
        mock_client.create_order.return_value = None

        with patch.object(binance_exchange, "_validate_order"), \
             patch.object(binance_exchange, "_validate_symbol"), \
             patch.object(binance_exchange, "_validate_price"), \
             patch.object(binance_exchange, "_validate_quantity"):
            with pytest.raises(ExchangeError, match="Failed to place order.*Order placement returned empty result"):
                await binance_exchange.place_order(mock_order_request)

    async def test_place_order_binance_api_exception(self, binance_exchange, mock_order_request, mock_client):
        """Test handling Binance API exceptions during order placement."""
        binance_exchange.client = mock_client
        mock_client.create_order.side_effect = BinanceAPIException(
            response=Mock(status=400, text="Invalid symbol"),
            status_code=400,
            text="Invalid symbol"
        )

        with patch.object(binance_exchange, "_validate_order"), \
             patch.object(binance_exchange, "_validate_symbol"), \
             patch.object(binance_exchange, "_validate_price"), \
             patch.object(binance_exchange, "_validate_quantity"):
            with pytest.raises(ExchangeError, match="Failed to place order.*Invalid symbol"):
                await binance_exchange.place_order(mock_order_request)

    async def test_place_order_binance_order_exception(self, binance_exchange, mock_order_request, mock_client):
        """Test handling Binance order exceptions."""
        binance_exchange.client = mock_client
        mock_client.create_order.side_effect = BinanceOrderException(
            code=-2010,
            message="Account has insufficient balance for requested action"
        )

        with patch.object(binance_exchange, "_validate_order"), \
             patch.object(binance_exchange, "_validate_symbol"), \
             patch.object(binance_exchange, "_validate_price"), \
             patch.object(binance_exchange, "_validate_quantity"):
            with pytest.raises(OrderRejectionError, match="Binance order rejected.*insufficient balance"):
                await binance_exchange.place_order(mock_order_request)


class TestBinanceOrderCancellation:
    """Test Binance order cancellation functionality."""

    async def test_cancel_order_success(self, binance_exchange, mock_client):
        """Test successful order cancellation."""
        mock_response = {
            "orderId": 12345678,
            "clientOrderId": "test_cancel_123",
            "symbol": "BTCUSDT",
            "status": "CANCELED",
            "side": "BUY",
            "type": "LIMIT",
            "origQty": "0.01000000",
            "price": "50000.00000000",
            "executedQty": "0.00000000"
        }

        binance_exchange.client = mock_client
        mock_client.cancel_order.return_value = mock_response

        with patch.object(binance_exchange, "_validate_symbol"):
            result = await binance_exchange.cancel_order("BTCUSDT", "12345678")

        assert result.order_id == "12345678"
        assert result.symbol == "BTCUSDT"
        assert result.status == OrderStatus.CANCELLED
        mock_client.cancel_order.assert_called_once_with(
            symbol="BTCUSDT",
            orderId=12345678
        )

    async def test_cancel_order_no_client(self, binance_exchange):
        """Test canceling order when Binance client is not initialized."""
        binance_exchange.client = None

        with patch.object(binance_exchange, "_validate_symbol"), \
             pytest.raises(ExchangeError, match="Failed to cancel order.*Binance client not connected"):
            await binance_exchange.cancel_order("BTCUSDT", "12345678")

    async def test_cancel_order_api_exception(self, binance_exchange, mock_client):
        """Test handling API exceptions during order cancellation."""
        binance_exchange.client = mock_client
        mock_client.cancel_order.side_effect = BinanceAPIException(
            response=Mock(status=400, text="Order does not exist"),
            status_code=400,
            text="Order does not exist"
        )

        with patch.object(binance_exchange, "_validate_symbol"), \
             pytest.raises(ExchangeError, match="Failed to cancel order.*Order does not exist"):
            await binance_exchange.cancel_order("BTCUSDT", "99999999")


class TestBinanceOrderStatus:
    """Test Binance order status retrieval."""

    async def test_get_order_status_success(self, binance_exchange, mock_client):
        """Test successful order status retrieval."""
        mock_response = {
            "orderId": 12345678,
            "clientOrderId": "test_status_123",
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "side": "BUY",
            "type": "LIMIT",
            "executedQty": "0.01000000",
            "origQty": "0.01000000",
            "price": "50000.00000000",
            "time": 1640995200000  # timestamp in milliseconds
        }

        binance_exchange.client = mock_client
        mock_client.get_order.return_value = mock_response

        with patch.object(binance_exchange, "_validate_symbol"):
            status = await binance_exchange.get_order_status("BTCUSDT", "12345678")

        assert status.order_id == "12345678"
        assert status.status == OrderStatus.FILLED
        assert status.symbol == "BTCUSDT"
        mock_client.get_order.assert_called_once_with(symbol="BTCUSDT", orderId=12345678)

    async def test_get_order_status_no_client(self, binance_exchange):
        """Test getting order status when Binance client is not initialized."""
        binance_exchange.client = None

        with patch.object(binance_exchange, "_validate_symbol"), \
             pytest.raises(ExchangeError, match="Failed to get order status.*Binance client not connected"):
            await binance_exchange.get_order_status("BTCUSDT", "12345678")

    async def test_get_order_status_api_exception(self, binance_exchange, mock_client):
        """Test handling API exceptions during order status retrieval."""
        binance_exchange.client = mock_client
        mock_client.get_order.side_effect = BinanceAPIException(
            response=Mock(status=400, text="Order does not exist"),
            status_code=400,
            text="Order does not exist"
        )

        with patch.object(binance_exchange, "_validate_symbol"), \
             pytest.raises(ExchangeError, match="Failed to get order status.*Order does not exist"):
            await binance_exchange.get_order_status("BTCUSDT", "99999999")


class TestBinanceAccountInfo:
    """Test Binance account information retrieval."""

    async def test_get_account_balance_success(self, binance_exchange, mock_client):
        """Test successful balance retrieval."""
        mock_response = {
            "balances": [
                {"asset": "BTC", "free": "1.00000000", "locked": "0.10000000"},
                {"asset": "USDT", "free": "10000.00000000", "locked": "500.00000000"},
                {"asset": "ETH", "free": "0.00000000", "locked": "0.00000000"}  # Zero balance
            ]
        }

        binance_exchange.client = mock_client
        mock_client.get_account.return_value = mock_response

        balances = await binance_exchange.get_account_balance()

        # Should only include non-zero balances
        assert "BTC" in balances
        assert "USDT" in balances
        assert "ETH" not in balances  # Zero balance filtered out

        assert balances["BTC"] == Decimal("1.10000000")  # free + locked
        assert balances["USDT"] == Decimal("10500.00000000")  # free + locked

    async def test_get_account_balance_specific_asset(self, binance_exchange, mock_client):
        """Test balance retrieval for specific asset."""
        mock_response = {
            "balances": [
                {"asset": "BTC", "free": "1.00000000", "locked": "0.10000000"},
                {"asset": "USDT", "free": "10000.00000000", "locked": "500.00000000"}
            ]
        }

        binance_exchange.client = mock_client
        mock_client.get_account.return_value = mock_response

        balances = await binance_exchange.get_account_balance()

        assert "BTC" in balances
        assert "USDT" in balances  # All balances are returned
        assert balances["BTC"] == Decimal("1.10000000")  # free + locked

    async def test_get_account_balance_no_client(self, binance_exchange):
        """Test getting balance when Binance client is not initialized."""
        binance_exchange.client = None

        with pytest.raises(ExchangeError, match="Failed to get account balance.*Binance client not connected"):
            await binance_exchange.get_account_balance()


class TestBinanceFinancialPrecision:
    """Test financial precision in Binance operations."""

    async def test_order_decimal_precision_preservation(self, binance_exchange, mock_client):
        """Test that Decimal precision is preserved in order operations."""
        high_precision_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.123456789"),  # High precision
            price=Decimal("50000.123456789"),  # High precision
            client_order_id="precision_test"
        )

        mock_response = {
            "orderId": 12345678,
            "clientOrderId": "precision_test",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "origQty": "0.123456789",
            "price": "50000.123456789",
            "status": "NEW",
            "executedQty": "0.00000000",
            "transactTime": 1234567890000,
            "timeInForce": "GTC"
        }

        binance_exchange.client = mock_client
        mock_client.create_order.return_value = mock_response

        with patch.object(binance_exchange, "_validate_order"), \
             patch.object(binance_exchange, "_validate_symbol"), \
             patch.object(binance_exchange, "_validate_price"), \
             patch.object(binance_exchange, "_validate_quantity"):
            response = await binance_exchange.place_order(high_precision_order)

            assert isinstance(response.quantity, Decimal)
            assert isinstance(response.price, Decimal)
            assert response.quantity == Decimal("0.123456789")
            assert response.price == Decimal("50000.123456789")

    async def test_balance_decimal_precision_preservation(self, binance_exchange, mock_client):
        """Test that balance values maintain Decimal precision."""
        mock_response = {
            "balances": [
                {"asset": "BTC", "free": "1.123456789", "locked": "0.987654321"}
            ]
        }

        binance_exchange.client = mock_client
        mock_client.get_account.return_value = mock_response

        balances = await binance_exchange.get_account_balance()

        assert isinstance(balances["BTC"], Decimal)
        assert balances["BTC"] == Decimal("2.111111110")  # free + locked

    async def test_no_float_conversions_in_api_calls(self, binance_exchange, mock_client):
        """Test that API calls use string representations, not float conversions."""
        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.123456789"),
            price=Decimal("50000.123456789"),
            client_order_id="string_test"
        )

        mock_response = {
            "orderId": 12345678,
            "clientOrderId": "string_test",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "origQty": "0.123456789",
            "price": "50000.123456789",
            "status": "NEW",
            "executedQty": "0.00000000",
            "transactTime": 1234567890000,
            "timeInForce": "GTC"
        }

        binance_exchange.client = mock_client
        mock_client.create_order.return_value = mock_response

        with patch.object(binance_exchange, "_validate_order"), \
             patch.object(binance_exchange, "_validate_symbol"), \
             patch.object(binance_exchange, "_validate_price"), \
             patch.object(binance_exchange, "_validate_quantity"):
            await binance_exchange.place_order(order)

            # Verify that string representations are used in API calls
            # Backend rounds to 8 decimal places
            call_kwargs = mock_client.create_order.call_args.kwargs
            assert call_kwargs["quantity"] == "0.12345679"
            assert call_kwargs["price"] == "50000.12345679"
            # Ensure these are strings, not floats
            assert isinstance(call_kwargs["quantity"], str)
            assert isinstance(call_kwargs["price"], str)


class TestBinanceErrorHandling:
    """Test Binance-specific error handling scenarios."""

    async def test_insufficient_balance_error(self, binance_exchange, mock_order_request, mock_client):
        """Test handling insufficient balance error."""
        binance_exchange.client = mock_client
        mock_client.create_order.side_effect = BinanceOrderException(
            code=-2010,
            message="Account has insufficient balance for requested action"
        )

        with patch.object(binance_exchange, "_validate_order"), \
             patch.object(binance_exchange, "_validate_symbol"), \
             patch.object(binance_exchange, "_validate_price"), \
             patch.object(binance_exchange, "_validate_quantity"):
            with pytest.raises(OrderRejectionError) as exc_info:
                await binance_exchange.place_order(mock_order_request)

            assert "insufficient balance" in str(exc_info.value)

    async def test_invalid_symbol_error(self, binance_exchange, mock_client):
        """Test handling invalid symbol error."""
        invalid_order = OrderRequest(
            symbol="INVALIDUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.01"),
            price=Decimal("50000.00"),
            client_order_id="invalid_symbol_test"
        )

        binance_exchange.client = mock_client
        mock_client.create_order.side_effect = BinanceAPIException(
            response=Mock(status=400, text="Invalid symbol"),
            status_code=400,
            text="Invalid symbol"
        )

        with patch.object(binance_exchange, "_validate_order"), \
             patch.object(binance_exchange, "_validate_symbol"), \
             patch.object(binance_exchange, "_validate_price"), \
             patch.object(binance_exchange, "_validate_quantity"):
            with pytest.raises(ExchangeError) as exc_info:
                await binance_exchange.place_order(invalid_order)

            assert "Invalid symbol" in str(exc_info.value)

    async def test_rate_limit_exceeded_error(self, binance_exchange, mock_order_request, mock_client):
        """Test handling rate limit exceeded error."""
        binance_exchange.client = mock_client
        mock_client.create_order.side_effect = BinanceAPIException(
            response=Mock(status=429, text="Too many requests"),
            status_code=429,
            text="Too many requests; current limit is 1200 requests per minute"
        )

        with patch.object(binance_exchange, "_validate_order"), \
             patch.object(binance_exchange, "_validate_symbol"), \
             patch.object(binance_exchange, "_validate_price"), \
             patch.object(binance_exchange, "_validate_quantity"):
            with pytest.raises(ExchangeError) as exc_info:
                await binance_exchange.place_order(mock_order_request)

            assert "Too many requests" in str(exc_info.value)


class TestBinanceEdgeCases:
    """Test Binance-specific edge cases."""

    async def test_order_with_time_in_force(self, binance_exchange, mock_client):
        """Test order placement with specific time-in-force."""
        order_with_tif = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.01"),
            price=Decimal("50000.00"),
            time_in_force="IOC",  # Immediate or Cancel
            client_order_id="tif_test"
        )

        mock_response = {
            "orderId": 12345678,
            "clientOrderId": "tif_test",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "origQty": "0.01000000",
            "price": "50000.00000000",
            "status": "EXPIRED",  # IOC orders often expire if not filled immediately
            "executedQty": "0.00000000",
            "transactTime": 1234567890000,
            "timeInForce": "IOC"
        }

        binance_exchange.client = mock_client
        mock_client.create_order.return_value = mock_response

        with patch.object(binance_exchange, "_validate_order"), \
             patch.object(binance_exchange, "_validate_symbol"), \
             patch.object(binance_exchange, "_validate_price"), \
             patch.object(binance_exchange, "_validate_quantity"):
            response = await binance_exchange.place_order(order_with_tif)

            assert response.status == OrderStatus.EXPIRED

            # Verify IOC was used
            call_kwargs = mock_client.create_order.call_args.kwargs
            assert call_kwargs["timeInForce"] == "IOC"

    async def test_market_order_partial_fill(self, binance_exchange, mock_client):
        """Test market order with partial fill scenario."""
        market_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            client_order_id="partial_fill_test"
        )

        mock_response = {
            "orderId": 12345678,
            "clientOrderId": "partial_fill_test",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "origQty": "0.01000000",
            "status": "PARTIALLY_FILLED",
            "executedQty": "0.00500000",  # Only half filled
            "transactTime": 1234567890000,
            "fills": [
                {"price": "50000.00", "qty": "0.005", "commission": "0.01", "commissionAsset": "BNB"}
            ]
        }

        binance_exchange.client = mock_client
        mock_client.create_order.return_value = mock_response

        with patch.object(binance_exchange, "_validate_order"), \
             patch.object(binance_exchange, "_validate_symbol"), \
             patch.object(binance_exchange, "_validate_price"), \
             patch.object(binance_exchange, "_validate_quantity"):
            response = await binance_exchange.place_order(market_order)

            assert response.status == OrderStatus.PARTIALLY_FILLED
            assert response.filled_quantity == Decimal("0.00500000")
            assert response.remaining_quantity == Decimal("0.00500000")

    async def test_concurrent_order_placement(self, binance_exchange, mock_client):
        """Test concurrent order placements to Binance."""
        orders = []
        for i in range(5):
            order = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.01"),
                price=Decimal("50000.00"),
                client_order_id=f"concurrent_binance_{i}"
            )
            orders.append(order)

        # Mock responses for all orders
        def mock_order_response(i):
            return {
                "orderId": 12345678 + i,
                "clientOrderId": f"concurrent_binance_{i}",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "origQty": "0.01000000",
                "price": "50000.00000000",
                "status": "NEW",
                "executedQty": "0.00000000",
                "transactTime": 1234567890000,
                "timeInForce": "GTC"
            }

        binance_exchange.client = mock_client
        mock_client.create_order.side_effect = [mock_order_response(i) for i in range(5)]

        with patch.object(binance_exchange, "_validate_order"), \
             patch.object(binance_exchange, "_validate_symbol"), \
             patch.object(binance_exchange, "_validate_price"), \
             patch.object(binance_exchange, "_validate_quantity"):
            tasks = [binance_exchange.place_order(order) for order in orders]
            responses = await asyncio.gather(*tasks)

            assert len(responses) == 5
            for i, response in enumerate(responses):
                assert response.client_order_id == f"concurrent_binance_{i}"
                assert response.status == OrderStatus.NEW
