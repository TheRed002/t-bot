"""
Unit tests for Coinbase Exchange Implementation (P-006)

This module contains comprehensive unit tests for the Coinbase exchange implementation,
including tests for all abstract methods from BaseExchange and Coinbase-specific functionality.

CRITICAL: These tests ensure the Coinbase implementation correctly follows the unified
exchange interface and handles all error scenarios properly.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import (
    ExchangeConnectionError,
)

# Import core types and exceptions
from src.core.types import (
    ExchangeInfo,
    MarketData,
    OrderBook,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Ticker,
    Trade,
)

# Import the classes to test
from src.exchanges.coinbase import CoinbaseExchange
from src.exchanges.coinbase_orders import CoinbaseOrderManager
from src.exchanges.coinbase_websocket import CoinbaseWebSocketHandler

# Import test utilities


class TestCoinbaseExchange:
    """Test cases for CoinbaseExchange class."""

    @pytest.fixture
    def coinbase_exchange(self, config):
        """Create CoinbaseExchange instance for testing."""
        return CoinbaseExchange(config, "coinbase")

    @pytest.fixture
    def mock_rest_client(self):
        """Create mock REST client."""
        mock_client = AsyncMock()

        # Mock account data
        mock_client.get_accounts.return_value = [
            {
                "currency": "USD",
                "available_balance": {"value": "1000.00"},
                "hold": {"value": "0.00"},
            },
            {"currency": "BTC", "available_balance": {"value": "0.5"}, "hold": {"value": "0.0"}},
        ]

        # Mock product data
        mock_client.get_product.return_value = {"product_id": "BTC-USD", "status": "online"}

        # Mock ticker data
        mock_client.get_product_ticker.return_value = {
            "product_id": "BTC-USD",
            "price": "50000.00",
            "volume_24h": "1000.0",
            "time": "2024-01-01T00:00:00.000Z",
            "bid": "49999.00",
            "ask": "50001.00",
        }

        # Mock candles data
        mock_client.get_product_candles.return_value = [
            {
                "open": "50000.00",
                "high": "50100.00",
                "low": "49900.00",
                "close": "50050.00",
                "volume": "100.0",
                "time": "2024-01-01T00:00:00.000Z",
            }
        ]

        # Mock order book data
        mock_client.get_product_book.return_value = {
            "bids": [["50000.00", "1.0"], ["49999.00", "2.0"]],
            "asks": [["50001.00", "1.0"], ["50002.00", "2.0"]],
        }

        # Mock trade data
        mock_client.get_product_trades.return_value = [
            {
                "trade_id": "12345",
                "product_id": "BTC-USD",
                "side": "buy",
                "size": "0.1",
                "price": "50000.00",
                "time": "2024-01-01T00:00:00.000Z",
            }
        ]

        # Mock products data
        mock_client.get_products.return_value = [
            {"product_id": "BTC-USD", "status": "online"},
            {"product_id": "ETH-USD", "status": "online"},
        ]

        # Mock products for connection testing
        mock_client.get_products.return_value = [
            {"product_id": "BTC-USD", "status": "online"},
            {"product_id": "ETH-USD", "status": "online"},
        ]

        return mock_client

    @pytest.fixture
    def mock_ws_client(self):
        """Create mock WebSocket client."""
        mock_client = AsyncMock()
        mock_client.open = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client.ticker = AsyncMock()
        mock_client.level2 = AsyncMock()
        mock_client.matches = AsyncMock()
        mock_client.user = AsyncMock()
        return mock_client

    @pytest.mark.asyncio
    async def test_initialization(self, coinbase_exchange, config):
        """Test CoinbaseExchange initialization."""
        assert coinbase_exchange.exchange_name == "coinbase"
        assert coinbase_exchange.api_key == config.exchanges.coinbase_api_key
        assert coinbase_exchange.api_secret == config.exchanges.coinbase_api_secret
        assert coinbase_exchange.sandbox == config.exchanges.coinbase_sandbox
        assert coinbase_exchange.connected is False
        assert coinbase_exchange.status == "initializing"

    @pytest.mark.asyncio
    async def test_connect_success(self, coinbase_exchange, mock_rest_client, mock_ws_client):
        """Test successful connection to Coinbase."""
        with (
            patch("src.exchanges.coinbase.RESTClient", return_value=mock_rest_client),
            patch("src.exchanges.coinbase.WSClient", return_value=mock_ws_client),
        ):
            result = await coinbase_exchange.connect()

            assert result is True
            assert coinbase_exchange.connected is True
            assert coinbase_exchange.status == "connected"
            assert coinbase_exchange.client is not None
            assert coinbase_exchange.ws_client is not None

    @pytest.mark.asyncio
    async def test_connect_failure(self, coinbase_exchange):
        """Test connection failure."""
        with patch("src.exchanges.coinbase.RESTClient", side_effect=Exception("Connection failed")):
            result = await coinbase_exchange.connect()

            assert result is False
            assert coinbase_exchange.connected is False
            assert coinbase_exchange.status == "connection_failed"

    @pytest.mark.asyncio
    async def test_disconnect(self, coinbase_exchange, mock_rest_client, mock_ws_client):
        """Test disconnection from Coinbase."""
        coinbase_exchange.client = mock_rest_client
        coinbase_exchange.ws_client = mock_ws_client
        coinbase_exchange.connected = True

        await coinbase_exchange.disconnect()

        assert coinbase_exchange.connected is False
        assert coinbase_exchange.status == "disconnected"
        mock_ws_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_account_balance(self, coinbase_exchange, mock_rest_client):
        """Test getting account balance."""
        coinbase_exchange.client = mock_rest_client

        balances = await coinbase_exchange.get_account_balance()

        assert isinstance(balances, dict)
        assert "USD" in balances
        assert "BTC" in balances
        assert balances["USD"] == Decimal("1000.00")
        assert balances["BTC"] == Decimal("0.5")
        mock_rest_client.get_accounts.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_account_balance_not_connected(self, coinbase_exchange):
        """Test getting account balance when not connected."""
        coinbase_exchange.client = None

        with pytest.raises(ExchangeConnectionError):
            await coinbase_exchange.get_account_balance()

    @pytest.mark.asyncio
    async def test_place_order_success(self, coinbase_exchange, mock_rest_client):
        """Test successful order placement."""
        coinbase_exchange.client = mock_rest_client

        # Mock order creation response
        mock_rest_client.create_order.return_value = {
            "order_id": "test_order_123",
            "product_id": "BTC-USD",
            "side": "BUY",
            "order_configuration": {"market_market_ioc": {"quote_size": "100.00"}},
            "status": "OPEN",
            "created_time": "2024-01-01T00:00:00.000Z",
            "filled_size": "0.0",
        }

        order_request = OrderRequest(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100.00"),
        )

        response = await coinbase_exchange.place_order(order_request)

        assert isinstance(response, OrderResponse)
        assert response.id == "test_order_123"
        assert response.symbol == "BTC-USD"
        assert response.side == OrderSide.BUY
        assert response.order_type == OrderType.MARKET
        mock_rest_client.create_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_order_not_connected(self, coinbase_exchange):
        """Test order placement when not connected."""
        coinbase_exchange.client = None

        order_request = OrderRequest(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100.00"),
        )

        with pytest.raises(ExchangeConnectionError):
            await coinbase_exchange.place_order(order_request)

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, coinbase_exchange, mock_rest_client):
        """Test successful order cancellation."""
        coinbase_exchange.client = mock_rest_client
        mock_rest_client.cancel_orders.return_value = {"results": [{"order_id": "test_order_123"}]}

        result = await coinbase_exchange.cancel_order("test_order_123")

        assert result is True
        mock_rest_client.cancel_orders.assert_called_once_with(["test_order_123"])

    @pytest.mark.asyncio
    async def test_cancel_order_failure(self, coinbase_exchange, mock_rest_client):
        """Test order cancellation failure."""
        coinbase_exchange.client = mock_rest_client
        mock_rest_client.cancel_orders.side_effect = Exception("Cancellation failed")

        result = await coinbase_exchange.cancel_order("test_order_123")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_order_status(self, coinbase_exchange, mock_rest_client):
        """Test getting order status."""
        coinbase_exchange.client = mock_rest_client
        mock_rest_client.get_order.return_value = {"order_id": "test_order_123", "status": "FILLED"}

        status = await coinbase_exchange.get_order_status("test_order_123")

        assert status == OrderStatus.FILLED
        mock_rest_client.get_order.assert_called_once_with("test_order_123")

    @pytest.mark.asyncio
    async def test_get_market_data(self, coinbase_exchange, mock_rest_client):
        """Test getting market data."""
        coinbase_exchange.client = mock_rest_client

        market_data = await coinbase_exchange.get_market_data("BTC-USD")

        assert isinstance(market_data, MarketData)
        assert market_data.symbol == "BTC-USD"
        assert market_data.price == Decimal("50000.00")
        assert market_data.volume == Decimal("1000.0")
        assert market_data.bid == Decimal("49999.00")
        assert market_data.ask == Decimal("50001.00")

    @pytest.mark.asyncio
    async def test_get_order_book(self, coinbase_exchange, mock_rest_client):
        """Test getting order book."""
        coinbase_exchange.client = mock_rest_client

        order_book = await coinbase_exchange.get_order_book("BTC-USD")

        assert isinstance(order_book, OrderBook)
        assert order_book.symbol == "BTC-USD"
        assert len(order_book.bids) == 2
        assert len(order_book.asks) == 2
        assert order_book.bids[0][0] == Decimal("50000.00")
        assert order_book.bids[0][1] == Decimal("1.0")

    @pytest.mark.asyncio
    async def test_get_trade_history(self, coinbase_exchange, mock_rest_client):
        """Test getting trade history."""
        coinbase_exchange.client = mock_rest_client

        trades = await coinbase_exchange.get_trade_history("BTC-USD")

        assert isinstance(trades, list)
        assert len(trades) == 1
        assert isinstance(trades[0], Trade)
        assert trades[0].id == "12345"
        assert trades[0].symbol == "BTC-USD"
        assert trades[0].side == OrderSide.BUY
        assert trades[0].amount == Decimal("0.1")
        assert trades[0].price == Decimal("50000.00")

    @pytest.mark.asyncio
    async def test_get_exchange_info(self, coinbase_exchange, mock_rest_client):
        """Test getting exchange information."""
        coinbase_exchange.client = mock_rest_client

        exchange_info = await coinbase_exchange.get_exchange_info()

        assert isinstance(exchange_info, ExchangeInfo)
        assert exchange_info.name == "coinbase"
        assert "BTC-USD" in exchange_info.supported_symbols
        assert "ETH-USD" in exchange_info.supported_symbols
        assert "spot_trading" in exchange_info.features

    @pytest.mark.asyncio
    async def test_get_ticker(self, coinbase_exchange, mock_rest_client):
        """Test getting ticker information."""
        coinbase_exchange.client = mock_rest_client

        ticker = await coinbase_exchange.get_ticker("BTC-USD")

        assert isinstance(ticker, Ticker)
        assert ticker.symbol == "BTC-USD"
        assert ticker.last_price == Decimal("50000.00")
        assert ticker.bid == Decimal("49999.00")
        assert ticker.ask == Decimal("50001.00")
        assert ticker.volume_24h == Decimal("1000.0")

    @pytest.mark.asyncio
    async def test_health_check_success(self, coinbase_exchange, mock_rest_client):
        """Test successful health check."""
        coinbase_exchange.client = mock_rest_client

        result = await coinbase_exchange.health_check()

        assert result is True
        mock_rest_client.get_products.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failure(self, coinbase_exchange):
        """Test health check failure."""
        coinbase_exchange.client = None

        result = await coinbase_exchange.health_check()

        assert result is False

    def test_get_rate_limits(self, coinbase_exchange):
        """Test getting rate limits."""
        rate_limits = coinbase_exchange.get_rate_limits()

        assert isinstance(rate_limits, dict)
        assert "requests_per_minute" in rate_limits
        assert "orders_per_second" in rate_limits
        assert "websocket_connections" in rate_limits
        assert rate_limits["requests_per_minute"] == 600
        assert rate_limits["orders_per_second"] == 15

    def test_convert_timeframe_to_granularity(self, coinbase_exchange):
        """Test timeframe to granularity conversion."""
        assert coinbase_exchange._convert_timeframe_to_granularity("1m") == "ONE_MINUTE"
        assert coinbase_exchange._convert_timeframe_to_granularity("5m") == "FIVE_MINUTE"
        assert coinbase_exchange._convert_timeframe_to_granularity("1h") == "ONE_HOUR"
        assert coinbase_exchange._convert_timeframe_to_granularity("1d") == "ONE_DAY"
        assert coinbase_exchange._convert_timeframe_to_granularity("unknown") == "ONE_MINUTE"

    def test_convert_coinbase_status_to_order_status(self, coinbase_exchange):
        """Test Coinbase status to OrderStatus conversion."""
        assert (
            coinbase_exchange._convert_coinbase_status_to_order_status("OPEN")
            == OrderStatus.PENDING
        )
        assert (
            coinbase_exchange._convert_coinbase_status_to_order_status("FILLED")
            == OrderStatus.FILLED
        )
        assert (
            coinbase_exchange._convert_coinbase_status_to_order_status("CANCELLED")
            == OrderStatus.CANCELLED
        )
        assert (
            coinbase_exchange._convert_coinbase_status_to_order_status("EXPIRED")
            == OrderStatus.EXPIRED
        )
        assert (
            coinbase_exchange._convert_coinbase_status_to_order_status("REJECTED")
            == OrderStatus.REJECTED
        )
        assert (
            coinbase_exchange._convert_coinbase_status_to_order_status("PARTIALLY_FILLED")
            == OrderStatus.PARTIALLY_FILLED
        )
        assert (
            coinbase_exchange._convert_coinbase_status_to_order_status("UNKNOWN_STATUS")
            == OrderStatus.UNKNOWN
        )


class TestCoinbaseWebSocketHandler:
    """Test cases for CoinbaseWebSocketHandler class."""

    @pytest.fixture
    def ws_handler(self, config):
        """Create CoinbaseWebSocketHandler instance for testing."""
        return CoinbaseWebSocketHandler(config, "coinbase")

    @pytest.fixture
    def mock_ws_client(self):
        """Create mock WebSocket client."""
        mock_client = AsyncMock()
        mock_client.open = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client.ticker = AsyncMock()
        mock_client.level2 = AsyncMock()
        mock_client.matches = AsyncMock()
        mock_client.user = AsyncMock()
        mock_client.ticker_unsubscribe = AsyncMock()
        mock_client.level2_unsubscribe = AsyncMock()
        mock_client.matches_unsubscribe = AsyncMock()
        mock_client.user_unsubscribe = AsyncMock()
        return mock_client

    @pytest.mark.asyncio
    async def test_initialization(self, ws_handler, config):
        """Test CoinbaseWebSocketHandler initialization."""
        assert ws_handler.exchange_name == "coinbase"
        assert ws_handler.api_key == config.exchanges.coinbase_api_key
        assert ws_handler.api_secret == config.exchanges.coinbase_api_secret
        assert ws_handler.sandbox == config.exchanges.coinbase_sandbox
        assert ws_handler.connected is False
        assert len(ws_handler.active_streams) == 0

    @pytest.mark.asyncio
    async def test_connect_success(self, ws_handler, mock_ws_client):
        """Test successful WebSocket connection."""
        with patch("src.exchanges.coinbase_websocket.WSClient", return_value=mock_ws_client):
            result = await ws_handler.connect()

            assert result is True
            assert ws_handler.connected is True
            assert ws_handler.ws_client is not None
            mock_ws_client.open.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure(self, ws_handler):
        """Test WebSocket connection failure."""
        with patch(
            "src.exchanges.coinbase_websocket.WSClient", side_effect=Exception("Connection failed")
        ):
            result = await ws_handler.connect()

            assert result is False
            assert ws_handler.connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, ws_handler, mock_ws_client):
        """Test WebSocket disconnection."""
        ws_handler.ws_client = mock_ws_client
        ws_handler.connected = True

        await ws_handler.disconnect()

        assert ws_handler.connected is False
        mock_ws_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_subscribe_to_ticker(self, ws_handler, mock_ws_client):
        """Test subscribing to ticker stream."""
        ws_handler.ws_client = mock_ws_client
        ws_handler.connected = True

        callback = AsyncMock()
        await ws_handler.subscribe_to_ticker("BTC-USD", callback)

        assert "ticker_BTC-USD" in ws_handler.active_streams
        assert "BTC-USD" in ws_handler.callbacks
        mock_ws_client.ticker.assert_called_once_with("BTC-USD", callback)

    @pytest.mark.asyncio
    async def test_subscribe_to_orderbook(self, ws_handler, mock_ws_client):
        """Test subscribing to order book stream."""
        ws_handler.ws_client = mock_ws_client
        ws_handler.connected = True

        callback = AsyncMock()
        await ws_handler.subscribe_to_orderbook("BTC-USD", callback)

        assert "orderbook_BTC-USD" in ws_handler.active_streams
        mock_ws_client.level2.assert_called_once_with("BTC-USD", callback)

    @pytest.mark.asyncio
    async def test_subscribe_to_trades(self, ws_handler, mock_ws_client):
        """Test subscribing to trades stream."""
        ws_handler.ws_client = mock_ws_client
        ws_handler.connected = True

        callback = AsyncMock()
        await ws_handler.subscribe_to_trades("BTC-USD", callback)

        assert "trades_BTC-USD" in ws_handler.active_streams
        mock_ws_client.matches.assert_called_once_with("BTC-USD", callback)

    @pytest.mark.asyncio
    async def test_subscribe_to_user_data(self, ws_handler, mock_ws_client):
        """Test subscribing to user data stream."""
        ws_handler.ws_client = mock_ws_client
        ws_handler.connected = True

        callback = AsyncMock()
        await ws_handler.subscribe_to_user_data(callback)

        assert "user_data" in ws_handler.active_streams
        mock_ws_client.user.assert_called_once_with(callback)

    @pytest.mark.asyncio
    async def test_unsubscribe_from_stream(self, ws_handler, mock_ws_client):
        """Test unsubscribing from a stream."""
        ws_handler.ws_client = mock_ws_client
        ws_handler.active_streams["ticker_BTC-USD"] = {
            "type": "ticker",
            "symbol": "BTC-USD",
            "callback": AsyncMock(),
        }

        result = await ws_handler.unsubscribe_from_stream("ticker_BTC-USD")

        assert result is True
        assert "ticker_BTC-USD" not in ws_handler.active_streams
        mock_ws_client.ticker_unsubscribe.assert_called_once_with("BTC-USD")

    @pytest.mark.asyncio
    async def test_unsubscribe_all(self, ws_handler, mock_ws_client):
        """Test unsubscribing from all streams."""
        ws_handler.ws_client = mock_ws_client
        ws_handler.active_streams["ticker_BTC-USD"] = {
            "type": "ticker",
            "symbol": "BTC-USD",
            "callback": AsyncMock(),
        }
        ws_handler.active_streams["orderbook_BTC-USD"] = {
            "type": "orderbook",
            "symbol": "BTC-USD",
            "callback": AsyncMock(),
        }

        await ws_handler.unsubscribe_all()

        assert len(ws_handler.active_streams) == 0
        assert len(ws_handler.callbacks) == 0

    @pytest.mark.asyncio
    async def test_handle_ticker_message(self, ws_handler):
        """Test handling ticker message."""
        callback = AsyncMock()
        ws_handler.callbacks["BTC-USD"] = [callback]

        message = {
            "product_id": "BTC-USD",
            "price": "50000.00",
            "volume_24h": "1000.0",
            "time": "2024-01-01T00:00:00.000Z",
            "bid": "49999.00",
            "ask": "50001.00",
            "price_change_24h": "100.00",
        }

        await ws_handler.handle_ticker_message(message)

        callback.assert_called_once()
        ticker = callback.call_args[0][0]
        assert isinstance(ticker, Ticker)
        assert ticker.symbol == "BTC-USD"
        assert ticker.last_price == Decimal("50000.00")

    @pytest.mark.asyncio
    async def test_handle_orderbook_message(self, ws_handler):
        """Test handling order book message."""
        callback = AsyncMock()
        ws_handler.callbacks["BTC-USD"] = [callback]

        message = {
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"], ["49999.00", "2.0"]],
            "asks": [["50001.00", "1.0"], ["50002.00", "2.0"]],
        }

        await ws_handler.handle_orderbook_message(message)

        callback.assert_called_once()
        order_book = callback.call_args[0][0]
        assert isinstance(order_book, OrderBook)
        assert order_book.symbol == "BTC-USD"
        assert len(order_book.bids) == 2
        assert len(order_book.asks) == 2

    @pytest.mark.asyncio
    async def test_handle_trade_message(self, ws_handler):
        """Test handling trade message."""
        callback = AsyncMock()
        ws_handler.callbacks["BTC-USD"] = [callback]

        message = {
            "trade_id": "12345",
            "product_id": "BTC-USD",
            "side": "buy",
            "size": "0.1",
            "price": "50000.00",
            "time": "2024-01-01T00:00:00.000Z",
        }

        await ws_handler.handle_trade_message(message)

        callback.assert_called_once()
        trade = callback.call_args[0][0]
        assert isinstance(trade, Trade)
        assert trade.id == "12345"
        assert trade.symbol == "BTC-USD"
        assert trade.side == OrderSide.BUY
        assert trade.amount == Decimal("0.1")
        assert trade.price == Decimal("50000.00")

    @pytest.mark.asyncio
    async def test_health_check_success(self, ws_handler):
        """Test successful health check."""
        ws_handler.connected = True
        ws_handler.ws_client = MagicMock()  # Add mock ws_client
        ws_handler.last_heartbeat = datetime.now(timezone.utc)

        result = await ws_handler.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, ws_handler):
        """Test health check failure."""
        ws_handler.connected = False

        result = await ws_handler.health_check()

        assert result is False

    def test_is_connected(self, ws_handler):
        """Test connection status check."""
        ws_handler.connected = True
        ws_handler.ws_client = MagicMock()

        assert ws_handler.is_connected() is True

        ws_handler.connected = False
        assert ws_handler.is_connected() is False

    def test_get_active_streams(self, ws_handler):
        """Test getting active streams."""
        ws_handler.active_streams["test_stream"] = {"type": "test"}

        streams = ws_handler.get_active_streams()

        assert isinstance(streams, dict)
        assert "test_stream" in streams


class TestCoinbaseOrderManager:
    """Test cases for CoinbaseOrderManager class."""

    @pytest.fixture
    def order_manager(self, config):
        """Create CoinbaseOrderManager instance for testing."""
        return CoinbaseOrderManager(config, "coinbase")

    @pytest.fixture
    def mock_rest_client(self):
        """Create mock REST client."""
        mock_client = AsyncMock()

        # Mock order creation response
        mock_client.create_order.return_value = {
            "order_id": "test_order_123",
            "product_id": "BTC-USD",
            "side": "BUY",
            "order_configuration": {"market_market_ioc": {"quote_size": "100.00"}},
            "status": "OPEN",
            "created_time": "2024-01-01T00:00:00.000Z",
            "filled_size": "0.0",
        }

        # Mock order details
        mock_client.get_order.return_value = {
            "order_id": "test_order_123",
            "product_id": "BTC-USD",
            "side": "BUY",
            "order_configuration": {"market_market_ioc": {"quote_size": "100.00"}},
            "status": "FILLED",
            "created_time": "2024-01-01T00:00:00.000Z",
            "filled_size": "100.0",
        }

        # Mock product data
        mock_client.get_product.return_value = {"product_id": "BTC-USD", "status": "online"}

        # Mock products for connection testing
        mock_client.get_products.return_value = [
            {"product_id": "BTC-USD", "status": "online"},
            {"product_id": "ETH-USD", "status": "online"},
        ]

        return mock_client

    @pytest.mark.asyncio
    async def test_initialization(self, order_manager, config):
        """Test CoinbaseOrderManager initialization."""
        assert order_manager.exchange_name == "coinbase"
        assert order_manager.api_key == config.exchanges.coinbase_api_key
        assert order_manager.api_secret == config.exchanges.coinbase_api_secret
        assert order_manager.sandbox == config.exchanges.coinbase_sandbox
        assert len(order_manager.pending_orders) == 0
        assert len(order_manager.order_history) == 0

    @pytest.mark.asyncio
    async def test_initialize_success(self, order_manager, mock_rest_client):
        """Test successful initialization."""
        with patch("src.exchanges.coinbase_orders.RESTClient", return_value=mock_rest_client):
            result = await order_manager.initialize()

            assert result is True
            assert order_manager.client is not None
            mock_rest_client.get_products.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_failure(self, order_manager):
        """Test initialization failure."""
        with patch(
            "src.exchanges.coinbase_orders.RESTClient", side_effect=Exception("Init failed")
        ):
            result = await order_manager.initialize()

            assert result is False

    @pytest.mark.asyncio
    async def test_place_order_success(self, order_manager, mock_rest_client):
        """Test successful order placement."""
        order_manager.client = mock_rest_client

        order_request = OrderRequest(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100.00"),
        )

        response = await order_manager.place_order(order_request)

        assert isinstance(response, OrderResponse)
        assert response.id == "test_order_123"
        assert response.symbol == "BTC-USD"
        assert response.side == OrderSide.BUY
        assert response.order_type == OrderType.MARKET
        assert "test_order_123" in order_manager.pending_orders
        assert len(order_manager.order_history) == 1

    @pytest.mark.asyncio
    async def test_place_order_not_connected(self, order_manager):
        """Test order placement when not connected."""
        order_manager.client = None

        order_request = OrderRequest(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100.00"),
        )

        with pytest.raises(ExchangeConnectionError):
            await order_manager.place_order(order_request)

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, order_manager, mock_rest_client):
        """Test successful order cancellation."""
        order_manager.client = mock_rest_client
        mock_rest_client.cancel_orders.return_value = {"results": [{"order_id": "test_order_123"}]}

        result = await order_manager.cancel_order("test_order_123")

        assert result is True
        mock_rest_client.cancel_orders.assert_called_once_with(["test_order_123"])

    @pytest.mark.asyncio
    async def test_cancel_order_failure(self, order_manager, mock_rest_client):
        """Test order cancellation failure."""
        order_manager.client = mock_rest_client
        mock_rest_client.cancel_orders.side_effect = Exception("Cancellation failed")

        result = await order_manager.cancel_order("test_order_123")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_order_status(self, order_manager, mock_rest_client):
        """Test getting order status."""
        order_manager.client = mock_rest_client

        status = await order_manager.get_order_status("test_order_123")

        assert status == OrderStatus.FILLED
        mock_rest_client.get_order.assert_called_once_with("test_order_123")

    @pytest.mark.asyncio
    async def test_get_order_details(self, order_manager, mock_rest_client):
        """Test getting order details."""
        order_manager.client = mock_rest_client

        order_response = await order_manager.get_order_details("test_order_123")

        assert isinstance(order_response, OrderResponse)
        assert order_response.id == "test_order_123"
        assert order_response.symbol == "BTC-USD"
        assert order_response.side == OrderSide.BUY
        assert order_response.status == "FILLED"

    @pytest.mark.asyncio
    async def test_calculate_fees(self, order_manager, mock_rest_client):
        """Test fee calculation."""
        order_manager.client = mock_rest_client

        order_request = OrderRequest(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100.00"),
        )

        fees = await order_manager.calculate_fees(order_request)

        assert isinstance(fees, dict)
        assert "fee_rate" in fees
        assert "fee_amount" in fees
        assert "fee_currency" in fees
        assert fees["fee_currency"] == "USD"
        assert fees["fee_rate"] == Decimal("0.006")  # Market order fee rate
        assert fees["fee_amount"] == Decimal("0.6")  # 100 * 0.006

    def test_get_total_fees(self, order_manager):
        """Test getting total fees."""
        order_manager.total_fees["USD"] = Decimal("10.50")
        order_manager.total_fees["BTC"] = Decimal("0.001")

        total_fees = order_manager.get_total_fees()

        assert isinstance(total_fees, dict)
        assert total_fees["USD"] == Decimal("10.50")
        assert total_fees["BTC"] == Decimal("0.001")

    def test_get_order_statistics(self, order_manager):
        """Test getting order statistics."""
        # Add some mock orders to history
        order_manager.order_history = [
            OrderResponse(
                id="1",
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                price=None,
                filled_quantity=Decimal("1.0"),
                status="FILLED",
                timestamp=datetime.now(timezone.utc),
                client_order_id="client_1",
            ),
            OrderResponse(
                id="2",
                symbol="BTC-USD",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.5"),
                price=Decimal("50000"),
                filled_quantity=Decimal("0.5"),
                status="CANCELLED",
                timestamp=datetime.now(timezone.utc),
                client_order_id="client_2",
            ),
        ]

        stats = order_manager.get_order_statistics()

        assert isinstance(stats, dict)
        assert stats["total_orders"] == 2
        assert stats["filled_orders"] == 1
        assert stats["cancelled_orders"] == 1
        assert stats["fill_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_validate_order_success(self, order_manager):
        """Test successful order validation."""
        order_request = OrderRequest(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100.00"),
        )

        result = await order_manager._validate_order(order_request)

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_order_failure(self, order_manager):
        """Test order validation failure."""
        # Test missing required fields
        order_request = OrderRequest(
            symbol="", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("100.00")
        )

        result = await order_manager._validate_order(order_request)

        assert result is False

    def test_convert_order_to_coinbase(self, order_manager):
        """Test order conversion to Coinbase format."""
        order_request = OrderRequest(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100.00"),
            client_order_id="test_client_id",
        )

        coinbase_order = order_manager._convert_order_to_coinbase(order_request)

        assert coinbase_order["product_id"] == "BTC-USD"
        assert coinbase_order["side"] == "BUY"
        assert "market_market_ioc" in coinbase_order["order_configuration"]
        assert coinbase_order["client_order_id"] == "test_client_id"

    def test_convert_coinbase_order_to_response(self, order_manager):
        """Test Coinbase order response conversion."""
        coinbase_response = {
            "order_id": "test_order_123",
            "product_id": "BTC-USD",
            "side": "BUY",
            "order_configuration": {"market_market_ioc": {"quote_size": "100.00"}},
            "status": "FILLED",
            "created_time": "2024-01-01T00:00:00.000Z",
            "filled_size": "100.0",
        }

        order_response = order_manager._convert_coinbase_order_to_response(coinbase_response)

        assert isinstance(order_response, OrderResponse)
        assert order_response.id == "test_order_123"
        assert order_response.symbol == "BTC-USD"
        assert order_response.side == OrderSide.BUY
        assert order_response.order_type == OrderType.MARKET
        assert order_response.status == "FILLED"

    def test_convert_coinbase_status_to_order_status(self, order_manager):
        """Test Coinbase status to OrderStatus conversion."""
        assert order_manager._convert_coinbase_status_to_order_status("OPEN") == OrderStatus.PENDING
        assert (
            order_manager._convert_coinbase_status_to_order_status("FILLED") == OrderStatus.FILLED
        )
        assert (
            order_manager._convert_coinbase_status_to_order_status("CANCELLED")
            == OrderStatus.CANCELLED
        )
        assert (
            order_manager._convert_coinbase_status_to_order_status("EXPIRED") == OrderStatus.EXPIRED
        )
        assert (
            order_manager._convert_coinbase_status_to_order_status("REJECTED")
            == OrderStatus.REJECTED
        )
        assert (
            order_manager._convert_coinbase_status_to_order_status("PARTIALLY_FILLED")
            == OrderStatus.PARTIALLY_FILLED
        )
        assert (
            order_manager._convert_coinbase_status_to_order_status("UNKNOWN_STATUS")
            == OrderStatus.UNKNOWN
        )
