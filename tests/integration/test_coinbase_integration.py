"""
Integration tests for Coinbase Exchange Implementation (P-006)

This module contains integration tests for the Coinbase exchange implementation,
testing the complete workflow from connection to order execution.

CRITICAL: These tests ensure the Coinbase implementation works correctly
in a real environment with proper error handling and recovery.
"""

import warnings
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import Config
from src.core.exceptions import (
    ExchangeConnectionError,
    ExchangeError,
)
from src.core.types import (
    MarketData,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Ticker,
)
from src.exchanges.coinbase import CoinbaseExchange
from src.exchanges.coinbase_orders import CoinbaseOrderManager
from src.exchanges.coinbase_websocket import CoinbaseWebSocketHandler
from src.exchanges.factory import ExchangeFactory

# Suppress unawaited coroutine RuntimeWarnings in this module-specific context
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Import core types and exceptions

# Import the classes to test

# Import test utilities


class TestCoinbaseIntegration:
    """Integration tests for Coinbase exchange."""

    @pytest.fixture
    def config(self):
        cfg = Config()
        cfg.exchanges.coinbase_sandbox = True
        cfg.exchanges.binance_testnet = True
        cfg.exchanges.okx_sandbox = True
        return cfg

    @pytest.fixture
    def exchange_factory(self, config):
        """Create exchange factory for testing."""
        return ExchangeFactory(config)

    @pytest.fixture
    def coinbase_exchange(self, config):
        """Create CoinbaseExchange instance for testing."""
        return CoinbaseExchange(config, "coinbase")

    @pytest.fixture
    def ws_handler(self, config):
        """Create CoinbaseWebSocketHandler instance for testing."""
        return CoinbaseWebSocketHandler(config, "coinbase")

    @pytest.fixture
    def order_manager(self, config):
        """Create CoinbaseOrderManager instance for testing."""
        return CoinbaseOrderManager(config, "coinbase")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_exchange_factory_registration(self, exchange_factory):
        """Test that Coinbase exchange is properly registered with factory."""
        from unittest.mock import AsyncMock, patch

        from src.exchanges import register_exchanges

        # Register exchanges
        register_exchanges(exchange_factory)

        # Check that Coinbase is registered
        supported_exchanges = exchange_factory.get_supported_exchanges()
        assert "coinbase" in supported_exchanges

        # Mock the Coinbase clients to avoid real API calls
        mock_rest_client = AsyncMock()
        mock_ws_client = AsyncMock()

        # Mock the get_products method for connection test
        mock_rest_client.get_products.return_value = [{"product_id": "BTC-USD", "status": "online"}]

        # Mock WebSocket methods
        mock_ws_client.open = AsyncMock()

        with (
            patch("src.exchanges.coinbase.RESTClient", return_value=mock_rest_client),
            patch("src.exchanges.coinbase.WSClient", return_value=mock_ws_client),
        ):
            # Check that we can create a Coinbase exchange instance
            exchange = await exchange_factory.create_exchange("coinbase")
            assert exchange is not None
            assert exchange.exchange_name == "coinbase"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_complete_workflow(self, coinbase_exchange, ws_handler, order_manager):
        """Test complete workflow from connection to order execution."""
        # Mock the Coinbase clients
        mock_rest_client = AsyncMock()
        mock_ws_client = AsyncMock()

        # Mock account data
        mock_rest_client.get_accounts.return_value = [
            {
                "currency": "USD",
                "available_balance": {"value": "10000.00"},
                "hold": {"value": "0.00"},
            }
        ]

        # Mock product data
        mock_rest_client.get_product.return_value = {"product_id": "BTC-USD", "status": "online"}

        # Mock ticker data
        mock_rest_client.get_product_ticker.return_value = {
            "product_id": "BTC-USD",
            "price": "50000.00",
            "volume_24h": "1000.0",
            "time": "2024-01-01T00:00:00.000Z",
            "bid": "49999.00",
            "ask": "50001.00",
        }

        # Mock candles data
        mock_rest_client.get_product_candles.return_value = [
            {
                "open": "50000.00",
                "high": "50100.00",
                "low": "49900.00",
                "close": "50050.00",
                "volume": "100.0",
                "time": "2024-01-01T00:00:00.000Z",
            }
        ]

        # Mock order creation
        mock_rest_client.create_order.return_value = {
            "order_id": "test_order_123",
            "product_id": "BTC-USD",
            "side": "BUY",
            "order_configuration": {"market_market_ioc": {"quote_size": "100.00"}},
            "status": "OPEN",
            "created_time": "2024-01-01T00:00:00.000Z",
            "filled_size": "0.0",
        }

        # Mock order status
        mock_rest_client.get_order.return_value = {
            "order_id": "test_order_123",
            "product_id": "BTC-USD",
            "side": "BUY",
            "order_configuration": {"market_market_ioc": {"quote_size": "100.00"}},
            "status": "FILLED",
            "created_time": "2024-01-01T00:00:00.000Z",
            "filled_size": "100.0",
        }

        # Mock time
        mock_rest_client.get_time.return_value = {"iso": "2024-01-01T00:00:00.000Z"}

        # Mock WebSocket methods
        mock_ws_client.open = AsyncMock()
        mock_ws_client.close = AsyncMock()
        mock_ws_client.ticker = AsyncMock()
        mock_ws_client.level2 = AsyncMock()
        mock_ws_client.matches = AsyncMock()
        mock_ws_client.user = AsyncMock()

        with (
            patch("src.exchanges.coinbase.RESTClient", return_value=mock_rest_client),
            patch("src.exchanges.coinbase.WSClient", return_value=mock_ws_client),
            patch("src.exchanges.coinbase_websocket.WSClient", return_value=mock_ws_client),
            patch("src.exchanges.coinbase_orders.RESTClient", return_value=mock_rest_client),
        ):
            # 1. Test exchange connection
            result = await coinbase_exchange.connect()
            assert result is True
            assert coinbase_exchange.connected is True

            # 2. Test WebSocket connection
            ws_result = await ws_handler.connect()
            assert ws_result is True
            assert ws_handler.connected is True

            # 3. Test order manager initialization
            om_result = await order_manager.initialize()
            assert om_result is True

            # 4. Test getting account balance
            balances = await coinbase_exchange.get_account_balance()
            assert isinstance(balances, dict)
            assert "USD" in balances
            assert balances["USD"] == Decimal("10000.00")

            # 5. Test getting market data
            market_data = await coinbase_exchange.get_market_data("BTC-USD")
            assert isinstance(market_data, MarketData)
            assert market_data.symbol == "BTC-USD"
            assert market_data.price == Decimal("50000.00")

            # 6. Test placing an order
            order_request = OrderRequest(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100.00"),
            )

            order_response = await coinbase_exchange.place_order(order_request)
            assert isinstance(order_response, OrderResponse)
            assert order_response.id == "test_order_123"
            assert order_response.symbol == "BTC-USD"
            assert order_response.side == OrderSide.BUY

            # 7. Test getting order status
            status = await coinbase_exchange.get_order_status("test_order_123")
            assert status == OrderStatus.FILLED

            # 8. Test WebSocket subscription
            callback = AsyncMock()
            await ws_handler.subscribe_to_ticker("BTC-USD", callback)
            assert "ticker_BTC-USD" in ws_handler.active_streams

            # 9. Test order cancellation
            cancel_result = await coinbase_exchange.cancel_order("test_order_123")
            assert cancel_result is True

            # 10. Test disconnection
            await coinbase_exchange.disconnect()
            assert coinbase_exchange.connected is False

            await ws_handler.disconnect()
            assert ws_handler.connected is False

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_error_handling_and_recovery(self, coinbase_exchange):
        """Test error handling and recovery scenarios."""
        # Test connection failure
        with patch("src.exchanges.coinbase.RESTClient", side_effect=Exception("Connection failed")):
            result = await coinbase_exchange.connect()
            assert result is False
            assert coinbase_exchange.connected is False
            assert coinbase_exchange.status == "connection_failed"

        # Test order placement without connection
        order_request = OrderRequest(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100.00"),
        )

        with pytest.raises(ExchangeConnectionError):
            await coinbase_exchange.place_order(order_request)

        # Test getting balance without connection
        with pytest.raises(ExchangeConnectionError):
            await coinbase_exchange.get_account_balance()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_rate_limiting_integration(self, coinbase_exchange):
        """Test rate limiting integration."""
        # Mock the rate limiter
        mock_rate_limiter = MagicMock()
        mock_rate_limiter.acquire = AsyncMock(return_value=True)
        mock_rate_limiter.release = AsyncMock()

        coinbase_exchange.rate_limiter = mock_rate_limiter

        # Mock REST client
        mock_rest_client = AsyncMock()
        mock_rest_client.get_accounts.return_value = []
        mock_rest_client.get_time.return_value = {"iso": "2024-01-01T00:00:00.000Z"}

        coinbase_exchange.client = mock_rest_client

        # Test that rate limiter is used
        await coinbase_exchange.get_account_balance()

        # Verify rate limiter was called
        mock_rate_limiter.acquire.assert_called()
        # Note: Rate limiter doesn't use release() method - it uses token
        # bucket approach

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_websocket_message_handling(self, ws_handler):
        """Test WebSocket message handling."""
        # Mock WebSocket client
        mock_ws_client = AsyncMock()
        mock_ws_client.open = AsyncMock()
        mock_ws_client.ticker = AsyncMock()

        ws_handler.ws_client = mock_ws_client
        ws_handler.connected = True

        # Test ticker message handling
        callback = AsyncMock()
        ws_handler.callbacks["BTC-USD"] = [callback]

        ticker_message = {
            "product_id": "BTC-USD",
            "price": "50000.00",
            "volume_24h": "1000.0",
            "time": "2024-01-01T00:00:00.000Z",
            "bid": "49999.00",
            "ask": "50001.00",
            "price_change_24h": "100.00",
        }

        await ws_handler.handle_ticker_message(ticker_message)

        # Verify callback was called
        callback.assert_called_once()
        ticker = callback.call_args[0][0]
        assert isinstance(ticker, Ticker)
        assert ticker.symbol == "BTC-USD"
        assert ticker.last_price == Decimal("50000.00")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_order_manager_integration(self, order_manager):
        """Test order manager integration."""
        # Mock REST client
        mock_rest_client = AsyncMock()
        mock_rest_client.get_time.return_value = {"iso": "2024-01-01T00:00:00.000Z"}
        mock_rest_client.get_product.return_value = {"product_id": "BTC-USD", "status": "online"}
        mock_rest_client.create_order.return_value = {
            "order_id": "test_order_123",
            "product_id": "BTC-USD",
            "side": "BUY",
            "order_configuration": {"market_market_ioc": {"quote_size": "100.00"}},
            "status": "OPEN",
            "created_time": "2024-01-01T00:00:00.000Z",
            "filled_size": "0.0",
        }

        with patch("src.exchanges.coinbase_orders.RESTClient", return_value=mock_rest_client):
            # Initialize order manager
            result = await order_manager.initialize()
            assert result is True

            # Test order placement
            order_request = OrderRequest(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100.00"),
            )

            order_response = await order_manager.place_order(order_request)
            assert isinstance(order_response, OrderResponse)
            assert order_response.id == "test_order_123"

            # Test fee calculation
            fees = await order_manager.calculate_fees(order_request)
            assert isinstance(fees, dict)
            assert "fee_rate" in fees
            assert "fee_amount" in fees
            assert fees["fee_currency"] == "USD"

            # Test order statistics
            stats = order_manager.get_order_statistics()
            assert isinstance(stats, dict)
            assert "total_orders" in stats
            assert "filled_orders" in stats
            assert "cancelled_orders" in stats
            assert "fill_rate" in stats

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_multi_exchange_compatibility(self, exchange_factory, config):
        """Test that Coinbase works alongside other exchanges."""
        from unittest.mock import AsyncMock, patch

        from src.exchanges import register_exchanges

        # Register all exchanges
        register_exchanges(exchange_factory)

        # Mock OKX clients to avoid real API calls
        mock_account_client = MagicMock()
        mock_market_client = MagicMock()
        mock_trade_client = MagicMock()
        mock_public_client = MagicMock()

        # Mock OKX balance response
        mock_account_client.get_balance.return_value = {
            "code": "0",
            "data": [
                {"ccy": "USDT", "availBal": "1000.00", "frozenBal": "0.00"},
                {"ccy": "BTC", "availBal": "0.1", "frozenBal": "0.00"},
            ],
        }

        # Mock Binance clients
        mock_binance_client = AsyncMock()
        mock_binance_client.get_account.return_value = {
            "makerCommission": 15,
            "takerCommission": 15,
            "buyerCommission": 0,
            "sellerCommission": 0,
            "canTrade": True,
            "canWithdraw": True,
            "canDeposit": True,
            "updateTime": 0,
            "accountType": "SPOT",
            "balances": [
                {"asset": "USDT", "free": "1000.00000000", "locked": "0.00000000"},
                {"asset": "BTC", "free": "0.10000000", "locked": "0.00000000"},
            ],
        }
        mock_binance_client.get_server_time.return_value = {"serverTime": 1754384454796}
        mock_binance_client.close_connection = AsyncMock()

        # Mock BinanceSocketManager
        mock_ws_manager = AsyncMock()
        mock_ws_manager.start_socket = AsyncMock()
        mock_ws_manager.start_user_socket = AsyncMock()

        # Mock Coinbase clients
        mock_coinbase_rest_client = AsyncMock()
        mock_coinbase_rest_client.get_products.return_value = [
            {"product_id": "BTC-USD", "status": "online"}
        ]
        mock_coinbase_ws_client = AsyncMock()
        mock_coinbase_ws_client.open = AsyncMock()

        with (
            patch("src.exchanges.okx.Account", return_value=mock_account_client),
            patch("src.exchanges.okx.Market", return_value=mock_market_client),
            patch("src.exchanges.okx.Trade", return_value=mock_trade_client),
            patch("src.exchanges.okx.Public", return_value=mock_public_client),
            patch("src.exchanges.binance.AsyncClient.create", return_value=mock_binance_client),
            patch("src.exchanges.binance.BinanceSocketManager", return_value=mock_ws_manager),
            patch("src.exchanges.coinbase.RESTClient", return_value=mock_coinbase_rest_client),
            patch("src.exchanges.coinbase.WSClient", return_value=mock_coinbase_ws_client),
        ):
            # Create multiple exchanges
            binance_exchange = await exchange_factory.get_exchange("binance")
            okx_exchange = await exchange_factory.get_exchange("okx")
            coinbase_exchange = await exchange_factory.get_exchange("coinbase")

            # Verify all exchanges are created
            assert binance_exchange is not None
            assert okx_exchange is not None
            assert coinbase_exchange is not None

            # Verify they have different names
            assert binance_exchange.exchange_name == "binance"
            assert okx_exchange.exchange_name == "okx"
            assert coinbase_exchange.exchange_name == "coinbase"

            # Test that they can coexist
            active_exchanges = await exchange_factory.get_all_active_exchanges()
            assert "binance" in active_exchanges
            assert "okx" in active_exchanges
            assert "coinbase" in active_exchanges

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_configuration_integration(self, config):
        """Test that Coinbase configuration is properly integrated."""
        # Test that Coinbase config is present
        assert hasattr(config.exchanges, "coinbase_api_key")
        assert hasattr(config.exchanges, "coinbase_api_secret")
        assert hasattr(config.exchanges, "coinbase_sandbox")

        # Test that Coinbase is in supported exchanges
        assert "coinbase" in config.exchanges.supported_exchanges

        # Test that Coinbase rate limits are configured
        assert "coinbase" in config.exchanges.rate_limits
        coinbase_limits = config.exchanges.rate_limits["coinbase"]
        assert "requests_per_minute" in coinbase_limits
        assert "orders_per_second" in coinbase_limits
        assert "websocket_connections" in coinbase_limits

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_error_recovery_scenarios(self, coinbase_exchange):
        """Test various error recovery scenarios."""
        # Test network disconnection recovery
        mock_rest_client = AsyncMock()
        mock_rest_client.get_accounts.side_effect = [
            Exception("Network error"),  # First call fails
            [
                {
                    "currency": "USD",
                    "available_balance": {"value": "1000.00"},
                    "hold": {"value": "0.00"},
                }
            ],  # Second call succeeds
        ]

        coinbase_exchange.client = mock_rest_client

        # First call should fail
        with pytest.raises(ExchangeError):
            await coinbase_exchange.get_account_balance()

        # Second call should succeed
        balances = await coinbase_exchange.get_account_balance()
        assert isinstance(balances, dict)
        assert "USD" in balances

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_data_consistency(self, coinbase_exchange):
        """Test data consistency across different API calls."""
        # Mock consistent data
        mock_rest_client = AsyncMock()
        mock_rest_client.get_product_ticker.return_value = {
            "product_id": "BTC-USD",
            "price": "50000.00",
            "volume_24h": "1000.0",
            "time": "2024-01-01T00:00:00.000Z",
            "bid": "50000.00",  # Match the best bid in order book
            "ask": "50001.00",
        }
        mock_rest_client.get_product_book.return_value = {
            "bids": [["50000.00", "1.0"], ["49999.00", "2.0"]],
            "asks": [["50001.00", "1.0"], ["50002.00", "2.0"]],
        }

        coinbase_exchange.client = mock_rest_client

        # Test that ticker and order book data are consistent
        ticker = await coinbase_exchange.get_ticker("BTC-USD")
        order_book = await coinbase_exchange.get_order_book("BTC-USD")

        # Verify data consistency
        assert ticker.symbol == order_book.symbol
        assert ticker.bid == order_book.bids[0][0]  # Best bid should match
        assert ticker.ask == order_book.asks[0][0]  # Best ask should match

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_performance_metrics(self, coinbase_exchange):
        """Test performance metrics and monitoring."""
        import time
        from unittest.mock import AsyncMock, patch

        # Mock REST client
        mock_rest_client = AsyncMock()
        mock_rest_client.get_accounts.return_value = [
            {
                "currency": "USD",
                "available_balance": {"value": "1000.00"},
                "hold": {"value": "0.00"},
            }
        ]
        mock_rest_client.get_products.return_value = [{"product_id": "BTC-USD", "status": "online"}]

        # Mock WebSocket client
        mock_ws_client = AsyncMock()
        mock_ws_client.open = AsyncMock()

        with (
            patch("src.exchanges.coinbase.RESTClient", return_value=mock_rest_client),
            patch("src.exchanges.coinbase.WSClient", return_value=mock_ws_client),
        ):
            # Test connection performance
            start_time = time.time()
            result = await coinbase_exchange.connect()
            connection_time = time.time() - start_time

            assert result is True
            assert connection_time < 1.0  # Should connect quickly

            # Test API call performance
            start_time = time.time()
            balances = await coinbase_exchange.get_account_balance()
            api_call_time = time.time() - start_time

            assert isinstance(balances, dict)
            assert api_call_time < 0.5  # Should be fast

            # Test health check performance
            start_time = time.time()
            health_result = await coinbase_exchange.health_check()
            health_check_time = time.time() - start_time

            assert health_result is True
            assert health_check_time < 0.5  # Should be fast
