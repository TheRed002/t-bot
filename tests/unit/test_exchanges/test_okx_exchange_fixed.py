"""
Fixed comprehensive tests for OKX Exchange implementation.

This module tests the actual OKXExchange class with proper mocking
of external dependencies to achieve 70%+ coverage.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.exceptions import (
    ExchangeConnectionError,
    ServiceError,
    ValidationError,
)
from src.core.types import (
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Ticker,
)
from src.exchanges.okx import OKXExchange


class TestOKXExchangeFixed:
    """Test OKX exchange implementation comprehensively with fixes."""

    @pytest.fixture
    def basic_config(self):
        """Basic OKX configuration for testing."""
        return {
            "api_key": "test_api_key",
            "api_secret": "test_api_secret",
            "passphrase": "test_passphrase",
            "sandbox": True,
        }

    @pytest.fixture
    def okx_exchange(self, basic_config):
        """Create OKXExchange instance with basic config."""
        with patch("src.exchanges.okx.Account", None), \
             patch("src.exchanges.okx.Market", None), \
             patch("src.exchanges.okx.Public", None), \
             patch("src.exchanges.okx.OKXTrade", None):
            return OKXExchange(basic_config)

    def test_init_with_basic_config(self, basic_config):
        """Test OKXExchange initialization with basic configuration."""
        with patch("src.exchanges.okx.Account", None):
            exchange = OKXExchange(basic_config)

            assert exchange.exchange_name == "okx"  # Use exchange_name property
            assert exchange.api_key == "test_api_key"
            assert exchange.api_secret == "test_api_secret"
            assert exchange.passphrase == "test_passphrase"
            assert exchange.sandbox is True
            assert exchange.connected is False  # Use property
            assert exchange.base_url == "https://www.okx.com"
            assert "wsaws.okx.com" in exchange.ws_url
            assert "wsaws.okx.com" in exchange.ws_private_url

    def test_init_production_config(self):
        """Test OKXExchange initialization with production configuration."""
        production_config = {
            "api_key": "prod_api_key",
            "api_secret": "prod_api_secret",
            "passphrase": "prod_passphrase",
            "sandbox": False,
        }

        with patch("src.exchanges.okx.Account", None):
            exchange = OKXExchange(production_config)

            assert exchange.sandbox is False
            assert exchange.base_url == "https://www.okx.com"
            assert "ws.okx.com" in exchange.ws_url
            assert "ws.okx.com" in exchange.ws_private_url

    def test_init_missing_credentials(self):
        """Test initialization with missing credentials."""
        config = {}
        with patch("src.exchanges.okx.Account", None):
            exchange = OKXExchange(config)

            assert exchange.api_key is None
            assert exchange.api_secret is None
            assert exchange.passphrase is None

    @pytest.mark.asyncio
    async def test_connect_success_with_mock_clients(self, basic_config):
        """Test successful connection with mocked OKX clients."""
        mock_account = Mock(return_value=Mock())
        mock_market = Mock(return_value=Mock())
        mock_public = Mock(return_value=Mock())
        mock_trade = Mock(return_value=Mock())

        with patch("src.exchanges.okx.Account", mock_account), \
             patch("src.exchanges.okx.Market", mock_market), \
             patch("src.exchanges.okx.Public", mock_public), \
             patch("src.exchanges.okx.OKXTrade", mock_trade):

            exchange = OKXExchange(basic_config)

            # Mock the connection test and prevent connected property setting
            with patch.object(exchange, "_test_okx_connection", new_callable=AsyncMock) as mock_test:
                # Mock _connected attribute to avoid property setter issue
                exchange._connected = False
                await exchange.connect()

                # Check that connection was established by checking the _connected attribute directly
                assert exchange._connected is True
                assert exchange.account_client is not None
                assert exchange.market_client is not None
                assert exchange.trade_client is not None
                assert exchange.public_client is not None
                mock_test.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure_missing_credentials(self):
        """Test connection failure with missing credentials."""
        minimal_config = {
            "api_key": "",
            "api_secret": "",
            "passphrase": "",
        }

        with patch("src.exchanges.okx.Account", Mock()):
            exchange = OKXExchange(minimal_config)

            with pytest.raises(ExchangeConnectionError, match="OKX API key, secret, and passphrase are required"):
                await exchange.connect()

    @pytest.mark.asyncio
    async def test_connect_without_okx_library(self, basic_config):
        """Test connection handling when OKX library is not available."""
        with patch("src.exchanges.okx.Account", None), \
             patch("src.exchanges.okx.Market", None), \
             patch("src.exchanges.okx.Public", None), \
             patch("src.exchanges.okx.OKXTrade", None):

            exchange = OKXExchange(basic_config)

            # Mock the connection test to succeed and prevent connected property setting
            with patch.object(exchange, "_test_okx_connection", new_callable=AsyncMock) as mock_test:
                # Mock _connected attribute to avoid property setter issue
                exchange._connected = False
                await exchange.connect()

                # Check that connection was established by checking the _connected attribute directly
                assert exchange._connected is True
                assert exchange.account_client is None
                assert exchange.market_client is None
                assert exchange.trade_client is None
                assert exchange.public_client is None
                mock_test.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_test_connection_fails(self, basic_config):
        """Test connection failure when connection test fails."""
        with patch("src.exchanges.okx.Account", Mock()), \
             patch("src.exchanges.okx.Market", Mock()), \
             patch("src.exchanges.okx.Public", Mock()), \
             patch("src.exchanges.okx.OKXTrade", Mock()):

            exchange = OKXExchange(basic_config)

            # Mock the connection test to fail
            with patch.object(exchange, "_test_okx_connection", new_callable=AsyncMock) as mock_test:
                mock_test.side_effect = Exception("Connection test failed")

                with pytest.raises(ExchangeConnectionError, match="OKX connection failed"):
                    await exchange.connect()

                assert exchange.connected is False

    @pytest.mark.asyncio
    async def test_disconnect_success(self, okx_exchange):
        """Test successful disconnection."""
        # Set up initial state using private attributes
        okx_exchange._connected = True  # Use private attribute
        okx_exchange.account_client = Mock()
        okx_exchange.market_client = Mock()
        okx_exchange.trade_client = Mock()
        okx_exchange.public_client = Mock()

        await okx_exchange.disconnect()

        assert okx_exchange.connected is False
        assert okx_exchange.account_client is None
        assert okx_exchange.market_client is None
        assert okx_exchange.trade_client is None
        assert okx_exchange.public_client is None

    @pytest.mark.asyncio
    async def test_ping_success(self, okx_exchange):
        """Test successful ping."""
        okx_exchange._connected = True  # Use private attribute

        with patch.object(okx_exchange, "_test_okx_connection", new_callable=AsyncMock) as mock_test:
            result = await okx_exchange.ping()

            assert result is True
            mock_test.assert_called_once()

    @pytest.mark.asyncio
    async def test_ping_not_connected(self, okx_exchange):
        """Test ping when not connected."""
        okx_exchange._connected = False  # Use private attribute

        result = await okx_exchange.ping()

        assert result is False

    @pytest.mark.asyncio
    async def test_ping_connection_test_fails(self, okx_exchange):
        """Test ping when connection test fails."""
        okx_exchange._connected = True  # Use private attribute

        with patch.object(okx_exchange, "_test_okx_connection", new_callable=AsyncMock) as mock_test:
            mock_test.side_effect = Exception("Connection test failed")

            result = await okx_exchange.ping()

            assert result is False

    @pytest.mark.asyncio
    async def test_get_ticker_not_connected(self, okx_exchange):
        """Test get_ticker when not connected."""
        okx_exchange._connected = False  # Use private attribute
        
        # Setup exchange info with supported symbols to avoid validation errors
        with patch.object(okx_exchange, '_trading_symbols', ['BTC-USDT']):
            with pytest.raises(ExchangeConnectionError, match="not connected"):
                await okx_exchange.get_ticker("BTC-USDT")

    @pytest.mark.asyncio
    async def test_get_ticker_success(self, okx_exchange):
        """Test successful ticker retrieval."""
        okx_exchange._connected = True  # Use private attribute
        okx_exchange.public_client = Mock()  # Use public_client as per implementation
        
        # Setup supported symbols to avoid validation errors
        with patch.object(okx_exchange, '_trading_symbols', ['BTC-USDT']):
            # Mock the API response
            mock_response = {
                "code": "0",
                "data": [{
                    "instId": "BTC-USDT",
                    "last": "50000.0",
                    "bidPx": "49999.0",
                    "askPx": "50001.0",
                    "bidSz": "10.0",
                    "askSz": "10.0",
                    "high24h": "51000.0",
                    "low24h": "49000.0",
                    "vol24h": "1000.5",
                    "open24h": "50000.0",
                    "ts": str(int(datetime.now(timezone.utc).timestamp() * 1000))
                }]
            }

            okx_exchange.public_client.get_ticker.return_value = mock_response

            result = await okx_exchange.get_ticker("BTC-USDT")

            assert isinstance(result, Ticker)
            assert result.symbol == "BTC-USDT"
            assert result.last_price == Decimal("50000.0")
            assert result.bid_price == Decimal("49999.0")
            assert result.ask_price == Decimal("50001.0")

    @pytest.mark.asyncio
    async def test_get_ticker_api_error(self, okx_exchange):
        """Test get_ticker when API returns error."""
        okx_exchange._connected = True  # Use private attribute
        okx_exchange.public_client = Mock()  # Use public_client as per implementation
        
        # Setup supported symbols to avoid validation errors
        with patch.object(okx_exchange, '_trading_symbols', ['BTC-USDT']):
            okx_exchange.public_client.get_ticker.side_effect = Exception("API Error")

            with pytest.raises(ServiceError, match="Ticker retrieval failed"):
                await okx_exchange.get_ticker("BTC-USDT")

    # Test utility methods
    def test_convert_symbol_to_okx_format(self, okx_exchange):
        """Test symbol conversion to OKX format."""
        # Based on actual implementation logic
        assert okx_exchange._convert_symbol_to_okx_format("BTC-USDT") == "BTC-USDT"  # Already correct
        assert okx_exchange._convert_symbol_to_okx_format("BTCUSDT") == "BTC-USDT"   # Maps to BTC-USDT
        assert okx_exchange._convert_symbol_to_okx_format("ETHUSDT") == "ETH-USDT"   # Maps to ETH-USDT
        assert okx_exchange._convert_symbol_to_okx_format("UNKNOWN") == "UNKNOWN"    # No mapping

    def test_convert_symbol_from_okx_format(self, okx_exchange):
        """Test symbol conversion from OKX format."""
        # Based on actual implementation - removes dash
        assert okx_exchange._convert_symbol_from_okx_format("BTC-USDT") == "BTCUSDT"
        assert okx_exchange._convert_symbol_from_okx_format("ETH-BTC") == "ETHBTC"

    def test_convert_order_type_to_okx(self, okx_exchange):
        """Test order type conversion to OKX format."""
        assert okx_exchange._convert_order_type_to_okx(OrderType.MARKET) == "market"
        assert okx_exchange._convert_order_type_to_okx(OrderType.LIMIT) == "limit"

    def test_convert_okx_order_type_to_unified(self, okx_exchange):
        """Test order type conversion from OKX format."""
        assert okx_exchange._convert_okx_order_type_to_unified("market") == OrderType.MARKET
        assert okx_exchange._convert_okx_order_type_to_unified("limit") == OrderType.LIMIT

    def test_convert_okx_status_to_order_status(self, okx_exchange):
        """Test OKX status conversion to unified status."""
        assert okx_exchange._convert_okx_status_to_order_status("live") == OrderStatus.NEW
        assert okx_exchange._convert_okx_status_to_order_status("filled") == OrderStatus.FILLED
        assert okx_exchange._convert_okx_status_to_order_status("canceled") == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_test_okx_connection_success(self, okx_exchange):
        """Test successful OKX connection test."""
        okx_exchange.account_client = Mock()
        okx_exchange.account_client.get_account_balance.return_value = {"data": []}

        # Should not raise exception
        await okx_exchange._test_okx_connection()

    @pytest.mark.asyncio
    async def test_test_okx_connection_no_client(self, okx_exchange):
        """Test OKX connection test without client."""
        okx_exchange.account_client = None

        # Should not raise exception when no client (mock mode)
        await okx_exchange._test_okx_connection()

    @pytest.mark.asyncio
    async def test_test_okx_connection_failure(self, okx_exchange):
        """Test OKX connection test failure."""
        okx_exchange.account_client = Mock()
        okx_exchange.public_client = Mock()
        okx_exchange.account_client.get_balance.side_effect = Exception("API Error")

        with pytest.raises(ExchangeConnectionError, match="OKX connection test failed"):
            await okx_exchange._test_okx_connection()

    @pytest.mark.asyncio
    async def test_validate_okx_order_success(self, okx_exchange):
        """Test successful order validation."""
        order_request = OrderRequest(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.0")
        )

        # Should not raise exception
        await okx_exchange._validate_okx_order(order_request)

    @pytest.mark.asyncio
    async def test_validate_okx_order_invalid_symbol(self, okx_exchange):
        """Test order validation with invalid symbol."""
        # Use mock to bypass OrderRequest validation
        order_request = Mock()
        order_request.symbol = ""
        order_request.side = OrderSide.BUY
        order_request.order_type = OrderType.LIMIT
        order_request.quantity = Decimal("0.1")
        order_request.price = Decimal("50000.0")

        with pytest.raises(ValidationError, match="Order must have symbol and quantity"):
            await okx_exchange._validate_okx_order(order_request)

    @pytest.mark.asyncio
    async def test_validate_okx_order_invalid_quantity(self, okx_exchange):
        """Test order validation with invalid quantity."""
        # Use mock to test zero quantity validation
        order_request = Mock()
        order_request.symbol = "BTC-USDT"  # Valid symbol
        order_request.quantity = Decimal("-0.1")  # Negative quantity to trigger the positive check
        order_request.side = OrderSide.BUY
        order_request.order_type = OrderType.LIMIT
        order_request.price = Decimal("50000.0")

        with pytest.raises(ValidationError, match="Order quantity must be positive"):
            await okx_exchange._validate_okx_order(order_request)

    @pytest.mark.asyncio
    async def test_validate_okx_order_invalid_price_for_limit(self, okx_exchange):
        """Test order validation with invalid price for limit order."""
        # Use mock to bypass OrderRequest validation
        order_request = Mock()
        order_request.symbol = "BTC-USDT"
        order_request.side = OrderSide.BUY
        order_request.order_type = OrderType.LIMIT
        order_request.quantity = Decimal("0.1")
        order_request.price = Decimal("0")

        with pytest.raises(ValidationError, match="Limit orders must have positive price"):
            await okx_exchange._validate_okx_order(order_request)

    @pytest.mark.asyncio
    async def test_method_calls_without_connection(self, okx_exchange):
        """Test that methods properly check connection status."""
        okx_exchange._connected = False  # Use private attribute
        
        # Setup supported symbols to avoid validation errors  
        with patch.object(okx_exchange, '_trading_symbols', ['BTC-USDT']):
            with pytest.raises(ExchangeConnectionError):
                await okx_exchange.get_ticker("BTC-USDT")

            with pytest.raises(ExchangeConnectionError):
                await okx_exchange.get_order_book("BTC-USDT")

            with pytest.raises(ExchangeConnectionError):
                await okx_exchange.get_recent_trades("BTC-USDT")

    def test_convert_order_to_okx(self, okx_exchange):
        """Test order conversion to OKX format."""
        order_request = OrderRequest(
            symbol="BTCUSDT",  # Will be converted to BTC-USDT
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.0")
        )

        result = okx_exchange._convert_order_to_okx(order_request)

        assert result["instId"] == "BTC-USDT"
        assert result["side"] == "buy"
        assert result["ordType"] == "limit"
        # Decimal converts to string with full precision
        assert result["sz"] == "0.10000000"
        assert result["px"] == "50000.00000000"

    def test_convert_okx_order_to_response(self, okx_exchange):
        """Test OKX order response conversion."""
        okx_order = {
            "ordId": "okx-123",
            "instId": "BTC-USDT",
            "state": "filled",
            "px": "50000.0",
            "sz": "0.1",
            "fillPx": "50000.0",
            "fillSz": "0.1"
        }

        result = okx_exchange._convert_okx_order_to_response(okx_order)

        assert isinstance(result, OrderResponse)
        assert result.order_id == "okx-123"
        assert result.symbol == "BTCUSDT"  # Converted from OKX format
        assert result.status == OrderStatus.FILLED
        assert result.price == Decimal("50000.0")
        assert result.quantity == Decimal("0.1")
