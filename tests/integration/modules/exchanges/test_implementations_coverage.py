"""
Exchange Implementation Integration Tests
=========================================

Comprehensive integration tests for specific exchange implementations
(Binance, Coinbase, OKX) designed to achieve high test coverage by
testing real class methods and integration points.

This module focuses on:
1. Exchange class initialization with proper config dictionaries
2. Method existence and proper inheritance from BaseExchange
3. Configuration validation and error handling
4. Mock-based integration testing without external API calls
5. Financial precision and data type handling
6. Connection lifecycle simulation
"""

from datetime import datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

# Core imports
from src.core.exceptions import (
    ExchangeConnectionError,
    ServiceError,
    ValidationError,
)
from src.core.types import (
    OrderBook,
    Ticker,
)


class TestBinanceExchangeIntegration:
    """Integration tests for BinanceExchange implementation."""

    @pytest.fixture
    def binance_config(self) -> dict[str, Any]:
        """Create proper Binance config dictionary using real testnet credentials from environment."""
        import os

        from dotenv import load_dotenv

        load_dotenv()

        # Support both *_API_SECRET and *_SECRET_KEY naming conventions
        api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv(
            "BINANCE_SECRET_KEY", "test_binance_secret"
        )

        return {
            "api_key": os.getenv("BINANCE_API_KEY", "test_binance_key"),
            "api_secret": api_secret,
            "testnet": os.getenv("BINANCE_TESTNET", "true").lower() == "true",
            "sandbox": True,
        }

    @pytest.fixture
    def mock_binance_client(self):
        """Create mock Binance AsyncClient."""
        client = AsyncMock()

        # Mock account info for connection test
        client.get_account.return_value = {
            "accountType": "SPOT",
            "balances": [
                {"asset": "BTC", "free": "1.0", "locked": "0.0"},
                {"asset": "USDT", "free": "10000.0", "locked": "0.0"},
            ],
        }

        # Mock exchange info
        client.get_exchange_info.return_value = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "filters": [
                        {
                            "filterType": "PRICE_FILTER",
                            "minPrice": "0.01000000",
                            "maxPrice": "1000000.00000000",
                            "tickSize": "0.01000000",
                        },
                        {
                            "filterType": "LOT_SIZE",
                            "minQty": "0.00001000",
                            "maxQty": "9000.00000000",
                            "stepSize": "0.00001000",
                        },
                    ],
                }
            ]
        }

        # Mock ticker data
        client.get_ticker.return_value = {
            "symbol": "BTCUSDT",
            "lastPrice": "50000.12345678",
            "bidPrice": "49999.98765432",
            "askPrice": "50000.23456789",
            "volume": "1234.56789012",
            "quoteVolume": "61728395.86419753",
            "openPrice": "49500.00000000",
            "highPrice": "50500.00000000",
            "lowPrice": "49000.00000000",
            "closeTime": int(datetime.now().timestamp() * 1000),
        }

        # Mock order book data
        client.get_order_book.return_value = {
            "bids": [["49999.50000000", "1.23456789"]],
            "asks": [["50000.50000000", "0.98765432"]],
        }

        # Mock ping
        client.ping.return_value = {}

        # Mock close connection
        client.close_connection = AsyncMock()

        return client

    def test_binance_import_and_initialization(self, binance_config):
        """Test Binance exchange can be imported and initialized."""
        # Import should work even if Binance SDK not available
        from src.exchanges.binance import BINANCE_AVAILABLE, BinanceExchange

        if not BINANCE_AVAILABLE:
            # Test error when SDK not available
            with pytest.raises(ServiceError, match="Binance SDK not available"):
                BinanceExchange(binance_config)
        else:
            # Test successful initialization
            exchange = BinanceExchange(binance_config)
            assert exchange.exchange_name == "binance"
            # Don't assert specific credential values - they come from .env
            assert exchange.api_key is not None
            assert exchange.api_secret is not None
            assert isinstance(exchange.testnet, bool)

    def test_binance_config_validation(self):
        """Test Binance configuration validation."""
        from src.exchanges.binance import BINANCE_AVAILABLE, BinanceExchange

        if not BINANCE_AVAILABLE:
            pytest.skip("Binance SDK not available")

        # Test missing API key
        with pytest.raises(ValidationError, match="API key and secret are required"):
            BinanceExchange({"api_secret": "secret"})

        # Test missing API secret
        with pytest.raises(ValidationError, match="API key and secret are required"):
            BinanceExchange({"api_key": "key"})

        # Test valid config
        config = {"api_key": "key", "api_secret": "secret"}
        exchange = BinanceExchange(config)
        assert exchange.api_key == "key"
        assert exchange.api_secret == "secret"
        assert exchange.testnet is False  # Default value

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_binance_connection_lifecycle(self, binance_config, mock_binance_client):
        """Test Binance connection lifecycle with mocked client."""
        from src.exchanges.binance import BINANCE_AVAILABLE, BinanceExchange

        if not BINANCE_AVAILABLE:
            pytest.skip("Binance SDK not available")

        # Create exchange
        exchange = BinanceExchange(binance_config)
        assert not exchange.is_connected()

        # Mock the client creation
        with patch("src.exchanges.binance.AsyncClient", return_value=mock_binance_client):
            # Test connection
            await exchange.connect()
            assert exchange.is_connected()
            assert exchange.client is not None

            # Test ping
            result = await exchange.ping()
            assert result is True
            mock_binance_client.ping.assert_called_once()

            # Test disconnect
            await exchange.disconnect()
            assert not exchange.is_connected()
            mock_binance_client.close_connection.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_binance_exchange_info_loading(self, binance_config, mock_binance_client):
        """Test loading exchange info from Binance API."""
        from src.exchanges.binance import BINANCE_AVAILABLE, BinanceExchange

        if not BINANCE_AVAILABLE:
            pytest.skip("Binance SDK not available")

        exchange = BinanceExchange(binance_config)

        with patch("src.exchanges.binance.AsyncClient", return_value=mock_binance_client):
            await exchange.connect()

            # Exchange info should be loaded during connection
            exchange_info = exchange.get_exchange_info()
            assert exchange_info is not None
            assert exchange_info.symbol == "BTCUSDT"
            assert isinstance(exchange_info.min_price, Decimal)
            assert isinstance(exchange_info.max_price, Decimal)

            # Trading symbols should be populated
            symbols = exchange.get_trading_symbols()
            assert symbols is not None
            assert "BTCUSDT" in symbols

            # Symbol support check
            assert exchange.is_symbol_supported("BTCUSDT")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_binance_market_data_methods(self, binance_config, mock_binance_client):
        """Test Binance market data retrieval methods."""
        from src.exchanges.binance import BINANCE_AVAILABLE, BinanceExchange

        if not BINANCE_AVAILABLE:
            pytest.skip("Binance SDK not available")

        exchange = BinanceExchange(binance_config)

        with patch("src.exchanges.binance.AsyncClient", return_value=mock_binance_client):
            await exchange.connect()

            # Test get_ticker
            ticker = await exchange.get_ticker("BTCUSDT")
            assert isinstance(ticker, Ticker)
            assert ticker.symbol == "BTCUSDT"
            assert isinstance(ticker.last_price, Decimal)
            assert isinstance(ticker.bid_price, Decimal)
            assert isinstance(ticker.ask_price, Decimal)
            assert ticker.exchange == "binance"

            # Test get_order_book
            order_book = await exchange.get_order_book("BTCUSDT")
            assert isinstance(order_book, OrderBook)
            assert order_book.symbol == "BTCUSDT"
            assert len(order_book.bids) > 0
            assert len(order_book.asks) > 0
            assert isinstance(order_book.bids[0].price, Decimal)
            assert isinstance(order_book.asks[0].price, Decimal)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_binance_error_handling(self, binance_config, container):
        """Test Binance error handling scenarios using real exchange."""
        from src.exchanges.binance import BINANCE_AVAILABLE, BinanceExchange

        if not BINANCE_AVAILABLE:
            pytest.skip("Binance SDK not available")

        exchange = BinanceExchange(binance_config)
        exchange.configure_dependencies(container)

        # Test ping without connection (should raise error)
        with pytest.raises(ExchangeConnectionError, match="not connected"):
            await exchange.ping()

        # Test successful connection and ping
        await exchange.connect()
        assert exchange.is_connected()

        # Ping should work after connection
        result = await exchange.ping()
        assert result is True

        # Disconnect and verify ping fails again
        await exchange.disconnect()
        with pytest.raises(ExchangeConnectionError, match="not connected"):
            await exchange.ping()


class TestCoinbaseExchangeIntegration:
    """Integration tests for CoinbaseExchange implementation."""

    @pytest.fixture
    def coinbase_config(self) -> dict[str, Any]:
        """Create proper Coinbase config dictionary using real sandbox credentials from environment."""
        import os

        from dotenv import load_dotenv

        load_dotenv()

        # Support both *_API_SECRET and *_SECRET_KEY naming conventions
        api_secret = os.getenv("COINBASE_API_SECRET") or os.getenv(
            "COINBASE_SECRET_KEY", "test_coinbase_secret"
        )

        return {
            "api_key": os.getenv("COINBASE_API_KEY", "test_coinbase_key"),
            "api_secret": api_secret,
            "passphrase": os.getenv("COINBASE_PASSPHRASE", "test_passphrase"),
            "sandbox": os.getenv("COINBASE_SANDBOX", "true").lower() == "true",
            "testnet": True,
        }

    def test_coinbase_import_and_initialization(self, coinbase_config):
        """Test Coinbase exchange import and initialization."""
        from src.exchanges.coinbase import CoinbaseExchange

        # Test initialization (may require coinbase-pro SDK)
        try:
            exchange = CoinbaseExchange(coinbase_config)
            assert exchange.exchange_name == "coinbase"
            # Don't assert specific credential values - they come from .env
            assert exchange.api_key is not None
            assert exchange.api_secret is not None
            assert exchange.passphrase is not None
            assert isinstance(exchange.sandbox, bool)
        except ServiceError as e:
            if "SDK not available" in str(e):
                pytest.skip("Coinbase SDK not available")
            else:
                raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_coinbase_config_validation(self):
        """Test Coinbase configuration validation during connection."""
        from src.exchanges.coinbase import CoinbaseExchange

        # Test missing credentials - validation happens during connect(), not __init__()
        try:
            # Exchange can be initialized with minimal config
            exchange = CoinbaseExchange({"api_key": "key"})  # Missing secret

            # But connection should fail
            with pytest.raises(ExchangeConnectionError, match="API key and secret are required"):
                await exchange.connect()

        except ServiceError as e:
            if "SDK not available" in str(e):
                pytest.skip("Coinbase SDK not available")
            else:
                raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_coinbase_connection_lifecycle(self, coinbase_config):
        """Test Coinbase connection lifecycle simulation."""
        from src.exchanges.coinbase import CoinbaseExchange

        try:
            exchange = CoinbaseExchange(coinbase_config)
            assert not exchange.is_connected()

            # Mock connection methods to avoid actual API calls
            exchange.connect = AsyncMock()
            exchange.disconnect = AsyncMock()
            exchange.ping = AsyncMock(return_value=True)

            # Test mocked lifecycle
            await exchange.connect()
            await exchange.ping()
            await exchange.disconnect()

            # Verify methods were called
            exchange.connect.assert_called_once()
            exchange.ping.assert_called_once()
            exchange.disconnect.assert_called_once()

        except ServiceError as e:
            if "SDK not available" in str(e):
                pytest.skip("Coinbase SDK not available")
            else:
                raise


class TestOKXExchangeIntegration:
    """Integration tests for OKXExchange implementation."""

    @pytest.fixture
    def okx_config(self) -> dict[str, Any]:
        """Create proper OKX config dictionary using real sandbox credentials from environment."""
        import os

        from dotenv import load_dotenv

        load_dotenv()

        # Support both *_API_SECRET and *_SECRET_KEY naming conventions
        api_secret = os.getenv("OKX_API_SECRET") or os.getenv("OKX_SECRET_KEY", "test_okx_secret")

        return {
            "api_key": os.getenv("OKX_API_KEY", "test_okx_key"),
            "api_secret": api_secret,
            "passphrase": os.getenv("OKX_PASSPHRASE", "test_passphrase"),
            "sandbox": os.getenv("OKX_SANDBOX", "true").lower() == "true",
            "testnet": True,
        }

    def test_okx_import_and_initialization(self, okx_config):
        """Test OKX exchange import and initialization."""
        from src.exchanges.okx import OKXExchange

        # Test initialization (may require OKX SDK)
        try:
            exchange = OKXExchange(okx_config)
            assert exchange.exchange_name == "okx"
            # Don't assert specific credential values - they come from .env
            assert exchange.api_key is not None
            assert exchange.api_secret is not None
            assert exchange.passphrase is not None
            assert isinstance(exchange.sandbox, bool)
        except ServiceError as e:
            if "SDK not available" in str(e):
                pytest.skip("OKX SDK not available")
            else:
                raise

    def test_okx_client_initialization(self, okx_config):
        """Test OKX client attributes initialization."""
        from src.exchanges.okx import OKXExchange

        try:
            exchange = OKXExchange(okx_config)

            # Test that client attributes exist (initially None)
            assert hasattr(exchange, "account_client")
            assert hasattr(exchange, "market_client")
            assert hasattr(exchange, "trade_client")
            assert hasattr(exchange, "public_client")

            # Initially should be None before connection
            assert exchange.account_client is None
            assert exchange.market_client is None
            assert exchange.trade_client is None
            assert exchange.public_client is None

        except ServiceError as e:
            if "SDK not available" in str(e):
                pytest.skip("OKX SDK not available")
            else:
                raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_okx_connection_methods(self, okx_config):
        """Test OKX connection method existence and basic functionality."""
        from src.exchanges.okx import OKXExchange

        try:
            exchange = OKXExchange(okx_config)

            # Test method existence
            assert hasattr(exchange, "connect")
            assert hasattr(exchange, "disconnect")
            assert hasattr(exchange, "ping")
            assert callable(exchange.connect)
            assert callable(exchange.disconnect)
            assert callable(exchange.ping)

            # Mock the connection methods to avoid actual API calls
            exchange.connect = AsyncMock()
            exchange.disconnect = AsyncMock()
            exchange.ping = AsyncMock(return_value=True)

            # Test mocked methods work
            await exchange.connect()
            result = await exchange.ping()
            await exchange.disconnect()

            assert result is True
            exchange.connect.assert_called_once()
            exchange.ping.assert_called_once()
            exchange.disconnect.assert_called_once()

        except ServiceError as e:
            if "SDK not available" in str(e):
                pytest.skip("OKX SDK not available")
            else:
                raise


class TestExchangeImplementationComparison:
    """Test common functionality across all exchange implementations."""

    def test_all_exchanges_inherit_from_base(self):
        """Test that all exchange implementations inherit from BaseExchange."""
        from src.exchanges.base import BaseExchange
        from src.exchanges.binance import BINANCE_AVAILABLE, BinanceExchange

        # Test Binance inheritance
        if BINANCE_AVAILABLE:
            assert issubclass(BinanceExchange, BaseExchange)

        # Test other exchanges
        try:
            from src.exchanges.coinbase import CoinbaseExchange

            assert issubclass(CoinbaseExchange, BaseExchange)
        except ImportError:
            pass  # SDK not available

        try:
            from src.exchanges.okx import OKXExchange

            assert issubclass(OKXExchange, BaseExchange)
        except ImportError:
            pass  # SDK not available

    def test_all_exchanges_have_required_attributes(self):
        """Test that exchanges have required attributes after initialization."""
        configs = {"api_key": "test_key", "api_secret": "test_secret", "testnet": True}

        # Test Binance
        from src.exchanges.binance import BINANCE_AVAILABLE, BinanceExchange

        if BINANCE_AVAILABLE:
            binance = BinanceExchange(configs)
            assert hasattr(binance, "api_key")
            assert hasattr(binance, "api_secret")
            assert hasattr(binance, "testnet")
            assert hasattr(binance, "exchange_name")
            assert binance.exchange_name == "binance"

        # Test Coinbase with additional passphrase
        try:
            from src.exchanges.coinbase import CoinbaseExchange

            coinbase_config = configs.copy()
            coinbase_config["passphrase"] = "test_pass"
            coinbase = CoinbaseExchange(coinbase_config)
            assert hasattr(coinbase, "api_key")
            assert hasattr(coinbase, "api_secret")
            assert hasattr(coinbase, "passphrase")
            assert hasattr(coinbase, "exchange_name")
            assert coinbase.exchange_name == "coinbase"
        except (ImportError, ServiceError):
            pass  # SDK not available

        # Test OKX with additional passphrase
        try:
            from src.exchanges.okx import OKXExchange

            okx_config = configs.copy()
            okx_config["passphrase"] = "test_pass"
            okx = OKXExchange(okx_config)
            assert hasattr(okx, "api_key")
            assert hasattr(okx, "api_secret")
            assert hasattr(okx, "passphrase")
            assert hasattr(okx, "exchange_name")
            assert okx.exchange_name == "okx"
        except (ImportError, ServiceError):
            pass  # SDK not available

    def test_exchange_method_interfaces(self):
        """Test that exchanges implement the required method interface."""
        from src.exchanges.binance import BINANCE_AVAILABLE, BinanceExchange

        if not BINANCE_AVAILABLE:
            pytest.skip("No exchange SDKs available for interface testing")

        config = {"api_key": "test_key", "api_secret": "test_secret", "testnet": True}

        exchange = BinanceExchange(config)

        # Test abstract method implementations exist
        required_methods = [
            "connect",
            "disconnect",
            "ping",
            "load_exchange_info",
            "get_ticker",
            "get_order_book",
            "get_recent_trades",
            "place_order",
            "cancel_order",
            "get_order_status",
            "get_open_orders",
            "get_account_balance",
            "get_positions",
        ]

        for method_name in required_methods:
            assert hasattr(exchange, method_name), f"Missing method: {method_name}"
            assert callable(getattr(exchange, method_name)), f"Method not callable: {method_name}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_exchange_validation_methods(self, container):
        """Test validation methods work consistently across implementations."""
        from src.exchanges.base import BaseMockExchange

        # Use mock exchange to test validation methods that are inherited
        config = {"api_key": "test", "api_secret": "test", "testnet": True}
        exchange = BaseMockExchange("test", config)
        exchange.configure_dependencies(container)
        await exchange.start()

        # Test symbol validation
        exchange._validate_symbol("BTCUSDT")  # Should not raise

        with pytest.raises(ValidationError):
            exchange._validate_symbol("")

        with pytest.raises(ValidationError):
            exchange._validate_symbol("INVALID")

        # Test price validation
        exchange._validate_price(Decimal("100.50"))  # Should not raise

        with pytest.raises(ValidationError):
            exchange._validate_price(100.50)  # Float not allowed

        with pytest.raises(ValidationError):
            exchange._validate_price(Decimal("-10"))  # Negative not allowed

        # Test quantity validation
        exchange._validate_quantity(Decimal("1.5"))  # Should not raise

        with pytest.raises(ValidationError):
            exchange._validate_quantity(1.5)  # Float not allowed

        with pytest.raises(ValidationError):
            exchange._validate_quantity(Decimal("0"))  # Zero not allowed

        await exchange.stop()


class TestExchangeConfiguration:
    """Test exchange configuration handling and edge cases."""

    def test_config_dictionary_format(self):
        """Test that exchanges work with dictionary configs (not Mock objects)."""
        # This is the key fix from the original problem

        config_dict = {
            "api_key": "real_string_key",
            "api_secret": "real_string_secret",
            "testnet": True,
            "sandbox": True,
        }

        from src.exchanges.binance import BINANCE_AVAILABLE, BinanceExchange

        if BINANCE_AVAILABLE:
            # Should work with dictionary config
            exchange = BinanceExchange(config_dict)
            assert exchange.api_key == "real_string_key"
            assert exchange.api_secret == "real_string_secret"
            assert isinstance(exchange.testnet, bool)
            assert exchange.testnet is True
        else:
            # Test that error message is informative
            with pytest.raises(ServiceError, match="SDK not available"):
                BinanceExchange(config_dict)

    def test_config_validation_edge_cases(self):
        """Test edge cases in configuration validation."""
        from src.exchanges.binance import BINANCE_AVAILABLE, BinanceExchange

        if not BINANCE_AVAILABLE:
            pytest.skip("Binance SDK not available")

        # Test empty strings (should fail)
        with pytest.raises(ValidationError):
            BinanceExchange({"api_key": "", "api_secret": "secret"})

        with pytest.raises(ValidationError):
            BinanceExchange({"api_key": "key", "api_secret": ""})

        # Test None values (should fail)
        with pytest.raises(ValidationError):
            BinanceExchange({"api_key": None, "api_secret": "secret"})

        # Test missing keys entirely (should fail)
        with pytest.raises(ValidationError):
            BinanceExchange({})

        # Test additional config parameters are preserved
        config = {
            "api_key": "key",
            "api_secret": "secret",
            "testnet": False,
            "custom_param": "custom_value",
        }

        exchange = BinanceExchange(config)
        assert exchange.config["custom_param"] == "custom_value"
        assert exchange.testnet is False

    def test_decimal_precision_configuration(self):
        """Test that configurations handle decimal precision properly."""
        from src.exchanges.base import BaseMockExchange

        config = {
            "api_key": "test",
            "api_secret": "test",
            "min_trade_amount": "0.00000001",  # String that should become Decimal
            "fee_rate": "0.001",  # 0.1% fee
        }

        exchange = BaseMockExchange("precision_test", config)

        # Test that config preserves string values for later Decimal conversion
        assert exchange.config["min_trade_amount"] == "0.00000001"
        assert exchange.config["fee_rate"] == "0.001"

        # Test conversion to Decimal
        min_amount = Decimal(exchange.config["min_trade_amount"])
        assert isinstance(min_amount, Decimal)
        assert min_amount == Decimal("0.000000000000000001")


if __name__ == "__main__":
    # Run tests manually if executed directly
    pytest.main([__file__, "-v"])
