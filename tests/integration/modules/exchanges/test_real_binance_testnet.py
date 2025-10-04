"""
Real Binance Testnet Integration Tests - NO MOCKS

These tests connect to the REAL Binance testnet API using actual credentials
from .env file. They validate actual API behavior, not simulated responses.

Requirements:
- Valid Binance testnet API keys in .env
- BINANCE_API_KEY
- BINANCE_API_SECRET
- BINANCE_TESTNET=true
"""

import asyncio
import os
from decimal import Decimal

import pytest
from dotenv import load_dotenv

from src.core.exceptions import ExchangeConnectionError, ExchangeError
from src.exchanges.binance import BinanceExchange, BINANCE_AVAILABLE

# Load real credentials
load_dotenv()

# Skip all tests if Binance SDK not available
pytestmark = pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance SDK not available")


@pytest.fixture
def real_binance_config():
    """Real Binance testnet configuration from environment."""
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        pytest.skip("Binance testnet credentials not configured in .env")

    return {
        "api_key": api_key,
        "api_secret": api_secret,
        "testnet": True,
        "sandbox": True
    }


@pytest.fixture
async def real_binance_exchange(real_binance_config, container):
    """Real Binance exchange connected to testnet."""
    exchange = BinanceExchange(real_binance_config)
    exchange.configure_dependencies(container)

    # Connect to REAL testnet API
    await exchange.connect()

    yield exchange

    # Cleanup
    await exchange.disconnect()


class TestRealBinanceConnection:
    """Test real Binance testnet API connection."""

    @pytest.mark.asyncio
    async def test_real_connection_and_ping(self, real_binance_exchange):
        """Test connecting to real Binance testnet and ping."""
        # Verify we're connected to REAL API
        assert real_binance_exchange.is_connected()

        # Ping REAL Binance servers
        result = await real_binance_exchange.ping()
        assert result is True

    @pytest.mark.asyncio
    async def test_real_account_info(self, real_binance_exchange):
        """Test getting real account information from testnet."""
        # Get REAL account balance from testnet
        balance = await real_binance_exchange.get_account_balance()

        # Should return dict with asset balances
        assert isinstance(balance, dict)
        # Testnet should have some balance
        assert len(balance) > 0

    @pytest.mark.asyncio
    async def test_real_exchange_info(self, real_binance_exchange):
        """Test loading real exchange info from Binance."""
        # Should have loaded exchange info during connection
        info = real_binance_exchange.get_exchange_info()
        assert info is not None

        # Should have trading symbols
        symbols = real_binance_exchange.get_trading_symbols()
        assert symbols is not None
        assert len(symbols) > 0


class TestRealBinanceMarketData:
    """Test real market data from Binance testnet."""

    @pytest.mark.asyncio
    async def test_real_ticker_data(self, real_binance_exchange):
        """Test getting real ticker data from Binance."""
        # Get REAL ticker from Binance testnet
        ticker = await real_binance_exchange.get_ticker("BTCUSDT")

        # Validate real data structure
        assert ticker is not None
        assert ticker.symbol == "BTCUSDT"
        assert isinstance(ticker.last_price, Decimal)
        assert ticker.last_price > Decimal("0")
        assert isinstance(ticker.bid_price, Decimal)
        assert isinstance(ticker.ask_price, Decimal)
        assert ticker.exchange == "binance"

    @pytest.mark.asyncio
    async def test_real_order_book(self, real_binance_exchange):
        """Test getting real order book from Binance."""
        # Get REAL order book from Binance testnet
        order_book = await real_binance_exchange.get_order_book("BTCUSDT", limit=10)

        # Validate real order book data
        assert order_book is not None
        assert order_book.symbol == "BTCUSDT"
        assert len(order_book.bids) > 0
        assert len(order_book.asks) > 0

        # Validate price levels are Decimal
        assert isinstance(order_book.bids[0].price, Decimal)
        assert isinstance(order_book.bids[0].quantity, Decimal)
        assert isinstance(order_book.asks[0].price, Decimal)
        assert isinstance(order_book.asks[0].quantity, Decimal)

        # Bids should be descending, asks ascending
        if len(order_book.bids) > 1:
            assert order_book.bids[0].price > order_book.bids[1].price
        if len(order_book.asks) > 1:
            assert order_book.asks[0].price < order_book.asks[1].price

    @pytest.mark.asyncio
    async def test_real_recent_trades(self, real_binance_exchange):
        """Test getting real recent trades from Binance."""
        # Get REAL recent trades from Binance testnet
        trades = await real_binance_exchange.get_recent_trades("BTCUSDT", limit=10)

        # Validate real trade data
        assert isinstance(trades, list)
        assert len(trades) > 0

        # Validate first trade
        trade = trades[0]
        assert trade.symbol == "BTCUSDT"
        assert isinstance(trade.price, Decimal)
        assert isinstance(trade.quantity, Decimal)
        assert trade.exchange == "binance"


class TestRealBinanceErrorHandling:
    """Test real error handling with Binance API."""

    @pytest.mark.asyncio
    async def test_real_invalid_symbol(self, real_binance_exchange):
        """Test real error response for invalid symbol."""
        from src.core.exceptions import ValidationError

        # Try to get ticker for non-existent symbol - should raise ValidationError
        with pytest.raises(ValidationError, match="not supported"):
            await real_binance_exchange.get_ticker("INVALIDSYMBOL")


class TestRealBinanceReconnection:
    """Test real reconnection behavior."""

    @pytest.mark.asyncio
    async def test_real_disconnect_and_reconnect(self, real_binance_exchange):
        """Test disconnecting and reconnecting to real Binance."""
        # Start connected
        assert real_binance_exchange.is_connected()

        # Disconnect from REAL API
        await real_binance_exchange.disconnect()
        assert not real_binance_exchange.is_connected()

        # Reconnect to REAL API
        await real_binance_exchange.connect()
        assert real_binance_exchange.is_connected()

        # Should be able to use API again
        result = await real_binance_exchange.ping()
        assert result is True
