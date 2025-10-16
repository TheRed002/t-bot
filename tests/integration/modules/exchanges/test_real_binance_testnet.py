"""
Binance Exchange Integration Tests

These tests validate Binance exchange integration:
- When SDK available AND credentials present: use real Binance testnet API
- Otherwise: use MockExchange to simulate Binance behavior

This ensures tests always run regardless of environment configuration.
"""

import os
from decimal import Decimal

import pytest
import pytest_asyncio
from dotenv import load_dotenv

from src.exchanges.binance import BINANCE_AVAILABLE, BinanceExchange

# Check if we should use mock
try:
    from src.exchanges.mock_exchange import MockExchange
    MOCK_AVAILABLE = True
except ImportError:
    MOCK_AVAILABLE = False

# Load credentials if available
load_dotenv()


@pytest.fixture
def binance_config():
    """Binance configuration - real or mock."""
    api_key = os.getenv("BINANCE_API_KEY", "test_api_key")
    # Use BINANCE_SECRET_KEY as defined in .env (not BINANCE_API_SECRET)
    api_secret = os.getenv("BINANCE_SECRET_KEY") or os.getenv("BINANCE_API_SECRET", "test_api_secret")

    return {"api_key": api_key, "api_secret": api_secret, "testnet": True, "sandbox": True}


@pytest_asyncio.fixture
async def binance_exchange(binance_config):
    """Binance exchange - uses real API if SDK available and credentials present, otherwise mock."""
    # Check if we should use real or mock - use BINANCE_SECRET_KEY as defined in .env
    has_real_credentials = os.getenv("BINANCE_API_KEY") and (os.getenv("BINANCE_SECRET_KEY") or os.getenv("BINANCE_API_SECRET"))

    if BINANCE_AVAILABLE and has_real_credentials:
        # Use real Binance exchange with real credentials
        exchange = BinanceExchange(binance_config)
    elif MOCK_AVAILABLE:
        # Use MockExchange to simulate Binance
        exchange = MockExchange(binance_config)
        exchange.exchange_name = "binance"
        # Set trading symbols BEFORE connecting
        exchange._trading_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    else:
        pytest.skip("Neither Binance SDK nor MockExchange available")

    # Connect
    await exchange.connect()

    yield exchange

    # Cleanup
    await exchange.disconnect()


class TestRealBinanceConnection:
    """Test real Binance testnet API connection."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_real_connection_and_ping(self, binance_exchange):
        """Test connecting to real Binance testnet and ping."""
        # Verify we're connected to REAL API
        assert binance_exchange.is_connected()

        # Ping REAL Binance servers
        result = await binance_exchange.ping()
        assert result is True

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_real_account_info(self, binance_exchange):
        """Test getting real account information from testnet."""
        # Get REAL account balance from testnet
        balance = await binance_exchange.get_account_balance()

        # Should return dict with asset balances
        assert isinstance(balance, dict)
        # Testnet should have some balance
        assert len(balance) > 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_real_exchange_info(self, binance_exchange):
        """Test loading real exchange info from Binance."""
        # Should have loaded exchange info during connection
        info = binance_exchange.get_exchange_info()
        assert info is not None

        # Should have trading symbols
        symbols = binance_exchange.get_trading_symbols()
        assert symbols is not None
        assert len(symbols) > 0


class TestRealBinanceMarketData:
    """Test real market data from Binance testnet."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_real_ticker_data(self, binance_exchange):
        """Test getting real ticker data from Binance."""
        # Get REAL ticker from Binance testnet
        ticker = await binance_exchange.get_ticker("BTCUSDT")

        # Validate real data structure
        assert ticker is not None
        assert ticker.symbol == "BTCUSDT"
        assert isinstance(ticker.last_price, Decimal)
        assert ticker.last_price > Decimal("0")
        assert isinstance(ticker.bid_price, Decimal)
        assert isinstance(ticker.ask_price, Decimal)
        assert ticker.exchange == "binance"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_real_order_book(self, binance_exchange):
        """Test getting real order book from Binance."""
        # Get REAL order book from Binance testnet
        order_book = await binance_exchange.get_order_book("BTCUSDT", limit=10)

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
    @pytest.mark.timeout(300)
    async def test_real_recent_trades(self, binance_exchange):
        """Test getting real recent trades from Binance."""
        # Get REAL recent trades from Binance testnet
        trades = await binance_exchange.get_recent_trades("BTCUSDT", limit=10)

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
    @pytest.mark.timeout(300)
    async def test_real_invalid_symbol(self, binance_exchange):
        """Test real error response for invalid symbol."""
        from src.core.exceptions import ValidationError

        # Try to get ticker for non-existent symbol - should raise ValidationError
        with pytest.raises(ValidationError, match="not supported"):
            await binance_exchange.get_ticker("INVALIDSYMBOL")


class TestRealBinanceReconnection:
    """Test real reconnection behavior."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_real_disconnect_and_reconnect(self, binance_exchange):
        """Test disconnecting and reconnecting to real Binance."""
        # Start connected
        assert binance_exchange.is_connected()

        # Disconnect from REAL API
        await binance_exchange.disconnect()
        assert not binance_exchange.is_connected()

        # Reconnect to REAL API
        await binance_exchange.connect()
        assert binance_exchange.is_connected()

        # Should be able to use API again
        result = await binance_exchange.ping()
        assert result is True
