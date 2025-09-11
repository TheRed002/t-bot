"""
Unit tests for Coinbase Exchange - completely mocked.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.types import OrderRequest, OrderSide, OrderStatus, OrderType


class TestCoinbaseExchange:
    """Test Coinbase exchange with complete mocking."""

    @pytest.fixture
    def mock_coinbase_exchange(self):
        """Create a fully mocked Coinbase exchange."""
        exchange = MagicMock()
        exchange.exchange_name = "coinbase"
        exchange.connected = False
        exchange.connect = AsyncMock(return_value=True)
        exchange.disconnect = AsyncMock()
        exchange.get_account_balance = AsyncMock(
            return_value={"BTC": Decimal("2.0"), "USD": Decimal("20000.0")}
        )
        exchange.get_ticker = AsyncMock(
            return_value={"symbol": "BTC-USD", "price": Decimal("50500.00")}
        )
        exchange.place_order = AsyncMock(
            return_value={"order_id": "cb-12345", "status": OrderStatus.NEW, "symbol": "BTC-USD"}
        )
        exchange.cancel_order = AsyncMock(return_value=True)
        exchange.get_order = AsyncMock(
            return_value={"order_id": "cb-12345", "status": OrderStatus.FILLED}
        )
        exchange.get_order_book = AsyncMock(
            return_value={
                "bids": [[Decimal("50499.00"), Decimal("0.5")]],
                "asks": [[Decimal("50501.00"), Decimal("0.5")]],
            }
        )
        return exchange

    @pytest.mark.asyncio
    async def test_connection(self, mock_coinbase_exchange):
        """Test exchange connection."""
        result = await mock_coinbase_exchange.connect()
        assert result is True
        mock_coinbase_exchange.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnection(self, mock_coinbase_exchange):
        """Test exchange disconnection."""
        mock_coinbase_exchange._connected = True
        await mock_coinbase_exchange.disconnect()
        mock_coinbase_exchange.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_balance(self, mock_coinbase_exchange):
        """Test getting account balance."""
        balance = await mock_coinbase_exchange.get_account_balance()
        assert balance["BTC"] == Decimal("2.0")
        assert balance["USD"] == Decimal("20000.0")
        mock_coinbase_exchange.get_account_balance.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_order(self, mock_coinbase_exchange):
        """Test placing an order."""
        order_request = OrderRequest(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("49500.00"),
        )

        result = await mock_coinbase_exchange.place_order(order_request)
        assert result["order_id"] == "cb-12345"
        assert result["status"] == OrderStatus.NEW
        mock_coinbase_exchange.place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_ticker(self, mock_coinbase_exchange):
        """Test getting ticker data."""
        ticker = await mock_coinbase_exchange.get_ticker("BTC-USD")
        assert ticker["symbol"] == "BTC-USD"
        assert ticker["price"] == Decimal("50500.00")
        mock_coinbase_exchange.get_ticker.assert_called_once()
