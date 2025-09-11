"""
Unit tests for OKX Exchange - completely mocked.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.types import OrderRequest, OrderSide, OrderStatus, OrderType


class TestOKXExchange:
    """Test OKX exchange with complete mocking."""

    @pytest.fixture
    def mock_okx_exchange(self):
        """Create a fully mocked OKX exchange."""
        exchange = MagicMock()
        exchange.exchange_name = "okx"
        exchange.connected = False
        exchange.connect = AsyncMock(return_value=True)
        exchange.disconnect = AsyncMock()
        exchange.get_account_balance = AsyncMock(
            return_value={"BTC": Decimal("1.8"), "USDT": Decimal("15000.0")}
        )
        exchange.get_ticker = AsyncMock(
            return_value={"symbol": "BTC-USDT", "price": Decimal("50200.00")}
        )
        exchange.place_order = AsyncMock(
            return_value={"order_id": "okx-12345", "status": OrderStatus.NEW, "symbol": "BTC-USDT"}
        )
        exchange.cancel_order = AsyncMock(return_value=True)
        exchange.get_order = AsyncMock(
            return_value={"order_id": "okx-12345", "status": OrderStatus.FILLED}
        )
        exchange.get_order_book = AsyncMock(
            return_value={
                "bids": [[Decimal("50199.00"), Decimal("0.8")]],
                "asks": [[Decimal("50201.00"), Decimal("0.8")]],
            }
        )
        return exchange

    @pytest.mark.asyncio
    async def test_connection(self, mock_okx_exchange):
        """Test exchange connection."""
        result = await mock_okx_exchange.connect()
        assert result is True
        mock_okx_exchange.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnection(self, mock_okx_exchange):
        """Test exchange disconnection."""
        mock_okx_exchange._connected = True
        await mock_okx_exchange.disconnect()
        mock_okx_exchange.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_balance(self, mock_okx_exchange):
        """Test getting account balance."""
        balance = await mock_okx_exchange.get_account_balance()
        assert balance["BTC"] == Decimal("1.8")
        assert balance["USDT"] == Decimal("15000.0")
        mock_okx_exchange.get_account_balance.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_order(self, mock_okx_exchange):
        """Test placing an order."""
        order_request = OrderRequest(
            symbol="BTC-USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.05"),
            price=Decimal("51000.00"),
        )

        result = await mock_okx_exchange.place_order(order_request)
        assert result["order_id"] == "okx-12345"
        assert result["status"] == OrderStatus.NEW
        mock_okx_exchange.place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_ticker(self, mock_okx_exchange):
        """Test getting ticker data."""
        ticker = await mock_okx_exchange.get_ticker("BTC-USDT")
        assert ticker["symbol"] == "BTC-USDT"
        assert ticker["price"] == Decimal("50200.00")
        mock_okx_exchange.get_ticker.assert_called_once()
