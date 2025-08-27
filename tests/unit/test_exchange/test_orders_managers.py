from decimal import Decimal

import pytest

from src.core.config import Config
from src.core.types import OrderRequest, OrderSide, OrderType
from src.exchanges.binance_orders import BinanceOrderManager
from src.exchanges.coinbase_orders import CoinbaseOrderManager
from src.exchanges.okx_orders import OKXOrderManager


@pytest.mark.asyncio
async def test_binance_order_manager_validation(monkeypatch):
    om = BinanceOrderManager(Config(), client=None)

    class DummyClient:
        async def order_market(self, **kwargs):
            return {
                "orderId": "1",
                "clientOrderId": "cid-1",
                "symbol": kwargs["symbol"],
                "status": "FILLED",
                "side": "BUY",
                "type": "MARKET",
                "origQty": "1.0",
                "executedQty": "1.0",
                "price": "0",
                "time": 1700000000000,
            }

    om.client = DummyClient()
    order = OrderRequest(symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("1"))
    resp = await om.place_market_order(order)
    assert resp.id


@pytest.mark.asyncio
async def test_coinbase_order_manager_init(monkeypatch):
    om = CoinbaseOrderManager(Config(), "coinbase")
    class Dummy:
        async def get_time(self):
            return {"iso": "2024-01-01T00:00:00Z"}
        async def get_product(self, symbol):
            return {"product_id": symbol}
        async def create_order(self, **kwargs):
            return {"order_id": "abc", "status": "OPEN", "product_id": kwargs.get("product_id", "BTC-USD")}
    monkeypatch.setattr("src.exchanges.coinbase_orders.RESTClient", lambda *a, **k: Dummy())
    await om.initialize()


@pytest.mark.asyncio
async def test_okx_order_manager_get_status(monkeypatch):
    class DummyTrade:
        def get_order_details(self, **kwargs):
            return {"code": "0", "data": [{"state": "filled"}]}
    om = OKXOrderManager(Config(), DummyTrade())
    status = await om.get_order_status("1")
    assert status


