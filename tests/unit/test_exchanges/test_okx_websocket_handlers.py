
import pytest

from src.core.config import Config
from src.exchanges.okx_websocket import OKXWebSocketManager


@pytest.mark.asyncio
async def test_okx_ws_message_handlers(monkeypatch):
    mgr = OKXWebSocketManager(Config())
    mgr.connected = True

    # Ticker data path
    data = {
        "arg": {"channel": "tickers", "instId": "BTC-USDT"},
        "data": [
            {
                "instId": "BTC-USDT",
                "bidPx": "49999",
                "askPx": "50001",
                "last": "50000",
                "vol24h": "1000",
                "change24h": "100",
                "ts": "1700000000000",
            }
        ],
    }
    await mgr._handle_data_message(data)

    # Order book path
    ob_data = {
        "arg": {"channel": "books", "instId": "BTC-USDT"},
        "data": [
            {
                "instId": "BTC-USDT",
                "bids": [["49999", "1"]],
                "asks": [["50001", "1"]],
                "ts": "1700000000000",
            }
        ],
    }
    await mgr._handle_data_message(ob_data)
