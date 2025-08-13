import asyncio

import pytest

from src.core.config import Config
from src.exchanges.binance_websocket import BinanceWebSocketHandler
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_binance_ws_init_and_reconnect(monkeypatch):
    # Provide a dummy client to avoid real manager init
    dummy_client = AsyncMock()
    ws = BinanceWebSocketHandler(Config(), dummy_client, "binance")
    ws.reconnect_attempts = ws.max_reconnect_attempts - 1

    async def dummy():
        return None

    # Directly invoke internal error handler to exercise reconnect logging path
    await ws._handle_stream_error("ticker")
