import asyncio
from decimal import Decimal

import pytest

from src.core.config import Config
from src.exchanges.okx_websocket import OKXWebSocketManager


@pytest.mark.asyncio
async def test_okx_ws_reconnect_logging(monkeypatch):
    manager = OKXWebSocketManager(Config())
    manager.reconnect_attempts = manager.max_reconnect_attempts - 1

    # Force reconnect path without real sockets
    async def dummy():
        return None

    monkeypatch.setattr(manager, "_connect_public_websocket", dummy)
    monkeypatch.setattr(manager, "_connect_private_websocket", dummy)

    await manager._reconnect_public_websocket()
    await manager._reconnect_private_websocket()
