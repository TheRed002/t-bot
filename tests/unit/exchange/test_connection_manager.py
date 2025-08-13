import pytest

from src.core.config import Config
from src.exchanges.connection_manager import ConnectionManager


@pytest.mark.asyncio
async def test_connection_manager_create_and_release():
    cm = ConnectionManager(Config(), "test_ex")
    # Create WebSocket connection
    ws = await cm.create_websocket_connection("wss://test", "ws1")
    assert ws is not None

    # Get connection
    got = await cm.get_connection("test_ex", "ticker")
    # Might be None depending on pool state, but call should not raise
    assert got is None or hasattr(got, "id")

    # Release connection (no-op if not pooled)
    if got is not None:
        await cm.release_connection("test_ex", got)

    # Disconnect all
    await cm.disconnect_all()

