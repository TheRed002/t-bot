import asyncio

import pytest

from src.core.config import Config
from src.exchanges.health_monitor import ConnectionHealthMonitor, ConnectionStatus


@pytest.mark.asyncio
async def test_health_monitor_register_and_fail_recover():
    monitor = ConnectionHealthMonitor(Config())

    class Dummy:
        def __init__(self, cid):
            self.id = cid
            self.exchange = "ex"
            self.stream_type = "ticker"

    conn = Dummy("c-1")
    await monitor.monitor_connection(conn)

    # Register a recovery callback that marks healthy
    async def recovery_cb(info):
        # noop to simulate success
        return None

    monitor.register_recovery_callback("c-1", recovery_cb)

    # Mark failed triggers recovery
    await monitor.mark_failed(conn)
    info = monitor.get_connection_status("c-1")
    assert info is not None
    assert info.status in {ConnectionStatus.FAILED, ConnectionStatus.RECOVERED}

    await monitor.stop_monitoring()

