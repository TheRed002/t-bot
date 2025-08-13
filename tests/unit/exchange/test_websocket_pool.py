import asyncio
from datetime import datetime, timedelta

import pytest

from src.core.config import Config
from src.exchanges.websocket_pool import (
    ConnectionType,
    PooledConnection,
    WebSocketConnectionPool,
)


@pytest.mark.asyncio
async def test_websocket_pool_basic_lifecycle():
    pool = WebSocketConnectionPool(exchange="test_ex")

    # Create a new connection
    conn = await pool.get_connection(ConnectionType.TICKER)
    assert conn is not None
    assert isinstance(conn, PooledConnection)
    assert conn.connection_id in pool.active_connections

    # Reuse connection
    reused = await pool.get_connection(ConnectionType.TICKER)
    assert reused is not None
    assert reused.connection_id == conn.connection_id

    # Record message and subscription
    pool.record_message(conn.connection_id)
    ok = pool.record_subscription(conn.connection_id)
    assert ok is True

    # Stats
    pool_stats = pool.get_pool_stats()
    assert pool_stats["total_connections"] >= 1
    conn_stats = pool.get_connection_stats(conn.connection_id)
    assert conn_stats["connection_id"] == conn.connection_id

    # Release connection
    await pool.release_connection(conn)

    # Cleanup: make it old to trigger removal
    conn.created_at = datetime.now() - timedelta(seconds=pool.connection_timeout + 1)
    removed = await pool.cleanup_old_connections()
    assert removed >= 1


@pytest.mark.asyncio
async def test_websocket_pool_limits_and_close_all():
    pool = WebSocketConnectionPool(exchange="test_ex", max_connections=2, max_subscriptions=1)

    c1 = await pool.get_connection(ConnectionType.TICKER)
    c2 = await pool.get_connection(ConnectionType.ORDERBOOK)
    assert c1 and c2

    # Exceed subscription limit
    pool.record_subscription(c1.connection_id)
    assert pool.record_subscription(c1.connection_id) is False

    # No third connection due to limit, should warn and return None
    none = await pool.get_connection(ConnectionType.TRADES)
    assert none is None

    # Close all
    await pool.close_all_connections()
    assert len(pool.active_connections) == 0
