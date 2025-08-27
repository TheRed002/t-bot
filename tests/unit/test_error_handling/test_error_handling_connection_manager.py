"""
Unit tests for connection manager functionality.

Tests connection state management, health monitoring, and resilience features.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.core.config import Config
from src.core.exceptions import ExchangeConnectionError
from src.error_handling.connection_manager import (
    ConnectionHealth,
    ConnectionManager,
    ConnectionState,
)


class TestConnectionState:
    """Test connection state enumeration."""

    def test_connection_state_values(self):
        """Test connection state enum values."""
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.FAILED.value == "failed"
        assert ConnectionState.MAINTENANCE.value == "maintenance"

    def test_connection_state_comparison(self):
        """Test connection state comparison."""
        assert ConnectionState.CONNECTED != ConnectionState.DISCONNECTED
        assert ConnectionState.CONNECTING == ConnectionState.CONNECTING


class TestConnectionHealth:
    """Test connection health metrics."""

    def test_connection_health_creation(self):
        """Test connection health creation."""
        timestamp = datetime.now(timezone.utc)
        health = ConnectionHealth(
            last_heartbeat=timestamp,
            latency_ms=50.0,
            packet_loss=0.01,
            connection_quality=0.95,
            uptime_seconds=3600,
            reconnect_count=2,
            last_error="Connection timeout",
        )

        assert health.last_heartbeat == timestamp
        assert health.latency_ms == 50.0
        assert health.packet_loss == 0.01
        assert health.connection_quality == 0.95
        assert health.uptime_seconds == 3600
        assert health.reconnect_count == 2
        assert health.last_error == "Connection timeout"

    def test_connection_health_to_dict(self):
        """Test connection health to dictionary conversion."""
        timestamp = datetime.now(timezone.utc)
        health = ConnectionHealth(
            last_heartbeat=timestamp,
            latency_ms=100.0,
            packet_loss=0.05,
            connection_quality=0.8,
            uptime_seconds=1800,
            reconnect_count=1,
        )

        health_dict = health.to_dict()
        assert health_dict["latency_ms"] == 100.0
        assert health_dict["packet_loss"] == 0.05
        assert health_dict["connection_quality"] == 0.8
        assert health_dict["uptime_seconds"] == 1800
        assert health_dict["reconnect_count"] == 1
        assert "last_heartbeat" in health_dict


class TestConnectionManager:
    """Test connection manager functionality."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Config()

    @pytest.fixture
    def connection_manager(self, config):
        """Provide connection manager instance."""
        return ConnectionManager(config)

    def test_connection_manager_initialization(self, config):
        """Test connection manager initialization."""
        manager = ConnectionManager(config)
        assert manager.config == config
        assert manager.connections == {}
        assert manager.health_monitors == {}
        assert "exchange" in manager.reconnect_policies
        assert "database" in manager.reconnect_policies
        assert "websocket" in manager.reconnect_policies
        assert "exchange" in manager.heartbeat_intervals

    @pytest.mark.asyncio
    async def test_establish_connection_success(self, connection_manager):
        """Test successful connection establishment."""

        async def mock_connect(**kwargs):
            return {"status": "connected", "connection_id": "test_conn"}

        # Mock the health monitoring to avoid hanging
        with patch.object(connection_manager, "_monitor_connection_health") as mock_monitor:
            result = await connection_manager.establish_connection(
                connection_id="test_conn",
                connection_type="exchange",
                connect_func=mock_connect,
                host="localhost",
                port=8080,
            )

            assert result is True
            assert "test_conn" in connection_manager.connections
            assert connection_manager.connections["test_conn"]["state"] == ConnectionState.CONNECTED
            # Verify health monitoring was called
            mock_monitor.assert_called_once_with("test_conn")

    @pytest.mark.asyncio
    async def test_establish_connection_failure(self, connection_manager):
        """Test connection establishment failure."""

        async def mock_connect(**kwargs):
            raise ExchangeConnectionError("Connection failed")

        # Mock the health monitoring to avoid hanging
        with patch.object(connection_manager, "_monitor_connection_health"):
            result = await connection_manager.establish_connection(
                connection_id="test_conn",
                connection_type="exchange",
                connect_func=mock_connect,
                host="localhost",
                port=8080,
            )

            assert result is False
            assert "test_conn" not in connection_manager.connections

    @pytest.mark.asyncio
    async def test_close_connection(self, connection_manager):
        """Test connection closure."""

        # First establish a connection
        async def mock_connect(**kwargs):
            return {"status": "connected", "connection_id": "test_conn"}

        # Mock the health monitoring to avoid hanging
        with patch.object(connection_manager, "_monitor_connection_health"):
            await connection_manager.establish_connection(
                connection_id="test_conn",
                connection_type="exchange",
                connect_func=mock_connect,
                host="localhost",
                port=8080,
            )

            # Then close it
            result = await connection_manager.close_connection("test_conn")
            assert result is True
            # The connection should still exist but with DISCONNECTED state
            assert "test_conn" in connection_manager.connections
            assert (
                connection_manager.connections["test_conn"]["state"] == ConnectionState.DISCONNECTED
            )

    @pytest.mark.asyncio
    async def test_reconnect_connection(self, connection_manager):
        """Test connection reconnection."""

        # First establish a connection
        async def mock_connect(**kwargs):
            return {"status": "connected", "connection_id": "test_conn"}

        # Mock the health monitoring to avoid hanging
        with patch.object(connection_manager, "_monitor_connection_health"):
            await connection_manager.establish_connection(
                connection_id="test_conn",
                connection_type="exchange",
                connect_func=mock_connect,
                host="localhost",
                port=8080,
            )

            # Then reconnect it
            result = await connection_manager.reconnect_connection("test_conn")
            assert result is True
            assert "test_conn" in connection_manager.connections

    def test_get_connection_status(self, connection_manager):
        """Test getting connection status."""
        # Add a mock connection with all required fields
        connection_manager.connections["test_conn"] = {
            "connection": {"status": "connected"},
            "type": "exchange",
            "state": ConnectionState.CONNECTED,
            "established_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "reconnect_count": 0,
        }

        # Add health monitor
        connection_manager.health_monitors["test_conn"] = ConnectionHealth(
            last_heartbeat=datetime.now(timezone.utc),
            latency_ms=50.0,
            packet_loss=0.01,
            connection_quality=0.95,
            uptime_seconds=3600,
            reconnect_count=0,
        )

        status = connection_manager.get_connection_status("test_conn")
        assert status is not None
        assert status["state"] == ConnectionState.CONNECTED.value
        assert status["type"] == "exchange"

    def test_get_all_connection_status(self, connection_manager):
        """Test getting all connection statuses."""
        # Add mock connections with all required fields
        now = datetime.now(timezone.utc)
        connection_manager.connections["conn1"] = {
            "connection": {"status": "connected"},
            "type": "exchange",
            "state": ConnectionState.CONNECTED,
            "established_at": now,
            "last_activity": now,
            "reconnect_count": 0,
        }
        connection_manager.connections["conn2"] = {
            "connection": {"status": "disconnected"},
            "type": "database",
            "state": ConnectionState.DISCONNECTED,
            "established_at": now,
            "last_activity": now,
            "reconnect_count": 1,
        }

        # Add health monitors
        connection_manager.health_monitors["conn1"] = ConnectionHealth(
            last_heartbeat=now,
            latency_ms=50.0,
            packet_loss=0.01,
            connection_quality=0.95,
            uptime_seconds=3600,
            reconnect_count=0,
        )
        connection_manager.health_monitors["conn2"] = ConnectionHealth(
            last_heartbeat=now,
            latency_ms=100.0,
            packet_loss=0.05,
            connection_quality=0.8,
            uptime_seconds=1800,
            reconnect_count=1,
        )

        all_status = connection_manager.get_all_connection_status()
        assert len(all_status) == 2
        assert "conn1" in all_status
        assert "conn2" in all_status

    def test_is_connection_healthy(self, connection_manager):
        """Test connection health check."""
        # Add a healthy connection with all required fields
        now = datetime.now(timezone.utc)
        connection_manager.connections["test_conn"] = {
            "connection": {"status": "connected"},
            "type": "exchange",
            "state": ConnectionState.CONNECTED,
            "established_at": now,
            "last_activity": now,
            "reconnect_count": 0,
        }

        # Add a healthy connection
        connection_manager.health_monitors["test_conn"] = ConnectionHealth(
            last_heartbeat=now,
            latency_ms=50.0,
            packet_loss=0.01,
            connection_quality=0.95,
            uptime_seconds=3600,
            reconnect_count=0,
        )

        assert connection_manager.is_connection_healthy("test_conn") is True

    @pytest.mark.asyncio
    async def test_queue_message(self, connection_manager):
        """Test message queuing."""
        message = {"type": "order", "data": {"symbol": "BTCUSDT", "side": "buy"}}

        result = await connection_manager.queue_message("test_conn", message)
        assert result is True
        assert "test_conn" in connection_manager.message_queues
        assert connection_manager.message_queues["test_conn"].size() == 1

    @pytest.mark.asyncio
    async def test_flush_message_queue(self, connection_manager):
        """Test message queue flushing."""
        # Add messages to queue
        messages = [
            {"type": "order", "data": {"symbol": "BTCUSDT", "side": "buy"}},
            {"type": "cancel", "data": {"order_id": "123"}},
        ]

        for message in messages:
            await connection_manager.queue_message("test_conn", message)

        # Flush the queue (no send function needed)
        flushed_count = await connection_manager.flush_message_queue("test_conn")
        assert flushed_count == 2
        assert connection_manager.message_queues["test_conn"].size() == 0
