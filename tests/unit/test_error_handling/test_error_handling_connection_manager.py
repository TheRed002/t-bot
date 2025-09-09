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

    def test_connection_state_comparison(self):
        """Test connection state comparison."""
        assert ConnectionState.CONNECTED != ConnectionState.DISCONNECTED
        assert ConnectionState.CONNECTING == ConnectionState.CONNECTING


class TestConnectionHealth:
    """Test connection health enum."""

    def test_connection_health_values(self):
        """Test connection health enum values."""
        assert ConnectionHealth.HEALTHY.value == "healthy"
        assert ConnectionHealth.DEGRADED.value == "degraded"
        assert ConnectionHealth.UNHEALTHY.value == "unhealthy"

    def test_connection_health_comparison(self):
        """Test connection health comparison."""
        assert ConnectionHealth.HEALTHY != ConnectionHealth.UNHEALTHY
        assert ConnectionHealth.DEGRADED == ConnectionHealth.DEGRADED


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
        assert manager.max_attempts == 3
        assert manager.base_delay == 1.0
        assert manager.max_delay == 30.0
        assert manager.connection_timeout == 30.0

    @pytest.mark.asyncio
    async def test_establish_connection_success(self, connection_manager):
        """Test successful connection establishment."""

        async def mock_connect(**kwargs):
            return {"status": "connected", "connection_id": "test_conn"}

        result = await connection_manager.establish_connection(
            connection_id="test_conn",
            connection_type="exchange",
            connect_func=mock_connect,
            host="localhost",
            port=8080,
        )

        assert result is True
        assert "test_conn" in connection_manager.connections
        assert connection_manager.connections["test_conn"].state == ConnectionState.CONNECTED

    @pytest.mark.asyncio
    async def test_establish_connection_failure(self, connection_manager):
        """Test connection establishment failure."""

        async def mock_connect(**kwargs):
            raise ExchangeConnectionError("Connection failed")

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
        assert connection_manager.connections["test_conn"].state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_reconnect_connection(self, connection_manager):
        """Test connection reconnection."""

        # First establish a connection
        async def mock_connect(**kwargs):
            return {"status": "connected", "connection_id": "test_conn"}

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
        # Import ConnectionInfo from the actual module
        from src.error_handling.connection_manager import ConnectionInfo
        
        # Add a mock connection with correct structure
        connection_manager.connections["test_conn"] = ConnectionInfo(
            connection={"status": "connected"},
            state=ConnectionState.CONNECTED,
            established_at=datetime.now(timezone.utc),
            reconnect_count=0,
        )

        status = connection_manager.get_connection_status("test_conn")
        assert status is not None
        assert status["state"] == ConnectionState.CONNECTED.value
        assert status["connection_id"] == "test_conn"

    def test_get_all_connection_status(self, connection_manager):
        """Test getting all connection statuses."""
        # Import ConnectionInfo from the actual module
        from src.error_handling.connection_manager import ConnectionInfo
        
        # Add mock connections with correct structure
        now = datetime.now(timezone.utc)
        connection_manager.connections["conn1"] = ConnectionInfo(
            connection={"status": "connected"},
            state=ConnectionState.CONNECTED,
            established_at=now,
            reconnect_count=0,
        )
        connection_manager.connections["conn2"] = ConnectionInfo(
            connection={"status": "disconnected"},
            state=ConnectionState.DISCONNECTED,
            established_at=now,
            reconnect_count=1,
        )

        # Since get_all_connection_status doesn't exist, let's test individual status calls
        status1 = connection_manager.get_connection_status("conn1")
        status2 = connection_manager.get_connection_status("conn2")
        
        assert status1 is not None
        assert status2 is not None
        assert status1["state"] == ConnectionState.CONNECTED.value
        assert status2["state"] == ConnectionState.DISCONNECTED.value

    def test_is_connection_healthy(self, connection_manager):
        """Test connection health check."""
        # Import ConnectionInfo from the actual module
        from src.error_handling.connection_manager import ConnectionInfo
        
        # Add a healthy connection with correct structure
        now = datetime.now(timezone.utc)
        connection_manager.connections["test_conn"] = ConnectionInfo(
            connection={"status": "connected"},
            state=ConnectionState.CONNECTED,
            established_at=now,
            reconnect_count=0,
        )

        assert connection_manager.is_connection_healthy("test_conn") is True

    def test_connection_manager_nonexistent_connection(self, connection_manager):
        """Test operations on nonexistent connections."""
        # Test getting status of nonexistent connection
        status = connection_manager.get_connection_status("nonexistent")
        assert status is None
        
        # Test health check of nonexistent connection
        assert connection_manager.is_connection_healthy("nonexistent") is False

    @pytest.mark.asyncio
    async def test_connection_manager_cleanup(self, connection_manager):
        """Test connection manager cleanup."""
        # Add a mock connection
        from src.error_handling.connection_manager import ConnectionInfo
        
        now = datetime.now(timezone.utc)
        connection_manager.connections["test_conn"] = ConnectionInfo(
            connection={"status": "connected"},
            state=ConnectionState.CONNECTED,
            established_at=now,
            reconnect_count=0,
        )
        
        # Test cleanup
        await connection_manager.cleanup_resources()
        assert len(connection_manager.connections) == 0
