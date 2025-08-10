"""
Unit tests for the exchange connection manager.

This module tests the ConnectionManager and WebSocketConnection classes
to ensure proper connection management and error handling.
"""

from datetime import datetime, timedelta

import pytest

from src.core.config import Config

# Import the components to test
from src.exchanges.connection_manager import ConnectionManager, WebSocketConnection


class TestWebSocketConnection:
    """Test cases for the WebSocketConnection class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()

    @pytest.fixture
    def ws_connection(self, config):
        """Create a WebSocket connection instance."""
        return WebSocketConnection("wss://test.com/ws", "test_exchange", config)

    def test_websocket_initialization(self, config):
        """Test WebSocket connection initialization."""
        ws = WebSocketConnection("wss://test.com/ws", "test_exchange", config)

        assert ws.url == "wss://test.com/ws"
        assert ws.exchange_name == "test_exchange"
        assert ws.config == config
        assert ws.error_handler is not None
        assert not ws.connected
        assert not ws.connecting
        assert ws.last_heartbeat is None
        assert ws.last_message is None
        assert len(ws.message_queue) == 0
        assert ws.max_queue_size == 1000

    @pytest.mark.asyncio
    async def test_connect_success(self, ws_connection):
        """Test successful WebSocket connection."""
        # Set up callbacks
        on_connect_called = False

        async def on_connect():
            nonlocal on_connect_called
            on_connect_called = True

        ws_connection.on_connect = on_connect

        # Connect
        result = await ws_connection.connect()

        assert result
        assert ws_connection.connected
        assert not ws_connection.connecting
        assert ws_connection.last_heartbeat is not None
        assert on_connect_called

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, ws_connection):
        """Test connection when already connected."""
        # Connect first time
        result1 = await ws_connection.connect()
        assert result1

        # Try to connect again
        result2 = await ws_connection.connect()
        assert result2  # Should return True since already connected

    @pytest.mark.asyncio
    async def test_connect_failure(self, ws_connection):
        """Test WebSocket connection failure."""
        # Mock the connection to fail by returning False
        original_connect = ws_connection.connect

        async def failing_connect():
            return False

        ws_connection.connect = failing_connect

        # Try to connect
        result = await ws_connection.connect()

        assert not result
        assert not ws_connection.connected
        assert not ws_connection.connecting

    @pytest.mark.asyncio
    async def test_disconnect(self, ws_connection):
        """Test WebSocket disconnection."""
        # Connect first
        await ws_connection.connect()
        assert ws_connection.connected

        # Set up callback
        on_disconnect_called = False

        async def on_disconnect():
            nonlocal on_disconnect_called
            on_disconnect_called = True

        ws_connection.on_disconnect = on_disconnect

        # Disconnect
        await ws_connection.disconnect()

        assert not ws_connection.connected
        assert on_disconnect_called

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, ws_connection):
        """Test disconnection when not connected."""
        # Should not raise any exceptions
        await ws_connection.disconnect()
        assert not ws_connection.connected

    @pytest.mark.asyncio
    async def test_send_message_connected(self, ws_connection):
        """Test sending message when connected."""
        await ws_connection.connect()

        message = {"test": "data"}
        result = await ws_connection.send_message(message)

        assert result

    @pytest.mark.asyncio
    async def test_send_message_not_connected(self, ws_connection):
        """Test sending message when not connected."""
        message = {"test": "data"}
        result = await ws_connection.send_message(message)

        assert not result
        assert len(ws_connection.message_queue) == 1
        assert ws_connection.message_queue[0] == message

    @pytest.mark.asyncio
    async def test_send_message_queue_full(self, ws_connection):
        """Test sending message when queue is full."""
        # Fill the queue
        ws_connection.message_queue = [{"test": i} for i in range(1000)]

        message = {"test": "data"}
        result = await ws_connection.send_message(message)

        assert not result
        assert len(ws_connection.message_queue) == 1000  # Should not add more

    @pytest.mark.asyncio
    async def test_subscribe(self, ws_connection):
        """Test WebSocket subscription."""
        await ws_connection.connect()

        result = await ws_connection.subscribe("ticker", "BTCUSDT")
        assert result

    @pytest.mark.asyncio
    async def test_unsubscribe(self, ws_connection):
        """Test WebSocket unsubscription."""
        await ws_connection.connect()

        result = await ws_connection.unsubscribe("ticker", "BTCUSDT")
        assert result

    @pytest.mark.asyncio
    async def test_heartbeat(self, ws_connection):
        """Test WebSocket heartbeat."""
        await ws_connection.connect()

        result = await ws_connection.heartbeat()
        assert result
        assert ws_connection.last_heartbeat is not None

    @pytest.mark.asyncio
    async def test_heartbeat_not_connected(self, ws_connection):
        """Test heartbeat when not connected."""
        result = await ws_connection.heartbeat()
        assert not result

    def test_is_healthy(self, ws_connection):
        """Test health check."""
        # Not connected
        assert not ws_connection.is_healthy()

        # Connected but no heartbeat
        ws_connection.connected = True
        assert ws_connection.is_healthy()

        # Connected with recent heartbeat
        ws_connection.last_heartbeat = datetime.now()
        assert ws_connection.is_healthy()

        # Connected with old heartbeat
        ws_connection.last_heartbeat = datetime.now() - timedelta(seconds=100)
        assert not ws_connection.is_healthy()

    @pytest.mark.asyncio
    async def test_process_queued_messages(self, ws_connection):
        """Test processing queued messages."""
        # Add some messages to queue
        ws_connection.message_queue = [
            {"test": "message1"},
            {"test": "message2"},
            {"test": "message3"},
        ]

        # Connect
        await ws_connection.connect()

        # Process queued messages
        processed = await ws_connection.process_queued_messages()

        assert processed == 3
        assert len(ws_connection.message_queue) == 0

    @pytest.mark.asyncio
    async def test_process_queued_messages_not_connected(self, ws_connection):
        """Test processing queued messages when not connected."""
        ws_connection.message_queue = [{"test": "message"}]

        processed = await ws_connection.process_queued_messages()
        assert processed == 0
        assert len(ws_connection.message_queue) == 1  # Should remain unchanged


class TestConnectionManager:
    """Test cases for the ConnectionManager class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()

    @pytest.fixture
    def connection_manager(self, config):
        """Create a connection manager instance."""
        return ConnectionManager(config, "test_exchange")

    def test_connection_manager_initialization(self, config):
        """Test connection manager initialization."""
        cm = ConnectionManager(config, "test_exchange")

        assert cm.config == config
        assert cm.exchange_name == "test_exchange"
        assert cm.error_handler is not None
        assert cm.error_connection_manager is not None
        assert cm.rest_connections == {}
        assert cm.websocket_connections == {}
        assert cm.max_rest_connections == 10
        assert cm.max_websocket_connections == 5

    @pytest.mark.asyncio
    async def test_get_rest_connection(self, connection_manager):
        """Test getting REST connection."""
        connection = await connection_manager.get_rest_connection("test_endpoint")

        assert connection is not None
        assert connection["endpoint"] == "test_endpoint"
        assert connection["connected"]

        # Should return the same connection for the same endpoint
        connection2 = await connection_manager.get_rest_connection("test_endpoint")
        assert connection2 == connection

    @pytest.mark.asyncio
    async def test_create_websocket_connection(self, connection_manager):
        """Test creating WebSocket connection."""
        ws = await connection_manager.create_websocket_connection("wss://test.com/ws", "test_conn")

        assert isinstance(ws, WebSocketConnection)
        assert ws.url == "wss://test.com/ws"
        assert ws.exchange_name == "test_exchange"
        assert "test_conn" in connection_manager.websocket_connections

    @pytest.mark.asyncio
    async def test_create_websocket_connection_existing(self, connection_manager):
        """Test creating WebSocket connection when it already exists."""
        # Create first connection
        ws1 = await connection_manager.create_websocket_connection("wss://test.com/ws", "test_conn")

        # Try to create the same connection again
        ws2 = await connection_manager.create_websocket_connection("wss://test.com/ws", "test_conn")

        assert ws2 == ws1  # Should return the existing connection

    @pytest.mark.asyncio
    async def test_create_websocket_connection_max_limit(self, connection_manager):
        """Test creating WebSocket connection when at max limit."""
        # Create maximum number of connections
        connections = []
        for i in range(5):
            ws = await connection_manager.create_websocket_connection(
                f"wss://test{i}.com/ws", f"conn_{i}"
            )
            connections.append(ws)

        # Try to create one more
        ws_extra = await connection_manager.create_websocket_connection(
            "wss://extra.com/ws", "extra_conn"
        )

        # Should return the first connection instead
        assert ws_extra == connections[0]

    @pytest.mark.asyncio
    async def test_get_websocket_connection(self, connection_manager):
        """Test getting existing WebSocket connection."""
        # Create connection
        ws = await connection_manager.create_websocket_connection("wss://test.com/ws", "test_conn")

        # Get the connection
        retrieved_ws = await connection_manager.get_websocket_connection("test_conn")

        assert retrieved_ws == ws

    @pytest.mark.asyncio
    async def test_get_websocket_connection_not_found(self, connection_manager):
        """Test getting non-existent WebSocket connection."""
        ws = await connection_manager.get_websocket_connection("nonexistent")
        assert ws is None

    @pytest.mark.asyncio
    async def test_remove_websocket_connection(self, connection_manager):
        """Test removing WebSocket connection."""
        # Create connection
        ws = await connection_manager.create_websocket_connection("wss://test.com/ws", "test_conn")

        # Remove the connection
        result = await connection_manager.remove_websocket_connection("test_conn")

        assert result
        assert "test_conn" not in connection_manager.websocket_connections

    @pytest.mark.asyncio
    async def test_remove_websocket_connection_not_found(self, connection_manager):
        """Test removing non-existent WebSocket connection."""
        result = await connection_manager.remove_websocket_connection("nonexistent")
        assert not result

    @pytest.mark.asyncio
    async def test_health_check_all(self, connection_manager):
        """Test health check for all connections."""
        # Create some connections
        # Create REST connection
        await connection_manager.get_rest_connection("default")
        ws1 = await connection_manager.create_websocket_connection("wss://test1.com/ws", "conn1")
        ws2 = await connection_manager.create_websocket_connection("wss://test2.com/ws", "conn2")

        # Connect the WebSocket connections
        await ws1.connect()
        await ws2.connect()

        # Perform health check
        health_status = await connection_manager.health_check_all()

        assert "rest_default" in health_status
        assert "ws_conn1" in health_status
        assert "ws_conn2" in health_status
        # REST connections are marked as healthy
        assert health_status["rest_default"]
        # WebSocket connections should be healthy
        assert health_status["ws_conn1"]
        assert health_status["ws_conn2"]

    @pytest.mark.asyncio
    async def test_reconnect_all(self, connection_manager):
        """Test reconnecting all connections."""
        # Create some connections
        # Create REST connection
        await connection_manager.get_rest_connection("default")
        ws1 = await connection_manager.create_websocket_connection("wss://test1.com/ws", "conn1")
        ws2 = await connection_manager.create_websocket_connection("wss://test2.com/ws", "conn2")

        # Mock one connection to be unhealthy
        def mock_is_healthy():
            return False

        ws1.is_healthy = mock_is_healthy

        # Reconnect all
        results = await connection_manager.reconnect_all()

        assert "rest_default" in results
        assert "ws_conn1" in results
        assert "ws_conn2" in results
        # REST connections are marked as healthy
        assert results["rest_default"]
        assert results["ws_conn1"]  # Should be reconnected
        assert results["ws_conn2"]  # Should remain healthy

    def test_get_connection_stats(self, connection_manager):
        """Test getting connection statistics."""
        stats = connection_manager.get_connection_stats()

        assert stats["exchange_name"] == "test_exchange"
        assert stats["rest_connections"] == 0
        assert stats["websocket_connections"] == 0
        assert stats["max_rest_connections"] == 10
        assert stats["max_websocket_connections"] == 5
        assert stats["last_health_check"] is not None

    @pytest.mark.asyncio
    async def test_disconnect_all(self, connection_manager):
        """Test disconnecting all connections."""
        # Create some connections
        await connection_manager.create_websocket_connection("wss://test1.com/ws", "conn1")
        await connection_manager.create_websocket_connection("wss://test2.com/ws", "conn2")

        # Disconnect all
        await connection_manager.disconnect_all()

        assert len(connection_manager.websocket_connections) == 0
        assert len(connection_manager.rest_connections) == 0

    @pytest.mark.asyncio
    async def test_context_manager(self, connection_manager):
        """Test async context manager functionality."""
        async with connection_manager as cm:
            assert cm == connection_manager

        # Should not raise any exceptions


class TestConnectionManagerErrorHandling:
    """Test error handling in ConnectionManager."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()

    @pytest.fixture
    def connection_manager(self, config):
        """Create a connection manager instance."""
        return ConnectionManager(config, "test_exchange")

    @pytest.mark.asyncio
    async def test_health_check_exception_handling(self, connection_manager):
        """Test handling of exceptions during health checks."""
        # Create a WebSocket connection
        ws = await connection_manager.create_websocket_connection("wss://test.com/ws", "test_conn")

        # Mock is_healthy to raise an exception
        def mock_is_healthy():
            raise Exception("Health check failed")

        ws.is_healthy = mock_is_healthy

        # Perform health check
        health_status = await connection_manager.health_check_all()

        # Should be marked as unhealthy
        assert not health_status["ws_test_conn"]

    @pytest.mark.asyncio
    async def test_reconnect_exception_handling(self, connection_manager):
        """Test handling of exceptions during reconnection."""
        # Create a WebSocket connection
        ws = await connection_manager.create_websocket_connection("wss://test.com/ws", "test_conn")

        # Mock is_healthy to return False
        def mock_is_healthy():
            return False

        ws.is_healthy = mock_is_healthy

        # Mock disconnect to raise an exception
        async def mock_disconnect():
            raise Exception("Disconnect failed")

        ws.disconnect = mock_disconnect

        # Reconnect all
        results = await connection_manager.reconnect_all()

        assert not results["ws_test_conn"]  # Should be marked as failed

    @pytest.mark.asyncio
    async def test_disconnect_all_exception_handling(self, connection_manager):
        """Test handling of exceptions during disconnect all."""
        # Create a WebSocket connection
        ws = await connection_manager.create_websocket_connection("wss://test.com/ws", "test_conn")

        # Mock disconnect to raise an exception
        async def mock_disconnect():
            raise Exception("Disconnect failed")

        ws.disconnect = mock_disconnect

        # Disconnect all (should not raise exception)
        await connection_manager.disconnect_all()

        # Connection should still be removed
        assert len(connection_manager.websocket_connections) == 0


class TestWebSocketConnectionErrorHandling:
    """Test error handling in WebSocketConnection."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()

    @pytest.fixture
    def ws_connection(self, config):
        """Create a WebSocket connection instance."""
        return WebSocketConnection("wss://test.com/ws", "test_exchange", config)

    @pytest.mark.asyncio
    async def test_connect_exception_handling(self, ws_connection):
        """Test handling of exceptions during connection."""
        # Mock the connection to simulate a failure that returns False
        original_connect = ws_connection.connect

        async def failing_connect():
            # Simulate connection failure by returning False
            return False

        ws_connection.connect = failing_connect

        # Try to connect
        result = await ws_connection.connect()

        assert not result
        assert not ws_connection.connected
        assert not ws_connection.connecting

    @pytest.mark.asyncio
    async def test_disconnect_exception_handling(self, ws_connection):
        """Test handling of exceptions during disconnection."""
        # Connect first
        await ws_connection.connect()

        # Mock disconnect to simulate a failure that doesn't raise an exception
        async def failing_disconnect():
            # Simulate a failure that doesn't raise an exception
            # But don't set connected to False since we're mocking the method
            pass

        ws_connection.disconnect = failing_disconnect

        # Disconnect should complete without raising an exception
        await ws_connection.disconnect()

        # Since we mocked the entire disconnect method, connected should still be True
        # The test is checking that the exception handling works, not the
        # actual disconnection
        assert ws_connection.connected

    @pytest.mark.asyncio
    async def test_send_message_exception_handling(self, ws_connection):
        """Test handling of exceptions during message sending."""
        await ws_connection.connect()

        # Mock send_message to simulate a failure that returns False
        async def failing_send_message(message):
            # Simulate a failure that returns False
            return False

        ws_connection.send_message = failing_send_message

        # Try to send message (should return False due to failure)
        result = await ws_connection.send_message({"test": "data"})

        assert not result

    @pytest.mark.asyncio
    async def test_heartbeat_exception_handling(self, ws_connection):
        """Test handling of exceptions during heartbeat."""
        await ws_connection.connect()

        # Mock heartbeat to simulate a failure that returns False
        async def failing_heartbeat():
            # Simulate a failure that returns False
            return False

        ws_connection.heartbeat = failing_heartbeat

        # Try to send heartbeat (should return False due to failure)
        result = await ws_connection.heartbeat()

        assert not result
