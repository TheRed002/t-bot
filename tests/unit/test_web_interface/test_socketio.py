"""
Unit tests for Socket.IO functionality in T-Bot Trading System.
"""

import asyncio
import unittest.mock
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.web_interface.socketio_manager import SocketIOManager, TradingNamespace


@pytest.fixture
def mock_sio_server():
    """Create a mock Socket.IO server."""
    server = MagicMock()
    server.emit = AsyncMock()
    server.enter_room = AsyncMock()
    server.leave_room = AsyncMock()
    return server


@pytest.fixture
def trading_namespace():
    """Create a TradingNamespace instance."""
    return TradingNamespace("/")


@pytest.fixture
def socketio_mgr():
    """Create a SocketIOManager instance."""
    return SocketIOManager()


class TestTradingNamespace:
    """Test cases for TradingNamespace."""

    @pytest.mark.asyncio
    async def test_on_connect_without_auth(self, trading_namespace):
        """Test client connection without authentication."""
        sid = "test_sid_123"
        environ = {}

        # Mock the emit method
        trading_namespace.emit = AsyncMock()

        result = await trading_namespace.on_connect(sid, environ)

        assert result is True
        assert sid in trading_namespace.connected_clients
        assert trading_namespace.connected_clients[sid]["authenticated"] is False
        assert sid not in trading_namespace.authenticated_sessions

        # Check welcome message was sent
        trading_namespace.emit.assert_called()

    @pytest.mark.asyncio
    async def test_on_connect_with_valid_token(self, trading_namespace):
        """Test client connection with valid authentication token."""
        sid = "test_sid_456"
        environ = {}
        auth = {"token": "valid_token_123"}

        mock_token_data = {
            "user_id": "user_123",
            "username": "testuser",
            "scopes": ["read", "write"]
        }
        with patch.object(trading_namespace, "_validate_token", new_callable=AsyncMock, return_value=mock_token_data):
            with patch.object(trading_namespace, "emit_standardized", new_callable=AsyncMock) as mock_emit_standardized:
                result = await trading_namespace.on_connect(sid, environ, auth)

        assert result is True
        assert sid in trading_namespace.connected_clients
        assert trading_namespace.connected_clients[sid]["authenticated"] is True
        assert sid in trading_namespace.authenticated_sessions

        # Check if authenticated event was emitted via emit_standardized
        mock_emit_standardized.assert_any_call("authenticated", {"status": "success"}, room=sid, processing_mode="request_reply")

    @pytest.mark.asyncio
    async def test_on_connect_with_invalid_token(self, trading_namespace):
        """Test client connection with invalid authentication token."""
        sid = "test_sid_789"
        environ = {}
        auth = {"token": "invalid_token"}

        with patch.object(trading_namespace, "_validate_token", new_callable=AsyncMock, return_value=None):
            with patch.object(trading_namespace, "emit") as mock_emit:
                result = await trading_namespace.on_connect(sid, environ, auth)

        assert result is True
        assert sid in trading_namespace.connected_clients
        assert trading_namespace.connected_clients[sid]["authenticated"] is False
        assert sid not in trading_namespace.authenticated_sessions

        # Check if auth failed event was emitted
        mock_emit.assert_any_call(
            "auth_error", {"error": "Invalid or expired token"}, room=sid
        )

    @pytest.mark.asyncio
    async def test_on_disconnect(self, trading_namespace):
        """Test client disconnection."""
        sid = "test_sid_disconnect"

        # First connect the client
        trading_namespace.connected_clients[sid] = {
            "connected_at": datetime.utcnow().isoformat(),
            "authenticated": True,
        }
        trading_namespace.authenticated_sessions.add(sid)

        # Disconnect
        await trading_namespace.on_disconnect(sid)

        assert sid not in trading_namespace.connected_clients
        assert sid not in trading_namespace.authenticated_sessions

    @pytest.mark.asyncio
    async def test_on_authenticate_success(self, trading_namespace):
        """Test successful authentication after connection."""
        sid = "test_sid_auth"
        data = {"token": "valid_token"}

        # Setup connected client
        trading_namespace.connected_clients[sid] = {"authenticated": False, "subscriptions": set()}

        with patch.object(trading_namespace, "_validate_token", new_callable=AsyncMock, return_value=True):
            with patch.object(trading_namespace, "emit") as mock_emit:
                with patch.object(trading_namespace, "enter_room") as mock_enter:
                    await trading_namespace.on_authenticate(sid, data)

        assert trading_namespace.connected_clients[sid]["authenticated"] is True
        assert sid in trading_namespace.authenticated_sessions
        # Get the actual call args to check the structure
        call_args = mock_emit.call_args
        assert call_args[0][0] == "authenticated"
        assert call_args[1]["room"] == sid
        # Check that response contains required fields
        response_data = call_args[0][1]
        assert response_data["status"] == "success"
        assert "timestamp" in response_data
        mock_enter.assert_called_with(sid, "authenticated")

    @pytest.mark.asyncio
    async def test_on_subscribe_authenticated(self, trading_namespace):
        """Test subscription to channels when authenticated."""
        sid = "test_sid_sub"
        channels = ["market_data", "bot_status", "portfolio"]
        data = {"channels": channels}

        # Setup authenticated client
        trading_namespace.connected_clients[sid] = {"authenticated": True, "subscriptions": set()}
        trading_namespace.authenticated_sessions.add(sid)

        with patch.object(trading_namespace, "enter_room") as mock_enter:
            with patch.object(trading_namespace, "emit") as mock_emit:
                await trading_namespace.on_subscribe(sid, data)

        # Check rooms were joined
        for channel in channels:
            mock_enter.assert_any_call(sid, channel)

        # Check subscriptions were recorded
        assert trading_namespace.connected_clients[sid]["subscriptions"] == set(channels)

        # Check confirmation was sent
        # Get the actual call args to check the structure
        call_args = mock_emit.call_args
        assert call_args[0][0] == "subscribed"
        assert call_args[1]["room"] == sid
        # Check that response contains required fields
        response_data = call_args[0][1]
        assert set(response_data["channels"]) == set(channels)
        assert "timestamp" in response_data

    @pytest.mark.asyncio
    async def test_on_subscribe_unauthenticated(self, trading_namespace):
        """Test subscription rejection when not authenticated."""
        sid = "test_sid_unauth"
        data = {"channels": ["market_data"]}

        # Setup unauthenticated client
        trading_namespace.connected_clients[sid] = {"authenticated": False, "subscriptions": set()}

        with patch.object(trading_namespace, "emit") as mock_emit:
            await trading_namespace.on_subscribe(sid, data)

        mock_emit.assert_called_with("error", {"error": "Authentication required"}, room=sid)

    @pytest.mark.asyncio
    async def test_on_ping(self, trading_namespace):
        """Test ping/pong functionality."""
        sid = "test_sid_ping"
        timestamp = datetime.utcnow().isoformat()
        data = {"timestamp": timestamp}

        with patch.object(trading_namespace, "emit") as mock_emit:
            await trading_namespace.on_ping(sid, data)

        # Get the actual call args to check the structure
        call_args = mock_emit.call_args
        assert call_args[0][0] == "pong"
        assert call_args[1]["room"] == sid
        # Check that response contains required fields
        response_data = call_args[0][1]
        assert response_data["latency"] == timestamp
        assert "timestamp" in response_data

    @pytest.mark.asyncio
    async def test_on_execute_order_authenticated(self, trading_namespace):
        """Test order execution when authenticated."""
        sid = "test_sid_order"
        data = {"type": "market", "symbol": "BTC/USDT", "side": "buy", "amount": 0.1}

        trading_namespace.authenticated_sessions.add(sid)

        with patch.object(trading_namespace, "emit") as mock_emit:
            await trading_namespace.on_execute_order(sid, data)

        mock_emit.assert_called()
        call_args = mock_emit.call_args[0]
        assert call_args[0] == "order.created"
        assert "order_id" in call_args[1]
        assert call_args[1]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_on_execute_order_missing_fields(self, trading_namespace):
        """Test order execution with missing required fields."""
        sid = "test_sid_order_invalid"
        data = {
            "type": "market",
            "symbol": "BTC/USDT",
            # Missing 'side' and 'amount'
        }

        trading_namespace.authenticated_sessions.add(sid)

        with patch.object(trading_namespace, "emit") as mock_emit:
            await trading_namespace.on_execute_order(sid, data)

        mock_emit.assert_called_with(
            "order_error", {"error": "Missing required order fields"}, room=sid
        )

    @pytest.mark.asyncio
    async def test_on_get_portfolio(self, trading_namespace):
        """Test portfolio data retrieval."""
        sid = "test_sid_portfolio"
        data = {}

        trading_namespace.authenticated_sessions.add(sid)

        with patch.object(trading_namespace, "emit") as mock_emit:
            await trading_namespace.on_get_portfolio(sid, data)

        mock_emit.assert_called()
        call_args = mock_emit.call_args[0]
        assert call_args[0] == "portfolio_data"
        portfolio = call_args[1]
        assert "total_value" in portfolio
        assert "available_balance" in portfolio
        assert "positions" in portfolio
        assert isinstance(portfolio["positions"], list)


class TestSocketIOManager:
    """Test cases for SocketIOManager."""

    def test_create_server(self, socketio_mgr):
        """Test Socket.IO server creation."""
        cors_origins = ["http://localhost:3000", "http://localhost:8080"]
        server = socketio_mgr.create_server(cors_origins)

        assert server is not None
        assert socketio_mgr.sio is server

    def test_create_app(self, socketio_mgr):
        """Test ASGI app creation."""
        # First create server
        socketio_mgr.create_server()

        # Then create app
        app = socketio_mgr.create_app()

        assert app is not None
        assert socketio_mgr.app is app

    def test_create_app_without_server_raises_error(self, socketio_mgr):
        """Test that creating app without server raises error."""
        with pytest.raises(RuntimeError, match="Socket.IO server not created"):
            socketio_mgr.create_app()

    @pytest.mark.asyncio
    async def test_start_background_tasks(self, socketio_mgr):
        """Test starting background tasks."""
        await socketio_mgr.start_background_tasks()

        assert socketio_mgr.is_running is True
        assert len(socketio_mgr.background_tasks) == 3  # market, bot, portfolio

        # Clean up
        await socketio_mgr.stop_background_tasks()

    @pytest.mark.asyncio
    async def test_stop_background_tasks(self, socketio_mgr):
        """Test stopping background tasks."""
        # Start tasks first
        await socketio_mgr.start_background_tasks()

        # Stop tasks
        await socketio_mgr.stop_background_tasks()

        assert socketio_mgr.is_running is False
        assert len(socketio_mgr.background_tasks) == 0

    @pytest.mark.asyncio
    async def test_broadcast_market_data(self, socketio_mgr):
        """Test market data broadcasting."""
        socketio_mgr.sio = MagicMock()
        socketio_mgr.sio.emit = AsyncMock()
        socketio_mgr._is_running = True

        # Run one iteration of broadcast
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            try:
                await socketio_mgr._broadcast_market_data()
            except asyncio.CancelledError:
                pass

        # Check emit was called
        socketio_mgr.sio.emit.assert_called()
        call_args = socketio_mgr.sio.emit.call_args[0]
        assert call_args[0] == "market_data"
        assert "data" in call_args[1]
        assert "type" in call_args[1]["data"]
        assert call_args[1]["data"]["type"] == "market_update"

    @pytest.mark.asyncio
    async def test_broadcast_bot_status(self, socketio_mgr):
        """Test bot status broadcasting."""
        socketio_mgr.sio = MagicMock()
        socketio_mgr.sio.emit = AsyncMock()
        socketio_mgr._is_running = True

        # Run one iteration of broadcast
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            try:
                await socketio_mgr._broadcast_bot_status()
            except asyncio.CancelledError:
                pass

        # Check emit was called
        socketio_mgr.sio.emit.assert_called()
        call_args = socketio_mgr.sio.emit.call_args[0]
        assert call_args[0] == "bot_status"
        assert "type" in call_args[1]
        assert call_args[1]["type"] == "bot_status"

    @pytest.mark.asyncio
    async def test_broadcast_portfolio_updates(self, socketio_mgr):
        """Test portfolio update broadcasting."""
        socketio_mgr.sio = MagicMock()
        socketio_mgr.sio.emit = AsyncMock()
        socketio_mgr._is_running = True

        # Run one iteration of broadcast
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            try:
                await socketio_mgr._broadcast_portfolio_updates()
            except asyncio.CancelledError:
                pass

        # Check emit was called
        socketio_mgr.sio.emit.assert_called()
        call_args = socketio_mgr.sio.emit.call_args[0]
        assert call_args[0] == "portfolio_update"
        assert "type" in call_args[1]
        assert call_args[1]["type"] == "portfolio_update"

    @pytest.mark.asyncio
    async def test_broadcast_method(self, socketio_mgr):
        """Test generic broadcast method."""
        socketio_mgr.sio = MagicMock()
        socketio_mgr.sio.emit = AsyncMock()

        event = "test_event"
        data = {"test": "data"}
        room = "test_room"

        await socketio_mgr.broadcast(event, data, room)

        socketio_mgr.sio.emit.assert_called_with(event, data, room=room)


class TestIntegration:
    """Integration tests for Socket.IO functionality."""

    @pytest.mark.asyncio
    async def test_full_connection_flow(self):
        """Test complete connection and authentication flow."""
        namespace = TradingNamespace("/")
        sid = "integration_test_sid"

        # Mock emit to track calls
        emit_calls = []

        async def mock_emit(event, data, room=None):
            emit_calls.append((event, data, room))

        namespace.emit = mock_emit
        namespace.enter_room = AsyncMock()

        # 1. Connect without auth
        await namespace.on_connect(sid, {})
        assert any(e[0] == "welcome" for e in emit_calls)

        # 2. Authenticate
        with patch.object(namespace, "_validate_token", new_callable=AsyncMock, return_value=True):
            await namespace.on_authenticate(sid, {"token": "valid"})
        assert any(e[0] == "authenticated" and e[1]["status"] == "success" for e in emit_calls)

        # 3. Subscribe to channels
        await namespace.on_subscribe(sid, {"channels": ["market_data", "bot_status"]})
        assert any(e[0] == "subscribed" for e in emit_calls)

        # 4. Execute order
        await namespace.on_execute_order(
            sid, {"type": "limit", "symbol": "ETH/USDT", "side": "sell", "amount": 1.0}
        )
        assert any(e[0] == "order.created" for e in emit_calls)

        # 5. Disconnect
        await namespace.on_disconnect(sid)
        assert sid not in namespace.connected_clients


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in Socket.IO operations."""
    namespace = TradingNamespace("/")
    sid = "error_test_sid"

    # Test authentication without token
    with patch.object(namespace, "emit") as mock_emit:
        await namespace.on_authenticate(sid, {})
    mock_emit.assert_called_with("auth_error", {"error": "Token required"}, room=sid)

    # Test order execution without authentication
    with patch.object(namespace, "emit") as mock_emit:
        await namespace.on_execute_order(sid, {"type": "market"})
    mock_emit.assert_called_with("error", {"error": "Authentication required"}, room=sid)


@pytest.mark.asyncio
async def test_concurrent_connections():
    """Test handling multiple concurrent connections."""
    # Create a mock server for the namespace
    mock_server = unittest.mock.AsyncMock()
    mock_server.emit = unittest.mock.AsyncMock()
    mock_server.enter_room = unittest.mock.AsyncMock()

    namespace = TradingNamespace("/")
    namespace.server = mock_server  # Assign the mock server

    # Create multiple connections
    sids = [f"concurrent_sid_{i}" for i in range(10)]

    # Connect all clients concurrently
    tasks = [namespace.on_connect(sid, {}) for sid in sids]
    results = await asyncio.gather(*tasks)

    assert all(r is True for r in results)
    assert len(namespace.connected_clients) == 10

    # Disconnect all clients concurrently
    tasks = [namespace.on_disconnect(sid) for sid in sids]
    await asyncio.gather(*tasks)

    assert len(namespace.connected_clients) == 0
