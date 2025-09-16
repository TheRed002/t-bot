"""
Test cases for web_interface websockets market_data module.

This module tests the WebSocket functionality for real-time market data streaming.
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import WebSocket, status

from src.web_interface.websockets.market_data import (
    ConnectionManager,
    MarketDataMessage,
    SubscriptionMessage,
    authenticate_websocket,
    market_data_websocket,
    send_initial_market_data,
    market_data_simulator,
    get_market_data_status,
    manager
)
from src.web_interface.security.auth import User


class TestConnectionManager:
    """Test ConnectionManager functionality."""

    @pytest.fixture
    def connection_manager(self):
        """Create ConnectionManager instance."""
        return ConnectionManager()

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket."""
        websocket = AsyncMock(spec=WebSocket)
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        return websocket

    def test_init(self, connection_manager):
        """Test connection manager initialization."""
        assert isinstance(connection_manager.active_connections, dict)
        assert isinstance(connection_manager.subscriptions, dict)
        assert isinstance(connection_manager.symbol_subscribers, dict)
        assert len(connection_manager.active_connections) == 0

    async def test_connect_success(self, connection_manager, mock_websocket):
        """Test successful WebSocket connection."""
        user_id = "user123"
        
        await connection_manager.connect(mock_websocket, user_id)
        
        mock_websocket.accept.assert_called_once()
        assert connection_manager.active_connections[user_id] == mock_websocket
        assert user_id in connection_manager.subscriptions
        assert connection_manager.subscriptions[user_id] == set()

    async def test_connect_failure(self, connection_manager, mock_websocket):
        """Test WebSocket connection failure."""
        user_id = "user123"
        mock_websocket.accept.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            await connection_manager.connect(mock_websocket, user_id)

    def test_disconnect_existing_user(self, connection_manager, mock_websocket):
        """Test disconnecting existing user."""
        user_id = "user123"
        symbol = "BTCUSDT"
        
        # Set up connections and subscriptions
        connection_manager.active_connections[user_id] = mock_websocket
        connection_manager.subscriptions[user_id] = {symbol}
        connection_manager.symbol_subscribers[symbol] = {user_id}
        
        connection_manager.disconnect(user_id)
        
        assert user_id not in connection_manager.active_connections
        assert user_id not in connection_manager.subscriptions
        assert symbol not in connection_manager.symbol_subscribers

    def test_disconnect_nonexistent_user(self, connection_manager):
        """Test disconnecting non-existent user."""
        connection_manager.disconnect("nonexistent_user")
        # Should not raise exception

    def test_subscribe_to_symbol_new_user(self, connection_manager):
        """Test subscribing new user to symbol."""
        user_id = "user123"
        symbol = "BTCUSDT"
        connection_manager.subscriptions[user_id] = set()
        
        connection_manager.subscribe_to_symbol(user_id, symbol)
        
        assert symbol in connection_manager.subscriptions[user_id]
        assert symbol in connection_manager.symbol_subscribers
        assert user_id in connection_manager.symbol_subscribers[symbol]

    def test_subscribe_to_symbol_existing_symbol(self, connection_manager):
        """Test subscribing to existing symbol."""
        user_id1 = "user123"
        user_id2 = "user456"
        symbol = "BTCUSDT"
        
        connection_manager.subscriptions[user_id1] = set()
        connection_manager.subscriptions[user_id2] = set()
        connection_manager.symbol_subscribers[symbol] = {user_id1}
        
        connection_manager.subscribe_to_symbol(user_id2, symbol)
        
        assert user_id1 in connection_manager.symbol_subscribers[symbol]
        assert user_id2 in connection_manager.symbol_subscribers[symbol]

    def test_unsubscribe_from_symbol_last_user(self, connection_manager):
        """Test unsubscribing last user from symbol."""
        user_id = "user123"
        symbol = "BTCUSDT"
        
        connection_manager.subscriptions[user_id] = {symbol}
        connection_manager.symbol_subscribers[symbol] = {user_id}
        
        connection_manager.unsubscribe_from_symbol(user_id, symbol)
        
        assert symbol not in connection_manager.subscriptions[user_id]
        assert symbol not in connection_manager.symbol_subscribers

    def test_unsubscribe_from_symbol_multiple_users(self, connection_manager):
        """Test unsubscribing from symbol with multiple users."""
        user_id1 = "user123"
        user_id2 = "user456"
        symbol = "BTCUSDT"
        
        connection_manager.subscriptions[user_id1] = {symbol}
        connection_manager.subscriptions[user_id2] = {symbol}
        connection_manager.symbol_subscribers[symbol] = {user_id1, user_id2}
        
        connection_manager.unsubscribe_from_symbol(user_id1, symbol)
        
        assert symbol not in connection_manager.subscriptions[user_id1]
        assert user_id1 not in connection_manager.symbol_subscribers[symbol]
        assert user_id2 in connection_manager.symbol_subscribers[symbol]

    async def test_send_to_user_success(self, connection_manager, mock_websocket):
        """Test successful message sending to user."""
        user_id = "user123"
        message = {"type": "test", "data": "value"}
        connection_manager.active_connections[user_id] = mock_websocket
        
        await connection_manager.send_to_user(user_id, message)
        
        mock_websocket.send_text.assert_called_once_with(json.dumps(message))

    async def test_send_to_user_timeout(self, connection_manager, mock_websocket):
        """Test message sending timeout."""
        user_id = "user123"
        message = {"type": "test", "data": "value"}
        connection_manager.active_connections[user_id] = mock_websocket
        mock_websocket.send_text.side_effect = asyncio.TimeoutError()
        
        with patch.object(connection_manager, 'disconnect') as mock_disconnect:
            await connection_manager.send_to_user(user_id, message)
            mock_disconnect.assert_called_once_with(user_id)

    async def test_send_to_user_exception(self, connection_manager, mock_websocket):
        """Test message sending exception."""
        user_id = "user123"
        message = {"type": "test", "data": "value"}
        connection_manager.active_connections[user_id] = mock_websocket
        mock_websocket.send_text.side_effect = Exception("Send failed")
        
        with patch.object(connection_manager, 'disconnect') as mock_disconnect:
            await connection_manager.send_to_user(user_id, message)
            mock_disconnect.assert_called_once_with(user_id)

    async def test_send_to_user_nonexistent(self, connection_manager):
        """Test sending to non-existent user."""
        message = {"type": "test", "data": "value"}
        await connection_manager.send_to_user("nonexistent", message)
        # Should not raise exception

    async def test_broadcast_to_symbol_subscribers_success(self, connection_manager):
        """Test successful broadcast to symbol subscribers."""
        symbol = "BTCUSDT"
        user_id1 = "user123"
        user_id2 = "user456"
        message = {"type": "ticker_update", "symbol": symbol}
        
        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()
        
        connection_manager.active_connections[user_id1] = mock_websocket1
        connection_manager.active_connections[user_id2] = mock_websocket2
        connection_manager.symbol_subscribers[symbol] = {user_id1, user_id2}
        
        with patch.object(connection_manager, '_send_with_timeout_broadcast', return_value=None) as mock_send:
            await connection_manager.broadcast_to_symbol_subscribers(symbol, message)
            assert mock_send.call_count == 2

    async def test_broadcast_to_symbol_subscribers_no_subscribers(self, connection_manager):
        """Test broadcast with no subscribers."""
        symbol = "BTCUSDT"
        message = {"type": "ticker_update", "symbol": symbol}
        
        await connection_manager.broadcast_to_symbol_subscribers(symbol, message)
        # Should not raise exception

    async def test_broadcast_to_symbol_subscribers_with_exception(self, connection_manager):
        """Test broadcast with send exceptions."""
        symbol = "BTCUSDT"
        user_id = "user123"
        message = {"type": "ticker_update", "symbol": symbol}
        
        mock_websocket = AsyncMock()
        connection_manager.active_connections[user_id] = mock_websocket
        connection_manager.symbol_subscribers[symbol] = {user_id}
        
        with patch.object(connection_manager, '_send_with_timeout_broadcast',
                         side_effect=Exception("Send failed")) as mock_send, \
             patch.object(connection_manager, 'disconnect') as mock_disconnect:
            await connection_manager.broadcast_to_symbol_subscribers(symbol, message)
            mock_disconnect.assert_called_once_with(user_id)

    async def test_send_with_timeout_success(self, connection_manager):
        """Test successful send with timeout."""
        mock_websocket = AsyncMock()
        message = {"type": "test"}
        user_id = "user123"
        
        await connection_manager._send_with_timeout(mock_websocket, message, user_id)
        mock_websocket.send_text.assert_called_once_with(json.dumps(message))

    async def test_send_with_timeout_exception(self, connection_manager):
        """Test send with timeout exception."""
        mock_websocket = AsyncMock()
        mock_websocket.send_text.side_effect = Exception("Send failed")
        message = {"type": "test"}
        user_id = "user123"
        
        with pytest.raises(Exception, match="Send failed"):
            await connection_manager._send_with_timeout(mock_websocket, message, user_id)


class TestMarketDataMessage:
    """Test MarketDataMessage model."""

    def test_market_data_message_valid(self):
        """Test valid MarketDataMessage creation."""
        message = MarketDataMessage(
            type="ticker",
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            data={"price": 45000.0}
        )
        
        assert message.type == "ticker"
        assert message.symbol == "BTCUSDT"
        assert isinstance(message.timestamp, datetime)
        assert message.data == {"price": 45000.0}


class TestSubscriptionMessage:
    """Test SubscriptionMessage model."""

    def test_subscription_message_defaults(self):
        """Test SubscriptionMessage with defaults."""
        message = SubscriptionMessage(
            action="subscribe",
            symbols=["BTCUSDT"]
        )
        
        assert message.action == "subscribe"
        assert message.symbols == ["BTCUSDT"]
        assert message.data_types == ["ticker", "orderbook", "trades"]

    def test_subscription_message_custom(self):
        """Test SubscriptionMessage with custom data types."""
        message = SubscriptionMessage(
            action="unsubscribe",
            symbols=["BTCUSDT", "ETHUSDT"],
            data_types=["ticker"]
        )
        
        assert message.action == "unsubscribe"
        assert message.symbols == ["BTCUSDT", "ETHUSDT"]
        assert message.data_types == ["ticker"]


class TestAuthenticateWebSocket:
    """Test WebSocket authentication function."""

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket."""
        websocket = Mock(spec=WebSocket)
        websocket.query_params = {}
        websocket.headers = {}
        websocket.close = AsyncMock()
        return websocket

    @pytest.fixture
    def mock_jwt_handler(self):
        """Create mock JWT handler."""
        handler = Mock()
        handler.validate_token = Mock()
        return handler

    @pytest.fixture
    def mock_token_data(self):
        """Create mock token data."""
        token_data = Mock()
        token_data.user_id = "user123"
        token_data.username = "testuser"
        token_data.scopes = ["read", "write"]
        return token_data

    async def test_authenticate_websocket_query_param_success(self, mock_websocket, mock_jwt_handler, mock_token_data):
        """Test successful authentication with query parameter token."""
        mock_websocket.query_params = {"token": "valid_token"}
        mock_jwt_handler.validate_token.return_value = mock_token_data
        
        with patch("src.web_interface.security.auth.jwt_handler", mock_jwt_handler):
            user = await authenticate_websocket(mock_websocket)
            
        assert isinstance(user, User)
        assert user.user_id == "user123"
        assert user.username == "testuser"

    async def test_authenticate_websocket_header_success(self, mock_websocket, mock_jwt_handler, mock_token_data):
        """Test successful authentication with Authorization header."""
        mock_websocket.headers = {"authorization": "Bearer valid_token"}
        mock_jwt_handler.validate_token.return_value = mock_token_data
        
        with patch("src.web_interface.security.auth.jwt_handler", mock_jwt_handler):
            user = await authenticate_websocket(mock_websocket)
            
        assert isinstance(user, User)
        assert user.user_id == "user123"

    async def test_authenticate_websocket_no_token(self, mock_websocket):
        """Test authentication failure with no token."""
        user = await authenticate_websocket(mock_websocket)
        
        assert user is None
        mock_websocket.close.assert_called_once_with(
            code=status.WS_1008_POLICY_VIOLATION, reason="Authentication required"
        )

    async def test_authenticate_websocket_invalid_token(self, mock_websocket, mock_jwt_handler):
        """Test authentication failure with invalid token."""
        mock_websocket.query_params = {"token": "invalid_token"}
        mock_jwt_handler.validate_token.return_value = None
        
        with patch("src.web_interface.security.auth.jwt_handler", mock_jwt_handler):
            user = await authenticate_websocket(mock_websocket)
            
        assert user is None
        mock_websocket.close.assert_called_once_with(
            code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed"
        )

    async def test_authenticate_websocket_async_validate(self, mock_websocket, mock_jwt_handler, mock_token_data):
        """Test authentication with async validate_token."""
        mock_websocket.query_params = {"token": "valid_token"}
        mock_jwt_handler.validate_token = AsyncMock(return_value=mock_token_data)
        
        with patch("src.web_interface.security.auth.jwt_handler", mock_jwt_handler):
            user = await authenticate_websocket(mock_websocket)
            
        assert isinstance(user, User)
        assert user.user_id == "user123"

    async def test_authenticate_websocket_exception(self, mock_websocket):
        """Test authentication exception handling."""
        mock_websocket.query_params = {"token": "error_token"}
        
        with patch("src.web_interface.security.auth.jwt_handler", side_effect=Exception("JWT error")):
            user = await authenticate_websocket(mock_websocket)
            
        assert user is None
        mock_websocket.close.assert_called_once_with(
            code=status.WS_1008_POLICY_VIOLATION, reason="Authentication error"
        )


class TestSendInitialMarketData:
    """Test send_initial_market_data function."""

    @pytest.fixture
    def mock_manager(self):
        """Create mock connection manager."""
        manager = Mock()
        manager.send_to_user = AsyncMock()
        return manager

    async def test_send_initial_market_data_ticker(self, mock_manager):
        """Test sending initial ticker data."""
        user_id = "user123"
        symbol = "BTCUSDT"
        data_types = ["ticker"]
        
        with patch("src.web_interface.websockets.market_data.manager", mock_manager):
            await send_initial_market_data(user_id, symbol, data_types)
            
        mock_manager.send_to_user.assert_called_once()
        call_args = mock_manager.send_to_user.call_args
        assert call_args[0][0] == user_id
        message = call_args[0][1]
        assert message["type"] == "ticker"
        assert message["symbol"] == symbol

    async def test_send_initial_market_data_orderbook(self, mock_manager):
        """Test sending initial orderbook data."""
        user_id = "user123"
        symbol = "BTCUSDT"
        data_types = ["orderbook"]
        
        with patch("src.web_interface.websockets.market_data.manager", mock_manager):
            await send_initial_market_data(user_id, symbol, data_types)
            
        mock_manager.send_to_user.assert_called_once()
        message = mock_manager.send_to_user.call_args[0][1]
        assert message["type"] == "orderbook"
        assert "bids" in message["data"]
        assert "asks" in message["data"]

    async def test_send_initial_market_data_trades(self, mock_manager):
        """Test sending initial trades data."""
        user_id = "user123"
        symbol = "BTCUSDT"
        data_types = ["trades"]
        
        with patch("src.web_interface.websockets.market_data.manager", mock_manager):
            await send_initial_market_data(user_id, symbol, data_types)
            
        mock_manager.send_to_user.assert_called_once()
        message = mock_manager.send_to_user.call_args[0][1]
        assert message["type"] == "trades"
        assert "trades" in message["data"]

    async def test_send_initial_market_data_all_types(self, mock_manager):
        """Test sending all data types."""
        user_id = "user123"
        symbol = "BTCUSDT"
        data_types = ["ticker", "orderbook", "trades"]
        
        with patch("src.web_interface.websockets.market_data.manager", mock_manager):
            await send_initial_market_data(user_id, symbol, data_types)
            
        assert mock_manager.send_to_user.call_count == 3

    async def test_send_initial_market_data_exception(self, mock_manager):
        """Test send initial market data exception handling."""
        user_id = "user123"
        symbol = "BTCUSDT"
        data_types = ["ticker"]
        mock_manager.send_to_user.side_effect = Exception("Send failed")
        
        with patch("src.web_interface.websockets.market_data.manager", mock_manager):
            # Should not raise exception
            await send_initial_market_data(user_id, symbol, data_types)


class TestMarketDataSimulator:
    """Test market_data_simulator function."""

    async def test_market_data_simulator_with_subscribers(self):
        """Test market data simulator with active subscribers."""
        mock_manager = Mock()
        mock_manager.symbol_subscribers = {"BTCUSDT": {"user123"}}
        mock_manager.broadcast_to_symbol_subscribers = AsyncMock()
        
        # Run one iteration
        with patch("src.web_interface.websockets.market_data.manager", mock_manager), \
             patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError()]):
            
            try:
                await market_data_simulator()
            except asyncio.CancelledError:
                pass
        
        mock_manager.broadcast_to_symbol_subscribers.assert_called()

    async def test_market_data_simulator_no_subscribers(self):
        """Test market data simulator with no subscribers."""
        mock_manager = Mock()
        mock_manager.symbol_subscribers = {}
        mock_manager.broadcast_to_symbol_subscribers = AsyncMock()
        
        # Run one iteration
        with patch("src.web_interface.websockets.market_data.manager", mock_manager), \
             patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError()]):
            
            try:
                await market_data_simulator()
            except asyncio.CancelledError:
                pass
        
        mock_manager.broadcast_to_symbol_subscribers.assert_not_called()

    async def test_market_data_simulator_exception_handling(self):
        """Test market data simulator exception handling."""
        mock_manager = Mock()
        mock_manager.symbol_subscribers.get.side_effect = Exception("Simulator error")
        
        # Run one iteration with exception, then cancel
        with patch("src.web_interface.websockets.market_data.manager", mock_manager), \
             patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError()]):
            
            try:
                await market_data_simulator()
            except asyncio.CancelledError:
                pass
        
        # Should handle exception gracefully


class TestGetMarketDataStatus:
    """Test get_market_data_status function."""

    async def test_get_market_data_status(self):
        """Test getting market data status."""
        mock_manager = Mock()
        mock_manager.active_connections = {"user1": Mock(), "user2": Mock()}
        mock_manager.subscriptions = {
            "user1": {"BTCUSDT", "ETHUSDT"},
            "user2": {"BTCUSDT"}
        }
        mock_manager.symbol_subscribers = {
            "BTCUSDT": {"user1", "user2"},
            "ETHUSDT": {"user1"}
        }
        
        with patch("src.web_interface.websockets.market_data.manager", mock_manager):
            status = await get_market_data_status()
        
        assert status["active_connections"] == 2
        assert status["total_subscriptions"] == 3
        assert set(status["symbols_with_subscribers"]) == {"BTCUSDT", "ETHUSDT"}
        assert "connection_details" in status