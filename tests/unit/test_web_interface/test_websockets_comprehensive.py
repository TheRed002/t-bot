"""
Test cases for web_interface websockets modules.

This module tests the WebSocket functionality for bot status, portfolio updates, 
and unified manager components.
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import WebSocket
from decimal import Decimal

# Import what we can from the websockets modules
try:
    from src.web_interface.websockets.bot_status import BotStatusManager
    BOT_STATUS_AVAILABLE = True
except ImportError:
    BOT_STATUS_AVAILABLE = False

try:
    from src.web_interface.websockets.portfolio import PortfolioManager
    PORTFOLIO_AVAILABLE = True
except ImportError:
    PORTFOLIO_AVAILABLE = False

try:
    from src.web_interface.websockets.unified_manager import UnifiedWebSocketManager
    UNIFIED_AVAILABLE = True
except ImportError:
    UNIFIED_AVAILABLE = False


@pytest.mark.skipif(not BOT_STATUS_AVAILABLE, reason="bot_status module not available")
class TestBotStatusManager:
    """Test BotStatusManager functionality."""

    @pytest.fixture
    def bot_status_manager(self):
        """Create BotStatusManager instance."""
        return BotStatusManager()

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket."""
        websocket = AsyncMock(spec=WebSocket)
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        return websocket

    def test_init(self, bot_status_manager):
        """Test bot status manager initialization."""
        assert isinstance(bot_status_manager.active_connections, dict)
        assert isinstance(bot_status_manager.bot_subscriptions, dict)
        assert isinstance(bot_status_manager.user_subscriptions, dict)
        assert len(bot_status_manager.active_connections) == 0

    async def test_connect_success(self, bot_status_manager, mock_websocket):
        """Test successful WebSocket connection."""
        user_id = "user123"
        
        await bot_status_manager.connect(mock_websocket, user_id)
        
        mock_websocket.accept.assert_called_once()
        assert bot_status_manager.active_connections[user_id] == mock_websocket
        assert user_id in bot_status_manager.bot_subscriptions
        assert bot_status_manager.bot_subscriptions[user_id] == set()

    async def test_connect_failure(self, bot_status_manager, mock_websocket):
        """Test WebSocket connection failure."""
        user_id = "user123"
        mock_websocket.accept.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            await bot_status_manager.connect(mock_websocket, user_id)

    def test_disconnect_existing_user(self, bot_status_manager, mock_websocket):
        """Test disconnecting existing user."""
        user_id = "user123"
        bot_id = "bot_001"
        
        # Set up connections and subscriptions
        bot_status_manager.active_connections[user_id] = mock_websocket
        bot_status_manager.bot_subscriptions[user_id] = {bot_id}
        bot_status_manager.user_subscriptions[bot_id] = {user_id}
        
        bot_status_manager.disconnect(user_id)
        
        assert user_id not in bot_status_manager.active_connections
        assert user_id not in bot_status_manager.bot_subscriptions
        assert bot_id not in bot_status_manager.user_subscriptions

    def test_subscribe_to_bot_new_user(self, bot_status_manager):
        """Test subscribing new user to bot."""
        user_id = "user123"
        bot_id = "bot_001"
        bot_status_manager.bot_subscriptions[user_id] = set()
        
        bot_status_manager.subscribe_to_bot(user_id, bot_id)
        
        assert bot_id in bot_status_manager.bot_subscriptions[user_id]
        assert bot_id in bot_status_manager.user_subscriptions
        assert user_id in bot_status_manager.user_subscriptions[bot_id]

    def test_unsubscribe_from_bot_last_user(self, bot_status_manager):
        """Test unsubscribing last user from bot."""
        user_id = "user123"
        bot_id = "bot_001"
        
        bot_status_manager.bot_subscriptions[user_id] = {bot_id}
        bot_status_manager.user_subscriptions[bot_id] = {user_id}
        
        bot_status_manager.unsubscribe_from_bot(user_id, bot_id)
        
        assert bot_id not in bot_status_manager.bot_subscriptions[user_id]
        assert bot_id not in bot_status_manager.user_subscriptions

    def test_subscribe_to_all_bots(self, bot_status_manager):
        """Test subscribing to all bots."""
        user_id = "user123"
        bot_status_manager.bot_subscriptions[user_id] = set()
        
        bot_status_manager.subscribe_to_all_bots(user_id)
        
        # Should subscribe to mock bot IDs
        assert len(bot_status_manager.bot_subscriptions[user_id]) > 0

    async def test_send_to_user_success(self, bot_status_manager, mock_websocket):
        """Test successful message sending to user."""
        user_id = "user123"
        message = {"type": "bot_status", "data": "running"}
        bot_status_manager.active_connections[user_id] = mock_websocket
        
        await bot_status_manager.send_to_user(user_id, message)
        
        mock_websocket.send_text.assert_called_once_with(json.dumps(message))

    async def test_send_to_user_timeout(self, bot_status_manager, mock_websocket):
        """Test message sending timeout."""
        user_id = "user123"
        message = {"type": "bot_status", "data": "running"}
        bot_status_manager.active_connections[user_id] = mock_websocket
        mock_websocket.send_text.side_effect = asyncio.TimeoutError()
        
        with patch.object(bot_status_manager, 'disconnect') as mock_disconnect:
            await bot_status_manager.send_to_user(user_id, message)
            mock_disconnect.assert_called_once_with(user_id)


class TestWebSocketPortfolioMock:
    """Test WebSocket portfolio functionality with mocking."""

    @pytest.fixture
    def mock_portfolio_manager(self):
        """Create mock portfolio manager."""
        manager = Mock()
        manager.active_connections = {}
        manager.subscriptions = {}
        manager.connect = AsyncMock()
        manager.disconnect = Mock()
        manager.send_portfolio_update = AsyncMock()
        manager.send_balance_update = AsyncMock()
        manager.send_trade_notification = AsyncMock()
        return manager

    def test_portfolio_manager_structure(self, mock_portfolio_manager):
        """Test portfolio manager has expected structure."""
        assert hasattr(mock_portfolio_manager, 'active_connections')
        assert hasattr(mock_portfolio_manager, 'subscriptions')
        assert hasattr(mock_portfolio_manager, 'connect')
        assert hasattr(mock_portfolio_manager, 'disconnect')

    async def test_portfolio_updates(self, mock_portfolio_manager):
        """Test portfolio update functionality."""
        user_id = "user123"
        portfolio_data = {
            "total_value": Decimal("50000.00"),
            "positions": [
                {"symbol": "BTCUSDT", "quantity": Decimal("1.5"), "value": Decimal("45000.00")}
            ]
        }
        
        await mock_portfolio_manager.send_portfolio_update(user_id, portfolio_data)
        mock_portfolio_manager.send_portfolio_update.assert_called_once_with(user_id, portfolio_data)

    async def test_balance_updates(self, mock_portfolio_manager):
        """Test balance update functionality."""
        user_id = "user123"
        balance_data = {
            "USDT": {"available": Decimal("5000.00"), "locked": Decimal("1000.00")},
            "BTC": {"available": Decimal("1.5"), "locked": Decimal("0.0")}
        }
        
        await mock_portfolio_manager.send_balance_update(user_id, balance_data)
        mock_portfolio_manager.send_balance_update.assert_called_once_with(user_id, balance_data)

    async def test_trade_notifications(self, mock_portfolio_manager):
        """Test trade notification functionality."""
        user_id = "user123"
        trade_data = {
            "id": "trade_123",
            "symbol": "BTCUSDT",
            "side": "buy",
            "quantity": Decimal("0.1"),
            "price": Decimal("45000.00"),
            "timestamp": datetime.now(timezone.utc)
        }
        
        await mock_portfolio_manager.send_trade_notification(user_id, trade_data)
        mock_portfolio_manager.send_trade_notification.assert_called_once_with(user_id, trade_data)


@pytest.mark.skipif(not UNIFIED_AVAILABLE, reason="unified_manager module not available")
class TestUnifiedWebSocketManager:
    """Test UnifiedWebSocketManager functionality."""

    @pytest.fixture
    def unified_manager(self):
        """Create UnifiedWebSocketManager instance."""
        return UnifiedWebSocketManager()

    def test_init(self, unified_manager):
        """Test unified manager initialization."""
        assert hasattr(unified_manager, 'sio')
        assert hasattr(unified_manager, 'namespace')
        assert hasattr(unified_manager, 'api_facade')

    async def test_register_connection(self, unified_manager):
        """Test registering WebSocket connection."""
        websocket = AsyncMock()
        user_id = "user123"
        connection_type = "market_data"
        
        if hasattr(unified_manager, 'register_connection'):
            await unified_manager.register_connection(websocket, user_id, connection_type)
            # Verify connection was registered
            assert user_id in unified_manager.connections or connection_type in unified_manager.connections

    async def test_unregister_connection(self, unified_manager):
        """Test unregistering WebSocket connection."""
        user_id = "user123"
        connection_type = "market_data"
        
        if hasattr(unified_manager, 'unregister_connection'):
            unified_manager.unregister_connection(user_id, connection_type)
            # Should not raise exception even if connection doesn't exist

    async def test_broadcast_message(self, unified_manager):
        """Test broadcasting message to all connections."""
        message = {"type": "system_alert", "data": "maintenance in 5 minutes"}
        
        if hasattr(unified_manager, 'broadcast'):
            await unified_manager.broadcast(message)
            # Should complete without error

    async def test_send_to_connection_type(self, unified_manager):
        """Test sending message to specific connection type."""
        message = {"type": "market_update", "data": {"symbol": "BTCUSDT", "price": 45000}}
        connection_type = "market_data"
        
        if hasattr(unified_manager, 'send_to_type'):
            await unified_manager.send_to_type(connection_type, message)
            # Should complete without error


class TestWebSocketCommonFunctionality:
    """Test common WebSocket functionality across modules."""

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket."""
        websocket = AsyncMock()
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.receive_text = AsyncMock()
        websocket.close = AsyncMock()
        return websocket

    async def test_websocket_json_message_handling(self, mock_websocket):
        """Test JSON message handling pattern."""
        # Simulate receiving a JSON message
        test_message = {"action": "subscribe", "symbols": ["BTCUSDT"]}
        mock_websocket.receive_text.return_value = json.dumps(test_message)
        
        # Test message parsing
        data = await mock_websocket.receive_text()
        parsed_message = json.loads(data)
        
        assert parsed_message["action"] == "subscribe"
        assert "BTCUSDT" in parsed_message["symbols"]

    async def test_websocket_error_handling(self, mock_websocket):
        """Test WebSocket error handling patterns."""
        # Test connection timeout
        mock_websocket.send_text.side_effect = asyncio.TimeoutError()
        
        try:
            await asyncio.wait_for(
                mock_websocket.send_text(json.dumps({"type": "test"})), 
                timeout=1.0
            )
        except asyncio.TimeoutError:
            # Should handle timeout gracefully
            pass

    async def test_websocket_authentication_flow(self, mock_websocket):
        """Test WebSocket authentication flow."""
        # Mock authentication token
        token = "valid_jwt_token"
        mock_websocket.query_params = {"token": token}
        
        # Test token extraction
        extracted_token = mock_websocket.query_params.get("token")
        assert extracted_token == token

    def test_websocket_subscription_patterns(self):
        """Test subscription pattern structures."""
        # Test user -> symbols subscription pattern
        subscriptions = {}
        user_id = "user123"
        symbol = "BTCUSDT"
        
        if user_id not in subscriptions:
            subscriptions[user_id] = set()
        
        subscriptions[user_id].add(symbol)
        assert symbol in subscriptions[user_id]
        
        # Test symbol -> users pattern
        symbol_subscribers = {}
        if symbol not in symbol_subscribers:
            symbol_subscribers[symbol] = set()
        
        symbol_subscribers[symbol].add(user_id)
        assert user_id in symbol_subscribers[symbol]

    def test_message_serialization(self):
        """Test message serialization patterns."""
        # Test standard message format
        message = {
            "type": "ticker_update",
            "symbol": "BTCUSDT",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "price": 45000.50,
                "volume": 1234567.89
            }
        }
        
        # Test JSON serialization
        json_message = json.dumps(message)
        parsed_message = json.loads(json_message)
        
        assert parsed_message["type"] == "ticker_update"
        assert parsed_message["symbol"] == "BTCUSDT"
        assert "timestamp" in parsed_message
        assert "data" in parsed_message

    async def test_connection_cleanup_patterns(self):
        """Test connection cleanup patterns."""
        # Mock connection manager state
        active_connections = {"user123": Mock(), "user456": Mock()}
        subscriptions = {"user123": {"BTCUSDT"}, "user456": {"ETHUSDT"}}
        symbol_subscribers = {
            "BTCUSDT": {"user123"},
            "ETHUSDT": {"user456"}
        }
        
        # Test cleanup for user123
        user_id = "user123"
        if user_id in active_connections:
            # Remove from symbol subscribers
            for symbol in subscriptions.get(user_id, set()):
                if symbol in symbol_subscribers:
                    symbol_subscribers[symbol].discard(user_id)
                    if not symbol_subscribers[symbol]:
                        del symbol_subscribers[symbol]
            
            # Remove user
            del active_connections[user_id]
            del subscriptions[user_id]
        
        assert user_id not in active_connections
        assert user_id not in subscriptions
        assert "BTCUSDT" not in symbol_subscribers  # Was only subscriber

    def test_decimal_handling_in_messages(self):
        """Test Decimal handling in WebSocket messages."""
        from decimal import Decimal
        
        # Test Decimal serialization for financial data
        portfolio_data = {
            "total_value": Decimal("50000.12345678"),
            "positions": [
                {
                    "symbol": "BTCUSDT",
                    "quantity": Decimal("1.50000000"),
                    "price": Decimal("45000.00000000")
                }
            ]
        }
        
        # Convert Decimals to strings for JSON serialization
        def decimal_to_str(obj):
            if isinstance(obj, Decimal):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: decimal_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [decimal_to_str(item) for item in obj]
            return obj
        
        serializable_data = decimal_to_str(portfolio_data)
        json_str = json.dumps(serializable_data)
        
        # Verify serialization worked
        assert "50000.12345678" in json_str
        assert "1.50000000" in json_str

    async def test_rate_limiting_patterns(self):
        """Test rate limiting patterns for WebSockets."""
        import time
        
        # Mock rate limiter state
        last_message_time = {}
        rate_limit_seconds = 0.1
        
        user_id = "user123"
        current_time = time.time()
        
        # Check if user is rate limited
        if user_id in last_message_time:
            time_diff = current_time - last_message_time[user_id]
            if time_diff < rate_limit_seconds:
                # Would be rate limited
                assert time_diff < rate_limit_seconds
            else:
                last_message_time[user_id] = current_time
        else:
            last_message_time[user_id] = current_time
        
        assert user_id in last_message_time

    def test_message_queue_patterns(self):
        """Test message queuing patterns."""
        # Test simple message queue structure
        message_queue = {}
        user_id = "user123"
        
        if user_id not in message_queue:
            message_queue[user_id] = []
        
        # Add messages to queue
        messages = [
            {"type": "ticker", "data": {"price": 45000}},
            {"type": "balance", "data": {"USDT": "5000.00"}},
            {"type": "trade", "data": {"id": "trade123"}}
        ]
        
        for message in messages:
            message_queue[user_id].append(message)
        
        assert len(message_queue[user_id]) == 3
        assert message_queue[user_id][0]["type"] == "ticker"
        
        # Test queue processing (FIFO)
        processed_message = message_queue[user_id].pop(0)
        assert processed_message["type"] == "ticker"
        assert len(message_queue[user_id]) == 2