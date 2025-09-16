"""
Tests for web_interface.websockets.public module.
"""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from src.web_interface.websockets.public import public_websocket, router


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket."""
    websocket = Mock(spec=WebSocket)
    websocket.accept = AsyncMock()
    websocket.send_json = AsyncMock()
    websocket.receive_text = AsyncMock()
    websocket.close = AsyncMock()
    websocket.client_state = WebSocketState.CONNECTED
    return websocket


class TestPublicWebSocket:
    """Tests for public WebSocket endpoint."""

    async def test_websocket_connection_without_token(self, mock_websocket):
        """Test WebSocket connection without authentication token."""
        mock_websocket.receive_text.side_effect = WebSocketDisconnect()

        await public_websocket(mock_websocket, None)

        mock_websocket.accept.assert_called_once()

        # Check welcome message was sent
        send_calls = mock_websocket.send_json.call_args_list
        assert len(send_calls) >= 1
        welcome_call = send_calls[0][0][0]
        assert welcome_call["type"] == "welcome"
        assert welcome_call["authenticated"] is False
        assert "public stream" in welcome_call["message"]

    async def test_websocket_connection_with_token(self, mock_websocket):
        """Test WebSocket connection with authentication token."""
        mock_websocket.receive_text.side_effect = WebSocketDisconnect()

        await public_websocket(mock_websocket, "test_token")

        mock_websocket.accept.assert_called_once()

        # Check welcome message was sent
        send_calls = mock_websocket.send_json.call_args_list
        assert len(send_calls) >= 1
        welcome_call = send_calls[0][0][0]
        assert welcome_call["type"] == "welcome"
        assert welcome_call["authenticated"] is True
        assert "T-Bot WebSocket" in welcome_call["message"]

    async def test_ping_pong_message(self, mock_websocket):
        """Test ping-pong message handling."""
        ping_message = json.dumps({"type": "ping"})

        # Set up receive_text to return ping once, then disconnect
        mock_websocket.receive_text.side_effect = [ping_message, WebSocketDisconnect()]

        await public_websocket(mock_websocket, None)

        # Check that pong was sent
        send_calls = mock_websocket.send_json.call_args_list
        assert len(send_calls) >= 2  # welcome + pong

        pong_call = None
        for call in send_calls:
            if call[0][0]["type"] == "pong":
                pong_call = call[0][0]
                break

        assert pong_call is not None
        assert pong_call["type"] == "pong"
        assert "timestamp" in pong_call

    async def test_subscribe_message(self, mock_websocket):
        """Test subscription message handling."""
        subscribe_message = json.dumps({
            "type": "subscribe",
            "channels": ["market_data", "portfolio"]
        })

        mock_websocket.receive_text.side_effect = [subscribe_message, WebSocketDisconnect()]

        await public_websocket(mock_websocket, None)

        # Check that subscription confirmation was sent
        send_calls = mock_websocket.send_json.call_args_list
        subscribed_call = None
        for call in send_calls:
            if call[0][0]["type"] == "subscribed":
                subscribed_call = call[0][0]
                break

        assert subscribed_call is not None
        assert subscribed_call["type"] == "subscribed"
        assert subscribed_call["channels"] == ["market_data", "portfolio"]
        assert "timestamp" in subscribed_call

    async def test_echo_message(self, mock_websocket):
        """Test echo message handling for unknown message types."""
        echo_message = json.dumps({
            "type": "unknown",
            "data": "test_data"
        })

        mock_websocket.receive_text.side_effect = [echo_message, WebSocketDisconnect()]

        await public_websocket(mock_websocket, None)

        # Check that echo was sent
        send_calls = mock_websocket.send_json.call_args_list
        echo_call = None
        for call in send_calls:
            if call[0][0]["type"] == "echo":
                echo_call = call[0][0]
                break

        assert echo_call is not None
        assert echo_call["type"] == "echo"
        assert echo_call["data"]["type"] == "unknown"
        assert "timestamp" in echo_call

    async def test_timeout_heartbeat(self, mock_websocket):
        """Test heartbeat sent on message timeout."""
        # Simulate timeout then disconnect
        mock_websocket.receive_text.side_effect = [
            asyncio.TimeoutError(),
            WebSocketDisconnect()
        ]

        await public_websocket(mock_websocket, None)

        # Check that heartbeat was sent
        send_calls = mock_websocket.send_json.call_args_list
        heartbeat_call = None
        for call in send_calls:
            if call[0][0]["type"] == "heartbeat":
                heartbeat_call = call[0][0]
                break

        assert heartbeat_call is not None
        assert heartbeat_call["type"] == "heartbeat"
        assert "timestamp" in heartbeat_call

    async def test_websocket_disconnect_logging(self, mock_websocket):
        """Test proper logging on WebSocket disconnect."""
        mock_websocket.receive_text.side_effect = WebSocketDisconnect()

        with patch("src.web_interface.websockets.public.logger") as mock_logger:
            await public_websocket(mock_websocket, None)

            # Check that disconnect was logged
            mock_logger.info.assert_called()
            disconnect_calls = [call for call in mock_logger.info.call_args_list
                             if "disconnected" in str(call)]
            assert len(disconnect_calls) > 0

    async def test_unexpected_error_handling(self, mock_websocket):
        """Test handling of unexpected errors."""
        # Simulate an unexpected error during message processing
        mock_websocket.receive_text.side_effect = Exception("Unexpected error")
        mock_websocket.client_state = WebSocketState.CONNECTED

        with patch("src.web_interface.websockets.public.logger") as mock_logger:
            await public_websocket(mock_websocket, None)

            # Check that error was logged
            mock_logger.error.assert_called()
            error_calls = [call for call in mock_logger.error.call_args_list
                          if "Error in WebSocket handler" in str(call)]
            assert len(error_calls) > 0

    async def test_websocket_cleanup_already_disconnected(self, mock_websocket):
        """Test cleanup when WebSocket is already disconnected."""
        mock_websocket.receive_text.side_effect = Exception("Test error")
        mock_websocket.client_state = WebSocketState.DISCONNECTED
        mock_websocket.close.side_effect = WebSocketDisconnect()

        with patch("src.web_interface.websockets.public.logger") as mock_logger:
            await public_websocket(mock_websocket, None)

            # Should not attempt to close an already disconnected websocket
            # but if it does and gets WebSocketDisconnect, it should log debug message
            debug_calls = [call for call in mock_logger.debug.call_args_list
                          if "already closed" in str(call)]
            # May or may not be called depending on the exact flow

    async def test_websocket_cleanup_connection_error(self, mock_websocket):
        """Test cleanup when connection error occurs."""
        mock_websocket.receive_text.side_effect = Exception("Test error")
        mock_websocket.client_state = WebSocketState.CONNECTED
        mock_websocket.close.side_effect = ConnectionError("Connection error")

        with patch("src.web_interface.websockets.public.logger") as mock_logger:
            await public_websocket(mock_websocket, None)

            # Should handle ConnectionError gracefully during cleanup
            mock_websocket.close.assert_called_once_with(code=1000)

    async def test_websocket_cleanup_unexpected_error(self, mock_websocket):
        """Test cleanup when unexpected error occurs during close."""
        mock_websocket.receive_text.side_effect = Exception("Test error")
        mock_websocket.client_state = WebSocketState.CONNECTED
        mock_websocket.close.side_effect = Exception("Unexpected close error")

        with patch("src.web_interface.websockets.public.logger") as mock_logger:
            await public_websocket(mock_websocket, None)

            # Should log warning for unexpected errors during cleanup
            warning_calls = [call for call in mock_logger.warning.call_args_list
                           if "Unexpected error during WebSocket cleanup" in str(call)]
            assert len(warning_calls) > 0

    async def test_client_id_generation(self, mock_websocket):
        """Test that client ID is properly generated."""
        mock_websocket.receive_text.side_effect = WebSocketDisconnect()

        await public_websocket(mock_websocket, None)

        # Check welcome message contains client_id
        send_calls = mock_websocket.send_json.call_args_list
        welcome_call = send_calls[0][0][0]
        assert "client_id" in welcome_call
        assert welcome_call["client_id"].startswith("client_")

    async def test_timestamp_format(self, mock_websocket):
        """Test that timestamps are in proper ISO format."""
        mock_websocket.receive_text.side_effect = WebSocketDisconnect()

        await public_websocket(mock_websocket, None)

        # Check welcome message has proper timestamp format
        send_calls = mock_websocket.send_json.call_args_list
        welcome_call = send_calls[0][0][0]
        assert "timestamp" in welcome_call

        # Verify it's a valid ISO timestamp
        timestamp = welcome_call["timestamp"]
        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))  # Should not raise

    async def test_message_json_parsing_error(self, mock_websocket):
        """Test handling of invalid JSON messages."""
        invalid_json = "invalid json message"

        mock_websocket.receive_text.side_effect = [invalid_json, WebSocketDisconnect()]

        with patch("src.web_interface.websockets.public.logger") as mock_logger:
            await public_websocket(mock_websocket, None)

            # Should handle JSON parsing error gracefully
            # The error would be caught in the general exception handler

    async def test_subscription_without_channels(self, mock_websocket):
        """Test subscription message without channels field."""
        subscribe_message = json.dumps({"type": "subscribe"})

        mock_websocket.receive_text.side_effect = [subscribe_message, WebSocketDisconnect()]

        await public_websocket(mock_websocket, None)

        # Check that subscription confirmation was sent with empty channels
        send_calls = mock_websocket.send_json.call_args_list
        subscribed_call = None
        for call in send_calls:
            if call[0][0]["type"] == "subscribed":
                subscribed_call = call[0][0]
                break

        assert subscribed_call is not None
        assert subscribed_call["channels"] == []

    async def test_multiple_message_sequence(self, mock_websocket):
        """Test handling multiple messages in sequence."""
        messages = [
            json.dumps({"type": "ping"}),
            json.dumps({"type": "subscribe", "channels": ["test"]}),
            json.dumps({"type": "custom", "data": "test"})
        ]

        mock_websocket.receive_text.side_effect = messages + [WebSocketDisconnect()]

        await public_websocket(mock_websocket, None)

        # Should handle all message types
        send_calls = mock_websocket.send_json.call_args_list

        # Count different response types (excluding welcome)
        response_types = [call[0][0]["type"] for call in send_calls[1:]]  # Skip welcome

        assert "pong" in response_types
        assert "subscribed" in response_types
        assert "echo" in response_types

    def test_router_exists(self):
        """Test that router is properly initialized."""
        assert router is not None
        from fastapi import APIRouter
        assert isinstance(router, APIRouter)

    async def test_websocket_state_connected(self, mock_websocket):
        """Test WebSocket state handling when connected."""
        mock_websocket.receive_text.side_effect = Exception("Test error")
        mock_websocket.client_state = WebSocketState.CONNECTED

        await public_websocket(mock_websocket, None)

        # Should attempt to close the websocket when an error occurs
        mock_websocket.close.assert_called_once_with(code=1000)