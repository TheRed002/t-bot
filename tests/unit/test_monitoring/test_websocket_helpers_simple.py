"""
Simple test suite for monitoring websocket helpers module.

Basic tests to improve coverage of WebSocket connection management.
"""

from unittest.mock import Mock

import pytest

from src.monitoring.websocket_helpers import (
    WebSocketConfig,
    WebSocketManager,
    WebSocketMetrics,
    WebSocketState,
    create_websocket_config,
)


class TestWebSocketState:
    """Test WebSocket state enumeration."""
    
    def test_websocket_state_values(self):
        """Test WebSocket state enum has correct values."""
        assert WebSocketState.DISCONNECTED.value == "disconnected"
        assert WebSocketState.CONNECTING.value == "connecting"
        assert WebSocketState.CONNECTED.value == "connected"
        assert WebSocketState.RECONNECTING.value == "reconnecting"
        assert WebSocketState.ERROR.value == "error"
        assert WebSocketState.CLOSED.value == "closed"


class TestWebSocketConfig:
    """Test WebSocket configuration dataclass."""
    
    def test_websocket_config_creation(self):
        """Test WebSocket config creation with required parameters."""
        config = WebSocketConfig(url="wss://example.com")
        assert config.url == "wss://example.com"
        assert config.connect_timeout == 10.0
        assert config.heartbeat_interval == 30.0
        assert config.heartbeat_timeout == 5.0
        assert config.max_reconnect_attempts == 5
        assert config.reconnect_backoff_base == 2.0
        assert config.reconnect_backoff_max == 60.0
        assert config.message_queue_size == 1000
        assert config.enable_compression is True

    def test_websocket_config_custom_values(self):
        """Test WebSocket config creation with custom values."""
        config = WebSocketConfig(
            url="wss://test.com",
            connect_timeout=5.0,
            heartbeat_interval=60.0,
            heartbeat_timeout=10.0,
            max_reconnect_attempts=3,
            reconnect_backoff_base=1.5,
            reconnect_backoff_max=30.0,
            message_queue_size=500,
            enable_compression=False,
        )
        assert config.url == "wss://test.com"
        assert config.connect_timeout == 5.0
        assert config.heartbeat_interval == 60.0
        assert config.heartbeat_timeout == 10.0
        assert config.max_reconnect_attempts == 3
        assert config.reconnect_backoff_base == 1.5
        assert config.reconnect_backoff_max == 30.0
        assert config.message_queue_size == 500
        assert config.enable_compression is False


class TestWebSocketMetrics:
    """Test WebSocket metrics dataclass."""
    
    def test_websocket_metrics_defaults(self):
        """Test WebSocket metrics default values."""
        metrics = WebSocketMetrics()
        assert metrics.connection_time == 0.0
        assert metrics.last_message_time == 0.0
        assert metrics.messages_sent == 0
        assert metrics.messages_received == 0
        assert metrics.connection_errors == 0
        assert metrics.reconnection_attempts == 0
        assert metrics.last_heartbeat_time == 0.0
        assert metrics.last_heartbeat_latency == 0.0

    def test_websocket_metrics_custom_values(self):
        """Test WebSocket metrics with custom values."""
        metrics = WebSocketMetrics(
            connection_time=1234567890.0,
            last_message_time=1234567891.0,
            messages_sent=100,
            messages_received=200,
            connection_errors=5,
            reconnection_attempts=3,
            last_heartbeat_time=1234567892.0,
            last_heartbeat_latency=0.05,
        )
        assert metrics.connection_time == 1234567890.0
        assert metrics.last_message_time == 1234567891.0
        assert metrics.messages_sent == 100
        assert metrics.messages_received == 200
        assert metrics.connection_errors == 5
        assert metrics.reconnection_attempts == 3
        assert metrics.last_heartbeat_time == 1234567892.0
        assert metrics.last_heartbeat_latency == 0.05


class TestWebSocketManager:
    """Test WebSocket manager class."""
    
    def test_websocket_manager_initialization(self):
        """Test WebSocket manager initialization."""
        config = WebSocketConfig(url="wss://test.com")
        manager = WebSocketManager(config)
        assert manager.config == config
        assert manager.message_handler is None
        assert manager.error_handler is None
        assert manager.state == WebSocketState.DISCONNECTED
        assert isinstance(manager.metrics, WebSocketMetrics)
        assert manager._websocket is None
        assert manager._connection_task is None
        assert manager._heartbeat_task is None
        assert manager._reconnect_attempts == 0
        assert manager._last_reconnect_time == 0.0

    def test_websocket_manager_with_handlers(self):
        """Test WebSocket manager initialization with handlers."""
        config = WebSocketConfig(url="wss://test.com")
        message_handler = Mock()
        error_handler = Mock()
        
        manager = WebSocketManager(
            config,
            message_handler=message_handler,
            error_handler=error_handler
        )
        assert manager.message_handler == message_handler
        assert manager.error_handler == error_handler

    def test_websocket_manager_properties(self):
        """Test WebSocket manager properties."""
        config = WebSocketConfig(url="wss://test.com")
        manager = WebSocketManager(config)
        assert manager.state == WebSocketState.DISCONNECTED
        assert isinstance(manager.metrics, WebSocketMetrics)
        assert manager.is_connected is False

    def test_websocket_manager_is_connected_states(self):
        """Test WebSocket manager is_connected property logic."""
        config = WebSocketConfig(url="wss://test.com")
        manager = WebSocketManager(config)
        
        # Test disconnected state
        manager._state = WebSocketState.DISCONNECTED
        assert manager.is_connected is False
        
        # Test connecting state
        manager._state = WebSocketState.CONNECTING
        assert manager.is_connected is False
        
        # Test connected state
        manager._state = WebSocketState.CONNECTED
        assert manager.is_connected is True
        
        # Test error state
        manager._state = WebSocketState.ERROR
        assert manager.is_connected is False


class TestCreateWebSocketConfig:
    """Test create_websocket_config utility function."""
    
    def test_create_websocket_config_defaults(self):
        """Test creating WebSocket config with defaults."""
        config = create_websocket_config("wss://example.com")
        assert config.url == "wss://example.com"
        assert config.connect_timeout == 10.0
        assert config.heartbeat_interval == 30.0
        assert config.heartbeat_timeout == 5.0
        assert config.max_reconnect_attempts == 5
        assert config.reconnect_backoff_base == 2.0
        assert config.reconnect_backoff_max == 60.0
        assert config.message_queue_size == 1000
        assert config.enable_compression is True

    def test_create_websocket_config_custom(self):
        """Test creating WebSocket config with custom values."""
        config = create_websocket_config(
            "wss://custom.com",
            connect_timeout=5.0,
            heartbeat_interval=60.0,
            heartbeat_timeout=10.0,
            max_reconnect_attempts=3,
            reconnect_backoff_base=1.5,
            reconnect_backoff_max=30.0,
            message_queue_size=500,
            enable_compression=False,
        )
        assert config.url == "wss://custom.com"
        assert config.connect_timeout == 5.0
        assert config.heartbeat_interval == 60.0
        assert config.heartbeat_timeout == 10.0
        assert config.max_reconnect_attempts == 3
        assert config.reconnect_backoff_base == 1.5
        assert config.reconnect_backoff_max == 30.0
        assert config.message_queue_size == 500
        assert config.enable_compression is False


class TestWebSocketEdgeCases:
    """Test edge cases for WebSocket components."""
    
    def test_websocket_config_edge_values(self):
        """Test WebSocket config with edge case values."""
        config = WebSocketConfig(
            url="wss://test.com",
            connect_timeout=0.1,  # Very short timeout
            heartbeat_interval=0.1,  # Very frequent heartbeat
            heartbeat_timeout=0.01,  # Very short heartbeat timeout
            max_reconnect_attempts=100,  # Many attempts
            reconnect_backoff_base=1.1,  # Small backoff
            reconnect_backoff_max=0.5,  # Small max backoff
            message_queue_size=1,  # Tiny queue
            enable_compression=True,
        )
        
        assert config.connect_timeout == 0.1
        assert config.heartbeat_interval == 0.1
        assert config.heartbeat_timeout == 0.01
        assert config.max_reconnect_attempts == 100
        assert config.reconnect_backoff_base == 1.1
        assert config.reconnect_backoff_max == 0.5
        assert config.message_queue_size == 1

    def test_websocket_manager_name_generation(self):
        """Test WebSocket manager name generation."""
        config = WebSocketConfig(url="wss://example.com/path")
        manager = WebSocketManager(config)
        # Name should include the URL
        assert "WebSocketManager" in manager.name
        assert "example.com" in manager.name

    def test_websocket_state_enumeration_completeness(self):
        """Test WebSocket state enum contains all expected states."""
        states = list(WebSocketState)
        assert len(states) == 6
        assert WebSocketState.DISCONNECTED in states
        assert WebSocketState.CONNECTING in states
        assert WebSocketState.CONNECTED in states
        assert WebSocketState.RECONNECTING in states
        assert WebSocketState.ERROR in states
        assert WebSocketState.CLOSED in states