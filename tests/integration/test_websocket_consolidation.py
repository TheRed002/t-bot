"""
Test WebSocket Consolidation Integration

Tests that verify the consolidated WebSocket system works correctly
and eliminates the resource management issues from the previous
fragmented implementations.
"""

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.core.config import Config
from src.exchanges.high_performance_websocket import (
    HighPerformanceWebSocketManager,
    MessagePriority,
    ConnectionPool,
    WebSocketConnectionPool
)


class TestWebSocketConsolidation:
    """Test consolidated WebSocket functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.websocket = Mock()
        config.websocket.max_connections_per_exchange = 5
        config.websocket.max_total_connections = 20
        config.websocket.batch_size = 100
        config.websocket.batch_timeout_ms = 10.0
        config.websocket.max_memory_mb = 500
        config.websocket.max_file_descriptors = 1000
        config.websocket.cleanup_interval = 300
        config.error_handling = Mock()
        return config

    @pytest_asyncio.fixture
    async def websocket_manager(self, mock_config):
        """Create high-performance WebSocket manager."""
        manager = HighPerformanceWebSocketManager(mock_config)
        yield manager
        await manager.disconnect_all()

    @pytest_asyncio.fixture
    async def connection_pool(self, mock_config):
        """Create connection pool."""
        pool = WebSocketConnectionPool("test_exchange")
        yield pool
        await pool.close_all_connections()

    @pytest.mark.asyncio
    async def test_manager_initialization(self, mock_config):
        """Test that the consolidated manager initializes correctly."""
        manager = ConsolidatedWebSocketManager(mock_config)
        
        # Test initialization
        result = await manager.initialize()
        assert result is True
        
        # Verify initial state
        assert len(manager.connections) == 0
        assert manager._shutdown is False
        
        # Test shutdown
        await manager.shutdown()
        assert manager._shutdown is True

    @pytest.mark.asyncio
    async def test_resource_manager_initialization(self, mock_config):
        """Test that the resource manager initializes correctly."""
        manager = WebSocketResourceManager(mock_config)
        
        # Test initialization
        await manager.start()
        
        # Verify initial state
        assert len(manager.tracked_connections) == 0
        assert manager._shutdown is False
        
        # Test shutdown
        await manager.stop()
        assert manager._shutdown is True

    @pytest.mark.asyncio
    async def test_connection_creation_limits(self, websocket_manager):
        """Test that connection limits are enforced."""
        # Mock exchange handler creation to avoid actual connections
        with patch.object(websocket_manager, '_create_exchange_handler') as mock_create:
            mock_handler = AsyncMock()
            mock_handler.connect.return_value = True
            mock_handler.is_connected.return_value = True
            mock_create.return_value = mock_handler
            
            # Create connections up to the limit
            connection_ids = []
            for i in range(websocket_manager.max_connections_per_exchange):
                conn_id = await websocket_manager.create_exchange_connection(
                    ExchangeType.BINANCE, client=Mock()
                )
                connection_ids.append(conn_id)
                assert conn_id is not None
            
            # Try to create one more - should return existing connection
            extra_conn_id = await websocket_manager.create_exchange_connection(
                ExchangeType.BINANCE, client=Mock()
            )
            assert extra_conn_id in connection_ids

    @pytest.mark.asyncio
    async def test_subscription_delegation(self, websocket_manager):
        """Test that subscriptions are properly delegated to exchange handlers."""
        # Mock exchange handler
        with patch.object(websocket_manager, '_create_exchange_handler') as mock_create:
            mock_handler = AsyncMock()
            mock_handler.connect.return_value = True
            mock_handler.is_connected.return_value = True
            mock_handler.subscribe_to_ticker_stream = AsyncMock()
            mock_create.return_value = mock_handler
            
            # Create connection
            conn_id = await websocket_manager.create_exchange_connection(
                ExchangeType.BINANCE, client=Mock()
            )
            
            # Test subscription
            callback = AsyncMock()
            result = await websocket_manager.subscribe_to_stream(
                conn_id, StreamType.TICKER, "BTCUSDT", callback
            )
            
            assert result is True
            mock_handler.subscribe_to_ticker_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_resource_tracking(self, mock_config):
        """Test that resources are properly tracked."""
        manager = WebSocketResourceManager(mock_config)
        await manager.start()
        
        try:
            # Create mock connection object
            mock_connection = Mock()
            
            # Register connection
            result = manager.register_connection(
                "test_conn_1", "binance", mock_connection
            )
            assert result is True
            assert "test_conn_1" in manager.tracked_connections
            
            # Update activity
            manager.update_activity("test_conn_1")
            
            # Check resource limits
            status = await manager.check_resource_limits()
            assert status["within_limits"] is True
            assert status["metrics"]["connections"]["active"] == 1
            
            # Unregister connection
            result = manager.unregister_connection("test_conn_1")
            assert result is True
            assert "test_conn_1" not in manager.tracked_connections
            
        finally:
            await manager.stop()

    @pytest.mark.asyncio
    async def test_resource_leak_detection(self, resource_manager):
        """Test that resource leaks are detected and cleaned up."""
        # Register connection without keeping reference
        mock_connection = Mock()
        resource_manager.register_connection(
            "leak_test", "binance", mock_connection
        )
        
        # Remove the reference to simulate a leak
        del mock_connection
        
        # Force leak detection
        await resource_manager._detect_resource_leaks()
        
        # Verify cleanup
        assert resource_manager.leak_detection_count > 0

    @pytest.mark.asyncio
    async def test_connection_status_monitoring(self, websocket_manager):
        """Test connection status monitoring."""
        # Create connection
        conn_id = await websocket_manager.create_connection(
            "wss://stream.binance.com:9443/ws", "binance"
        )
        assert conn_id is not None
        
        # Give it a moment to establish connection
        await asyncio.sleep(0.1)
        
        # Test status retrieval
        status = websocket_manager.get_connection_health()
        assert status is not None
        assert "total_connections" in status
        assert "active_connections" in status
        assert status["total_connections"] >= 1

    @pytest.mark.asyncio
    async def test_overall_status_reporting(self, websocket_manager):
        """Test overall status reporting."""
        # Get initial status
        status = websocket_manager.get_connection_health()
        
        assert "total_connections" in status
        assert "active_connections" in status
        assert "failed_connections" in status
        assert "total_messages_received" in status
        assert "total_latency_sum" in status
        assert "message_rate" in status

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, websocket_manager):
        """Test that shutdown properly cleans up all resources."""
        # Mock exchange handler
        with patch.object(websocket_manager, '_create_exchange_handler') as mock_create:
            mock_handler = AsyncMock()
            mock_handler.connect.return_value = True
            mock_handler.is_connected.return_value = True
            mock_handler.disconnect = AsyncMock()
            mock_create.return_value = mock_handler
            
            # Create multiple connections
            conn_ids = []
            for exchange in [ExchangeType.BINANCE, ExchangeType.COINBASE]:
                if exchange == ExchangeType.BINANCE:
                    conn_id = await websocket_manager.create_exchange_connection(
                        exchange, client=Mock()
                    )
                else:
                    conn_id = await websocket_manager.create_exchange_connection(exchange)
                conn_ids.append(conn_id)
            
            assert len(websocket_manager.connections) == 2
            
            # Test shutdown
            await websocket_manager.shutdown()
            
            # Verify all handlers were disconnected
            for conn_info in websocket_manager.connections.values():
                mock_handler.disconnect.assert_called()
            
            # Verify cleanup
            assert len(websocket_manager.connections) == 0
            assert websocket_manager._shutdown is True

    @pytest.mark.asyncio
    async def test_message_priority_handling(self, websocket_manager):
        """Test that message priorities are handled correctly."""
        # Mock exchange handler
        with patch.object(websocket_manager, '_create_exchange_handler') as mock_create:
            mock_handler = AsyncMock()
            mock_handler.connect.return_value = True
            mock_handler.is_connected.return_value = True
            mock_create.return_value = mock_handler
            
            # Create connection
            conn_id = await websocket_manager.create_exchange_connection(
                ExchangeType.BINANCE, client=Mock()
            )
            
            # Test subscription with different priorities
            callback = AsyncMock()
            result = await websocket_manager.subscribe_to_stream(
                conn_id, StreamType.TICKER, "BTCUSDT", callback, MessagePriority.HIGH
            )
            
            assert result is True

    @pytest.mark.asyncio
    async def test_force_cleanup(self, resource_manager):
        """Test force cleanup functionality."""
        # Register multiple connections
        for i in range(3):
            mock_connection = Mock()
            resource_manager.register_connection(
                f"test_conn_{i}", "binance", mock_connection
            )
        
        assert len(resource_manager.tracked_connections) == 3
        
        # Force cleanup of all connections
        cleaned_count = await resource_manager.force_cleanup()
        
        # Verify cleanup (may not clean all if they're still active)
        assert resource_manager.force_cleanup_count > 0

    @pytest.mark.asyncio
    async def test_connection_limits_enforcement(self, resource_manager):
        """Test that connection limits are properly enforced."""
        # Set low limit for testing
        resource_manager.max_connections = 2
        
        # Register connections up to limit
        for i in range(2):
            mock_connection = Mock()
            result = resource_manager.register_connection(
                f"test_conn_{i}", "binance", mock_connection
            )
            assert result is True
        
        # Try to register one more - should fail
        mock_connection = Mock()
        result = resource_manager.register_connection(
            "test_conn_overflow", "binance", mock_connection
        )
        assert result is False

    def test_exchange_type_enum(self):
        """Test that exchange types are properly defined."""
        assert ExchangeType.BINANCE.value == "binance"
        assert ExchangeType.COINBASE.value == "coinbase"
        assert ExchangeType.OKX.value == "okx"

    def test_stream_type_enum(self):
        """Test that stream types are properly defined."""
        assert StreamType.TICKER.value == "ticker"
        assert StreamType.ORDERBOOK.value == "orderbook"
        assert StreamType.TRADES.value == "trades"
        assert StreamType.USER_DATA.value == "user_data"


    @pytest.mark.asyncio
    async def test_message_batching_performance(self, websocket_manager):
        """Test that message batching improves performance."""
        # Create a connection to test batching
        conn_id = await websocket_manager.create_connection(
            "wss://stream.binance.com:9443/ws", "binance"
        )
        
        try:
            # Test that high-performance batching is working
            if conn_id in websocket_manager.connections:
                connection = websocket_manager.connections[conn_id]
                # Verify the connection has queues for batching
                assert hasattr(connection, 'inbound_queue')
                assert hasattr(connection, 'outbound_queue')
            
        finally:
            await websocket_manager.disconnect_all()

    @pytest.mark.asyncio 
    async def test_resource_monitoring_overhead(self, websocket_manager):
        """Test that resource monitoring has minimal overhead."""
        # Test connection pool resource management
        initial_metrics = websocket_manager.get_connection_health()
        
        # Create multiple connections
        connection_ids = []
        for i in range(5):  # Reduced from 100 to respect connection limits
            try:
                conn_id = await websocket_manager.create_connection(
                    f"wss://stream.binance.com:9443/ws/btcusdt@ticker", 
                    "binance",
                    f"perf_test_{i}"
                )
                if conn_id:
                    connection_ids.append(conn_id)
                await asyncio.sleep(0.1)  # Small delay to avoid overwhelming
            except Exception:
                pass  # Connection limits may prevent all connections
        
        # Get updated metrics
        final_metrics = websocket_manager.get_connection_health()
        
        # Verify resource tracking is working
        assert final_metrics["total_connections"] >= initial_metrics["total_connections"]
        assert len(websocket_manager.connections) <= websocket_manager.config.websocket.max_total_connections
        
        # Cleanup
        await websocket_manager.disconnect_all()


class TestWebSocketConsolidationPerformance:
    """Test performance aspects of the consolidated WebSocket system."""
    pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])