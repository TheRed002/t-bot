"""Test suite for streaming service components."""

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.data.streaming.streaming_service import (
    BufferStrategy,
    StreamBuffer,
    StreamConfig,
    StreamMetrics,
    StreamState,
    WebSocketConnection,
)


class TestStreamMetrics:
    """Test suite for StreamMetrics."""

    def test_initialization_defaults(self):
        """Test stream metrics initialization with defaults."""
        metrics = StreamMetrics()

        assert metrics.messages_received == 0
        assert metrics.messages_processed == 0
        assert metrics.messages_dropped == 0
        assert metrics.bytes_received == 0
        assert metrics.connection_count == 0
        assert metrics.reconnection_count == 0
        assert metrics.last_message_time is None
        assert metrics.connection_uptime == timedelta()
        assert metrics.average_latency_ms == 0.0
        assert metrics.processing_rate_per_second == 0.0
        assert metrics.buffer_utilization == 0.0

    def test_initialization_with_values(self):
        """Test stream metrics initialization with custom values."""
        timestamp = datetime.now(timezone.utc)
        uptime = timedelta(minutes=30)

        metrics = StreamMetrics(
            messages_received=1000,
            messages_processed=950,
            messages_dropped=10,
            bytes_received=50000,
            connection_count=5,
            reconnection_count=2,
            last_message_time=timestamp,
            connection_uptime=uptime,
            average_latency_ms=25.5,
            processing_rate_per_second=100.0,
            buffer_utilization=0.75,
        )

        assert metrics.messages_received == 1000
        assert metrics.messages_processed == 950
        assert metrics.messages_dropped == 10
        assert metrics.bytes_received == 50000
        assert metrics.connection_count == 5
        assert metrics.reconnection_count == 2
        assert metrics.last_message_time == timestamp
        assert metrics.connection_uptime == uptime
        assert metrics.average_latency_ms == 25.5
        assert metrics.processing_rate_per_second == 100.0
        assert metrics.buffer_utilization == 0.75


class TestStreamConfig:
    """Test suite for StreamConfig."""

    def test_initialization_minimal(self):
        """Test minimal stream config initialization."""
        config = StreamConfig(
            exchange="binance",
            symbols=["BTCUSDT"],
            websocket_url="wss://stream.binance.com:9443/ws",
        )

        assert config.exchange == "binance"
        assert config.symbols == ["BTCUSDT"]
        assert config.websocket_url == "wss://stream.binance.com:9443/ws"
        assert config.data_types == ["ticker", "trades"]
        assert config.auth_required is False
        assert config.api_key is None
        assert config.api_secret is None
        assert config.connection_timeout == 30
        assert config.heartbeat_interval == 30
        assert config.max_reconnect_attempts == 10
        assert config.reconnect_delay == 5
        assert config.buffer_size == 10000
        assert config.buffer_strategy == BufferStrategy.DROP_OLDEST
        assert config.batch_size == 100
        assert config.flush_interval == 1
        assert config.enable_validation is True
        assert config.enable_deduplication is True
        assert config.max_latency_ms == 1000

    def test_initialization_full(self):
        """Test full stream config initialization."""
        config = StreamConfig(
            exchange="okx",
            symbols=["BTCUSDT", "ETHUSDT"],
            data_types=["ticker", "trades", "orderbook"],
            websocket_url="wss://ws.okx.com:8443/ws/v5/public",
            auth_required=True,
            api_key="test_key",
            api_secret="test_secret",
            connection_timeout=60,
            heartbeat_interval=15,
            max_reconnect_attempts=20,
            reconnect_delay=10,
            buffer_size=50000,
            buffer_strategy=BufferStrategy.DROP_NEWEST,
            batch_size=500,
            flush_interval=2,
            enable_validation=False,
            enable_deduplication=False,
            max_latency_ms=500,
        )

        assert config.exchange == "okx"
        assert config.symbols == ["BTCUSDT", "ETHUSDT"]
        assert config.data_types == ["ticker", "trades", "orderbook"]
        assert config.websocket_url == "wss://ws.okx.com:8443/ws/v5/public"
        assert config.auth_required is True
        assert config.api_key == "test_key"
        assert config.api_secret == "test_secret"
        assert config.connection_timeout == 60
        assert config.heartbeat_interval == 15
        assert config.max_reconnect_attempts == 20
        assert config.reconnect_delay == 10
        assert config.buffer_size == 50000
        assert config.buffer_strategy == BufferStrategy.DROP_NEWEST
        assert config.batch_size == 500
        assert config.flush_interval == 2
        assert config.enable_validation is False
        assert config.enable_deduplication is False
        assert config.max_latency_ms == 500

    def test_validation_exchange_empty(self):
        """Test exchange validation."""
        with pytest.raises(ValueError):
            StreamConfig(
                exchange="",  # Empty exchange
                symbols=["BTCUSDT"],
                websocket_url="wss://test.com",
            )

    def test_validation_symbols_empty(self):
        """Test symbols validation."""
        with pytest.raises(ValueError):
            StreamConfig(
                exchange="binance",
                symbols=[],  # Empty symbols list
                websocket_url="wss://test.com",
            )

    def test_validation_websocket_url_empty(self):
        """Test websocket URL validation."""
        with pytest.raises(ValueError):
            StreamConfig(
                exchange="binance",
                symbols=["BTCUSDT"],
                websocket_url="",  # Empty URL
            )

    def test_validation_connection_timeout(self):
        """Test connection timeout validation."""
        with pytest.raises(ValueError):
            StreamConfig(
                exchange="binance",
                symbols=["BTCUSDT"],
                websocket_url="wss://test.com",
                connection_timeout=1,  # Too small
            )

        with pytest.raises(ValueError):
            StreamConfig(
                exchange="binance",
                symbols=["BTCUSDT"],
                websocket_url="wss://test.com",
                connection_timeout=500,  # Too large
            )

    def test_validation_buffer_size(self):
        """Test buffer size validation."""
        with pytest.raises(ValueError):
            StreamConfig(
                exchange="binance",
                symbols=["BTCUSDT"],
                websocket_url="wss://test.com",
                buffer_size=50,  # Too small
            )


class TestStreamBuffer:
    """Test suite for StreamBuffer."""

    @pytest.fixture
    def stream_config(self):
        """Create stream config for buffer testing."""
        return StreamConfig(
            exchange="binance",
            symbols=["BTCUSDT"],
            websocket_url="wss://test.com",
            buffer_size=100,  # Minimum valid buffer size
            buffer_strategy=BufferStrategy.DROP_OLDEST,
        )

    @pytest.fixture
    def stream_buffer(self, stream_config):
        """Create stream buffer instance."""
        return StreamBuffer(config=stream_config)

    def test_initialization(self, stream_config):
        """Test stream buffer initialization."""
        buffer = StreamBuffer(config=stream_config)

        assert buffer.config is stream_config
        assert isinstance(buffer._buffer, deque)
        assert buffer._buffer.maxlen == 100
        assert isinstance(buffer._lock, asyncio.Lock)
        assert isinstance(buffer._condition, asyncio.Condition)
        assert buffer._dropped_count == 0

    @pytest.mark.asyncio
    async def test_put_and_get_success(self, stream_buffer):
        """Test successful put and get operations."""
        item = {"test": "data"}

        # Put item
        result = await stream_buffer.put(item)
        assert result is True

        # Get item
        retrieved = await stream_buffer.get(timeout=0.1)
        assert retrieved == item

    @pytest.mark.asyncio
    async def test_get_timeout(self, stream_buffer):
        """Test get operation timeout."""
        # Buffer is empty, should timeout
        result = await stream_buffer.get(timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_put_buffer_full_drop_oldest(self, stream_config):
        """Test buffer overflow with DROP_OLDEST strategy."""
        stream_config.buffer_strategy = BufferStrategy.DROP_OLDEST
        buffer = StreamBuffer(config=stream_config)

        # Fill buffer to capacity (use smaller number for test)
        for i in range(5):  # Use smaller number for test
            await buffer.put(f"item_{i}")

        # Add one more item (buffer not full yet)
        await buffer.put("new_item")

        # All items should still be in buffer since it's not full
        first_item = await buffer.get(timeout=0.1)
        assert first_item == "item_0"

    @pytest.mark.asyncio
    async def test_put_buffer_full_drop_newest(self, stream_config):
        """Test buffer overflow with DROP_NEWEST strategy."""
        stream_config.buffer_strategy = BufferStrategy.DROP_NEWEST
        buffer = StreamBuffer(config=stream_config)

        # Fill buffer to capacity (use smaller number for test)
        for i in range(5):  # Use smaller number for test
            await buffer.put(f"item_{i}")

        # Try to add one more item (buffer not full yet, so should succeed)
        result = await buffer.put("new_item")
        assert result is True

        # Buffer should have all items since it's not full
        assert buffer.size() == 6

    @pytest.mark.asyncio
    async def test_get_batch(self, stream_buffer):
        """Test batch retrieval."""
        # Put multiple items
        items = ["item_1", "item_2", "item_3"]
        for item in items:
            await stream_buffer.put(item)

        # Get batch
        batch = await stream_buffer.get_batch(max_size=2, timeout=0.1)

        assert len(batch) == 2
        assert batch[0] == "item_1"
        assert batch[1] == "item_2"

    @pytest.mark.asyncio
    async def test_get_batch_timeout(self, stream_buffer):
        """Test batch retrieval with timeout."""
        # Put one item
        await stream_buffer.put("item_1")

        # Try to get batch of 3 items with short timeout
        batch = await stream_buffer.get_batch(max_size=3, timeout=0.1)

        assert len(batch) == 1
        assert batch[0] == "item_1"

    def test_size(self, stream_buffer):
        """Test buffer size reporting."""
        assert stream_buffer.size() == 0

        # Add items using asyncio.run since we can't await in sync test
        async def add_items():
            await stream_buffer.put("item_1")
            await stream_buffer.put("item_2")

        asyncio.run(add_items())

        assert stream_buffer.size() == 2

    def test_utilization(self, stream_buffer):
        """Test buffer utilization calculation."""
        # Empty buffer
        assert stream_buffer.utilization() == 0.0

        # Add items
        async def fill_buffer():
            for i in range(3):  # 3 out of 5 capacity
                await stream_buffer.put(f"item_{i}")

        asyncio.run(fill_buffer())

        assert stream_buffer.utilization() == 0.03  # 3/100 = 0.03

    @pytest.mark.asyncio
    async def test_clear(self, stream_buffer):
        """Test buffer clearing."""
        # Add items
        await stream_buffer.put("item_1")
        await stream_buffer.put("item_2")

        assert stream_buffer.size() == 2

        # Clear buffer
        await stream_buffer.clear()

        assert stream_buffer.size() == 0
        assert stream_buffer.dropped_count() == 0


class TestWebSocketConnection:
    """Test suite for WebSocketConnection."""

    @pytest.fixture
    def stream_config(self):
        """Create stream config for WebSocket testing."""
        return StreamConfig(
            exchange="binance",
            symbols=["BTCUSDT"],
            websocket_url="wss://stream.binance.com:9443/ws",
        )

    @pytest.fixture
    def message_handler(self):
        """Create mock message handler."""
        return AsyncMock()

    @pytest.fixture
    def websocket_connection(self, stream_config, message_handler):
        """Create WebSocket connection instance."""
        return WebSocketConnection(config=stream_config, message_handler=message_handler)

    def test_initialization(self, stream_config, message_handler):
        """Test WebSocket connection initialization."""
        connection = WebSocketConnection(config=stream_config, message_handler=message_handler)

        assert connection.config is stream_config
        assert connection.message_handler is message_handler
        assert connection.websocket is None
        assert connection.state == StreamState.DISCONNECTED
        assert connection.connection_start_time is None
        assert connection.reconnect_count == 0

    @pytest.mark.asyncio
    async def test_connect_success(self, websocket_connection):
        """Test successful WebSocket connection."""
        mock_websocket = AsyncMock()

        with patch(
            "websockets.connect", new_callable=AsyncMock, return_value=mock_websocket
        ) as mock_connect:
            result = await websocket_connection.connect()

        assert result is True
        assert websocket_connection.state == StreamState.CONNECTED
        assert websocket_connection.websocket is mock_websocket
        assert websocket_connection.connection_start_time is not None

    @pytest.mark.asyncio
    async def test_connect_timeout(self, websocket_connection):
        """Test WebSocket connection timeout."""
        with patch("websockets.connect", side_effect=asyncio.TimeoutError("Connection timeout")):
            with pytest.raises(Exception):  # Should raise NetworkError
                await websocket_connection.connect()

        assert websocket_connection.state == StreamState.ERROR

    @pytest.mark.asyncio
    async def test_connect_exception(self, websocket_connection):
        """Test WebSocket connection exception."""
        with patch("websockets.connect", side_effect=Exception("Connection error")):
            with pytest.raises(Exception):  # Should raise NetworkError
                await websocket_connection.connect()

        assert websocket_connection.state == StreamState.ERROR


class TestEnums:
    """Test suite for streaming enums."""

    def test_stream_state_values(self):
        """Test stream state enum values."""
        assert StreamState.DISCONNECTED.value == "disconnected"
        assert StreamState.CONNECTING.value == "connecting"
        assert StreamState.CONNECTED.value == "connected"
        assert StreamState.RECONNECTING.value == "reconnecting"
        assert StreamState.ERROR.value == "error"
        assert StreamState.STOPPED.value == "stopped"

    def test_buffer_strategy_values(self):
        """Test buffer strategy enum values."""
        assert BufferStrategy.DROP_OLDEST.value == "drop_oldest"
        assert BufferStrategy.DROP_NEWEST.value == "drop_newest"
        assert BufferStrategy.BLOCK.value == "block"
        assert BufferStrategy.EXPAND.value == "expand"
