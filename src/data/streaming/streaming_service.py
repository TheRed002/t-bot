"""
StreamingDataService - Real-time Market Data Streaming

This module implements enterprise-grade real-time market data streaming
with WebSocket management, automatic reconnection, backpressure handling,
and sophisticated buffering for high-frequency trading applications.

Dependencies:
- P-001: Core types, exceptions, logging
- P-002: Database models, queries, and connections
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

import asyncio
import json
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

import websockets
from pydantic import BaseModel, Field
from websockets.exceptions import ConnectionClosed, WebSocketException

from src.base import BaseComponent
from src.core.config import Config
from src.core.types import MarketData
from src.data.services.data_service import DataService
from src.data.validation.data_validator import DataValidator


class StreamState(Enum):
    """Streaming connection state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    STOPPED = "stopped"


class BufferStrategy(Enum):
    """Buffer overflow strategy."""

    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"
    EXPAND = "expand"


@dataclass
class StreamMetrics:
    """Streaming metrics."""

    messages_received: int = 0
    messages_processed: int = 0
    messages_dropped: int = 0
    bytes_received: int = 0
    connection_count: int = 0
    reconnection_count: int = 0
    last_message_time: datetime | None = None
    connection_uptime: timedelta = timedelta()
    average_latency_ms: float = 0.0
    processing_rate_per_second: float = 0.0
    buffer_utilization: float = 0.0


class StreamConfig(BaseModel):
    """Stream configuration model."""

    exchange: str = Field(..., min_length=1)
    symbols: list[str] = Field(..., min_length=1)
    data_types: list[str] = Field(default_factory=lambda: ["ticker", "trades"])
    websocket_url: str = Field(..., min_length=1)
    auth_required: bool = False
    api_key: str | None = None
    api_secret: str | None = None

    # Connection settings
    connection_timeout: int = Field(30, ge=5, le=300)
    heartbeat_interval: int = Field(30, ge=10, le=300)
    max_reconnect_attempts: int = Field(10, ge=1, le=100)
    reconnect_delay: int = Field(5, ge=1, le=60)

    # Buffer settings
    buffer_size: int = Field(10000, ge=100, le=1000000)
    buffer_strategy: BufferStrategy = BufferStrategy.DROP_OLDEST
    batch_size: int = Field(100, ge=1, le=1000)
    flush_interval: int = Field(1, ge=1, le=60)

    # Quality settings
    enable_validation: bool = True
    enable_deduplication: bool = True
    max_latency_ms: int = Field(1000, ge=10, le=10000)

    class Config:
        use_enum_values = True


class StreamBuffer:
    """High-performance streaming data buffer."""

    def __init__(self, config: StreamConfig):
        self.config = config
        self._buffer: deque = deque(maxlen=config.buffer_size)
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._dropped_count = 0

    async def put(self, item: Any) -> bool:
        """Add item to buffer."""
        async with self._condition:
            if len(self._buffer) >= self.config.buffer_size:
                if self.config.buffer_strategy == BufferStrategy.DROP_OLDEST:
                    self._buffer.popleft()
                elif self.config.buffer_strategy == BufferStrategy.DROP_NEWEST:
                    self._dropped_count += 1
                    return False
                elif self.config.buffer_strategy == BufferStrategy.BLOCK:
                    await self._condition.wait()
                elif self.config.buffer_strategy == BufferStrategy.EXPAND:
                    # Allow buffer to grow beyond max size
                    pass

            self._buffer.append(item)
            self._condition.notify()
            return True

    async def get(self, timeout: float | None = None) -> Any | None:
        """Get item from buffer."""
        async with self._condition:
            if not self._buffer:
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    return None

            if self._buffer:
                return self._buffer.popleft()
            return None

    async def get_batch(self, max_size: int, timeout: float = 1.0) -> list[Any]:
        """Get batch of items from buffer."""
        items = []
        start_time = time.time()

        while len(items) < max_size and (time.time() - start_time) < timeout:
            item = await self.get(timeout=0.1)
            if item is not None:
                items.append(item)
            else:
                break

        return items

    def size(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)

    def utilization(self) -> float:
        """Get buffer utilization percentage."""
        return len(self._buffer) / self.config.buffer_size if self.config.buffer_size > 0 else 0.0

    def dropped_count(self) -> int:
        """Get number of dropped messages."""
        return self._dropped_count

    async def clear(self) -> None:
        """Clear buffer."""
        async with self._lock:
            self._buffer.clear()
            self._dropped_count = 0


class WebSocketConnection:
    """WebSocket connection manager with automatic reconnection."""

    def __init__(self, config: StreamConfig, message_handler: Callable):
        self.config = config
        self.message_handler = message_handler
        self.websocket: websockets.WebSocketServerProtocol | None = None
        self.state = StreamState.DISCONNECTED
        self.connection_start_time: datetime | None = None
        self.reconnect_count = 0

    async def connect(self) -> bool:
        """Connect to WebSocket.
        
        SECURITY NOTE: API keys should be provided via environment variables
        or secure vault, not hardcoded in configuration.
        """
        try:
            self.state = StreamState.CONNECTING

            # Prepare connection headers
            headers = {}
            if self.config.auth_required:
                import os
                # Try to get API key from environment first
                api_key = os.environ.get(f"{self.config.exchange.upper()}_API_KEY") or self.config.api_key
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

            # Connect to WebSocket
            self.websocket = await websockets.connect(
                self.config.websocket_url,
                extra_headers=headers,
                timeout=self.config.connection_timeout,
                ping_interval=self.config.heartbeat_interval,
                ping_timeout=self.config.heartbeat_interval * 2,
            )

            self.state = StreamState.CONNECTED
            self.connection_start_time = datetime.now(timezone.utc)

            # Send subscription message
            await self._send_subscription()

            return True

        except Exception as e:
            self.state = StreamState.ERROR
            raise ConnectionError(f"WebSocket connection failed: {e}")

    async def _send_subscription(self) -> None:
        """Send subscription message for symbols and data types."""
        if not self.websocket:
            return

        # Generic subscription format - needs to be customized per exchange
        subscription = {
            "method": "subscribe",
            "params": {
                "symbols": self.config.symbols,
                "types": self.config.data_types,
            },
            "id": str(uuid.uuid4()),
        }

        await self.websocket.send(json.dumps(subscription))

    async def listen(self) -> AsyncGenerator[dict[str, Any], None]:
        """Listen for messages from WebSocket."""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")

        try:
            async for message in self.websocket:
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        yield data
                    except json.JSONDecodeError:
                        # Log invalid JSON but continue
                        continue
                elif isinstance(message, bytes):
                    try:
                        data = json.loads(message.decode("utf-8"))
                        yield data
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

        except ConnectionClosed:
            self.state = StreamState.DISCONNECTED
            raise
        except WebSocketException:
            self.state = StreamState.ERROR
            raise

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self.state = StreamState.DISCONNECTED

    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.state == StreamState.CONNECTED and self.websocket is not None

    def get_uptime(self) -> timedelta:
        """Get connection uptime."""
        if self.connection_start_time:
            return datetime.now(timezone.utc) - self.connection_start_time
        return timedelta()


class StreamingDataService(BaseComponent):
    """
    Enterprise-grade real-time market data streaming service.

    Features:
    - WebSocket connection management with auto-reconnection
    - High-performance buffering with configurable strategies
    - Real-time data validation and quality monitoring
    - Backpressure handling and flow control
    - Comprehensive metrics and monitoring
    - Support for multiple exchanges and data types
    """

    def __init__(
        self,
        config: Config,
        data_service: DataService | None = None,
        validator: DataValidator | None = None,
    ):
        """Initialize streaming data service."""
        super().__init__()
        self.config = config
        self.data_service = data_service
        self.validator = validator

        # Stream configurations
        self._stream_configs: dict[str, StreamConfig] = {}

        # Active connections
        self._connections: dict[str, WebSocketConnection] = {}
        self._buffers: dict[str, StreamBuffer] = {}

        # Metrics
        self._metrics: dict[str, StreamMetrics] = {}

        # Background tasks
        self._background_tasks: list[asyncio.Task] = []

        # Message handlers
        self._message_handlers: dict[str, Callable] = {}

        # Deduplication cache
        self._dedup_cache: dict[str, set] = {}
        self._dedup_cache_size = 10000

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize streaming service."""
        try:
            if self._initialized:
                return

            self.logger.info("Initializing StreamingDataService...")

            # Load stream configurations
            await self._load_stream_configurations()

            # Initialize message handlers
            await self._setup_message_handlers()

            self._initialized = True
            self.logger.info("StreamingDataService initialized successfully")

        except Exception as e:
            self.logger.error(f"StreamingDataService initialization failed: {e}")
            raise

    async def _load_stream_configurations(self) -> None:
        """Load streaming configurations from config."""
        streaming_config = getattr(self.config, "streaming", {})

        for exchange, exchange_config in streaming_config.items():
            if isinstance(exchange_config, dict):
                try:
                    stream_config = StreamConfig(exchange=exchange, **exchange_config)
                    self._stream_configs[exchange] = stream_config
                    self.logger.info(f"Loaded stream config for {exchange}")
                except Exception as e:
                    self.logger.error(f"Invalid stream config for {exchange}: {e}")

    async def _setup_message_handlers(self) -> None:
        """Setup message handlers for different exchanges."""
        # Default message handler
        self._message_handlers["default"] = self._handle_generic_message

        # Exchange-specific handlers
        self._message_handlers["binance"] = self._handle_binance_message
        self._message_handlers["coinbase"] = self._handle_coinbase_message
        self._message_handlers["okx"] = self._handle_okx_message

    async def add_stream(self, exchange: str, config: StreamConfig) -> bool:
        """Add new streaming configuration."""
        try:
            self._stream_configs[exchange] = config
            self._metrics[exchange] = StreamMetrics()

            # Initialize buffer
            self._buffers[exchange] = StreamBuffer(config)

            self.logger.info(f"Added stream configuration for {exchange}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add stream for {exchange}: {e}")
            return False

    async def start_stream(self, exchange: str) -> bool:
        """Start streaming for an exchange."""
        try:
            if exchange not in self._stream_configs:
                raise ValueError(f"No configuration for exchange: {exchange}")

            config = self._stream_configs[exchange]

            # Create connection
            message_handler = self._message_handlers.get(
                exchange, self._message_handlers["default"]
            )
            connection = WebSocketConnection(config, message_handler)

            # Connect
            await connection.connect()

            # Store connection and initialize metrics
            self._connections[exchange] = connection
            if exchange not in self._metrics:
                self._metrics[exchange] = StreamMetrics()

            # Start background tasks
            stream_task = asyncio.create_task(self._stream_task(exchange))
            processor_task = asyncio.create_task(self._processor_task(exchange))

            self._background_tasks.extend([stream_task, processor_task])

            self.logger.info(f"Started streaming for {exchange}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start stream for {exchange}: {e}")
            return False

    async def stop_stream(self, exchange: str) -> bool:
        """Stop streaming for an exchange."""
        try:
            # Disconnect WebSocket
            if exchange in self._connections:
                await self._connections[exchange].disconnect()
                del self._connections[exchange]

            # Clear buffer
            if exchange in self._buffers:
                await self._buffers[exchange].clear()

            self.logger.info(f"Stopped streaming for {exchange}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop stream for {exchange}: {e}")
            return False

    async def _stream_task(self, exchange: str) -> None:
        """Background task for streaming data."""
        connection = self._connections[exchange]
        buffer = self._buffers[exchange]
        metrics = self._metrics[exchange]
        self._stream_configs[exchange]

        while connection.is_connected():
            try:
                async for message in connection.listen():
                    # Update metrics
                    metrics.messages_received += 1
                    metrics.last_message_time = datetime.now(timezone.utc)

                    if isinstance(message, str):
                        metrics.bytes_received += len(message.encode("utf-8"))
                    elif isinstance(message, dict):
                        metrics.bytes_received += len(json.dumps(message).encode("utf-8"))

                    # Add to buffer
                    success = await buffer.put(message)
                    if not success:
                        metrics.messages_dropped += 1

                    # Update buffer utilization
                    metrics.buffer_utilization = buffer.utilization()

            except ConnectionClosed:
                self.logger.warning(f"Connection closed for {exchange}, attempting reconnect...")
                await self._reconnect(exchange)
            except Exception as e:
                self.logger.error(f"Streaming error for {exchange}: {e}")
                await asyncio.sleep(1)

    async def _processor_task(self, exchange: str) -> None:
        """Background task for processing buffered data."""
        buffer = self._buffers[exchange]
        metrics = self._metrics[exchange]
        config = self._stream_configs[exchange]

        while True:
            try:
                # Get batch of messages
                messages = await buffer.get_batch(config.batch_size, timeout=config.flush_interval)

                if messages:
                    # Process batch
                    await self._process_message_batch(exchange, messages)

                    # Update metrics
                    metrics.messages_processed += len(messages)

                    # Calculate processing rate
                    if metrics.last_message_time:
                        elapsed = (
                            datetime.now(timezone.utc) - metrics.last_message_time
                        ).total_seconds()
                        if elapsed > 0:
                            metrics.processing_rate_per_second = len(messages) / elapsed

            except Exception as e:
                self.logger.error(f"Processing error for {exchange}: {e}")
                await asyncio.sleep(1)

    async def _process_message_batch(self, exchange: str, messages: list[dict[str, Any]]) -> None:
        """Process batch of messages."""
        config = self._stream_configs[exchange]
        handler = self._message_handlers.get(exchange, self._message_handlers["default"])

        # Convert messages to MarketData objects
        market_data_list = []

        for message in messages:
            try:
                market_data = await handler(message, exchange)
                if market_data:
                    # Deduplication check
                    if config.enable_deduplication:
                        if await self._is_duplicate(exchange, market_data):
                            continue

                    market_data_list.append(market_data)

            except Exception as e:
                self.logger.error(f"Message processing error: {e}")
                continue

        if not market_data_list:
            return

        # Validate data if enabled
        if config.enable_validation and self.validator:
            try:
                validation_results = await self.validator.validate_market_data(
                    market_data_list, include_statistical=True
                )

                # Filter valid data
                valid_data = []
                for i, result in enumerate(validation_results):
                    if result.is_valid or result.quality_score.overall > 0.7:
                        valid_data.append(market_data_list[i])

                market_data_list = valid_data

            except Exception as e:
                self.logger.error(f"Validation error: {e}")

        # Store data if data service available
        if self.data_service and market_data_list:
            try:
                await self.data_service.store_market_data(
                    market_data_list, exchange, validate=False
                )
            except Exception as e:
                self.logger.error(f"Data storage error: {e}")

    async def _is_duplicate(self, exchange: str, data: MarketData) -> bool:
        """Check if data is duplicate."""
        if exchange not in self._dedup_cache:
            self._dedup_cache[exchange] = set()

        cache = self._dedup_cache[exchange]

        # Create unique key
        key = f"{data.symbol}:{data.price}:{data.timestamp}"

        if key in cache:
            return True

        # Add to cache
        cache.add(key)

        # Limit cache size
        if len(cache) > self._dedup_cache_size:
            # Remove oldest entries (simplified - should use LRU)
            cache.clear()

        return False

    async def _reconnect(self, exchange: str) -> None:
        """Reconnect to exchange stream."""
        config = self._stream_configs[exchange]
        metrics = self._metrics[exchange]

        for attempt in range(config.max_reconnect_attempts):
            try:
                self.logger.info(f"Reconnecting to {exchange} (attempt {attempt + 1})")

                # Wait before reconnecting
                await asyncio.sleep(config.reconnect_delay)

                # Create new connection
                message_handler = self._message_handlers.get(
                    exchange, self._message_handlers["default"]
                )
                connection = WebSocketConnection(config, message_handler)

                # Connect
                await connection.connect()

                # Update connection
                self._connections[exchange] = connection
                metrics.reconnection_count += 1

                self.logger.info(f"Reconnected to {exchange}")
                return

            except Exception as e:
                self.logger.error(f"Reconnection attempt {attempt + 1} failed for {exchange}: {e}")

        self.logger.error(
            f"Failed to reconnect to {exchange} after {config.max_reconnect_attempts} attempts"
        )

    async def _handle_generic_message(
        self, message: dict[str, Any], exchange: str
    ) -> MarketData | None:
        """Generic message handler."""
        try:
            # Extract common fields
            symbol = message.get("symbol", message.get("s"))
            price = message.get("price", message.get("p"))
            volume = message.get("volume", message.get("v"))
            timestamp_ms = message.get("timestamp", message.get("t"))

            if not symbol or not price:
                return None

            # Convert timestamp
            timestamp = None
            if timestamp_ms:
                timestamp = datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)

            return MarketData(
                symbol=str(symbol).upper(),
                price=Decimal(str(price)),
                volume=Decimal(str(volume)) if volume else None,
                timestamp=timestamp,
            )

        except Exception as e:
            self.logger.error(f"Generic message parsing error: {e}")
            return None

    async def _handle_binance_message(
        self, message: dict[str, Any], exchange: str
    ) -> MarketData | None:
        """Binance-specific message handler."""
        try:
            # Binance ticker format
            if "stream" in message and "data" in message:
                data = message["data"]

                return MarketData(
                    symbol=data.get("s", "").upper(),
                    price=Decimal(str(data.get("c", 0))),  # Close price
                    volume=Decimal(str(data.get("v", 0))),  # Volume
                    high_price=Decimal(str(data.get("h", 0))),  # High
                    low_price=Decimal(str(data.get("l", 0))),  # Low
                    open_price=Decimal(str(data.get("o", 0))),  # Open
                    timestamp=datetime.now(timezone.utc),
                )

            return await self._handle_generic_message(message, exchange)

        except Exception as e:
            self.logger.error(f"Binance message parsing error: {e}")
            return None

    async def _handle_coinbase_message(
        self, message: dict[str, Any], exchange: str
    ) -> MarketData | None:
        """Coinbase-specific message handler."""
        try:
            # Coinbase ticker format
            if message.get("type") == "ticker":
                return MarketData(
                    symbol=message.get("product_id", "").replace("-", ""),
                    price=Decimal(str(message.get("price", 0))),
                    volume=Decimal(str(message.get("volume_24h", 0))),
                    bid=Decimal(str(message.get("best_bid", 0))),
                    ask=Decimal(str(message.get("best_ask", 0))),
                    timestamp=datetime.now(timezone.utc),
                )

            return await self._handle_generic_message(message, exchange)

        except Exception as e:
            self.logger.error(f"Coinbase message parsing error: {e}")
            return None

    async def _handle_okx_message(
        self, message: dict[str, Any], exchange: str
    ) -> MarketData | None:
        """OKX-specific message handler."""
        try:
            # OKX ticker format
            if "data" in message:
                for data in message["data"]:
                    return MarketData(
                        symbol=data.get("instId", "").replace("-", ""),
                        price=Decimal(str(data.get("last", 0))),
                        volume=Decimal(str(data.get("vol24h", 0))),
                        high_price=Decimal(str(data.get("high24h", 0))),
                        low_price=Decimal(str(data.get("low24h", 0))),
                        timestamp=datetime.now(timezone.utc),
                    )

            return await self._handle_generic_message(message, exchange)

        except Exception as e:
            self.logger.error(f"OKX message parsing error: {e}")
            return None

    async def get_stream_status(self, exchange: str | None = None) -> dict[str, Any]:
        """Get streaming status."""
        if exchange:
            if exchange in self._connections:
                connection = self._connections[exchange]
                metrics = self._metrics.get(exchange, StreamMetrics())

                return {
                    "exchange": exchange,
                    "state": connection.state.value,
                    "connected": connection.is_connected(),
                    "uptime": connection.get_uptime().total_seconds(),
                    "reconnect_count": connection.reconnect_count,
                    "metrics": metrics,
                }
            else:
                return {"exchange": exchange, "state": "not_configured"}
        else:
            # Return status for all exchanges
            status = {}
            for exchange_name in self._stream_configs.keys():
                status[exchange_name] = await self.get_stream_status(exchange_name)
            return status

    async def get_metrics(self) -> dict[str, StreamMetrics]:
        """Get streaming metrics for all exchanges."""
        return self._metrics.copy()

    async def health_check(self) -> dict[str, Any]:
        """Perform streaming service health check."""
        health = {
            "status": "healthy",
            "initialized": self._initialized,
            "active_streams": len(self._connections),
            "configured_streams": len(self._stream_configs),
            "streams": {},
        }

        for exchange in self._stream_configs.keys():
            stream_status = await self.get_stream_status(exchange)
            health["streams"][exchange] = stream_status

            if not stream_status.get("connected", False):
                health["status"] = "degraded"

        return health

    async def cleanup(self) -> None:
        """Cleanup streaming service resources."""
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Disconnect all streams
            for exchange in list(self._connections.keys()):
                await self.stop_stream(exchange)

            # Clear resources
            self._connections.clear()
            self._buffers.clear()
            self._metrics.clear()
            self._dedup_cache.clear()

            self._initialized = False
            self.logger.info("StreamingDataService cleanup completed")

        except Exception as e:
            self.logger.error(f"StreamingDataService cleanup error: {e}")
