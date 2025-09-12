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
from pydantic import BaseModel, ConfigDict, Field
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed, WebSocketException

from src.core import BaseComponent
from src.core.config import Config
from src.core.exceptions import (
    ConfigurationError,
    DataError,
    DataProcessingError,
    DataValidationError,
    NetworkError,
)
from src.core.types import MarketData
from src.data.interfaces import DataServiceInterface, DataValidatorInterface
from src.utils.messaging_patterns import MessagingCoordinator


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

    model_config = ConfigDict(use_enum_values=False)


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
        items: list[Any] = []
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
        self.websocket: WebSocketClientProtocol | None = None
        self.state = StreamState.DISCONNECTED
        self.connection_start_time: datetime | None = None
        self.reconnect_count = 0
        self._heartbeat_task: asyncio.Task | None = None
        self._close_event = asyncio.Event()

    async def connect(self) -> bool:
        """Connect to WebSocket.

        SECURITY NOTE: API keys should be provided via environment variables
        or secure vault, not hardcoded in configuration.
        """
        websocket = None
        try:
            self.state = StreamState.CONNECTING

            # Prepare connection headers
            headers = {}
            if self.config.auth_required:
                import os

                # Try to get API key from environment first
                api_key = (
                    os.environ.get(f"{self.config.exchange.upper()}_API_KEY") or self.config.api_key
                )
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

            # Connect to WebSocket
            websocket = await websockets.connect(
                self.config.websocket_url,
                extra_headers=headers,
                timeout=self.config.connection_timeout,
                ping_interval=self.config.heartbeat_interval,
                ping_timeout=self.config.heartbeat_interval * 2,
            )

            self.websocket = websocket
            websocket = None  # Prevent cleanup of successfully assigned websocket
            self.state = StreamState.CONNECTED
            self.connection_start_time = datetime.now(timezone.utc)
            self._close_event.clear()

            # Send subscription message
            await self._send_subscription()

            # Start heartbeat task for connection monitoring
            self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

            return True

        except Exception as e:
            self.state = StreamState.ERROR
            # Clean up websocket connection on error
            if self.websocket:
                try:
                    await self.websocket.close()
                except Exception as close_e:
                    if hasattr(self, "logger"):
                        self.logger.error(
                            f"Failed to close assigned websocket during cleanup: {close_e}"
                        )
                finally:
                    self.websocket = None

            raise NetworkError(f"WebSocket connection failed: {e}")
        finally:
            # Clean up any unassigned websocket connection
            if websocket:
                try:
                    await websocket.close()
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.error(f"Failed to close websocket in finally block: {e}")

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
            raise NetworkError("WebSocket not connected")

        try:
            async for message in self.websocket:
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        yield data
                    except json.JSONDecodeError:
                        # Log invalid JSON but continue
                        if hasattr(self, "logger"):
                            self.logger.warning(f"Invalid JSON received: {message[:100]}")
                        continue
                elif isinstance(message, bytes):
                    try:
                        data = json.loads(message.decode("utf-8"))
                        yield data
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        if hasattr(self, "logger"):
                            self.logger.warning("Invalid bytes message received")
                        continue

        except ConnectionClosed:
            self.state = StreamState.DISCONNECTED
            if hasattr(self, "logger"):
                self.logger.info("WebSocket connection closed")
            raise
        except WebSocketException as e:
            self.state = StreamState.ERROR
            if hasattr(self, "logger"):
                self.logger.error(f"WebSocket exception: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._close_event.set()  # Signal connection to close

        # Cancel heartbeat task
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        websocket = None
        try:
            if self.websocket:
                websocket = self.websocket
                self.websocket = None
                await asyncio.wait_for(websocket.close(), timeout=5.0)
        except asyncio.TimeoutError:
            if hasattr(self, "logger"):
                self.logger.warning("WebSocket close timeout, forcing disconnect")
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.warning(f"Failed to close WebSocket during cleanup: {e}")
        finally:
            if websocket and not websocket.closed:
                try:
                    await asyncio.wait_for(websocket.close(), timeout=2.0)
                except asyncio.TimeoutError:
                    if hasattr(self, "logger"):
                        self.logger.warning("WebSocket final close timeout")
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.warning(f"Failed to close WebSocket in finally block: {e}")
            self.state = StreamState.DISCONNECTED

    async def _heartbeat_monitor(self) -> None:
        """Monitor WebSocket connection health with heartbeat."""
        try:
            while not self._close_event.is_set():
                if self.websocket and not self.websocket.closed:
                    try:
                        # Send ping and wait for pong
                        pong_waiter = await self.websocket.ping()
                        await asyncio.wait_for(pong_waiter, timeout=10.0)
                    except (asyncio.TimeoutError, ConnectionClosed, WebSocketException):
                        if hasattr(self, "logger"):
                            self.logger.warning("WebSocket heartbeat failed")
                        self.state = StreamState.ERROR
                        break
                    except Exception as e:
                        if hasattr(self, "logger"):
                            self.logger.error(f"Heartbeat error: {e}")
                        break

                # Wait for next heartbeat interval or close signal
                try:
                    await asyncio.wait_for(
                        self._close_event.wait(), timeout=self.config.heartbeat_interval
                    )
                    break  # Close event was set
                except asyncio.TimeoutError:
                    continue  # Normal timeout, continue heartbeat

        except asyncio.CancelledError:
            pass

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
        data_service: DataServiceInterface | None = None,
        validator: DataValidatorInterface | None = None,
    ):
        """Initialize streaming data service."""
        super().__init__()
        self.config = config

        # Service dependencies should be injected
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
        self._shutdown_events: dict[str, asyncio.Event] = {}  # Shutdown events per exchange

        # Message handlers
        self._message_handlers: dict[str, Callable] = {}

        # Deduplication cache
        self._dedup_cache: dict[str, set] = {}
        self._dedup_cache_size = 10000
        self._dedup_lock = asyncio.Lock()  # Protect concurrent access to dedup cache

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
                raise ConfigurationError(f"No configuration for exchange: {exchange}")

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

            # Initialize shutdown event for this exchange
            self._shutdown_events[exchange] = asyncio.Event()

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
            # Signal shutdown to background tasks
            if exchange in self._shutdown_events:
                self._shutdown_events[exchange].set()

            # Disconnect WebSocket
            if exchange in self._connections:
                await self._connections[exchange].disconnect()
                del self._connections[exchange]

            # Clear buffer
            if exchange in self._buffers:
                await self._buffers[exchange].clear()

            # Cleanup shutdown event
            if exchange in self._shutdown_events:
                del self._shutdown_events[exchange]

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
        shutdown_event = self._shutdown_events[exchange]

        try:
            while connection.is_connected() and not shutdown_event.is_set():
                try:
                    # Use timeout to allow periodic shutdown checks
                    listen_task = asyncio.create_task(self._listen_with_timeout(connection, shutdown_event))
                    async for message in listen_task:
                        # Check for shutdown during message processing
                        if shutdown_event.is_set():
                            break

                        # Update metrics
                        metrics.messages_received += 1
                        metrics.last_message_time = datetime.now(timezone.utc)

                        if isinstance(message, str):
                            metrics.bytes_received += len(message.encode("utf-8"))
                        elif isinstance(message, dict):
                            metrics.bytes_received += len(json.dumps(message).encode("utf-8"))

                        # Add to buffer with timeout to prevent blocking
                        try:
                            success = await asyncio.wait_for(buffer.put(message), timeout=1.0)
                            if not success:
                                metrics.messages_dropped += 1
                        except asyncio.TimeoutError:
                            metrics.messages_dropped += 1
                            self.logger.warning(f"Buffer put timeout for {exchange}")

                        # Update buffer utilization
                        metrics.buffer_utilization = buffer.utilization()

                except ConnectionClosed:
                    if not shutdown_event.is_set():
                        self.logger.warning(
                            f"Connection closed for {exchange}, attempting reconnect..."
                        )
                        await self._reconnect(exchange)
                    else:
                        break
                except Exception as e:
                    self.logger.error(f"Streaming error for {exchange}: {e}")
                    if shutdown_event.is_set():
                        break
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            self.logger.info(f"Stream task cancelled for {exchange}")
        finally:
            self.logger.info(f"Stream task stopped for {exchange}")

    async def _listen_with_timeout(self, connection: WebSocketConnection, shutdown_event: asyncio.Event) -> AsyncGenerator[dict[str, Any], None]:
        """Listen for messages with timeout to allow shutdown checks."""
        try:
            async for message in connection.listen():
                if shutdown_event.is_set():
                    break
                yield message
        except asyncio.CancelledError:
            # Handle graceful cancellation
            pass

    async def _processor_task(self, exchange: str) -> None:
        """Background task for processing buffered data."""
        buffer = self._buffers[exchange]
        metrics = self._metrics[exchange]
        config = self._stream_configs[exchange]
        shutdown_event = self._shutdown_events[exchange]

        try:
            while not shutdown_event.is_set():
                try:
                    # Get batch of messages with timeout to allow shutdown check
                    messages = await buffer.get_batch(
                        config.batch_size, timeout=config.flush_interval
                    )

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
                    # Check if we should exit due to shutdown
                    if shutdown_event.is_set():
                        break
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            self.logger.info(f"Processor task cancelled for {exchange}")
        finally:
            self.logger.info(f"Processor task stopped for {exchange}")

    async def _process_message_batch(self, exchange: str, messages: list[dict[str, Any]]) -> None:
        """Process batch of messages using consistent data flow patterns from pipeline and messaging coordinator."""
        config = self._stream_configs[exchange]
        handler = self._message_handlers.get(exchange, self._message_handlers["default"])

        # Module boundary validation using messaging patterns
        if not config or handler is None:
            raise DataProcessingError(
                f"Invalid streaming configuration for {exchange}",
                error_code="DATA_201",
                processing_step="configuration_validation",
                data_source=exchange,
                pipeline_stage="ingestion",
            )

        # Use consistent processing paradigm alignment
        from src.utils.messaging_patterns import ProcessingParadigmAligner

        # Align stream to batch processing for consistency with pipeline
        stream_items = [{"message": msg, "exchange": exchange} for msg in messages]
        batch_data = ProcessingParadigmAligner.create_batch_from_stream(stream_items)

        # Apply processing mode alignment for consistent flow
        aligned_data = ProcessingParadigmAligner.align_processing_modes(
            source_mode="stream", target_mode="batch", data=batch_data
        )

        # Use consistent pub/sub messaging pattern for streaming data processing
        coordinator = MessagingCoordinator("StreamingDataProcessing")

        market_data_list = []
        processing_errors = []

        for message in messages:
            try:
                # Validate raw message at module boundary using messaging patterns
                if not isinstance(message, dict) or not message:
                    continue

                market_data = await handler(message, exchange)
                if market_data:
                    # Apply consistent data transformation from messaging coordinator using pub/sub pattern
                    market_data_dict = (
                        market_data.model_dump()
                        if hasattr(market_data, "model_dump")
                        else market_data.__dict__
                    )
                    transformed_data = coordinator._apply_data_transformation(market_data_dict)

                    # Ensure consistent financial data transformation matching utils patterns
                    if "price" in transformed_data and transformed_data["price"] is not None:
                        from src.utils.data_utils import normalize_price
                        if hasattr(market_data, "symbol") and market_data.symbol:
                            transformed_data["price"] = normalize_price(
                                transformed_data["price"], market_data.symbol, precision=8
                            )

                    # Recreate MarketData with consistent transformed data
                    market_data = MarketData(**transformed_data)

                    # Deduplication check with consistent pattern
                    if config.enable_deduplication:
                        if await self._is_duplicate(exchange, market_data):
                            continue

                    market_data_list.append(market_data)

            except Exception as e:
                # Use consistent error propagation pattern from pipeline
                processing_error = DataProcessingError(
                    f"Message processing failed for {exchange}",
                    error_code="DATA_202",
                    processing_step="streaming_message_conversion",
                    input_data_sample={
                        "exchange": exchange,
                        "message_keys": list(message.keys()) if isinstance(message, dict) else [],
                    },
                    data_source=exchange,
                    data_type="streaming_message",
                    pipeline_stage="ingestion",
                )
                processing_errors.append(processing_error)
                self.logger.error(f"Message processing error: {e}")

        if not market_data_list:
            return

        # Module boundary validation - ensure we have valid data
        if not market_data_list:
            if processing_errors:
                # Log processing errors but don't fail the entire batch
                self.logger.warning(
                    f"All messages failed processing for {exchange}: {len(processing_errors)} errors"
                )
            return

        # Use consistent validation patterns from pipeline at module boundary
        if config.enable_validation and self.validator:
            try:
                # Apply boundary validation using consistent patterns
                from src.utils.messaging_patterns import BoundaryValidator

                for data in market_data_list:
                    BoundaryValidator.validate_database_entity(data.model_dump(), "create")

                # Validate at module boundary before storage
                if hasattr(self.validator, "validate_market_data"):
                    validated_data = await self.validator.validate_market_data(market_data_list)
                    market_data_list = validated_data
                else:
                    # Use basic validation for each item
                    for data in market_data_list:
                        self.validator.validate(data.model_dump())

            except Exception as e:
                raise DataValidationError(
                    f"Streaming data validation failed for {exchange}",
                    error_code="DATA_203",
                    validation_rule="market_data_batch_validation",
                    invalid_fields=["batch_validation"],
                    data_source=exchange,
                ) from e

        # Store data using consistent patterns with pipeline
        if market_data_list:
            if not self.data_service:
                self.logger.warning(
                    f"No data service available - discarding {len(market_data_list)} market data records for {exchange}"
                )
                return

            try:
                # Use consistent storage interface with pipeline
                success = await self.data_service.store_market_data(
                    data=market_data_list,
                    exchange=exchange,
                    validate=False,  # Already validated at boundary
                )

                if not success:
                    raise DataError(
                        f"Data storage operation failed for {exchange}",
                        error_code="DATA_204",
                        data_type="market_data_batch",
                        data_source=exchange,
                        context={"batch_size": len(market_data_list)},
                    )

            except Exception as e:
                # Consistent error propagation with pipeline patterns
                if not isinstance(e, (DataError, DataValidationError)):
                    # Wrap generic errors in consistent format
                    raise DataError(
                        f"Streaming storage error for {exchange}",
                        error_code="DATA_205",
                        data_type="market_data_batch",
                        data_source=exchange,
                        context={"original_error": str(e)},
                    ) from e
                raise

    async def _is_duplicate(self, exchange: str, data: MarketData) -> bool:
        """Check if data is duplicate with proper concurrency control."""
        async with self._dedup_lock:
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

                # Connect with timeout
                await asyncio.wait_for(connection.connect(), timeout=config.connection_timeout)

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
                open=Decimal(str(price)),
                high=Decimal(str(price)),
                low=Decimal(str(price)),
                close=Decimal(str(price)),
                volume=Decimal(str(volume)) if volume else Decimal("0"),
                timestamp=timestamp,
                exchange=exchange,
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
                    close=Decimal(str(data.get("c", 0))),  # Close price
                    volume=Decimal(str(data.get("v", 0))),  # Volume
                    high=Decimal(str(data.get("h", 0))),  # High
                    low=Decimal(str(data.get("l", 0))),  # Low
                    open=Decimal(str(data.get("o", 0))),  # Open
                    timestamp=datetime.now(timezone.utc),
                    exchange=exchange,
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
                price = Decimal(str(message.get("price", 0)))
                return MarketData(
                    symbol=message.get("product_id", "").replace("-", ""),
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=Decimal(str(message.get("volume_24h", 0))),
                    timestamp=datetime.now(timezone.utc),
                    exchange=exchange,
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
                    price = Decimal(str(data.get("last", 0)))
                    return MarketData(
                        symbol=data.get("instId", "").replace("-", ""),
                        open=price,
                        high=Decimal(str(data.get("high24h", 0))),
                        low=Decimal(str(data.get("low24h", 0))),
                        close=price,
                        volume=Decimal(str(data.get("vol24h", 0))),
                        timestamp=datetime.now(timezone.utc),
                        exchange=exchange,
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

    def get_metrics(self) -> dict[str, Any]:
        """Get streaming metrics for all exchanges."""
        return self._metrics.copy()

    async def health_check(self) -> dict[str, Any]:
        """Perform streaming service health check."""
        status = "healthy"
        streams_info = {}

        for exchange in self._stream_configs.keys():
            stream_status = await self.get_stream_status(exchange)
            streams_info[exchange] = stream_status

            if not stream_status.get("connected", False):
                status = "degraded"

        return {
            "status": status,
            "message": f"Streaming service: {status}",
            "initialized": self._initialized,
            "active_streams": len(self._connections),
            "configured_streams": len(self._stream_configs),
            "streams": streams_info,
        }

    async def cleanup(self) -> None:
        """Cleanup streaming service resources."""
        background_tasks = []
        connections = {}
        try:
            # Collect resources for cleanup
            background_tasks = list(self._background_tasks)
            connections = dict(self._connections)

            # Signal shutdown to all exchanges first
            for exchange in connections.keys():
                if exchange in self._shutdown_events:
                    self._shutdown_events[exchange].set()

            # Cancel background tasks with timeout
            cancel_tasks = []
            for task in background_tasks:
                if not task.done():
                    task.cancel()
                    cancel_tasks.append(task)

            if cancel_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*cancel_tasks, return_exceptions=True), timeout=10.0
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("Timeout waiting for background tasks to cancel")

            # Disconnect all streams with timeout
            disconnect_tasks = [self.stop_stream(exchange) for exchange in list(connections.keys())]
            if disconnect_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*disconnect_tasks, return_exceptions=True), timeout=30.0
                    )
                except asyncio.TimeoutError:
                    self.logger.warning(
                        "Timeout during stream cleanup, some connections may not have closed cleanly"
                    )

            # Clear resources
            self._connections.clear()
            self._buffers.clear()
            self._metrics.clear()
            self._dedup_cache.clear()
            self._background_tasks.clear()
            self._shutdown_events.clear()

            self._initialized = False
            self.logger.info("StreamingDataService cleanup completed")

        except Exception as e:
            self.logger.error(f"StreamingDataService cleanup error: {e}")
        finally:
            # Force cleanup any remaining resources with proper async context
            try:
                # Force cancel any remaining background tasks
                remaining_tasks = [task for task in background_tasks if not task.done()]
                if remaining_tasks:
                    for task in remaining_tasks:
                        task.cancel()
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*remaining_tasks, return_exceptions=True), timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        self.logger.warning("Force cancel timeout - some tasks may remain")

                # Force disconnect any remaining connections with async context managers
                disconnect_coros = []
                for exchange, connection in connections.items():
                    if connection.is_connected():
                        disconnect_coros.append(connection.disconnect())

                if disconnect_coros:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*disconnect_coros, return_exceptions=True), timeout=10.0
                        )
                    except asyncio.TimeoutError:
                        self.logger.warning("Force disconnect timeout")

                # Clear all resources
                self._connections.clear()
                self._buffers.clear()
                self._metrics.clear()
                self._dedup_cache.clear()
                self._background_tasks.clear()
                self._shutdown_events.clear()
                self._initialized = False
            except Exception as e:
                self.logger.warning(f"Error in final streaming cleanup: {e}")
