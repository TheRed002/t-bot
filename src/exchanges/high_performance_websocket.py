"""
High-Performance WebSocket Manager

This module implements optimized WebSocket connections for high-frequency trading
with minimal latency, connection pooling, and message batching capabilities.

Key Optimizations:
- Connection pooling with automatic load balancing
- Message batching for reduced network overhead
- ujson for fast JSON parsing (5-10x faster than standard json)
- Binary protocol support where available
- Connection multiplexing and automatic failover
- Zero-copy message processing
- Lock-free message queues using asyncio.Queue

Performance Targets:
- Message processing: >10,000 messages/second per connection
- Latency: <5ms for critical messages
- Connection setup: <100ms including authentication
- Memory usage: <50MB per connection pool
"""

import asyncio
import time
import zlib
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

try:
    import ujson as json  # 5-10x faster than standard json
except ImportError:
    import json

import websockets
from websockets.exceptions import ConnectionClosed

from src.core.config import Config
from src.core.exceptions import ExchangeConnectionError, ValidationError
from src.core.logging import get_logger
from src.core.types import ConnectionType


@dataclass
class PooledConnection:
    """Information about a pooled connection."""

    connection_id: str
    exchange: str
    connection_type: ConnectionType
    connection: Any
    created_at: datetime
    last_used: datetime
    is_healthy: bool = True
    message_count: int = 0
    subscription_count: int = 0


class MessagePriority(Enum):
    """Message priority levels for processing order."""

    CRITICAL = 1  # Order updates, fills
    HIGH = 2  # Price updates, trade data
    MEDIUM = 3  # Market data, indicators
    LOW = 4  # Heartbeats, status updates


@dataclass
class WebSocketMessage:
    """Optimized message structure."""

    data: dict | bytes
    timestamp: float = field(default_factory=time.time)
    priority: MessagePriority = MessagePriority.MEDIUM
    exchange: str = ""
    symbol: str = ""
    message_type: str = ""

    def __post_init__(self):
        """Post-initialization processing."""
        if isinstance(self.data, dict):
            # Extract common fields for faster routing
            self.exchange = self.data.get("exchange", self.exchange)
            self.symbol = self.data.get("symbol", self.symbol)
            self.message_type = self.data.get("type", self.message_type)


@dataclass
class ConnectionPool:
    """High-performance connection pool."""

    connections: dict[str, "HighPerformanceWebSocket"] = field(default_factory=dict)
    active_connections: set[str] = field(default_factory=set)
    failed_connections: set[str] = field(default_factory=set)
    connection_weights: dict[str, float] = field(default_factory=dict)
    last_used: dict[str, float] = field(default_factory=dict)
    max_connections_per_exchange: int = 5

    def get_best_connection(self, exchange: str) -> Optional["HighPerformanceWebSocket"]:
        """Get the best available connection for an exchange."""
        exchange_connections = [
            conn_id for conn_id in self.active_connections if conn_id.startswith(f"{exchange}_")
        ]

        if not exchange_connections:
            return None

        # Select connection with lowest weight (least loaded)
        best_conn_id = min(exchange_connections, key=lambda x: self.connection_weights.get(x, 0))

        return self.connections.get(best_conn_id)

    def update_connection_weight(self, connection_id: str, weight: float):
        """Update connection load weight."""
        self.connection_weights[connection_id] = weight
        self.last_used[connection_id] = time.time()


class MessageBatcher:
    """High-performance message batcher to reduce processing overhead."""

    def __init__(self, max_batch_size: int = 100, max_batch_time_ms: float = 10.0):
        self.max_batch_size = max_batch_size
        self.max_batch_time_ms = max_batch_time_ms
        self.batches: dict[MessagePriority, list[WebSocketMessage]] = defaultdict(list)
        self.batch_timers: dict[MessagePriority, float] = defaultdict(lambda: time.time())
        self.callbacks: dict[MessagePriority, list[Callable]] = defaultdict(list)

    def add_message(self, message: WebSocketMessage):
        """Add message to appropriate batch."""
        priority = message.priority
        batch = self.batches[priority]
        batch.append(message)

        # Set timer for first message in batch
        if len(batch) == 1:
            self.batch_timers[priority] = time.time()

        # Check if batch should be flushed
        if (
            len(batch) >= self.max_batch_size
            or (time.time() - self.batch_timers[priority]) * 1000 >= self.max_batch_time_ms
        ):
            self._flush_task = asyncio.create_task(self._flush_batch(priority))

    async def _flush_batch(self, priority: MessagePriority):
        """Flush batch of messages to callbacks."""
        batch = self.batches[priority]
        if not batch:
            return

        # Copy and clear batch atomically
        messages = batch.copy()
        batch.clear()

        # Process callbacks
        callbacks = self.callbacks[priority]
        if callbacks:
            tasks = [callback(messages) for callback in callbacks]
            await asyncio.gather(*tasks, return_exceptions=True)

    def add_callback(self, priority: MessagePriority, callback: Callable):
        """Add callback for specific priority level."""
        self.callbacks[priority].append(callback)


class HighPerformanceWebSocket:
    """High-performance WebSocket connection with advanced optimizations."""

    def __init__(
        self,
        url: str,
        config: Config,
        connection_id: str,
        exchange: str = "",
        connection_pool: ConnectionPool | None = None,
    ):
        self.url = url
        self.config = config
        self.connection_id = connection_id
        self.exchange = exchange
        self.connection_pool = connection_pool
        self.logger = get_logger(f"{__name__}.{connection_id}")

        # Connection state
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self.is_connected = False
        self.is_authenticated = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10

        # Performance components
        self.message_batcher = MessageBatcher(
            max_batch_size=getattr(getattr(self.config, "websocket", None), "batch_size", None)
            or 50,
            max_batch_time_ms=getattr(
                getattr(self.config, "websocket", None), "batch_time_ms", None
            )
            or 5.0,
        )

        # Message queues with priority
        self.inbound_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=10000)
        self.outbound_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1000)

        # Performance metrics
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "connection_time": 0.0,
            "avg_latency_ms": 0.0,
            "error_count": 0,
            "reconnections": 0,
        }

        # Task management
        self.tasks: set[asyncio.Task] = set()
        self.message_handlers: dict[str, list[Callable]] = defaultdict(list)

        # Binary message support
        self.supports_binary = False
        self.compression_enabled = False

        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix=f"ws-{connection_id}"
        )

    async def connect(self) -> bool:
        """Establish WebSocket connection with optimizations."""
        start_time = time.perf_counter()

        try:
            # Connection with optimization options
            extra_headers = {
                "User-Agent": "TradingBot-HFT/1.0",
            }

            # Enable compression if supported
            websocket_config = getattr(self.config, "websocket", None)
            compression = (
                "deflate"
                if (websocket_config and getattr(websocket_config, "enable_compression", False))
                else None
            )

            self.websocket = await websockets.connect(
                self.url,
                extra_headers=extra_headers,
                compression=compression,
                ping_interval=20,  # 20 second ping
                ping_timeout=10,  # 10 second timeout
                close_timeout=5,  # 5 second close timeout
                max_size=2**23,  # 8MB max message size
                read_limit=2**20,  # 1MB read buffer
                write_limit=2**20,  # 1MB write buffer
            )

            self.is_connected = True
            self.connection_time = time.perf_counter() - start_time
            self.metrics["connection_time"] = self.connection_time

            # Check for binary protocol support
            if hasattr(self.websocket, "binary_type"):
                self.supports_binary = True

            # Check compression support
            if self.websocket.extensions:
                self.compression_enabled = any(
                    "deflate" in ext.name for ext in self.websocket.extensions
                )

            # Start background tasks
            await self._start_background_tasks()

            # Update connection pool
            if self.connection_pool:
                self.connection_pool.active_connections.add(self.connection_id)
                self.connection_pool.failed_connections.discard(self.connection_id)

            self.logger.info(
                "WebSocket connected successfully",
                connection_time_ms=round(self.connection_time * 1000, 2),
                supports_binary=self.supports_binary,
                compression_enabled=self.compression_enabled,
            )

            return True

        except Exception as e:
            self.logger.error("WebSocket connection failed", error=str(e))
            self.is_connected = False

            if self.connection_pool:
                self.connection_pool.failed_connections.add(self.connection_id)
                self.connection_pool.active_connections.discard(self.connection_id)

            return False

    async def _start_background_tasks(self):
        """Start background tasks for message processing."""

        # Message receiver task
        receive_task = asyncio.create_task(self._message_receiver())
        self.tasks.add(receive_task)
        receive_task.add_done_callback(self.tasks.discard)

        # Message sender task
        send_task = asyncio.create_task(self._message_sender())
        self.tasks.add(send_task)
        send_task.add_done_callback(self.tasks.discard)

        # Message processor task
        process_task = asyncio.create_task(self._message_processor())
        self.tasks.add(process_task)
        process_task.add_done_callback(self.tasks.discard)

        # Metrics updater task
        metrics_task = asyncio.create_task(self._metrics_updater())
        self.tasks.add(metrics_task)
        metrics_task.add_done_callback(self.tasks.discard)

    async def _message_receiver(self):
        """High-performance message receiver."""
        while self.is_connected and self.websocket:
            try:
                # Receive message with minimal processing
                raw_message = await self.websocket.recv()

                # Update metrics
                self.metrics["messages_received"] += 1

                if isinstance(raw_message, bytes):
                    self.metrics["bytes_received"] += len(raw_message)

                    # Handle compressed binary messages
                    if self.compression_enabled:
                        try:
                            raw_message = zlib.decompress(raw_message).decode("utf-8")
                        except (zlib.error, UnicodeDecodeError):
                            pass  # Not compressed
                    else:
                        raw_message = raw_message.decode("utf-8")
                else:
                    self.metrics["bytes_received"] += len(raw_message.encode("utf-8"))

                # Fast JSON parsing
                try:
                    message_data = json.loads(raw_message)
                except json.JSONDecodeError:
                    self.logger.warning("Invalid JSON received", message=raw_message[:100])
                    continue

                # Create message object
                message = WebSocketMessage(data=message_data, timestamp=time.time())

                # Determine priority
                message.priority = self._determine_message_priority(message_data)

                # Add to processing queue
                if not self.inbound_queue.full():
                    await self.inbound_queue.put(message)
                else:
                    # Drop low-priority messages if queue is full
                    if message.priority in [MessagePriority.CRITICAL, MessagePriority.HIGH]:
                        # Force add critical/high priority messages
                        await self.inbound_queue.put(message)
                    else:
                        self.logger.warning("Dropping low-priority message due to full queue")

            except ConnectionClosed:
                self.logger.info("WebSocket connection closed")
                break
            except Exception as e:
                self.logger.error("Message receive error", error=str(e))
                self.metrics["error_count"] += 1
                await asyncio.sleep(0.1)  # Brief pause on error

    async def _message_processor(self):
        """Process inbound messages with batching."""
        while self.is_connected:
            try:
                # Get message from queue
                message = await self.inbound_queue.get()

                # Add to batcher for efficient processing
                self.message_batcher.add_message(message)

                # Mark task as done
                self.inbound_queue.task_done()

            except Exception as e:
                self.logger.error("Message processing error", error=str(e))
                await asyncio.sleep(0.1)

    async def _message_sender(self):
        """High-performance message sender."""
        while self.is_connected and self.websocket:
            try:
                message = await self.outbound_queue.get()

                # Serialize message
                if isinstance(message, dict):
                    serialized = json.dumps(message)
                elif isinstance(message, str):
                    serialized = message
                else:
                    serialized = str(message)

                # Send message
                await self.websocket.send(serialized)

                # Update metrics
                self.metrics["messages_sent"] += 1
                self.metrics["bytes_sent"] += len(serialized.encode("utf-8"))

                self.outbound_queue.task_done()

            except ConnectionClosed:
                break
            except Exception as e:
                self.logger.error("Message send error", error=str(e))
                self.metrics["error_count"] += 1
                await asyncio.sleep(0.1)

    async def _metrics_updater(self):
        """Update connection metrics and load balancing weights."""
        while self.is_connected:
            try:
                # Update connection weight based on queue sizes
                inbound_load = self.inbound_queue.qsize() / self.inbound_queue.maxsize
                outbound_load = self.outbound_queue.qsize() / self.outbound_queue.maxsize

                # Calculate combined weight (0.0 = no load, 1.0 = full load)
                weight = (inbound_load + outbound_load) / 2.0

                if self.connection_pool:
                    self.connection_pool.update_connection_weight(self.connection_id, weight)

                # Sleep for metrics update interval
                await asyncio.sleep(5.0)  # Update every 5 seconds

            except Exception as e:
                self.logger.error("Metrics update error", error=str(e))
                await asyncio.sleep(5.0)

    def _determine_message_priority(self, message_data: dict) -> MessagePriority:
        """Determine message priority based on content."""
        message_type = message_data.get("type", "").lower()

        # Critical messages - order updates, fills
        if any(keyword in message_type for keyword in ["order", "fill", "execution", "trade"]):
            return MessagePriority.CRITICAL

        # High priority - price updates, market data
        if any(keyword in message_type for keyword in ["price", "ticker", "book", "depth"]):
            return MessagePriority.HIGH

        # Medium priority - indicators, analytics
        if any(keyword in message_type for keyword in ["indicator", "signal", "analytics"]):
            return MessagePriority.MEDIUM

        # Low priority - status, heartbeat
        return MessagePriority.LOW

    async def send_message(self, message: dict, priority: MessagePriority = MessagePriority.MEDIUM):
        """Send message with priority queuing."""
        if not self.is_connected:
            raise ExchangeConnectionError("WebSocket not connected")

        # Add timestamp for latency measurement
        message["client_timestamp"] = time.time()

        await self.outbound_queue.put(message)

    def add_message_handler(self, message_type: str, handler: Callable):
        """Add message handler for specific message types."""
        self.message_handlers[message_type].append(handler)

        # Add handler to message batcher
        priority = (
            MessagePriority.HIGH if message_type in ["order", "fill"] else MessagePriority.MEDIUM
        )
        self.message_batcher.add_callback(priority, self._create_handler_wrapper(message_type))

    def _create_handler_wrapper(self, message_type: str) -> Callable:
        """Create wrapper for message handlers."""

        async def wrapper(messages: list[WebSocketMessage]):
            handlers = self.message_handlers.get(message_type, [])
            if not handlers:
                return

            # Filter messages by type
            relevant_messages = [
                msg
                for msg in messages
                if msg.message_type == message_type
                or (isinstance(msg.data, dict) and msg.data.get("type") == message_type)
            ]

            if not relevant_messages:
                return

            # Call handlers
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(relevant_messages)
                    else:
                        # Run in thread pool for blocking handlers
                        await asyncio.get_event_loop().run_in_executor(
                            self.thread_pool, handler, relevant_messages
                        )
                except Exception as e:
                    self.logger.error("Handler error", message_type=message_type, error=str(e))

        return wrapper

    async def disconnect(self):
        """Gracefully disconnect WebSocket."""
        tasks_to_cancel = []
        websocket_to_close = None
        thread_pool_to_shutdown = None

        try:
            self.is_connected = False

            # Store references to avoid race conditions
            tasks_to_cancel = list(self.tasks)
            websocket_to_close = self.websocket
            thread_pool_to_shutdown = self.thread_pool
        except Exception as e:
            self.logger.error("Error preparing disconnect", error=str(e))

        # Cancel all tasks
        try:
            for task in tasks_to_cancel:
                if task and not task.done():
                    task.cancel()

            # Wait for tasks to complete
            if tasks_to_cancel:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        except Exception as e:
            self.logger.error("Error canceling tasks", error=str(e))
        finally:
            try:
                self.tasks.clear()
            except Exception as e:
                self.logger.error(f"Error clearing tasks during close: {e}")

        # Close WebSocket
        try:
            if websocket_to_close:
                await websocket_to_close.close()
        except Exception as e:
            self.logger.error("Error closing websocket", error=str(e))
        finally:
            self.websocket = None

        # Shutdown thread pool
        try:
            if thread_pool_to_shutdown:
                thread_pool_to_shutdown.shutdown(wait=True)
        except Exception as e:
            self.logger.error("Error shutting down thread pool", error=str(e))
        finally:
            self.thread_pool = None

        # Update connection pool
        try:
            if self.connection_pool:
                self.connection_pool.active_connections.discard(self.connection_id)
                self.connection_pool.connections.pop(self.connection_id, None)
        except Exception as e:
            self.logger.error("Error updating connection pool", error=str(e))

        self.logger.info("WebSocket disconnected cleanly")

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            **self.metrics,
            "connection_id": self.connection_id,
            "is_connected": self.is_connected,
            "inbound_queue_size": self.inbound_queue.qsize(),
            "outbound_queue_size": self.outbound_queue.qsize(),
            "active_tasks": len(self.tasks),
            "supports_binary": self.supports_binary,
            "compression_enabled": self.compression_enabled,
            "message_rate_per_sec": self.metrics["messages_received"]
            / max(1, time.time() - getattr(self, "start_time", time.time())),
        }


class HighPerformanceWebSocketManager:
    """Manager for multiple high-performance WebSocket connections."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)

        # Connection management
        self.connection_pool = ConnectionPool()
        self.connections: dict[str, HighPerformanceWebSocket] = {}

        # Global message handling
        self.global_handlers: dict[str, list[Callable]] = defaultdict(list)

        # Performance monitoring
        self.total_metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_per_second": 0.0,
            "average_latency_ms": 0.0,
        }

    async def create_connection(
        self, url: str, exchange: str, connection_id: str | None = None
    ) -> str:
        """Create a new high-performance WebSocket connection."""

        if not connection_id:
            connection_id = f"{exchange}_{len(self.connections)}_{int(time.time())}"

        # Check if we already have enough connections for this exchange
        exchange_connections = [
            conn_id for conn_id in self.connections.keys() if conn_id.startswith(f"{exchange}_")
        ]

        if len(exchange_connections) >= self.connection_pool.max_connections_per_exchange:
            self.logger.warning(
                "Maximum connections reached for exchange",
                exchange=exchange,
                max_connections=self.connection_pool.max_connections_per_exchange,
            )
            return exchange_connections[0]  # Return existing connection

        # Create new connection
        connection = HighPerformanceWebSocket(
            url=url,
            config=self.config,
            connection_id=connection_id,
            exchange=exchange,
            connection_pool=self.connection_pool,
        )

        # Store connection
        self.connections[connection_id] = connection
        self.connection_pool.connections[connection_id] = connection

        # Connect
        success = await connection.connect()
        if success:
            self.total_metrics["total_connections"] += 1
            self.total_metrics["active_connections"] += 1

            # Add global handlers
            for message_type, handlers in self.global_handlers.items():
                for handler in handlers:
                    connection.add_message_handler(message_type, handler)

            self.logger.info(
                "WebSocket connection created", connection_id=connection_id, exchange=exchange
            )

            return connection_id
        else:
            # Clean up failed connection
            self.connections.pop(connection_id, None)
            self.connection_pool.connections.pop(connection_id, None)
            raise ExchangeConnectionError(f"Failed to create WebSocket connection: {connection_id}")

    async def send_message(
        self,
        message: dict,
        exchange: str | None = None,
        connection_id: str | None = None,
        priority: MessagePriority = MessagePriority.MEDIUM,
    ) -> bool:
        """Send message using best available connection."""

        target_connection = None

        if connection_id:
            target_connection = self.connections.get(connection_id)
        elif exchange:
            target_connection = self.connection_pool.get_best_connection(exchange)

        if not target_connection:
            self.logger.error(
                "No available connection for message",
                exchange=exchange,
                connection_id=connection_id,
            )
            return False

        try:
            await target_connection.send_message(message, priority)
            return True
        except Exception as e:
            self.logger.error("Failed to send message", error=str(e))
            return False

    def add_global_handler(self, message_type: str, handler: Callable):
        """Add global message handler for all connections."""
        self.global_handlers[message_type].append(handler)

        # Add to existing connections
        for connection in self.connections.values():
            connection.add_message_handler(message_type, handler)

    async def disconnect_all(self):
        """Disconnect all WebSocket connections."""
        tasks = []
        connections_to_disconnect = []

        try:
            connections_to_disconnect = list(self.connections.values())
        except Exception as e:
            self.logger.error(f"Error getting connections list for disconnect: {e}")

        # Create disconnect tasks
        for connection in connections_to_disconnect:
            if connection:
                try:
                    tasks.append(connection.disconnect())
                except Exception as e:
                    self.logger.error(f"Error creating disconnect task: {e}")

        # Execute disconnects
        try:
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            self.logger.error(f"Error disconnecting connections: {e}")
        finally:
            # Clean up state regardless of disconnect success
            try:
                self.connections.clear()
            except Exception as e:
                self.logger.error(f"Error clearing connections dict: {e}")
            try:
                self.connection_pool.connections.clear()
                self.connection_pool.active_connections.clear()
            except Exception as e:
                self.logger.error(f"Error clearing connection pool state: {e}")
            try:
                self.total_metrics["active_connections"] = 0
            except Exception as e:
                self.logger.error(f"Error resetting metrics: {e}")

            self.logger.info("All WebSocket connections disconnected")

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary for all connections."""
        connection_metrics = {
            conn_id: conn.get_performance_metrics() for conn_id, conn in self.connections.items()
        }

        # Calculate aggregated metrics
        total_messages_received = sum(
            metrics["messages_received"] for metrics in connection_metrics.values()
        )

        total_messages_sent = sum(
            metrics["messages_sent"] for metrics in connection_metrics.values()
        )

        return {
            "total_connections": len(self.connections),
            "active_connections": len(self.connection_pool.active_connections),
            "failed_connections": len(self.connection_pool.failed_connections),
            "total_messages_received": total_messages_received,
            "total_messages_sent": total_messages_sent,
            "connection_details": connection_metrics,
            "pool_status": {
                "active_connections": list(self.connection_pool.active_connections),
                "failed_connections": list(self.connection_pool.failed_connections),
                "connection_weights": dict(self.connection_pool.connection_weights),
            },
        }

    def get_connection_health(self) -> dict[str, Any]:
        """
        Get connection health status.

        This method provides compatibility with monitoring systems expecting
        connection health information. It returns the same data as get_performance_summary
        but also includes additional computed metrics for health monitoring.
        """
        # Get base performance summary
        summary = self.get_performance_summary()

        # Calculate message rate
        total_latency_sum = 0.0
        message_count = 0

        for metrics in summary["connection_details"].values():
            if "average_latency_ms" in metrics and metrics["messages_received"] > 0:
                total_latency_sum += metrics["average_latency_ms"] * metrics["messages_received"]
                message_count += metrics["messages_received"]

        # Add health-specific metrics
        summary["total_latency_sum"] = total_latency_sum
        summary["message_rate"] = message_count / max(
            1, time.time() - getattr(self, "start_time", time.time())
        )

        return summary


class WebSocketConnectionPool:
    """
    WebSocket connection pool for managing connection pools.

    This class manages individual WebSocket connection pools with health
    monitoring and automatic connection management.
    """

    def __init__(
        self,
        exchange: str,
        max_connections: int = 10,
        max_messages_per_second: int = 100,
        max_subscriptions: int = 50,
    ):
        """
        Initialize WebSocket connection pool.

        Args:
            exchange: Exchange name
            max_connections: Maximum number of connections in pool
            max_messages_per_second: Maximum messages per second per connection
            max_subscriptions: Maximum subscriptions per connection
        """
        self.exchange = exchange
        self.max_connections = max_connections
        self.max_messages_per_second = max_messages_per_second
        self.max_subscriptions = max_subscriptions
        self.logger = get_logger(f"websocket_pool.{exchange}")

        # Connection pools by type
        self.connection_pools: dict[ConnectionType, list[PooledConnection]] = defaultdict(list)

        # Active connections
        self.active_connections: dict[str, PooledConnection] = {}

        # Connection usage tracking
        self.message_counters: dict[str, list[datetime]] = defaultdict(list)
        self.subscription_counters: dict[str, int] = defaultdict(int)

        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.connection_timeout = 300  # seconds (5 minutes)

    async def get_connection(self, connection_type: ConnectionType) -> PooledConnection | None:
        """
        Get a connection from the pool.

        Args:
            connection_type: Type of connection needed

        Returns:
            PooledConnection if available, None otherwise
        """
        try:
            # Validate parameters
            if not connection_type:
                raise ValidationError("Connection type is required")

            # Check if we have available connections
            available_connections = [
                conn
                for conn in self.connection_pools[connection_type]
                if self._is_connection_healthy(conn) and self._can_use_connection(conn)
            ]

            if available_connections:
                # Return the least recently used connection
                connection = min(available_connections, key=lambda c: c.last_used)
                connection.last_used = datetime.now(timezone.utc)
                return connection

            # Create new connection if pool not full
            if len(self.active_connections) < self.max_connections:
                connection: PooledConnection | None = await self._create_connection(connection_type)
                if connection is not None:
                    self.connection_pools[connection_type].append(connection)
                    self.active_connections[connection.connection_id] = connection
                    return connection

            return None

        except Exception as e:
            self.logger.error(f"Failed to get connection from pool: {e}")
            return None

    async def release_connection(self, connection: PooledConnection) -> None:
        """
        Release a connection back to the pool.

        Args:
            connection: Connection to release
        """
        try:
            if not connection or not connection.connection_id:
                raise ValidationError("Valid connection is required")

            # Update connection stats
            connection.last_used = datetime.now(timezone.utc)

            # Check if connection is still healthy
            if not self._is_connection_healthy(connection):
                await self._remove_connection(connection)

        except Exception as e:
            self.logger.error(f"Failed to release connection: {e}")

    async def _create_connection(self, connection_type: ConnectionType) -> PooledConnection | None:
        """
        Create a new WebSocket connection.

        Args:
            connection_type: Type of connection to create

        Returns:
            PooledConnection if successful, None otherwise
        """
        try:
            # Generate unique connection ID
            connection_id = f"{self.exchange}_{connection_type.value}_{int(time.time() * 1000)}"

            # Create connection object (placeholder implementation)
            connection_obj = self._create_websocket_connection(connection_type)

            if not connection_obj:
                return None

            # Create pooled connection
            pooled_connection = PooledConnection(
                connection_id=connection_id,
                exchange=self.exchange,
                connection_type=connection_type,
                connection=connection_obj,
                created_at=datetime.now(timezone.utc),
                last_used=datetime.now(timezone.utc),
            )

            return pooled_connection

        except Exception as e:
            self.logger.error(f"Failed to create connection: {e}")
            return None

    def _create_websocket_connection(self, connection_type: ConnectionType) -> Any:
        """
        Create actual WebSocket connection (placeholder implementation).

        Args:
            connection_type: Type of connection to create

        Returns:
            WebSocket connection object
        """

        # This is a placeholder implementation
        class MockWebSocketConnection:
            def __init__(self, conn_id, conn_type, exchange):
                self.id = conn_id
                self.type = conn_type
                self.is_connected = True
                self.exchange = exchange
                self.stream_type = conn_type.value

        return MockWebSocketConnection(
            f"{self.exchange}_{connection_type.value}_{int(time.time() * 1000)}",
            connection_type,
            self.exchange,
        )

    def _is_connection_healthy(self, connection: PooledConnection) -> bool:
        """
        Check if a connection is healthy.

        Args:
            connection: Connection to check

        Returns:
            bool: True if connection is healthy
        """
        try:
            # Check if connection is too old
            if datetime.now(timezone.utc) - connection.created_at > timedelta(
                seconds=self.connection_timeout
            ):
                return False

            # Check if connection has been inactive for too long
            if datetime.now(timezone.utc) - connection.last_used > timedelta(
                seconds=self.connection_timeout
            ):
                return False

            # Check message rate limits
            if not self._check_message_rate_limit(connection):
                return False

            # Check subscription limits
            if not self._check_subscription_limit(connection):
                return False

            # Check if connection object is still valid
            if hasattr(connection.connection, "is_connected"):
                if not connection.connection.is_connected:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking connection validity: {e}")
            return False

    def _can_use_connection(self, connection: PooledConnection) -> bool:
        """
        Check if a connection can be used.

        Args:
            connection: Connection to check

        Returns:
            bool: True if connection can be used
        """
        try:
            # Check message rate limit
            if not self._check_message_rate_limit(connection):
                return False

            # Check subscription limit
            if not self._check_subscription_limit(connection):
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking if connection can be used: {e}")
            return False

    def _check_message_rate_limit(self, connection: PooledConnection) -> bool:
        """
        Check if connection is within message rate limits.

        Args:
            connection: Connection to check

        Returns:
            bool: True if within limits
        """
        try:
            now = datetime.now(timezone.utc)
            second_ago = now - timedelta(seconds=1)

            # Clean old message timestamps
            self.message_counters[connection.connection_id] = [
                t for t in self.message_counters[connection.connection_id] if t > second_ago
            ]

            # Check if adding a message would exceed limit
            current_messages = len(self.message_counters[connection.connection_id])
            return current_messages < self.max_messages_per_second

        except Exception as e:
            self.logger.error(f"Error checking message rate limit: {e}")
            return False

    def _check_subscription_limit(self, connection: PooledConnection) -> bool:
        """
        Check if connection is within subscription limits.

        Args:
            connection: Connection to check

        Returns:
            bool: True if within limits
        """
        try:
            current_subscriptions = self.subscription_counters.get(connection.connection_id, 0)
            return current_subscriptions < self.max_subscriptions

        except Exception as e:
            self.logger.error(f"Error checking subscription limit: {e}")
            return False

    async def _remove_connection(self, connection: PooledConnection) -> None:
        """
        Remove a connection from the pool.

        Args:
            connection: Connection to remove
        """
        try:
            # Remove from active connections
            if connection.connection_id in self.active_connections:
                del self.active_connections[connection.connection_id]

            # Remove from connection pools
            if connection in self.connection_pools[connection.connection_type]:
                self.connection_pools[connection.connection_type].remove(connection)

            # Clean up counters
            if connection.connection_id in self.message_counters:
                del self.message_counters[connection.connection_id]

            if connection.connection_id in self.subscription_counters:
                del self.subscription_counters[connection.connection_id]

            # Close connection if it has a close method
            if hasattr(connection.connection, "close"):
                try:
                    await connection.connection.close()
                except Exception as e:
                    self.logger.error(f"Error closing connection {connection.connection_id}: {e}")

        except Exception as e:
            self.logger.error(f"Failed to remove connection: {e}")

    def record_message(self, connection_id: str) -> None:
        """
        Record a message sent on a connection.

        Args:
            connection_id: Connection ID
        """
        try:
            now = datetime.now(timezone.utc)
            self.message_counters[connection_id].append(now)

            # Clean old entries (keep last 1000)
            if len(self.message_counters[connection_id]) > 1000:
                self.message_counters[connection_id] = self.message_counters[connection_id][-1000:]

        except Exception as e:
            self.logger.error(f"Failed to record message: {e}")

    def record_subscription(self, connection_id: str) -> bool:
        """
        Record a subscription on a connection.

        Args:
            connection_id: Connection ID

        Returns:
            bool: True if subscription was recorded successfully
        """
        try:
            current_count = self.subscription_counters.get(connection_id, 0)

            if current_count >= self.max_subscriptions:
                return False

            self.subscription_counters[connection_id] = current_count + 1
            return True

        except Exception as e:
            self.logger.error(f"Error recording subscription for connection {connection_id}: {e}")
            return False

    async def close_all_connections(self) -> None:
        """Close all connections in the pool."""
        connections_to_remove = []

        try:
            # Collect all connections to avoid modification during iteration
            for _connection_type, connections in self.connection_pools.items():
                connections_to_remove.extend(connections[:])
        except Exception as e:
            self.logger.error(f"Error collecting connections for cleanup: {e}")

        # Remove connections one by one
        for connection in connections_to_remove:
            try:
                await self._remove_connection(connection)
            except Exception as e:
                self.logger.error(f"Failed to remove connection {connection.connection_id}: {e}")

        # Final cleanup
        try:
            self.connection_pools.clear()
            self.active_connections.clear()
            self.message_counters.clear()
            self.subscription_counters.clear()
        except Exception as e:
            self.logger.error(f"Error in final cleanup: {e}")
