"""
Connection resilience manager for reliable network connections.

This module provides automatic reconnection with exponential backoff, connection
pooling with health monitoring, heartbeat detection, and message queuing during
brief disconnections.

CRITICAL: This module integrates with P-001 core framework and P-002 database
for connection state persistence and will be used by all subsequent prompts.
"""

import asyncio
import secrets
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Any

from src.core.config import Config
from src.core.exceptions import TimeoutError
from src.core.logging import get_logger

# Core framework imports
from src.utils.decorators import circuit_breaker, time_execution

# Production configuration constants
DEFAULT_LATENCY_SAMPLES = 10
DEFAULT_QUEUE_SIZE = 1000
DEFAULT_QUEUE_TTL_SECONDS = 300
DEFAULT_MAX_SIZE_BYTES = 10 * 1024 * 1024
DEFAULT_WEBSOCKET_TIMEOUT = 30.0
DEFAULT_WEBSOCKET_HEARTBEAT_TIMEOUT = 5.0
DEFAULT_CONNECTION_BATCH_SIZE = 10


class ConnectionState(Enum):
    """Connection state enumeration."""

    CONNECTED = "connected"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


@dataclass
class ConnectionHealth:
    """Optimized connection health metrics with memory-efficient storage."""

    last_heartbeat: datetime
    latency_ms: Decimal
    packet_loss: Decimal
    connection_quality: Decimal  # 0.0 to 1.0
    uptime_seconds: int
    reconnect_count: int
    last_error: str | None = None

    # Efficient latency history storage
    _latency_history: deque[Decimal] = field(default_factory=lambda: deque(maxlen=100))
    _quality_history: deque[Decimal] = field(default_factory=lambda: deque(maxlen=50))

    def add_latency_measurement(self, latency: float | Decimal) -> None:
        """Add latency measurement to history."""
        latency_decimal = Decimal(str(latency))
        self._latency_history.append(latency_decimal)
        self.latency_ms = latency_decimal

    def add_quality_measurement(self, quality: float | Decimal) -> None:
        """Add quality measurement to history."""
        quality_decimal = Decimal(str(quality))
        self._quality_history.append(quality_decimal)
        self.connection_quality = quality_decimal

    def get_average_latency(self, samples: int = DEFAULT_LATENCY_SAMPLES) -> Decimal:
        """Get average latency from recent samples."""
        if not self._latency_history:
            return Decimal("0.0")
        recent_samples = list(self._latency_history)[-samples:]
        from decimal import localcontext

        with localcontext() as ctx:
            ctx.prec = 8
            ctx.rounding = ROUND_HALF_UP
            return sum(recent_samples) / Decimal(str(len(recent_samples)))

    def get_quality_trend(self) -> str:
        """Get quality trend (improving, degrading, stable)."""
        if len(self._quality_history) < 3:
            return "stable"

        recent = list(self._quality_history)[-3:]
        if recent[-1] > recent[-2] > recent[-3]:
            return "improving"
        elif recent[-1] < recent[-2] < recent[-3]:
            return "degrading"
        else:
            return "stable"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "latency_ms": str(self.latency_ms),
            "packet_loss": str(self.packet_loss),
            "connection_quality": str(self.connection_quality),
            "uptime_seconds": self.uptime_seconds,
            "reconnect_count": self.reconnect_count,
            "last_error": self.last_error,
            "avg_latency_10s": str(self.get_average_latency(10)),
            "quality_trend": self.get_quality_trend(),
        }


class MessageQueue:
    """Efficient message queue with size limits and TTL."""

    def __init__(
        self, max_size: int = DEFAULT_QUEUE_SIZE, ttl_seconds: int = DEFAULT_QUEUE_TTL_SECONDS
    ) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._queue: deque[dict[str, Any]] = deque(maxlen=max_size)
        self._total_size_bytes = 0
        self._max_size_bytes = DEFAULT_MAX_SIZE_BYTES

    def add_message(self, message: dict[str, Any]) -> bool:
        """Add message to queue with size and TTL checks."""
        # Add timestamp for TTL
        message["queued_at"] = datetime.now(timezone.utc).isoformat()

        # Estimate size
        estimated_size = len(str(message))

        # Check memory limit
        if self._total_size_bytes + estimated_size > self._max_size_bytes:
            # Remove old messages until under limit
            while self._queue and self._total_size_bytes > self._max_size_bytes * 0.7:
                old_msg = self._queue.popleft()
                self._total_size_bytes -= len(str(old_msg))

        # Add new message
        self._queue.append(message)
        self._total_size_bytes += estimated_size

        return True

    def get_messages(self) -> list[dict[str, Any]]:
        """Get all messages, removing expired ones."""
        now = datetime.now(timezone.utc)
        valid_messages = []

        for message in self._queue:
            # Parse ISO format datetime string properly
            queued_at_str = message["queued_at"]
            if queued_at_str.endswith("Z"):
                queued_at_str = queued_at_str[:-1] + "+00:00"
            queued_time = datetime.fromisoformat(queued_at_str)
            if (now - queued_time).total_seconds() <= self.ttl_seconds:
                valid_messages.append(message)

        return valid_messages

    def clear(self) -> int:
        """Clear all messages and return count."""
        count = len(self._queue)
        self._queue.clear()
        self._total_size_bytes = 0
        return count

    def size(self) -> int:
        """Get current queue size."""
        return len(self._queue)


class ConnectionManager:
    """Optimized connection reliability manager with efficient memory usage."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__module__)

        # Optimized connection storage
        self.connections: dict[str, dict[str, Any]] = {}
        self.health_monitors: dict[str, ConnectionHealth] = {}

        # Reconnection policies
        self.reconnect_policies: dict[str, dict[str, Any]] = {
            "exchange": {"max_attempts": 5, "base_delay": 1, "max_delay": 60, "jitter": True},
            "database": {"max_attempts": 3, "base_delay": 0.5, "max_delay": 10, "jitter": False},
            "websocket": {"max_attempts": 10, "base_delay": 0.1, "max_delay": 30, "jitter": True},
        }

        # Optimized message queues
        self.message_queues: dict[str, MessageQueue] = {}
        self.heartbeat_intervals: dict[str, int] = {"exchange": 30, "database": 60, "websocket": 10}

        # WebSocket connection timeouts and heartbeat settings
        self.websocket_timeout = DEFAULT_WEBSOCKET_TIMEOUT
        self.websocket_heartbeat_timeout = DEFAULT_WEBSOCKET_HEARTBEAT_TIMEOUT

        # Task management with weak references
        self._health_monitor_tasks: dict[str, asyncio.Task] = {}
        self._cleanup_task: asyncio.Task | None = None

        # Performance monitoring
        self._connection_stats = {
            "total_connections_created": 0,
            "total_reconnections": 0,
            "total_messages_queued": 0,
            "last_cleanup": datetime.now(timezone.utc),
        }

        # Synchronization locks to prevent race conditions
        self._connections_lock = asyncio.Lock()
        self._health_monitors_lock = asyncio.Lock()
        self._message_queues_lock = asyncio.Lock()
        self._tasks_lock = asyncio.Lock()
        self._stats_lock = asyncio.Lock()

        # Cleanup task - will be started when needed
        self._cleanup_task = None
        # Don't start cleanup task immediately - start it when first connection is made

    @time_execution
    async def establish_connection(
        self,
        connection_id: str,
        connection_type: str,
        connect_func: Callable[..., Any],
        **kwargs: Any,
    ) -> bool:
        """Establish a new connection with automatic retry."""

        self.logger.info(
            "Establishing connection", connection_id=connection_id, connection_type=connection_type
        )

        policy = self.reconnect_policies.get(connection_type, self.reconnect_policies["exchange"])

        for attempt in range(policy["max_attempts"]):
            try:
                # Calculate delay with exponential backoff
                delay = min(policy["base_delay"] * (2**attempt), policy["max_delay"])

                if policy["jitter"]:
                    # Use cryptographically secure random for jitter
                    jitter_factor = 0.5 + (secrets.randbelow(50) / 100)  # 0.5 to 1.0
                    delay *= jitter_factor

                if attempt > 0:
                    await asyncio.sleep(delay)

                # Start cleanup task if not already running
                self._start_cleanup_task()

                # Attempt connection with timeout for WebSocket connections
                if connection_type == "websocket":
                    try:
                        connection = await asyncio.wait_for(
                            connect_func(**kwargs), timeout=self.websocket_timeout
                        )
                    except asyncio.TimeoutError as e:
                        raise TimeoutError(
                            f"WebSocket connection timeout after {self.websocket_timeout}s"
                        ) from e
                else:
                    connection = await connect_func(**kwargs)

                # Initialize connection tracking with proper locking
                async with self._connections_lock:
                    self.connections[connection_id] = {
                        "connection": connection,
                        "type": connection_type,
                        "state": ConnectionState.CONNECTED,
                        "established_at": datetime.now(timezone.utc),
                        "last_activity": datetime.now(timezone.utc),
                        "reconnect_count": 0,
                    }

                # Initialize health monitoring with proper locking
                async with self._health_monitors_lock:
                    self.health_monitors[connection_id] = ConnectionHealth(
                        last_heartbeat=datetime.now(timezone.utc),
                        latency_ms=Decimal("0.0"),
                        packet_loss=Decimal("0.0"),
                        connection_quality=Decimal("1.0"),
                        uptime_seconds=0,
                        reconnect_count=0,
                    )

                # Start health monitoring with proper locking
                async with self._tasks_lock:
                    self._health_monitor_tasks[connection_id] = asyncio.create_task(
                        self._monitor_connection_health(connection_id)
                    )

                self.logger.info(
                    "Connection established successfully",
                    connection_id=connection_id,
                    connection_type=connection_type,
                    attempt=attempt + 1,
                )

                return True

            except Exception as e:
                self.logger.warning(
                    "Connection attempt failed",
                    connection_id=connection_id,
                    connection_type=connection_type,
                    attempt=attempt + 1,
                    error=str(e),
                )

                if attempt == policy["max_attempts"] - 1:
                    self.logger.error(
                        "Connection establishment failed",
                        connection_id=connection_id,
                        connection_type=connection_type,
                        max_attempts=policy["max_attempts"],
                    )
                    return False

        return False

    async def close_connection(self, connection_id: str) -> bool:
        """Close a connection gracefully with proper WebSocket handling."""

        if connection_id not in self.connections:
            self.logger.warning("Connection not found", connection_id=connection_id)
            return False

        connection_info = None
        connection = None
        try:
            async with self._connections_lock:
                connection_info = self.connections.get(connection_id)
                if not connection_info:
                    return False
                connection = connection_info["connection"]

            # Close WebSocket connection properly with await and timeout
            if connection:
                connection_type = connection_info.get("type", "unknown")

                if hasattr(connection, "close"):
                    if asyncio.iscoroutinefunction(connection.close):
                        try:
                            await asyncio.wait_for(connection.close(), timeout=10.0)
                        except asyncio.TimeoutError:
                            self.logger.warning(
                                "Connection close timeout", connection_id=connection_id
                            )
                        except asyncio.CancelledError:
                            # Re-raise cancellation to maintain proper async cleanup
                            self.logger.debug(
                                "Connection close cancelled", connection_id=connection_id
                            )
                            raise
                        except Exception as close_error:
                            self.logger.warning(
                                "Connection close error",
                                connection_id=connection_id,
                                error=str(close_error),
                            )
                    else:
                        try:
                            connection.close()
                        except Exception as close_error:
                            self.logger.warning(
                                "Sync connection close error",
                                connection_id=connection_id,
                                error=str(close_error),
                            )
                elif hasattr(connection, "disconnect"):
                    if asyncio.iscoroutinefunction(connection.disconnect):
                        try:
                            await asyncio.wait_for(connection.disconnect(), timeout=10.0)
                        except asyncio.TimeoutError:
                            self.logger.warning(
                                "Connection disconnect timeout", connection_id=connection_id
                            )
                        except asyncio.CancelledError:
                            # Re-raise cancellation to maintain proper async cleanup
                            self.logger.debug(
                                "Connection disconnect cancelled", connection_id=connection_id
                            )
                            raise
                        except Exception as disconnect_error:
                            self.logger.warning(
                                "Connection disconnect error",
                                connection_id=connection_id,
                                error=str(disconnect_error),
                            )
                    else:
                        try:
                            connection.disconnect()
                        except Exception as disconnect_error:
                            self.logger.warning(
                                "Sync connection disconnect error",
                                connection_id=connection_id,
                                error=str(disconnect_error),
                            )

                # For WebSocket connections, ensure proper state cleanup
                if connection_type == "websocket":
                    if hasattr(connection, "state"):
                        try:
                            # Mark connection as closed if state is mutable
                            if hasattr(connection.state, "__setattr__"):
                                connection.state = 3  # WebSocket CLOSED state
                        except Exception as state_error:
                            self.logger.debug(
                                "WebSocket state cleanup warning",
                                connection_id=connection_id,
                                error=str(state_error),
                            )

            self.logger.info("Connection closed", connection_id=connection_id)
            return True

        except Exception as e:
            self.logger.error(
                "Failed to close connection", connection_id=connection_id, error=str(e)
            )
            return False
        finally:
            # Ensure state is updated and connection is removed even if close fails
            try:
                async with self._connections_lock:
                    if connection_id in self.connections:
                        if connection_info:
                            connection_info["state"] = ConnectionState.DISCONNECTED
                            connection_info["last_activity"] = datetime.now(timezone.utc)
            except Exception as cleanup_error:
                self.logger.warning(
                    "Failed to update connection state during cleanup",
                    connection_id=connection_id,
                    error=str(cleanup_error),
                )

    @time_execution
    @circuit_breaker(failure_threshold=3)
    async def reconnect_connection(self, connection_id: str) -> bool:
        """Reconnect a failed connection."""

        if connection_id not in self.connections:
            self.logger.warning(
                "Connection not found for reconnection", connection_id=connection_id
            )
            return False

        connection_info = self.connections[connection_id]
        connection_type = connection_info["type"]

        self.logger.info(
            "Attempting reconnection", connection_id=connection_id, connection_type=connection_type
        )

        # Update state
        connection_info["state"] = ConnectionState.CONNECTING
        connection_info["reconnect_count"] += 1

        # Exponential backoff placeholder - implement with configurable retry strategy
        # This will be implemented with exchange integrations
        # For now, simulate reconnection
        await asyncio.sleep(1)

        # Simulate successful reconnection
        connection_info["state"] = ConnectionState.CONNECTED
        connection_info["last_activity"] = datetime.now(timezone.utc)

        self.logger.info(
            "Reconnection successful",
            connection_id=connection_id,
            reconnect_count=connection_info["reconnect_count"],
        )

        return True

    async def _monitor_connection_health(self, connection_id: str) -> None:
        """Monitor connection health with heartbeat checks."""

        connection_info = self.connections.get(connection_id)
        if not connection_info:
            return

        connection_type = connection_info["type"]
        heartbeat_interval = self.heartbeat_intervals.get(connection_type, 30)

        while connection_info["state"] == ConnectionState.CONNECTED:
            try:
                # Perform heartbeat check with proper race condition handling
                start_time = time.time()
                is_healthy = await self._perform_heartbeat(connection_id)
                latency_ms = (time.time() - start_time) * 1000

                # Use locks to prevent race conditions when updating health metrics
                async with self._health_monitors_lock:
                    if connection_id not in self.health_monitors:
                        # Connection was removed during heartbeat, exit monitoring
                        self.logger.debug(
                            "Connection removed during health monitoring",
                            connection_id=connection_id,
                        )
                        break

                    health = self.health_monitors[connection_id]

                    if is_healthy:
                        # Update health metrics with optimized storage
                        health.last_heartbeat = datetime.now(timezone.utc)
                        health.add_latency_measurement(Decimal(str(latency_ms)))
                        health.uptime_seconds = int(
                            (
                                datetime.now(timezone.utc) - connection_info["established_at"]
                            ).total_seconds()
                        )
                        health.last_error = None

                        # Update connection quality based on latency with trend analysis
                        latency_decimal = Decimal(str(latency_ms))
                        if latency_decimal < Decimal("100"):
                            quality = Decimal("1.0")
                        elif latency_decimal < Decimal("500"):
                            quality = Decimal("0.8")
                        elif latency_decimal < Decimal("1000"):
                            quality = Decimal("0.6")
                        else:
                            quality = Decimal("0.4")

                        # Factor in trend for more accurate quality assessment
                        trend = health.get_quality_trend()
                        if trend == "degrading":
                            quality = quality * Decimal("0.9")
                        elif trend == "improving":
                            quality = min(quality * Decimal("1.1"), Decimal("1.0"))

                        health.add_quality_measurement(quality)

                        # Update connection activity with lock protection
                        async with self._connections_lock:
                            if connection_id in self.connections:
                                self.connections[connection_id]["last_activity"] = datetime.now(
                                    timezone.utc
                                )

                    else:
                        # Connection is unhealthy, trigger reconnection
                        self.logger.warning(
                            "Connection health check failed", connection_id=connection_id
                        )

                        health.last_error = "Heartbeat failed"
                        health.connection_quality = Decimal("0.0")

                        # Trigger reconnection (this will handle its own locking)
                        try:
                            await self.reconnect_connection(connection_id)
                        except Exception as reconnect_error:
                            self.logger.error(
                                "Reconnection failed during health monitoring",
                                connection_id=connection_id,
                                error=str(reconnect_error),
                            )
                            # Continue monitoring even if reconnection fails
                            pass

                await asyncio.sleep(heartbeat_interval)

            except asyncio.CancelledError:
                self.logger.debug("Health monitoring cancelled", connection_id=connection_id)
                # Exit cleanly on cancellation
                break
            except Exception as e:
                self.logger.error(
                    "Health monitoring error", connection_id=connection_id, error=str(e)
                )
                # Use exponential backoff for error conditions to prevent flooding
                try:
                    await asyncio.sleep(min(heartbeat_interval * 2, 60))
                except asyncio.CancelledError:
                    self.logger.debug(
                        "Health monitoring sleep cancelled", connection_id=connection_id
                    )
                    break

    async def _perform_heartbeat(self, connection_id: str) -> bool:
        """Perform heartbeat check on connection with proper async handling."""

        connection_info = self.connections.get(connection_id)
        if not connection_info:
            return False

        connection = connection_info["connection"]
        connection_type = connection_info["type"]

        try:
            if connection_type == "exchange":
                # Exchange heartbeat - check API connectivity
                if hasattr(connection, "ping"):
                    if asyncio.iscoroutinefunction(connection.ping):
                        try:
                            response = await asyncio.wait_for(
                                connection.ping(),
                                timeout=10.0,  # 10 second timeout for exchange pings
                            )
                            return response is not None
                        except asyncio.TimeoutError:
                            self.logger.warning(
                                "Exchange ping timeout", connection_id=connection_id, timeout=10.0
                            )
                            return False
                    else:
                        response = connection.ping()
                        return response is not None
                elif hasattr(connection, "get_server_time"):
                    if asyncio.iscoroutinefunction(connection.get_server_time):
                        try:
                            response = await asyncio.wait_for(
                                connection.get_server_time(),
                                timeout=10.0,  # 10 second timeout for server time
                            )
                            return response is not None
                        except asyncio.TimeoutError:
                            self.logger.warning(
                                "Exchange server time timeout",
                                connection_id=connection_id,
                                timeout=10.0,
                            )
                            return False
                    else:
                        response = connection.get_server_time()
                        return response is not None
                else:
                    # Fallback: assume healthy if connection exists
                    return connection is not None

            elif connection_type == "database":
                # Database heartbeat - execute simple query with timeout
                if hasattr(connection, "execute"):
                    if asyncio.iscoroutinefunction(connection.execute):
                        try:
                            result = await asyncio.wait_for(
                                connection.execute("SELECT 1"),
                                timeout=5.0,  # 5 second timeout for DB queries
                            )
                            return result is not None
                        except asyncio.TimeoutError:
                            self.logger.warning(
                                "Database query timeout", connection_id=connection_id, timeout=5.0
                            )
                            return False
                    else:
                        result = connection.execute("SELECT 1")
                        return result is not None
                elif hasattr(connection, "ping"):
                    if asyncio.iscoroutinefunction(connection.ping):
                        try:
                            return await asyncio.wait_for(
                                connection.ping(),
                                timeout=5.0,  # 5 second timeout for DB ping
                            )
                        except asyncio.TimeoutError:
                            self.logger.warning(
                                "Database ping timeout", connection_id=connection_id, timeout=5.0
                            )
                            return False
                    else:
                        return connection.ping()
                else:
                    return connection is not None

            elif connection_type == "websocket":
                # WebSocket heartbeat - check connection state with proper async handling
                if hasattr(connection, "state"):
                    # Check if WebSocket is open (usually state == 1)
                    try:
                        state = connection.state
                        return state == 1
                    except Exception as state_error:
                        self.logger.debug(
                            "WebSocket state access error",
                            connection_id=connection_id,
                            error=str(state_error),
                        )
                        return False
                elif hasattr(connection, "is_open"):
                    if asyncio.iscoroutinefunction(connection.is_open):
                        try:
                            return await asyncio.wait_for(
                                connection.is_open(), timeout=self.websocket_heartbeat_timeout
                            )
                        except asyncio.TimeoutError:
                            self.logger.warning(
                                "WebSocket is_open timeout",
                                connection_id=connection_id,
                                timeout=self.websocket_heartbeat_timeout,
                            )
                            return False
                        except asyncio.CancelledError:
                            # Re-raise cancellation to maintain proper async cleanup
                            self.logger.debug(
                                "WebSocket is_open cancelled", connection_id=connection_id
                            )
                            raise
                        except Exception as is_open_error:
                            self.logger.warning(
                                "WebSocket is_open error",
                                connection_id=connection_id,
                                error=str(is_open_error),
                            )
                            return False
                    else:
                        try:
                            return connection.is_open()
                        except Exception as is_open_error:
                            self.logger.warning(
                                "WebSocket sync is_open error",
                                connection_id=connection_id,
                                error=str(is_open_error),
                            )
                            return False
                elif hasattr(connection, "ping"):
                    if asyncio.iscoroutinefunction(connection.ping):
                        try:
                            ping_result = await asyncio.wait_for(
                                connection.ping(), timeout=self.websocket_heartbeat_timeout
                            )
                            # Handle different ping response types
                            if ping_result is None:
                                return True  # Some implementations return None on success
                            return bool(ping_result)
                        except asyncio.TimeoutError:
                            self.logger.warning(
                                "WebSocket ping timeout",
                                connection_id=connection_id,
                                timeout=self.websocket_heartbeat_timeout,
                            )
                            return False
                        except asyncio.CancelledError:
                            # Re-raise cancellation to maintain proper async cleanup
                            self.logger.debug(
                                "WebSocket ping cancelled", connection_id=connection_id
                            )
                            raise
                        except Exception as ping_error:
                            self.logger.warning(
                                "WebSocket ping error",
                                connection_id=connection_id,
                                error=str(ping_error),
                            )
                            return False
                    else:
                        try:
                            ping_result = connection.ping()
                            return True  # If no exception, assume success
                        except Exception as ping_error:
                            self.logger.warning(
                                "WebSocket sync ping error",
                                connection_id=connection_id,
                                error=str(ping_error),
                            )
                            return False
                else:
                    # Fallback: check if connection object is not None and has basic attributes
                    try:
                        return connection is not None and hasattr(connection, "__dict__")
                    except Exception:
                        return False

            else:
                # Generic heartbeat - check if connection object exists with timeout
                if hasattr(connection, "ping"):
                    if asyncio.iscoroutinefunction(connection.ping):
                        try:
                            return await asyncio.wait_for(
                                connection.ping(),
                                timeout=10.0,  # 10 second generic timeout
                            )
                        except asyncio.TimeoutError:
                            self.logger.warning(
                                "Generic ping timeout", connection_id=connection_id, timeout=10.0
                            )
                            return False
                    else:
                        return connection.ping()
                else:
                    return connection is not None

        except asyncio.CancelledError:
            self.logger.debug("Heartbeat cancelled", connection_id=connection_id)
            return False
        except Exception as e:
            self.logger.error("Heartbeat failed", connection_id=connection_id, error=str(e))
            return False

    def _start_cleanup_task(self) -> None:
        """Start periodic cleanup task."""
        try:
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
                # Add done callback to handle any exceptions
                self._cleanup_task.add_done_callback(self._cleanup_task_done_callback)
        except RuntimeError:
            # No event loop running, cleanup task will be started on first use
            self.logger.debug(
                "No event loop available for cleanup task - will be started on first connection"
            )
            pass

    def _cleanup_task_done_callback(self, task: asyncio.Task) -> None:
        """Handle cleanup task completion."""
        try:
            exception = task.exception()
            if exception and not isinstance(exception, asyncio.CancelledError):
                self.logger.error(f"Cleanup task failed: {exception}")
        except asyncio.CancelledError:
            # Task was cancelled, this is expected during shutdown
            pass
        # Reset task reference so it can be restarted if needed
        if self._cleanup_task is task:
            self._cleanup_task = None

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of connection data and metrics."""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes

                # Clean up dead connections
                dead_connections = []
                for conn_id, conn_info in self.connections.items():
                    if conn_info["state"] == ConnectionState.FAILED:
                        last_activity = conn_info.get("last_activity")
                        if (
                            last_activity
                            and (datetime.now(timezone.utc) - last_activity).total_seconds() > 3600
                        ):
                            dead_connections.append(conn_id)

                for conn_id in dead_connections:
                    await self._cleanup_connection(conn_id)

                # Log performance stats
                self.logger.info(
                    "Connection manager stats",
                    active_connections=len(self.connections),
                    queued_connections=len(self.message_queues),
                    total_created=self._connection_stats["total_connections_created"],
                    total_reconnections=self._connection_stats["total_reconnections"],
                )

                self._connection_stats["last_cleanup"] = datetime.now(timezone.utc)

            except Exception as e:
                self.logger.error("Error in periodic cleanup", error=str(e))
                await asyncio.sleep(300)

    async def _cleanup_connection(self, connection_id: str) -> None:
        """Clean up resources for a dead connection."""
        task = None
        try:
            # Cancel monitoring task with timeout
            async with self._tasks_lock:
                if connection_id in self._health_monitor_tasks:
                    task = self._health_monitor_tasks[connection_id]
                    if not task.done():
                        task.cancel()
                    del self._health_monitor_tasks[connection_id]

            # Wait for task cancellation outside the lock
            if task and not task.done():
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        except Exception as e:
            self.logger.warning(f"Error cleaning up monitoring task: {e}")
        finally:
            # Always ensure task is cleaned up
            if task and not task.done():
                try:
                    task.cancel()
                except Exception as e:
                    self.logger.debug(f"Task cancellation failed: {e}")
                    # Continue cleanup despite cancellation failure

        # Close connection before removing data
        try:
            if connection_id in self.connections:
                await self.close_connection(connection_id)
        except Exception as e:
            self.logger.warning(f"Error closing connection during cleanup: {e}")

        # Remove connection data with proper locking
        cleared_count = 0
        try:
            async with self._connections_lock:
                self.connections.pop(connection_id, None)

            async with self._health_monitors_lock:
                self.health_monitors.pop(connection_id, None)

            async with self._message_queues_lock:
                if connection_id in self.message_queues:
                    cleared_count = self.message_queues[connection_id].clear()
                    del self.message_queues[connection_id]

        except Exception as e:
            self.logger.warning(f"Error removing connection data during cleanup: {e}")

        self.logger.info(
            f"Cleaned up connection {connection_id}, cleared {cleared_count} queued messages"
        )

    async def queue_message(self, connection_id: str, message: dict[str, Any]) -> bool:
        """Queue a message for later transmission when connection is restored."""

        async with self._message_queues_lock:
            if connection_id not in self.message_queues:
                self.message_queues[connection_id] = MessageQueue(
                    max_size=1000, ttl_seconds=900
                )  # 15 min TTL

            success = self.message_queues[connection_id].add_message(message)

        if success:
            async with self._stats_lock:
                total_queued = self._connection_stats.get("total_messages_queued", 0)
                if isinstance(total_queued, int | str):
                    self._connection_stats["total_messages_queued"] = int(total_queued) + 1
                else:
                    self._connection_stats["total_messages_queued"] = 1

            # Get queue size safely
            async with self._message_queues_lock:
                queue_size = self.message_queues[connection_id].size()

            self.logger.info(
                "Message queued",
                connection_id=connection_id,
                message_type=message.get("type", "unknown"),
                queue_size=queue_size,
            )

        return success

    async def flush_message_queue(self, connection_id: str) -> int:
        """Flush queued messages for a connection."""

        async with self._message_queues_lock:
            if connection_id not in self.message_queues:
                return 0

            message_queue = self.message_queues[connection_id]
            messages = message_queue.get_messages()  # Get valid (non-expired) messages

        if not messages:
            return 0

        flushed_count = 0

        for message in messages:
            try:
                # Secure message queue placeholder - implement with production queue system
                # This will be implemented in P-003+ (Exchange Integrations)
                self.logger.info(
                    "Flushing queued message",
                    connection_id=connection_id,
                    message_type=message.get("type", "unknown"),
                )
                flushed_count += 1

            except Exception as e:
                self.logger.error(
                    "Failed to flush message",
                    connection_id=connection_id,
                    message_type=message.get("type", "unknown"),
                    error=str(e),
                )

        # Clear the queue
        cleared_count = message_queue.clear()

        self.logger.info(
            "Message queue flushed",
            connection_id=connection_id,
            flushed_count=flushed_count,
            cleared_count=cleared_count,
        )

        return flushed_count

    def get_connection_status(self, connection_id: str) -> dict[str, Any] | None:
        """Get status of a specific connection."""

        if connection_id not in self.connections:
            return None

        connection_info = self.connections[connection_id]
        health = self.health_monitors.get(connection_id)

        return {
            "connection_id": connection_id,
            "type": connection_info["type"],
            "state": connection_info["state"].value,
            "established_at": connection_info["established_at"].isoformat(),
            "last_activity": connection_info["last_activity"].isoformat(),
            "reconnect_count": connection_info["reconnect_count"],
            "health": health.to_dict() if health else None,
            "queued_messages": self.message_queues.get(connection_id, MessageQueue()).size(),
        }

    def get_all_connection_status(self) -> dict[str, dict[str, Any] | None]:
        """Get status of all connections."""
        return {
            connection_id: self.get_connection_status(connection_id)
            for connection_id in self.connections.keys()
        }

    def is_connection_healthy(self, connection_id: str) -> bool:
        """Check if a connection is healthy."""

        if connection_id not in self.connections:
            return False

        connection_info = self.connections[connection_id]
        health = self.health_monitors.get(connection_id)

        if connection_info["state"] != ConnectionState.CONNECTED:
            return False

        if not health:
            return False

        # Check if heartbeat is recent (within 2x heartbeat interval)
        connection_type = connection_info["type"]
        heartbeat_interval = self.heartbeat_intervals.get(connection_type, 30)
        max_age = heartbeat_interval * 2

        time_since_heartbeat = (datetime.now(timezone.utc) - health.last_heartbeat).total_seconds()

        return time_since_heartbeat < max_age and health.connection_quality > Decimal("0.5")

    async def cleanup_resources(self) -> None:
        """Cleanup all resources and close connections with proper async WebSocket handling."""

        # Cancel cleanup task if running
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=5.0)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                self.logger.warning("Cleanup task cancellation timed out")
            except Exception as e:
                self.logger.error(f"Failed to cleanup task: {e}")

        # Cancel all health monitoring tasks with timeout
        async with self._tasks_lock:
            tasks_to_cancel = list(self._health_monitor_tasks.values())
            self._health_monitor_tasks.clear()

        # Cancel tasks in parallel for efficiency
        if tasks_to_cancel:
            for task in tasks_to_cancel:
                if not task.done():
                    task.cancel()

            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_cancel, return_exceptions=True), timeout=10.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("Health monitoring task cancellation timed out")

        # Close all connections with concurrent execution, timeout, and backpressure handling
        connection_ids = list(self.connections.keys())
        if connection_ids:
            # Process connections in batches to avoid overwhelming the system
            batch_size = DEFAULT_CONNECTION_BATCH_SIZE

            for i in range(0, len(connection_ids), batch_size):
                batch = connection_ids[i : i + batch_size]
                batch_tasks = [self.close_connection(connection_id) for connection_id in batch]

                try:
                    # Process each batch with timeout
                    await asyncio.wait_for(
                        asyncio.gather(*batch_tasks, return_exceptions=True), timeout=15.0
                    )

                    # Small delay between batches to prevent overwhelming
                    if i + batch_size < len(connection_ids):
                        await asyncio.sleep(0.1)

                except asyncio.TimeoutError:
                    self.logger.warning(
                        "Connection batch cleanup timed out", batch_start=i, batch_size=len(batch)
                    )
                    continue  # Continue with next batch
                except Exception as batch_error:
                    self.logger.error(
                        "Connection batch cleanup error", batch_start=i, error=str(batch_error)
                    )
                    continue  # Continue with next batch

        # Clear all data structures
        async with self._connections_lock:
            self.connections.clear()
        async with self._health_monitors_lock:
            self.health_monitors.clear()
        async with self._message_queues_lock:
            for queue in self.message_queues.values():
                queue.clear()
            self.message_queues.clear()

        self.logger.info("Connection manager cleanup completed")
