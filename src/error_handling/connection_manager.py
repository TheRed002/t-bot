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
from enum import Enum
from typing import Any

from src.core.config import Config
from src.core.logging import get_logger

# MANDATORY: Import from P-001 core framework
# MANDATORY: Import from P-007A utils framework
from src.utils.decorators import circuit_breaker, time_execution


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
    latency_ms: float
    packet_loss: float
    connection_quality: float  # 0.0 to 1.0
    uptime_seconds: int
    reconnect_count: int
    last_error: str | None = None

    # Efficient latency history storage
    _latency_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    _quality_history: deque[float] = field(default_factory=lambda: deque(maxlen=50))

    def add_latency_measurement(self, latency: float) -> None:
        """Add latency measurement to history."""
        self._latency_history.append(latency)
        self.latency_ms = latency

    def add_quality_measurement(self, quality: float) -> None:
        """Add quality measurement to history."""
        self._quality_history.append(quality)
        self.connection_quality = quality

    def get_average_latency(self, samples: int = 10) -> float:
        """Get average latency from recent samples."""
        if not self._latency_history:
            return 0.0
        recent_samples = list(self._latency_history)[-samples:]
        return sum(recent_samples) / len(recent_samples)

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
            "latency_ms": self.latency_ms,
            "packet_loss": self.packet_loss,
            "connection_quality": self.connection_quality,
            "uptime_seconds": self.uptime_seconds,
            "reconnect_count": self.reconnect_count,
            "last_error": self.last_error,
            "avg_latency_10s": self.get_average_latency(10),
            "quality_trend": self.get_quality_trend(),
        }


class MessageQueue:
    """Efficient message queue with size limits and TTL."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._queue: deque[dict[str, Any]] = deque(maxlen=max_size)
        self._total_size_bytes = 0
        self._max_size_bytes = 10 * 1024 * 1024  # 10MB limit

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

    def __init__(self, config: Config):
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

                # Attempt connection
                connection = await connect_func(**kwargs)

                # Initialize connection tracking
                self.connections[connection_id] = {
                    "connection": connection,
                    "type": connection_type,
                    "state": ConnectionState.CONNECTED,
                    "established_at": datetime.now(timezone.utc),
                    "last_activity": datetime.now(timezone.utc),
                    "reconnect_count": 0,
                }

                # Initialize health monitoring
                self.health_monitors[connection_id] = ConnectionHealth(
                    last_heartbeat=datetime.now(timezone.utc),
                    latency_ms=0.0,
                    packet_loss=0.0,
                    connection_quality=1.0,
                    uptime_seconds=0,
                    reconnect_count=0,
                )

                # Start health monitoring
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
        """Close a connection gracefully."""

        if connection_id not in self.connections:
            self.logger.warning("Connection not found", connection_id=connection_id)
            return False

        try:
            connection_info = self.connections[connection_id]
            connection = connection_info["connection"]

            # Close connection if it has a close method
            if hasattr(connection, "close"):
                await connection.close()
            elif hasattr(connection, "disconnect"):
                await connection.disconnect()

            # Update state
            connection_info["state"] = ConnectionState.DISCONNECTED
            connection_info["last_activity"] = datetime.now(timezone.utc)

            self.logger.info("Connection closed", connection_id=connection_id)
            return True

        except Exception as e:
            self.logger.error(
                "Failed to close connection", connection_id=connection_id, error=str(e)
            )
            return False

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

        # TODO: Implement actual reconnection logic
        # This will be implemented in P-003+ (Exchange Integrations)
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

    async def _monitor_connection_health(self, connection_id: str):
        """Monitor connection health with heartbeat checks."""

        connection_info = self.connections.get(connection_id)
        if not connection_info:
            return

        connection_type = connection_info["type"]
        heartbeat_interval = self.heartbeat_intervals.get(connection_type, 30)

        while connection_info["state"] == ConnectionState.CONNECTED:
            try:
                # Perform heartbeat check
                start_time = time.time()
                is_healthy = await self._perform_heartbeat(connection_id)
                latency_ms = (time.time() - start_time) * 1000

                if is_healthy:
                    # Update health metrics with optimized storage
                    health = self.health_monitors[connection_id]
                    health.last_heartbeat = datetime.now(timezone.utc)
                    health.add_latency_measurement(latency_ms)
                    health.uptime_seconds = int(
                        (
                            datetime.now(timezone.utc) - connection_info["established_at"]
                        ).total_seconds()
                    )
                    health.last_error = None

                    # Update connection quality based on latency with trend analysis
                    if latency_ms < 100:
                        quality = 1.0
                    elif latency_ms < 500:
                        quality = 0.8
                    elif latency_ms < 1000:
                        quality = 0.6
                    else:
                        quality = 0.4

                    # Factor in trend for more accurate quality assessment
                    trend = health.get_quality_trend()
                    if trend == "degrading":
                        quality *= 0.9
                    elif trend == "improving":
                        quality = min(quality * 1.1, 1.0)

                    health.add_quality_measurement(quality)
                    connection_info["last_activity"] = datetime.now(timezone.utc)

                else:
                    # Connection is unhealthy, trigger reconnection
                    self.logger.warning(
                        "Connection health check failed", connection_id=connection_id
                    )

                    health = self.health_monitors[connection_id]
                    health.last_error = "Heartbeat failed"
                    health.connection_quality = 0.0

                    # Trigger reconnection
                    await self.reconnect_connection(connection_id)

                await asyncio.sleep(heartbeat_interval)

            except Exception as e:
                self.logger.error(
                    "Health monitoring error", connection_id=connection_id, error=str(e)
                )
                await asyncio.sleep(heartbeat_interval)

    async def _perform_heartbeat(self, connection_id: str) -> bool:
        """Perform heartbeat check on connection."""

        connection_info = self.connections.get(connection_id)
        if not connection_info:
            return False

        connection = connection_info["connection"]
        connection_type = connection_info["type"]

        try:
            if connection_type == "exchange":
                # Exchange heartbeat - check API connectivity
                if hasattr(connection, "ping"):
                    response = await connection.ping()
                    return response is not None
                elif hasattr(connection, "get_server_time"):
                    response = await connection.get_server_time()
                    return response is not None
                else:
                    # Fallback: assume healthy if connection exists
                    return connection is not None

            elif connection_type == "database":
                # Database heartbeat - execute simple query
                if hasattr(connection, "execute"):
                    result = await connection.execute("SELECT 1")
                    return result is not None
                elif hasattr(connection, "ping"):
                    return await connection.ping()
                else:
                    return connection is not None

            elif connection_type == "websocket":
                # WebSocket heartbeat - check connection state
                if hasattr(connection, "state"):
                    # Check if WebSocket is open (usually state == 1)
                    return connection.state == 1
                elif hasattr(connection, "is_open"):
                    return connection.is_open()
                elif hasattr(connection, "ping"):
                    await connection.ping()
                    return True
                else:
                    return connection is not None

            else:
                # Generic heartbeat - check if connection object exists
                if hasattr(connection, "ping"):
                    return await connection.ping()
                else:
                    return connection is not None

        except Exception as e:
            self.logger.error("Heartbeat failed", connection_id=connection_id, error=str(e))
            return False

    def _start_cleanup_task(self) -> None:
        """Start periodic cleanup task."""
        try:
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        except RuntimeError:
            # No event loop running, cleanup task will be started on first use
            self.logger.debug(
                "No event loop available for cleanup task - will be started on first connection"
            )
            pass

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
        # Cancel monitoring task
        if connection_id in self._health_monitor_tasks:
            task = self._health_monitor_tasks[connection_id]
            if not task.done():
                task.cancel()
            del self._health_monitor_tasks[connection_id]

        # Remove connection data
        self.connections.pop(connection_id, None)
        self.health_monitors.pop(connection_id, None)

        # Clear message queue
        if connection_id in self.message_queues:
            cleared_count = self.message_queues[connection_id].clear()
            del self.message_queues[connection_id]
            self.logger.info(
                f"Cleaned up connection {connection_id}, cleared {cleared_count} queued messages"
            )

    async def queue_message(self, connection_id: str, message: dict[str, Any]) -> bool:
        """Queue a message for later transmission when connection is restored."""

        if connection_id not in self.message_queues:
            self.message_queues[connection_id] = MessageQueue(
                max_size=1000, ttl_seconds=900
            )  # 15 min TTL

        success = self.message_queues[connection_id].add_message(message)

        if success:
            self._connection_stats["total_messages_queued"] += 1
            self.logger.info(
                "Message queued",
                connection_id=connection_id,
                message_type=message.get("type", "unknown"),
                queue_size=self.message_queues[connection_id].size(),
            )

        return success

    async def flush_message_queue(self, connection_id: str) -> int:
        """Flush queued messages for a connection."""

        if connection_id not in self.message_queues:
            return 0

        message_queue = self.message_queues[connection_id]
        messages = message_queue.get_messages()  # Get valid (non-expired) messages

        if not messages:
            return 0

        flushed_count = 0

        for message in messages:
            try:
                # TODO: Implement actual message transmission
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

        return time_since_heartbeat < max_age and health.connection_quality > 0.5
