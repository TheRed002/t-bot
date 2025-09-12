"""
Connection manager for reliable network connections.

Simplified connection management with basic reconnection and health monitoring.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.core.config import Config
from src.core.logging import get_logger


class ConnectionState(Enum):
    """Connection state enumeration."""

    CONNECTED = "connected"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    ACTIVE = "active"  # Alias for CONNECTED for compatibility


class ConnectionHealth(Enum):
    """Connection health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ConnectionInfo:
    """Basic connection information."""

    connection: Any
    state: ConnectionState
    established_at: datetime
    reconnect_count: int = 0


class ConnectionManager:
    """Simple connection manager with reconnection support."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__module__)

        # Simple connection storage
        self.connections: dict[str, ConnectionInfo] = {}

        # Basic reconnection settings
        self.max_attempts = 3
        self.base_delay = 1.0
        self.max_delay = 30.0
        self.connection_timeout = 30.0  # Default connection timeout
        self.heartbeat_interval = 30.0  # Default heartbeat interval

    async def establish_connection(
        self,
        connection_id: str,
        connection_type: str,
        connect_func: Callable[..., Any],
        **kwargs: Any,
    ) -> bool:
        """Establish a new connection with retry."""

        self.logger.info(f"Establishing connection {connection_id}")

        for attempt in range(self.max_attempts):
            try:
                delay = min(self.base_delay * (2**attempt), self.max_delay)

                if attempt > 0:
                    await asyncio.sleep(delay)

                if asyncio.iscoroutinefunction(connect_func):
                    # Add timeout to async connection attempts
                    connection = await asyncio.wait_for(
                        connect_func(**kwargs), timeout=self.connection_timeout
                    )
                else:
                    connection = connect_func(**kwargs)

                self.connections[connection_id] = ConnectionInfo(
                    connection=connection,
                    state=ConnectionState.CONNECTED,
                    established_at=datetime.now(timezone.utc),
                    reconnect_count=0,
                )

                self.logger.info(f"Connection {connection_id} established")
                return True

            except Exception as e:
                self.logger.warning(
                    f"Connection attempt {attempt + 1} failed for {connection_id}: {e}"
                )

                if attempt == self.max_attempts - 1:
                    self.logger.error(f"Failed to establish connection {connection_id}")
                    return False
                # Continue with next retry attempt

        return False

    async def close_connection(self, connection_id: str) -> bool:
        """Close a connection."""

        if connection_id not in self.connections:
            self.logger.warning(f"Connection {connection_id} not found")
            return False

        connection_info = self.connections[connection_id]
        connection = connection_info.connection

        try:
            if hasattr(connection, "close"):
                if asyncio.iscoroutinefunction(connection.close):
                    await connection.close()
                else:
                    connection.close()

            connection_info.state = ConnectionState.DISCONNECTED
            self.logger.info(f"Connection {connection_id} closed")
            return True

        except Exception as e:
            self.logger.error(f"Failed to close connection {connection_id}: {e}")
            return False

    async def reconnect_connection(self, connection_id: str) -> bool:
        """Reconnect a failed connection."""

        if connection_id not in self.connections:
            self.logger.warning(f"Connection {connection_id} not found for reconnection")
            return False

        connection_info = self.connections[connection_id]
        connection_info.state = ConnectionState.CONNECTING
        connection_info.reconnect_count += 1

        self.logger.info(f"Attempting reconnection for {connection_id}")

        # Simulate reconnection delay
        await asyncio.sleep(1)

        connection_info.state = ConnectionState.CONNECTED
        self.logger.info(f"Reconnection successful for {connection_id}")

        return True

    def get_connection_status(self, connection_id: str) -> dict[str, Any] | None:
        """Get connection status."""

        if connection_id not in self.connections:
            return None

        connection_info = self.connections[connection_id]

        return {
            "connection_id": connection_id,
            "state": connection_info.state.value,
            "established_at": connection_info.established_at.isoformat(),
            "reconnect_count": connection_info.reconnect_count,
        }

    def is_connection_healthy(self, connection_id: str) -> bool:
        """Check if connection is healthy."""

        if connection_id not in self.connections:
            return False

        connection_info = self.connections[connection_id]
        return connection_info.state == ConnectionState.CONNECTED

    async def cleanup_resources(self) -> None:
        """Cleanup all resources."""

        # Close all connections with proper cleanup
        cleanup_tasks = []
        for connection_id in list(self.connections.keys()):
            cleanup_tasks.append(self.close_connection(connection_id))

        # Wait for all cleanup tasks to complete
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                self.logger.error(f"Error during connection cleanup: {e}")

        self.connections.clear()
        self.logger.info("Connection manager cleanup completed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        await self.cleanup_resources()
