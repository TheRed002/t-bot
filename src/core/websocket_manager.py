"""
WebSocket Connection Manager for T-Bot Trading System.

Provides async context manager for WebSocket connections with proper cleanup,
reconnection logic, heartbeat mechanisms, and resource management.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.core.exceptions import WebSocketError
from src.core.resource_manager import ResourceManager, ResourceType


class WebSocketState(Enum):
    """WebSocket connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    ERROR = "error"


class WebSocketManager:
    """
    Async WebSocket connection manager with proper resource cleanup.

    Features:
    - Async context manager for proper connection lifecycle
    - Automatic reconnection with exponential backoff
    - Heartbeat mechanism to detect connection issues
    - Resource registration with ResourceManager
    - Memory leak prevention
    - Proper error handling and logging
    """

    def __init__(
        self,
        url: str,
        resource_manager: ResourceManager | None = None,
        heartbeat_interval: int = 30,
        reconnect_attempts: int = 5,
        reconnect_delay: float = 1.0,
        connection_timeout: float = 30.0,
    ):
        """
        Initialize WebSocket manager.

        Args:
            url: WebSocket URL to connect to
            resource_manager: Optional resource manager for tracking
            heartbeat_interval: Heartbeat interval in seconds
            reconnect_attempts: Maximum reconnection attempts
            reconnect_delay: Initial delay between reconnect attempts (exponential backoff)
            connection_timeout: Connection timeout in seconds
        """
        self.url = url
        self.resource_manager = resource_manager
        self.heartbeat_interval = heartbeat_interval
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.connection_timeout = connection_timeout

        self.websocket = None
        self.state = WebSocketState.DISCONNECTED
        self.reconnect_count = 0
        self.last_heartbeat = None

        # Tasks
        self.heartbeat_task: asyncio.Task | None = None
        self.message_handler_task: asyncio.Task | None = None

        # Callbacks
        self.message_callback: Callable[[dict], None] | None = None
        self.error_callback: Callable[[Exception], None] | None = None
        self.disconnect_callback: Callable[[], None] | None = None

        # Resource tracking
        self.resource_id: str | None = None

        self.logger = logging.getLogger(__name__)

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator["WebSocketManager", None]:
        """
        Async context manager for WebSocket connection with proper cleanup.

        Usage:
            async with ws_manager.connection() as ws:
                await ws.send_message({"type": "subscribe", "channel": "ticker"})
                # Connection is automatically closed on exit
        """
        try:
            await self._connect()
            yield self
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            raise WebSocketError(f"Failed to establish WebSocket connection: {e}") from e
        finally:
            await self._disconnect()

    async def _connect(self):
        """Establish WebSocket connection with timeout and error handling."""
        import websockets

        self.state = WebSocketState.CONNECTING

        try:
            # Connect with timeout
            async with asyncio.timeout(self.connection_timeout):
                self.websocket = await websockets.connect(self.url)

            self.state = WebSocketState.CONNECTED
            self.reconnect_count = 0
            self.last_heartbeat = datetime.now(timezone.utc)

            # Register resource if resource manager is available
            if self.resource_manager:
                self.resource_id = self.resource_manager.register_resource(
                    resource=self.websocket,
                    resource_type=ResourceType.WEBSOCKET_CONNECTION,
                    async_cleanup_callback=self._cleanup_connection,
                    metadata={"url": self.url, "state": self.state.value},
                )

            # Start background tasks
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.message_handler_task = asyncio.create_task(self._message_handler_loop())

            self.logger.info(f"WebSocket connected to {self.url}")

        except asyncio.TimeoutError:
            self.state = WebSocketState.ERROR
            raise WebSocketError(
                f"WebSocket connection timeout after {self.connection_timeout}s",
                websocket_state=self.state.value,
            )
        except Exception as e:
            self.state = WebSocketState.ERROR
            raise WebSocketError(
                f"WebSocket connection failed: {e}", websocket_state=self.state.value
            ) from e

    async def _disconnect(self):
        """Properly disconnect and cleanup WebSocket resources."""
        if self.state == WebSocketState.DISCONNECTED:
            return

        self.state = WebSocketState.CLOSING

        try:
            # Use TaskGroup for proper resource cleanup
            tasks = []
            if self.heartbeat_task and not self.heartbeat_task.done():
                tasks.append(self.heartbeat_task)
            if self.message_handler_task and not self.message_handler_task.done():
                tasks.append(self.message_handler_task)

            if tasks:
                try:
                    # Cancel all tasks
                    for task in tasks:
                        task.cancel()

                    # Wait for cancellation with timeout to prevent hanging
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("Task cancellation timed out")
                except Exception as e:
                    self.logger.error(f"Error cancelling tasks: {e}")

            # Close WebSocket connection
            await self._cleanup_connection()

            # Unregister resource
            if self.resource_manager and self.resource_id:
                await self.resource_manager.unregister_resource(self.resource_id)
                self.resource_id = None

            self.state = WebSocketState.DISCONNECTED
            self.logger.info(f"WebSocket disconnected from {self.url}")

            # Call disconnect callback
            if self.disconnect_callback:
                try:
                    self.disconnect_callback()
                except Exception as e:
                    self.logger.error(f"Error in disconnect callback: {e}")

        except Exception as e:
            self.logger.error(f"Error during WebSocket disconnect: {e}")
        finally:
            self.state = WebSocketState.DISCONNECTED

    async def _cleanup_connection(self):
        """Cleanup WebSocket connection with proper error handling."""
        if self.websocket:
            try:
                # Close connection gracefully
                if not self.websocket.closed:
                    await asyncio.wait_for(self.websocket.close(), timeout=5.0)

                # Wait for connection to be fully closed
                if hasattr(self.websocket, "wait_closed"):
                    await asyncio.wait_for(self.websocket.wait_closed(), timeout=5.0)

            except asyncio.TimeoutError:
                self.logger.warning("WebSocket close operation timed out")
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None

    async def _heartbeat_loop(self):
        """Background heartbeat loop to detect connection issues."""
        try:
            while self.state == WebSocketState.CONNECTED:
                try:
                    await asyncio.sleep(self.heartbeat_interval)

                    if self.websocket and not self.websocket.closed:
                        # Send ping
                        await asyncio.wait_for(self.websocket.ping(), timeout=10.0)
                        self.last_heartbeat = datetime.now(timezone.utc)

                        # Touch resource to indicate activity
                        if self.resource_manager and self.resource_id:
                            self.resource_manager.touch_resource(self.resource_id)
                    else:
                        break

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Heartbeat failed: {e}")
                    if self.error_callback:
                        self.error_callback(e)
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Heartbeat loop error: {e}")

    async def _message_handler_loop(self):
        """Background message handler loop with consistent data processing."""
        try:
            async for message in self.websocket:
                try:
                    # Parse JSON message with consistent format handling
                    if isinstance(message, str):
                        raw_data = json.loads(message)
                    else:
                        raw_data = json.loads(message.decode())

                    # Standardize incoming message format for consistent processing
                    if isinstance(raw_data, dict):
                        # Check if already standardized
                        if "data_format" not in raw_data:
                            # Transform to standard format
                            standardized_data = {
                                "data_format": "websocket_message_v1",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "source": "external",
                                "payload": raw_data,
                                "message_type": raw_data.get("type", "unknown"),
                                "processing_stage": "message_received",
                            }
                        else:
                            standardized_data = raw_data
                    else:
                        # Handle non-dict messages
                        standardized_data = {
                            "data_format": "websocket_message_v1",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "source": "external",
                            "payload": {"raw_message": raw_data},
                            "message_type": "raw_data",
                            "processing_stage": "message_received",
                        }

                    # Call message callback with standardized data
                    if self.message_callback:
                        try:
                            self.message_callback(standardized_data)
                        except Exception as e:
                            self.logger.error(f"Error in message callback: {e}")

                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON message received: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Message handler loop error: {e}")

    async def send_message(self, message: dict):
        """
        Send message through WebSocket with consistent data flow patterns.

        Args:
            message: Dictionary message to send as JSON (standardized format)

        Raises:
            WebSocketError: If sending fails
        """
        # Thread-safe state check with atomic access
        current_state = self.state
        current_websocket = self.websocket

        if (
            not current_websocket
            or current_websocket.closed
            or current_state != WebSocketState.CONNECTED
        ):
            raise WebSocketError("WebSocket is not connected", websocket_state=current_state.value)

        try:
            # Standardize message format for consistent processing across modules
            standardized_message = {
                "data_format": "websocket_message_v1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "websocket_manager",
                "payload": message,
                "message_type": message.get("type", "unknown"),
                "correlation_id": message.get("correlation_id"),
            }

            json_message = json.dumps(standardized_message)
            await asyncio.wait_for(self.websocket.send(json_message), timeout=10.0)

            # Touch resource to indicate activity
            if self.resource_manager and self.resource_id:
                self.resource_manager.touch_resource(self.resource_id)

        except asyncio.TimeoutError:
            raise WebSocketError("Message send timeout", websocket_state=self.state.value)
        except Exception as e:
            raise WebSocketError(
                f"Failed to send message: {e}", websocket_state=self.state.value
            ) from e

    def set_message_callback(self, callback: Callable[[dict], None]):
        """Set callback for incoming messages."""
        self.message_callback = callback

    def set_error_callback(self, callback: Callable[[Exception], None]):
        """Set callback for connection errors."""
        self.error_callback = callback

    def set_disconnect_callback(self, callback: Callable[[], None]):
        """Set callback for disconnection events."""
        self.disconnect_callback = callback

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is currently connected."""
        return bool(
            self.websocket and not self.websocket.closed and self.state == WebSocketState.CONNECTED
        )

    def get_stats(self) -> dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            "url": self.url,
            "state": self.state.value,
            "is_connected": self.is_connected,
            "reconnect_count": self.reconnect_count,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "resource_id": self.resource_id,
        }


# Factory function for creating WebSocket managers with resource management
def create_websocket_manager(
    url: str, resource_manager: ResourceManager | None = None, **kwargs
) -> WebSocketManager:
    """
    Factory function to create a WebSocket manager.

    Args:
        url: WebSocket URL
        resource_manager: Optional resource manager
        **kwargs: Additional configuration options

    Returns:
        Configured WebSocket manager
    """
    return WebSocketManager(url=url, resource_manager=resource_manager, **kwargs)
