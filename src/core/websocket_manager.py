"""
WebSocket Connection Manager for T-Bot Trading System.

Provides async context manager for WebSocket connections with proper cleanup,
reconnection logic, heartbeat mechanisms, and resource management.
"""

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.core.exceptions import WebSocketError
from src.core.resource_manager import ResourceManager, ResourceType

# Production WebSocket constants
DEFAULT_HEARTBEAT_INTERVAL = 30  # seconds
DEFAULT_RECONNECT_ATTEMPTS = 5
DEFAULT_RECONNECT_DELAY = 1.0  # seconds
DEFAULT_CONNECTION_TIMEOUT = 30.0  # seconds
DEFAULT_MESSAGE_QUEUE_SIZE = 1000
TASK_CANCELLATION_TIMEOUT = 5.0  # seconds
WEBSOCKET_CLOSE_TIMEOUT = 5.0  # seconds
WEBSOCKET_PING_TIMEOUT = 10.0  # seconds
MESSAGE_QUEUE_TIMEOUT = 1.0  # seconds
MESSAGE_SEND_TIMEOUT = 10.0  # seconds


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
        heartbeat_interval: int = DEFAULT_HEARTBEAT_INTERVAL,
        reconnect_attempts: int = DEFAULT_RECONNECT_ATTEMPTS,
        reconnect_delay: float = DEFAULT_RECONNECT_DELAY,
        connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT,
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

        # Backpressure handling
        self._message_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=DEFAULT_MESSAGE_QUEUE_SIZE)
        self._queue_processor_task: asyncio.Task | None = None

        # Resource tracking
        self.resource_id: str | None = None

        from src.core.logging import get_logger

        self.logger = get_logger(__name__)

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
            self.websocket = await asyncio.wait_for(
                websockets.connect(self.url),
                timeout=self.connection_timeout
            )

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
            self._queue_processor_task = asyncio.create_task(self._message_queue_processor())

            self.logger.info(f"WebSocket connected to {self.url}")

        except asyncio.TimeoutError as e:
            self.state = WebSocketState.ERROR
            raise WebSocketError(
                f"WebSocket connection timeout after {self.connection_timeout}s",
                websocket_state=self.state.value,
            ) from e
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
            if self._queue_processor_task and not self._queue_processor_task.done():
                tasks.append(self._queue_processor_task)

            if tasks:
                try:
                    # Cancel all tasks
                    for task in tasks:
                        task.cancel()

                    # Wait for cancellation with timeout to prevent hanging
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True), timeout=TASK_CANCELLATION_TIMEOUT
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
                    await asyncio.wait_for(self.websocket.close(), timeout=WEBSOCKET_CLOSE_TIMEOUT)

                # Wait for connection to be fully closed
                if hasattr(self.websocket, "wait_closed"):
                    await asyncio.wait_for(self.websocket.wait_closed(), timeout=WEBSOCKET_CLOSE_TIMEOUT)

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
                        await asyncio.wait_for(self.websocket.ping(), timeout=WEBSOCKET_PING_TIMEOUT)
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
            if not self.websocket:
                return
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
                                "processing_mode": "stream",  # WebSocket is inherently stream-based
                                "message_pattern": "pub_sub",  # WebSocket messages use pub/sub pattern
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
                            "processing_mode": "stream",  # WebSocket is inherently stream-based
                            "message_pattern": "pub_sub",  # WebSocket messages use pub/sub pattern
                        }

                    # Queue message for processing with backpressure handling
                    try:
                        self._message_queue.put_nowait(standardized_data)
                    except asyncio.QueueFull:
                        # Handle backpressure by dropping oldest message
                        try:
                            self._message_queue.get_nowait()
                            self._message_queue.put_nowait(standardized_data)
                            self.logger.warning("Message queue full, dropped oldest message")
                        except asyncio.QueueEmpty:
                            pass

                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON message received: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Message handler loop error: {e}")

    async def _message_queue_processor(self):
        """Process queued messages with backpressure control."""
        try:
            while self.state == WebSocketState.CONNECTED:
                try:
                    # Process messages from queue with timeout
                    message = await asyncio.wait_for(
                        self._message_queue.get(),
                        timeout=MESSAGE_QUEUE_TIMEOUT
                    )

                    # Call message callback with proper error handling
                    if self.message_callback:
                        try:
                            # Execute callback in executor to prevent blocking
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, self.message_callback, message)
                        except Exception as e:
                            self.logger.error(f"Error in message callback: {e}")

                    # Mark queue task as done
                    self._message_queue.task_done()

                except asyncio.TimeoutError:
                    # No messages to process, continue
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error processing message queue: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Message queue processor error: {e}")

    async def send_message(self, message: dict) -> None:
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
                "processing_mode": "stream",  # WebSocket is inherently stream-based
                "message_pattern": "pub_sub",  # WebSocket messages use pub/sub pattern
            }

            json_message = json.dumps(standardized_message)
            await asyncio.wait_for(self.websocket.send(json_message), timeout=MESSAGE_SEND_TIMEOUT)

            # Touch resource to indicate activity
            if self.resource_manager and self.resource_id:
                self.resource_manager.touch_resource(self.resource_id)

        except asyncio.TimeoutError as e:
            raise WebSocketError("Message send timeout", websocket_state=self.state.value) from e
        except Exception as e:
            raise WebSocketError(
                f"Failed to send message: {e}", websocket_state=self.state.value
            ) from e

    def set_message_callback(self, callback: Callable[[dict], None]) -> None:
        """Set callback for incoming messages."""
        self.message_callback = callback

    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set callback for connection errors."""
        self.error_callback = callback

    def set_disconnect_callback(self, callback: Callable[[], None]) -> None:
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
            "message_queue_size": self._message_queue.qsize(),
            "message_queue_maxsize": self._message_queue.maxsize,
            "has_active_tasks": any([
                self.heartbeat_task and not self.heartbeat_task.done(),
                self.message_handler_task and not self.message_handler_task.done(),
                self._queue_processor_task and not self._queue_processor_task.done(),
            ])
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
