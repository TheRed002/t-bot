"""
WebSocket Connection Management Helpers for T-Bot Monitoring.

This module provides reusable patterns and utilities for managing WebSocket
connections in the monitoring system, including connection timeout handling,
heartbeat mechanisms, and proper async cleanup.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.core.base.component import BaseComponent
from src.core.exceptions import MonitoringError


class WebSocketState(Enum):
    """WebSocket connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class WebSocketConfig:
    """WebSocket connection configuration."""

    url: str
    connect_timeout: float = 10.0
    heartbeat_interval: float = 30.0
    heartbeat_timeout: float = 5.0
    max_reconnect_attempts: int = 5
    reconnect_backoff_base: float = 2.0
    reconnect_backoff_max: float = 60.0
    message_queue_size: int = 1000
    enable_compression: bool = True


@dataclass
class WebSocketMetrics:
    """WebSocket connection metrics."""

    connection_time: float = 0.0
    last_message_time: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0
    connection_errors: int = 0
    reconnection_attempts: int = 0
    last_heartbeat_time: float = 0.0
    last_heartbeat_latency: float = 0.0


class WebSocketManager(BaseComponent):
    """
    Managed WebSocket connection with automatic reconnection and monitoring.

    Provides proper async context management, heartbeat mechanisms, and
    comprehensive error handling for WebSocket connections.
    """

    def __init__(
        self,
        config: WebSocketConfig,
        message_handler: Callable[[Any], None] | None = None,
        error_handler: Callable[[Exception], None] | None = None,
    ):
        """
        Initialize WebSocket manager.

        Args:
            config: WebSocket configuration
            message_handler: Optional callback for handling incoming messages
            error_handler: Optional callback for handling connection errors
        """
        super().__init__(name=f"WebSocketManager-{config.url}")

        self.config = config
        self.message_handler = message_handler
        self.error_handler = error_handler

        self._state = WebSocketState.DISCONNECTED
        self._metrics = WebSocketMetrics()
        self._websocket: Any = None
        self._connection_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=config.message_queue_size)

        # Async synchronization
        self._state_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._connected_event = asyncio.Event()

        # Reconnection control
        self._reconnect_attempts = 0
        self._last_reconnect_time = 0.0

    @property
    def state(self) -> WebSocketState:
        """Get current connection state."""
        return self._state

    @property
    def metrics(self) -> WebSocketMetrics:
        """Get connection metrics."""
        return self._metrics

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._state == WebSocketState.CONNECTED

    async def connect(self, timeout: float | None = None) -> None:
        """
        Connect to WebSocket with timeout protection.

        Args:
            timeout: Optional connection timeout (uses config default if not provided)

        Raises:
            MonitoringError: If connection fails
            asyncio.TimeoutError: If connection times out
        """
        connect_timeout = timeout or self.config.connect_timeout

        async with self._state_lock:
            if self._state in [WebSocketState.CONNECTED, WebSocketState.CONNECTING]:
                return

            self._state = WebSocketState.CONNECTING
            self._connected_event.clear()

        try:
            await asyncio.wait_for(self._establish_connection(), timeout=connect_timeout)
            self.logger.info(f"WebSocket connected to {self.config.url}")
        except asyncio.TimeoutError:
            await self._set_error_state("Connection timeout")
            raise asyncio.TimeoutError(f"WebSocket connection timeout after {connect_timeout}s")
        except Exception as e:
            await self._set_error_state(f"Connection failed: {e}")
            raise MonitoringError(f"WebSocket connection failed: {e}") from e

    async def disconnect(self, timeout: float = 5.0) -> None:
        """
        Disconnect WebSocket with proper cleanup.

        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        self.logger.info("Disconnecting WebSocket...")

        # Signal shutdown to all tasks
        self._shutdown_event.set()

        async with self._state_lock:
            self._state = WebSocketState.DISCONNECTED
            self._connected_event.clear()

        # Cancel and wait for tasks with timeout
        tasks_to_cancel = []
        if self._connection_task and not self._connection_task.done():
            tasks_to_cancel.append(self._connection_task)
        if self._heartbeat_task and not self._heartbeat_task.done():
            tasks_to_cancel.append(self._heartbeat_task)

        if tasks_to_cancel:
            for task in tasks_to_cancel:
                task.cancel()

            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_cancel, return_exceptions=True), timeout=timeout
                )
            except asyncio.TimeoutError:
                self.logger.warning("Some tasks did not cancel within timeout")

        # Close WebSocket connection with proper async context
        if self._websocket:
            try:
                from src.monitoring.config import WEBSOCKET_CLOSE_TIMEOUT

                if hasattr(self._websocket, "close"):
                    await asyncio.wait_for(self._websocket.close(), timeout=WEBSOCKET_CLOSE_TIMEOUT)
                elif hasattr(self._websocket, "aclose"):
                    await asyncio.wait_for(
                        self._websocket.aclose(), timeout=WEBSOCKET_CLOSE_TIMEOUT
                    )
            except (asyncio.TimeoutError, Exception) as e:
                self.logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self._websocket = None

        self.logger.info("WebSocket disconnected")

    async def send_message(self, message: Any, timeout: float = 5.0) -> None:
        """
        Send message with backpressure handling.

        Args:
            message: Message to send
            timeout: Send timeout

        Raises:
            MonitoringError: If not connected or send fails
            asyncio.TimeoutError: If send times out
        """
        if not self.is_connected:
            raise MonitoringError("WebSocket not connected")

        try:
            # Check queue size for backpressure handling
            if self._message_queue.qsize() >= self.config.message_queue_size * 0.9:
                self.logger.warning("Message queue near capacity - applying backpressure")

            # Add to message queue with backpressure protection
            await asyncio.wait_for(self._message_queue.put(message), timeout=timeout)
            self._metrics.messages_sent += 1

            # Process message queue if WebSocket is ready
            await self._process_message_queue()

        except asyncio.TimeoutError:
            raise asyncio.TimeoutError("Message queue full - backpressure detected")
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            raise MonitoringError(f"Failed to send message: {e}") from e

    @asynccontextmanager
    async def connection_context(self) -> AsyncIterator["WebSocketManager"]:
        """
        Async context manager for WebSocket connection lifecycle.

        Ensures proper connection setup and cleanup even if exceptions occur.
        """
        try:
            await self.connect()
            yield self
        finally:
            await self.disconnect()

    async def wait_connected(self, timeout: float | None = None) -> None:
        """
        Wait for WebSocket to be connected.

        Args:
            timeout: Optional timeout for waiting

        Raises:
            asyncio.TimeoutError: If connection not established within timeout
        """
        if timeout is not None:
            await asyncio.wait_for(self._connected_event.wait(), timeout=timeout)
        else:
            await self._connected_event.wait()

    async def _establish_connection(self) -> None:
        """Establish WebSocket connection (implement with actual WebSocket library)."""
        # This would be implemented with the actual WebSocket library (websockets, aiohttp, etc.)
        # For now, simulate connection establishment with proper timeout handling
        try:
            connection_start = time.time()
            await asyncio.wait_for(self._simulate_connection(), timeout=self.config.connect_timeout)

            self._metrics.connection_time = time.time()
            self._metrics.last_message_time = time.time()

            async with self._state_lock:
                self._state = WebSocketState.CONNECTED
                self._connected_event.set()
                self._reconnect_attempts = 0

            # Start heartbeat mechanism with proper error handling
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Start message processing loop
            self._connection_task = asyncio.create_task(self._message_processing_loop())

            connection_duration = time.time() - connection_start
            self.logger.debug(f"WebSocket connection established in {connection_duration:.3f}s")

        except asyncio.TimeoutError:
            await self._set_error_state("Connection establishment timeout")
            raise
        except Exception as e:
            await self._set_error_state(f"Connection establishment failed: {e}")
            raise

    async def _simulate_connection(self) -> None:
        """Simulate WebSocket connection establishment."""
        # In real implementation, this would establish actual WebSocket connection
        await asyncio.sleep(0.1)  # Simulate connection time

    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop to maintain connection health."""
        while not self._shutdown_event.is_set() and self.is_connected:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                if self._shutdown_event.is_set():
                    break

                # Send heartbeat and measure latency
                heartbeat_start = time.time()
                await self._send_heartbeat()

                # Wait for heartbeat response (simulated)
                await asyncio.wait_for(
                    self._wait_heartbeat_response(), timeout=self.config.heartbeat_timeout
                )

                heartbeat_latency = time.time() - heartbeat_start
                self._metrics.last_heartbeat_time = time.time()
                self._metrics.last_heartbeat_latency = heartbeat_latency

                self.logger.debug(f"Heartbeat latency: {heartbeat_latency:.3f}s")

            except asyncio.TimeoutError:
                self.logger.warning("Heartbeat timeout - connection may be unhealthy")
                await self._handle_connection_error("Heartbeat timeout")
                break
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await self._handle_connection_error(f"Heartbeat error: {e}")
                break

    async def _send_heartbeat(self) -> None:
        """Send heartbeat message (implement with actual WebSocket library)."""
        # This would send an actual ping/heartbeat message
        pass

    async def _wait_heartbeat_response(self) -> None:
        """Wait for heartbeat response (implement with actual WebSocket library)."""
        # This would wait for pong/heartbeat response
        await asyncio.sleep(0.01)  # Simulate response time

    async def _handle_connection_error(self, error_msg: str) -> None:
        """Handle connection errors with optional reconnection."""
        self._metrics.connection_errors += 1

        if self.error_handler:
            try:
                self.error_handler(Exception(error_msg))
            except Exception as e:
                self.logger.error(f"Error handler failed: {e}")

        # Attempt reconnection if enabled
        if self._reconnect_attempts < self.config.max_reconnect_attempts:
            await self._attempt_reconnection(error_msg)
        else:
            await self._set_error_state(f"Max reconnect attempts exceeded: {error_msg}")

    async def _attempt_reconnection(self, reason: str) -> None:
        """Attempt to reconnect with exponential backoff."""
        async with self._state_lock:
            self._state = WebSocketState.RECONNECTING
            self._connected_event.clear()

        self._reconnect_attempts += 1
        self._metrics.reconnection_attempts += 1

        # Calculate backoff delay
        backoff_delay = min(
            self.config.reconnect_backoff_base**self._reconnect_attempts,
            self.config.reconnect_backoff_max,
        )

        self.logger.info(
            f"Attempting reconnection "
            f"{self._reconnect_attempts}/{self.config.max_reconnect_attempts} "
            f"after {backoff_delay}s (reason: {reason})"
        )

        try:
            await asyncio.sleep(backoff_delay)
            await self._establish_connection()
        except Exception as e:
            await self._handle_connection_error(f"Reconnection failed: {e}")

    async def _set_error_state(self, error_msg: str) -> None:
        """Set connection to error state."""
        async with self._state_lock:
            self._state = WebSocketState.ERROR
            self._connected_event.clear()

        self.logger.error(f"WebSocket error: {error_msg}")

    async def _process_message_queue(self) -> None:
        """Process pending messages in the queue."""
        if not self.is_connected or not self._websocket:
            return

        try:
            while not self._message_queue.empty() and self.is_connected:
                try:
                    message = self._message_queue.get_nowait()
                    # Send message through WebSocket (implementation depends on WebSocket library)
                    # This is a placeholder - actual implementation would send via WebSocket
                    await asyncio.sleep(0.001)  # Simulate send time
                    self._message_queue.task_done()
                except asyncio.QueueEmpty:
                    break
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    break
        except Exception as e:
            self.logger.error(f"Error in message queue processing: {e}")

    async def _message_processing_loop(self) -> None:
        """Background loop for processing incoming WebSocket messages."""
        while not self._shutdown_event.is_set() and self.is_connected:
            try:
                # This would typically listen for incoming WebSocket messages
                # For now, just yield control and check for shutdown
                await asyncio.sleep(0.01)

                # Update last message time for connection health
                if self._websocket and self.is_connected:
                    self._metrics.last_message_time = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {e}")
                await self._handle_connection_error(f"Message processing error: {e}")
                break


@asynccontextmanager
async def managed_websocket(
    config: WebSocketConfig,
    message_handler: Callable[[Any], None] | None = None,
    error_handler: Callable[[Exception], None] | None = None,
) -> AsyncIterator[WebSocketManager]:
    """
    Context manager for easy WebSocket management.

    Args:
        config: WebSocket configuration
        message_handler: Optional message handler
        error_handler: Optional error handler

    Yields:
        Configured and connected WebSocket manager
    """
    manager = WebSocketManager(config, message_handler, error_handler)
    async with manager.connection_context():
        yield manager


# Utility functions for common WebSocket patterns


async def with_websocket_timeout(
    coro, timeout: float, error_msg: str = "WebSocket operation timed out"
):
    """Execute WebSocket operation with timeout protection."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise MonitoringError(error_msg) from None


def create_websocket_config(url: str, **kwargs) -> WebSocketConfig:
    """Create WebSocket config with sensible defaults for trading systems."""
    return WebSocketConfig(
        url=url,
        connect_timeout=kwargs.get("connect_timeout", 10.0),
        heartbeat_interval=kwargs.get("heartbeat_interval", 30.0),
        heartbeat_timeout=kwargs.get("heartbeat_timeout", 5.0),
        max_reconnect_attempts=kwargs.get("max_reconnect_attempts", 5),
        reconnect_backoff_base=kwargs.get("reconnect_backoff_base", 2.0),
        reconnect_backoff_max=kwargs.get("reconnect_backoff_max", 60.0),
        message_queue_size=kwargs.get("message_queue_size", 1000),
        enable_compression=kwargs.get("enable_compression", True),
    )
