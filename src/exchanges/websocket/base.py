"""Enhanced Base WebSocket manager for exchanges."""

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import websockets
from websockets.client import WebSocketClientProtocol

from src.core.config import Config
from src.core.exceptions import ExchangeConnectionError
from src.core.logging import get_logger
from src.error_handling.error_handler import ErrorHandler


class EnhancedBaseWebSocketManager(ABC):
    """
    Enhanced shared WebSocket management logic for all exchanges.

    This eliminates duplication of WebSocket connection handling,
    reconnection logic, subscription management, and provides
    unified resource management and health monitoring.

    Features:
    - Automatic reconnection with exponential backoff
    - Health monitoring and metrics collection
    - Proper resource cleanup and leak prevention
    - Unified error handling
    - Performance optimizations
    """

    def __init__(
        self,
        config: Config,
        url: str,
        exchange_name: str,
        reconnect_delay: int = 1,
        max_reconnect_attempts: int = 10,
        ping_interval: int = 20,
        ping_timeout: int = 10,
        message_timeout: int = 60,
    ):
        """
        Initialize enhanced WebSocket manager.

        Args:
            config: Application configuration
            url: WebSocket URL
            exchange_name: Exchange name for logging
            reconnect_delay: Base delay between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts
            ping_interval: Ping interval in seconds
            ping_timeout: Ping timeout in seconds
            message_timeout: Message timeout in seconds
        """
        self.config = config
        self.url = url
        self.exchange_name = exchange_name
        self.base_reconnect_delay = reconnect_delay
        self.max_reconnect_delay = 60  # Maximum delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.message_timeout = message_timeout

        # WebSocket connection
        self.ws: WebSocketClientProtocol | None = None
        self._connection_lock = asyncio.Lock()

        # Subscription and handler management
        self.subscriptions: set[str] = set()
        self.handlers: dict[str, Callable] = {}
        self.callbacks: dict[str, list[Callable]] = {}

        # Connection state
        self._running = False
        self._shutdown = False
        self.connected = False
        self._reconnect_count = 0

        # Health monitoring
        self.last_message_time: datetime | None = None
        self.last_heartbeat: datetime | None = None
        self._health_check_task: asyncio.Task | None = None

        # Connection metrics
        self._connection_start_time: datetime | None = None
        self._total_messages_received = 0
        self._total_messages_sent = 0
        self._total_reconnections = 0
        self._error_count = 0

        # Background tasks
        self._tasks: set[asyncio.Task] = set()
        self._listener_task: asyncio.Task | None = None
        self._ping_task: asyncio.Task | None = None
        self._reconnect_task: asyncio.Task | None = None

        # Error handling
        self.error_handler = ErrorHandler(config)

        # Logger
        self._logger = get_logger(f"{exchange_name}.websocket.base")

        self._logger.info(f"Initialized enhanced WebSocket manager for {exchange_name}")

    async def connect(self) -> bool:
        """
        Connect to WebSocket with enhanced error handling and reconnection.

        Returns:
            bool: True if connected successfully, False otherwise
        """
        async with self._connection_lock:
            try:
                if self._shutdown:
                    return False

                self._logger.info(f"Connecting to {self.exchange_name} WebSocket: {self.url}")

                # Connect with enhanced options
                self.ws = await websockets.connect(
                    self.url,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                    close_timeout=10,
                    max_size=2**20,  # 1MB max message size
                    read_limit=2**16,  # 64KB read buffer
                    write_limit=2**16,  # 64KB write buffer
                )

                self.connected = True
                self._running = True
                self._reconnect_count = 0
                self._connection_start_time = datetime.now(timezone.utc)
                self.last_message_time = datetime.now(timezone.utc)
                self.last_heartbeat = datetime.now(timezone.utc)

                # Call connection hook
                await self._on_connect()

                # Start background tasks
                await self._start_background_tasks()

                # Resubscribe to channels
                await self._resubscribe()

                self._logger.info(f"Successfully connected to {self.exchange_name} WebSocket")
                return True

            except Exception as e:
                self._logger.error(f"Failed to connect to {self.exchange_name} WebSocket: {e!s}")
                self.connected = False
                await self._schedule_reconnect()
                return False

    async def disconnect(self) -> None:
        """Disconnect from WebSocket with proper cleanup."""
        async with self._connection_lock:
            try:
                self._shutdown = True
                self._running = False
                self.connected = False

                self._logger.info(f"Disconnecting from {self.exchange_name} WebSocket")

                # Cancel all background tasks
                for task in [
                    self._health_check_task,
                    self._listener_task,
                    self._ping_task,
                    self._reconnect_task,
                ]:
                    if task and not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                # Cancel all other tasks
                for task in self._tasks:
                    if not task.done():
                        task.cancel()

                # Wait for all tasks to complete
                if self._tasks:
                    await asyncio.gather(*self._tasks, return_exceptions=True)

                # Close WebSocket connection
                if self.ws and not self.ws.closed:
                    await self.ws.close()
                    self.ws = None

                # Clear state
                self._tasks.clear()
                self.subscriptions.clear()
                self.handlers.clear()
                self.callbacks.clear()

                self._logger.info(f"Successfully disconnected from {self.exchange_name} WebSocket")

            except Exception as e:
                self._logger.error(
                    f"Error disconnecting from {self.exchange_name} WebSocket: {e!s}"
                )

    async def _start_background_tasks(self) -> None:
        """Start all background tasks."""
        try:
            # Start message listener
            self._listener_task = asyncio.create_task(self._listen())
            self._tasks.add(self._listener_task)
            self._listener_task.add_done_callback(self._tasks.discard)

            # Start ping task
            self._ping_task = asyncio.create_task(self._ping_loop())
            self._tasks.add(self._ping_task)
            self._ping_task.add_done_callback(self._tasks.discard)

            # Start health monitoring
            if not self._health_check_task or self._health_check_task.done():
                self._health_check_task = asyncio.create_task(self._health_monitor())
                self._tasks.add(self._health_check_task)
                self._health_check_task.add_done_callback(self._tasks.discard)

        except Exception as e:
            self._logger.error(f"Failed to start background tasks: {e!s}")
            raise

    async def _listen(self) -> None:
        """Listen for messages from WebSocket with enhanced error handling."""
        try:
            while not self._shutdown and self.connected and self.ws and not self.ws.closed:
                try:
                    message = await self.ws.recv()

                    # Update metrics
                    self.last_message_time = datetime.now(timezone.utc)
                    self._total_messages_received += 1

                    # Parse and route message
                    data = self._parse_message(message)
                    await self._route_message(data)

                except websockets.exceptions.ConnectionClosed:
                    self._logger.warning(f"{self.exchange_name} WebSocket connection closed")
                    break
                except websockets.exceptions.WebSocketException as e:
                    self._logger.error(f"{self.exchange_name} WebSocket error: {e!s}")
                    break
                except Exception as e:
                    self._logger.error(f"Error processing message: {e!s}")
                    self._error_count += 1

        except Exception as e:
            self._logger.error(f"Error in message listener: {e!s}")
        finally:
            if self.connected and not self._shutdown:
                await self._handle_disconnect()

    async def _ping_loop(self) -> None:
        """Send periodic pings to keep connection alive."""
        while not self._shutdown and self.connected:
            try:
                if self.ws and not self.ws.closed:
                    await self.ws.ping()
                    self.last_heartbeat = datetime.now(timezone.utc)
                    self._logger.debug(f"{self.exchange_name} ping sent")

                await asyncio.sleep(self.ping_interval)

            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                self._logger.error(f"Ping error: {e!s}")
                break

    async def _health_monitor(self) -> None:
        """Monitor connection health and trigger reconnection if needed."""
        while not self._shutdown and self.connected:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                if self._shutdown:
                    break

                # Check if we've received messages recently
                if self.last_message_time:
                    time_since_last_message = (
                        datetime.now(timezone.utc) - self.last_message_time
                    ).total_seconds()

                    if time_since_last_message > self.message_timeout:
                        self._logger.warning(
                            f"No messages received for {time_since_last_message:.1f}s, "
                            "triggering health check reconnection"
                        )
                        self.connected = False
                        await self._schedule_reconnect()
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in health monitor: {e!s}")

    async def _handle_disconnect(self) -> None:
        """Handle disconnection and prepare for reconnection."""
        try:
            self.connected = False
            self._total_reconnections += 1

            if self.ws:
                await self.ws.close()
                self.ws = None

            await self._on_disconnect()

            if not self._shutdown:
                await self._schedule_reconnect()

        except Exception as e:
            self._logger.error(f"Error handling disconnect: {e!s}")

    async def _schedule_reconnect(self) -> None:
        """Schedule reconnection with exponential backoff."""
        if self._shutdown or self._reconnect_count >= self.max_reconnect_attempts:
            if self._reconnect_count >= self.max_reconnect_attempts:
                self._logger.error(
                    f"Max reconnection attempts ({self.max_reconnect_attempts}) reached"
                )
            return

        self._reconnect_count += 1

        # Calculate exponential backoff delay
        delay = min(
            self.base_reconnect_delay * (2 ** (self._reconnect_count - 1)),
            self.max_reconnect_delay,
        )

        self._logger.info(
            f"Scheduling reconnection in {delay:.1f}s "
            f"(attempt {self._reconnect_count}/{self.max_reconnect_attempts})"
        )

        # Schedule reconnection
        self._reconnect_task = asyncio.create_task(self._reconnect_after_delay(delay))

    async def _reconnect_after_delay(self, delay: float) -> None:
        """Reconnect after specified delay."""
        try:
            await asyncio.sleep(delay)

            if self._shutdown:
                return

            self._logger.info(f"Attempting to reconnect {self.exchange_name}...")

            # Store current subscriptions for resubscription
            subscriptions_to_restore = self.subscriptions.copy()
            callbacks_to_restore = dict(self.callbacks)

            # Attempt reconnection
            if await self.connect():
                self._logger.info(
                    f"{self.exchange_name} reconnection successful, restoring subscriptions..."
                )

                # Restore subscriptions
                for subscription in subscriptions_to_restore:
                    try:
                        await self._send_subscribe(subscription)
                    except Exception as e:
                        self._logger.error(f"Failed to restore subscription {subscription}: {e!s}")

                # Restore callbacks
                self.callbacks.update(callbacks_to_restore)

                self._logger.info(f"{self.exchange_name} subscription restoration completed")
            else:
                self._logger.error(f"{self.exchange_name} reconnection failed")

        except Exception as e:
            self._logger.error(f"Error during reconnection: {e!s}")

    async def _resubscribe(self) -> None:
        """Resubscribe to all channels after reconnection."""
        for channel in self.subscriptions.copy():
            try:
                await self._send_subscribe(channel)
                self._logger.debug(f"Resubscribed to {channel}")
            except Exception as e:
                self._logger.error(f"Failed to resubscribe to {channel}: {e!s}")

    async def subscribe(self, channel: str, handler: Callable) -> None:
        """
        Subscribe to a channel with enhanced tracking.

        Args:
            channel: Channel to subscribe to
            handler: Handler function for channel messages
        """
        try:
            self.subscriptions.add(channel)
            self.handlers[channel] = handler

            # Track callbacks for this channel
            if channel not in self.callbacks:
                self.callbacks[channel] = []
            self.callbacks[channel].append(handler)

            if self.ws and not self.ws.closed:
                await self._send_subscribe(channel)

            self._logger.info(f"Subscribed to {channel}")

        except Exception as e:
            self._logger.error(f"Failed to subscribe to {channel}: {e!s}")
            raise

    async def unsubscribe(self, channel: str) -> None:
        """
        Unsubscribe from a channel with enhanced cleanup.

        Args:
            channel: Channel to unsubscribe from
        """
        try:
            self.subscriptions.discard(channel)
            self.handlers.pop(channel, None)
            self.callbacks.pop(channel, None)

            if self.ws and not self.ws.closed:
                await self._send_unsubscribe(channel)

            self._logger.info(f"Unsubscribed from {channel}")

        except Exception as e:
            self._logger.error(f"Failed to unsubscribe from {channel}: {e!s}")

    async def _route_message(self, data: dict[str, Any]) -> None:
        """
        Route message to appropriate handler with enhanced error handling.

        Args:
            data: Parsed message data
        """
        try:
            channel = self._extract_channel(data)

            if channel and channel in self.handlers:
                handler = self.handlers[channel]

                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)

                except Exception as e:
                    self._logger.error(f"Handler error for {channel}: {e!s}")
                    self._error_count += 1
            else:
                # Default handler for unmatched messages
                await self._on_message(data)

        except Exception as e:
            self._logger.error(f"Error routing message: {e!s}")
            self._error_count += 1

    async def send_message(self, message: dict[str, Any]) -> None:
        """
        Send a message through WebSocket with enhanced error handling.

        Args:
            message: Message to send
        """
        try:
            if not self.ws or self.ws.closed:
                raise ExchangeConnectionError("WebSocket not connected")

            message_str = json.dumps(message)
            await self.ws.send(message_str)

            self._total_messages_sent += 1
            self._logger.debug(f"Sent message: {message_str}")

        except Exception as e:
            self._logger.error(f"Failed to send message: {e!s}")
            self._error_count += 1
            raise

    def is_connected(self) -> bool:
        """Check if WebSocket is connected and healthy."""
        return self.connected and self.ws is not None and not self.ws.closed and not self._shutdown

    async def health_check(self) -> bool:
        """
        Perform comprehensive health check.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            if self._shutdown or not self.connected:
                return False

            if not self.ws or self.ws.closed:
                return False

            # Check if we've received messages recently
            if self.last_message_time:
                time_since_last_message = (
                    datetime.now(timezone.utc) - self.last_message_time
                ).total_seconds()

                if time_since_last_message > self.message_timeout:
                    return False

            return True

        except Exception as e:
            self._logger.error(f"Health check failed: {e!s}")
            return False

    def get_connection_metrics(self) -> dict[str, Any]:
        """
        Get comprehensive connection metrics.

        Returns:
            Dict containing metrics
        """
        try:
            uptime = None
            if self._connection_start_time:
                uptime = (datetime.now(timezone.utc) - self._connection_start_time).total_seconds()

            time_since_last_message = None
            if self.last_message_time:
                time_since_last_message = (
                    datetime.now(timezone.utc) - self.last_message_time
                ).total_seconds()

            return {
                "exchange": self.exchange_name,
                "connected": self.connected,
                "healthy": self.health_check(),
                "uptime_seconds": uptime,
                "total_messages_received": self._total_messages_received,
                "total_messages_sent": self._total_messages_sent,
                "total_reconnections": self._total_reconnections,
                "error_count": self._error_count,
                "reconnect_attempts": self._reconnect_count,
                "max_reconnect_attempts": self.max_reconnect_attempts,
                "subscriptions": len(self.subscriptions),
                "time_since_last_message": time_since_last_message,
                "message_timeout": self.message_timeout,
                "shutdown": self._shutdown,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self._logger.error(f"Failed to get metrics: {e!s}")
            return {"error": str(e)}

    def get_subscriptions(self) -> set[str]:
        """Get current subscriptions."""
        return self.subscriptions.copy()

    # Abstract methods for exchange-specific implementation

    @abstractmethod
    def _parse_message(self, message: str) -> dict[str, Any]:
        """
        Parse raw message from WebSocket.

        Args:
            message: Raw message string

        Returns:
            Parsed message data
        """
        pass

    @abstractmethod
    def _extract_channel(self, data: dict[str, Any]) -> str | None:
        """
        Extract channel from message data.

        Args:
            data: Message data

        Returns:
            Channel name or None
        """
        pass

    @abstractmethod
    async def _send_subscribe(self, channel: str) -> None:
        """
        Send subscription message for a channel.

        Args:
            channel: Channel to subscribe to
        """
        pass

    @abstractmethod
    async def _send_unsubscribe(self, channel: str) -> None:
        """
        Send unsubscription message for a channel.

        Args:
            channel: Channel to unsubscribe from
        """
        pass

    # Hooks for subclasses

    async def _on_connect(self) -> None:
        """Hook called when connected."""
        pass

    async def _on_disconnect(self) -> None:
        """Hook called when disconnected."""
        pass

    async def _on_message(self, data: dict[str, Any]) -> None:
        """
        Hook for handling unmatched messages.

        Args:
            data: Message data
        """
        self._logger.debug(f"Unhandled message: {data}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


# Keep old class for backward compatibility
BaseWebSocketManager = EnhancedBaseWebSocketManager
