"""
Common WebSocket Connection Management Utilities

This module contains shared WebSocket utilities to eliminate duplication
across exchange WebSocket implementations. It provides:
- Connection lifecycle management
- Reconnection logic with exponential backoff
- Health monitoring and heartbeat handling
- Message queuing during disconnections
- Stream subscription management
"""

import asyncio
import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import websockets
from websockets.client import WebSocketClientProtocol

from src.core.logging import get_logger
from src.utils.decorators import retry


class WebSocketConnectionManager:
    """Base WebSocket connection manager with common functionality."""

    def __init__(
        self,
        exchange_name: str,
        ws_url: str,
        max_reconnect_attempts: int = 10,
        base_reconnect_delay: int = 1,
        max_reconnect_delay: int = 60,
        message_timeout: int = 60,
        ping_interval: int = 20,
        ping_timeout: int = 10,
    ):
        """
        Initialize WebSocket connection manager.

        Args:
            exchange_name: Name of the exchange
            ws_url: WebSocket URL to connect to
            max_reconnect_attempts: Maximum reconnection attempts
            base_reconnect_delay: Base delay between reconnection attempts (seconds)
            max_reconnect_delay: Maximum delay between reconnections (seconds)
            message_timeout: Timeout for message reception (seconds)
            ping_interval: Ping interval for connection health (seconds)
            ping_timeout: Ping timeout (seconds)
        """
        self.exchange_name = exchange_name
        self.ws_url = ws_url
        self.max_reconnect_attempts = max_reconnect_attempts
        self.base_reconnect_delay = base_reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.message_timeout = message_timeout
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout

        # Connection management
        self.ws: WebSocketClientProtocol | None = None
        self._connection_lock = asyncio.Lock()
        self.connected = False
        self._shutdown = False

        # Reconnection state
        self.reconnect_attempts = 0
        self._reconnect_task: asyncio.Task | None = None

        # Health monitoring
        self.last_message_time: datetime | None = None
        self.last_heartbeat: datetime | None = None
        self._health_check_task: asyncio.Task | None = None

        # Message handling
        self._listener_task: asyncio.Task | None = None
        self.message_queue: list[dict[str, Any]] = []
        self.max_queue_size = 1000

        # Subscriptions and callbacks
        self.active_streams: dict[str, Any] = {}
        self.callbacks: dict[str, list[Callable]] = {}
        self.subscribed_channels: set[str] = set()

        # Metrics
        self._connection_start_time: datetime | None = None
        self._total_messages_received = 0
        self._total_reconnections = 0

        # Initialize logger
        self.logger = get_logger(f"{exchange_name}.websocket")

    async def connect(self) -> bool:
        """
        Establish WebSocket connection with error handling.

        Returns:
            bool: True if connection successful, False otherwise
        """
        async with self._connection_lock:
            try:
                if self._shutdown:
                    return False

                self.logger.info(f"Connecting to {self.exchange_name} WebSocket: {self.ws_url}")

                # Create WebSocket connection with timeout
                self.ws = await asyncio.wait_for(
                    websockets.connect(
                        self.ws_url,
                        ping_interval=self.ping_interval,
                        ping_timeout=self.ping_timeout,
                        close_timeout=10,
                    ),
                    timeout=30.0,  # Connection timeout
                )

                # Update connection state
                self.connected = True
                self.reconnect_attempts = 0
                self._connection_start_time = datetime.now(timezone.utc)
                self.last_message_time = datetime.now(timezone.utc)
                self.last_heartbeat = datetime.now(timezone.utc)

                # Start message listener
                self._listener_task = asyncio.create_task(self._listen_messages())

                # Start health monitoring
                if not self._health_check_task or self._health_check_task.done():
                    self._health_check_task = asyncio.create_task(self._health_monitor())

                # Re-subscribe to channels after reconnection
                await self._resubscribe_channels()

                self.logger.info(f"Successfully connected to {self.exchange_name} WebSocket")
                return True

            except asyncio.TimeoutError:
                self.logger.error(f"WebSocket connection to {self.exchange_name} timed out")
                self.connected = False
                await self._schedule_reconnect()
                return False
            except Exception as e:
                self.logger.error(f"Failed to connect to {self.exchange_name} WebSocket: {e}")
                self.connected = False
                await self._schedule_reconnect()
                return False

    async def disconnect(self) -> None:
        """Disconnect WebSocket with proper cleanup."""
        async with self._connection_lock:
            try:
                self._shutdown = True
                self.logger.info(f"Disconnecting from {self.exchange_name} WebSocket")

                # Cancel tasks
                await self._cancel_tasks()

                # Close WebSocket connection with timeout
                if self.ws and not self.ws.closed:
                    try:
                        await asyncio.wait_for(self.ws.close(), timeout=10.0)
                    except asyncio.TimeoutError:
                        self.logger.warning(f"WebSocket close timed out for {self.exchange_name}")
                    except Exception as e:
                        self.logger.error(f"Error closing WebSocket: {e}")
                    finally:
                        self.ws = None

                # Clear state
                self.connected = False
                self.active_streams.clear()
                self.callbacks.clear()
                self.subscribed_channels.clear()
                self.message_queue.clear()

                self.logger.info(f"Successfully disconnected from {self.exchange_name} WebSocket")

            except Exception as e:
                self.logger.error(f"Error disconnecting from {self.exchange_name} WebSocket: {e}")

    async def send_message(self, message: dict[str, Any]) -> bool:
        """
        Send message through WebSocket connection.

        Args:
            message: Message to send

        Returns:
            bool: True if message sent successfully
        """
        try:
            if not self.ws or self.ws.closed:
                self.logger.warning("WebSocket not connected, queueing message")
                if len(self.message_queue) < self.max_queue_size:
                    self.message_queue.append(message)
                return False

            message_str = json.dumps(message)
            # Add timeout to send operation
            await asyncio.wait_for(self.ws.send(message_str), timeout=10.0)
            self.logger.debug(f"Sent message: {message}")
            return True

        except asyncio.TimeoutError:
            self.logger.error(f"Message send timed out for {self.exchange_name}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False

    def add_callback(self, channel: str, callback: Callable) -> None:
        """
        Add callback for a channel.

        Args:
            channel: Channel name
            callback: Callback function
        """
        if channel not in self.callbacks:
            self.callbacks[channel] = []
        self.callbacks[channel].append(callback)

    def remove_callback(self, channel: str, callback: Callable) -> None:
        """
        Remove callback for a channel.

        Args:
            channel: Channel name
            callback: Callback function to remove
        """
        if channel in self.callbacks:
            try:
                self.callbacks[channel].remove(callback)
                if not self.callbacks[channel]:
                    del self.callbacks[channel]
            except ValueError:
                pass

    async def subscribe_channel(self, channel: str, subscription_data: dict[str, Any]) -> bool:
        """
        Subscribe to a channel.

        Args:
            channel: Channel name
            subscription_data: Subscription message data

        Returns:
            bool: True if subscription successful
        """
        try:
            success = await self.send_message(subscription_data)
            if success:
                self.subscribed_channels.add(channel)
                self.logger.info(f"Subscribed to channel: {channel}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to subscribe to channel {channel}: {e}")
            return False

    async def unsubscribe_channel(self, channel: str, unsubscription_data: dict[str, Any]) -> bool:
        """
        Unsubscribe from a channel.

        Args:
            channel: Channel name
            unsubscription_data: Unsubscription message data

        Returns:
            bool: True if unsubscription successful
        """
        try:
            success = await self.send_message(unsubscription_data)
            if success:
                self.subscribed_channels.discard(channel)
                if channel in self.callbacks:
                    del self.callbacks[channel]
                self.logger.info(f"Unsubscribed from channel: {channel}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from channel {channel}: {e}")
            return False

    async def _listen_messages(self) -> None:
        """Listen for WebSocket messages."""
        try:
            if self.ws is None:
                return
            async for message in self.ws:
                try:
                    # Update message reception time
                    self.last_message_time = datetime.now(timezone.utc)
                    self._total_messages_received += 1

                    # Parse message
                    if isinstance(message, str):
                        data = json.loads(message)
                    else:
                        data = message

                    # Process message with timeout to prevent blocking
                    await asyncio.wait_for(
                        self._process_message(data), timeout=5.0
                    )  # Message processing timeout

                except asyncio.TimeoutError:
                    self.logger.warning(f"Message processing timed out for {self.exchange_name}")
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse message: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")

        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
            self.connected = False
            if not self._shutdown:
                await self._schedule_reconnect()
        except Exception as e:
            self.logger.error(f"Error in message listener: {e}")
            self.connected = False
            if not self._shutdown:
                await self._schedule_reconnect()

    async def _process_message(self, data: dict[str, Any]) -> None:
        """
        Process received message. Should be overridden by subclasses.

        Args:
            data: Parsed message data
        """
        self.logger.debug(f"Received message: {data}")

        # Basic message routing based on channel/type
        message_type = data.get("type") or data.get("channel")
        if message_type and message_type in self.callbacks:
            # Process callbacks concurrently to avoid blocking
            callback_tasks: list[asyncio.Task[Any] | asyncio.Future[Any]] = []
            for callback in self.callbacks[message_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        task = asyncio.create_task(callback(data))
                        callback_tasks.append(task)
                    else:
                        # Run sync callbacks in thread pool to avoid blocking
                        loop = asyncio.get_event_loop()
                        task = loop.run_in_executor(None, callback, data)
                        callback_tasks.append(task)
                except Exception as e:
                    self.logger.error(f"Error creating callback task for {message_type}: {e}")

            # Execute all callbacks concurrently with timeout
            if callback_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*callback_tasks, return_exceptions=True),
                        timeout=10.0,  # Callback execution timeout
                    )
                except asyncio.TimeoutError:
                    self.logger.warning(f"Callback execution timed out for {message_type}")
                except Exception as e:
                    self.logger.error(f"Error executing callbacks for {message_type}: {e}")

    async def _health_monitor(self) -> None:
        """Monitor connection health."""
        try:
            while self.connected and not self._shutdown:
                await asyncio.sleep(30)  # Check every 30 seconds

                if not self.connected or self._shutdown:
                    break

                current_time = datetime.now(timezone.utc)

                # Check if we've received messages recently
                if (
                    self.last_message_time
                    and (current_time - self.last_message_time).total_seconds()
                    > self.message_timeout
                ):
                    self.logger.warning("No messages received within timeout, reconnecting")
                    self.connected = False
                    await self._schedule_reconnect()
                    break

        except Exception as e:
            self.logger.error(f"Error in health monitor: {e}")

    @retry(max_attempts=3, delay=1)
    async def _schedule_reconnect(self) -> None:
        """Schedule reconnection with exponential backoff."""
        if self._shutdown or (self._reconnect_task and not self._reconnect_task.done()):
            return

        self._reconnect_task = asyncio.create_task(self._reconnect())

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        try:
            while (
                self.reconnect_attempts < self.max_reconnect_attempts
                and not self._shutdown
                and not self.connected
            ):
                # Calculate delay with exponential backoff
                delay = min(
                    self.base_reconnect_delay * (2**self.reconnect_attempts),
                    self.max_reconnect_delay,
                )

                self.logger.info(
                    f"Reconnecting in {delay}s (attempt {self.reconnect_attempts + 1}/"
                    f"{self.max_reconnect_attempts})"
                )

                await asyncio.sleep(delay)

                if self._shutdown:
                    break

                self.reconnect_attempts += 1
                self._total_reconnections += 1

                # Attempt to reconnect
                success = await self.connect()
                if success:
                    self.logger.info("Successfully reconnected")
                    break

        except Exception as e:
            self.logger.error(f"Error during reconnection: {e}")

    async def _resubscribe_channels(self) -> None:
        """Re-subscribe to channels after reconnection."""
        if not self.subscribed_channels:
            return

        self.logger.info(f"Re-subscribing to {len(self.subscribed_channels)} channels")

        # Send queued messages first with concurrency limit
        if self.message_queue:
            # Process messages in batches to avoid overwhelming the connection
            batch_size = 10
            semaphore = asyncio.Semaphore(batch_size)

            async def send_queued_message(message: dict[str, Any]) -> None:
                async with semaphore:
                    success = await self.send_message(message)
                    if success:
                        try:
                            self.message_queue.remove(message)
                        except ValueError:
                            pass  # Message already removed by another task
                    return success

            # Send all queued messages concurrently
            tasks = [send_queued_message(msg) for msg in self.message_queue[:]]
            if tasks:
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as e:
                    self.logger.error(f"Error sending queued messages: {e}")

        # This is a basic implementation - subclasses should override
        # to provide exchange-specific resubscription logic

    async def _cancel_tasks(self) -> None:
        """Cancel all running tasks with timeout protection."""
        tasks_to_cancel = [self._health_check_task, self._listener_task, self._reconnect_task]

        cancel_tasks = []
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                cancel_tasks.append(task)

        if cancel_tasks:
            try:
                # Wait for all tasks to cancel with timeout
                await asyncio.wait_for(
                    asyncio.gather(*cancel_tasks, return_exceptions=True), timeout=5.0
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"Task cancellation timed out for {self.exchange_name}")
            except Exception as e:
                self.logger.error(f"Error during task cancellation: {e}")

    def get_connection_stats(self) -> dict[str, Any]:
        """Get connection statistics."""
        return {
            "connected": self.connected,
            "reconnect_attempts": self.reconnect_attempts,
            "total_reconnections": self._total_reconnections,
            "total_messages_received": self._total_messages_received,
            "active_channels": len(self.subscribed_channels),
            "connection_start_time": self._connection_start_time.isoformat()
            if self._connection_start_time
            else None,
            "last_message_time": self.last_message_time.isoformat()
            if self.last_message_time
            else None,
            "message_queue_size": len(self.message_queue),
        }


class AuthenticatedWebSocketManager(WebSocketConnectionManager):
    """WebSocket manager with authentication support."""

    def __init__(
        self,
        exchange_name: str,
        ws_url: str,
        api_key: str,
        api_secret: str,
        passphrase: str = "",
        **kwargs,
    ):
        """
        Initialize authenticated WebSocket manager.

        Args:
            exchange_name: Name of the exchange
            ws_url: WebSocket URL
            api_key: API key
            api_secret: API secret
            passphrase: Optional passphrase
            **kwargs: Additional arguments for base class
        """
        super().__init__(exchange_name, ws_url, **kwargs)
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase

    async def authenticate(self) -> bool:
        """
        Authenticate the WebSocket connection.
        Should be implemented by exchange-specific subclasses.

        Returns:
            bool: True if authentication successful
        """
        self.logger.warning("Authentication not implemented in base class")
        return True


class MultiStreamWebSocketManager:
    """Manager for multiple WebSocket connections."""

    def __init__(self, exchange_name: str) -> None:
        """
        Initialize multi-stream manager.

        Args:
            exchange_name: Name of the exchange
        """
        self.exchange_name = exchange_name
        self.connections: dict[str, WebSocketConnectionManager] = {}
        self.logger = get_logger(f"{exchange_name}.multistream")

    def add_connection(self, name: str, connection: WebSocketConnectionManager) -> None:
        """
        Add a WebSocket connection.

        Args:
            name: Connection name
            connection: WebSocket connection manager
        """
        self.connections[name] = connection

    async def connect_all(self) -> dict[str, bool]:
        """
        Connect all WebSocket connections with proper error handling.

        Returns:
            Dict[str, bool]: Connection results by name
        """
        results: dict[str, bool] = {}

        if not self.connections:
            return results

        # Use asyncio.gather for concurrent connections with timeout
        connection_tasks = [
            asyncio.create_task(connection.connect(), name=f"connect_{name}")
            for name, connection in self.connections.items()
        ]

        try:
            # Execute all connection tasks with timeout
            completed_results = await asyncio.wait_for(
                asyncio.gather(*connection_tasks, return_exceptions=True),
                timeout=60.0,  # Overall timeout for all connections
            )

            # Process results
            for (name, _), result in zip(self.connections.items(), completed_results, strict=False):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to connect {name}: {result}")
                    results[name] = False
                else:
                    results[name] = bool(result)

        except asyncio.TimeoutError:
            self.logger.error("Connection timeout - some connections may have failed")
            # Mark all as failed if timeout occurred
            for name in self.connections.keys():
                results[name] = False
        except Exception as e:
            self.logger.error(f"Unexpected error during connections: {e}")
            for name in self.connections.keys():
                results[name] = False

        return results

    async def disconnect_all(self) -> None:
        """Disconnect all WebSocket connections with proper error handling."""
        if not self.connections:
            return

        # Create disconnect tasks for all connections
        disconnect_tasks = [
            asyncio.create_task(connection.disconnect(), name=f"disconnect_{name}")
            for name, connection in self.connections.items()
        ]

        if disconnect_tasks:
            try:
                # Execute all disconnect tasks with timeout
                await asyncio.wait_for(
                    asyncio.gather(*disconnect_tasks, return_exceptions=True),
                    timeout=30.0,  # Timeout for all disconnections
                )
            except asyncio.TimeoutError:
                self.logger.warning(
                    "Disconnect timeout - some connections may not have closed properly"
                )
            except Exception as e:
                self.logger.error(f"Error during disconnections: {e}")

    def get_connection(self, name: str) -> WebSocketConnectionManager | None:
        """
        Get a specific connection.

        Args:
            name: Connection name

        Returns:
            Optional[WebSocketConnectionManager]: Connection if found
        """
        return self.connections.get(name)

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all connections."""
        return {
            name: connection.get_connection_stats() for name, connection in self.connections.items()
        }


class WebSocketMessageBuffer:
    """Buffer for WebSocket messages during connection issues."""

    def __init__(self, max_size: int = 1000) -> None:
        """
        Initialize message buffer.

        Args:
            max_size: Maximum number of messages to buffer
        """
        self.max_size = max_size
        self.messages: list[dict[str, Any]] = []

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.messages) >= self.max_size

    def add_message(self, message: dict[str, Any]) -> None:
        """
        Add message to buffer.

        Args:
            message: Message to add
        """
        if len(self.messages) >= self.max_size:
            # Remove oldest message
            self.messages.pop(0)
        self.messages.append(message)

    def get_messages(self, count: int | None = None) -> list[dict[str, Any]]:
        """
        Get messages from buffer and clear them.

        Args:
            count: Number of messages to get (all if None)

        Returns:
            List of messages
        """
        if count is None:
            messages = self.messages.copy()
            self.messages.clear()
            return messages
        else:
            messages = self.messages[:count]
            self.messages = self.messages[count:]
            return messages

    def get_message_count(self) -> int:
        """Get number of messages in buffer."""
        return len(self.messages)


class WebSocketHeartbeatManager:
    """Manager for WebSocket heartbeat/ping functionality."""

    def __init__(
        self,
        connection_manager: WebSocketConnectionManager,
        ping_interval: int = 30,
        ping_timeout: int = 10,
    ):
        """
        Initialize heartbeat manager.

        Args:
            connection_manager: WebSocket connection manager
            ping_interval: Interval between pings (seconds)
            ping_timeout: Ping timeout (seconds)
        """
        self.connection_manager = connection_manager
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.is_running = False
        self._heartbeat_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start heartbeat monitoring."""
        if self.is_running:
            return

        self.is_running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Stop heartbeat monitoring."""
        self.is_running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    async def _heartbeat_loop(self) -> None:
        """Main heartbeat loop."""
        while self.is_running:
            try:
                await asyncio.sleep(self.ping_interval)
                if hasattr(self.connection_manager, "send_ping"):
                    # Add timeout to prevent heartbeat hanging
                    try:
                        await asyncio.wait_for(self.connection_manager.send_ping(), timeout=5.0)
                    except asyncio.TimeoutError:
                        if hasattr(self.connection_manager, "logger"):
                            self.connection_manager.logger.warning("Heartbeat ping timed out")
                    except Exception as ping_error:
                        if hasattr(self.connection_manager, "logger"):
                            self.connection_manager.logger.error(
                                f"Heartbeat ping error: {ping_error}"
                            )
            except Exception as e:
                # Log error but continue heartbeat
                if hasattr(self.connection_manager, "logger"):
                    # Logger error is not a coroutine - call directly
                    try:
                        self.connection_manager.logger.error(f"Heartbeat error: {e}")
                    except Exception:
                        # Fallback if logger call fails
                        pass
                continue


class WebSocketSubscriptionManager:
    """Manager for WebSocket subscriptions and callbacks."""

    def __init__(self) -> None:
        """Initialize subscription manager."""
        self.active_subscriptions: set[str] = set()
        self.subscription_callbacks: dict[str, Callable] = {}

    def add_subscription(self, stream_name: str, callback: Callable) -> None:
        """
        Add subscription.

        Args:
            stream_name: Name of the stream
            callback: Callback function
        """
        self.active_subscriptions.add(stream_name)
        self.subscription_callbacks[stream_name] = callback

    def remove_subscription(self, stream_name: str) -> None:
        """
        Remove subscription.

        Args:
            stream_name: Name of the stream
        """
        self.active_subscriptions.discard(stream_name)
        self.subscription_callbacks.pop(stream_name, None)

    def get_subscriptions(self) -> list[str]:
        """Get list of active subscriptions."""
        return list(self.active_subscriptions)

    async def handle_message(self, message: dict[str, Any]) -> None:
        """
        Handle received message.

        Args:
            message: Received message
        """
        stream_name = message.get("stream")
        if stream_name and stream_name in self.subscription_callbacks:
            callback = self.subscription_callbacks[stream_name]
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Add timeout to prevent callback hanging
                    await asyncio.wait_for(callback(message), timeout=10.0)
                else:
                    # Run sync callbacks in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, callback, message)
            except asyncio.TimeoutError:
                logger.warning(f"Callback timeout for stream {stream_name}")
            except Exception as e:
                logger.error(f"Callback error for stream {stream_name}: {e}")


class WebSocketStreamManager:
    """Manager for WebSocket streams."""

    def __init__(self, max_streams: int = 50) -> None:
        """
        Initialize stream manager.

        Args:
            max_streams: Maximum number of streams
        """
        self.max_streams = max_streams
        self.active_streams: dict[str, dict[str, Any]] = {}
        self.stream_handlers: dict[str, Callable] = {}

    def add_stream(self, stream_id: str, stream_config: dict[str, Any], handler: Callable) -> bool:
        """
        Add stream.

        Args:
            stream_id: Stream identifier
            stream_config: Stream configuration
            handler: Stream handler

        Returns:
            True if added successfully
        """
        if len(self.active_streams) >= self.max_streams:
            return False

        self.active_streams[stream_id] = stream_config
        self.stream_handlers[stream_id] = handler
        return True

    def remove_stream(self, stream_id: str) -> bool:
        """
        Remove stream.

        Args:
            stream_id: Stream identifier

        Returns:
            True if removed successfully
        """
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
            self.stream_handlers.pop(stream_id, None)
            return True
        return False

    def get_stream_count(self) -> int:
        """Get number of active streams."""
        return len(self.active_streams)

    def is_at_capacity(self) -> bool:
        """Check if stream manager is at capacity."""
        return len(self.active_streams) >= self.max_streams

    async def handle_stream_message(self, stream_id: str, message: dict[str, Any]) -> None:
        """
        Handle stream message.

        Args:
            stream_id: Stream identifier
            message: Stream message
        """
        if stream_id in self.stream_handlers:
            handler = self.stream_handlers[stream_id]
            try:
                if asyncio.iscoroutinefunction(handler):
                    # Add timeout to prevent handler hanging
                    await asyncio.wait_for(handler(message), timeout=10.0)
                else:
                    # Run sync handlers in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, handler, message)
            except asyncio.TimeoutError:
                logger.warning(f"Handler timeout for stream {stream_id}")
            except Exception as e:
                logger.error(f"Handler error for stream {stream_id}: {e}")
