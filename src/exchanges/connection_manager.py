"""
Connection manager for exchange APIs and WebSocket streams.

This module manages connections to exchange REST APIs and WebSocket streams,
providing automatic reconnection, heartbeat monitoring, and connection pooling.

CRITICAL: This integrates with P-001 (core types, exceptions, config)
and P-002A (error handling) components.
"""

import asyncio
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from typing import Any

# LoggerMixin not needed - BaseComponent already provides logging
from src.core.config import Config

# MANDATORY: Import from P-001
from src.core.exceptions import ExchangeConnectionError, ValidationError

# Logger is provided by BaseExchange (via BaseComponent)
from src.core.types import ConnectionType
from src.error_handling.connection_manager import ConnectionManager as ErrorConnectionManager

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

# Recovery scenarios - imported at top level for better performance
from src.error_handling.recovery_scenarios import NetworkDisconnectionRecovery
# MANDATORY: Import from P-007 (advanced rate limiting)
from src.exchanges.advanced_rate_limiter import get_global_rate_limiter
from src.exchanges.health_monitor import ConnectionHealthMonitor
from src.exchanges.high_performance_websocket import WebSocketConnectionPool
from src.utils import TIMEOUTS, log_calls, retry, measure_latency, test_connection


class WebSocketConnection:
    """
    WebSocket connection wrapper with automatic reconnection.

    This class manages a single WebSocket connection with automatic
    reconnection, heartbeat monitoring, and message queuing.
    """

    def __init__(self, url: str, exchange_name: str, config: Config):
        """
        Initialize WebSocket connection.

        Args:
            url: WebSocket URL
            exchange_name: Name of the exchange
            config: Application configuration
        """
        self.url = url
        self.exchange_name = exchange_name
        self.config = config
        self.error_handler = ErrorHandler(config.error_handling)

        # Connection state
        self.connected = False
        self.connecting = False
        self.last_heartbeat = None
        self.last_message = None

        # Message queue for reconnection
        self.message_queue: list[dict[str, Any]] = []
        self.max_queue_size = 1000

        # Reconnection settings
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0
        self.max_reconnect_delay = 60.0
        self.heartbeat_interval = 30.0

        # Callbacks
        self.on_message: Callable | None = None
        self.on_connect: Callable | None = None
        self.on_disconnect: Callable | None = None
        self.on_error: Callable | None = None

        self.logger.info(f"Initialized WebSocket connection for {exchange_name}")

    @retry(max_attempts=3, base_delay=1.0)
    @log_calls
    async def connect(self) -> bool:
        """
        Connect to the WebSocket.

        Returns:
            bool: True if connection successful, False otherwise
        """
        if self.connecting or self.connected:
            return self.connected

        self.connecting = True

        for attempt in range(self.max_reconnect_attempts):
            try:
                # TODO: Remove in production - Implement actual WebSocket connection
                # This is a placeholder implementation
                self.logger.info(f"Connecting to WebSocket {self.url}(attempt {attempt + 1})")

                # Simulate connection delay
                await asyncio.sleep(0.1)

                self.connected = True
                self.connecting = False
                self.last_heartbeat = datetime.now()

                if self.on_connect:
                    await self.on_connect()

                self.logger.info(f"WebSocket connected to {self.exchange_name}")
                return True

            except Exception as e:
                self.logger.error(f"WebSocket connection failed(attempt {attempt + 1}): {e!s}")

                if attempt < self.max_reconnect_attempts - 1:
                    delay = min(self.reconnect_delay * (2**attempt), self.max_reconnect_delay)
                    await asyncio.sleep(delay)

        self.connecting = False
        self.logger.error(
            f"Failed to connect to WebSocket after {self.max_reconnect_attempts} attempts"
        )
        return False

    async def disconnect(self) -> None:
        """Disconnect from the WebSocket."""
        if not self.connected:
            return

        try:
            # TODO: Remove in production - Implement actual WebSocket
            # disconnection
            self.logger.info(f"Disconnecting from WebSocket {self.url}")

            self.connected = False

            if self.on_disconnect:
                await self.on_disconnect()

            self.logger.info(f"WebSocket disconnected from {self.exchange_name}")

        except Exception as e:
            self.logger.error(f"Error during WebSocket disconnection: {e!s}")

    async def send_message(self, message: dict[str, Any]) -> bool:
        """
        Send a message through the WebSocket.

        Args:
            message: Message to send

        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.connected:
            # Queue message for later
            if len(self.message_queue) < self.max_queue_size:
                self.message_queue.append(message)
                self.logger.debug(f"Queued message for {self.exchange_name}")
            return False

        try:
            # TODO: Remove in production - Implement actual WebSocket message
            # sending
            self.logger.debug(f"Sending message to {self.exchange_name}: {message}")

            # Simulate message sending
            await asyncio.sleep(0.001)

            return True

        except Exception as e:
            self.logger.error(f"Failed to send message to {self.exchange_name}: {e!s}")
            return False

    async def subscribe(self, channel: str, symbol: str | None = None) -> bool:
        """
        Subscribe to a WebSocket channel.

        Args:
            channel: Channel to subscribe to
            symbol: Optional symbol for the channel

        Returns:
            bool: True if subscription successful, False otherwise
        """
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": [f"{channel}{f'@{symbol}' if symbol else ''}"],
            "id": int(time.time() * 1000),
        }

        return await self.send_message(subscribe_message)

    async def unsubscribe(self, channel: str, symbol: str | None = None) -> bool:
        """
        Unsubscribe from a WebSocket channel.

        Args:
            channel: Channel to unsubscribe from
            symbol: Optional symbol for the channel

        Returns:
            bool: True if unsubscription successful, False otherwise
        """
        unsubscribe_message = {
            "method": "UNSUBSCRIBE",
            "params": [f"{channel}{f'@{symbol}' if symbol else ''}"],
            "id": int(time.time() * 1000),
        }

        return await self.send_message(unsubscribe_message)

    async def heartbeat(self) -> bool:
        """
        Send heartbeat to keep connection alive.

        Returns:
            bool: True if heartbeat successful, False otherwise
        """
        if not self.connected:
            return False

        try:
            # TODO: Remove in production - Implement actual heartbeat
            heartbeat_message = {"method": "ping"}
            success = await self.send_message(heartbeat_message)

            if success:
                self.last_heartbeat = datetime.now()

            return success

        except Exception as e:
            self.logger.error(f"Heartbeat failed for {self.exchange_name}: {e!s}")
            return False

    def is_healthy(self) -> bool:
        """
        Check if the WebSocket connection is healthy.

        Returns:
            bool: True if healthy, False otherwise
        """
        if not self.connected:
            return False

        if not self.last_heartbeat:
            return True  # No heartbeat yet, assume healthy

        # Check if heartbeat is within acceptable range
        time_since_heartbeat = datetime.now() - self.last_heartbeat
        return time_since_heartbeat.total_seconds() < self.heartbeat_interval * 2

    async def process_queued_messages(self) -> int:
        """
        Process queued messages after reconnection.

        Returns:
            int: Number of messages processed
        """
        if not self.connected or not self.message_queue:
            return 0

        processed = 0
        while self.message_queue and self.connected:
            message = self.message_queue.pop(0)
            if await self.send_message(message):
                processed += 1
            else:
                break

        self.logger.info(f"Processed {processed} queued messages for {self.exchange_name}")
        return processed


class ConnectionManager:
    """
    Connection manager for exchange APIs and WebSocket streams.

    This class manages multiple connections to exchange APIs and WebSocket
    streams, providing connection pooling, health monitoring, and automatic
    reconnection capabilities.
    """

    def __init__(self, config: Config, exchange_name: str):
        """
        Initialize connection manager.

        Args:
            config: Application configuration
            exchange_name: Name of the exchange
        """
        self.config = config
        self.exchange_name = exchange_name
        self.error_handler = ErrorHandler(config.error_handling)
        self.error_connection_manager = ErrorConnectionManager(config.error_handling)

        # P-007: Advanced rate limiting and connection management integration
        self.advanced_rate_limiter = get_global_rate_limiter(config)
        self.health_monitor = ConnectionHealthMonitor(config)
        self.websocket_pool = WebSocketConnectionPool(
            exchange=exchange_name,
            max_connections=10,
            max_messages_per_second=100,
            max_subscriptions=50,
        )

        # Connection pools
        # Will be populated by actual implementations
        self.rest_connections: dict[str, Any] = {}
        self.websocket_connections: dict[str, WebSocketConnection] = {}

        # Connection monitoring
        self.connection_health: dict[str, bool] = defaultdict(lambda: True)
        self.last_health_check = datetime.now()
        self.health_check_interval = 60.0  # seconds

        # Connection limits
        self.max_rest_connections = 10
        self.max_websocket_connections = 5

        # TODO: Remove in production
        self.logger.debug(
            f"ConnectionManager initialized with P-007 components for {exchange_name}"
        )
        self.logger.info(f"Initialized connection manager for {exchange_name}")

    async def _handle_connection_error(
        self, error: Exception, operation: str, connection_id: str | None = None
    ) -> None:
        """
        Handle connection-related errors using the error handler.

        Args:
            error: The exception that occurred
            operation: The operation being performed
            connection_id: The connection identifier
        """
        try:
            # Create error context
            error_context = self.error_handler.create_error_context(
                error=error,
                component="exchange_connection_manager",
                operation=operation,
                details={
                    "exchange_name": self.exchange_name,
                    "operation": operation,
                    "connection_id": connection_id,
                },
            )

            # Use network disconnection recovery for connection failures
            recovery_scenario = NetworkDisconnectionRecovery(self.config)

            # Handle the error
            await self.error_handler.handle_error(error, error_context, recovery_scenario)

        except Exception as e:
            # Fallback to basic logging if error handling fails
            self.logger.error(f"Error handling failed for {operation}: {e!s}")

    async def get_rest_connection(self, endpoint: str = "default") -> Any | None:
        """
        Get a REST API connection.

        Args:
            endpoint: Endpoint identifier

        Returns:
            Optional[Any]: REST connection or None if not available
        """
        # TODO: Remove in production - Implement actual REST connection pooling
        # This is a placeholder implementation
        if endpoint not in self.rest_connections:
            self.logger.debug(f"Creating new REST connection for {endpoint}")
            # Simulate connection creation
            self.rest_connections[endpoint] = {"endpoint": endpoint, "connected": True}

        return self.rest_connections.get(endpoint)

    async def create_websocket_connection(
        self, url: str, connection_id: str = "default"
    ) -> WebSocketConnection:
        """
        Create a new WebSocket connection.

        Args:
            url: WebSocket URL
            connection_id: Unique identifier for the connection

        Returns:
            WebSocketConnection: WebSocket connection instance
        """
        if connection_id in self.websocket_connections:
            self.logger.warning(f"WebSocket connection {connection_id} already exists")
            return self.websocket_connections[connection_id]

        if len(self.websocket_connections) >= self.max_websocket_connections:
            self.logger.warning(f"Maximum WebSocket connections reached for {self.exchange_name}")
            # Return the first available connection
            return next(iter(self.websocket_connections.values()))

        connection = WebSocketConnection(url, self.exchange_name, self.config)
        self.websocket_connections[connection_id] = connection

        self.logger.info(f"Created WebSocket connection {connection_id} for {self.exchange_name}")
        return connection

    async def get_connection(self, exchange: str, stream_type: str) -> Any | None:
        """
        Get a WebSocket connection from the pool.

        Args:
            exchange: Exchange name
            stream_type: Type of stream (ticker, orderbook, trades, etc.)

        Returns:
            Optional[Any]: WebSocket connection or None if not available

        Raises:
            ExchangeConnectionError: If connection retrieval fails
            ValidationError: If parameters are invalid
        """
        try:
            # Validate parameters
            if not exchange or not stream_type:
                raise ValidationError("Exchange and stream_type are required")

            # Convert stream_type to ConnectionType
            try:
                connection_type = ConnectionType(stream_type)
            except ValueError:
                raise ValidationError(f"Invalid stream type: {stream_type}")

            # Get connection from pool
            pooled_connection = await self.websocket_pool.get_connection(connection_type)

            if pooled_connection:
                # Register with health monitor
                await self.health_monitor.monitor_connection(pooled_connection.connection)

                self.logger.debug(
                    "Retrieved connection from pool",
                    exchange=exchange,
                    stream_type=stream_type,
                    connection_id=pooled_connection.connection_id,
                )
                return pooled_connection.connection
            else:
                self.logger.warning(
                    "No available connection in pool", exchange=exchange, stream_type=stream_type
                )
                return None

        except (ValidationError, ExchangeConnectionError):
            raise
        except Exception as e:
            self.logger.error(
                "Failed to get connection",
                exchange=exchange,
                stream_type=stream_type,
                error=str(e),
            )
            raise ExchangeConnectionError(f"Connection retrieval failed: {e!s}")

    async def release_connection(self, exchange: str, connection: Any) -> None:
        """
        Release a connection back to the pool.

        Args:
            exchange: Exchange name
            connection: WebSocket connection to release

        Raises:
            ExchangeConnectionError: If connection release fails
            ValidationError: If parameters are invalid
        """
        try:
            # Validate parameters
            if not exchange or not connection:
                raise ValidationError("Exchange and connection are required")

            # Find the pooled connection
            connection_id = getattr(connection, "id", str(id(connection)))

            # Release to pool
            for pooled_conn in self.websocket_pool.active_connections.values():
                if pooled_conn.connection_id == connection_id:
                    await self.websocket_pool.release_connection(pooled_conn)

                    # Unregister from health monitor
                    self.health_monitor.unregister_connection(connection_id)

                    self.logger.debug(
                        "Released connection back to pool",
                        exchange=exchange,
                        connection_id=connection_id,
                    )
                    return

            self.logger.warning(
                "Connection not found in pool",
                exchange=exchange,
                connection_id=connection_id,
            )

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error("Failed to release connection", exchange=exchange, error=str(e))
            raise ExchangeConnectionError(f"Connection release failed: {e!s}")

    async def handle_connection_failure(self, exchange: str, connection: Any) -> None:
        """
        Handle connection failure and trigger recovery.

        Args:
            exchange: Exchange name
            connection: Failed WebSocket connection

        Raises:
            ExchangeConnectionError: If failure handling fails
            ValidationError: If parameters are invalid
        """
        try:
            # Validate parameters
            if not exchange or not connection:
                raise ValidationError("Exchange and connection are required")

            # Mark connection as failed in health monitor
            await self.health_monitor.mark_failed(connection)

            # Handle connection failure using error handler
            connection_id = getattr(connection, "id", str(id(connection)))
            await self._handle_connection_error(
                ExchangeConnectionError("Connection failure detected"),
                "handle_connection_failure",
                connection_id,
            )

            self.logger.warning(
                "Handled connection failure",
                exchange=exchange,
                connection_id=connection_id,
            )

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to handle connection failure", exchange=exchange, error=str(e)
            )
            raise ExchangeConnectionError(f"Connection failure handling failed: {e!s}")

    async def get_websocket_connection(
        self, connection_id: str = "default"
    ) -> WebSocketConnection | None:
        """
        Get an existing WebSocket connection.

        Args:
            connection_id: Connection identifier

        Returns:
            Optional[WebSocketConnection]: WebSocket connection or None if not found
        """
        return self.websocket_connections.get(connection_id)

    async def remove_websocket_connection(self, connection_id: str) -> bool:
        """
        Remove a WebSocket connection.

        Args:
            connection_id: Connection identifier

        Returns:
            bool: True if removed successfully, False otherwise
        """
        if connection_id in self.websocket_connections:
            connection = self.websocket_connections[connection_id]
            await connection.disconnect()
            del self.websocket_connections[connection_id]
            self.logger.info(f"Removed WebSocket connection {connection_id}")
            return True

        return False

    async def health_check_all(self) -> dict[str, bool]:
        """
        Perform health check on all connections.

        Returns:
            Dict[str, bool]: Dictionary mapping connection IDs to health status
        """
        health_status = {}

        # Check REST connections
        for endpoint, connection in self.rest_connections.items():
            try:
                # TODO: Remove in production - Implement actual REST health
                # check
                health_status[f"rest_{endpoint}"] = connection.get("connected", False)
            except Exception as e:
                self.logger.error(f"REST health check failed for {endpoint}: {e!s}")
                health_status[f"rest_{endpoint}"] = False

        # Check WebSocket connections
        for connection_id, connection in self.websocket_connections.items():
            try:
                health_status[f"ws_{connection_id}"] = connection.is_healthy()
            except Exception as e:
                self.logger.error(f"WebSocket health check failed for {connection_id}: {e!s}")
                health_status[f"ws_{connection_id}"] = False

        self.last_health_check = datetime.now()
        return health_status

    async def check_network_health(self) -> dict[str, Any]:
        """
        Check network connectivity and latency using utils helpers.

        Returns:
            Dict[str, Any]: Network health information
        """
        try:
            network_health = {}

            # Test connection to exchange endpoints
            for endpoint, _connection in self.rest_connections.items():
                try:
                    # Extract host and port from endpoint
                    if "://" in endpoint:
                        host = endpoint.split("://")[1].split("/")[0]
                        if ":" in host:
                            host, port_str = host.split(":")
                            port = int(port_str)
                        else:
                            port = 443 if "https" in endpoint else 80
                    else:
                        host = endpoint
                        port = 80

                    # Test connection using utils helpers
                    is_connected = test_connection(host, port, timeout=TIMEOUTS.get("short", 5.0))
                    latency = measure_latency(host, port, timeout=TIMEOUTS.get("short", 5.0))

                    network_health[f"network_{endpoint}"] = {
                        "connected": is_connected,
                        "latency_ms": latency,
                        "timestamp": datetime.now().isoformat(),
                    }

                except Exception as e:
                    self.logger.error(f"Network health check failed for {endpoint}: {e!s}")
                    network_health[f"network_{endpoint}"] = {
                        "connected": False,
                        "latency_ms": -1,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }

            return network_health

        except Exception as e:
            self.logger.error(f"Network health check failed: {e!s}")
            return {"error": str(e)}

    async def reconnect_all(self) -> dict[str, bool]:
        """
        Reconnect all unhealthy connections.

        Returns:
            Dict[str, bool]: Dictionary mapping connection IDs to reconnection success
        """
        reconnection_results = {}

        # Reconnect WebSocket connections
        for connection_id, connection in self.websocket_connections.items():
            if not connection.is_healthy():
                try:
                    await connection.disconnect()
                    success = await connection.connect()
                    reconnection_results[f"ws_{connection_id}"] = success

                    if success:
                        await connection.process_queued_messages()

                except Exception as e:
                    self.logger.error(f"Failed to reconnect WebSocket {connection_id}: {e!s}")
                    reconnection_results[f"ws_{connection_id}"] = False

        # TODO: Remove in production - Implement REST connection reconnection
        # For now, just mark REST connections as healthy
        for endpoint in self.rest_connections:
            reconnection_results[f"rest_{endpoint}"] = True

        return reconnection_results

    def get_connection_stats(self) -> dict[str, Any]:
        """
        Get connection statistics.

        Returns:
            Dict[str, Any]: Connection statistics
        """
        return {
            "exchange_name": self.exchange_name,
            "rest_connections": len(self.rest_connections),
            "websocket_connections": len(self.websocket_connections),
            "max_rest_connections": self.max_rest_connections,
            "max_websocket_connections": self.max_websocket_connections,
            "last_health_check": (
                self.last_health_check.isoformat() if self.last_health_check else None
            ),
        }

    async def disconnect_all(self) -> None:
        """Disconnect all connections."""
        # Disconnect WebSocket connections
        for connection_id, connection in self.websocket_connections.items():
            try:
                await connection.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting WebSocket {connection_id}: {e!s}")

        # Clear connection pools
        self.websocket_connections.clear()
        self.rest_connections.clear()

        self.logger.info(f"Disconnected all connections for {self.exchange_name}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect_all()
