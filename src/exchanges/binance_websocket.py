"""
Binance WebSocket Handler (P-004)

This module implements real-time data streaming for Binance exchange,
including ticker streams, order book depth streams, and user data streams.

CRITICAL: This integrates with P-001 (core types, exceptions, config), P-002A (error handling),
and P-003 (base exchange interface) components.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

# Binance-specific imports
from binance import BinanceSocketManager

from src.core.config import Config
from src.core.exceptions import ExchangeError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import OrderBook, OrderSide, Ticker, Trade

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler


class BinanceWebSocketHandler:
    """
    Binance WebSocket handler for real-time data streaming.

    Provides real-time data streams for:
    - Ticker price streams for all configured symbols
    - Order book depth streams with configurable levels
    - Trade execution stream for portfolio tracking
    - User data stream for account balance updates

    CRITICAL: This class handles all WebSocket connections and data processing.
    """

    def __init__(self, config: Config, client, exchange_name: str = "binance"):
        """
        Initialize Binance WebSocket handler.

        Args:
            config: Application configuration
            client: Binance client instance
            exchange_name: Exchange name (default: "binance")
        """
        self.config = config
        self.client = client
        self.exchange_name = exchange_name

        # Initialize WebSocket manager
        self.ws_manager = BinanceSocketManager(client)

        # Stream management
        self.active_streams: dict[str, Any] = {}
        self.callbacks: dict[str, list[Callable]] = {}
        self.stream_handlers: dict[str, asyncio.Task] = {}

        # Connection state
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.base_reconnect_delay = 1  # Base delay in seconds
        self.max_reconnect_delay = 60  # Max delay in seconds
        self._shutdown = False
        self._connection_lock = asyncio.Lock()

        # Health monitoring
        self.last_message_time = None
        self.message_timeout = 60  # seconds
        self._health_check_task: asyncio.Task | None = None

        # Connection metrics
        self._connection_start_time = None
        self._total_messages_received = 0
        self._total_reconnections = 0
        self._reconnect_task: asyncio.Task | None = None

        # Error handling
        self.error_handler = ErrorHandler(config.error_handling)

        # Keep track of listen key renewal task
        self.listen_key_task: asyncio.Task | None = None

        # Initialize logger
        self.logger = get_logger(f"binance.websocket.{exchange_name}")

        self.logger.info("Initialized Binance WebSocket handler")

    async def connect(self) -> bool:
        """
        Establish WebSocket connections with enhanced error handling.

        Returns:
            bool: True if connection successful, False otherwise
        """
        async with self._connection_lock:
            try:
                if self._shutdown:
                    return False

                self.logger.info("Connecting Binance WebSocket streams...")

                # Test connection by getting server time
                server_time = await self.client.get_server_time()
                self.logger.info(f"Binance server time: {server_time}")

                self.connected = True
                self.reconnect_attempts = 0
                self._connection_start_time = datetime.now(timezone.utc)
                self.last_message_time = datetime.now(timezone.utc)

                # Start health monitoring
                if not self._health_check_task or self._health_check_task.done():
                    self._health_check_task = asyncio.create_task(self._health_monitor())

                self.logger.info("Successfully connected Binance WebSocket streams")
                return True

            except Exception as e:
                self.logger.error(f"Failed to connect Binance WebSocket streams: {e!s}")
                self.connected = False
                await self._schedule_reconnect()
                return False

    async def disconnect(self) -> None:
        """Disconnect all WebSocket streams with proper cleanup."""
        async with self._connection_lock:
            try:
                self._shutdown = True
                self.logger.info("Disconnecting Binance WebSocket streams...")

                # Cancel health monitoring
                if self._health_check_task and not self._health_check_task.done():
                    self._health_check_task.cancel()
                    try:
                        await self._health_check_task
                    except asyncio.CancelledError:
                        pass

                # Cancel all stream handlers
                for task in list(self.stream_handlers.values()):
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                # Close all active streams
                for stream_name in list(self.active_streams.keys()):
                    await self._close_stream(stream_name)

                # Clear state
                self.stream_handlers.clear()
                self.callbacks.clear()

                self.connected = False
                self.logger.info("Successfully disconnected Binance WebSocket streams")

            except Exception as e:
                self.logger.error(f"Error disconnecting Binance WebSocket streams: {e!s}")

    async def subscribe_to_ticker_stream(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to ticker price stream for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            callback: Callback function to handle ticker data
        """
        try:
            stream_name = f"{symbol.lower()}@ticker"

            # Register callback
            if stream_name not in self.callbacks:
                self.callbacks[stream_name] = []
            self.callbacks[stream_name].append(callback)

            # Start stream if not already active
            if stream_name not in self.active_streams:
                stream = self.ws_manager.symbol_ticker_socket(symbol)
                self.active_streams[stream_name] = stream

                # Start async stream handler
                task = asyncio.create_task(self._handle_ticker_stream(stream_name, stream))
                self.stream_handlers[stream_name] = task

                self.logger.info(f"Subscribed to ticker stream: {stream_name}")

        except Exception as e:
            self.logger.error(f"Error subscribing to ticker stream: {e!s}")
            raise ExchangeError(f"Failed to subscribe to ticker stream: {e!s}")

    async def subscribe_to_orderbook_stream(
        self, symbol: str, depth: str = "20", callback: Callable | None = None
    ) -> None:
        """
        Subscribe to order book depth stream for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            depth: Order book depth ("5", "10", "20")
            callback: Callback function to handle order book data
        """
        try:
            stream_name = f"{symbol.lower()}@depth{depth}@100ms"

            # Register callback
            if stream_name not in self.callbacks:
                self.callbacks[stream_name] = []
            if callback:
                self.callbacks[stream_name].append(callback)

            # Start stream if not already active
            if stream_name not in self.active_streams:
                stream = self.ws_manager.depth_socket(symbol, depth=depth)
                self.active_streams[stream_name] = stream

                # Start async stream handler
                task = asyncio.create_task(self._handle_orderbook_stream(stream_name, stream))
                self.stream_handlers[stream_name] = task

                self.logger.info(f"Subscribed to orderbook stream: {stream_name}")

        except Exception as e:
            self.logger.error(f"Error subscribing to orderbook stream: {e!s}")
            raise ExchangeError(f"Failed to subscribe to orderbook stream: {e!s}")

    async def subscribe_to_trade_stream(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to trade execution stream for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            callback: Callback function to handle trade data
        """
        try:
            stream_name = f"{symbol.lower()}@trade"

            # Register callback
            if stream_name not in self.callbacks:
                self.callbacks[stream_name] = []
            self.callbacks[stream_name].append(callback)

            # Start stream if not already active
            if stream_name not in self.active_streams:
                stream = self.ws_manager.trade_socket(symbol)
                self.active_streams[stream_name] = stream

                # Start async stream handler
                task = asyncio.create_task(self._handle_trade_stream(stream_name, stream))
                self.stream_handlers[stream_name] = task

                self.logger.info(f"Subscribed to trade stream: {stream_name}")

        except Exception as e:
            self.logger.error(f"Error subscribing to trade stream: {e!s}")
            raise ExchangeError(f"Failed to subscribe to trade stream: {e!s}")

    async def subscribe_to_user_data_stream(self, callback: Callable) -> None:
        """
        Subscribe to user data stream for account updates.

        Args:
            callback: Callback function to handle user data updates
        """
        try:
            stream_name = "user_data"

            # Register callback
            if stream_name not in self.callbacks:
                self.callbacks[stream_name] = []
            self.callbacks[stream_name].append(callback)

            # Start stream if not already active
            if stream_name not in self.active_streams:
                stream = self.ws_manager.user_socket()
                self.active_streams[stream_name] = stream

                # Start async stream handler
                task = asyncio.create_task(self._handle_user_data_stream(stream_name, stream))
                self.stream_handlers[stream_name] = task

                self.logger.info(f"Subscribed to user data stream: {stream_name}")

        except Exception as e:
            self.logger.error(f"Error subscribing to user data stream: {e!s}")
            raise ExchangeError(f"Failed to subscribe to user data stream: {e!s}")

    async def unsubscribe_from_stream(self, stream_name: str) -> bool:
        """
        Unsubscribe from a specific stream.

        Args:
            stream_name: Name of the stream to unsubscribe from

        Returns:
            bool: True if unsubscribed successfully, False otherwise
        """
        try:
            if stream_name in self.active_streams:
                # Cancel stream handler
                if stream_name in self.stream_handlers:
                    self.stream_handlers[stream_name].cancel()
                    del self.stream_handlers[stream_name]

                # Close stream
                await self._close_stream(stream_name)

                self.logger.info(f"Unsubscribed from stream: {stream_name}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error unsubscribing from stream {stream_name}: {e!s}")
            return False

    # Stream handlers

    async def _handle_ticker_stream(self, stream_name: str, stream) -> None:
        """Handle ticker price stream data with enhanced error handling."""
        try:
            async with stream as stream_handler:
                async for msg in stream_handler:
                    if self._shutdown:
                        break

                    try:
                        # Update last message time
                        self.last_message_time = datetime.now(timezone.utc)
                        self._total_messages_received += 1

                        # Convert to Ticker format
                        ticker_data = self._convert_ticker_message(msg)

                        # Call registered callbacks
                        if stream_name in self.callbacks:
                            for callback in self.callbacks[stream_name]:
                                try:
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(ticker_data)
                                    else:
                                        callback(ticker_data)
                                except Exception as e:
                                    self.logger.error(f"Error in ticker callback: {e!s}")

                    except Exception as e:
                        self.logger.error(f"Error processing ticker message: {e!s}")

        except asyncio.CancelledError:
            self.logger.info(f"Ticker stream {stream_name} cancelled")
        except Exception as e:
            self.logger.error(f"Error handling ticker stream {stream_name}: {e!s}")
            await self._handle_stream_error(stream_name)

    async def _handle_orderbook_stream(self, stream_name: str, stream) -> None:
        """Handle order book depth stream data with enhanced error handling."""
        try:
            async with stream as stream_handler:
                async for msg in stream_handler:
                    if self._shutdown:
                        break

                    try:
                        # Update last message time
                        self.last_message_time = datetime.now(timezone.utc)
                        self._total_messages_received += 1

                        # Convert to OrderBook format
                        orderbook_data = self._convert_orderbook_message(msg)

                        # Call registered callbacks
                        if stream_name in self.callbacks:
                            for callback in self.callbacks[stream_name]:
                                try:
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(orderbook_data)
                                    else:
                                        callback(orderbook_data)
                                except Exception as e:
                                    self.logger.error(f"Error in orderbook callback: {e!s}")

                    except Exception as e:
                        self.logger.error(f"Error processing orderbook message: {e!s}")

        except asyncio.CancelledError:
            self.logger.info(f"Orderbook stream {stream_name} cancelled")
        except Exception as e:
            self.logger.error(f"Error handling orderbook stream {stream_name}: {e!s}")
            await self._handle_stream_error(stream_name)

    async def _handle_trade_stream(self, stream_name: str, stream) -> None:
        """Handle trade execution stream data with enhanced error handling."""
        try:
            async with stream as stream_handler:
                async for msg in stream_handler:
                    if self._shutdown:
                        break

                    try:
                        # Update last message time
                        self.last_message_time = datetime.now(timezone.utc)
                        self._total_messages_received += 1

                        # Convert to Trade format
                        trade_data = self._convert_trade_message(msg)

                        # Call registered callbacks
                        if stream_name in self.callbacks:
                            for callback in self.callbacks[stream_name]:
                                try:
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(trade_data)
                                    else:
                                        callback(trade_data)
                                except Exception as e:
                                    self.logger.error(f"Error in trade callback: {e!s}")

                    except Exception as e:
                        self.logger.error(f"Error processing trade message: {e!s}")

        except asyncio.CancelledError:
            self.logger.info(f"Trade stream {stream_name} cancelled")
        except Exception as e:
            self.logger.error(f"Error handling trade stream {stream_name}: {e!s}")
            await self._handle_stream_error(stream_name)

    async def _handle_user_data_stream(self, stream_name: str, stream) -> None:
        """Handle user data stream for account updates with enhanced error handling."""
        try:
            async with stream as stream_handler:
                async for msg in stream_handler:
                    if self._shutdown:
                        break

                    try:
                        # Update last message time
                        self.last_message_time = datetime.now(timezone.utc)
                        self._total_messages_received += 1

                        # Process different types of user data updates
                        if msg.get("e") == "outboundAccountPosition":
                            # Account balance update
                            await self._handle_account_position_update(msg)
                        elif msg.get("e") == "executionReport":
                            # Order execution update
                            await self._handle_execution_report(msg)
                        elif msg.get("e") == "balanceUpdate":
                            # Balance update
                            await self._handle_balance_update(msg)

                        # Call registered callbacks
                        if stream_name in self.callbacks:
                            for callback in self.callbacks[stream_name]:
                                try:
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(msg)
                                    else:
                                        callback(msg)
                                except Exception as e:
                                    self.logger.error(f"Error in user data callback: {e!s}")

                    except Exception as e:
                        self.logger.error(f"Error processing user data message: {e!s}")

        except asyncio.CancelledError:
            self.logger.info(f"User data stream {stream_name} cancelled")
        except Exception as e:
            self.logger.error(f"Error handling user data stream {stream_name}: {e!s}")
            await self._handle_stream_error(stream_name)

    # Message conversion methods

    def _convert_ticker_message(self, msg: dict) -> Ticker:
        """Convert Binance ticker message to Ticker format."""
        return Ticker(
            symbol=msg["s"],
            bid=Decimal(str(msg["b"])),
            ask=Decimal(str(msg["a"])),
            last_price=Decimal(str(msg["c"])),
            volume_24h=Decimal(str(msg["v"])),
            price_change_24h=Decimal(str(msg["p"])),
            timestamp=datetime.fromtimestamp(msg["E"] / 1000, tz=timezone.utc),
        )

    def _convert_orderbook_message(self, msg: dict) -> OrderBook:
        """Convert Binance order book message to OrderBook format."""
        bids = [[Decimal(str(price)), Decimal(str(qty))] for price, qty in msg["b"]]
        asks = [[Decimal(str(price)), Decimal(str(qty))] for price, qty in msg["a"]]

        return OrderBook(
            symbol=msg["s"],
            bids=bids,
            asks=asks,
            timestamp=datetime.fromtimestamp(msg["E"] / 1000, tz=timezone.utc),
        )

    def _convert_trade_message(self, msg: dict) -> Trade:
        """Convert Binance trade message to Trade format."""
        return Trade(
            id=str(msg["t"]),
            symbol=msg["s"],
            # m=True means maker is seller
            side=OrderSide.BUY if msg["m"] else OrderSide.SELL,
            amount=Decimal(str(msg["q"])),
            price=Decimal(str(msg["p"])),
            timestamp=datetime.fromtimestamp(msg["T"] / 1000, tz=timezone.utc),
            fee=Decimal("0"),  # Fee not available in trade stream
        )

    # User data handlers

    async def _handle_account_position_update(self, msg: dict) -> None:
        """Handle account position update message."""
        try:
            balances = {}
            for balance in msg["B"]:
                asset = balance["a"]
                free = Decimal(balance["f"])
                locked = Decimal(balance["l"])
                total = free + locked

                if total > 0:
                    balances[asset] = total

            self.logger.debug(f"Account position updated: {len(balances)} assets")

        except Exception as e:
            self.logger.error(f"Error handling account position update: {e!s}")

    async def _handle_execution_report(self, msg: dict) -> None:
        """Handle order execution report message."""
        try:
            order_id = str(msg["i"])
            status = msg["X"]
            executed_qty = Decimal(str(msg["z"]))

            self.logger.debug(
                f"Order execution: {order_id}, status: {status}, executed: {executed_qty}"
            )

        except Exception as e:
            self.logger.error(f"Error handling execution report: {e!s}")

    async def _handle_balance_update(self, msg: dict) -> None:
        """Handle balance update message."""
        try:
            asset = msg["a"]
            delta = Decimal(str(msg["d"]))

            self.logger.debug(f"Balance update: {asset}, delta: {delta}")

        except Exception as e:
            self.logger.error(f"Error handling balance update: {e!s}")

    # Utility methods

    async def _close_stream(self, stream_name: str) -> None:
        """Close a WebSocket stream."""
        try:
            if stream_name in self.active_streams:
                stream = self.active_streams[stream_name]
                await stream.close()
                del self.active_streams[stream_name]

                if stream_name in self.callbacks:
                    del self.callbacks[stream_name]

                self.logger.info(f"Closed stream: {stream_name}")
        except Exception as e:
            self.logger.error(f"Error closing stream {stream_name}: {e!s}")

    async def _handle_stream_error(self, stream_name: str) -> None:
        """Handle stream error with enhanced reconnection logic."""
        try:
            self.logger.warning(f"Stream {stream_name} encountered an error")

            # Mark as disconnected
            self.connected = False
            self._total_reconnections += 1

            # Schedule reconnection
            await self._schedule_reconnect()

        except Exception as e:
            self.logger.error(f"Error handling stream error: {e!s}")

    async def _schedule_reconnect(self) -> None:
        """Schedule reconnection with exponential backoff."""
        if self._shutdown or self.reconnect_attempts >= self.max_reconnect_attempts:
            if self.reconnect_attempts >= self.max_reconnect_attempts:
                self.logger.error(
                    f"Max reconnection attempts ({self.max_reconnect_attempts}) reached"
                )
            return

        self.reconnect_attempts += 1

        # Calculate exponential backoff delay
        delay = min(
            self.base_reconnect_delay * (2 ** (self.reconnect_attempts - 1)),
            self.max_reconnect_delay,
        )

        self.logger.info(
            f"Scheduling reconnection in {delay:.1f}s "
            f"(attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})"
        )

        # Schedule reconnection
        task = asyncio.create_task(self._reconnect_after_delay(delay))
        # Store task to prevent garbage collection
        self._reconnect_task = task

    async def _reconnect_after_delay(self, delay: float) -> None:
        """Reconnect after specified delay."""
        try:
            await asyncio.sleep(delay)

            if self._shutdown:
                return

            self.logger.info("Attempting to reconnect...")

            # Store current subscriptions for resubscription
            subscriptions_to_restore = {
                stream_name: callbacks for stream_name, callbacks in self.callbacks.items()
            }

            # Attempt reconnection
            if await self.connect():
                self.logger.info("Reconnection successful, restoring subscriptions...")

                # Restore subscriptions
                for stream_name, callbacks in subscriptions_to_restore.items():
                    try:
                        # Parse stream name to determine subscription type
                        if "@ticker" in stream_name:
                            symbol = stream_name.split("@")[0].upper()
                            for callback in callbacks:
                                await self.subscribe_to_ticker_stream(symbol, callback)
                        elif "@depth" in stream_name:
                            symbol = stream_name.split("@")[0].upper()
                            depth = "20"  # Default depth
                            for callback in callbacks:
                                await self.subscribe_to_orderbook_stream(symbol, depth, callback)
                        elif "@trade" in stream_name:
                            symbol = stream_name.split("@")[0].upper()
                            for callback in callbacks:
                                await self.subscribe_to_trade_stream(symbol, callback)
                        elif stream_name == "user_data":
                            for callback in callbacks:
                                await self.subscribe_to_user_data_stream(callback)

                    except Exception as e:
                        self.logger.error(f"Failed to restore subscription {stream_name}: {e!s}")

                self.logger.info("Subscription restoration completed")
            else:
                self.logger.error("Reconnection failed")

        except Exception as e:
            self.logger.error(f"Error during reconnection: {e!s}")

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
                        self.logger.warning(
                            f"No messages received for {time_since_last_message:.1f}s, "
                            "triggering health check reconnection"
                        )
                        self.connected = False
                        await self._schedule_reconnect()
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e!s}")

    def get_active_streams(self) -> list[str]:
        """Get list of active stream names."""
        return list(self.active_streams.keys())

    def is_connected(self) -> bool:
        """Check if WebSocket handler is connected and healthy."""
        return self.connected and not self._shutdown

    async def health_check(self) -> bool:
        """
        Perform comprehensive health check on WebSocket connections.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            if self._shutdown:
                return False

            if not self.connected:
                self.logger.debug("Health check failed: not connected")
                return False

            # Check if we've received messages recently
            if self.last_message_time:
                time_since_last_message = (
                    datetime.now(timezone.utc) - self.last_message_time
                ).total_seconds()

                if time_since_last_message > self.message_timeout:
                    self.logger.warning(
                        f"Health check failed: no messages for {time_since_last_message:.1f}s"
                    )
                    return False

            # Check if stream handlers are running
            failed_streams = []
            for stream_name, task in self.stream_handlers.items():
                if task.done():
                    failed_streams.append(stream_name)

            if failed_streams:
                self.logger.warning(f"Health check failed: inactive streams: {failed_streams}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"WebSocket health check failed: {e!s}")
            return False

    def get_connection_metrics(self) -> dict[str, Any]:
        """
        Get comprehensive connection metrics.

        Returns:
            Dict containing connection metrics
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
                "uptime_seconds": uptime,
                "total_messages_received": self._total_messages_received,
                "total_reconnections": self._total_reconnections,
                "reconnect_attempts": self.reconnect_attempts,
                "max_reconnect_attempts": self.max_reconnect_attempts,
                "active_streams": len(self.active_streams),
                "stream_handlers": len(self.stream_handlers),
                "time_since_last_message": time_since_last_message,
                "message_timeout": self.message_timeout,
                "health_status": "healthy" if self.health_check() else "unhealthy",
                "stream_names": list(self.active_streams.keys()),
                "shutdown": self._shutdown,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get connection metrics: {e!s}")
            return {"error": str(e)}

    async def get_stream_health(self) -> dict[str, dict[str, Any]]:
        """
        Get health status of individual streams.

        Returns:
            Dict mapping stream names to their health status
        """
        stream_health = {}

        try:
            for stream_name in self.active_streams.keys():
                task = self.stream_handlers.get(stream_name)
                stream_health[stream_name] = {
                    "active": stream_name in self.active_streams,
                    "handler_running": task is not None and not task.done(),
                    "callback_count": len(self.callbacks.get(stream_name, [])),
                    "task_done": task.done() if task else True,
                    "task_exception": (
                        str(task.exception()) if task and task.done() and task.exception() else None
                    ),
                }

        except Exception as e:
            self.logger.error(f"Failed to get stream health: {e!s}")

        return stream_health
