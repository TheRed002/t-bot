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
from src.core.types.market import OrderBook, OrderBookLevel, Ticker, Trade
from src.core.types.trading import OrderSide

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler
from src.utils.websocket_manager_utils import ExchangeWebSocketReconnectionManager


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

    def __init__(self, config: Config, client, exchange_name: str = "binance", error_handler: ErrorHandler | None = None):
        """
        Initialize Binance WebSocket handler.

        Args:
            config: Application configuration
            client: Binance client instance
            exchange_name: Exchange name (default: "binance")
            error_handler: Error handler service (injected)
        """
        self.config = config
        self.client = client
        self.exchange_name = exchange_name

        # Store injected error handler
        self._injected_error_handler = error_handler

        # Initialize WebSocket manager
        self.ws_manager = BinanceSocketManager(client)

        # Stream management
        self.active_streams: dict[str, Any] = {}
        self.callbacks: dict[str, list[Callable]] = {}
        self.stream_handlers: dict[str, asyncio.Task] = {}

        # Get WebSocket configuration
        self.ws_config = getattr(config, "websocket", {})
        if hasattr(config, "exchange") and hasattr(config.exchange, "get_websocket_config"):
            self.ws_config = config.exchange.get_websocket_config(exchange_name)

        # Connection state
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = self.ws_config.get("reconnect_attempts", 10)
        self.base_reconnect_delay = self.ws_config.get("reconnect_delay", 1.0)
        self.max_reconnect_delay = self.ws_config.get("max_reconnect_delay", 60.0)
        self._shutdown = False
        self._connection_lock = asyncio.Lock()

        # Health monitoring
        self.last_message_time = None
        self.message_timeout = self.ws_config.get("message_timeout", 60)
        self._health_check_task: asyncio.Task | None = None

        # Connection metrics
        self._connection_start_time = None
        self._total_messages_received = 0
        self._total_reconnections = 0
        self._reconnect_task: asyncio.Task | None = None

        # Error handling - use injected handler if available
        self.error_handler = self._injected_error_handler or ErrorHandler(config)

        # Reconnection management using shared utility
        self.reconnection_manager = ExchangeWebSocketReconnectionManager(
            exchange_name, self.max_reconnect_attempts
        )

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
                try:
                    if hasattr(self.client, "get_server_time"):
                        if asyncio.iscoroutinefunction(self.client.get_server_time):
                            server_time = await self.client.get_server_time()
                        else:
                            server_time = self.client.get_server_time()
                        self.logger.info(f"Binance server time: {server_time}")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to get server time: {e}, proceeding with connection"
                    )

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
            health_task = None
            stream_tasks = []
            stream_names = []

            try:
                self._shutdown = True
                self.logger.info("Disconnecting Binance WebSocket streams...")

                # Store references before modifying
                health_task = self._health_check_task
                stream_tasks = list(self.stream_handlers.values())
                stream_names = list(self.active_streams.keys())
            except Exception as e:
                self.logger.error(f"Error preparing disconnect: {e!s}")

            # Cancel health monitoring
            try:
                if health_task and not health_task.done():
                    health_task.cancel()
                    try:
                        await health_task
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        self.logger.error(f"Error canceling health task: {e!s}")
            except Exception as e:
                self.logger.error(f"Error with health monitoring cleanup: {e!s}")
            finally:
                # Always clear task reference regardless of cancellation success
                self._health_check_task = None

            # Cancel all stream handlers with proper resource cleanup
            try:
                for task in stream_tasks:
                    if task and not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            self.logger.error(f"Error canceling stream task: {e!s}")
            except Exception as e:
                self.logger.error(f"Error canceling stream handlers: {e!s}")
            finally:
                # Clear all stream handler references
                try:
                    self.stream_handlers.clear()
                except Exception as e:
                    self.logger.error(f"Error clearing stream handlers: {e}")

            # Close all active streams
            try:
                for stream_name in stream_names:
                    try:
                        await self._close_stream(stream_name)
                    except Exception as e:
                        self.logger.error(f"Error closing stream {stream_name}: {e!s}")
            except Exception as e:
                self.logger.error(f"Error closing streams: {e!s}")
            finally:
                # Clear remaining state references
                try:
                    self.callbacks.clear()
                    self.active_streams.clear()
                except Exception as e:
                    self.logger.warning(f"Error clearing stream data: {e}")

                self.connected = False
                self.logger.info("Successfully disconnected Binance WebSocket streams")

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
                    task = self.stream_handlers[stream_name]
                    if task and not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            self.logger.error(f"Error waiting for stream task cancellation: {e}")
                    del self.stream_handlers[stream_name]

                # Close stream with proper error handling
                try:
                    await self._close_stream(stream_name)
                except Exception as e:
                    self.logger.error(f"Error closing stream {stream_name}: {e}")

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

                        # Call registered callbacks with proper concurrency
                        if stream_name in self.callbacks:
                            callback_tasks = []
                            for callback in self.callbacks[stream_name]:
                                try:
                                    if asyncio.iscoroutinefunction(callback):
                                        callback_tasks.append(callback(ticker_data))
                                    else:
                                        # Run sync callback in executor to avoid blocking
                                        callback_tasks.append(
                                            asyncio.get_event_loop().run_in_executor(
                                                None, callback, ticker_data
                                            )
                                        )
                                except Exception as e:
                                    self.logger.error(f"Error preparing ticker callback: {e!s}")

                            # Execute all callbacks concurrently
                            if callback_tasks:
                                await asyncio.gather(*callback_tasks, return_exceptions=True)

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
            bid_price=Decimal(str(msg["b"])),
            bid_quantity=Decimal(str(msg.get("B", "0"))),  # Binance uses "B" for bid quantity
            ask_price=Decimal(str(msg["a"])),
            ask_quantity=Decimal(str(msg.get("A", "0"))),  # Binance uses "A" for ask quantity
            last_price=Decimal(str(msg["c"])),
            last_quantity=Decimal(str(msg.get("Q", "0"))) if msg.get("Q") else None,
            open_price=Decimal(str(msg.get("o", msg["c"]))),  # Use "o" or fallback to close
            high_price=Decimal(str(msg.get("h", msg["c"]))),  # Use "h" or fallback to close
            low_price=Decimal(str(msg.get("l", msg["c"]))),  # Use "l" or fallback to close
            volume=Decimal(str(msg["v"])),
            quote_volume=Decimal(str(msg.get("q", "0"))) if msg.get("q") else None,
            timestamp=datetime.fromtimestamp(msg["E"] / 1000, tz=timezone.utc),
            exchange="binance",
            price_change=Decimal(str(msg["p"])) if msg.get("p") else None,
            price_change_percent=Decimal(str(msg.get("P", "0"))) if msg.get("P") else None,
        )

    def _convert_orderbook_message(self, msg: dict) -> OrderBook:
        """Convert Binance order book message to OrderBook format."""
        bids = [
            OrderBookLevel(price=Decimal(str(price)), quantity=Decimal(str(qty)))
            for price, qty in msg["b"]
        ]
        asks = [
            OrderBookLevel(price=Decimal(str(price)), quantity=Decimal(str(qty)))
            for price, qty in msg["a"]
        ]

        return OrderBook(
            symbol=msg["s"],
            bids=bids,
            asks=asks,
            timestamp=datetime.fromtimestamp(msg["E"] / 1000, tz=timezone.utc),
            exchange="binance",
        )

    def _convert_trade_message(self, msg: dict) -> Trade:
        """Convert Binance trade message to Trade format."""
        return Trade(
            id=str(msg["t"]),
            symbol=msg["s"],
            exchange="binance",
            side=OrderSide.BUY.value if not msg["m"] else OrderSide.SELL.value,
            price=Decimal(str(msg["p"])),
            quantity=Decimal(str(msg["q"])),
            timestamp=datetime.fromtimestamp(msg["T"] / 1000, tz=timezone.utc),
            maker=msg["m"],
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
        """Close a WebSocket stream with proper resource cleanup."""
        stream = None
        try:
            stream = self.active_streams.get(stream_name)
            if stream:
                # Ensure proper async context manager closing
                if hasattr(stream, "__aexit__"):
                    try:
                        await stream.__aexit__(None, None, None)
                    except Exception as e:
                        self.logger.debug(f"Stream context manager exit error: {e}")
                elif hasattr(stream, "close"):
                    try:
                        await stream.close()
                    except Exception as e:
                        self.logger.debug(f"Stream close error: {e}")
        except Exception as e:
            self.logger.error(f"Error closing stream {stream_name}: {e!s}")
        finally:
            # Clean up references regardless of close success - use finally to ensure cleanup
            try:
                if stream_name in self.active_streams:
                    del self.active_streams[stream_name]
            except Exception as e:
                self.logger.warning(f"Error removing stream {stream_name} from active_streams: {e}")

            try:
                if stream_name in self.callbacks:
                    del self.callbacks[stream_name]
            except Exception as e:
                self.logger.warning(f"Error removing callbacks for stream {stream_name}: {e}")

            try:
                self.logger.info(f"Closed stream: {stream_name}")
            except Exception:
                # Even logging can fail in edge cases
                pass

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
                await asyncio.sleep(
                    self.ws_config.get("health_check_interval", 30.0)
                )  # Health check interval

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

    async def get_connection_metrics(self) -> dict[str, Any]:
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
                "health_status": "healthy" if await self.health_check() else "unhealthy",
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
