"""
Coinbase WebSocket Handler (P-006)

This module implements the Coinbase-specific WebSocket client for real-time data streaming,
including ticker updates, order book changes, and trade notifications.

CRITICAL: This integrates with P-001 (core types, exceptions, config), P-002A (error handling),
and P-003 (base exchange interface) components.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

# WebSocket imports
import websockets
from websockets.client import WebSocketClientProtocol

from src.core.config import Config
from src.core.exceptions import ExchangeConnectionError, ExchangeError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import OrderBook, OrderSide, Ticker, Trade

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler


class CoinbaseWebSocketHandler:
    """
    Coinbase WebSocket handler for real-time data streaming.

    Provides real-time market data, order updates, and account notifications
    through Coinbase's WebSocket API.
    """

    def __init__(self, config: Config, exchange_name: str = "coinbase"):
        """
        Initialize Coinbase WebSocket handler.

        Args:
            config: Application configuration
            exchange_name: Exchange name (default: "coinbase")
        """
        self.config = config
        self.exchange_name = exchange_name
        self.error_handler = ErrorHandler(config)

        # Coinbase-specific configuration
        self.api_key = config.exchange.coinbase_api_key
        self.api_secret = config.exchange.coinbase_api_secret
        self.passphrase = getattr(config.exchanges, "coinbase_passphrase", "")
        self.sandbox = config.exchange.coinbase_sandbox

        # WebSocket URLs
        if self.sandbox:
            self.ws_url = "wss://ws-feed-public.sandbox.exchange.coinbase.com"
        else:
            self.ws_url = "wss://ws-feed.exchange.coinbase.com"

        # WebSocket connection
        self.ws: WebSocketClientProtocol | None = None
        self._connection_lock = asyncio.Lock()

        # Stream management
        self.active_streams: dict[str, Any] = {}
        self.callbacks: dict[str, list[Callable]] = {}
        self.connected = False
        self.last_heartbeat = None

        # Connection state
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.base_reconnect_delay = 1  # Base delay in seconds
        self.max_reconnect_delay = 60  # Max delay in seconds
        self._shutdown = False

        # Health monitoring
        self.last_message_time = None
        self.message_timeout = 60  # seconds
        self._health_check_task: asyncio.Task | None = None

        # Connection metrics
        self._connection_start_time = None
        self._total_messages_received = 0
        self._total_reconnections = 0

        # Message queue for reconnection
        self.message_queue: list[dict] = []
        self.max_queue_size = 1000

        # Subscribed channels for reconnection
        self.subscribed_channels: set[str] = set()
        self._listener_task: asyncio.Task | None = None

        # Initialize logger
        self.logger = get_logger(f"coinbase.websocket.{exchange_name}")

        self.logger.info(f"Initialized {exchange_name} WebSocket handler")

    async def connect(self) -> bool:
        """
        Establish WebSocket connection to Coinbase with enhanced error handling.

        Returns:
            bool: True if connection successful, False otherwise
        """
        async with self._connection_lock:
            try:
                if self._shutdown:
                    return False

                self.logger.info(f"Connecting to {self.exchange_name} WebSocket: {self.ws_url}")

                # Connect to WebSocket
                self.ws = await websockets.connect(
                    self.ws_url, ping_interval=20, ping_timeout=10, close_timeout=10
                )

                self.connected = True
                self.reconnect_attempts = 0
                self._connection_start_time = datetime.now(timezone.utc)
                self.last_heartbeat = datetime.now(timezone.utc)
                self.last_message_time = datetime.now(timezone.utc)

                # Start message listener
                listener_task = asyncio.create_task(self._listen_messages())
                # Store task to prevent garbage collection
                self._listener_task = listener_task

                # Start health monitoring
                if not self._health_check_task or self._health_check_task.done():
                    self._health_check_task = asyncio.create_task(self._health_monitor())

                self.logger.info(f"Successfully connected to {self.exchange_name} WebSocket")
                return True

            except Exception as e:
                self.logger.error(f"Failed to connect to {self.exchange_name} WebSocket: {e!s}")
                self.connected = False
                await self._schedule_reconnect()
                return False

    async def disconnect(self) -> None:
        """Disconnect from Coinbase WebSocket with proper cleanup."""
        async with self._connection_lock:
            try:
                self._shutdown = True
                self.logger.info(f"Disconnecting from {self.exchange_name} WebSocket")

                # Cancel health monitoring
                if self._health_check_task and not self._health_check_task.done():
                    self._health_check_task.cancel()
                    try:
                        await self._health_check_task
                    except asyncio.CancelledError:
                        pass

                # Close WebSocket connection
                if self.ws and not self.ws.closed:
                    await self.ws.close()
                    self.ws = None

                # Clear state
                self.connected = False
                self.active_streams.clear()
                self.callbacks.clear()
                self.subscribed_channels.clear()

                self.logger.info(f"Successfully disconnected from {self.exchange_name} WebSocket")

            except Exception as e:
                self.logger.error(f"Error disconnecting from {self.exchange_name} WebSocket: {e!s}")

    async def subscribe_to_ticker(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to ticker updates for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            callback: Callback function to handle ticker updates
        """
        try:
            if not self.ws or self.ws.closed:
                raise ExchangeConnectionError("WebSocket not connected")

            # Create subscription message
            subscribe_msg = {"type": "subscribe", "product_ids": [symbol], "channels": ["ticker"]}

            # Add authentication if available
            if self.api_key and self.api_secret:
                subscribe_msg.update(self._create_auth_headers())

            # Send subscription
            await self.ws.send(json.dumps(subscribe_msg))

            # Track subscription
            channel_key = f"ticker_{symbol}"
            self.subscribed_channels.add(channel_key)

            if channel_key not in self.callbacks:
                self.callbacks[channel_key] = []
            self.callbacks[channel_key].append(callback)

            self.logger.info(f"Subscribed to ticker stream for {symbol}")

        except Exception as e:
            self.logger.error(f"Failed to subscribe to ticker for {symbol}: {e!s}")
            raise ExchangeError(f"Failed to subscribe to ticker: {e!s}")

    async def subscribe_to_orderbook(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to order book updates for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            callback: Callback function to handle order book updates
        """
        try:
            if not self.ws or self.ws.closed:
                raise ExchangeConnectionError("WebSocket not connected")

            # Create subscription message
            subscribe_msg = {"type": "subscribe", "product_ids": [symbol], "channels": ["level2"]}

            # Add authentication if available
            if self.api_key and self.api_secret:
                subscribe_msg.update(self._create_auth_headers())

            # Send subscription
            await self.ws.send(json.dumps(subscribe_msg))

            # Track subscription
            channel_key = f"level2_{symbol}"
            self.subscribed_channels.add(channel_key)

            if channel_key not in self.callbacks:
                self.callbacks[channel_key] = []
            self.callbacks[channel_key].append(callback)

            self.logger.info(f"Subscribed to order book stream for {symbol}")

        except Exception as e:
            self.logger.error(f"Failed to subscribe to order book for {symbol}: {e!s}")
            raise ExchangeError(f"Failed to subscribe to order book: {e!s}")

    async def subscribe_to_trades(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to trade updates for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            callback: Callback function to handle trade updates
        """
        try:
            if not self.ws or self.ws.closed:
                raise ExchangeConnectionError("WebSocket not connected")

            # Create subscription message
            subscribe_msg = {"type": "subscribe", "product_ids": [symbol], "channels": ["matches"]}

            # Add authentication if available
            if self.api_key and self.api_secret:
                subscribe_msg.update(self._create_auth_headers())

            # Send subscription
            await self.ws.send(json.dumps(subscribe_msg))

            # Track subscription
            channel_key = f"matches_{symbol}"
            self.subscribed_channels.add(channel_key)

            if channel_key not in self.callbacks:
                self.callbacks[channel_key] = []
            self.callbacks[channel_key].append(callback)

            self.logger.info(f"Subscribed to trades stream for {symbol}")

        except Exception as e:
            self.logger.error(f"Failed to subscribe to trades for {symbol}: {e!s}")
            raise ExchangeError(f"Failed to subscribe to trades: {e!s}")

    async def subscribe_to_user_data(self, callback: Callable) -> None:
        """
        Subscribe to user data updates (orders, balances).

        Args:
            callback: Callback function to handle user data updates
        """
        try:
            if not self.ws or self.ws.closed:
                raise ExchangeConnectionError("WebSocket not connected")

            if not (self.api_key and self.api_secret):
                raise ExchangeError("API credentials required for user data subscription")

            # Create subscription message with authentication
            subscribe_msg = {"type": "subscribe", "channels": ["user"]}
            subscribe_msg.update(self._create_auth_headers())

            # Send subscription
            await self.ws.send(json.dumps(subscribe_msg))

            # Track subscription
            channel_key = "user"
            self.subscribed_channels.add(channel_key)

            if channel_key not in self.callbacks:
                self.callbacks[channel_key] = []
            self.callbacks[channel_key].append(callback)

            self.logger.info("Subscribed to user data stream")

        except Exception as e:
            self.logger.error(f"Failed to subscribe to user data: {e!s}")
            raise ExchangeError(f"Failed to subscribe to user data: {e!s}")

    async def unsubscribe_from_stream(self, stream_key: str) -> bool:
        """
        Unsubscribe from a specific stream.

        Args:
            stream_key: Key of the stream to unsubscribe from

        Returns:
            bool: True if unsubscribed successfully, False otherwise
        """
        try:
            if stream_key not in self.active_streams:
                self.logger.warning(f"Stream {stream_key} not found in active streams")
                return False

            stream_info = self.active_streams[stream_key]

            # Build unsubscribe message based on stream type
            unsubscribe_msg = {"type": "unsubscribe", "channels": []}

            if stream_info["type"] == "ticker":
                unsubscribe_msg["channels"].append(
                    {"name": "ticker", "product_ids": [stream_info["symbol"]]}
                )
            elif stream_info["type"] == "orderbook":
                unsubscribe_msg["channels"].append(
                    {"name": "level2", "product_ids": [stream_info["symbol"]]}
                )
            elif stream_info["type"] == "trades":
                unsubscribe_msg["channels"].append(
                    {"name": "matches", "product_ids": [stream_info["symbol"]]}
                )
            elif stream_info["type"] == "user_data":
                unsubscribe_msg["channels"].append({"name": "user", "product_ids": ["*"]})

            # Send unsubscribe message
            await self.ws.send(json.dumps(unsubscribe_msg))

            # Remove from tracking
            del self.active_streams[stream_key]

            self.logger.info(f"Unsubscribed from {stream_key}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from {stream_key}: {e!s}")
            return False

    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all active streams."""
        try:
            for stream_key in list(self.active_streams.keys()):
                await self.unsubscribe_from_stream(stream_key)

            self.active_streams.clear()
            self.callbacks.clear()

            self.logger.info("Unsubscribed from all streams")

        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from all streams: {e!s}")

    async def handle_ticker_message(self, message: dict) -> None:
        """
        Handle ticker message from WebSocket.

        Args:
            message: Ticker message from Coinbase
        """
        try:
            # Convert to unified Ticker format
            ticker = Ticker(
                symbol=message.get("product_id", ""),
                bid=Decimal(str(message.get("bid", "0"))),
                ask=Decimal(str(message.get("ask", "0"))),
                last_price=Decimal(str(message.get("price", "0"))),
                volume_24h=Decimal(str(message.get("volume_24h", "0"))),
                price_change_24h=Decimal(str(message.get("price_change_24h", "0"))),
                timestamp=datetime.fromisoformat(message.get("time", "").replace("Z", "+00:00")),
            )

            # Call registered callbacks
            symbol = message.get("product_id", "")
            if symbol in self.callbacks:
                for callback in self.callbacks[symbol]:
                    try:
                        await callback(ticker)
                    except Exception as e:
                        self.logger.error(f"Error in ticker callback: {e!s}")

        except Exception as e:
            self.logger.error(f"Error handling ticker message: {e!s}")

    async def handle_orderbook_message(self, message: dict) -> None:
        """
        Handle order book message from WebSocket.

        Args:
            message: Order book message from Coinbase
        """
        try:
            # Convert to unified OrderBook format
            order_book = OrderBook(
                symbol=message.get("product_id", ""),
                bids=[
                    [Decimal(str(level[0])), Decimal(str(level[1]))]
                    for level in message.get("bids", [])
                ],
                asks=[
                    [Decimal(str(level[0])), Decimal(str(level[1]))]
                    for level in message.get("asks", [])
                ],
                timestamp=datetime.now(timezone.utc),
            )

            # Call registered callbacks
            symbol = message.get("product_id", "")
            if symbol in self.callbacks:
                for callback in self.callbacks[symbol]:
                    try:
                        await callback(order_book)
                    except Exception as e:
                        self.logger.error(f"Error in order book callback: {e!s}")

        except Exception as e:
            self.logger.error(f"Error handling order book message: {e!s}")

    async def handle_trade_message(self, message: dict) -> None:
        """
        Handle trade message from WebSocket.

        Args:
            message: Trade message from Coinbase
        """
        try:
            # Convert to unified Trade format
            trade = Trade(
                id=message.get("trade_id", ""),
                symbol=message.get("product_id", ""),
                # Convert to lowercase for consistency
                side=message.get("side", "buy").lower(),
                amount=Decimal(str(message.get("size", "0"))),
                price=Decimal(str(message.get("price", "0"))),
                timestamp=datetime.fromisoformat(message.get("time", "").replace("Z", "+00:00")),
                fee=Decimal("0"),  # Coinbase doesn't provide fee in trade data
                fee_currency="USD",
            )

            # Call registered callbacks
            symbol = message.get("product_id", "")
            if symbol in self.callbacks:
                for callback in self.callbacks[symbol]:
                    try:
                        await callback(trade)
                    except Exception as e:
                        self.logger.error(f"Error in trade callback: {e!s}")

        except Exception as e:
            self.logger.error(f"Error handling trade message: {e!s}")

    async def handle_user_message(self, message: dict) -> None:
        """
        Handle user data message from WebSocket.

        Args:
            message: User data message from Coinbase
        """
        try:
            # Handle different types of user messages
            message_type = message.get("channel", {}).get("name", "")

            if message_type == "orders":
                # Handle order updates
                await self._handle_order_update(message)
            elif message_type == "accounts":
                # Handle account updates
                await self._handle_account_update(message)

            # Call registered callbacks
            if "user_data" in self.callbacks:
                for callback in self.callbacks["user_data"]:
                    try:
                        await callback(message)
                    except Exception as e:
                        self.logger.error(f"Error in user data callback: {e!s}")

        except Exception as e:
            self.logger.error(f"Error handling user message: {e!s}")

    async def _handle_order_update(self, message: dict) -> None:
        """Handle order update message."""
        try:
            order_id = message.get("order_id", "")
            status = message.get("status", "")

            self.logger.info(f"Order {order_id} status updated to {status}")

        except Exception as e:
            self.logger.error(f"Error handling order update: {e!s}")

    async def _handle_account_update(self, message: dict) -> None:
        """Handle account update message."""
        try:
            account_id = message.get("account_id", "")
            _ = message.get("balance", {})

            self.logger.info(f"Account {account_id} balance updated")

        except Exception as e:
            self.logger.error(f"Error handling account update: {e!s}")

    async def health_check(self) -> bool:
        """
        Perform comprehensive health check on WebSocket connection.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            if self._shutdown:
                return False

            if not self.connected or not self.ws or self.ws.closed:
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

            return True

        except Exception as e:
            self.logger.warning(f"WebSocket health check failed: {e!s}")
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
                "subscribed_channels": len(self.subscribed_channels),
                "time_since_last_message": time_since_last_message,
                "message_timeout": self.message_timeout,
                "health_status": "healthy" if self.connected else "unhealthy",
                "channel_names": list(self.subscribed_channels),
                "shutdown": self._shutdown,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get connection metrics: {e!s}")
            return {"error": str(e)}

    def is_connected(self) -> bool:
        """
        Check if WebSocket is connected and healthy.

        Returns:
            bool: True if connected, False otherwise
        """
        return self.connected and self.ws is not None and not self.ws.closed and not self._shutdown

    def get_active_streams(self) -> dict[str, Any]:
        """
        Get information about active streams.

        Returns:
            Dict[str, Any]: Dictionary of active streams
        """
        return self.active_streams.copy()

    def _create_auth_headers(self) -> dict[str, str]:
        """
        Create authentication headers for WebSocket subscription.

        Returns:
            Dict containing authentication headers
        """
        timestamp = str(int(time.time()))
        message = timestamp + "GET" + "/users/self/verify"

        signature = base64.b64encode(
            hmac.new(
                base64.b64decode(self.api_secret), message.encode("utf-8"), hashlib.sha256
            ).digest()
        ).decode("utf-8")

        return {
            "signature": signature,
            "key": self.api_key,
            "passphrase": self.passphrase,
            "timestamp": timestamp,
        }

    async def _listen_messages(self) -> None:
        """
        Listen for messages from WebSocket.
        """
        try:
            while not self._shutdown and self.ws and not self.ws.closed:
                try:
                    message = await self.ws.recv()
                    await self._handle_message(message)

                except websockets.exceptions.ConnectionClosed:
                    self.logger.warning("WebSocket connection closed")
                    break
                except Exception as e:
                    self.logger.error(f"Error receiving message: {e!s}")

        except Exception as e:
            self.logger.error(f"Error in message listener: {e!s}")
        finally:
            if self.connected and not self._shutdown:
                await self._handle_disconnect()

    async def _handle_message(self, message: str) -> None:
        """
        Handle incoming WebSocket message.

        Args:
            message: Raw WebSocket message
        """
        try:
            # Update last message time
            self.last_message_time = datetime.now(timezone.utc)
            self._total_messages_received += 1

            data = json.loads(message)
            message_type = data.get("type", "")

            # Route message to appropriate handler
            if message_type == "ticker":
                await self._handle_ticker_message(data)
            elif message_type == "l2update":
                await self._handle_orderbook_message(data)
            elif message_type == "match":
                await self._handle_trade_message(data)
            elif message_type in ["received", "open", "done", "change", "activate"]:
                await self._handle_user_message(data)
            elif message_type == "subscriptions":
                self.logger.info(f"Subscription confirmation: {data}")
            elif message_type == "error":
                self.logger.error(f"WebSocket error: {data}")
            else:
                self.logger.debug(f"Unhandled message type: {message_type}")

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse message: {e!s}")
        except Exception as e:
            self.logger.error(f"Error handling message: {e!s}")

    async def _handle_ticker_message(self, data: dict) -> None:
        """
        Handle ticker message.

        Args:
            data: Ticker message data
        """
        try:
            symbol = data.get("product_id", "")
            channel_key = f"ticker_{symbol}"

            if channel_key in self.callbacks:
                # Convert to unified Ticker format
                ticker = Ticker(
                    symbol=symbol,
                    bid=Decimal(str(data.get("best_bid", "0"))),
                    ask=Decimal(str(data.get("best_ask", "0"))),
                    last_price=Decimal(str(data.get("price", "0"))),
                    volume_24h=Decimal(str(data.get("volume_24h", "0"))),
                    price_change_24h=Decimal("0"),  # Not available in ticker
                    timestamp=datetime.fromisoformat(data.get("time", "").replace("Z", "+00:00")),
                )

                # Call registered callbacks
                for callback in self.callbacks[channel_key]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(ticker)
                        else:
                            callback(ticker)
                    except Exception as e:
                        self.logger.error(f"Error in ticker callback: {e!s}")

        except Exception as e:
            self.logger.error(f"Error handling ticker message: {e!s}")

    async def _handle_orderbook_message(self, data: dict) -> None:
        """
        Handle order book message.

        Args:
            data: Order book message data
        """
        try:
            symbol = data.get("product_id", "")
            channel_key = f"level2_{symbol}"

            if channel_key in self.callbacks:
                # Convert to unified OrderBook format
                changes = data.get("changes", [])
                bids = []
                asks = []

                for change in changes:
                    side, price, size = change
                    price_decimal = Decimal(str(price))
                    size_decimal = Decimal(str(size))

                    if side == "buy":
                        bids.append([price_decimal, size_decimal])
                    else:
                        asks.append([price_decimal, size_decimal])

                order_book = OrderBook(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    timestamp=datetime.fromisoformat(data.get("time", "").replace("Z", "+00:00")),
                )

                # Call registered callbacks
                for callback in self.callbacks[channel_key]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(order_book)
                        else:
                            callback(order_book)
                    except Exception as e:
                        self.logger.error(f"Error in orderbook callback: {e!s}")

        except Exception as e:
            self.logger.error(f"Error handling orderbook message: {e!s}")

    async def _handle_trade_message(self, data: dict) -> None:
        """
        Handle trade message.

        Args:
            data: Trade message data
        """
        try:
            symbol = data.get("product_id", "")
            channel_key = f"matches_{symbol}"

            if channel_key in self.callbacks:
                # Convert to unified Trade format
                trade = Trade(
                    id=str(data.get("trade_id", "")),
                    symbol=symbol,
                    side=OrderSide.BUY if data.get("side") == "buy" else OrderSide.SELL,
                    amount=Decimal(str(data.get("size", "0"))),
                    price=Decimal(str(data.get("price", "0"))),
                    timestamp=datetime.fromisoformat(data.get("time", "").replace("Z", "+00:00")),
                    fee=Decimal("0"),
                )

                # Call registered callbacks
                for callback in self.callbacks[channel_key]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(trade)
                        else:
                            callback(trade)
                    except Exception as e:
                        self.logger.error(f"Error in trade callback: {e!s}")

        except Exception as e:
            self.logger.error(f"Error handling trade message: {e!s}")

    async def _handle_user_message(self, data: dict) -> None:
        """
        Handle user data message.

        Args:
            data: User data message
        """
        try:
            channel_key = "user"

            if channel_key in self.callbacks:
                # Call registered callbacks
                for callback in self.callbacks[channel_key]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data)
                        else:
                            callback(data)
                    except Exception as e:
                        self.logger.error(f"Error in user data callback: {e!s}")

        except Exception as e:
            self.logger.error(f"Error handling user message: {e!s}")

    async def _handle_disconnect(self) -> None:
        """
        Handle disconnection and prepare for reconnection.
        """
        self.connected = False
        self._total_reconnections += 1

        if not self._shutdown:
            await self._schedule_reconnect()

    async def _schedule_reconnect(self) -> None:
        """
        Schedule reconnection with exponential backoff.
        """
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
        asyncio.create_task(self._reconnect_after_delay(delay))

    async def _reconnect_after_delay(self, delay: float) -> None:
        """
        Reconnect after specified delay.

        Args:
            delay: Delay in seconds
        """
        try:
            await asyncio.sleep(delay)

            if self._shutdown:
                return

            self.logger.info("Attempting to reconnect...")

            # Store current subscriptions for resubscription
            channels_to_restore = self.subscribed_channels.copy()

            # Attempt reconnection
            if await self.connect():
                self.logger.info("Reconnection successful, restoring subscriptions...")

                # Restore subscriptions
                for channel_key in channels_to_restore:
                    try:
                        callbacks = self.callbacks.get(channel_key, [])
                        if not callbacks:
                            continue

                        # Parse channel key to determine subscription type
                        if channel_key.startswith("ticker_"):
                            symbol = channel_key.replace("ticker_", "")
                            for callback in callbacks:
                                await self.subscribe_to_ticker(symbol, callback)
                        elif channel_key.startswith("level2_"):
                            symbol = channel_key.replace("level2_", "")
                            for callback in callbacks:
                                await self.subscribe_to_orderbook(symbol, callback)
                        elif channel_key.startswith("matches_"):
                            symbol = channel_key.replace("matches_", "")
                            for callback in callbacks:
                                await self.subscribe_to_trades(symbol, callback)
                        elif channel_key == "user":
                            for callback in callbacks:
                                await self.subscribe_to_user_data(callback)

                    except Exception as e:
                        self.logger.error(f"Failed to restore subscription {channel_key}: {e!s}")

                self.logger.info("Subscription restoration completed")
            else:
                self.logger.error("Reconnection failed")

        except Exception as e:
            self.logger.error(f"Error during reconnection: {e!s}")

    async def _health_monitor(self) -> None:
        """
        Monitor connection health and trigger reconnection if needed.
        """
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

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
