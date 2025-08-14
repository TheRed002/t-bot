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

logger = get_logger(__name__)


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
        self.max_reconnect_attempts = 5

        # Error handling
        self.error_handler = ErrorHandler(config.error_handling)

        logger.info("Initialized Binance WebSocket handler")

    async def connect(self) -> bool:
        """
        Establish WebSocket connections.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info("Connecting Binance WebSocket streams...")

            # Test connection by getting server time
            server_time = await self.client.get_server_time()
            logger.info(f"Binance server time: {server_time}")

            self.connected = True
            self.reconnect_attempts = 0

            logger.info("Successfully connected Binance WebSocket streams")
            return True

        except Exception as e:
            logger.error(f"Failed to connect Binance WebSocket streams: {e!s}")
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect all WebSocket streams."""
        try:
            logger.info("Disconnecting Binance WebSocket streams...")

            # Cancel all stream handlers
            for task in self.stream_handlers.values():
                task.cancel()

            # Close all active streams
            for stream_name in list(self.active_streams.keys()):
                await self._close_stream(stream_name)

            self.connected = False
            logger.info("Successfully disconnected Binance WebSocket streams")

        except Exception as e:
            logger.error(f"Error disconnecting Binance WebSocket streams: {e!s}")

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

                logger.info(f"Subscribed to ticker stream: {stream_name}")

        except Exception as e:
            logger.error(f"Error subscribing to ticker stream: {e!s}")
            raise ExchangeError(f"Failed to subscribe to ticker stream: {e!s}")

    async def subscribe_to_orderbook_stream(
        self, symbol: str, depth: str = "20", callback: Callable = None
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

                logger.info(f"Subscribed to orderbook stream: {stream_name}")

        except Exception as e:
            logger.error(f"Error subscribing to orderbook stream: {e!s}")
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

                logger.info(f"Subscribed to trade stream: {stream_name}")

        except Exception as e:
            logger.error(f"Error subscribing to trade stream: {e!s}")
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

                logger.info(f"Subscribed to user data stream: {stream_name}")

        except Exception as e:
            logger.error(f"Error subscribing to user data stream: {e!s}")
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

                logger.info(f"Unsubscribed from stream: {stream_name}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error unsubscribing from stream {stream_name}: {e!s}")
            return False

    # Stream handlers

    async def _handle_ticker_stream(self, stream_name: str, stream) -> None:
        """Handle ticker price stream data."""
        try:
            async with stream as stream_handler:
                async for msg in stream_handler:
                    try:
                        # Convert to Ticker format
                        ticker_data = self._convert_ticker_message(msg)

                        # Call registered callbacks
                        if stream_name in self.callbacks:
                            for callback in self.callbacks[stream_name]:
                                try:
                                    await callback(ticker_data)
                                except Exception as e:
                                    logger.error(f"Error in ticker callback: {e!s}")

                    except Exception as e:
                        logger.error(f"Error processing ticker message: {e!s}")

        except asyncio.CancelledError:
            logger.info(f"Ticker stream {stream_name} cancelled")
        except Exception as e:
            logger.error(f"Error handling ticker stream {stream_name}: {e!s}")
            await self._handle_stream_error(stream_name)

    async def _handle_orderbook_stream(self, stream_name: str, stream) -> None:
        """Handle order book depth stream data."""
        try:
            async with stream as stream_handler:
                async for msg in stream_handler:
                    try:
                        # Convert to OrderBook format
                        orderbook_data = self._convert_orderbook_message(msg)

                        # Call registered callbacks
                        if stream_name in self.callbacks:
                            for callback in self.callbacks[stream_name]:
                                try:
                                    await callback(orderbook_data)
                                except Exception as e:
                                    logger.error(f"Error in orderbook callback: {e!s}")

                    except Exception as e:
                        logger.error(f"Error processing orderbook message: {e!s}")

        except asyncio.CancelledError:
            logger.info(f"Orderbook stream {stream_name} cancelled")
        except Exception as e:
            logger.error(f"Error handling orderbook stream {stream_name}: {e!s}")
            await self._handle_stream_error(stream_name)

    async def _handle_trade_stream(self, stream_name: str, stream) -> None:
        """Handle trade execution stream data."""
        try:
            async with stream as stream_handler:
                async for msg in stream_handler:
                    try:
                        # Convert to Trade format
                        trade_data = self._convert_trade_message(msg)

                        # Call registered callbacks
                        if stream_name in self.callbacks:
                            for callback in self.callbacks[stream_name]:
                                try:
                                    await callback(trade_data)
                                except Exception as e:
                                    logger.error(f"Error in trade callback: {e!s}")

                    except Exception as e:
                        logger.error(f"Error processing trade message: {e!s}")

        except asyncio.CancelledError:
            logger.info(f"Trade stream {stream_name} cancelled")
        except Exception as e:
            logger.error(f"Error handling trade stream {stream_name}: {e!s}")
            await self._handle_stream_error(stream_name)

    async def _handle_user_data_stream(self, stream_name: str, stream) -> None:
        """Handle user data stream for account updates."""
        try:
            async with stream as stream_handler:
                async for msg in stream_handler:
                    try:
                        # Process different types of user data updates
                        if msg["e"] == "outboundAccountPosition":
                            # Account balance update
                            await self._handle_account_position_update(msg)
                        elif msg["e"] == "executionReport":
                            # Order execution update
                            await self._handle_execution_report(msg)
                        elif msg["e"] == "balanceUpdate":
                            # Balance update
                            await self._handle_balance_update(msg)

                        # Call registered callbacks
                        if stream_name in self.callbacks:
                            for callback in self.callbacks[stream_name]:
                                try:
                                    await callback(msg)
                                except Exception as e:
                                    logger.error(f"Error in user data callback: {e!s}")

                    except Exception as e:
                        logger.error(f"Error processing user data message: {e!s}")

        except asyncio.CancelledError:
            logger.info(f"User data stream {stream_name} cancelled")
        except Exception as e:
            logger.error(f"Error handling user data stream {stream_name}: {e!s}")
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
            quantity=Decimal(str(msg["q"])),
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

            logger.debug(f"Account position updated: {len(balances)} assets")

        except Exception as e:
            logger.error(f"Error handling account position update: {e!s}")

    async def _handle_execution_report(self, msg: dict) -> None:
        """Handle order execution report message."""
        try:
            order_id = str(msg["i"])
            status = msg["X"]
            executed_qty = Decimal(str(msg["z"]))

            logger.debug(f"Order execution: {order_id}, status: {status}, executed: {executed_qty}")

        except Exception as e:
            logger.error(f"Error handling execution report: {e!s}")

    async def _handle_balance_update(self, msg: dict) -> None:
        """Handle balance update message."""
        try:
            asset = msg["a"]
            delta = Decimal(str(msg["d"]))

            logger.debug(f"Balance update: {asset}, delta: {delta}")

        except Exception as e:
            logger.error(f"Error handling balance update: {e!s}")

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

                logger.info(f"Closed stream: {stream_name}")
        except Exception as e:
            logger.error(f"Error closing stream {stream_name}: {e!s}")

    async def _handle_stream_error(self, stream_name: str) -> None:
        """Handle stream error with reconnection logic."""
        try:
            logger.warning(f"Stream {stream_name} encountered an error")

            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_attempts += 1
                logger.info(
                    f"Attempting to reconnect stream {stream_name}"
                    f"(attempt {self.reconnect_attempts})"
                )

                # Wait before reconnecting
                await asyncio.sleep(2**self.reconnect_attempts)

                # Reconnect logic would go here
                # For now, just log the error
                logger.error(f"Stream {stream_name} reconnection not implemented")
            else:
                logger.error(f"Max reconnection attempts reached for stream {stream_name}")

        except Exception as e:
            logger.error(f"Error handling stream error: {e!s}")

    def get_active_streams(self) -> list[str]:
        """Get list of active stream names."""
        return list(self.active_streams.keys())

    def is_connected(self) -> bool:
        """Check if WebSocket handler is connected."""
        return self.connected

    async def health_check(self) -> bool:
        """
        Perform health check on WebSocket connections.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            if not self.connected:
                return False

            # Check if any streams are active
            if not self.active_streams:
                return True  # No streams is considered healthy

            # Check if stream handlers are running
            for stream_name, task in self.stream_handlers.items():
                if task.done():
                    logger.warning(f"Stream handler {stream_name} is done")
                    return False

            return True

        except Exception as e:
            logger.error(f"WebSocket health check failed: {e!s}")
            return False
