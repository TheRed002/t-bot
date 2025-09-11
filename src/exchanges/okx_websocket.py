"""
OKX WebSocket Implementation (P-005)

This module implements WebSocket connections for OKX exchange,
providing real-time market data streaming capabilities.

CRITICAL: This integrates with P-001 (core types, exceptions, config), P-002A (error handling),
and P-003 (base exchange interface) components.

OKX WebSocket Features:
- Public and private WebSocket streams
- Authentication with API key, secret, and passphrase
- Automatic reconnection with exponential backoff
- Message queuing during disconnections
- Heartbeat monitoring for connection health
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

from src.core.config import Config
from src.core.exceptions import ExchangeConnectionError, ExchangeError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types.market import OrderBook, OrderBookLevel, Ticker, Trade
from src.core.types.trading import OrderSide

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler


class OKXWebSocketManager:
    """
    OKX WebSocket manager for real-time data streaming.

    Handles WebSocket connections to OKX for:
    - Public market data streams (tickers, order books, trades)
    - Private account data streams (orders, positions, balances)
    - Automatic reconnection and error handling
    """

    def __init__(self, config: Config, exchange_name: str = "okx", error_handler: ErrorHandler | None = None):
        """
        Initialize OKX WebSocket manager.

        Args:
            config: Application configuration
            exchange_name: Exchange name (default: "okx")
            error_handler: Error handler service (injected)
        """
        self.config = config
        self.exchange_name = exchange_name

        # Store injected error handler
        self._injected_error_handler = error_handler

        # OKX API credentials
        self.api_key = getattr(config.exchange, "okx_api_key", "")
        self.api_secret = getattr(config.exchange, "okx_api_secret", "")
        self.passphrase = getattr(config.exchange, "okx_passphrase", "")
        self.sandbox = getattr(config.exchange, "okx_sandbox", False)

        # WebSocket URLs
        if self.sandbox:
            self.public_ws_url = "wss://wspap.okx.com:8443/ws/v5/public"
            self.private_ws_url = "wss://wspap.okx.com:8443/ws/v5/private"
        else:
            self.public_ws_url = "wss://ws.okx.com:8443/ws/v5/public"
            self.private_ws_url = "wss://ws.okx.com:8443/ws/v5/private"

        # WebSocket connections
        self.public_ws: websockets.WebSocketClientProtocol | None = None
        self.private_ws: websockets.WebSocketClientProtocol | None = None

        # Stream subscriptions
        self.public_subscriptions: dict[str, list[Callable]] = {}
        self.private_subscriptions: dict[str, list[Callable]] = {}

        # Connection state
        self.connected = False
        self.last_heartbeat = None
        self.reconnect_attempts = 0
        # Configuration-based reconnection parameters
        ws_config = getattr(config, "websocket", {})
        self.max_reconnect_attempts = getattr(ws_config, "max_reconnect_attempts", 10)
        self.base_reconnect_delay = getattr(ws_config, "base_reconnect_delay_seconds", 1)
        self.max_reconnect_delay = getattr(ws_config, "max_reconnect_delay_seconds", 60)
        self._shutdown = False
        self._connection_lock = asyncio.Lock()

        # Health monitoring
        self.last_message_time = None
        self.message_timeout = getattr(ws_config, "message_timeout_seconds", 60)
        self._health_check_task: asyncio.Task | None = None

        # Connection metrics
        self._connection_start_time = None
        self._total_messages_received = 0
        self._total_reconnections = 0

        # Message listener tasks
        self._public_listener_task: asyncio.Task | None = None
        self._private_listener_task: asyncio.Task | None = None
        self._reconnect_task: asyncio.Task | None = None

        # Message queue for reconnection
        self.message_queue: list[dict] = []

        # Initialize error handling
        # Use injected error handler if available
        self.error_handler = self._injected_error_handler or ErrorHandler(config)

        # Initialize logger
        self.logger = get_logger(f"okx.websocket.{exchange_name}")

        self.logger.info(f"Initialized OKX WebSocket manager: {exchange_name}")

    async def connect(self) -> bool:
        """
        Establish WebSocket connections to OKX with enhanced error handling.

        Returns:
            bool: True if connection successful, False otherwise
        """
        async with self._connection_lock:
            try:
                if self._shutdown:
                    return False

                self.logger.info("Connecting to OKX WebSocket...")

                # Connect to public WebSocket
                await self._connect_public_websocket()

                # Connect to private WebSocket (if credentials provided)
                if self.api_key and self.api_secret and self.passphrase:
                    await self._connect_private_websocket()

                self.connected = True
                self.last_heartbeat = datetime.now(timezone.utc)
                self.last_message_time = datetime.now(timezone.utc)
                self._connection_start_time = datetime.now(timezone.utc)
                self.reconnect_attempts = 0

                # Start health monitoring
                if not self._health_check_task or self._health_check_task.done():
                    self._health_check_task = asyncio.create_task(self._health_monitor())

                self.logger.info("Successfully connected to OKX WebSocket")
                return True

            except Exception as e:
                self.logger.error(f"Failed to connect to OKX WebSocket: {e!s}")
                self.connected = False
                await self._schedule_reconnect()
                return False

    async def disconnect(self) -> None:
        """Disconnect from OKX WebSocket with proper cleanup."""
        async with self._connection_lock:
            try:
                self._shutdown = True
                self.logger.info("Disconnecting from OKX WebSocket...")

                # Cancel health monitoring with proper resource cleanup
                if self._health_check_task and not self._health_check_task.done():
                    self._health_check_task.cancel()
                    try:
                        await self._health_check_task
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        self.logger.error(f"Error waiting for health check task: {e}")
                    finally:
                        self._health_check_task = None

                # Cancel listener tasks with proper resource cleanup
                if self._public_listener_task and not self._public_listener_task.done():
                    self._public_listener_task.cancel()
                    try:
                        await self._public_listener_task
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        self.logger.error(f"Error canceling public listener task: {e}")
                    finally:
                        # Always clear task reference
                        self._public_listener_task = None

                if self._private_listener_task and not self._private_listener_task.done():
                    self._private_listener_task.cancel()
                    try:
                        await self._private_listener_task
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        self.logger.error(f"Error canceling private listener task: {e}")
                    finally:
                        # Always clear task reference
                        self._private_listener_task = None

                # Close public WebSocket with proper resource cleanup
                if self.public_ws and not self.public_ws.closed:
                    try:
                        await self.public_ws.close()
                    except Exception as e:
                        self.logger.error(f"Error closing public WebSocket: {e}")
                    finally:
                        # Always clear websocket reference
                        self.public_ws = None

                # Close private WebSocket with proper resource cleanup
                if self.private_ws and not self.private_ws.closed:
                    try:
                        await self.private_ws.close()
                    except Exception as e:
                        self.logger.error(f"Error closing private WebSocket: {e}")
                    finally:
                        # Always clear websocket reference
                        self.private_ws = None

                # Clear state
                self.connected = False
                self.public_subscriptions.clear()
                self.private_subscriptions.clear()

                self.logger.info("Successfully disconnected from OKX WebSocket")

            except Exception as e:
                self.logger.error(f"Error disconnecting from OKX WebSocket: {e!s}")

    async def subscribe_to_ticker(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to ticker stream for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USDT')
            callback: Callback function to handle ticker data
        """
        try:
            stream_name = f"tickers.{symbol}"

            if stream_name not in self.public_subscriptions:
                self.public_subscriptions[stream_name] = []

            self.public_subscriptions[stream_name].append(callback)

            # Subscribe to stream with connection check
            subscribe_message = {
                "op": "subscribe",
                "args": [{"channel": "tickers", "instId": symbol}],
            }

            if not self.public_ws or self.public_ws.closed:
                raise ExchangeConnectionError("Public WebSocket not connected")
            await self._send_public_message(subscribe_message)
            self.logger.info(f"Subscribed to ticker stream for {symbol}")

        except Exception as e:
            self.logger.error(f"Failed to subscribe to ticker for {symbol}: {e!s}")
            raise ExchangeError(f"Failed to subscribe to ticker: {e!s}")

    async def subscribe_to_orderbook(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to order book stream for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USDT')
            callback: Callback function to handle order book data
        """
        try:
            stream_name = f"books.{symbol}"

            if stream_name not in self.public_subscriptions:
                self.public_subscriptions[stream_name] = []

            self.public_subscriptions[stream_name].append(callback)

            # Subscribe to stream
            subscribe_message = {
                "op": "subscribe",
                "args": [{"channel": "books", "instId": symbol}],
            }

            await self._send_public_message(subscribe_message)
            self.logger.info(f"Subscribed to order book stream for {symbol}")

        except Exception as e:
            self.logger.error(f"Failed to subscribe to order book for {symbol}: {e!s}")
            raise ExchangeError(f"Failed to subscribe to order book: {e!s}")

    async def subscribe_to_trades(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to trades stream for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USDT')
            callback: Callback function to handle trade data
        """
        try:
            stream_name = f"trades.{symbol}"

            if stream_name not in self.public_subscriptions:
                self.public_subscriptions[stream_name] = []

            self.public_subscriptions[stream_name].append(callback)

            # Subscribe to stream
            subscribe_message = {
                "op": "subscribe",
                "args": [{"channel": "trades", "instId": symbol}],
            }

            await self._send_public_message(subscribe_message)
            self.logger.info(f"Subscribed to trades stream for {symbol}")

        except Exception as e:
            self.logger.error(f"Failed to subscribe to trades for {symbol}: {e!s}")
            raise ExchangeError(f"Failed to subscribe to trades: {e!s}")

    async def subscribe_to_account(self, callback: Callable) -> None:
        """
        Subscribe to account data stream.

        Args:
            callback: Callback function to handle account data
        """
        try:
            if not self.private_ws:
                raise ExchangeConnectionError("Private WebSocket not connected")

            stream_name = "account"

            if stream_name not in self.private_subscriptions:
                self.private_subscriptions[stream_name] = []

            self.private_subscriptions[stream_name].append(callback)

            # Subscribe to account stream
            subscribe_message = {"op": "subscribe", "args": [{"channel": "account"}]}

            await self._send_private_message(subscribe_message)
            self.logger.info("Subscribed to account stream")

        except Exception as e:
            self.logger.error(f"Failed to subscribe to account stream: {e!s}")
            raise ExchangeError(f"Failed to subscribe to account stream: {e!s}")

    async def _connect_public_websocket(self) -> None:
        """
        Connect to OKX public WebSocket with proper resource cleanup.
        """
        websocket_connection = None
        try:
            websocket_connection = await websockets.connect(
                self.public_ws_url, ping_interval=20, ping_timeout=10
            )

            # Only assign after successful connection
            self.public_ws = websocket_connection

            # Start listening for messages
            self._public_listener_task = asyncio.create_task(self._listen_public_messages())

            self.logger.info("Connected to OKX public WebSocket")

        except Exception as e:
            self.logger.error(f"Failed to connect to OKX public WebSocket: {e!s}")

            # Cleanup websocket connection on failure
            if websocket_connection and not websocket_connection.closed:
                try:
                    await websocket_connection.close()
                except Exception as cleanup_error:
                    self.logger.error(f"Error closing websocket during cleanup: {cleanup_error}")

            raise ExchangeConnectionError(f"Failed to connect to public WebSocket: {e!s}")

    async def _connect_private_websocket(self) -> None:
        """
        Connect to OKX private WebSocket with authentication and proper resource cleanup.
        """
        websocket_connection = None
        try:
            websocket_connection = await websockets.connect(
                self.private_ws_url, ping_interval=20, ping_timeout=10
            )

            # Only assign after successful connection
            self.private_ws = websocket_connection

            # Authenticate private WebSocket
            await self._authenticate_private_websocket()

            # Start listening for messages
            self._private_listener_task = asyncio.create_task(self._listen_private_messages())

            self.logger.info("Connected to OKX private WebSocket")

        except Exception as e:
            self.logger.error(f"Failed to connect to OKX private WebSocket: {e!s}")

            # Cleanup websocket connection on failure
            if websocket_connection and not websocket_connection.closed:
                try:
                    await websocket_connection.close()
                except Exception as cleanup_error:
                    self.logger.error(f"Error closing websocket during cleanup: {cleanup_error}")

            raise ExchangeConnectionError(f"Failed to connect to private WebSocket: {e!s}")

    async def _authenticate_private_websocket(self) -> None:
        """
        Authenticate private WebSocket connection.
        """
        try:
            timestamp = str(int(time.time()))

            # Create signature
            message = timestamp + "GET" + "/users/self/verify"
            signature = base64.b64encode(
                hmac.new(
                    self.api_secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
                ).digest()
            ).decode("utf-8")

            # Authentication message
            auth_message = {
                "op": "login",
                "args": [
                    {
                        "apiKey": self.api_key,
                        "passphrase": self.passphrase,
                        "timestamp": timestamp,
                        "sign": signature,
                    }
                ],
            }

            await self._send_private_message(auth_message)

            # Wait for authentication response with timeout
            try:
                if not self.private_ws:
                    raise ExchangeError("Private WebSocket connection lost during authentication")
                response = await asyncio.wait_for(self.private_ws.recv(), timeout=10.0)
                response_data = json.loads(response)
            except asyncio.TimeoutError:
                raise ExchangeError("Authentication timeout")
            except Exception as e:
                raise ExchangeError(f"Authentication response error: {e}")

            if response_data.get("code") != "0":
                raise ExchangeError(
                    f"Authentication failed: {response_data.get('msg', 'Unknown error')}"
                )

            self.logger.info("Successfully authenticated private WebSocket")

        except Exception as e:
            self.logger.error(f"Failed to authenticate private WebSocket: {e!s}")
            raise ExchangeError(f"Authentication failed: {e!s}")

    async def _listen_public_messages(self) -> None:
        """
        Listen for messages from public WebSocket with enhanced error handling.
        """
        try:
            while not self._shutdown and self.connected and self.public_ws:
                try:
                    # Add timeout to prevent hanging on recv()
                    message = await asyncio.wait_for(self.public_ws.recv(), timeout=30.0)

                    # Update last message time
                    self.last_message_time = datetime.now(timezone.utc)
                    self._total_messages_received += 1

                    await self._handle_public_message(message)
                except websockets.exceptions.ConnectionClosed:
                    self.logger.warning("Public WebSocket connection closed")
                    break
                except asyncio.TimeoutError:
                    self.logger.warning("Public WebSocket receive timeout, connection may be stale")
                    break
                except Exception as e:
                    self.logger.error(f"Error handling public message: {e!s}")

        except Exception as e:
            self.logger.error(f"Error in public message listener: {e!s}")
        finally:
            if self.connected and not self._shutdown:
                await self._handle_disconnect()

    async def _listen_private_messages(self) -> None:
        """
        Listen for messages from private WebSocket with enhanced error handling.
        """
        try:
            while not self._shutdown and self.connected and self.private_ws:
                try:
                    # Add timeout to prevent hanging on recv()
                    message = await asyncio.wait_for(self.private_ws.recv(), timeout=30.0)

                    # Update last message time
                    self.last_message_time = datetime.now(timezone.utc)
                    self._total_messages_received += 1

                    await self._handle_private_message(message)
                except websockets.exceptions.ConnectionClosed:
                    self.logger.warning("Private WebSocket connection closed")
                    break
                except asyncio.TimeoutError:
                    self.logger.warning("Private WebSocket receive timeout, connection may be stale")
                    break
                except Exception as e:
                    self.logger.error(f"Error handling private message: {e!s}")

        except Exception as e:
            self.logger.error(f"Error in private message listener: {e!s}")
        finally:
            if self.connected and not self._shutdown:
                await self._handle_disconnect()

    async def _handle_public_message(self, message: str) -> None:
        """
        Handle incoming public WebSocket message.

        Args:
            message: Raw WebSocket message
        """
        try:
            data = json.loads(message)

            # Handle different message types
            if "event" in data:
                # Event message (subscription confirmation, etc.)
                await self._handle_event_message(data)
            elif "data" in data:
                # Data message (ticker, order book, trades)
                await self._handle_data_message(data)
            else:
                self.logger.warning(f"Unknown public message format: {data}")

        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse public message: {message}")
        except Exception as e:
            self.logger.error(f"Error handling public message: {e!s}")

    async def _handle_private_message(self, message: str) -> None:
        """
        Handle incoming private WebSocket message.

        Args:
            message: Raw WebSocket message
        """
        try:
            data = json.loads(message)

            # Handle different message types
            if "event" in data:
                # Event message (authentication, subscription confirmation)
                await self._handle_private_event_message(data)
            elif "data" in data:
                # Data message (account updates, orders)
                await self._handle_private_data_message(data)
            else:
                self.logger.warning(f"Unknown private message format: {data}")

        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse private message: {message}")
        except Exception as e:
            self.logger.error(f"Error handling private message: {e!s}")

    async def _handle_event_message(self, data: dict) -> None:
        """
        Handle event messages from public WebSocket.

        Args:
            data: Event message data
        """
        event = data.get("event", "")

        if event == "subscribe":
            self.logger.info(f"Successfully subscribed to channel: {data.get('arg', {})}")
        elif event == "error":
            self.logger.error(f"Public WebSocket error: {data.get('msg', 'Unknown error')}")
        else:
            self.logger.debug(f"Public event: {event}")

    async def _handle_private_event_message(self, data: dict) -> None:
        """
        Handle event messages from private WebSocket.

        Args:
            data: Event message data
        """
        event = data.get("event", "")

        if event == "login":
            self.logger.info("Successfully logged in to private WebSocket")
        elif event == "subscribe":
            self.logger.info(f"Successfully subscribed to private channel: {data.get('arg', {})}")
        elif event == "error":
            self.logger.error(f"Private WebSocket error: {data.get('msg', 'Unknown error')}")
        else:
            self.logger.debug(f"Private event: {event}")

    async def _handle_data_message(self, data: dict) -> None:
        """
        Handle data messages from public WebSocket.

        Args:
            data: Data message
        """
        try:
            arg = data.get("arg", {})
            channel = arg.get("channel", "")
            inst_id = arg.get("instId", "")
            stream_data = data.get("data", [])

            if not stream_data:
                return

            # Route to appropriate handlers
            if channel == "tickers":
                await self._handle_ticker_data(inst_id, stream_data)
            elif channel == "books":
                await self._handle_orderbook_data(inst_id, stream_data)
            elif channel == "trades":
                await self._handle_trades_data(inst_id, stream_data)
            else:
                self.logger.debug(f"Unhandled public channel: {channel}")

        except Exception as e:
            self.logger.error(f"Error handling data message: {e!s}")

    async def _handle_private_data_message(self, data: dict) -> None:
        """
        Handle data messages from private WebSocket.

        Args:
            data: Data message
        """
        try:
            arg = data.get("arg", {})
            channel = arg.get("channel", "")
            stream_data = data.get("data", [])

            if not stream_data:
                return

            # Route to appropriate handlers
            if channel == "account":
                await self._handle_account_data(stream_data)
            else:
                self.logger.debug(f"Unhandled private channel: {channel}")

        except Exception as e:
            self.logger.error(f"Error handling private data message: {e!s}")

    async def _handle_ticker_data(self, symbol: str, data: list[dict]) -> None:
        """
        Handle ticker data from WebSocket.

        Args:
            symbol: Trading symbol
            data: Ticker data
        """
        try:
            stream_name = f"tickers.{symbol}"

            if stream_name in self.public_subscriptions:
                for ticker_data in data:
                    # Convert to unified Ticker format
                    ticker = Ticker(
                        symbol=symbol,
                        bid_price=Decimal(ticker_data.get("bidPx", "0")),
                        bid_quantity=Decimal(ticker_data.get("bidSz", "0")),
                        ask_price=Decimal(ticker_data.get("askPx", "0")),
                        ask_quantity=Decimal(ticker_data.get("askSz", "0")),
                        last_price=Decimal(ticker_data.get("last", "0")),
                        volume=Decimal(ticker_data.get("vol24h", "0")),
                        timestamp=datetime.now(timezone.utc),
                        exchange="okx",
                        price_change=Decimal(ticker_data.get("change24h", "0")),
                    )

                    # Call all registered callbacks concurrently
                    callback_tasks = []
                    for callback in self.public_subscriptions[stream_name]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                callback_tasks.append(callback(ticker))
                            else:
                                # Run sync callback in executor to avoid blocking
                                callback_tasks.append(
                                    asyncio.get_event_loop().run_in_executor(
                                        None, callback, ticker
                                    )
                                )
                        except Exception as e:
                            self.logger.error(f"Error preparing ticker callback: {e!s}")

                    # Execute all callbacks concurrently with error handling
                    if callback_tasks:
                        await asyncio.gather(*callback_tasks, return_exceptions=True)

        except Exception as e:
            self.logger.error(f"Error handling ticker data: {e!s}")

    async def _handle_orderbook_data(self, symbol: str, data: list[dict]) -> None:
        """
        Handle order book data from WebSocket.

        Args:
            symbol: Trading symbol
            data: Order book data
        """
        try:
            stream_name = f"books.{symbol}"

            if stream_name in self.public_subscriptions:
                for book_data in data:
                    # Convert to unified OrderBook format
                    bids = [
                        OrderBookLevel(price=Decimal(str(price)), quantity=Decimal(str(size)))
                        for price, size in book_data.get("bids", [])
                    ]
                    asks = [
                        OrderBookLevel(price=Decimal(str(price)), quantity=Decimal(str(size)))
                        for price, size in book_data.get("asks", [])
                    ]

                    order_book = OrderBook(
                        symbol=symbol,
                        bids=bids,
                        asks=asks,
                        timestamp=datetime.now(timezone.utc),
                        exchange="okx",
                    )

                    # Call all registered callbacks concurrently
                    callback_tasks = []
                    for callback in self.public_subscriptions[stream_name]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                callback_tasks.append(callback(order_book))
                            else:
                                # Run sync callback in executor to avoid blocking
                                callback_tasks.append(
                                    asyncio.get_event_loop().run_in_executor(
                                        None, callback, order_book
                                    )
                                )
                        except Exception as e:
                            self.logger.error(f"Error preparing order book callback: {e!s}")

                    # Execute all callbacks concurrently with error handling
                    if callback_tasks:
                        await asyncio.gather(*callback_tasks, return_exceptions=True)

        except Exception as e:
            self.logger.error(f"Error handling order book data: {e!s}")

    async def _handle_trades_data(self, symbol: str, data: list[dict]) -> None:
        """
        Handle trades data from WebSocket.

        Args:
            symbol: Trading symbol
            data: Trades data
        """
        try:
            stream_name = f"trades.{symbol}"

            if stream_name in self.public_subscriptions:
                for trade_data in data:
                    # Convert to unified Trade format
                    trade = Trade(
                        trade_id=str(trade_data.get("tradeId", "")),
                        order_id="",  # Not available in public trade data
                        symbol=symbol,
                        side=OrderSide.BUY if trade_data.get("side") == "buy" else OrderSide.SELL,
                        quantity=Decimal(str(trade_data.get("sz", "0"))),
                        price=Decimal(str(trade_data.get("px", "0"))),
                        timestamp=datetime.fromtimestamp(
                            int(trade_data.get("ts", 0)) / 1000, tz=timezone.utc
                        ),
                        fee=Decimal("0"),  # Not available in public trade data
                        fee_currency="USDT",  # Default fee currency
                        exchange="okx",
                        is_maker=False,  # Not available in public trade data
                    )

                    # Call all registered callbacks
                    for callback in self.public_subscriptions[stream_name]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(trade)
                            else:
                                callback(trade)
                        except Exception as e:
                            self.logger.error(f"Error in trades callback: {e!s}")

        except Exception as e:
            self.logger.error(f"Error handling trades data: {e!s}")

    async def _handle_account_data(self, data: list[dict]) -> None:
        """
        Handle account data from WebSocket.

        Args:
            data: Account data
        """
        try:
            stream_name = "account"

            if stream_name in self.private_subscriptions:
                # Call all registered callbacks
                for callback in self.private_subscriptions[stream_name]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data)
                        else:
                            callback(data)
                    except Exception as e:
                        self.logger.error(f"Error in account callback: {e!s}")

        except Exception as e:
            self.logger.error(f"Error handling account data: {e!s}")

    async def _send_public_message(self, message: dict) -> None:
        """
        Send message to public WebSocket.

        Args:
            message: Message to send
        """
        try:
            if not self.public_ws:
                raise ExchangeConnectionError("Public WebSocket not connected")

            await self.public_ws.send(json.dumps(message))

        except Exception as e:
            self.logger.error(f"Failed to send public message: {e!s}")
            raise ExchangeError(f"Failed to send public message: {e!s}")

    async def _send_private_message(self, message: dict) -> None:
        """
        Send message to private WebSocket.

        Args:
            message: Message to send
        """
        try:
            if not self.private_ws:
                raise ExchangeConnectionError("Private WebSocket not connected")

            await self.private_ws.send(json.dumps(message))

        except Exception as e:
            self.logger.error(f"Failed to send private message: {e!s}")
            raise ExchangeError(f"Failed to send private message: {e!s}")

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
        task = asyncio.create_task(self._reconnect_after_delay(delay))
        # Store task to prevent garbage collection
        self._reconnect_task = task

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
            public_subs_to_restore = dict(self.public_subscriptions)
            private_subs_to_restore = dict(self.private_subscriptions)

            # Attempt reconnection
            if await self.connect():
                self.logger.info("Reconnection successful, restoring subscriptions...")

                # Restore public subscriptions
                for stream_name, callbacks in public_subs_to_restore.items():
                    try:
                        # Parse stream name to determine subscription type
                        if stream_name.startswith("tickers."):
                            symbol = stream_name.replace("tickers.", "")
                            for callback in callbacks:
                                await self.subscribe_to_ticker(symbol, callback)
                        elif stream_name.startswith("books."):
                            symbol = stream_name.replace("books.", "")
                            for callback in callbacks:
                                await self.subscribe_to_orderbook(symbol, callback)
                        elif stream_name.startswith("trades."):
                            symbol = stream_name.replace("trades.", "")
                            for callback in callbacks:
                                await self.subscribe_to_trades(symbol, callback)

                    except Exception as e:
                        self.logger.error(
                            f"Failed to restore public subscription {stream_name}: {e!s}"
                        )

                # Restore private subscriptions
                for stream_name, callbacks in private_subs_to_restore.items():
                    try:
                        if stream_name == "account":
                            for callback in callbacks:
                                await self.subscribe_to_account(callback)

                    except Exception as e:
                        self.logger.error(
                            f"Failed to restore private subscription {stream_name}: {e!s}"
                        )

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

    def is_connected(self) -> bool:
        """
        Check if WebSocket is connected and healthy.

        Returns:
            bool: True if connected, False otherwise
        """
        return self.connected and not self._shutdown and (self.public_ws or self.private_ws)

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

            # Check if at least one WebSocket connection is active
            public_connected = self.public_ws and not self.public_ws.closed
            private_connected = self.private_ws and not self.private_ws.closed

            if not (public_connected or private_connected):
                self.logger.warning("Health check failed: no active WebSocket connections")
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
                "public_subscriptions": len(self.public_subscriptions),
                "private_subscriptions": len(self.private_subscriptions),
                "time_since_last_message": time_since_last_message,
                "message_timeout": self.message_timeout,
                "health_status": "monitoring",  # Would require async context to check
                "public_connected": self.public_ws is not None and not self.public_ws.closed,
                "private_connected": self.private_ws is not None and not self.private_ws.closed,
                "shutdown": self._shutdown,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get connection metrics: {e!s}")
            return {"error": str(e)}

    def get_status(self) -> str:
        """
        Get WebSocket connection status.

        Returns:
            str: Connection status
        """
        if not self.connected:
            return "disconnected"
        elif self.public_ws and self.private_ws:
            return "connected"
        elif self.public_ws:
            return "public_only"
        else:
            return "error"
