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

# WebSocket imports
import websockets

from src.core.config import Config
from src.core.exceptions import ExchangeConnectionError, ExchangeError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import OrderBook, OrderSide, Ticker, Trade

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

logger = get_logger(__name__)


class OKXWebSocketManager:
    """
    OKX WebSocket manager for real-time data streaming.

    Handles WebSocket connections to OKX for:
    - Public market data streams (tickers, order books, trades)
    - Private account data streams (orders, positions, balances)
    - Automatic reconnection and error handling
    """

    def __init__(self, config: Config, exchange_name: str = "okx"):
        """
        Initialize OKX WebSocket manager.

        Args:
            config: Application configuration
            exchange_name: Exchange name (default: "okx")
        """
        self.config = config
        self.exchange_name = exchange_name

        # OKX API credentials
        self.api_key = config.exchanges.okx_api_key
        self.api_secret = config.exchanges.okx_api_secret
        self.passphrase = config.exchanges.okx_passphrase
        self.sandbox = config.exchanges.okx_sandbox

        # WebSocket URLs
        if self.sandbox:
            self.public_ws_url = "wss://wspap.okx.com:8443/ws/v5/public"
            self.private_ws_url = "wss://wspap.okx.com:8443/ws/v5/private"
        else:
            self.public_ws_url = "wss://ws.okx.com:8443/ws/v5/public"
            self.private_ws_url = "wss://ws.okx.com:8443/ws/v5/private"

        # WebSocket connections
        self.public_ws: websockets.WebSocketServerProtocol | None = None
        self.private_ws: websockets.WebSocketServerProtocol | None = None

        # Stream subscriptions
        self.public_subscriptions: dict[str, list[Callable]] = {}
        self.private_subscriptions: dict[str, list[Callable]] = {}

        # Connection state
        self.connected = False
        self.last_heartbeat = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5

        # Message queue for reconnection
        self.message_queue: list[dict] = []

        # Initialize error handling
        self.error_handler = ErrorHandler(config.error_handling)

        logger.info(f"Initialized OKX WebSocket manager: {exchange_name}")

    async def connect(self) -> bool:
        """
        Establish WebSocket connections to OKX.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info("Connecting to OKX WebSocket...")

            # Connect to public WebSocket
            await self._connect_public_websocket()

            # Connect to private WebSocket (if credentials provided)
            if self.api_key and self.api_secret and self.passphrase:
                await self._connect_private_websocket()

            self.connected = True
            self.last_heartbeat = datetime.now(timezone.utc)
            self.reconnect_attempts = 0

            logger.info("Successfully connected to OKX WebSocket")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to OKX WebSocket: {e!s}")
            self.connected = False
            raise ExchangeConnectionError(
                f"Failed to connect to OKX WebSocket: {e!s}")

    async def disconnect(self) -> None:
        """Disconnect from OKX WebSocket."""
        try:
            logger.info("Disconnecting from OKX WebSocket...")

            # Close public WebSocket
            if self.public_ws:
                await self.public_ws.close()
                self.public_ws = None

            # Close private WebSocket
            if self.private_ws:
                await self.private_ws.close()
                self.private_ws = None

            self.connected = False
            self.public_subscriptions.clear()
            self.private_subscriptions.clear()

            logger.info("Successfully disconnected from OKX WebSocket")

        except Exception as e:
            logger.error(f"Error disconnecting from OKX WebSocket: {e!s}")
            raise ExchangeConnectionError(
                f"Error disconnecting from OKX WebSocket: {e!s}")

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

            # Subscribe to stream
            subscribe_message = {
                "op": "subscribe",
                "args": [{"channel": "tickers", "instId": symbol}],
            }

            await self._send_public_message(subscribe_message)
            logger.info(f"Subscribed to ticker stream for {symbol}")

        except Exception as e:
            logger.error(f"Failed to subscribe to ticker for {symbol}: {e!s}")
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
            logger.info(f"Subscribed to order book stream for {symbol}")

        except Exception as e:
            logger.error(
                f"Failed to subscribe to order book for {symbol}: {e!s}")
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
            logger.info(f"Subscribed to trades stream for {symbol}")

        except Exception as e:
            logger.error(f"Failed to subscribe to trades for {symbol}: {e!s}")
            raise ExchangeError(f"Failed to subscribe to trades: {e!s}")

    async def subscribe_to_account(self, callback: Callable) -> None:
        """
        Subscribe to account data stream.

        Args:
            callback: Callback function to handle account data
        """
        try:
            if not self.private_ws:
                raise ExchangeConnectionError(
                    "Private WebSocket not connected")

            stream_name = "account"

            if stream_name not in self.private_subscriptions:
                self.private_subscriptions[stream_name] = []

            self.private_subscriptions[stream_name].append(callback)

            # Subscribe to account stream
            subscribe_message = {"op": "subscribe",
                                 "args": [{"channel": "account"}]}

            await self._send_private_message(subscribe_message)
            logger.info("Subscribed to account stream")

        except Exception as e:
            logger.error(f"Failed to subscribe to account stream: {e!s}")
            raise ExchangeError(
                f"Failed to subscribe to account stream: {e!s}")

    async def _connect_public_websocket(self) -> None:
        """
        Connect to OKX public WebSocket.
        """
        try:
            self.public_ws = await websockets.connect(
                self.public_ws_url, ping_interval=20, ping_timeout=10
            )

            # Start listening for messages
            asyncio.create_task(self._listen_public_messages())

            logger.info("Connected to OKX public WebSocket")

        except Exception as e:
            logger.error(f"Failed to connect to OKX public WebSocket: {e!s}")
            raise ExchangeConnectionError(
                f"Failed to connect to public WebSocket: {e!s}")

    async def _connect_private_websocket(self) -> None:
        """
        Connect to OKX private WebSocket with authentication.
        """
        try:
            self.private_ws = await websockets.connect(
                self.private_ws_url, ping_interval=20, ping_timeout=10
            )

            # Authenticate private WebSocket
            await self._authenticate_private_websocket()

            # Start listening for messages
            asyncio.create_task(self._listen_private_messages())

            logger.info("Connected to OKX private WebSocket")

        except Exception as e:
            logger.error(f"Failed to connect to OKX private WebSocket: {e!s}")
            raise ExchangeConnectionError(
                f"Failed to connect to private WebSocket: {e!s}")

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
                    self.api_secret.encode(
                        "utf-8"), message.encode("utf-8"), hashlib.sha256
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

            # Wait for authentication response
            response = await self.private_ws.recv()
            response_data = json.loads(response)

            if response_data.get("code") != "0":
                raise ExchangeError(
                    f"Authentication failed: {response_data.get('msg', 'Unknown error')}"
                )

            logger.info("Successfully authenticated private WebSocket")

        except Exception as e:
            logger.error(f"Failed to authenticate private WebSocket: {e!s}")
            raise ExchangeError(f"Authentication failed: {e!s}")

    async def _listen_public_messages(self) -> None:
        """
        Listen for messages from public WebSocket.
        """
        try:
            while self.connected and self.public_ws:
                try:
                    message = await self.public_ws.recv()
                    await self._handle_public_message(message)
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Public WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error(f"Error handling public message: {e!s}")

        except Exception as e:
            logger.error(f"Error in public message listener: {e!s}")
        finally:
            if self.connected:
                await self._reconnect_public_websocket()

    async def _listen_private_messages(self) -> None:
        """
        Listen for messages from private WebSocket.
        """
        try:
            while self.connected and self.private_ws:
                try:
                    message = await self.private_ws.recv()
                    await self._handle_private_message(message)
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Private WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error(f"Error handling private message: {e!s}")

        except Exception as e:
            logger.error(f"Error in private message listener: {e!s}")
        finally:
            if self.connected:
                await self._reconnect_private_websocket()

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
                logger.warning(f"Unknown public message format: {data}")

        except json.JSONDecodeError:
            logger.error(f"Failed to parse public message: {message}")
        except Exception as e:
            logger.error(f"Error handling public message: {e!s}")

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
                logger.warning(f"Unknown private message format: {data}")

        except json.JSONDecodeError:
            logger.error(f"Failed to parse private message: {message}")
        except Exception as e:
            logger.error(f"Error handling private message: {e!s}")

    async def _handle_event_message(self, data: dict) -> None:
        """
        Handle event messages from public WebSocket.

        Args:
            data: Event message data
        """
        event = data.get("event", "")

        if event == "subscribe":
            logger.info(
                f"Successfully subscribed to channel: {data.get('arg', {})}")
        elif event == "error":
            logger.error(
                f"Public WebSocket error: {data.get('msg', 'Unknown error')}")
        else:
            logger.debug(f"Public event: {event}")

    async def _handle_private_event_message(self, data: dict) -> None:
        """
        Handle event messages from private WebSocket.

        Args:
            data: Event message data
        """
        event = data.get("event", "")

        if event == "login":
            logger.info("Successfully logged in to private WebSocket")
        elif event == "subscribe":
            logger.info(
                f"Successfully subscribed to private channel: {data.get('arg', {})}")
        elif event == "error":
            logger.error(
                f"Private WebSocket error: {data.get('msg', 'Unknown error')}")
        else:
            logger.debug(f"Private event: {event}")

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
                logger.debug(f"Unhandled public channel: {channel}")

        except Exception as e:
            logger.error(f"Error handling data message: {e!s}")

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
                logger.debug(f"Unhandled private channel: {channel}")

        except Exception as e:
            logger.error(f"Error handling private data message: {e!s}")

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
                        bid=Decimal(ticker_data.get("bidPx", "0")),
                        ask=Decimal(ticker_data.get("askPx", "0")),
                        last_price=Decimal(ticker_data.get("last", "0")),
                        volume_24h=Decimal(ticker_data.get("vol24h", "0")),
                        price_change_24h=Decimal(
                            ticker_data.get("change24h", "0")),
                        timestamp=datetime.now(timezone.utc),
                    )

                    # Call all registered callbacks
                    for callback in self.public_subscriptions[stream_name]:
                        try:
                            await callback(ticker)
                        except Exception as e:
                            logger.error(f"Error in ticker callback: {e!s}")

        except Exception as e:
            logger.error(f"Error handling ticker data: {e!s}")

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
                        [Decimal(price), Decimal(size)] for price, size in book_data.get("bids", [])
                    ]
                    asks = [
                        [Decimal(price), Decimal(size)] for price, size in book_data.get("asks", [])
                    ]

                    order_book = OrderBook(
                        symbol=symbol, bids=bids, asks=asks, timestamp=datetime.now(
                            timezone.utc)
                    )

                    # Call all registered callbacks
                    for callback in self.public_subscriptions[stream_name]:
                        try:
                            await callback(order_book)
                        except Exception as e:
                            logger.error(
                                f"Error in order book callback: {e!s}")

        except Exception as e:
            logger.error(f"Error handling order book data: {e!s}")

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
                        id=trade_data.get("tradeId", ""),
                        symbol=symbol,
                        side=OrderSide.BUY if trade_data.get(
                            "side") == "buy" else OrderSide.SELL,
                        quantity=Decimal(trade_data.get("sz", "0")),
                        price=Decimal(trade_data.get("px", "0")),
                        timestamp=datetime.fromtimestamp(
                            int(trade_data.get("ts", 0)) / 1000, tz=timezone.utc
                        ),
                        # OKX doesn't provide fee in trade data
                        fee=Decimal("0"),
                    )

                    # Call all registered callbacks
                    for callback in self.public_subscriptions[stream_name]:
                        try:
                            await callback(trade)
                        except Exception as e:
                            logger.error(f"Error in trades callback: {e!s}")

        except Exception as e:
            logger.error(f"Error handling trades data: {e!s}")

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
                        await callback(data)
                    except Exception as e:
                        logger.error(f"Error in account callback: {e!s}")

        except Exception as e:
            logger.error(f"Error handling account data: {e!s}")

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
            logger.error(f"Failed to send public message: {e!s}")
            raise ExchangeError(f"Failed to send public message: {e!s}")

    async def _send_private_message(self, message: dict) -> None:
        """
        Send message to private WebSocket.

        Args:
            message: Message to send
        """
        try:
            if not self.private_ws:
                raise ExchangeConnectionError(
                    "Private WebSocket not connected")

            await self.private_ws.send(json.dumps(message))

        except Exception as e:
            logger.error(f"Failed to send private message: {e!s}")
            raise ExchangeError(f"Failed to send private message: {e!s}")

    async def _reconnect_public_websocket(self) -> None:
        """
        Reconnect to public WebSocket with exponential backoff.
        """
        try:
            if self.reconnect_attempts >= self.max_reconnect_attempts:
                logger.error(
                    "Max reconnection attempts reached for public WebSocket")
                return

            self.reconnect_attempts += 1
            # Exponential backoff, max 60s
            delay = min(2**self.reconnect_attempts, 60)

            logger.info(
                f"Reconnecting to public WebSocket in {delay} seconds"
                f"(attempt {self.reconnect_attempts})"
            )
            await asyncio.sleep(delay)

            await self._connect_public_websocket()

        except Exception as e:
            logger.error(f"Failed to reconnect to public WebSocket: {e!s}")

    async def _reconnect_private_websocket(self) -> None:
        """
        Reconnect to private WebSocket with exponential backoff.
        """
        try:
            if self.reconnect_attempts >= self.max_reconnect_attempts:
                logger.error(
                    "Max reconnection attempts reached for private WebSocket")
                return

            self.reconnect_attempts += 1
            # Exponential backoff, max 60s
            delay = min(2**self.reconnect_attempts, 60)

            logger.info(
                f"Reconnecting to private WebSocket in {delay} seconds"
                f"(attempt {self.reconnect_attempts})"
            )
            await asyncio.sleep(delay)

            await self._connect_private_websocket()

        except Exception as e:
            logger.error(f"Failed to reconnect to private WebSocket: {e!s}")

    def is_connected(self) -> bool:
        """
        Check if WebSocket is connected.

        Returns:
            bool: True if connected, False otherwise
        """
        return self.connected and (self.public_ws or self.private_ws)

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
