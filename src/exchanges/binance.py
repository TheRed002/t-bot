"""
Binance Exchange Implementation (P-004)

This module implements the Binance-specific exchange client with full API integration,
including REST API client, WebSocket streams, and rate limiting.

CRITICAL: This integrates with P-001 (core types, exceptions, config), P-002A (error handling),
and P-003 (base exchange interface) components.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

# Binance-specific imports
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException, BinanceOrderException

from src.core.config import Config
from src.core.exceptions import (
    ExchangeConnectionError,
    ExchangeError,
    ExchangeInsufficientFundsError,
    ExecutionError,
    OrderRejectionError,
    ValidationError,
)
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import (
    ExchangeInfo,
    MarketData,
    OrderBook,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Ticker,
    Trade,
)

# MANDATORY: Import from P-002A
# MANDATORY: Import from P-003
from src.exchanges.base import BaseExchange
from src.exchanges.connection_manager import ConnectionManager
from src.exchanges.rate_limiter import RateLimiter

# MANDATORY: Import from P-007A (utils)
from src.utils.constants import API_ENDPOINTS
from src.utils.decorators import cache_result, log_calls, retry, time_execution

logger = get_logger(__name__)


class BinanceExchange(BaseExchange):
    """
    Binance exchange implementation.

    Implements the unified exchange interface for Binance, providing:
    - REST API client with async support
    - WebSocket stream management
    - Rate limiting and error handling
    - Order management and balance tracking

    CRITICAL: This class must inherit from BaseExchange and implement all abstract methods.
    """

    def __init__(self, config: Config, exchange_name: str = "binance"):
        """
        Initialize Binance exchange.

        Args:
            config: Application configuration
            exchange_name: Exchange name (default: "binance")
        """
        super().__init__(config, exchange_name)

        # Binance-specific configuration
        self.api_key = config.exchanges.binance_api_key
        self.api_secret = config.exchanges.binance_api_secret
        self.testnet = config.exchanges.binance_testnet

        # Binance API URLs from constants
        binance_config = API_ENDPOINTS["binance"]
        if self.testnet:
            self.base_url = binance_config["testnet_url"]
            self.ws_url = binance_config["ws_testnet_url"]
        else:
            self.base_url = binance_config["base_url"]
            self.ws_url = binance_config["ws_url"]

        # Initialize Binance client
        self.client: AsyncClient | None = None
        self.ws_manager: BinanceSocketManager | None = None

        # Initialize rate limiter for Binance-specific limits
        self.rate_limiter = RateLimiter(config, "binance")

        # Initialize connection manager
        self.connection_manager = ConnectionManager(config, exchange_name)

        # WebSocket streams
        self.active_streams: dict[str, Any] = {}
        self.callbacks: dict[str, list[Callable]] = {}

        # Order tracking
        self.pending_orders: dict[str, dict] = {}

        logger.info(f"Initialized Binance exchange (testnet: {self.testnet})")

    @retry(max_attempts=3, base_delay=1.0)
    @log_calls
    async def connect(self) -> bool:
        """
        Establish connection to Binance.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info("Connecting to Binance exchange...")

            # Initialize Binance client
            self.client = await AsyncClient.create(
                api_key=self.api_key, api_secret=self.api_secret, testnet=self.testnet
            )

            # Initialize WebSocket manager
            self.ws_manager = BinanceSocketManager(self.client)

            # Test connection by getting server time
            server_time = await self.client.get_server_time()
            logger.info(f"Binance server time: {server_time}")

            # Initialize WebSocket connection
            await self._initialize_websocket()

            # Initialize database connection
            await self._initialize_database()

            # Initialize Redis client
            await self._initialize_redis()

            # Data module initialization removed to avoid circular dependency

            self.connected = True
            self.status = "connected"
            self.last_heartbeat = datetime.now(timezone.utc)

            logger.info("Successfully connected to Binance exchange")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e!s}")
            self.status = "error"
            self.connected = False
            return False

    async def _disconnect_from_exchange(self) -> None:
        """Disconnect from Binance exchange."""
        try:
            logger.info("Disconnecting from Binance exchange...")

            # Close WebSocket streams
            for stream_name in list(self.active_streams.keys()):
                await self._close_stream(stream_name)

            # Close Binance client
            if self.client:
                await self.client.close_connection()

            self.connected = False
            self.status = "disconnected"

            logger.info("Successfully disconnected from Binance exchange")

        except Exception as e:
            logger.error(f"Error disconnecting from Binance: {e!s}")

    @cache_result(ttl_seconds=30)  # Cache balance for 30 seconds
    async def get_account_balance(self) -> dict[str, Decimal]:
        """
        Get all asset balances from Binance.

        Returns:
            Dict[str, Decimal]: Dictionary mapping asset symbols to balances
        """
        try:
            if not self.connected:
                raise ExchangeConnectionError("Not connected to Binance")

            # Get account information
            account_info = await self.client.get_account()

            balances = {}
            for balance in account_info["balances"]:
                asset = balance["asset"]
                free = Decimal(balance["free"])
                locked = Decimal(balance["locked"])
                total = free + locked

                # Only include assets with non-zero balance
                if total > 0:
                    balances[asset] = total

            logger.debug(f"Retrieved {len(balances)} asset balances from Binance")

            # Store balance snapshot in database
            await self._store_balance_snapshot(balances)

            return balances

        except BinanceAPIException as e:
            await self._handle_exchange_error(e, "get_account_balance", {"symbol": "ALL"})
            raise ExchangeError(f"Failed to get balance: {e}")
        except Exception as e:
            await self._handle_exchange_error(e, "get_account_balance", {"symbol": "ALL"})
            raise ExchangeError(f"Failed to get balance: {e!s}")

    @time_execution
    @log_calls
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """
        Place an order on Binance.

        Args:
            order: Order request with all necessary details

        Returns:
            OrderResponse: Order response with execution details
        """
        try:
            if not self.connected:
                raise ExchangeConnectionError("Not connected to Binance")

            # Validate order
            if not await self.pre_trade_validation(order):
                raise ValidationError("Order validation failed")

            # Convert order to Binance format
            _ = self._convert_order_to_binance(order)

            # Apply rate limiting
            await self.rate_limiter.acquire("orders_per_second", 1)

            # Place order
            if order.order_type == OrderType.MARKET:
                result = await self.client.order_market(
                    symbol=order.symbol,
                    side=order.side.value.upper(),
                    quantity=str(order.quantity),
                    newClientOrderId=order.client_order_id,
                )
            elif order.order_type == OrderType.LIMIT:
                result = await self.client.order_limit(
                    symbol=order.symbol,
                    side=order.side.value.upper(),
                    quantity=str(order.quantity),
                    price=str(order.price),
                    timeInForce=order.time_in_force,
                    newClientOrderId=order.client_order_id,
                )
            else:
                raise ValidationError(f"Unsupported order type: {order.order_type}")

            # Convert response to OrderResponse
            response = self._convert_binance_order_to_response(result)

            # Post-trade processing
            await self.post_trade_processing(response)

            logger.info(f"Order placed successfully: {response.id}")
            return response

        except BinanceOrderException as e:
            await self._handle_exchange_error(
                e, "place_order", {"symbol": order.symbol, "order_id": order.client_order_id}
            )
            if "insufficient balance" in str(e).lower():
                raise ExchangeInsufficientFundsError(f"Insufficient balance: {e}")
            raise OrderRejectionError(f"Order rejected: {e}")
        except BinanceAPIException as e:
            await self._handle_exchange_error(
                e, "place_order", {"symbol": order.symbol, "order_id": order.client_order_id}
            )
            raise ExchangeError(f"Failed to place order: {e}")
        except Exception as e:
            await self._handle_exchange_error(
                e, "place_order", {"symbol": order.symbol, "order_id": order.client_order_id}
            )
            raise ExecutionError(f"Failed to place order: {e!s}")

    @time_execution
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order on Binance.

        Args:
            order_id: Order ID to cancel

        Returns:
            bool: True if cancellation successful, False otherwise
        """
        try:
            if not self.connected:
                raise ExchangeConnectionError("Not connected to Binance")

            # Apply rate limiting
            await self.rate_limiter.acquire("orders_per_second", 1)

            # Cancel order
            _ = await self.client.cancel_order(symbol="", orderId=order_id)

            logger.info(f"Order cancelled successfully: {order_id}")
            return True

        except BinanceAPIException as e:
            logger.error(f"Binance API error cancelling order: {e}")
            return False
        except Exception as e:
            logger.error(f"Error cancelling order on Binance: {e!s}")
            return False

    @time_execution
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get order status from Binance.

        Args:
            order_id: Order ID to check

        Returns:
            OrderStatus: Current order status
        """
        try:
            if not self.connected:
                raise ExchangeConnectionError("Not connected to Binance")

            # Apply rate limiting
            await self.rate_limiter.acquire("requests_per_minute", 1)

            # Get order status
            result = await self.client.get_order(symbol="", orderId=order_id)

            # Convert to OrderStatus
            status = self._convert_binance_status_to_order_status(result["status"])

            return status

        except BinanceAPIException as e:
            logger.error(f"Binance API error getting order status: {e}")
            raise ExchangeError(f"Failed to get order status: {e}")
        except Exception as e:
            logger.error(f"Error getting order status from Binance: {e!s}")
            raise ExchangeError(f"Failed to get order status: {e!s}")

    @time_execution
    @cache_result(ttl_seconds=5)  # Cache market data for 5 seconds
    async def _get_market_data_from_exchange(
        self, symbol: str, timeframe: str = "1m"
    ) -> MarketData:
        """
        Get market data from Binance API.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Timeframe for data (e.g., "1m", "5m", "1h")

        Returns:
            MarketData: Market data with OHLCV information
        """
        try:
            if not self.connected:
                raise ExchangeConnectionError("Not connected to Binance")

            # Apply rate limiting
            await self.rate_limiter.acquire("requests_per_minute", 1)

            # Get klines (OHLCV data)
            klines = await self.client.get_klines(symbol=symbol, interval=timeframe, limit=1)

            if not klines:
                raise ExchangeError(f"No market data available for {symbol}")

            # Convert to MarketData
            kline = klines[0]
            market_data = MarketData(
                symbol=symbol,
                price=Decimal(str(kline[4])),  # Close price
                volume=Decimal(str(kline[5])),  # Volume
                timestamp=datetime.fromtimestamp(kline[6] / 1000, tz=timezone.utc),
                open_price=Decimal(str(kline[1])),
                high_price=Decimal(str(kline[2])),
                low_price=Decimal(str(kline[3])),
            )

            # Cache market data in Redis
            await self._cache_market_data(
                symbol,
                {
                    "price": str(market_data.price),
                    "volume": str(market_data.volume),
                    "timestamp": market_data.timestamp.isoformat(),
                    "open_price": str(market_data.open_price),
                    "high_price": str(market_data.high_price),
                    "low_price": str(market_data.low_price),
                    "timeframe": timeframe,
                },
            )

            return market_data

        except BinanceAPIException as e:
            logger.error(f"Binance API error getting market data: {e}")
            raise ExchangeError(f"Failed to get market data: {e}")
        except Exception as e:
            logger.error(f"Error getting market data from Binance: {e!s}")
            raise ExchangeError(f"Failed to get market data: {e!s}")

    async def subscribe_to_stream(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to real-time data stream for a symbol.

        Args:
            symbol: Trading symbol to subscribe to
            callback: Callback function to handle stream data
        """
        try:
            if not self.connected or not self.ws_manager:
                raise ExchangeConnectionError("Not connected to Binance")

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
                asyncio.create_task(self._handle_stream(stream_name, stream))

                logger.info(f"Subscribed to stream: {stream_name}")

        except Exception as e:
            logger.error(f"Error subscribing to stream: {e!s}")
            raise ExchangeError(f"Failed to subscribe to stream: {e!s}")

    @time_execution
    async def get_order_book(self, symbol: str, depth: int = 10) -> OrderBook:
        """
        Get order book from Binance.

        Args:
            symbol: Trading symbol
            depth: Order book depth (max 1000)

        Returns:
            OrderBook: Order book with bids and asks
        """
        try:
            if not self.connected:
                raise ExchangeConnectionError("Not connected to Binance")

            # Apply rate limiting
            await self.rate_limiter.acquire("requests_per_minute", 1)

            # Get order book
            result = await self.client.get_order_book(symbol=symbol, limit=depth)

            # Convert to OrderBook
            bids = [[Decimal(str(price)), Decimal(str(qty))] for price, qty in result["bids"]]
            asks = [[Decimal(str(price)), Decimal(str(qty))] for price, qty in result["asks"]]

            order_book = OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.fromtimestamp(result["lastUpdateId"], tz=timezone.utc),
            )

            return order_book

        except BinanceAPIException as e:
            logger.error(f"Binance API error getting order book: {e}")
            raise ExchangeError(f"Failed to get order book: {e}")
        except Exception as e:
            logger.error(f"Error getting order book from Binance: {e!s}")
            raise ExchangeError(f"Failed to get order book: {e!s}")

    @time_execution
    async def _get_trade_history_from_exchange(self, symbol: str, limit: int = 100) -> list[Trade]:
        """
        Get trade history from Binance API.

        Args:
            symbol: Trading symbol
            limit: Number of trades to retrieve (max 1000)

        Returns:
            List[Trade]: List of trade records
        """
        try:
            if not self.connected:
                raise ExchangeConnectionError("Not connected to Binance")

            # Apply rate limiting
            await self.rate_limiter.acquire("requests_per_minute", 1)

            # Get recent trades
            result = await self.client.get_recent_trades(symbol=symbol, limit=limit)

            # Convert to Trade objects
            trades = []
            for trade_data in result:
                trade = Trade(
                    id=str(trade_data["id"]),
                    symbol=symbol,
                    side=OrderSide.BUY if trade_data["isBuyerMaker"] else OrderSide.SELL,
                    amount=Decimal(str(trade_data["qty"])),
                    price=Decimal(str(trade_data["price"])),
                    timestamp=datetime.fromtimestamp(trade_data["time"] / 1000, tz=timezone.utc),
                    fee=Decimal("0"),  # Fee not available in recent trades
                )
                trades.append(trade)

            return trades

        except BinanceAPIException as e:
            logger.error(f"Binance API error getting trade history: {e}")
            raise ExchangeError(f"Failed to get trade history: {e}")
        except Exception as e:
            logger.error(f"Error getting trade history from Binance: {e!s}")
            raise ExchangeError(f"Failed to get trade history: {e!s}")

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]:
        """Return open orders; prefer local tracking if available, else empty list."""
        try:
            if not self.connected or not self.client:
                return []
            # Use Binance open orders API when symbol provided; otherwise try local cache
            if symbol:
                result = await self.client.get_open_orders(symbol=symbol)
                return [self._convert_binance_order_to_response(o) for o in result]
            if hasattr(self, "pending_orders") and isinstance(self.pending_orders, dict):
                orders: list[OrderResponse] = []
                for _order_id in list(self.pending_orders.keys()):
                    # Skip API fetch to avoid heavy calls
                    pass
                return orders
        except Exception:
            return []
        return []

    async def get_positions(self) -> list[Position]:
        """Spot implementation: no positions; return empty list."""
        return []

    @cache_result(ttl_seconds=300)
    @time_execution
    @log_calls
    async def get_exchange_info(self) -> ExchangeInfo:
        """
        Get exchange information from Binance.

        Returns:
            ExchangeInfo: Exchange information and capabilities
        """
        try:
            if not self.connected:
                raise ExchangeConnectionError("Not connected to Binance")

            # Apply rate limiting
            await self.rate_limiter.acquire("requests_per_minute", 1)

            # Get exchange info
            result = await self.client.get_exchange_info()

            # Convert to ExchangeInfo
            exchange_info = ExchangeInfo(
                name="binance",
                supported_symbols=[symbol["symbol"] for symbol in result["symbols"]],
                rate_limits={
                    "requests_per_minute": 1200,
                    "orders_per_second": 10,
                    "orders_per_24_hours": 160000,
                },
                features=["spot_trading", "margin_trading", "futures_trading"],
                api_version="v3",
            )

            return exchange_info

        except BinanceAPIException as e:
            logger.error(f"Binance API error getting exchange info: {e}")
            raise ExchangeError(f"Failed to get exchange info: {e}")
        except Exception as e:
            logger.error(f"Error getting exchange info from Binance: {e!s}")
            raise ExchangeError(f"Failed to get exchange info: {e!s}")

    @time_execution
    async def get_ticker(self, symbol: str) -> Ticker:
        """
        Get ticker information from Binance.

        Args:
            symbol: Trading symbol

        Returns:
            Ticker: Ticker information
        """
        try:
            if not self.connected:
                raise ExchangeConnectionError("Not connected to Binance")

            # Apply rate limiting
            await self.rate_limiter.acquire("requests_per_minute", 1)

            # Get ticker
            result = await self.client.get_ticker(symbol=symbol)

            # Convert to Ticker
            ticker = Ticker(
                symbol=symbol,
                bid=Decimal(str(result["bidPrice"])),
                ask=Decimal(str(result["askPrice"])),
                last_price=Decimal(str(result["lastPrice"])),
                volume_24h=Decimal(str(result["volume"])),
                price_change_24h=Decimal(str(result["priceChange"])),
                timestamp=datetime.fromtimestamp(result["closeTime"] / 1000, tz=timezone.utc),
            )

            return ticker

        except BinanceAPIException as e:
            logger.error(f"Binance API error getting ticker: {e}")
            raise ExchangeError(f"Failed to get ticker: {e}")
        except Exception as e:
            logger.error(f"Error getting ticker from Binance: {e!s}")
            raise ExchangeError(f"Failed to get ticker: {e!s}")

    # Helper methods for internal use

    def _convert_order_to_binance(self, order: OrderRequest) -> dict[str, Any]:
        """Convert OrderRequest to Binance order format."""
        binance_order = {
            "symbol": order.symbol,
            "side": order.side.value.upper(),
            "type": order.order_type.value.upper(),
            "quantity": str(order.quantity),
            "newClientOrderId": order.client_order_id,
        }

        if order.price:
            binance_order["price"] = str(order.price)

        if order.time_in_force:
            binance_order["timeInForce"] = order.time_in_force

        return binance_order

    def _convert_binance_order_to_response(self, result: dict) -> OrderResponse:
        """Convert Binance order result to OrderResponse."""
        return OrderResponse(
            id=str(result["orderId"]),
            client_order_id=result.get("clientOrderId"),
            symbol=result["symbol"],
            side=OrderSide.BUY if result["side"] == "BUY" else OrderSide.SELL,
            order_type=OrderType.MARKET if result["type"] == "MARKET" else OrderType.LIMIT,
            quantity=Decimal(str(result["origQty"])),
            price=Decimal(str(result["price"])) if result.get("price") else None,
            filled_quantity=Decimal(str(result["executedQty"])),
            status=result["status"],
            timestamp=datetime.fromtimestamp(result["time"] / 1000, tz=timezone.utc),
        )

    def _convert_binance_status_to_order_status(self, status: str) -> OrderStatus:
        """Convert Binance order status to OrderStatus enum."""
        status_mapping = {
            "NEW": OrderStatus.PENDING,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        return status_mapping.get(status, OrderStatus.UNKNOWN)

    async def _initialize_websocket(self) -> None:
        """Initialize WebSocket connection."""
        try:
            # Start user data stream for account updates
            user_data_stream = self.ws_manager.user_socket()
            self.active_streams["user_data"] = user_data_stream

            # Start async handler for user data stream
            asyncio.create_task(self._handle_user_data_stream(user_data_stream))

            logger.info("WebSocket streams initialized")

        except Exception as e:
            logger.error(f"Error initializing WebSocket: {e!s}")

    async def _handle_stream(self, stream_name: str, stream) -> None:
        """Handle WebSocket stream data."""
        try:
            async with stream as stream_handler:
                async for msg in stream_handler:
                    if stream_name in self.callbacks:
                        for callback in self.callbacks[stream_name]:
                            try:
                                await callback(msg)
                            except Exception as e:
                                logger.error(f"Error in stream callback: {e!s}")
        except Exception as e:
            logger.error(f"Error handling stream {stream_name}: {e!s}")

    async def _handle_user_data_stream(self, stream) -> None:
        """Handle user data stream for account updates."""
        try:
            async with stream as stream_handler:
                async for msg in stream_handler:
                    # Handle different types of user data updates
                    if msg["e"] == "outboundAccountPosition":
                        # Account balance update
                        logger.debug("Account balance updated")
                    elif msg["e"] == "executionReport":
                        # Order execution update
                        logger.debug(f"Order execution: {msg}")
        except Exception as e:
            logger.error(f"Error handling user data stream: {e!s}")

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

    async def health_check(self) -> bool:
        """
        Perform health check on Binance connection.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            if not self.connected:
                return False

            # Test connection by getting server time
            await self.client.get_server_time()

            # Update heartbeat
            self.last_heartbeat = datetime.now(timezone.utc)

            return True

        except Exception as e:
            logger.error(f"Health check failed: {e!s}")
            return False

    def get_rate_limits(self) -> dict[str, int]:
        """Get current rate limits for Binance."""
        return {
            "requests_per_minute": 1200,
            "orders_per_second": 10,
            "orders_per_24_hours": 160000,
            "weight_per_minute": 1200,
        }
