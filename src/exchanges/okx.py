"""
OKX Exchange Implementation (P-005)

This module implements the OKX-specific exchange client with full API integration,
including REST API client, WebSocket streams, and rate limiting.

CRITICAL: This integrates with P-001 (core types, exceptions, config), P-002A (error handling),
and P-003 (base exchange interface) components.

OKX API Differences from Binance:
- Different authentication (API key + secret + passphrase)
- Unified account model (trading/funding accounts)
- Different order types and parameters
- Alternative rate limiting structure (60 requests/2 seconds per endpoint)
- Different WebSocket authentication requirements
"""

from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

# OKX-specific imports
from okx.api import Account, Market, Public, Trade as OKXTrade

from src.core.config import Config
from src.core.exceptions import (
    ExchangeConnectionError,
    ExchangeError,
    ExchangeInsufficientFundsError,
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
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class OKXExchange(BaseExchange):
    """
    OKX exchange implementation.

    Implements the unified exchange interface for OKX, providing:
    - REST API client with async support
    - WebSocket stream management
    - Rate limiting and error handling
    - Order management and balance tracking

    CRITICAL: This class must inherit from BaseExchange and implement all abstract methods.

    OKX-specific features:
    - Unified account model (trading/funding accounts)
    - Passphrase-based authentication
    - Different rate limiting structure
    - Alternative order types and parameters
    """

    def __init__(self, config: Config, exchange_name: str = "okx"):
        """
        Initialize OKX exchange.

        Args:
            config: Application configuration
            exchange_name: Exchange name (default: "okx")
        """
        super().__init__(config, exchange_name)

        # OKX-specific configuration
        self.api_key = config.exchanges.okx_api_key
        self.api_secret = config.exchanges.okx_api_secret
        self.passphrase = config.exchanges.okx_passphrase
        self.sandbox = config.exchanges.okx_sandbox

        # OKX API URLs from constants
        okx_config = API_ENDPOINTS["okx"]
        if self.sandbox:
            self.base_url = okx_config["sandbox_url"]
            self.ws_url = okx_config["ws_url"]
            self.ws_private_url = okx_config["ws_private_url"]
        else:
            self.base_url = okx_config["base_url"]
            self.ws_url = okx_config["ws_url"]
            self.ws_private_url = okx_config["ws_private_url"]

        # Initialize OKX clients
        self.account_client: Account | None = None
        self.market_client: Market | None = None
        self.trade_client: OKXTrade | None = None
        self.public_client: Public | None = None

        # Initialize rate limiter for OKX-specific limits
        self.rate_limiter = RateLimiter(config, "okx")

        # Initialize connection manager
        self.connection_manager = ConnectionManager(config, exchange_name)

        # WebSocket streams
        self.active_streams: dict[str, Any] = {}
        self.callbacks: dict[str, list[Callable]] = {}

        # Order tracking
        self.active_orders: dict[str, dict] = {}

        # Account balances cache
        self._balance_cache: dict[str, Decimal] = {}
        self._last_balance_update: datetime | None = None

        logger.info(f"Initialized OKX exchange: {exchange_name}")

    async def connect(self) -> bool:
        """
        Establish connection to OKX exchange.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info("Connecting to OKX exchange...")

            # Initialize OKX API clients
            self.account_client = Account(
                key=self.api_key,
                secret=self.api_secret,
                passphrase=self.passphrase,
                flag="0",  # 0: live trading, 1: demo trading
            )

            self.market_client = Market(
                key=self.api_key, secret=self.api_secret, passphrase=self.passphrase, flag="0"
            )

            self.trade_client = OKXTrade(
                key=self.api_key, secret=self.api_secret, passphrase=self.passphrase, flag="0"
            )

            self.public_client = Public(
                key=self.api_key, secret=self.api_secret, passphrase=self.passphrase, flag="0"
            )

            # Test connection by getting account balance
            await self.get_account_balance()

            self.connected = True
            self.status = "connected"
            self.last_heartbeat = datetime.now(timezone.utc)

            logger.info("Successfully connected to OKX exchange")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to OKX exchange: {e!s}")
            self.connected = False
            self.status = "error"
            raise ExchangeConnectionError(f"Failed to connect to OKX: {e!s}")

    async def disconnect(self) -> None:
        """Disconnect from OKX exchange."""
        try:
            logger.info("Disconnecting from OKX exchange...")

            # Close WebSocket connections
            for stream_name in list(self.active_streams.keys()):
                await self._close_stream(stream_name)

            # Clear clients
            self.account_client = None
            self.market_client = None
            self.trade_client = None
            self.public_client = None

            self.connected = False
            self.status = "disconnected"

            logger.info("Successfully disconnected from OKX exchange")

        except Exception as e:
            logger.error(f"Error disconnecting from OKX exchange: {e!s}")
            raise ExchangeConnectionError(f"Error disconnecting from OKX: {e!s}")

    async def get_account_balance(self) -> dict[str, Decimal]:
        """
        Get all asset balances from OKX exchange.

        Returns:
            Dict[str, Decimal]: Dictionary mapping asset symbols to balances
        """
        try:
            if not self.account_client:
                raise ExchangeConnectionError("OKX account client not initialized")

            # Get account balance from OKX
            result = self.account_client.get_balance()

            if result.get("code") != "0":
                raise ExchangeError(
                    f"Failed to get account balance: {result.get('msg', 'Unknown error')}"
                )

            balances = {}
            data = result.get("data", [])

            for account in data:
                currency = account.get("ccy", "")
                available = Decimal(account.get("availBal", "0"))
                frozen = Decimal(account.get("frozenBal", "0"))
                total = available + frozen

                if total > 0:
                    balances[currency] = total

            # Update cache
            self._balance_cache = balances
            self._last_balance_update = datetime.now(timezone.utc)

            logger.info(f"Retrieved {len(balances)} asset balances from OKX")
            return balances

        except Exception as e:
            logger.error(f"Failed to get account balance from OKX: {e!s}")
            raise ExchangeError(f"Failed to get account balance from OKX: {e!s}")

    @time_execution
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """
        Execute a trade order on OKX exchange.

        Args:
            order: Order request with all necessary details

        Returns:
            OrderResponse: Order response with execution details

        Raises:
            ExchangeError: If order placement fails
            ValidationError: If order request is invalid
        """
        try:
            if not self.trade_client:
                raise ExchangeConnectionError("OKX trade client not initialized")

            # Validate order request
            if not await self.pre_trade_validation(order):
                raise ValidationError("Order validation failed")

            # Convert order to OKX format
            okx_order = self._convert_order_to_okx(order)

            # Place order on OKX
            result = self.trade_client.place_order(**okx_order)

            if result.get("code") != "0":
                error_msg = result.get("msg", "Unknown error")
                if "insufficient" in error_msg.lower():
                    raise ExchangeInsufficientFundsError(f"Insufficient funds: {error_msg}")
                else:
                    raise ExchangeError(f"Order placement failed: {error_msg}")

            # Convert response to unified format
            order_response = self._convert_okx_order_to_response(result.get("data", [{}])[0])

            # Track active order
            self.active_orders[order_response.id] = {
                "order": order,
                "response": order_response,
                "timestamp": datetime.now(timezone.utc),
            }

            # Post-trade processing
            await self.post_trade_processing(order_response)

            logger.info(f"Successfully placed order on OKX: {order_response.id}")
            return order_response

        except Exception as e:
            logger.error(f"Failed to place order on OKX: {e!s}")
            if isinstance(e, (ExchangeError, ValidationError)):
                raise
            raise ExchangeError(f"Failed to place order on OKX: {e!s}")

    @time_execution
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order on OKX exchange.

        Args:
            order_id: ID of the order to cancel

        Returns:
            bool: True if cancellation successful, False otherwise
        """
        try:
            if not self.trade_client:
                raise ExchangeConnectionError("OKX trade client not initialized")

            # Cancel order on OKX
            result = self.trade_client.cancel_order(ordId=order_id)

            if result.get("code") != "0":
                logger.warning(
                    f"Failed to cancel order {order_id}: {result.get('msg', 'Unknown error')}"
                )
                return False

            # Remove from active orders
            if order_id in self.active_orders:
                del self.active_orders[order_id]

            logger.info(f"Successfully cancelled order on OKX: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id} on OKX: {e!s}")
            return False

    @time_execution
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Check the status of an order on OKX exchange.

        Args:
            order_id: ID of the order to check

        Returns:
            OrderStatus: Current status of the order
        """
        try:
            if not self.trade_client:
                raise ExchangeConnectionError("OKX trade client not initialized")

            # Get order status from OKX
            result = self.trade_client.get_order_details(ordId=order_id)

            if result.get("code") != "0":
                logger.warning(
                    f"Failed to get order status for {order_id}: {
                        result.get('msg', 'Unknown error')
                    }"
                )
                return OrderStatus.UNKNOWN

            data = result.get("data", [{}])[0]
            status = data.get("state", "")

            return self._convert_okx_status_to_order_status(status)

        except Exception as e:
            logger.error(f"Failed to get order status for {order_id} on OKX: {e!s}")
            return OrderStatus.UNKNOWN

    @time_execution
    async def get_market_data(self, symbol: str, timeframe: str = "1m") -> MarketData:
        """
        Get OHLCV market data for a symbol from OKX.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USDT')
            timeframe: Timeframe for the data (e.g., '1m', '1H', '1D')

        Returns:
            MarketData: Market data with OHLCV information
        """
        try:
            if not self.public_client:
                raise ExchangeConnectionError("OKX public client not initialized")

            # Convert timeframe to OKX format
            okx_timeframe = self._convert_timeframe_to_okx(timeframe)

            # Get candlestick data from OKX
            result = self.public_client.get_candlesticks(instId=symbol, bar=okx_timeframe, limit=1)

            if result.get("code") != "0":
                raise ExchangeError(
                    f"Failed to get market data: {result.get('msg', 'Unknown error')}"
                )

            data = result.get("data", [])
            if not data:
                raise ExchangeError(f"No market data available for {symbol}")

            # Parse candlestick data
            candle = data[0]
            timestamp = datetime.fromtimestamp(int(candle[0]) / 1000, tz=timezone.utc)

            market_data = MarketData(
                symbol=symbol,
                price=Decimal(candle[4]),  # Close price
                volume=Decimal(candle[5]),
                timestamp=timestamp,
                open_price=Decimal(candle[1]),
                high_price=Decimal(candle[2]),
                low_price=Decimal(candle[3]),
            )

            logger.debug(f"Retrieved market data for {symbol}: {market_data.price}")
            return market_data

        except Exception as e:
            logger.error(f"Failed to get market data for {symbol} from OKX: {e!s}")
            raise ExchangeError(f"Failed to get market data from OKX: {e!s}")

    async def subscribe_to_stream(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to real-time data stream for a symbol.

        Args:
            symbol: Trading symbol to subscribe to
            callback: Callback function to handle stream data
        """
        try:
            stream_name = f"tickers.{symbol}"

            if stream_name not in self.callbacks:
                self.callbacks[stream_name] = []

            self.callbacks[stream_name].append(callback)

            # Initialize WebSocket connection if not already active
            if stream_name not in self.active_streams:
                await self._initialize_websocket(stream_name, symbol)

            logger.info(f"Subscribed to {stream_name} stream on OKX")

        except Exception as e:
            logger.error(f"Failed to subscribe to stream for {symbol} on OKX: {e!s}")
            raise ExchangeError(f"Failed to subscribe to stream on OKX: {e!s}")

    @time_execution
    async def get_order_book(self, symbol: str, depth: int = 10) -> OrderBook:
        """
        Get order book for a symbol from OKX.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USDT')
            depth: Number of order book levels to retrieve

        Returns:
            OrderBook: Order book with bids and asks
        """
        try:
            if not self.public_client:
                raise ExchangeConnectionError("OKX public client not initialized")

            # Get order book from OKX
            result = self.public_client.get_orderbook(instId=symbol, sz=depth)

            if result.get("code") != "0":
                raise ExchangeError(
                    f"Failed to get order book: {result.get('msg', 'Unknown error')}"
                )

            data = result.get("data", [{}])[0]
            bids = [[Decimal(price), Decimal(size)] for price, size in data.get("bids", [])]
            asks = [[Decimal(price), Decimal(size)] for price, size in data.get("asks", [])]

            order_book = OrderBook(
                symbol=symbol, bids=bids, asks=asks, timestamp=datetime.now(timezone.utc)
            )

            logger.debug(f"Retrieved order book for {symbol}: {len(bids)} bids, {len(asks)} asks")
            return order_book

        except Exception as e:
            logger.error(f"Failed to get order book for {symbol} from OKX: {e!s}")
            raise ExchangeError(f"Failed to get order book from OKX: {e!s}")

    @time_execution
    async def get_trade_history(self, symbol: str, limit: int = 100) -> list[Trade]:
        """
        Get recent trade history for a symbol from OKX.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USDT')
            limit: Number of trades to retrieve

        Returns:
            List[Trade]: List of recent trades
        """
        try:
            if not self.public_client:
                raise ExchangeConnectionError("OKX public client not initialized")

            # Get trade history from OKX
            result = self.public_client.get_trades(instId=symbol, limit=limit)

            if result.get("code") != "0":
                raise ExchangeError(
                    f"Failed to get trade history: {result.get('msg', 'Unknown error')}"
                )

            trades = []
            data = result.get("data", [])

            for trade_data in data:
                trade = Trade(
                    id=trade_data.get("tradeId", ""),
                    symbol=symbol,
                    side=OrderSide.BUY if trade_data.get("side") == "buy" else OrderSide.SELL,
                    quantity=Decimal(trade_data.get("sz", "0")),
                    price=Decimal(trade_data.get("px", "0")),
                    timestamp=datetime.fromtimestamp(
                        int(trade_data.get("ts", 0)) / 1000, tz=timezone.utc
                    ),
                    # OKX doesn't provide fee in trade history
                    fee=Decimal("0"),
                )
                trades.append(trade)

            logger.debug(f"Retrieved {len(trades)} trades for {symbol} from OKX")
            return trades

        except Exception as e:
            logger.error(f"Failed to get trade history for {symbol} from OKX: {e!s}")
            raise ExchangeError(f"Failed to get trade history from OKX: {e!s}")

    @time_execution
    async def get_exchange_info(self) -> ExchangeInfo:
        """
        Get exchange information from OKX.

        Returns:
            ExchangeInfo: Exchange information and capabilities
        """
        try:
            if not self.public_client:
                raise ExchangeConnectionError("OKX public client not initialized")

            # Get instruments (symbols) from OKX
            result = self.public_client.get_instruments(instType="SPOT")

            if result.get("code") != "0":
                raise ExchangeError(
                    f"Failed to get exchange info: {result.get('msg', 'Unknown error')}"
                )

            data = result.get("data", [])
            supported_symbols = [item.get("instId", "") for item in data]

            exchange_info = ExchangeInfo(
                name="OKX",
                supported_symbols=supported_symbols,
                rate_limits={
                    "requests_per_minute": 600,
                    "orders_per_second": 20,
                    "websocket_connections": 3,
                },
                features=["spot_trading", "margin_trading", "futures"],
                api_version="v5",
            )

            logger.info(f"Retrieved exchange info from OKX: {len(supported_symbols)} symbols")
            return exchange_info

        except Exception as e:
            logger.error(f"Failed to get exchange info from OKX: {e!s}")
            raise ExchangeError(f"Failed to get exchange info from OKX: {e!s}")

    @time_execution
    async def get_ticker(self, symbol: str) -> Ticker:
        """
        Get ticker information for a symbol from OKX.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USDT')

        Returns:
            Ticker: Ticker information with price and volume data
        """
        try:
            if not self.public_client:
                raise ExchangeConnectionError("OKX public client not initialized")

            # Get ticker from OKX
            result = self.public_client.get_ticker(instId=symbol)

            if result.get("code") != "0":
                raise ExchangeError(f"Failed to get ticker: {result.get('msg', 'Unknown error')}")

            data = result.get("data", [{}])[0]

            ticker = Ticker(
                symbol=symbol,
                bid=Decimal(data.get("bidPx", "0")),
                ask=Decimal(data.get("askPx", "0")),
                last_price=Decimal(data.get("last", "0")),
                volume_24h=Decimal(data.get("vol24h", "0")),
                price_change_24h=Decimal(data.get("change24h", "0")),
                timestamp=datetime.now(timezone.utc),
            )

            logger.debug(f"Retrieved ticker for {symbol} from OKX: {ticker.last_price}")
            return ticker

        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol} from OKX: {e!s}")
            raise ExchangeError(f"Failed to get ticker from OKX: {e!s}")

    def _convert_order_to_okx(self, order: OrderRequest) -> dict[str, Any]:
        """
        Convert unified order request to OKX format.

        Args:
            order: Unified order request

        Returns:
            Dict[str, Any]: OKX-formatted order parameters
        """
        okx_order = {
            "instId": order.symbol,
            "tdMode": "cash",  # Spot trading
            "side": order.side.value.lower(),
            "ordType": self._convert_order_type_to_okx(order.order_type),
            "sz": str(order.quantity),
        }

        if order.price:
            okx_order["px"] = str(order.price)

        if order.client_order_id:
            okx_order["clOrdId"] = order.client_order_id

        return okx_order

    def _convert_okx_order_to_response(self, result: dict) -> OrderResponse:
        """
        Convert OKX order response to unified format.

        Args:
            result: OKX order response data

        Returns:
            OrderResponse: Unified order response
        """
        return OrderResponse(
            id=result.get("ordId", ""),
            client_order_id=result.get("clOrdId"),
            symbol=result.get("instId", ""),
            side=OrderSide.BUY if result.get("side") == "buy" else OrderSide.SELL,
            order_type=self._convert_okx_order_type_to_unified(result.get("ordType", "")),
            quantity=Decimal(result.get("sz", "0")),
            price=Decimal(result.get("px", "0")) if result.get("px") else None,
            filled_quantity=Decimal(result.get("accFillSz", "0")),
            status=self._convert_okx_status_to_order_status(result.get("state", "")).value,
            timestamp=datetime.now(timezone.utc),
        )

    def _convert_okx_status_to_order_status(self, status: str) -> OrderStatus:
        """
        Convert OKX order status to unified OrderStatus.

        Args:
            status: OKX order status string

        Returns:
            OrderStatus: Unified order status
        """
        status_mapping = {
            "live": OrderStatus.PENDING,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
        }

        return status_mapping.get(status, OrderStatus.UNKNOWN)

    def _convert_order_type_to_okx(self, order_type: OrderType) -> str:
        """
        Convert unified order type to OKX format.

        Args:
            order_type: Unified order type

        Returns:
            str: OKX order type string
        """
        type_mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_LOSS: "conditional",
            OrderType.TAKE_PROFIT: "conditional",
        }

        return type_mapping.get(order_type, "limit")

    def _convert_okx_order_type_to_unified(self, okx_type: str) -> OrderType:
        """
        Convert OKX order type to unified format.

        Args:
            okx_type: OKX order type string

        Returns:
            OrderType: Unified order type
        """
        type_mapping = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "conditional": OrderType.STOP_LOSS,  # Default mapping
        }

        return type_mapping.get(okx_type, OrderType.LIMIT)

    def _convert_timeframe_to_okx(self, timeframe: str) -> str:
        """
        Convert unified timeframe to OKX format.

        Args:
            timeframe: Unified timeframe string

        Returns:
            str: OKX timeframe string
        """
        timeframe_mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
            "1w": "1W",
        }

        return timeframe_mapping.get(timeframe, "1m")

    async def _initialize_websocket(self, stream_name: str, symbol: str) -> None:
        """
        Initialize WebSocket connection for a stream.

        Args:
            stream_name: Name of the stream
            symbol: Trading symbol
        """
        try:
            # TODO: Implement WebSocket connection for OKX
            # This will be implemented in okx_websocket.py
            logger.info(f"Initializing WebSocket for {stream_name} on OKX")

        except Exception as e:
            logger.error(f"Failed to initialize WebSocket for {stream_name} on OKX: {e!s}")
            raise ExchangeConnectionError(f"Failed to initialize WebSocket on OKX: {e!s}")

    async def _handle_stream(self, stream_name: str, stream) -> None:
        """
        Handle incoming WebSocket stream data.

        Args:
            stream_name: Name of the stream
            stream: WebSocket stream object
        """
        try:
            # TODO: Implement stream handling for OKX
            # This will be implemented in okx_websocket.py
            logger.debug(f"Handling stream {stream_name} on OKX")

        except Exception as e:
            logger.error(f"Error handling stream {stream_name} on OKX: {e!s}")

    async def _close_stream(self, stream_name: str) -> None:
        """
        Close a WebSocket stream.

        Args:
            stream_name: Name of the stream to close
        """
        try:
            if stream_name in self.active_streams:
                # TODO: Implement stream closing for OKX
                # This will be implemented in okx_websocket.py
                del self.active_streams[stream_name]
                logger.info(f"Closed stream {stream_name} on OKX")

        except Exception as e:
            logger.error(f"Error closing stream {stream_name} on OKX: {e!s}")

    async def health_check(self) -> bool:
        """
        Perform health check on OKX exchange.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            if not self.connected:
                return False

            # Test connection by getting account balance
            await self.get_account_balance()

            self.last_heartbeat = datetime.now(timezone.utc)
            return True

        except Exception as e:
            logger.error(f"Health check failed for OKX: {e!s}")
            return False

    def get_rate_limits(self) -> dict[str, int]:
        """
        Get OKX rate limits.

        Returns:
            Dict[str, int]: Rate limits for OKX
        """
        return {"requests_per_minute": 600, "orders_per_second": 20, "websocket_connections": 3}
