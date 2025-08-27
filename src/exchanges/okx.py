"""
Enhanced OKX Exchange Implementation

Refactored implementation using the unified infrastructure from base.py.
This eliminates all duplication and leverages:
- Unified connection pooling and session management
- Advanced rate limiting with local and global enforcement
- Unified WebSocket management with auto-reconnection
- Comprehensive error handling and recovery
- Market data caching and optimization
- Health monitoring and automatic recovery
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

# OKX-specific imports
from okx.api import Account, Market, Public, Trade as OKXTrade

from src.core.config import Config
from src.core.exceptions import (
    ExchangeConnectionError,
    ExchangeError,
    ExchangeErrorMapper,
    ExchangeInsufficientFundsError,
    ExchangeRateLimitError,
    ExecutionError,
    OrderRejectionError,
    ValidationError,
)

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

# MANDATORY: Import enhanced base
from src.exchanges.base import EnhancedBaseExchange

# MANDATORY: Import from P-007A (utils)
from src.utils import API_ENDPOINTS


class OKXExchange(EnhancedBaseExchange):
    """
    Enhanced OKX exchange implementation with unified infrastructure.

    Provides complete OKX API integration with:
    - All common functionality inherited from EnhancedBaseExchange
    - OKX-specific API implementations with passphrase authentication
    - Automatic error mapping and handling
    - Optimized connection and WebSocket management
    """

    def __init__(
        self,
        config: Config,
        exchange_name: str = "okx",
        state_service: Any | None = None,
        trade_lifecycle_manager: Any | None = None,
        metrics_collector: Any | None = None,
    ):
        """
        Initialize enhanced OKX exchange.

        Args:
            config: Application configuration
            exchange_name: Exchange name (default: "okx")
            state_service: Optional state service for persistence
            trade_lifecycle_manager: Optional trade lifecycle manager
        """
        super().__init__(config, exchange_name, state_service, trade_lifecycle_manager, metrics_collector)

        # OKX-specific configuration
        self.api_key = config.exchange.okx_api_key
        self.api_secret = config.exchange.okx_api_secret
        self.passphrase = config.exchange.okx_passphrase
        self.sandbox = config.exchange.okx_testnet  # Use okx_testnet, not okx_sandbox

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

        # OKX-specific clients (will be initialized in connect)
        self.account_client: Account | None = None
        self.market_client: Market | None = None
        self.trade_client: OKXTrade | None = None
        self.public_client: Public | None = None

        self.logger.info(f"Enhanced OKX exchange initialized (sandbox: {self.sandbox})")

    # === ENHANCED BASE IMPLEMENTATION ===

    async def _connect_to_exchange(self) -> bool:
        """
        OKX-specific connection logic.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.logger.info("Establishing OKX API connection...")

            # Initialize OKX API clients
            self.account_client = Account(
                key=self.api_key,
                secret=self.api_secret,
                passphrase=self.passphrase,
                flag="0" if not self.sandbox else "1",  # 0: live trading, 1: demo trading
            )

            self.market_client = Market(
                key=self.api_key,
                secret=self.api_secret,
                passphrase=self.passphrase,
                flag="0" if not self.sandbox else "1",
            )

            self.trade_client = OKXTrade(
                key=self.api_key,
                secret=self.api_secret,
                passphrase=self.passphrase,
                flag="0" if not self.sandbox else "1",
            )

            self.public_client = Public(
                key=self.api_key,
                secret=self.api_secret,
                passphrase=self.passphrase,
                flag="0" if not self.sandbox else "1",
            )

            # Test connection by getting account balance
            await self._test_okx_connection()

            self.logger.info("OKX API connection established successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to establish OKX connection: {e!s}")
            await self._handle_okx_error(e, "connect")
            return False

    async def _test_okx_connection(self) -> None:
        """Test OKX connection by making a simple API call."""
        try:
            # Test with account balance call
            result = self.account_client.get_balance()

            # OKX API returns status code in response
            if result.get("code") != "0":
                error_msg = result.get("msg", "Unknown error")
                raise ExchangeError(f"OKX connection test failed: {error_msg}")

            # Also test public API
            server_time = self.public_client.get_system_time()
            if server_time.get("code") != "0":
                error_msg = server_time.get("msg", "Unknown error")
                raise ExchangeError(f"OKX public API test failed: {error_msg}")

            self.logger.info(
                f"OKX connection test successful. Server time: {server_time.get('data', [{}])[0].get('ts', 'N/A')}"
            )

        except Exception as e:
            raise ExchangeConnectionError(f"OKX connection test failed: {e!s}")

    async def _disconnect_from_exchange(self) -> None:
        """OKX-specific disconnection logic."""
        try:
            self.logger.info("Closing OKX connections...")

            # Clear OKX clients
            self.account_client = None
            self.market_client = None
            self.trade_client = None
            self.public_client = None

            self.logger.info("OKX connections closed successfully")

        except Exception as e:
            self.logger.error(f"Error closing OKX connections: {e!s}")

    async def _create_websocket_stream(self, symbol: str, stream_name: str) -> Any:
        """
        Create OKX-specific WebSocket stream.

        Args:
            symbol: Trading symbol
            stream_name: Name for the stream

        Returns:
            WebSocket connection object
        """
        try:
            # OKX WebSocket implementation would go here
            # For now, return a placeholder
            self.logger.debug(f"Created OKX WebSocket stream for {symbol}")
            return {"symbol": symbol, "stream_name": stream_name, "type": "okx_stream"}

        except Exception as e:
            self.logger.error(f"Failed to create OKX WebSocket stream for {symbol}: {e!s}")
            return None

    async def _handle_exchange_stream(self, stream_name: str, stream: Any) -> None:
        """
        Handle OKX-specific WebSocket stream messages.

        Args:
            stream_name: Name of the stream
            stream: Stream connection object
        """
        try:
            # OKX-specific stream handling would go here
            self.logger.debug(f"Handling OKX stream {stream_name}")

        except Exception as e:
            self.logger.error(f"Error handling OKX stream {stream_name}: {e!s}")
            raise

    async def _close_exchange_stream(self, stream_name: str, stream: Any) -> None:
        """
        Close OKX-specific WebSocket stream.

        Args:
            stream_name: Name of the stream
            stream: Stream connection object
        """
        try:
            self.logger.debug(f"Closed OKX stream {stream_name}")

        except Exception as e:
            self.logger.error(f"Error closing OKX stream {stream_name}: {e!s}")

    # === EXCHANGE API IMPLEMENTATIONS ===

    async def _place_order_on_exchange(self, order: OrderRequest) -> OrderResponse:
        """
        OKX-specific order placement logic.

        Args:
            order: Order request

        Returns:
            OrderResponse: Order response
        """
        try:
            if not self.trade_client:
                raise ExchangeConnectionError("OKX trade client not initialized")

            # Validate order parameters
            await self._validate_okx_order(order)

            # Convert order to OKX format
            okx_order = self._convert_order_to_okx(order)

            # Log the order for debugging
            self.logger.debug(f"Placing OKX order: {okx_order}")

            # Place order on OKX
            result = self.trade_client.place_order(**okx_order)

            # Check if order placement was successful
            if result.get("code") != "0":
                error_msg = result.get("msg", "Unknown error")
                error_code = result.get("code", "")

                self.logger.error(f"OKX order placement failed: {error_code} - {error_msg}")

                # Map specific error types
                if "insufficient" in error_msg.lower() or "balance" in error_msg.lower():
                    raise ExchangeInsufficientFundsError(f"Insufficient funds: {error_msg}")
                elif "invalid" in error_msg.lower() or "parameter" in error_msg.lower():
                    raise ValidationError(f"Invalid order parameters: {error_msg}")
                elif "rate limit" in error_msg.lower():
                    raise ExchangeRateLimitError(f"Rate limit exceeded: {error_msg}")
                else:
                    raise OrderRejectionError(f"Order placement failed [{error_code}]: {error_msg}")

            # Check if we received order data
            order_data_list = result.get("data", [])
            if not order_data_list:
                raise ExecutionError("No order data returned from OKX")

            order_data = order_data_list[0]

            # Convert response to unified format
            order_response = self._convert_okx_order_to_response(order_data)

            self.logger.info(
                f"OKX order placed successfully: {order_response.id} for {order.symbol}"
            )
            return order_response

        except Exception as e:
            await self._handle_okx_error(
                e,
                "place_order",
                {
                    "symbol": order.symbol,
                    "order_id": order.client_order_id,
                    "order_type": order.order_type.value,
                },
            )
            raise

    async def get_account_balance(self) -> dict[str, Decimal]:
        """
        Get all asset balances from OKX.

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

            self.logger.debug(f"Retrieved {len(balances)} asset balances from OKX")

            # Store balance snapshot (handled by base class)
            await self._store_balance_snapshot(balances)

            return balances

        except Exception as e:
            await self._handle_okx_error(e, "get_account_balance")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order on OKX.

        Args:
            order_id: Order ID to cancel

        Returns:
            bool: True if cancellation successful, False otherwise
        """
        try:
            if not self.trade_client:
                raise ExchangeConnectionError("OKX trade client not initialized")

            # Cancel order on OKX (requires instrument ID)
            symbol = self._get_symbol_for_order(order_id)

            result = self.trade_client.cancel_order(ordId=order_id, instId=symbol)

            if result.get("code") != "0":
                self.logger.warning(
                    f"Failed to cancel OKX order {order_id}: {result.get('msg', 'Unknown error')}"
                )
                return False

            # Update local tracking
            if order_id in self.pending_orders:
                self.pending_orders[order_id]["status"] = "cancelled"

            self.logger.info(f"OKX order cancelled successfully: {order_id}")
            return True

        except Exception as e:
            await self._handle_okx_error(e, "cancel_order", {"order_id": order_id})
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get order status from OKX.

        Args:
            order_id: Order ID to check

        Returns:
            OrderStatus: Current order status
        """
        try:
            if not self.trade_client:
                raise ExchangeConnectionError("OKX trade client not initialized")

            # Get order status from OKX
            result = self.trade_client.get_order_details(ordId=order_id)

            if result.get("code") != "0":
                self.logger.warning(
                    f"Failed to get OKX order status for {order_id}: {result.get('msg', 'Unknown error')}"
                )
                return OrderStatus.UNKNOWN

            data = result.get("data", [{}])[0]
            status = data.get("state", "")

            # Convert to unified status
            unified_status = self._convert_okx_status_to_order_status(status)

            # Update local tracking
            if order_id in self.pending_orders:
                self.pending_orders[order_id]["status"] = unified_status.value

            return unified_status

        except Exception as e:
            await self._handle_okx_error(e, "get_order_status", {"order_id": order_id})
            return OrderStatus.UNKNOWN

    async def _get_market_data_from_exchange(
        self, symbol: str, timeframe: str = "1m"
    ) -> MarketData:
        """
        Get market data from OKX API.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USDT')
            timeframe: Timeframe for data (e.g., '1m', '1H', '1D')

        Returns:
            MarketData: Market data with OHLCV information
        """
        try:
            if not self.public_client:
                raise ExchangeConnectionError("OKX public client not initialized")

            # Convert symbol to OKX format if needed
            okx_symbol = self._convert_symbol_to_okx_format(symbol)

            # Convert timeframe to OKX format
            okx_timeframe = self._convert_timeframe_to_okx(timeframe)

            # Get ticker data
            ticker_result = self.public_client.get_ticker(instId=okx_symbol)

            if ticker_result.get("code") != "0":
                error_msg = ticker_result.get("msg", "Failed to get ticker data")
                raise ExchangeError(f"OKX ticker request failed: {error_msg}")

            if not ticker_result.get("data"):
                raise ExchangeError(f"No ticker data available for {okx_symbol}")

            ticker = ticker_result["data"][0]

            # Get OHLCV data for additional information
            kline_result = self.public_client.get_candlesticks(
                instId=okx_symbol, bar=okx_timeframe, limit=1
            )
            kline = None

            if kline_result.get("code") == "0" and kline_result.get("data"):
                klines = kline_result["data"]
                if klines:
                    kline = klines[0]

            # Parse timestamp from ticker or current time
            ticker_timestamp = ticker.get("ts")
            if ticker_timestamp:
                timestamp = datetime.fromtimestamp(int(ticker_timestamp) / 1000, tz=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)

            # Create MarketData object with proper error handling
            market_data = MarketData(
                symbol=symbol,
                price=Decimal(ticker.get("last", "0")),
                volume=Decimal(ticker.get("vol24h", "0")),
                timestamp=timestamp,
                bid=(
                    Decimal(ticker.get("bidPx", "0"))
                    if ticker.get("bidPx") and ticker.get("bidPx") != ""
                    else None
                ),
                ask=(
                    Decimal(ticker.get("askPx", "0"))
                    if ticker.get("askPx") and ticker.get("askPx") != ""
                    else None
                ),
                open_price=Decimal(kline[1]) if kline and len(kline) > 1 and kline[1] else None,
                high_price=Decimal(kline[2]) if kline and len(kline) > 2 and kline[2] else None,
                low_price=Decimal(kline[3]) if kline and len(kline) > 3 and kline[3] else None,
                volume_24h=Decimal(ticker.get("vol24h", "0")),
                quote_volume_24h=Decimal(ticker.get("volCcy24h", "0")),
                price_change_24h=(
                    Decimal(ticker.get("open24h", "0")) - Decimal(ticker.get("last", "0"))
                    if ticker.get("open24h") and ticker.get("last")
                    else Decimal("0")
                ),
            )

            return market_data

        except Exception as e:
            await self._handle_okx_error(e, "get_market_data", {"symbol": symbol})
            raise

    async def get_order_book(self, symbol: str, depth: int = 10) -> OrderBook:
        """
        Get order book from OKX.

        Args:
            symbol: Trading symbol
            depth: Order book depth

        Returns:
            OrderBook: Order book with bids and asks
        """
        try:
            if not self.public_client:
                raise ExchangeConnectionError("OKX public client not initialized")

            # Convert symbol to OKX format
            okx_symbol = self._convert_symbol_to_okx_format(symbol)

            # Get order book from OKX
            result = self.public_client.get_orderbook(instId=okx_symbol, sz=depth)

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

            return order_book

        except Exception as e:
            await self._handle_okx_error(e, "get_order_book", {"symbol": symbol})
            raise

    async def _get_trade_history_from_exchange(self, symbol: str, limit: int = 100) -> list[Trade]:
        """
        Get trade history from OKX API.

        Args:
            symbol: Trading symbol
            limit: Number of trades to retrieve

        Returns:
            List[Trade]: List of trade records
        """
        try:
            if not self.trade_client:
                raise ExchangeConnectionError("OKX trade client not initialized")

            # Convert symbol to OKX format
            okx_symbol = self._convert_symbol_to_okx_format(symbol)

            # Get trade history from account API
            trades_result = self.trade_client.get_fills(instId=okx_symbol, limit=str(limit))

            if trades_result.get("code") != "0":
                self.logger.warning(
                    f"Failed to get OKX trade history: {trades_result.get('msg', 'Unknown error')}"
                )
                return []

            trades = []
            for trade_data in trades_result.get("data", []):
                try:
                    trade = Trade(
                        id=trade_data.get("tradeId", ""),
                        order_id=trade_data.get("ordId"),
                        symbol=symbol,
                        side=OrderSide.BUY if trade_data.get("side") == "buy" else OrderSide.SELL,
                        amount=Decimal(trade_data.get("fillSz", "0")),
                        price=Decimal(trade_data.get("fillPx", "0")),
                        fee=Decimal(trade_data.get("fee", "0")),
                        fee_currency=trade_data.get("feeCcy", "USDT"),
                        timestamp=datetime.fromtimestamp(
                            int(trade_data.get("ts", "0")) / 1000, tz=timezone.utc
                        ),
                    )
                    trades.append(trade)
                except Exception as trade_error:
                    self.logger.warning(f"Failed to parse OKX trade data: {trade_error}")
                    continue

            return trades

        except Exception as e:
            await self._handle_okx_error(e, "get_trade_history", {"symbol": symbol})
            return []

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

            return exchange_info

        except Exception as e:
            await self._handle_okx_error(e, "get_exchange_info")
            raise

    async def get_ticker(self, symbol: str) -> Ticker:
        """
        Get ticker information from OKX.

        Args:
            symbol: Trading symbol

        Returns:
            Ticker: Ticker information
        """
        try:
            if not self.public_client:
                raise ExchangeConnectionError("OKX public client not initialized")

            # Convert symbol to OKX format
            okx_symbol = self._convert_symbol_to_okx_format(symbol)

            # Get ticker from OKX
            result = self.public_client.get_ticker(instId=okx_symbol)

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

            return ticker

        except Exception as e:
            await self._handle_okx_error(e, "get_ticker", {"symbol": symbol})
            raise

    # === HELPER METHODS ===

    async def _handle_okx_error(
        self, error: Exception, operation: str, context: dict | None = None
    ) -> None:
        """
        Handle OKX-specific errors using unified error mapping.

        Args:
            error: The OKX exception
            operation: Operation being performed
            context: Additional context
        """
        try:
            # Extract error data for mapping
            if hasattr(error, "code") and hasattr(error, "msg"):
                error_data = {
                    "code": getattr(error, "code", None),
                    "msg": getattr(error, "msg", str(error)),
                }
            else:
                error_data = {"msg": str(error)}

            # Map to unified exception
            unified_error = ExchangeErrorMapper.map_okx_error(error_data)

            # Handle using base class error handling
            await self._handle_exchange_error(unified_error, operation, context)

        except Exception as e:
            self.logger.error(f"Error in OKX error handling: {e!s}")

    def _convert_order_to_okx(self, order: OrderRequest) -> dict[str, Any]:
        """Convert unified order request to OKX format."""
        okx_order = {
            "instId": self._convert_symbol_to_okx_format(order.symbol),
            "tdMode": "cash",  # Spot trading (could be "isolated" for margin)
            "side": order.side.value.lower(),
            "ordType": self._convert_order_type_to_okx(order.order_type),
            "sz": str(order.quantity),
        }

        # Add price for limit orders
        if order.price and order.order_type == OrderType.LIMIT:
            okx_order["px"] = str(order.price)

        # Add stop price for stop orders
        if order.stop_price and order.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
            okx_order["stopPx"] = str(order.stop_price)

        # Add time in force
        if order.time_in_force:
            if order.time_in_force == "IOC":
                okx_order["tgtCcy"] = "base_ccy"  # Default for IOC
            elif order.time_in_force == "FOK":
                okx_order["ordType"] = "fok"  # FOK is a separate order type in OKX

        # Add client order ID
        if order.client_order_id:
            okx_order["clOrdId"] = order.client_order_id

        # Add reduce-only flag if specified (for derivatives)
        if hasattr(order, "reduce_only") and order.reduce_only:
            okx_order["reduceOnly"] = "true"

        return okx_order

    def _convert_okx_order_to_response(self, result: dict) -> OrderResponse:
        """Convert OKX order response to unified format."""
        return OrderResponse(
            id=result.get("ordId", ""),
            client_order_id=result.get("clOrdId"),
            symbol=self._convert_symbol_from_okx_format(result.get("instId", "")),
            side=OrderSide.BUY if result.get("side") == "buy" else OrderSide.SELL,
            order_type=self._convert_okx_order_type_to_unified(result.get("ordType", "")),
            quantity=Decimal(result.get("sz", "0")),
            price=Decimal(result.get("px", "0")) if result.get("px") else None,
            filled_quantity=Decimal(result.get("accFillSz", "0")),
            status=self._convert_okx_status_to_order_status(result.get("state", "")).value,
            timestamp=datetime.now(timezone.utc),
        )

    def _convert_okx_status_to_order_status(self, status: str) -> OrderStatus:
        """Convert OKX order status to unified OrderStatus."""
        status_mapping = {
            "live": OrderStatus.PENDING,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
        }
        return status_mapping.get(status, OrderStatus.UNKNOWN)

    def _convert_order_type_to_okx(self, order_type: OrderType) -> str:
        """Convert unified order type to OKX format."""
        type_mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_LOSS: "conditional",
            OrderType.TAKE_PROFIT: "conditional",
        }
        return type_mapping.get(order_type, "limit")

    def _convert_okx_order_type_to_unified(self, okx_type: str) -> OrderType:
        """Convert OKX order type to unified format."""
        type_mapping = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "conditional": OrderType.STOP_LOSS,  # Default mapping
        }
        return type_mapping.get(okx_type, OrderType.LIMIT)

    def _convert_timeframe_to_okx(self, timeframe: str) -> str:
        """Convert unified timeframe to OKX format."""
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

    def _convert_symbol_to_okx_format(self, symbol: str) -> str:
        """Convert symbol to OKX format."""
        # OKX uses dash-separated format like BTC-USDT
        if "-" in symbol:
            return symbol  # Already in correct format

        # Convert from formats like BTCUSDT to BTC-USDT
        symbol_mappings = {
            "BTCUSDT": "BTC-USDT",
            "ETHUSDT": "ETH-USDT",
            "BNBUSDT": "BNB-USDT",
            "ADAUSDT": "ADA-USDT",
            "DOTUSDT": "DOT-USDT",
            "LINKUSDT": "LINK-USDT",
            "LTCUSDT": "LTC-USDT",
            "SOLUSDT": "SOL-USDT",
            "XRPUSDT": "XRP-USDT",
        }

        if symbol in symbol_mappings:
            return symbol_mappings[symbol]

        # Generic conversion for other symbols
        if len(symbol) >= 6:
            if symbol.endswith("USDT"):
                base = symbol[:-4]
                return f"{base}-USDT"
            elif symbol.endswith("USDC"):
                base = symbol[:-4]
                return f"{base}-USDC"
            elif symbol.endswith("BTC"):
                base = symbol[:-3]
                return f"{base}-BTC"
            elif symbol.endswith("ETH"):
                base = symbol[:-3]
                return f"{base}-ETH"
            elif symbol.endswith("USD"):
                base = symbol[:-3]
                return f"{base}-USD"

        return symbol  # Return as-is if no conversion found

    def _convert_symbol_from_okx_format(self, okx_symbol: str) -> str:
        """Convert OKX symbol format to standard format."""
        # Convert BTC-USDT to BTCUSDT
        return okx_symbol.replace("-", "")

    def _get_symbol_for_order(self, order_id: str) -> str:
        """Get symbol for order from local tracking."""
        if order_id in self.pending_orders:
            order_data = self.pending_orders[order_id]
            if "order" in order_data:
                return self._convert_symbol_to_okx_format(order_data["order"].symbol)
            elif "response" in order_data:
                return self._convert_symbol_to_okx_format(order_data["response"].symbol)

        # Fallback - this should be improved with proper order tracking
        raise ValidationError(f"Cannot find symbol for order {order_id}")

    async def _validate_okx_order(self, order: OrderRequest) -> None:
        """Validate order parameters against OKX requirements."""
        try:
            # Basic validation
            if not order.symbol or not order.quantity:
                raise ValidationError("Order must have symbol and quantity")

            if order.quantity <= 0:
                raise ValidationError("Order quantity must be positive")

            # Validate symbol format
            okx_symbol = self._convert_symbol_to_okx_format(order.symbol)
            if not okx_symbol or "-" not in okx_symbol:
                raise ValidationError(f"Invalid symbol format for OKX: {order.symbol}")

            # Validate order type specific parameters
            if order.order_type == OrderType.LIMIT:
                if not order.price or order.price <= 0:
                    raise ValidationError("Limit orders must have positive price")

            elif order.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
                if not order.stop_price or order.stop_price <= 0:
                    raise ValidationError("Stop orders must have positive stop price")

            # Validate minimum order size (could be enhanced with real exchange info)
            min_size = Decimal("0.00001")  # Generic minimum
            if order.quantity < min_size:
                raise ValidationError(f"Order quantity {order.quantity} below minimum {min_size}")

            # Validate maximum order size to prevent accidental large orders
            max_size = Decimal("1000000")  # Generic maximum
            if order.quantity > max_size:
                raise ValidationError(f"Order quantity {order.quantity} above maximum {max_size}")

        except Exception as e:
            self.logger.error(f"OKX order validation failed: {e!s}")
            raise ValidationError(f"Order validation failed: {e!s}")

    def get_rate_limits(self) -> dict[str, int]:
        """Get current rate limits for OKX."""
        return {
            "requests_per_minute": 600,
            "orders_per_second": 20,
            "websocket_connections": 3,
            "weight_per_minute": 600,
        }
