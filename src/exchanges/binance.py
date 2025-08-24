"""
Enhanced Binance Exchange Implementation

Refactored implementation using the unified infrastructure from base.py.
This eliminates all duplication and leverages:
- Unified connection pooling and session management
- Advanced rate limiting with local and global enforcement
- Unified WebSocket management with auto-reconnection
- Comprehensive error handling and recovery
- Market data caching and optimization
- Health monitoring and automatic recovery
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

# Binance-specific imports
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException, BinanceOrderException

from src.core.config import Config
from src.core.exceptions import (
    ExchangeConnectionError,
    ExchangeError,
    ExchangeErrorMapper,
    ExecutionError,
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
# NOTE: utils.constants doesn't exist, removing this import


class BinanceExchange(EnhancedBaseExchange):
    """
    Enhanced Binance exchange implementation with unified infrastructure.

    Provides complete Binance API integration with:
    - All common functionality inherited from EnhancedBaseExchange
    - Binance-specific API implementations
    - Automatic error mapping and handling
    - Optimized connection and WebSocket management
    """

    def __init__(
        self,
        config: Config,
        exchange_name: str = "binance",
        state_service: Any | None = None,
        trade_lifecycle_manager: Any | None = None,
    ):
        """
        Initialize enhanced Binance exchange.

        Args:
            config: Application configuration
            exchange_name: Exchange name (default: "binance")
            state_service: Optional state service for persistence
            trade_lifecycle_manager: Optional trade lifecycle manager
        """
        super().__init__(config, exchange_name, state_service, trade_lifecycle_manager)

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

        # Binance-specific clients (will be initialized in connect)
        self.binance_client: AsyncClient | None = None
        self.binance_ws_manager: BinanceSocketManager | None = None

        self.logger.info(f"Enhanced Binance exchange initialized (testnet: {self.testnet})")

    # === ENHANCED BASE IMPLEMENTATION ===

    async def _connect_to_exchange(self) -> bool:
        """
        Binance-specific connection logic.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.logger.info("Establishing Binance API connection...")

            # Validate API credentials
            if not self.api_key or not self.api_secret:
                raise ExchangeConnectionError("Binance API key and secret are required")

            # Initialize Binance client with proper configuration
            self.binance_client = await AsyncClient.create(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
                tld="us" if not self.testnet else None,  # Use US TLD for production
                requests_params={"timeout": 30},  # Set timeout
            )

            # Test connection by getting server time and exchange info
            server_time = await self.binance_client.get_server_time()
            self.logger.info(f"Binance server time: {server_time['serverTime']}")

            # Test account access
            account_info = await self.binance_client.get_account()
            self.logger.info(
                f"Connected to Binance account: {account_info.get('accountType', 'SPOT')}"
            )

            # Initialize WebSocket manager
            self.binance_ws_manager = BinanceSocketManager(self.binance_client)

            self.logger.info("Binance API connection established successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to establish Binance connection: {e!s}")
            await self._handle_binance_error(e, "connect")
            return False

    async def _disconnect_from_exchange(self) -> None:
        """Binance-specific disconnection logic."""
        try:
            self.logger.info("Closing Binance connections...")

            # Close Binance client
            if self.binance_client:
                await self.binance_client.close_connection()
                self.binance_client = None

            # WebSocket manager will be cleaned up by unified system
            self.binance_ws_manager = None

            self.logger.info("Binance connections closed successfully")

        except Exception as e:
            self.logger.error(f"Error closing Binance connections: {e!s}")

    async def _create_websocket_stream(self, symbol: str, stream_name: str) -> Any:
        """
        Create Binance-specific WebSocket stream.

        Args:
            symbol: Trading symbol
            stream_name: Name for the stream

        Returns:
            WebSocket connection object
        """
        try:
            if not self.binance_ws_manager:
                raise ExchangeConnectionError("Binance WebSocket manager not initialized")

            # Create ticker stream for the symbol
            stream = self.binance_ws_manager.symbol_ticker_socket(symbol)

            self.logger.debug(f"Created Binance WebSocket stream for {symbol}")
            return stream

        except Exception as e:
            self.logger.error(f"Failed to create Binance WebSocket stream for {symbol}: {e!s}")
            return None

    async def _handle_exchange_stream(self, stream_name: str, stream: Any) -> None:
        """
        Handle Binance-specific WebSocket stream messages.

        Args:
            stream_name: Name of the stream
            stream: Stream connection object
        """
        try:
            async with stream as stream_handler:
                async for message in stream_handler:
                    # Process Binance message format
                    if stream_name in self.stream_callbacks:
                        for callback in self.stream_callbacks[stream_name]:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(message)
                                else:
                                    callback(message)
                            except Exception as e:
                                self.logger.error(f"Error in Binance stream callback: {e!s}")

        except Exception as e:
            self.logger.error(f"Error handling Binance stream {stream_name}: {e!s}")
            raise

    async def _close_exchange_stream(self, stream_name: str, stream: Any) -> None:
        """
        Close Binance-specific WebSocket stream.

        Args:
            stream_name: Name of the stream
            stream: Stream connection object
        """
        try:
            # Binance streams are context managers and will close automatically
            self.logger.debug(f"Closed Binance stream {stream_name}")

        except Exception as e:
            self.logger.error(f"Error closing Binance stream {stream_name}: {e!s}")

    # === EXCHANGE API IMPLEMENTATIONS ===

    async def _place_order_on_exchange(self, order: OrderRequest) -> OrderResponse:
        """
        Binance-specific order placement logic.

        Args:
            order: Order request

        Returns:
            OrderResponse: Order response
        """
        try:
            if not self.binance_client:
                raise ExchangeConnectionError("Binance client not initialized")

            # Validate order parameters
            await self._validate_order_parameters(order)

            # Place order based on type with proper error handling
            result = None

            if order.order_type == OrderType.MARKET:
                if order.side.value.upper() == "BUY":
                    # For market buy orders, use quoteOrderQty for precision
                    if hasattr(order, "quote_quantity") and order.quote_quantity:
                        result = await self.binance_client.order_market_buy(
                            symbol=order.symbol,
                            quoteOrderQty=str(order.quote_quantity),
                            newClientOrderId=order.client_order_id,
                        )
                    else:
                        result = await self.binance_client.order_market_buy(
                            symbol=order.symbol,
                            quantity=str(order.quantity),
                            newClientOrderId=order.client_order_id,
                        )
                else:
                    result = await self.binance_client.order_market_sell(
                        symbol=order.symbol,
                        quantity=str(order.quantity),
                        newClientOrderId=order.client_order_id,
                    )

            elif order.order_type == OrderType.LIMIT:
                result = await self.binance_client.order_limit(
                    symbol=order.symbol,
                    side=order.side.value.upper(),
                    quantity=str(order.quantity),
                    price=str(order.price),
                    timeInForce=order.time_in_force or "GTC",
                    newClientOrderId=order.client_order_id,
                )

            elif order.order_type == OrderType.STOP_LOSS:
                result = await self.binance_client.order_stop_loss_limit(
                    symbol=order.symbol,
                    side=order.side.value.upper(),
                    quantity=str(order.quantity),
                    price=str(order.price),
                    stopPrice=str(order.stop_price),
                    timeInForce=order.time_in_force or "GTC",
                    newClientOrderId=order.client_order_id,
                )

            elif order.order_type == OrderType.TAKE_PROFIT:
                result = await self.binance_client.order_take_profit_limit(
                    symbol=order.symbol,
                    side=order.side.value.upper(),
                    quantity=str(order.quantity),
                    price=str(order.price),
                    stopPrice=str(order.stop_price),
                    timeInForce=order.time_in_force or "GTC",
                    newClientOrderId=order.client_order_id,
                )
            else:
                raise ValidationError(f"Unsupported order type: {order.order_type}")

            if not result:
                raise ExecutionError("Order placement returned empty result")

            # Convert response to unified format
            order_response = self._convert_binance_order_to_response(result)

            self.logger.info(
                f"Binance order placed successfully: {order_response.id} ({order.order_type.value})"
            )
            return order_response

        except (BinanceOrderException, BinanceAPIException) as e:
            await self._handle_binance_error(
                e,
                "place_order",
                {
                    "symbol": order.symbol,
                    "order_id": order.client_order_id,
                    "order_type": order.order_type.value,
                },
            )
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error placing Binance order: {e!s}")
            raise ExecutionError(f"Failed to place order: {e!s}")

    async def get_account_balance(self) -> dict[str, Decimal]:
        """
        Get all asset balances from Binance.

        Returns:
            Dict[str, Decimal]: Dictionary mapping asset symbols to balances
        """
        try:
            if not self.binance_client:
                raise ExchangeConnectionError("Binance client not initialized")

            # Get account information
            account_info = await self.binance_client.get_account()

            balances = {}
            for balance in account_info["balances"]:
                asset = balance["asset"]
                free = Decimal(balance["free"])
                locked = Decimal(balance["locked"])
                total = free + locked

                # Only include assets with non-zero balance
                if total > 0:
                    balances[asset] = total

            self.logger.debug(f"Retrieved {len(balances)} asset balances from Binance")

            # Store balance snapshot (handled by base class)
            await self._store_balance_snapshot(balances)

            return balances

        except BinanceAPIException as e:
            await self._handle_binance_error(e, "get_account_balance")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error getting Binance balance: {e!s}")
            raise ExchangeError(f"Failed to get account balance: {e!s}")

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order on Binance.

        Args:
            order_id: Order ID to cancel

        Returns:
            bool: True if cancellation successful, False otherwise
        """
        try:
            if not self.binance_client:
                raise ExchangeConnectionError("Binance client not initialized")

            # Cancel order (symbol is required but can be extracted from order tracking)
            symbol = self._get_symbol_for_order(order_id)

            await self.binance_client.cancel_order(symbol=symbol, orderId=order_id)

            # Update local tracking
            if order_id in self.pending_orders:
                self.pending_orders[order_id]["status"] = "cancelled"

            self.logger.info(f"Binance order cancelled successfully: {order_id}")
            return True

        except BinanceAPIException as e:
            await self._handle_binance_error(e, "cancel_order", {"order_id": order_id})
            return False
        except Exception as e:
            self.logger.error(f"Error cancelling Binance order {order_id}: {e!s}")
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get order status from Binance.

        Args:
            order_id: Order ID to check

        Returns:
            OrderStatus: Current order status
        """
        try:
            if not self.binance_client:
                raise ExchangeConnectionError("Binance client not initialized")

            # Get order status (symbol is required)
            symbol = self._get_symbol_for_order(order_id)

            result = await self.binance_client.get_order(symbol=symbol, orderId=order_id)

            # Convert to unified status
            status = self._convert_binance_status_to_order_status(result["status"])

            # Update local tracking
            if order_id in self.pending_orders:
                self.pending_orders[order_id]["status"] = status.value

            return status

        except BinanceAPIException as e:
            await self._handle_binance_error(e, "get_order_status", {"order_id": order_id})
            return OrderStatus.UNKNOWN
        except Exception as e:
            self.logger.error(f"Error getting Binance order status {order_id}: {e!s}")
            return OrderStatus.UNKNOWN

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
            if not self.binance_client:
                raise ExchangeConnectionError("Binance client not initialized")

            # Get klines (OHLCV data)
            klines = await self.binance_client.get_klines(
                symbol=symbol, interval=timeframe, limit=1
            )

            if not klines:
                raise ExchangeError(f"No market data available for {symbol}")

            # Convert to unified format
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

            return market_data

        except BinanceAPIException as e:
            await self._handle_binance_error(e, "get_market_data", {"symbol": symbol})
            raise
        except Exception as e:
            self.logger.error(f"Error getting Binance market data for {symbol}: {e!s}")
            raise ExchangeError(f"Failed to get market data: {e!s}")

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
            if not self.binance_client:
                raise ExchangeConnectionError("Binance client not initialized")

            # Get order book
            result = await self.binance_client.get_order_book(symbol=symbol, limit=depth)

            # Convert to unified format
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
            await self._handle_binance_error(e, "get_order_book", {"symbol": symbol})
            raise
        except Exception as e:
            self.logger.error(f"Error getting Binance order book for {symbol}: {e!s}")
            raise ExchangeError(f"Failed to get order book: {e!s}")

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
            if not self.binance_client:
                raise ExchangeConnectionError("Binance client not initialized")

            # Get recent trades
            result = await self.binance_client.get_recent_trades(symbol=symbol, limit=limit)

            # Convert to unified format
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
            await self._handle_binance_error(e, "get_trade_history", {"symbol": symbol})
            raise
        except Exception as e:
            self.logger.error(f"Error getting Binance trade history for {symbol}: {e!s}")
            return []  # Return empty list on error

    async def get_exchange_info(self) -> ExchangeInfo:
        """
        Get exchange information from Binance.

        Returns:
            ExchangeInfo: Exchange information and capabilities
        """
        try:
            if not self.binance_client:
                raise ExchangeConnectionError("Binance client not initialized")

            # Get exchange info
            result = await self.binance_client.get_exchange_info()

            # Convert to unified format
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
            await self._handle_binance_error(e, "get_exchange_info")
            raise
        except Exception as e:
            self.logger.error(f"Error getting Binance exchange info: {e!s}")
            raise ExchangeError(f"Failed to get exchange info: {e!s}")

    async def get_ticker(self, symbol: str) -> Ticker:
        """
        Get ticker information from Binance.

        Args:
            symbol: Trading symbol

        Returns:
            Ticker: Ticker information
        """
        try:
            if not self.binance_client:
                raise ExchangeConnectionError("Binance client not initialized")

            # Get ticker
            result = await self.binance_client.get_ticker(symbol=symbol)

            # Convert to unified format
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
            await self._handle_binance_error(e, "get_ticker", {"symbol": symbol})
            raise
        except Exception as e:
            self.logger.error(f"Error getting Binance ticker for {symbol}: {e!s}")
            raise ExchangeError(f"Failed to get ticker: {e!s}")

    # === HELPER METHODS ===

    async def _handle_binance_error(
        self, error: Exception, operation: str, context: dict | None = None
    ) -> None:
        """
        Handle Binance-specific errors using unified error mapping.

        Args:
            error: The Binance exception
            operation: Operation being performed
            context: Additional context
        """
        try:
            # Extract error data for mapping
            if isinstance(error, BinanceAPIException | BinanceOrderException):
                error_data = {
                    "code": getattr(error, "code", None),
                    "msg": str(error),
                    "response": getattr(error, "response", None),
                }
            else:
                error_data = {"msg": str(error)}

            # Map to unified exception
            unified_error = ExchangeErrorMapper.map_binance_error(error_data)

            # Handle using base class error handling
            await self._handle_exchange_error(unified_error, operation, context)

        except Exception as e:
            self.logger.error(f"Error in Binance error handling: {e!s}")

    def _convert_binance_order_to_response(self, result: dict) -> OrderResponse:
        """Convert Binance order result to unified OrderResponse."""
        # Handle both single order and list responses
        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        return OrderResponse(
            id=str(result["orderId"]),
            client_order_id=result.get("clientOrderId"),
            symbol=result["symbol"],
            side=OrderSide.BUY if result["side"] == "BUY" else OrderSide.SELL,
            order_type=self._convert_binance_type_to_order_type(result.get("type", "LIMIT")),
            quantity=Decimal(str(result["origQty"])),
            price=(
                Decimal(str(result["price"]))
                if result.get("price") and result["price"] != "0.00000000"
                else None
            ),
            filled_quantity=Decimal(str(result.get("executedQty", "0"))),
            status=self._convert_binance_status_to_order_status(result.get("status", "NEW")).value,
            timestamp=(
                datetime.fromtimestamp(result["transactTime"] / 1000, tz=timezone.utc)
                if "transactTime" in result
                else datetime.now(timezone.utc)
            ),
            average_price=(
                Decimal(str(result.get("cummulativeQuoteQty", "0")))
                / Decimal(str(result.get("executedQty", "1")))
                if Decimal(str(result.get("executedQty", "0"))) > 0
                else None
            ),
            commission=(
                Decimal(str(result.get("commission", "0"))) if "commission" in result else None
            ),
            commission_asset=result.get("commissionAsset"),
        )

    def _convert_binance_status_to_order_status(self, status: str) -> OrderStatus:
        """Convert Binance order status to unified OrderStatus enum."""
        status_mapping = {
            "NEW": OrderStatus.PENDING,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        return status_mapping.get(status, OrderStatus.UNKNOWN)

    def _get_symbol_for_order(self, order_id: str) -> str:
        """Get symbol for order from local tracking."""
        if order_id in self.pending_orders:
            order_data = self.pending_orders[order_id]
            if "order" in order_data:
                return order_data["order"].symbol
            elif "response" in order_data:
                return order_data["response"].symbol

        # Fallback - this should be improved with proper order tracking
        raise ValidationError(f"Cannot find symbol for order {order_id}")

    async def _validate_order_parameters(self, order: OrderRequest) -> None:
        """Validate order parameters against Binance exchange rules."""
        try:
            if not self.binance_client:
                raise ValidationError("Binance client not initialized")

            # Get exchange info for symbol validation
            exchange_info = await self.binance_client.get_exchange_info()
            symbol_info = None

            for symbol in exchange_info.get("symbols", []):
                if symbol["symbol"] == order.symbol:
                    symbol_info = symbol
                    break

            if not symbol_info:
                raise ValidationError(f"Symbol {order.symbol} not found on Binance")

            if symbol_info["status"] != "TRADING":
                raise ValidationError(f"Symbol {order.symbol} is not currently trading")

            # Validate order quantity and price against symbol filters
            for filter_info in symbol_info.get("filters", []):
                if filter_info["filterType"] == "LOT_SIZE":
                    min_qty = Decimal(filter_info["minQty"])
                    max_qty = Decimal(filter_info["maxQty"])
                    step_size = Decimal(filter_info["stepSize"])

                    if order.quantity < min_qty:
                        raise ValidationError(
                            f"Order quantity {order.quantity} below minimum {min_qty}"
                        )
                    if order.quantity > max_qty:
                        raise ValidationError(
                            f"Order quantity {order.quantity} above maximum {max_qty}"
                        )

                    # Check step size compliance using proper decimal arithmetic
                    if step_size > 0:
                        # Calculate how many steps from min_qty
                        steps = (order.quantity - min_qty) / step_size
                        # Round to nearest integer and check if it's a whole number
                        rounded_steps = steps.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
                        if abs(steps - rounded_steps) > Decimal('0.0000000001'):  # Small epsilon for float precision
                            # Calculate the nearest valid quantity
                            nearest_valid = min_qty + (rounded_steps * step_size)
                            raise ValidationError(
                                f"Order quantity {order.quantity} does not comply with step size {step_size}. "
                                f"Nearest valid quantity: {nearest_valid}"
                            )

                elif filter_info["filterType"] == "PRICE_FILTER" and order.price:
                    min_price = Decimal(filter_info["minPrice"])
                    max_price = Decimal(filter_info["maxPrice"])
                    tick_size = Decimal(filter_info["tickSize"])

                    if order.price < min_price:
                        raise ValidationError(
                            f"Order price {order.price} below minimum {min_price}"
                        )
                    if order.price > max_price:
                        raise ValidationError(
                            f"Order price {order.price} above maximum {max_price}"
                        )

                    # Check tick size compliance
                    remainder = (order.price - min_price) % tick_size
                    if remainder != 0:
                        raise ValidationError(
                            f"Order price {order.price} does not comply with tick size {tick_size}"
                        )

                elif filter_info["filterType"] == "MIN_NOTIONAL":
                    min_notional = Decimal(filter_info["minNotional"])
                    notional_value = order.quantity * (order.price or Decimal("0"))

                    if order.order_type != OrderType.MARKET and notional_value < min_notional:
                        raise ValidationError(
                            f"Order notional value {notional_value} below minimum {min_notional}"
                        )

        except Exception as e:
            self.logger.error(f"Order validation failed: {e!s}")
            raise ValidationError(f"Order validation failed: {e!s}")

    def _convert_binance_type_to_order_type(self, binance_type: str) -> OrderType:
        """Convert Binance order type to unified OrderType."""
        type_mapping = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP_LOSS": OrderType.STOP_LOSS,
            "STOP_LOSS_LIMIT": OrderType.STOP_LOSS,
            "TAKE_PROFIT": OrderType.TAKE_PROFIT,
            "TAKE_PROFIT_LIMIT": OrderType.TAKE_PROFIT,
        }
        return type_mapping.get(binance_type, OrderType.LIMIT)

    def get_rate_limits(self) -> dict[str, int]:
        """Get current rate limits for Binance."""
        return {
            "requests_per_minute": 1200,
            "orders_per_second": 10,
            "orders_per_24_hours": 160000,
            "weight_per_minute": 1200,
        }
