"""
Binance Exchange Implementation

Production-ready Binance exchange implementation following the project's
service layer pattern with proper dependency injection and error handling.

Key Features:
- BaseExchange inheritance for proper lifecycle management
- Financial precision with Decimal types
- Proper error handling with decorators
- Real Binance API integration
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

# MANDATORY: Core imports as per CLAUDE.md
# Import capital management exception
from src.core.exceptions import (
    CapitalAllocationError,
    ExchangeConnectionError,
    ExchangeError,
    ExecutionError,
    OrderRejectionError,
    ServiceError,
    ValidationError,
)
from src.core.types import (
    ExchangeInfo,
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
from src.core.types.market import Trade as MarketTrade

# MANDATORY: Import from error_handling decorators
from src.error_handling.decorators import with_circuit_breaker, with_retry

# Import the new BaseExchange
from src.exchanges.base import BaseExchange

# Import unified patterns
from src.exchanges.data_transformer import TransformerFactory, standardize_decimal_precision
from src.utils.data_flow_integrity import (
    validate_exchange_market_data,
)
from src.utils.messaging_patterns import (
    get_message_queue_manager,
    publish_market_data,
)

try:
    # Try to import Binance SDK
    from binance import AsyncClient, BinanceSocketManager
    from binance.exceptions import BinanceAPIException, BinanceOrderException

    BINANCE_AVAILABLE = True
except ImportError:
    # If Binance SDK is not available, create dummy classes
    AsyncClient = None
    BinanceSocketManager = None
    BinanceAPIException = Exception
    BinanceOrderException = Exception
    BINANCE_AVAILABLE = False


class BinanceExchange(BaseExchange):
    """
    Binance exchange implementation following service layer pattern.

    This class provides full Binance API integration while following
    the project's mandatory service patterns.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize Binance exchange service.

        Args:
            config: Binance configuration including API keys
        """
        super().__init__(name="binance", config=config)

        if not BINANCE_AVAILABLE:
            raise ServiceError(
                "Binance SDK not available. Install with: pip install python-binance"
            )

        # Extract API credentials
        self.api_key = config.get("api_key")
        self.api_secret = config.get("api_secret")
        self.testnet = config.get("testnet", False)

        if not self.api_key or not self.api_secret:
            raise ValidationError("Binance API key and secret are required")

        # Binance client (will be initialized on connect)
        self.client: AsyncClient | None = None
        self.socket_manager: BinanceSocketManager | None = None

        # Cache for symbols and exchange info

        # Initialize unified data transformer
        self.transformer = TransformerFactory.create_transformer("binance")

        # Initialize message queue manager
        self.message_manager = get_message_queue_manager()
        self._symbol_info_cache: dict[str, dict] = {}

        self.logger.info(f"Binance exchange initialized (testnet={self.testnet})")

    def get_symbol_precision(self, symbol: str) -> tuple[int, int]:
        """
        Get price and quantity precision for a symbol.

        Returns:
            tuple: (price_precision, quantity_precision)
        """
        if symbol not in self._symbol_info_cache:
            # Default fallback precision
            self.logger.warning(f"Symbol {symbol} not in cache, using default precision 8")
            return (8, 8)

        symbol_info = self._symbol_info_cache[symbol]
        filters = symbol_info.get("filters", [])

        price_precision = 8  # Default
        quantity_precision = 8  # Default

        for f in filters:
            if f.get("filterType") == "PRICE_FILTER":
                tick_size_str = f.get("tickSize", "0.00000001")
                tick_size = Decimal(str(tick_size_str))
                # Normalize to remove trailing zeros, then calculate precision
                normalized = tick_size.normalize()
                price_precision = abs(normalized.as_tuple().exponent)
            elif f.get("filterType") == "LOT_SIZE":
                step_size_str = f.get("stepSize", "0.00000001")
                step_size = Decimal(str(step_size_str))
                # Normalize to remove trailing zeros, then calculate precision
                normalized = step_size.normalize()
                quantity_precision = abs(normalized.as_tuple().exponent)

        return (price_precision, quantity_precision)

    def round_price(self, price: Decimal, symbol: str) -> Decimal:
        """Round price to exchange-specific precision."""
        if price is None:
            return price

        price_precision, _ = self.get_symbol_precision(symbol)
        quantizer = Decimal(10) ** -price_precision
        return price.quantize(quantizer)

    def round_quantity(self, quantity: Decimal, symbol: str) -> Decimal:
        """Round quantity to exchange-specific precision."""
        if quantity is None:
            return quantity

        _, quantity_precision = self.get_symbol_precision(symbol)
        quantizer = Decimal(10) ** -quantity_precision
        return quantity.quantize(quantizer)

    def _validate_service_config(self, config: dict[str, Any]) -> bool:
        """Validate Binance-specific configuration."""
        if not config:
            return False

        # Check required fields
        api_key = config.get("api_key")
        api_secret = config.get("api_secret")

        if not api_key or not api_secret:
            return False

        # Check that they are strings
        if not isinstance(api_key, str) or not isinstance(api_secret, str):
            return False

        return True

    @with_retry(max_attempts=3, base_delay=2.0)
    async def connect(self) -> None:
        """Establish connection to Binance API."""
        self.logger.info("Connecting to Binance API")

        try:
            # Create Binance async client
            self.client = AsyncClient(
                api_key=self.api_key, api_secret=self.api_secret, testnet=self.testnet
            )

            # Test connection by getting account info
            account_info = await self.client.get_account()
            self.logger.info(
                f"Connected to Binance successfully. Account type: {account_info.get('accountType', 'UNKNOWN')}"
            )

            self._connected = True
            self._last_heartbeat = datetime.now(timezone.utc)

            # Load exchange info
            if not self._exchange_info:
                await self.load_exchange_info()

            # Update state through StateService
            if hasattr(self, "_update_connection_state"):
                await self._update_connection_state(True)

        except BinanceAPIException as e:
            self.logger.error(f"Binance API error during connection: {e}")
            raise ExchangeConnectionError(f"Failed to connect to Binance: {e}") from e
        except Exception as e:
            self.logger.error(f"Connection error to Binance: {e}")
            raise ExchangeConnectionError(f"Failed to connect to Binance: {e}") from e

    async def disconnect(self) -> None:
        """Close connection to Binance API with proper resource cleanup."""
        self.logger.info("Disconnecting from Binance")

        socket_manager_to_close = None
        client_to_close = None

        try:
            # Store references to avoid race conditions
            socket_manager_to_close = self.socket_manager
            client_to_close = self.client
        except Exception as e:
            self.logger.error(f"Error preparing disconnect: {e}")

        # Close socket manager
        if socket_manager_to_close:
            try:
                await socket_manager_to_close.close()
            except Exception as e:
                self.logger.error(f"Error closing socket manager: {e}")
            finally:
                self.socket_manager = None

        # Close client connection
        if client_to_close:
            try:
                await client_to_close.close_connection()
            except Exception as e:
                self.logger.error(f"Error closing client connection: {e}")
            finally:
                self.client = None

        # Always update connected state
        self._connected = False

        # Update state through StateService
        if hasattr(self, "_update_connection_state"):
            await self._update_connection_state(False)

        self.logger.info("Disconnected from Binance successfully")

    @with_retry(max_attempts=2, base_delay=1.0)
    async def ping(self) -> bool:
        """Test Binance API connectivity."""
        if not self.client:
            raise ExchangeConnectionError("Binance client not connected")

        # Use Binance ping endpoint
        await self.client.ping()
        self._last_heartbeat = datetime.now(timezone.utc)
        return True

    async def load_exchange_info(self) -> ExchangeInfo:
        """Load Binance exchange information and trading rules."""
        try:
            self.logger.info("Loading Binance exchange info")

            if not self.client:
                raise ExchangeConnectionError("Binance client not connected")

            # Get exchange info from Binance
            exchange_info = await self.client.get_exchange_info()

            # Extract trading symbols
            symbols = []
            for symbol_info in exchange_info.get("symbols", []):
                if symbol_info.get("status") == "TRADING":
                    symbols.append(symbol_info["symbol"])
                    self._symbol_info_cache[symbol_info["symbol"]] = symbol_info

            self._trading_symbols = symbols

            # Create a default ExchangeInfo for BTCUSDT as per the base class expectation
            btc_symbol_info = self._symbol_info_cache.get("BTCUSDT", {})

            # Extract price and quantity filters
            price_filter = {}
            lot_size_filter = {}
            for f in btc_symbol_info.get("filters", []):
                if f["filterType"] == "PRICE_FILTER":
                    price_filter = f
                elif f["filterType"] == "LOT_SIZE":
                    lot_size_filter = f

            self._exchange_info = ExchangeInfo(
                symbol="BTCUSDT",
                base_asset="BTC",
                quote_asset="USDT",
                status="TRADING",
                min_price=Decimal(price_filter.get("minPrice", "0.01")),
                max_price=Decimal(price_filter.get("maxPrice", "1000000")),
                tick_size=Decimal(price_filter.get("tickSize", "0.01")),
                min_quantity=Decimal(lot_size_filter.get("minQty", "0.00001")),
                max_quantity=Decimal(lot_size_filter.get("maxQty", "10000")),
                step_size=Decimal(lot_size_filter.get("stepSize", "0.00001")),
                exchange="binance",
            )

            self.logger.info(f"Loaded {len(self._trading_symbols)} Binance trading symbols")
            return self._exchange_info

        except Exception as e:
            self.logger.error(f"Failed to load Binance exchange info: {e}")
            raise ExchangeError(f"Failed to load Binance exchange info: {e}") from e

    @with_circuit_breaker(failure_threshold=5)
    @with_retry(max_attempts=3)
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker information from Binance."""
        self._validate_symbol(symbol)

        try:
            if not self.client:
                raise ExchangeConnectionError("Binance client not connected")

            # Get 24hr ticker statistics
            ticker_data = await self.client.get_ticker(symbol=symbol)

            # Create raw market data for validation
            raw_market_data = {
                "symbol": symbol,
                "price": str(ticker_data["lastPrice"]),
                "bid": str(ticker_data["bidPrice"]),
                "ask": str(ticker_data["askPrice"]),
                "volume": str(ticker_data["volume"]),
                "high": str(ticker_data["highPrice"]),
                "low": str(ticker_data["lowPrice"]),
                "open": str(ticker_data["openPrice"]),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Validate using boundary validator
            validated_data = validate_exchange_market_data(raw_market_data, "binance")

            ticker = Ticker(
                symbol=symbol,
                bid_price=standardize_decimal_precision(validated_data["bid"]),
                bid_quantity=Decimal(ticker_data.get("bidQty", "0")),
                ask_price=standardize_decimal_precision(validated_data["ask"]),
                ask_quantity=Decimal(ticker_data.get("askQty", "0")),
                last_price=standardize_decimal_precision(validated_data["price"]),
                open_price=standardize_decimal_precision(validated_data["open"]),
                high_price=standardize_decimal_precision(validated_data["high"]),
                low_price=standardize_decimal_precision(validated_data["low"]),
                volume=standardize_decimal_precision(validated_data["volume"]),
                exchange="binance",
                timestamp=datetime.now(timezone.utc),
            )

            # Publish to message queue for consistent distribution
            await publish_market_data("binance", symbol, validated_data)

            # Persist ticker to database
            await self._persist_market_data(ticker)

            # Track analytics
            await self._track_analytics(
                "market_data", {"symbol": symbol, "ticker": ticker.__dict__}
            )

            return ticker

        except BinanceAPIException as e:
            self.logger.error(f"Binance API error getting ticker for {symbol}: {e}")
            raise ExchangeError(f"Failed to get ticker: {e}") from e
        except Exception as e:
            self.logger.error(f"Error getting ticker for {symbol}: {e}")
            raise ExchangeError(f"Failed to get ticker: {e}") from e

    @with_circuit_breaker(failure_threshold=5)
    @with_retry(max_attempts=3)
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook:
        """Get order book from Binance."""
        self._validate_symbol(symbol)

        try:
            if not self.client:
                raise ExchangeConnectionError("Binance client not connected")

            # Limit must be one of: 5, 10, 20, 50, 100, 500, 1000, 5000
            valid_limits = [5, 10, 20, 50, 100, 500, 1000, 5000]
            binance_limit = min(valid_limits, key=lambda x: abs(x - limit))

            order_book_data = await self.client.get_order_book(symbol=symbol, limit=binance_limit)

            # Convert to our format with proper OrderBookLevel objects
            from src.core.types import OrderBookLevel

            bids = [
                OrderBookLevel(price=Decimal(price), quantity=Decimal(qty))
                for price, qty in order_book_data["bids"]
            ]
            asks = [
                OrderBookLevel(price=Decimal(price), quantity=Decimal(qty))
                for price, qty in order_book_data["asks"]
            ]

            return OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.now(timezone.utc),
                exchange="binance",
            )

        except BinanceAPIException as e:
            self.logger.error(f"Binance API error getting order book for {symbol}: {e}")
            raise ExchangeError(f"Failed to get order book: {e}") from e
        except Exception as e:
            self.logger.error(f"Error getting order book for {symbol}: {e}")
            raise ExchangeError(f"Failed to get order book: {e}") from e

    @with_circuit_breaker(failure_threshold=5)
    @with_retry(max_attempts=3)
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        """Get recent trades from Binance."""
        self._validate_symbol(symbol)

        try:
            if not self.client:
                raise ExchangeConnectionError("Binance client not connected")

            trades_data = await self.client.get_recent_trades(symbol=symbol, limit=limit)

            trades = []
            for trade_data in trades_data:
                trades.append(
                    MarketTrade(
                        id=str(trade_data.get("id", trade_data.get("aggTradeId", ""))),
                        symbol=symbol,
                        exchange="binance",
                        side="SELL" if trade_data["isBuyerMaker"] else "BUY",
                        price=Decimal(trade_data["price"]),
                        quantity=Decimal(trade_data["qty"]),
                        timestamp=datetime.fromtimestamp(trade_data["time"] / 1000, timezone.utc),
                        maker=trade_data["isBuyerMaker"],
                        fee=Decimal("0"),  # Use Decimal zero instead of None
                    )
                )

            return trades

        except BinanceAPIException as e:
            self.logger.error(f"Binance API error getting trades for {symbol}: {e}")
            raise ExchangeError(f"Failed to get trades: {e}") from e
        except Exception as e:
            self.logger.error(f"Error getting trades for {symbol}: {e}")
            raise ExchangeError(f"Failed to get trades: {e}") from e

    @with_circuit_breaker(failure_threshold=3)
    @with_retry(max_attempts=2)
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place order on Binance with full service integrations."""
        # Start performance profiling
        profiler_context = None
        if self.performance_profiler:
            profiler_context = self.performance_profiler.profile("binance_place_order")
            profiler_context.__enter__()

        try:
            # Track telemetry event
            if self.telemetry_service:
                await self.telemetry_service.track_event(
                    "order_placement_started",
                    {
                        "exchange": "binance",
                        "symbol": order_request.symbol,
                        "type": order_request.order_type.value
                        if hasattr(order_request.order_type, "value")
                        else str(order_request.order_type),
                        "quantity": str(order_request.quantity),
                        "price": str(order_request.price) if order_request.price else None,
                    },
                )

            # Track analytics event for order placement
            await self._track_analytics(
                "order_started",
                {
                    "exchange": "binance",
                    "symbol": order_request.symbol,
                    "side": order_request.side.value
                    if hasattr(order_request.side, "value")
                    else str(order_request.side),
                    "order_type": order_request.order_type.value
                    if hasattr(order_request.order_type, "value")
                    else str(order_request.order_type),
                    "quantity": order_request.quantity,
                    "price": order_request.price,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Check capital availability
            if self.capital_service:
                # Get current market price if not provided
                current_price = order_request.price
                if not current_price:
                    ticker = await self.get_ticker(order_request.symbol)
                    current_price = ticker.last_price

                required_capital = order_request.quantity * current_price

                capital_check = await self.capital_service.check_available_capital(
                    amount=required_capital, symbol=order_request.symbol, exchange="binance"
                )

                if not capital_check.has_sufficient_capital:
                    self.logger.warning(
                        f"Insufficient capital for order: {capital_check.reason}",
                        extra={
                            "required": str(required_capital),
                            "available": str(capital_check.available_capital),
                        },
                    )
                    if self.telemetry_service:
                        await self.telemetry_service.track_event(
                            "order_rejected_insufficient_capital",
                            {"exchange": "binance", "symbol": order_request.symbol},
                        )
                    raise CapitalAllocationError(
                        f"Insufficient capital: {capital_check.reason}",
                        required=required_capital,
                        available=capital_check.available_capital,
                    )

                # Reserve capital for this order
                await self.capital_service.reserve_capital(
                    amount=required_capital,
                    order_id=order_request.client_order_id
                    or f"binance_{order_request.symbol}_{datetime.now(timezone.utc).timestamp()}",
                    symbol=order_request.symbol,
                )

            # Validate order (includes mandatory risk checks and basic validation)
            await self._validate_order(order_request)

            if not self.client:
                raise ExchangeConnectionError("Binance client not connected")

            # Convert order types
            binance_side = "BUY" if order_request.side == OrderSide.BUY else "SELL"
            binance_type = self._map_to_binance_order_type(order_request.order_type)

            # Prepare order parameters
            order_params = {
                "symbol": order_request.symbol,
                "side": binance_side,
                "type": binance_type,
            }

            # Round quantity to exchange precision
            rounded_quantity = self.round_quantity(order_request.quantity, order_request.symbol)

            # Add quantity or quote quantity
            if order_request.quote_quantity is not None:
                order_params["quoteOrderQty"] = str(order_request.quote_quantity)
            else:
                order_params["quantity"] = str(rounded_quantity)

            # Add price and other parameters based on order type
            if order_request.order_type == OrderType.LIMIT:
                # Round price to exchange precision
                rounded_price = self.round_price(order_request.price, order_request.symbol)
                order_params["price"] = str(rounded_price)
                # Use time_in_force from request, default to GTC if not specified
                tif_mapping = {"GTC": "GTC", "IOC": "IOC", "FOK": "FOK"}
                tif_value = (
                    getattr(order_request.time_in_force, "value", order_request.time_in_force)
                    if hasattr(order_request.time_in_force, "value")
                    else order_request.time_in_force
                )
                order_params["timeInForce"] = tif_mapping.get(tif_value, "GTC")
            elif order_request.order_type == OrderType.STOP_LOSS:
                if order_request.price:
                    rounded_price = self.round_price(order_request.price, order_request.symbol)
                    order_params["price"] = str(rounded_price)
                if order_request.stop_price:
                    rounded_stop_price = self.round_price(order_request.stop_price, order_request.symbol)
                    order_params["stopPrice"] = str(rounded_stop_price)
                # Use time_in_force from request, default to GTC if not specified
                tif_mapping = {"GTC": "GTC", "IOC": "IOC", "FOK": "FOK"}
                tif_value = (
                    getattr(order_request.time_in_force, "value", order_request.time_in_force)
                    if hasattr(order_request.time_in_force, "value")
                    else order_request.time_in_force
                )
                order_params["timeInForce"] = tif_mapping.get(tif_value, "GTC")
            elif order_request.order_type == OrderType.TAKE_PROFIT:
                if order_request.price:
                    rounded_price = self.round_price(order_request.price, order_request.symbol)
                    order_params["price"] = str(rounded_price)
                if order_request.stop_price:
                    rounded_stop_price = self.round_price(order_request.stop_price, order_request.symbol)
                    order_params["stopPrice"] = str(rounded_stop_price)
                # Use time_in_force from request, default to GTC if not specified
                tif_mapping = {"GTC": "GTC", "IOC": "IOC", "FOK": "FOK"}
                tif_value = (
                    getattr(order_request.time_in_force, "value", order_request.time_in_force)
                    if hasattr(order_request.time_in_force, "value")
                    else order_request.time_in_force
                )
                order_params["timeInForce"] = tif_mapping.get(tif_value, "GTC")

            # Add client order ID if provided
            if order_request.client_order_id:
                order_params["newClientOrderId"] = order_request.client_order_id

            self.logger.info(
                f"Placing Binance order: {order_request.symbol} {binance_side} {order_request.quantity} @ {order_request.price}"
            )

            # Place the order
            order_result = await self.client.create_order(**order_params)

            # Validate order result
            if not order_result:
                raise ExecutionError("Order placement returned empty result")

            # Convert Binance order status to our format
            status_mapping = {
                "NEW": OrderStatus.NEW,
                "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
                "FILLED": OrderStatus.FILLED,
                "CANCELED": OrderStatus.CANCELLED,  # Binance uses CANCELED, we use CANCELLED
                "REJECTED": OrderStatus.REJECTED,
                "EXPIRED": OrderStatus.EXPIRED,
            }

            order_status = status_mapping.get(order_result["status"], OrderStatus.NEW)

            response = OrderResponse(
                order_id=str(order_result["orderId"]),
                client_order_id=order_result.get("clientOrderId"),
                symbol=order_result["symbol"],
                side=OrderSide.BUY if order_result["side"] == "BUY" else OrderSide.SELL,
                order_type=self._map_order_type(order_result["type"]),
                quantity=Decimal(order_result["origQty"]),
                price=Decimal(order_result.get("price", "0")),
                status=order_status,
                filled_quantity=Decimal(order_result.get("executedQty", "0")),
                created_at=datetime.fromtimestamp(
                    order_result["transactTime"] / 1000, timezone.utc
                ),
                exchange="binance",
            )

            # Add average execution price if available
            if "cummulativeQuoteQty" in order_result and "executedQty" in order_result:
                executed_qty = Decimal(order_result["executedQty"])
                if executed_qty > 0:
                    # OrderResponse uses average_price, not execution_price
                    # Use proper decimal context for division precision
                    from decimal import getcontext

                    getcontext().prec = 28  # Set high precision for financial calculations

                    cumulative_quote = Decimal(order_result["cummulativeQuoteQty"])
                    average_price = cumulative_quote / executed_qty
                    # Since Pydantic models are immutable after creation, we need to create a new response
                    response = OrderResponse(
                        order_id=response.order_id,
                        client_order_id=response.client_order_id,
                        symbol=response.symbol,
                        side=response.side,
                        order_type=response.order_type,
                        quantity=response.quantity,
                        price=response.price,
                        status=response.status,
                        filled_quantity=response.filled_quantity,
                        average_price=average_price,
                        created_at=response.created_at,
                        exchange=response.exchange,
                    )

            # Persist order to database
            await self._persist_order(response)

            # Track execution analytics
            if self.analytics_service:
                try:
                    execution_time = datetime.now(timezone.utc) - datetime.fromtimestamp(
                        order_result["transactTime"] / 1000, timezone.utc
                    )
                    slippage = Decimal("0")
                    if response.average_price and order_request.price:
                        # Use proper decimal context for slippage calculation
                        from decimal import getcontext

                        getcontext().prec = 28  # Set high precision for financial calculations

                        # Calculate percentage slippage for more meaningful metrics
                        price_diff = abs(response.average_price - order_request.price)
                        slippage = (price_diff / order_request.price) * Decimal(
                            "100"
                        )  # Percentage slippage

                    await self.analytics_service.track_execution(
                        order_id=response.order_id,
                        exchange="binance",
                        symbol=order_request.symbol,
                        side=order_request.side,
                        quantity=order_request.quantity,
                        requested_price=order_request.price,
                        execution_price=response.average_price or order_request.price,
                        slippage=slippage,
                        fees=Decimal(order_result.get("commission", "0"))
                        if "commission" in order_result
                        else Decimal("0"),
                        execution_time=response.created_at,
                        latency_ms=execution_time.total_seconds() * 1000,
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to track execution analytics: {e}")

            # Broadcast order placed event
            if self.event_bus:
                try:
                    from src.exchanges.base import OrderPlacedEvent

                    await self.event_bus.publish(
                        OrderPlacedEvent(
                            exchange="binance",
                            order_id=response.order_id,
                            symbol=order_request.symbol,
                            quantity=order_request.quantity,
                            price=order_request.price,
                            timestamp=datetime.now(timezone.utc),
                        )
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to broadcast order event: {e}")

            # Track success telemetry
            if self.telemetry_service:
                await self.telemetry_service.track_event(
                    "order_placement_succeeded",
                    {
                        "order_id": response.order_id,
                        "exchange": "binance",
                        "status": response.status.value,
                    },
                )

            self.logger.info(
                f"Binance order placed successfully: {response.order_id} status={response.status.value}"
            )
            return response

        except (BinanceOrderException, BinanceAPIException, Exception) as e:
            # Track failure telemetry
            if self.telemetry_service:
                await self.telemetry_service.track_event(
                    "order_placement_failed",
                    {"error": str(e), "exchange": "binance", "symbol": order_request.symbol},
                )

            # Send alert for critical failures
            if self.alerting_service:
                from src.monitoring.alerting import AlertLevel

                await self.alerting_service.send_alert(
                    level=AlertLevel.ERROR,
                    message="Order placement failed on Binance",
                    details={"error": str(e), "order": order_request.to_dict()},
                )

            # Release reserved capital if order failed
            if self.capital_service and "required_capital" in locals():
                try:
                    await self.capital_service.release_capital(
                        order_id=order_request.client_order_id or f"binance_{order_request.symbol}",
                        amount=required_capital,
                    )
                except Exception as release_error:
                    self.logger.error(
                        f"Failed to release capital after order failure: {release_error}"
                    )

            # Apply consistent error propagation patterns
            error_context = self._apply_messaging_pattern({
                "error_type": type(e).__name__,
                "error_message": str(e),
                "operation": "place_order",
                "symbol": order_request.symbol,
                "exchange": "binance"
            }, "order")

            # Re-raise with appropriate error type and consistent propagation
            if isinstance(e, BinanceOrderException):
                self.logger.error(f"Binance order rejected: {e}", extra=error_context)
                raise OrderRejectionError(f"Binance order rejected: {e}") from e
            elif isinstance(e, BinanceAPIException):
                self.logger.error(f"Binance API error placing order: {e}", extra=error_context)
                raise ExchangeError(f"Failed to place order: {e}") from e
            else:
                self.logger.error(f"Error placing order: {e}", extra=error_context)
                raise ExchangeError(f"Failed to place order: {e}") from e
        finally:
            # End profiling
            if profiler_context:
                profiler_context.__exit__(None, None, None)

    @with_circuit_breaker(failure_threshold=3)
    @with_retry(max_attempts=2)
    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse:
        """Cancel order on Binance."""
        self._validate_symbol(symbol)

        try:
            if not self.client:
                raise ExchangeConnectionError("Binance client not connected")

            result = await self.client.cancel_order(symbol=symbol, orderId=int(order_id))

            return OrderResponse(
                order_id=str(result["orderId"]),
                client_order_id=result.get("clientOrderId"),
                symbol=result["symbol"],
                side=OrderSide.BUY if result["side"] == "BUY" else OrderSide.SELL,
                order_type=OrderType.LIMIT if result["type"] == "LIMIT" else OrderType.MARKET,
                quantity=Decimal(result["origQty"]),
                price=Decimal(result.get("price", "0")),
                status=OrderStatus.CANCELLED,
                filled_quantity=Decimal(result.get("executedQty", "0")),
                created_at=datetime.now(timezone.utc),
                exchange="binance",
            )

        except BinanceAPIException as e:
            self.logger.error(f"Binance API error cancelling order {order_id}: {e}")
            raise ExchangeError(f"Failed to cancel order: {e}") from e
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            raise ExchangeError(f"Failed to cancel order: {e}") from e

    @with_circuit_breaker(failure_threshold=5)
    @with_retry(max_attempts=3)
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse:
        """Get order status from Binance."""
        self._validate_symbol(symbol)

        try:
            if not self.client:
                raise ExchangeConnectionError("Binance client not connected")

            order = await self.client.get_order(symbol=symbol, orderId=int(order_id))

            # Convert status
            status_mapping = {
                "NEW": OrderStatus.NEW,
                "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
                "FILLED": OrderStatus.FILLED,
                "CANCELED": OrderStatus.CANCELLED,  # Binance uses CANCELED, we use CANCELLED
                "REJECTED": OrderStatus.REJECTED,
                "EXPIRED": OrderStatus.EXPIRED,
            }

            return OrderResponse(
                order_id=str(order["orderId"]),
                client_order_id=order.get("clientOrderId"),
                symbol=order["symbol"],
                side=OrderSide.BUY if order["side"] == "BUY" else OrderSide.SELL,
                order_type=OrderType.LIMIT if order["type"] == "LIMIT" else OrderType.MARKET,
                quantity=Decimal(order["origQty"]),
                price=Decimal(order.get("price", "0")),
                status=status_mapping.get(order["status"], OrderStatus.NEW),
                filled_quantity=Decimal(order.get("executedQty", "0")),
                created_at=datetime.fromtimestamp(order["time"] / 1000, timezone.utc),
                exchange="binance",
            )

        except BinanceAPIException as e:
            self.logger.error(f"Binance API error getting order {order_id}: {e}")
            raise ExchangeError(f"Failed to get order status: {e}") from e
        except Exception as e:
            self.logger.error(f"Error getting order {order_id}: {e}")
            raise ExchangeError(f"Failed to get order status: {e}") from e

    @with_circuit_breaker(failure_threshold=5)
    @with_retry(max_attempts=3)
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]:
        """Get open orders from Binance."""
        try:
            if not self.client:
                raise ExchangeConnectionError("Binance client not connected")

            # Get open orders (optionally filtered by symbol)
            if symbol:
                self._validate_symbol(symbol)
                orders = await self.client.get_open_orders(symbol=symbol)
            else:
                orders = await self.client.get_open_orders()

            order_responses = []
            for order in orders:
                status_mapping = {
                    "NEW": OrderStatus.NEW,
                    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
                    "FILLED": OrderStatus.FILLED,
                    "CANCELED": OrderStatus.CANCELLED,  # Binance uses CANCELED, we use CANCELLED
                    "REJECTED": OrderStatus.REJECTED,
                    "EXPIRED": OrderStatus.EXPIRED,
                }

                order_responses.append(
                    OrderResponse(
                        order_id=str(order["orderId"]),
                        client_order_id=order.get("clientOrderId"),
                        symbol=order["symbol"],
                        side=OrderSide.BUY if order["side"] == "BUY" else OrderSide.SELL,
                        order_type=OrderType.LIMIT
                        if order["type"] == "LIMIT"
                        else OrderType.MARKET,
                        quantity=Decimal(order["origQty"]),
                        price=Decimal(order.get("price", "0")),
                        status=status_mapping.get(order["status"], OrderStatus.NEW),
                        filled_quantity=Decimal(order.get("executedQty", "0")),
                        created_at=datetime.fromtimestamp(order["time"] / 1000, timezone.utc),
                        exchange="binance",
                    )
                )

            return order_responses

        except BinanceAPIException as e:
            self.logger.error(f"Binance API error getting open orders: {e}")
            raise ExchangeError(f"Failed to get open orders: {e}") from e
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            raise ExchangeError(f"Failed to get open orders: {e}") from e

    @with_circuit_breaker(failure_threshold=5)
    @with_retry(max_attempts=3)
    async def get_account_balance(self) -> dict[str, Decimal]:
        """Get account balance from Binance."""
        try:
            if not self.client:
                raise ExchangeConnectionError("Binance client not connected")

            account = await self.client.get_account()

            balances = {}
            for balance in account["balances"]:
                asset = balance["asset"]
                free_balance = Decimal(balance["free"])
                locked_balance = Decimal(balance["locked"])
                total_balance = free_balance + locked_balance

                if total_balance > 0:  # Only include non-zero balances
                    balances[asset] = total_balance

            return balances

        except BinanceAPIException as e:
            self.logger.error(f"Binance API error getting account balance: {e}")
            raise ExchangeError(f"Failed to get account balance: {e}") from e
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            raise ExchangeError(f"Failed to get account balance: {e}") from e

    @with_circuit_breaker(failure_threshold=5)
    @with_retry(max_attempts=3)
    async def get_positions(self) -> list[Position]:
        """Get positions from Binance (for futures trading)."""
        # For spot trading, return empty list
        # This could be extended to support Binance Futures
        return []

    def _map_order_type(self, binance_type: str) -> OrderType:
        """Map Binance order type to our OrderType enum."""
        # Direct mapping without transformer for clarity
        type_mapping = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP_LOSS": OrderType.STOP_LOSS,
            "STOP_LOSS_LIMIT": OrderType.STOP_LOSS,
            "TAKE_PROFIT": OrderType.TAKE_PROFIT,
            "TAKE_PROFIT_LIMIT": OrderType.TAKE_PROFIT,
            "LIMIT_MAKER": OrderType.LIMIT,
        }
        return type_mapping.get(binance_type, OrderType.MARKET)

    def _map_to_binance_order_type(self, order_type: OrderType) -> str:
        """Map our OrderType enum to Binance order type."""
        # Direct mapping without transformer for clarity
        type_mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP_LOSS: "STOP_LOSS_LIMIT",
            OrderType.TAKE_PROFIT: "TAKE_PROFIT_LIMIT",
        }
        mapped_type = type_mapping.get(order_type)
        if not mapped_type:
            raise ValidationError(f"Unsupported order type: {order_type}")
        return mapped_type


# For backward compatibility
BinanceExchangeService = BinanceExchange
