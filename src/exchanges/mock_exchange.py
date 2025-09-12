"""
Mock exchange implementation for development and testing.

This implementation provides a simulated exchange that follows the project
standards and can be used for testing without connecting to real exchange APIs.
"""

import random
from decimal import Decimal
from typing import Any

from src.core.exceptions import ExchangeConnectionError, OrderRejectionError
from src.core.types import (
    ExchangeInfo,
    OrderBook,
    OrderBookLevel,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    OrderType,
    Position,
    Ticker,
    Trade,
)

# Import error handling decorators
from src.error_handling.decorators import with_retry
from src.exchanges.base import BaseExchange


class BaseMockExchange(BaseExchange):
    """Base mock exchange implementation."""

    def __init__(self, name: str = "mock", config: dict[str, Any] | None = None):
        super().__init__(name, config or {})
        self._mock_balances: dict[str, Decimal] = {
            "USDT": Decimal("10000.00000000"),
            "BTC": Decimal("0.50000000"),
            "ETH": Decimal("5.00000000"),
        }
        self._mock_orders: dict[str, OrderResponse] = {}


class MockExchange(BaseMockExchange):
    """
    Mock exchange implementation for development and testing.

    This class provides a complete exchange simulation that can be used
    for testing trading algorithms without connecting to real exchanges.
    It follows all the project standards including Decimal precision.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize mock exchange.

        Args:
            config: Exchange configuration (optional for mock)
        """
        super().__init__(name="mock", config=config)
        # BaseService already provides self.logger, don't override it

        # Enhanced mock data with more realistic scenarios
        self._mock_balances = {
            "USDT": Decimal("10000.00000000"),
            "BTC": Decimal("0.50000000"),
            "ETH": Decimal("5.00000000"),
            "BNB": Decimal("100.00000000"),
            "ADA": Decimal("1000.00000000"),
        }

        # Track orders with enhanced state management
        self._mock_orders: dict[str, OrderResponse] = {}
        self._order_counter = 0

        # Mock market prices (more realistic price movements)
        self._mock_prices = {
            "BTCUSDT": Decimal("45000.00000000"),
            "ETHUSDT": Decimal("3000.00000000"),
            "BNBUSDT": Decimal("300.00000000"),
            "ADAUSDT": Decimal("0.50000000"),
        }

    @property
    def orders(self) -> dict[str, OrderResponse]:
        """Get all orders for backward compatibility with tests."""
        return self._mock_orders

    @with_retry(max_attempts=2, base_delay=0.5)
    async def connect(self) -> None:
        """Establish mock connection."""
        self.logger.info("Connecting to mock exchange")
        self._connected = True
        await super().connect()

        # Also load exchange info so symbol validation works
        if not self._exchange_info:
            await self.load_exchange_info()

        # Set initial heartbeat on connection
        from datetime import datetime, timezone
        self._last_heartbeat = datetime.now(timezone.utc)

        self.logger.info("Mock exchange connected successfully")

    async def _do_start(self) -> None:
        """Override base _do_start to make dependencies optional for testing."""
        try:
            self.logger.info(f"Starting {self.exchange_name} exchange service")

            # Try to resolve dependencies but don't fail if they're not available
            try:
                self.state_service = self.resolve_dependency("state_service")
            except Exception:
                self.logger.warning("StateService not available for mock exchange - using None")
                self.state_service = None

            # Make all other dependencies optional too
            for dep in ["validation_service", "data_pipeline", "database_service",
                       "risk_management_service", "analytics_service", "trading_tracer",
                       "performance_profiler", "alert_manager", "capital_management_service",
                       "telemetry_service", "alerting_service", "event_bus"]:
                try:
                    setattr(self, dep, self.resolve_dependency(dep))
                except Exception:
                    setattr(self, dep, None)

            # Connect to mock exchange
            await self.connect()

            self.logger.info(f"{self.exchange_name} exchange service started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start {self.exchange_name} exchange: {e}")
            # Don't raise for mock exchange - just log and continue
            self.logger.warning("Mock exchange continuing with limited functionality")

    async def disconnect(self) -> None:
        """Close mock connection."""
        self.logger.info("Disconnecting from mock exchange")
        # Set connection state to False
        self._connected = False
        self.logger.info("Mock exchange disconnected")

    @with_retry(max_attempts=2, base_delay=0.5)
    async def load_exchange_info(self) -> ExchangeInfo:
        """Load enhanced mock exchange information."""
        self.logger.info("Loading mock exchange info")

        # Initialize supported trading symbols directly
        self._trading_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]

        # For single symbol ExchangeInfo, we can create a default one for BTCUSDT
        self._exchange_info = ExchangeInfo(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            status="TRADING",
            min_price=Decimal("0.01000000"),
            max_price=Decimal("1000000.00000000"),
            tick_size=Decimal("0.01000000"),
            min_quantity=Decimal("0.00001000"),
            max_quantity=Decimal("1000000.00000000"),
            step_size=Decimal("0.00001000"),
            exchange="mock",
        )

        self.logger.info(f"Loaded {len(self._trading_symbols)} trading symbols")
        return self._exchange_info

    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        """Get mock ticker with realistic price simulation (returns dict for backward compatibility)."""
        self._ensure_connected()
        ticker = await self._get_ticker_impl(symbol)
        return {
            "symbol": ticker.symbol,
            "price": ticker.last_price,
            "bid": ticker.bid_price,
            "ask": ticker.ask_price,
            "volume": ticker.volume,
            "timestamp": ticker.timestamp,
            "last": ticker.last_price,  # Some tests expect 'last' field
        }

    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook:
        """Get mock order book with realistic spread."""
        self._ensure_connected()
        self._validate_symbol(symbol)

        base_price = self._mock_prices.get(symbol, Decimal("100.00000000"))

        # Create realistic order book with multiple levels
        bids = []
        asks = []

        for i in range(min(limit, 10)):  # Limit to 10 levels for mock
            bid_price = base_price - Decimal(str(0.01 * (i + 1)))
            ask_price = base_price + Decimal(str(0.01 * (i + 1)))

            bid_qty = Decimal(str(random.uniform(0.1, 2.0)))
            ask_qty = Decimal(str(random.uniform(0.1, 2.0)))

            bids.append(OrderBookLevel(price=bid_price, quantity=bid_qty))
            asks.append(OrderBookLevel(price=ask_price, quantity=ask_qty))

        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=self._get_current_time(),
            exchange="mock"
        )

    async def _place_order_impl(self, order_request: OrderRequest) -> OrderResponse:
        """
        Place mock order with enhanced validation and state management.

        Simulates real exchange behavior including:
        - Balance checking
        - Order validation
        - Realistic order states
        """
        self._ensure_connected()
        self.logger.info(
            f"Placing mock order: {order_request.symbol} {order_request.side} {order_request.quantity} @ {order_request.price}"
        )

        # Validate order request
        self._validate_symbol(order_request.symbol)
        # Only validate price for non-market orders (market orders can have None price)
        if order_request.order_type != OrderType.MARKET and order_request.price is not None:
            self._validate_price(order_request.price)
        self._validate_quantity(order_request.quantity)

        # Check if we have sufficient balance (simplified for mock)
        if order_request.side.value.lower() == "buy":
            # Use market price for market orders, provided price for limit orders
            if order_request.order_type == OrderType.MARKET or order_request.price is None:
                market_price = self._mock_prices.get(order_request.symbol, Decimal("45000.00"))
                required_balance = order_request.quantity * market_price
            else:
                required_balance = order_request.quantity * order_request.price
            quote_currency = "USDT"  # Simplified assumption
            if self._mock_balances.get(quote_currency, Decimal("0")) < required_balance:
                raise OrderRejectionError(f"Insufficient {quote_currency} balance")

        # Generate order ID
        self._order_counter += 1
        order_id = f"mock_order_{self._order_counter}"

        # Determine execution price for the response
        if order_request.order_type == OrderType.MARKET or order_request.price is None:
            execution_price = self._mock_prices.get(order_request.symbol, Decimal("45000.00"))
        else:
            execution_price = order_request.price

        # Simulate order execution (for simplicity, most orders fill immediately)
        import random

        # Different order types have different execution logic
        if order_request.order_type == OrderType.MARKET:
            # Market orders always fill immediately
            status = OrderStatus.FILLED
            filled_qty = order_request.quantity
            remaining_qty = Decimal("0")
        elif order_request.order_type == OrderType.STOP_LOSS:
            # Stop orders start as NEW and wait for trigger
            status = OrderStatus.NEW
            filled_qty = Decimal("0")
            remaining_qty = order_request.quantity
        elif random.random() > 0.1:  # 90% fill rate for limit orders
            status = OrderStatus.FILLED
            filled_qty = order_request.quantity
            remaining_qty = Decimal("0")
        else:
            status = OrderStatus.NEW
            filled_qty = Decimal("0")
            remaining_qty = order_request.quantity

        order_response = OrderResponse(
            order_id=order_id,
            client_order_id=order_request.client_order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            price=execution_price,  # Use execution price instead of request price
            status=status,
            filled_quantity=filled_qty,
            created_at=self._get_current_time(),
            exchange="mock",
        )

        self._mock_orders[order_id] = order_response

        # Update mock balances if order filled
        if status == OrderStatus.FILLED:
            self._update_balances_for_filled_order(order_response)

        self.logger.info(f"Mock order placed successfully: {order_id} status={status.value}")
        return order_response

    async def get_account_balance(self) -> dict[str, Decimal]:
        """Get mock account balances."""
        self._ensure_connected()
        self.logger.debug("Retrieving mock account balances")
        return self._mock_balances.copy()

    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse:
        """Cancel mock order."""
        self._ensure_connected()
        if order_id not in self._mock_orders:
            raise OrderRejectionError(f"Order {order_id} not found")

        order = self._mock_orders[order_id]

        # Check if order can be cancelled
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            raise OrderRejectionError(f"Cannot cancel order {order_id} - status is {order.status.value}")

        order.status = OrderStatus.CANCELLED
        self.logger.info(f"Mock order {order_id} cancelled")
        return order

    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse:
        """Get mock order status."""
        self._ensure_connected()
        if order_id not in self._mock_orders:
            raise OrderRejectionError(f"Order {order_id} not found")
        return self._mock_orders[order_id]

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]:
        """Get mock open orders."""
        self._ensure_connected()
        orders = list(self._mock_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return [o for o in orders if o.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]]

    async def get_positions(self) -> list[Position]:
        """Get mock positions (empty for spot trading)."""
        self._ensure_connected()
        return []

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        """Get mock recent trades."""
        self._ensure_connected()
        self._validate_symbol(symbol)
        base_price = self._mock_prices.get(symbol, Decimal("100.00000000"))

        # Import OrderSide from the correct location
        from src.core.types import OrderSide

        return [
            Trade(
                trade_id=f"trade_{symbol}_{i}",
                order_id=f"order_{symbol}_{i}",
                symbol=symbol,
                side=OrderSide.BUY,
                price=base_price,
                quantity=Decimal("0.1"),
                fee=Decimal("0.001"),
                fee_currency="USDT",
                timestamp=self._get_current_time(),
                exchange="mock",
                is_maker=True,
            )
            for i in range(min(limit, 5))  # Generate up to 5 mock trades
        ]

    async def ping(self) -> bool:
        """Mock ping - always returns True."""
        if not self.connected:
            raise ExchangeConnectionError("Mock exchange not connected")
        self._last_heartbeat = self._get_current_time()
        return True

    # Backward compatibility methods for old tests

    def configure(
        self, latency: int = 10, failure_rate: float = 0.0, partial_fill_rate: float = 0.0
    ) -> None:
        """Configure mock exchange behavior (backward compatibility)."""
        # Store config for potential future use
        pass

    def set_balance(self, balances: dict[str, Decimal]) -> None:
        """Set mock balances (backward compatibility)."""
        self._mock_balances.update(balances)

    def set_price(self, symbol: str, price: Decimal) -> None:
        """Set mock price for symbol (backward compatibility)."""
        self._mock_prices[symbol] = price

    def set_order_book(self, symbol: str, order_book: dict) -> None:
        """Set mock order book (backward compatibility)."""
        # For simplicity, just store the data (not implemented fully)
        pass

    async def get_balance(self) -> dict[str, Decimal]:
        """Get balance (backward compatibility method)."""
        return await self.get_account_balance()

    async def get_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        """Get trades (backward compatibility method)."""
        return await self.get_recent_trades(symbol, limit)

    # Dictionary return versions for old tests compatibility (separate methods)

    async def place_order_dict(
        self, symbol: str, side: str, order_type: str, quantity: Decimal, price: Decimal, **kwargs
    ) -> dict[str, Any]:
        """
        Place order returning dictionary format for backward compatibility.
        """
        from src.core.types import OrderRequest, OrderSide, OrderType

        # Convert string types to enums
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        order_type_enum = OrderType.LIMIT if order_type.upper() == "LIMIT" else OrderType.MARKET

        order_request = OrderRequest(
            symbol=symbol,
            side=order_side,
            order_type=order_type_enum,
            quantity=quantity,
            price=price,
        )

        # Call the concrete implementation to get OrderResponse
        response = await self._place_order_impl(order_request)

        # Convert to dictionary for backward compatibility
        return {
            "order_id": response.order_id,
            "symbol": response.symbol,
            "side": response.side.value.upper() if hasattr(response.side, "value") else str(response.side).upper(),
            "order_type": response.order_type.value.upper()
            if hasattr(response.order_type, "value")
            else str(response.order_type).upper(),
            "quantity": response.quantity,
            "price": response.price,
            "status": response.status.value.upper()
            if hasattr(response.status, "value")
            else str(response.status).upper(),
            "filled_quantity": response.filled_quantity,
            "remaining_quantity": response.quantity - response.filled_quantity,
            "timestamp": response.created_at,
        }

    async def get_ticker_dict(self, symbol: str) -> dict[str, Any]:
        """Get ticker in dictionary format for backward compatibility."""
        ticker = await self._get_ticker_impl(symbol)

        return {
            "symbol": ticker.symbol,
            "price": ticker.last_price,
            "bid": ticker.bid_price,
            "ask": ticker.ask_price,
            "volume": ticker.volume,
            "timestamp": ticker.timestamp,
            "last": ticker.last_price,  # Some tests expect 'last' field
        }

    # Override methods to provide both interfaces dynamically
    async def place_order(self, *args, **kwargs):
        """Smart place_order that handles both OrderRequest and individual params."""
        if len(args) == 1 and hasattr(args[0], "symbol"):
            # New interface: OrderRequest object
            return await self._place_order_impl(args[0])
        else:
            # Old interface: individual parameters -> return dict
            return await self.place_order_dict(*args, **kwargs)

    async def get_ticker_fallback(self, symbol: str):
        """Fallback get_ticker that can return either Ticker or dict based on calling context."""
        # For simplicity, always return dict for now to pass tests
        # This can be refined later if needed
        return await self.get_ticker_dict(symbol)

    async def _get_ticker_impl(self, symbol: str) -> Ticker:
        """Get mock ticker with realistic price simulation (original implementation)."""
        self._validate_symbol(symbol)

        base_price = self._mock_prices.get(symbol, Decimal("100.00000000"))

        # Simulate small price movements
        import random

        price_change = Decimal(str(random.uniform(-0.02, 0.02)))  # ±2% change
        current_price = base_price * (Decimal("1") + price_change)

        # Update stored price for consistency
        self._mock_prices[symbol] = current_price

        return Ticker(
            symbol=symbol,
            bid_price=current_price - Decimal("0.01"),
            bid_quantity=Decimal("10.0"),
            ask_price=current_price + Decimal("0.01"),
            ask_quantity=Decimal("10.0"),
            last_price=current_price,
            last_quantity=Decimal("1.0"),
            open_price=current_price * Decimal("0.98"),  # 2% lower than current
            high_price=current_price * Decimal("1.02"),  # 2% higher than current
            low_price=current_price * Decimal("0.96"),  # 4% lower than current
            volume=Decimal("1000.00000000"),
            timestamp=self._get_current_time(),
            exchange="mock",
        )

    # Helper methods

    def _get_current_time(self):
        """Get current timestamp."""
        from datetime import datetime, timezone

        return datetime.now(timezone.utc)

    def _update_balances_for_filled_order(self, order: OrderResponse) -> None:
        """Update mock balances after order execution."""
        try:
            # Simplified balance update (assumes USDT quote currency)
            base_currency = order.symbol.replace("USDT", "")
            quote_currency = "USDT"

            if order.side.value.lower() == "buy":
                # Buying: reduce USDT, increase base currency
                cost = order.filled_quantity * order.price
                self._mock_balances[quote_currency] = (
                    self._mock_balances.get(quote_currency, Decimal("0")) - cost
                )
                self._mock_balances[base_currency] = (
                    self._mock_balances.get(base_currency, Decimal("0")) + order.filled_quantity
                )
            else:
                # Selling: reduce base currency, increase USDT
                proceeds = order.filled_quantity * order.price
                self._mock_balances[base_currency] = (
                    self._mock_balances.get(base_currency, Decimal("0")) - order.filled_quantity
                )
                self._mock_balances[quote_currency] = (
                    self._mock_balances.get(quote_currency, Decimal("0")) + proceeds
                )

            self.logger.debug(f"Updated balances after order {order.order_id}")
        except Exception as e:
            self.logger.warning(f"Failed to update balances for order {order.order_id}: {e}")


    def _ensure_connected(self) -> None:
        """Ensure the exchange is connected."""
        if not self.connected:
            raise ExchangeConnectionError("Mock exchange is not connected")


# For backward compatibility, create an alias
MockExchangeForTesting = MockExchange
