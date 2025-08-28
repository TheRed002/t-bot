"""Mock exchange implementation for development and testing."""

import asyncio
import random
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from src.core.exceptions import ExchangeError, ExchangeInsufficientFundsError, ExchangeOrderError
from src.core.types import (
    Balance,
    ExchangeInfo,
    MarketData,
    Order,
    OrderBook,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Ticker,
    TimeInForce,
    Trade,
)
from src.exchanges.base import BaseExchange


class MockExchange(BaseExchange):
    """Mock exchange for development and testing without real API keys."""

    def __init__(
        self,
        config,
        exchange_id: str = "mock",
        state_service=None,
        trade_lifecycle_manager=None,
        metrics_collector=None,
    ):
        """Initialize mock exchange."""
        # Import Config here to avoid circular imports
        from src.core.config import Config

        # Ensure config is a Config object
        if isinstance(config, dict):
            config_obj = Config()
        else:
            config_obj = config

        # Skip API key validation for mock mode
        super().__init__(
            config_obj, exchange_id, state_service, trade_lifecycle_manager, metrics_collector
        )

        # Initialize logger
        # Logger is provided by BaseExchange (via BaseComponent)

        # Set exchange_id for compatibility
        self.exchange_id = exchange_id

        # Initialize mock data
        self.orders: dict[str, Order] = {}
        self.balances: dict[str, Balance] = self._initialize_balances()
        self.positions: dict[str, Position] = {}
        self.market_prices: dict[str, Decimal] = self._initialize_prices()
        self.is_connected = False

        # Mock trading parameters
        self.maker_fee = Decimal("0.001")  # 0.1%
        self.taker_fee = Decimal("0.0015")  # 0.15%

    def _initialize_balances(self) -> dict[str, Balance]:
        """Initialize mock balances."""
        now = datetime.now(timezone.utc)
        return {
            "USDT": Balance(
                currency="USDT",
                available=Decimal("10000"),
                locked=Decimal("0"),
                total=Decimal("10000"),
                exchange=self.exchange_id,
                updated_at=now,
            ),
            "BTC": Balance(
                currency="BTC",
                available=Decimal("0.5"),
                locked=Decimal("0"),
                total=Decimal("0.5"),
                exchange=self.exchange_id,
                updated_at=now,
            ),
            "ETH": Balance(
                currency="ETH",
                available=Decimal("5"),
                locked=Decimal("0"),
                total=Decimal("5"),
                exchange=self.exchange_id,
                updated_at=now,
            ),
        }

    def _initialize_prices(self) -> dict[str, Decimal]:
        """Initialize mock market prices."""
        return {
            "BTC/USDT": Decimal("45000"),
            "ETH/USDT": Decimal("3000"),
            "BNB/USDT": Decimal("350"),
            "SOL/USDT": Decimal("100"),
            "ADA/USDT": Decimal("0.5"),
        }

    async def connect(self) -> bool:
        """Connect to mock exchange using BaseExchange connection logic."""
        # Call parent class connect() which handles all the connection logic
        return await super().connect()

    async def disconnect(self) -> None:
        """Disconnect from mock exchange using BaseExchange disconnect logic."""
        # Call parent class disconnect() which handles all the cleanup
        await super().disconnect()

    async def get_balance(self, currency: str | None = None) -> dict[str, Balance]:
        """Get mock account balance."""
        if not self.is_connected:
            await self.connect()

        if currency:
            if currency in self.balances:
                return {currency: self.balances[currency]}
            return {}
        return self.balances

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get mock ticker data."""
        if not self.is_connected:
            await self.connect()

        # Handle both BTC/USDT and BTCUSDT formats
        normalized_symbol = symbol
        if "/" not in symbol and len(symbol) >= 6:
            # Try to parse BTCUSDT -> BTC/USDT format
            if symbol.endswith("USDT"):
                base = symbol[:-4]
                normalized_symbol = f"{base}/USDT"
            elif symbol.endswith("BTC"):
                base = symbol[:-3]
                normalized_symbol = f"{base}/BTC"

        if normalized_symbol not in self.market_prices:
            raise ExchangeError(f"Symbol {symbol} not found")

        base_price = self.market_prices[normalized_symbol]
        # Add some random variation
        variation = Decimal(random.uniform(-0.01, 0.01))
        current_price = base_price * (Decimal("1") + variation)

        # Generate some volume and quantity data
        bid_quantity = Decimal(random.uniform(0.1, 2.0))
        ask_quantity = Decimal(random.uniform(0.1, 2.0))
        volume = Decimal(random.uniform(1000, 10000))
        price_change = Decimal(random.uniform(-100, 100))

        return Ticker(
            symbol=symbol,
            bid_price=current_price - Decimal("1"),
            bid_quantity=bid_quantity,
            ask_price=current_price + Decimal("1"),
            ask_quantity=ask_quantity,
            last_price=current_price,
            last_quantity=Decimal(random.uniform(0.01, 1.0)),
            open_price=base_price,
            high_price=current_price + Decimal("5"),
            low_price=current_price - Decimal("5"),
            volume=volume,
            quote_volume=volume * current_price,
            timestamp=datetime.now(timezone.utc),
            exchange=self.exchange_id,
            price_change=price_change,
            price_change_percent=float(price_change / base_price * 100) if base_price != 0 else 0.0,
        )

    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook:
        """Get mock order book."""
        if not self.is_connected:
            await self.connect()

        # Handle both BTC/USDT and BTCUSDT formats
        normalized_symbol = symbol
        if "/" not in symbol and len(symbol) >= 6:
            # Try to parse BTCUSDT -> BTC/USDT format
            if symbol.endswith("USDT"):
                base = symbol[:-4]
                normalized_symbol = f"{base}/USDT"
            elif symbol.endswith("BTC"):
                base = symbol[:-3]
                normalized_symbol = f"{base}/BTC"

        if normalized_symbol not in self.market_prices:
            raise ExchangeError(f"Symbol {symbol} not found")

        base_price = self.market_prices[normalized_symbol]

        # Import OrderBookLevel
        from src.core.types import OrderBookLevel

        # Generate mock bids and asks
        bids = []
        asks = []

        for i in range(min(limit, 20)):
            bid_price = base_price - Decimal(i + 1) * Decimal("0.5")
            ask_price = base_price + Decimal(i + 1) * Decimal("0.5")

            bids.append(
                OrderBookLevel(
                    price=bid_price,
                    quantity=Decimal(random.uniform(0.1, 2.0)),
                    order_count=random.randint(1, 10),
                )
            )
            asks.append(
                OrderBookLevel(
                    price=ask_price,
                    quantity=Decimal(random.uniform(0.1, 2.0)),
                    order_count=random.randint(1, 10),
                )
            )

        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=datetime.now(timezone.utc),
            exchange=self.exchange_id,
        )

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        amount: Decimal,
        price: Decimal | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        client_order_id: str | None = None,
    ) -> Order:
        """Place mock order."""
        if not self.is_connected:
            await self.connect()

        # Handle both BTC/USDT and BTCUSDT formats
        normalized_symbol = symbol
        if "/" not in symbol and len(symbol) >= 6:
            # Try to parse BTCUSDT -> BTC/USDT format
            if symbol.endswith("USDT"):
                base = symbol[:-4]
                normalized_symbol = f"{base}/USDT"
            elif symbol.endswith("BTC"):
                base = symbol[:-3]
                normalized_symbol = f"{base}/BTC"

        # Generate order ID
        order_id = client_order_id or str(uuid.uuid4())

        # Get current price for market orders
        if order_type == OrderType.MARKET:
            ticker = await self.get_ticker(symbol)
            price = ticker.last_price

        if not price:
            raise ExchangeError("Price required for limit orders")

        # Check balance
        base, quote = normalized_symbol.split("/")
        if side == OrderSide.BUY:
            required = amount * price
            quote_balance = self.balances.get(quote)
            if not quote_balance or quote_balance.available < required:
                raise ExchangeInsufficientFundsError(f"Insufficient {quote} balance")
        else:
            base_balance = self.balances.get(base)
            if not base_balance or base_balance.available < amount:
                raise ExchangeInsufficientFundsError(f"Insufficient {base} balance")

        # Create order
        order = Order(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            order_type=order_type,
            side=side,
            price=price,
            quantity=amount,
            filled_quantity=Decimal("0"),
            status=OrderStatus.OPEN,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            time_in_force=time_in_force,
            exchange=self.exchange_id,
        )

        self.orders[order_id] = order

        # Simulate immediate fill for market orders
        if order_type == OrderType.MARKET:
            await self._fill_order(order_id)

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel mock order."""
        if not self.is_connected:
            await self.connect()

        if order_id not in self.orders:
            raise ExchangeOrderError(f"Order {order_id} not found")

        order = self.orders[order_id]

        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            raise ExchangeError(f"Order {order_id} cannot be canceled")

        order.status = OrderStatus.CANCELLED

        # Release locked funds
        # Handle both BTC/USDT and BTCUSDT formats
        if "/" in order.symbol:
            base, quote = order.symbol.split("/")
        else:
            # Try to parse BTCUSDT -> BTC/USDT format
            if order.symbol.endswith("USDT"):
                base = order.symbol[:-4]
                quote = "USDT"
            elif order.symbol.endswith("BTC"):
                base = order.symbol[:-3]
                quote = "BTC"
            else:
                raise ExchangeError(f"Cannot parse symbol: {order.symbol}")
        if order.side == OrderSide.BUY:
            locked_amount = (order.quantity - order.filled_quantity) * order.price
            self.balances[quote].locked -= locked_amount
            self.balances[quote].available += locked_amount
        else:
            self.balances[base].locked -= order.quantity - order.filled_quantity
            self.balances[base].available += order.quantity - order.filled_quantity

        return True

    async def get_order(self, order_id: str, symbol: str | None = None) -> Order:
        """Get mock order status."""
        if not self.is_connected:
            await self.connect()

        if order_id not in self.orders:
            raise ExchangeOrderError(f"Order {order_id} not found")

        # Randomly fill some pending limit orders
        order = self.orders[order_id]
        if order.status == OrderStatus.OPEN and order.order_type == OrderType.LIMIT:
            if random.random() < 0.3:  # 30% chance to fill
                await self._fill_order(order_id)

        return self.orders[order_id]

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get mock open orders."""
        if not self.is_connected:
            await self.connect()

        open_orders = [
            order
            for order in self.orders.values()
            if order.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
        ]

        if symbol:
            open_orders = [o for o in open_orders if o.symbol == symbol]

        return open_orders

    async def get_order_history(self, symbol: str | None = None, limit: int = 100) -> list[Order]:
        """Get mock order history."""
        if not self.is_connected:
            await self.connect()

        all_orders = list(self.orders.values())

        if symbol:
            all_orders = [o for o in all_orders if o.symbol == symbol]

        # Sort by timestamp descending
        all_orders.sort(key=lambda x: x.timestamp, reverse=True)

        return all_orders[:limit]

    async def get_trades(self, symbol: str | None = None, limit: int = 100) -> list[Trade]:
        """Get mock trade history."""
        if not self.is_connected:
            await self.connect()

        # Generate mock trades from filled orders
        trades = []
        for order in self.orders.values():
            if order.status == OrderStatus.FILLED:
                trade = Trade(
                    trade_id=f"trade_{order.order_id}",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    price=order.price,
                    quantity=order.filled_quantity,
                    fee=order.filled_quantity * order.price * self.taker_fee,
                    fee_currency="USDT",
                    timestamp=order.created_at,
                    exchange=self.exchange_id,
                    is_maker=order.order_type == OrderType.LIMIT,
                )
                trades.append(trade)

        if symbol:
            trades = [t for t in trades if t.symbol == symbol]

        trades.sort(key=lambda x: x.timestamp, reverse=True)
        return trades[:limit]

    async def get_positions(self) -> list[Position]:
        """Get mock open positions."""
        if not self.is_connected:
            await self.connect()

        return list(self.positions.values())

    async def get_trade_history(self, symbol: str, limit: int = 100) -> list[Trade]:
        """Get trade history - alias for get_trades."""
        return await self.get_trades(symbol, limit)

    async def _fill_order(self, order_id: str) -> None:
        """Simulate order fill."""
        order = self.orders[order_id]

        if order.status != OrderStatus.OPEN:
            return

        # Handle both BTC/USDT and BTCUSDT formats
        if "/" in order.symbol:
            base, quote = order.symbol.split("/")
        else:
            # Try to parse BTCUSDT -> BTC/USDT format
            if order.symbol.endswith("USDT"):
                base = order.symbol[:-4]
                quote = "USDT"
            elif order.symbol.endswith("BTC"):
                base = order.symbol[:-3]
                quote = "BTC"
            else:
                raise ExchangeError(f"Cannot parse symbol: {order.symbol}")

        # Update balances
        if order.side == OrderSide.BUY:
            cost = order.quantity * order.price
            fee = cost * self.taker_fee

            self.balances[quote].available -= cost + fee
            self.balances[base].available += order.quantity
        else:
            fee = order.quantity * order.price * self.taker_fee

            self.balances[base].available -= order.quantity
            self.balances[quote].available += order.quantity * order.price - fee

        # Update order
        order.filled_quantity = order.quantity
        order.updated_at = datetime.now(timezone.utc)
        order.status = OrderStatus.FILLED

        self.logger.info(
            f"Mock order {order_id} filled: {order.quantity} {order.symbol} @ {order.price}"
        )

    async def get_market_data(self, symbol: str) -> MarketData:
        """Get mock market data."""
        if not self.is_connected:
            await self.connect()

        # Handle both BTC/USDT and BTCUSDT formats
        normalized_symbol = symbol
        if "/" not in symbol and len(symbol) >= 6:
            # Try to parse BTCUSDT -> BTC/USDT format
            if symbol.endswith("USDT"):
                base = symbol[:-4]
                normalized_symbol = f"{base}/USDT"
            elif symbol.endswith("BTC"):
                base = symbol[:-3]
                normalized_symbol = f"{base}/BTC"

        if normalized_symbol not in self.market_prices:
            raise ExchangeError(f"Symbol {symbol} not found")

        ticker = await self.get_ticker(symbol)
        order_book = await self.get_order_book(symbol, limit=10)

        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            open=ticker.open_price,
            high=ticker.high_price,
            low=ticker.low_price,
            close=ticker.last_price,
            volume=ticker.volume,
            bid=ticker.bid_price,
            ask=ticker.ask_price,
            order_book=order_book,
        )

    def validate_connection(self) -> bool:
        """Check if mock exchange is connected."""
        return self.is_connected

    async def _connect_to_exchange(self) -> bool:
        """Connect to mock exchange."""
        self.is_connected = True
        self.logger.info(f"Connected to mock exchange: {self.exchange_id}")
        return True

    async def _disconnect_from_exchange(self) -> None:
        """Disconnect from mock exchange."""
        self.is_connected = False
        self.logger.info(f"Disconnected from mock exchange: {self.exchange_id}")

    async def _get_market_data_from_exchange(self, symbol: str, timeframe: str = "1m"):
        """Get market data from mock exchange."""
        return await self.get_market_data(symbol)

    async def _get_trade_history_from_exchange(self, symbol: str, limit: int = 100):
        """Get trade history from mock exchange."""
        return await self.get_trades(symbol, limit)

    async def _place_order_on_exchange(self, order: OrderRequest) -> OrderResponse:
        """Place order on mock exchange."""
        # Convert OrderRequest to our place_order parameters
        placed_order = await self.place_order(
            symbol=order.symbol,
            side=order.side,
            order_type=order.type,
            amount=order.quantity,
            price=order.price,
            time_in_force=order.time_in_force,
            client_order_id=order.client_order_id,
        )

        # Convert Order to OrderResponse
        return OrderResponse(
            order_id=placed_order.order_id,
            client_order_id=placed_order.client_order_id,
            symbol=placed_order.symbol,
            side=placed_order.side,
            type=placed_order.order_type,
            price=placed_order.price,
            amount=placed_order.quantity,
            status=placed_order.status,
            timestamp=placed_order.created_at,
        )

    async def _create_websocket_stream(self, symbol: str, stream_name: str):
        """Create mock websocket stream."""
        # For mock, we just return a dummy stream object
        return {"symbol": symbol, "stream": stream_name, "active": True}

    async def _handle_exchange_stream(self, stream_name: str, stream) -> None:
        """Handle mock exchange stream."""
        # Mock implementation - just log
        self.logger.debug(f"Handling mock stream: {stream_name}")
        await asyncio.sleep(0.1)

    async def _close_exchange_stream(self, stream_name: str, stream) -> None:
        """Close mock exchange stream."""
        # Mock implementation
        if isinstance(stream, dict):
            stream["active"] = False
        self.logger.debug(f"Closed mock stream: {stream_name}")

    async def get_account_balance(self):
        """Get account balance from mock exchange."""
        return {currency: balance.available for currency, balance in self.balances.items()}

    async def get_exchange_info(self) -> list[ExchangeInfo]:
        """Get exchange information."""
        info_list = []

        # Define trading rules for each symbol
        symbol_info = {
            "BTC/USDT": {
                "base": "BTC",
                "quote": "USDT",
                "min_quantity": Decimal("0.0001"),
                "step_size": Decimal("0.0001"),
                "tick_size": Decimal("0.01"),
            },
            "ETH/USDT": {
                "base": "ETH",
                "quote": "USDT",
                "min_quantity": Decimal("0.001"),
                "step_size": Decimal("0.001"),
                "tick_size": Decimal("0.01"),
            },
            "BNB/USDT": {
                "base": "BNB",
                "quote": "USDT",
                "min_quantity": Decimal("0.01"),
                "step_size": Decimal("0.01"),
                "tick_size": Decimal("0.01"),
            },
            "SOL/USDT": {
                "base": "SOL",
                "quote": "USDT",
                "min_quantity": Decimal("0.1"),
                "step_size": Decimal("0.1"),
                "tick_size": Decimal("0.01"),
            },
            "ADA/USDT": {
                "base": "ADA",
                "quote": "USDT",
                "min_quantity": Decimal("1"),
                "step_size": Decimal("1"),
                "tick_size": Decimal("0.0001"),
            },
        }

        for symbol, info in symbol_info.items():
            info_list.append(
                ExchangeInfo(
                    symbol=symbol,
                    base_asset=info["base"],
                    quote_asset=info["quote"],
                    status="TRADING",
                    min_price=Decimal("0.01"),
                    max_price=Decimal("1000000"),
                    tick_size=info["tick_size"],
                    min_quantity=info["min_quantity"],
                    max_quantity=Decimal("10000000"),
                    step_size=info["step_size"],
                    min_notional=Decimal("10"),
                    exchange=self.exchange_id,
                    is_trading=True,
                )
            )

        return info_list

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        order = await self.get_order(order_id)
        return order.status

    async def subscribe_to_stream(self, symbol: str, callback):
        """Subscribe to mock data stream."""

        # For mock, we can simulate periodic updates
        async def mock_stream():
            while self.is_connected:
                ticker = await self.get_ticker(symbol)
                await callback(ticker)
                await asyncio.sleep(1)  # Update every second

        # Start the mock stream in background
        asyncio.create_task(mock_stream())
