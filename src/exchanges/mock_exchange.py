"""Mock exchange implementation for development and testing."""

import asyncio
import random
import uuid
from datetime import datetime
from decimal import Decimal

from src.core.exceptions import ExchangeError, ExchangeInsufficientFundsError, ExchangeOrderError
from src.core.types import (
    Balance,
    MarketData,
    Order,
    OrderBook,
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
        trade_lifecycle_manager=None
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
        super().__init__(config_obj, exchange_id, state_service, trade_lifecycle_manager)

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
        return {
            "USDT": Balance(
                currency="USDT", free=Decimal("10000"), locked=Decimal("0"), total=Decimal("10000")
            ),
            "BTC": Balance(
                currency="BTC", free=Decimal("0.5"), locked=Decimal("0"), total=Decimal("0.5")
            ),
            "ETH": Balance(
                currency="ETH", free=Decimal("5"), locked=Decimal("0"), total=Decimal("5")
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
        """Simulate connection to mock exchange."""
        await asyncio.sleep(0.1)  # Simulate network delay
        self.is_connected = True
        self.logger.info(f"Connected to mock exchange: {self.exchange_id}")
        return True

    async def disconnect(self) -> None:
        """Simulate disconnection from mock exchange."""
        self.is_connected = False
        self.logger.info(f"Disconnected from mock exchange: {self.exchange_id}")

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

        if symbol not in self.market_prices:
            raise ExchangeError(f"Symbol {symbol} not found")

        base_price = self.market_prices[symbol]
        # Add some random variation
        variation = Decimal(random.uniform(-0.01, 0.01))
        current_price = base_price * (Decimal("1") + variation)

        return Ticker(
            symbol=symbol,
            bid=current_price - Decimal("1"),
            ask=current_price + Decimal("1"),
            last_price=current_price,
            volume_24h=Decimal(random.uniform(1000, 10000)),
            price_change_24h=Decimal(random.uniform(-100, 100)),
            timestamp=datetime.now(),
        )

    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook:
        """Get mock order book."""
        if not self.is_connected:
            await self.connect()

        if symbol not in self.market_prices:
            raise ExchangeError(f"Symbol {symbol} not found")

        base_price = self.market_prices[symbol]

        # Generate mock bids and asks
        bids = []
        asks = []

        for i in range(min(limit, 20)):
            bid_price = base_price - Decimal(i + 1) * Decimal("0.5")
            ask_price = base_price + Decimal(i + 1) * Decimal("0.5")

            bids.append([bid_price, Decimal(random.uniform(0.1, 2.0))])
            asks.append([ask_price, Decimal(random.uniform(0.1, 2.0))])

        return OrderBook(symbol=symbol, bids=bids, asks=asks, timestamp=datetime.now())

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

        # Generate order ID
        order_id = client_order_id or str(uuid.uuid4())

        # Get current price for market orders
        if order_type == OrderType.MARKET:
            ticker = await self.get_ticker(symbol)
            price = ticker.last_price

        if not price:
            raise ExchangeError("Price required for limit orders")

        # Check balance
        base, quote = symbol.split("/")
        if side == OrderSide.BUY:
            required = amount * price
            quote_balance = self.balances.get(quote)
            if not quote_balance or quote_balance.free < required:
                raise ExchangeInsufficientFundsError(f"Insufficient {quote} balance")
        else:
            base_balance = self.balances.get(base)
            if not base_balance or base_balance.free < amount:
                raise ExchangeInsufficientFundsError(f"Insufficient {base} balance")

        # Create order
        order = Order(
            id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            type=order_type,
            side=side,
            price=price,
            amount=amount,
            filled=Decimal("0"),
            remaining=amount,
            status=OrderStatus.NEW,
            timestamp=datetime.now(),
            time_in_force=time_in_force,
        )

        self.orders[order_id] = order

        # Simulate immediate fill for market orders
        if order_type == OrderType.MARKET:
            await self._fill_order(order_id)

        return order

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> Order:
        """Cancel mock order."""
        if not self.is_connected:
            await self.connect()

        if order_id not in self.orders:
            raise ExchangeOrderError(f"Order {order_id} not found")

        order = self.orders[order_id]

        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED]:
            raise ExchangeError(f"Order {order_id} cannot be canceled")

        order.status = OrderStatus.CANCELED

        # Release locked funds
        base, quote = order.symbol.split("/")
        if order.side == OrderSide.BUY:
            locked_amount = order.remaining * order.price
            self.balances[quote].locked -= locked_amount
            self.balances[quote].free += locked_amount
        else:
            self.balances[base].locked -= order.remaining
            self.balances[base].free += order.remaining

        return order

    async def get_order(self, order_id: str, symbol: str | None = None) -> Order:
        """Get mock order status."""
        if not self.is_connected:
            await self.connect()

        if order_id not in self.orders:
            raise ExchangeOrderError(f"Order {order_id} not found")

        # Randomly fill some pending limit orders
        order = self.orders[order_id]
        if order.status == OrderStatus.NEW and order.type == OrderType.LIMIT:
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
            if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]
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
                    id=f"trade_{order.id}",
                    order_id=order.id,
                    symbol=order.symbol,
                    side=order.side,
                    price=order.price,
                    amount=order.filled,
                    fee=order.filled * order.price * self.taker_fee,
                    fee_currency="USDT",
                    timestamp=order.timestamp,
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

    async def _fill_order(self, order_id: str) -> None:
        """Simulate order fill."""
        order = self.orders[order_id]

        if order.status != OrderStatus.NEW:
            return

        base, quote = order.symbol.split("/")

        # Update balances
        if order.side == OrderSide.BUY:
            cost = order.amount * order.price
            fee = cost * self.taker_fee

            self.balances[quote].free -= cost + fee
            self.balances[base].free += order.amount
        else:
            fee = order.amount * order.price * self.taker_fee

            self.balances[base].free -= order.amount
            self.balances[quote].free += order.amount * order.price - fee

        # Update order
        order.filled = order.amount
        order.remaining = Decimal("0")
        order.status = OrderStatus.FILLED

        self.logger.info(
            f"Mock order {order_id} filled: {order.amount} {order.symbol} @ {order.price}"
        )

    async def get_market_data(self, symbol: str) -> MarketData:
        """Get mock market data."""
        if not self.is_connected:
            await self.connect()

        ticker = await self.get_ticker(symbol)
        order_book = await self.get_order_book(symbol, limit=10)

        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            open=ticker.last_price * Decimal("0.98"),
            high=ticker.last_price * Decimal("1.02"),
            low=ticker.last_price * Decimal("0.97"),
            close=ticker.last_price,
            volume=ticker.volume_24h,
            bid=ticker.bid,
            ask=ticker.ask,
            order_book=order_book,
        )

    def validate_connection(self) -> bool:
        """Check if mock exchange is connected."""
        return self.is_connected

    async def _disconnect_from_exchange(self) -> None:
        """Disconnect from mock exchange."""
        pass  # Nothing to do for mock

    async def _get_market_data_from_exchange(self, symbol: str, timeframe: str = "1m"):
        """Get market data from mock exchange."""
        return await self.get_market_data(symbol)

    async def _get_trade_history_from_exchange(self, symbol: str, limit: int = 100):
        """Get trade history from mock exchange."""
        return await self.get_trades(symbol, limit)

    async def get_account_balance(self):
        """Get account balance from mock exchange."""
        return {currency: balance.free for currency, balance in self.balances.items()}

    async def get_exchange_info(self):
        """Get exchange information."""
        return {
            "name": "mock",
            "trading_pairs": list(self.market_prices.keys()),
            "fees": {"maker": float(self.maker_fee), "taker": float(self.taker_fee)},
        }

    async def get_order_status(self, order_id: str):
        """Get order status."""
        return await self.get_order(order_id)

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
