"""
Unit tests for the base exchange interface.

This module tests the BaseExchange abstract class and related components
to ensure proper functionality and error handling.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.core.config import Config
from src.core.exceptions import (
    ExchangeConnectionError,
    ExchangeError,
)
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

# Import the components to test
from src.exchanges.base import BaseExchange


class MockExchange(BaseExchange):
    """Mock exchange implementation for testing."""

    def __init__(self, config: Config, exchange_name: str):
        super().__init__(config, exchange_name)
        self.mock_data = {
            "balances": {"BTC": Decimal("1.0"), "USDT": Decimal("10000.0")},
            "orders": {},
            "market_data": {},
            "connected": False,
        }

    async def connect(self) -> bool:
        """Mock connection implementation."""
        self.mock_data["connected"] = True
        self.connected = True
        self.status = "connected"
        return True

    async def disconnect(self) -> None:
        """Mock disconnection implementation."""
        self.mock_data["connected"] = False
        self.connected = False
        self.status = "disconnected"

    async def get_account_balance(self) -> dict:
        """Mock balance retrieval."""
        if not self.connected:
            raise ExchangeConnectionError("Not connected")
        return self.mock_data["balances"]

    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Mock order placement."""
        if not self.connected:
            raise ExchangeConnectionError("Not connected")

        order_id = f"order_{len(self.mock_data['orders']) + 1}"
        response = OrderResponse(
            id=order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            filled_quantity=Decimal("0"),
            status="pending",
            timestamp=datetime.now(timezone.utc),
        )

        self.mock_data["orders"][order_id] = response
        return response

    async def cancel_order(self, order_id: str) -> bool:
        """Mock order cancellation."""
        if not self.connected:
            raise ExchangeConnectionError("Not connected")

        if order_id in self.mock_data["orders"]:
            self.mock_data["orders"][order_id].status = "cancelled"
            return True
        return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Mock order status retrieval."""
        if not self.connected:
            raise ExchangeConnectionError("Not connected")

        if order_id in self.mock_data["orders"]:
            status_str = self.mock_data["orders"][order_id].status
            return OrderStatus(status_str)
        return OrderStatus.PENDING

    async def get_market_data(self, symbol: str, timeframe: str = "1m") -> MarketData:
        """Mock market data retrieval."""
        if not self.connected:
            raise ExchangeConnectionError("Not connected")

        return MarketData(
            symbol=symbol,
            price=Decimal("50000.0"),
            volume=Decimal("100.0"),
            timestamp=datetime.now(timezone.utc),
            bid=Decimal("49999.0"),
            ask=Decimal("50001.0"),
            open_price=Decimal("49900.0"),
            high_price=Decimal("50100.0"),
            low_price=Decimal("49800.0"),
        )

    async def subscribe_to_stream(self, symbol: str, callback) -> None:
        """Mock stream subscription."""
        if not self.connected:
            raise ExchangeConnectionError("Not connected")
        # Mock implementation - just log the subscription
        pass

    async def get_order_book(self, symbol: str, depth: int = 10) -> OrderBook:
        """Mock order book retrieval."""
        if not self.connected:
            raise ExchangeConnectionError("Not connected")

        return OrderBook(
            symbol=symbol,
            bids=[[Decimal("49999.0"), Decimal("1.0")], [Decimal("49998.0"), Decimal("2.0")]],
            asks=[[Decimal("50001.0"), Decimal("1.0")], [Decimal("50002.0"), Decimal("2.0")]],
            timestamp=datetime.now(timezone.utc),
        )

    async def get_trade_history(self, symbol: str, limit: int = 100) -> list:
        """Mock trade history retrieval."""
        if not self.connected:
            raise ExchangeConnectionError("Not connected")

        return [
            Trade(
                id=f"trade_{i}",
                symbol=symbol,
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                quantity=Decimal("0.1"),
                price=Decimal("50000.0"),
                timestamp=datetime.now(timezone.utc),
                fee=Decimal("0.001"),
                fee_currency="USDT",
            )
            for i in range(min(limit, 5))
        ]

    async def get_exchange_info(self) -> ExchangeInfo:
        """Mock exchange info retrieval."""
        if not self.connected:
            raise ExchangeConnectionError("Not connected")

        return ExchangeInfo(
            name=self.exchange_name,
            supported_symbols=["BTCUSDT", "ETHUSDT"],
            rate_limits={"requests_per_minute": 1200},
            features=["spot_trading"],
            api_version="1.0",
        )

    async def get_ticker(self, symbol: str) -> Ticker:
        """Mock ticker retrieval."""
        if not self.connected:
            raise ExchangeConnectionError("Not connected")

        return Ticker(
            symbol=symbol,
            bid=Decimal("49999.0"),
            ask=Decimal("50001.0"),
            last_price=Decimal("50000.0"),
            volume_24h=Decimal("1000.0"),
            price_change_24h=Decimal("100.0"),
            timestamp=datetime.now(timezone.utc),
        )


class TestBaseExchange:
    """Test cases for the BaseExchange abstract class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()

    @pytest.fixture
    def exchange(self, config):
        """Create a mock exchange instance."""
        return MockExchange(config, "test_exchange")

    def test_exchange_initialization(self, config):
        """Test exchange initialization."""
        exchange = MockExchange(config, "test_exchange")

        assert exchange.config == config
        assert exchange.exchange_name == "test_exchange"
        assert exchange.status == "initializing"
        assert not exchange.connected
        assert exchange.last_heartbeat is None
        assert exchange.error_handler is not None
        assert exchange.connection_manager is not None

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, exchange):
        """Test connection and disconnection."""
        # Test connection
        connected = await exchange.connect()
        assert connected
        assert exchange.connected
        assert exchange.status == "connected"

        # Test disconnection
        await exchange.disconnect()
        assert not exchange.connected
        assert exchange.status == "disconnected"

    @pytest.mark.asyncio
    async def test_get_account_balance(self, exchange):
        """Test account balance retrieval."""
        # Test without connection
        with pytest.raises(ExchangeConnectionError):
            await exchange.get_account_balance()

        # Test with connection
        await exchange.connect()
        balances = await exchange.get_account_balance()

        assert isinstance(balances, dict)
        assert "BTC" in balances
        assert "USDT" in balances
        assert isinstance(balances["BTC"], Decimal)
        assert isinstance(balances["USDT"], Decimal)

    @pytest.mark.asyncio
    async def test_place_order(self, exchange):
        """Test order placement."""
        await exchange.connect()

        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        response = await exchange.place_order(order)

        assert isinstance(response, OrderResponse)
        assert response.symbol == "BTCUSDT"
        assert response.side == OrderSide.BUY
        assert response.order_type == OrderType.MARKET
        assert response.quantity == Decimal("0.1")
        assert response.status == "pending"

    @pytest.mark.asyncio
    async def test_cancel_order(self, exchange):
        """Test order cancellation."""
        await exchange.connect()

        # Place an order first
        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )
        response = await exchange.place_order(order)

        # Cancel the order
        cancelled = await exchange.cancel_order(response.id)
        assert cancelled

        # Check order status
        status = await exchange.get_order_status(response.id)
        assert status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_get_market_data(self, exchange):
        """Test market data retrieval."""
        await exchange.connect()

        market_data = await exchange.get_market_data("BTCUSDT", "1m")

        assert isinstance(market_data, MarketData)
        assert market_data.symbol == "BTCUSDT"
        assert isinstance(market_data.price, Decimal)
        assert isinstance(market_data.volume, Decimal)
        assert market_data.bid is not None
        assert market_data.ask is not None

    @pytest.mark.asyncio
    async def test_get_order_book(self, exchange):
        """Test order book retrieval."""
        await exchange.connect()

        order_book = await exchange.get_order_book("BTCUSDT", 10)

        assert isinstance(order_book, OrderBook)
        assert order_book.symbol == "BTCUSDT"
        assert isinstance(order_book.bids, list)
        assert isinstance(order_book.asks, list)
        assert len(order_book.bids) > 0
        assert len(order_book.asks) > 0

    @pytest.mark.asyncio
    async def test_get_trade_history(self, exchange):
        """Test trade history retrieval."""
        await exchange.connect()

        trades = await exchange.get_trade_history("BTCUSDT", 5)

        assert isinstance(trades, list)
        assert len(trades) <= 5
        for trade in trades:
            assert isinstance(trade, Trade)
            assert trade.symbol == "BTCUSDT"
            assert isinstance(trade.quantity, Decimal)
            assert isinstance(trade.price, Decimal)

    @pytest.mark.asyncio
    async def test_get_exchange_info(self, exchange):
        """Test exchange info retrieval."""
        await exchange.connect()

        info = await exchange.get_exchange_info()

        assert isinstance(info, ExchangeInfo)
        assert info.name == "test_exchange"
        assert isinstance(info.supported_symbols, list)
        assert isinstance(info.rate_limits, dict)
        assert isinstance(info.features, list)

    @pytest.mark.asyncio
    async def test_get_ticker(self, exchange):
        """Test ticker retrieval."""
        await exchange.connect()

        ticker = await exchange.get_ticker("BTCUSDT")

        assert isinstance(ticker, Ticker)
        assert ticker.symbol == "BTCUSDT"
        assert isinstance(ticker.bid, Decimal)
        assert isinstance(ticker.ask, Decimal)
        assert isinstance(ticker.last_price, Decimal)

    @pytest.mark.asyncio
    async def test_pre_trade_validation(self, exchange):
        """Test pre-trade validation."""
        # Test valid order
        valid_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        result = await exchange.pre_trade_validation(valid_order)
        assert result

        # Test invalid order (missing symbol)
        invalid_order = OrderRequest(
            symbol="", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("0.1")
        )

        result = await exchange.pre_trade_validation(invalid_order)
        assert not result

        # Test invalid order (negative quantity)
        invalid_order2 = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("-0.1"),
        )

        result = await exchange.pre_trade_validation(invalid_order2)
        assert not result

    @pytest.mark.asyncio
    async def test_post_trade_processing(self, exchange):
        """Test post-trade processing."""
        order_response = OrderResponse(
            id="test_order",
            client_order_id="test_client_order",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=None,
            filled_quantity=Decimal("0.1"),
            status="filled",
            timestamp=datetime.now(timezone.utc),
        )

        # Should not raise any exceptions
        await exchange.post_trade_processing(order_response)

    def test_is_connected(self, exchange):
        """Test connection status check."""
        assert not exchange.is_connected()

        exchange.connected = True
        exchange.status = "connected"
        assert exchange.is_connected()

        exchange.status = "disconnected"
        assert not exchange.is_connected()

    def test_get_status(self, exchange):
        """Test status retrieval."""
        assert exchange.get_status() == "initializing"

        exchange.status = "connected"
        assert exchange.get_status() == "connected"

    def test_get_exchange_name(self, exchange):
        """Test exchange name retrieval."""
        assert exchange.get_exchange_name() == "test_exchange"

    @pytest.mark.asyncio
    async def test_health_check(self, exchange):
        """Test health check functionality."""
        # Test without connection
        result = await exchange.health_check()
        assert not result

        # Test with connection
        await exchange.connect()
        result = await exchange.health_check()
        assert result
        assert exchange.last_heartbeat is not None

    def test_get_rate_limits(self, exchange):
        """Test rate limits retrieval."""
        rate_limits = exchange.get_rate_limits()
        assert isinstance(rate_limits, dict)

    @pytest.mark.asyncio
    async def test_context_manager(self, exchange):
        """Test async context manager functionality."""
        async with exchange as e:
            assert e == exchange
            assert e.connected

        assert not exchange.connected


class TestBaseExchangeErrorHandling:
    """Test error handling in BaseExchange."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()

    @pytest.fixture
    def exchange(self, config):
        """Create a mock exchange instance."""
        return MockExchange(config, "test_exchange")

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, exchange):
        """Test handling of connection errors."""
        # Test operations without connection
        with pytest.raises(ExchangeConnectionError):
            await exchange.get_account_balance()

        with pytest.raises(ExchangeConnectionError):
            await exchange.place_order(Mock())

        with pytest.raises(ExchangeConnectionError):
            await exchange.get_market_data("BTCUSDT")

    @pytest.mark.asyncio
    async def test_validation_error_handling(self, exchange):
        """Test handling of validation errors."""
        await exchange.connect()

        # Test invalid order validation
        invalid_order = OrderRequest(
            symbol="", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("0")
        )

        result = await exchange.pre_trade_validation(invalid_order)
        assert not result

    @pytest.mark.asyncio
    async def test_exchange_error_propagation(self, exchange):
        """Test that exchange errors are properly propagated."""

        # Mock the place_order method to raise an ExchangeError
        async def mock_place_order(order):
            raise ExchangeError("Test exchange error")

        exchange.place_order = mock_place_order
        await exchange.connect()

        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        with pytest.raises(ExchangeError, match="Test exchange error"):
            await exchange.place_order(order)
