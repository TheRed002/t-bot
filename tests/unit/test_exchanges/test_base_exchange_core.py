"""
Core unit tests for BaseExchange functionality - PERFORMANCE OPTIMIZED.

This module provides comprehensive tests for the BaseExchange class following
the current architecture and coding standards.
All async operations are mocked for maximum speed.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import (
    ExchangeConnectionError,
    OrderRejectionError,
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
    Ticker,
)
from src.core.types.market import Trade
from src.exchanges.base import BaseMockExchange


# Pre-computed values for performance
FIXED_DATETIME = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
FIXED_DECIMALS = {
    "price": Decimal("50000.00"),
    "quantity": Decimal("1.00"),
    "volume": Decimal("1000.00"),
}


class TestMockExchange(BaseMockExchange):
    """Optimized test implementation of BaseExchange - instant operations."""

    def __init__(self, name: str = "test", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._test_connected = True  # Start connected for faster tests
        self._mock_connected = True  # Override for instant operations

    async def connect(self) -> None:
        """Instant connect - no delays."""
        self._test_connected = True
        self._mock_connected = True
        self._connected = True
        # Skip superclass call to avoid delays

    async def disconnect(self) -> None:
        """Instant disconnect - no delays."""
        self._test_connected = False
        self._mock_connected = False
        self._connected = False
        # Skip superclass call to avoid delays


class TestBaseExchangeCore:
    """Test cases for BaseExchange core functionality."""

    @pytest.fixture(scope="function")
    def mock_config(self):
        """Create optimized mock configuration."""
        return {
            "exchange": {
                "name": "test",
                "api_key": "test_key",
                "api_secret": "test_secret",
                "timeout": 1,  # Fast timeout
                "max_retries": 1,  # Single retry
            }
        }

    @pytest.fixture(scope="function")  
    def base_exchange(self, mock_config):
        """Create optimized BaseExchange instance."""
        exchange = TestMockExchange("test", mock_config)
        # Pre-configure for fast tests
        exchange._mock_connected = False
        exchange._connected = False
        exchange._last_heartbeat = FIXED_DATETIME
        return exchange
        
    def mock_dependencies(self, exchange):
        """Helper to mock DI dependencies with fast returns."""
        def fast_dependency_mock(name):
            mock = AsyncMock()
            # All async calls return immediately
            mock.return_value = None
            return mock
        return patch.object(exchange, 'resolve_dependency', side_effect=fast_dependency_mock)

    def test_initialization(self, base_exchange, mock_config):
        """Test BaseExchange initialization - instant."""
        assert base_exchange.exchange_name == "test"
        assert base_exchange.config == mock_config
        # Not connected initially
        assert base_exchange.connected is False
        assert base_exchange._mock_balances is not None

    @pytest.mark.asyncio
    async def test_service_lifecycle(self, base_exchange):
        """Test BaseService lifecycle methods - optimized."""
        # Pre-mock all dependencies for instant returns
        with patch.object(base_exchange, 'resolve_dependency') as mock_resolve:
            fast_mock = AsyncMock(return_value=None)
            mock_resolve.return_value = fast_mock
            
            # Test startup - should be instant
            await base_exchange.start()
            assert base_exchange.is_running is True
            assert base_exchange.connected is True

            # Test shutdown - should be instant
            await base_exchange.stop()
            assert base_exchange.is_running is False

    @pytest.mark.asyncio  
    async def test_connect_disconnect(self, base_exchange):
        """Test connection and disconnection - instant operations."""
        # Initially not connected
        assert base_exchange.connected is False
        
        # Test connection - instant return
        await base_exchange.connect()
        assert base_exchange.connected is True
        assert base_exchange.last_heartbeat is not None

        # Test disconnection - instant return
        await base_exchange.disconnect()
        assert base_exchange.connected is False

    @pytest.mark.asyncio
    async def test_ping(self, base_exchange):
        """Test ping functionality - mocked for speed."""
        # Connect first
        await base_exchange.connect()
        
        # Now ping should work immediately
        result = await base_exchange.ping()
        assert result is True
        assert base_exchange.last_heartbeat is not None

        # Test disconnected case by mocking
        base_exchange._mock_connected = False
        base_exchange._connected = False
        with pytest.raises(ExchangeConnectionError):
            await base_exchange.ping()

    @pytest.mark.asyncio
    async def test_load_exchange_info(self, base_exchange):
        """Test loading exchange info - pre-configured data."""
        # Already connected, instant return
        info = await base_exchange.load_exchange_info()
        assert isinstance(info, ExchangeInfo)
        assert info.exchange == "mock"
        assert info.symbol == "BTCUSDT"  # Single symbol info
        assert base_exchange._trading_symbols is not None

    @pytest.mark.asyncio
    async def test_get_ticker(self, base_exchange):
        """Test getting ticker data."""
        with self.mock_dependencies(base_exchange):
            await base_exchange.start()

            ticker = await base_exchange.get_ticker("BTCUSDT")
            assert isinstance(ticker, Ticker)
            assert ticker.symbol == "BTCUSDT"
            assert isinstance(ticker.last_price, Decimal)
            assert ticker.bid_price > Decimal("0")
            assert ticker.ask_price > Decimal("0")
            assert ticker.exchange == "mock"

    @pytest.mark.asyncio
    async def test_get_ticker_invalid_symbol(self, base_exchange):
        """Test getting ticker for invalid symbol."""
        with self.mock_dependencies(base_exchange):
            await base_exchange.start()

            with pytest.raises(ValidationError):
                await base_exchange.get_ticker("INVALID")

    @pytest.mark.asyncio
    async def test_get_order_book(self, base_exchange):
        """Test getting order book."""
        with self.mock_dependencies(base_exchange):
            await base_exchange.start()

            order_book = await base_exchange.get_order_book("BTCUSDT")
            assert isinstance(order_book, OrderBook)
            assert order_book.symbol == "BTCUSDT"
            assert len(order_book.bids) > 0
            assert len(order_book.asks) > 0

            # Verify bid/ask structure
            best_bid = order_book.bids[0]
            best_ask = order_book.asks[0]
            assert isinstance(best_bid.price, Decimal)
            assert isinstance(best_bid.quantity, Decimal)
            assert isinstance(best_ask.price, Decimal)
            assert isinstance(best_ask.quantity, Decimal)
            assert best_ask.price > best_bid.price  # Basic spread check

    @pytest.mark.asyncio
    async def test_get_recent_trades(self, base_exchange):
        """Test getting recent trades."""
        with self.mock_dependencies(base_exchange):
            await base_exchange.start()

            trades = await base_exchange.get_recent_trades("BTCUSDT")
            assert isinstance(trades, list)
            assert len(trades) > 0

            trade = trades[0]
            assert isinstance(trade, Trade)
            assert trade.symbol == "BTCUSDT"
            assert isinstance(trade.price, Decimal)
            assert isinstance(trade.quantity, Decimal)

    @pytest.mark.asyncio
    async def test_place_order(self, base_exchange):
        """Test placing an order."""
        with self.mock_dependencies(base_exchange):
            await base_exchange.start()

            order_request = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("45000.00")
            )

            response = await base_exchange.place_order(order_request)
            assert isinstance(response, OrderResponse)
            assert response.symbol == "BTCUSDT"
            assert response.side == OrderSide.BUY
            assert response.order_type == OrderType.LIMIT
            assert response.quantity == Decimal("0.001")
            assert response.price == Decimal("45000.00")
            assert response.order_id is not None

    @pytest.mark.asyncio
    async def test_place_order_validation(self, base_exchange):
        """Test order placement validation."""
        with self.mock_dependencies(base_exchange):
            await base_exchange.start()

            # Test invalid symbol
            with pytest.raises(ValidationError):
                order_request = OrderRequest(
                    symbol="INVALID",
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=Decimal("0.001"),
                    price=Decimal("45000.00")
                )
                await base_exchange.place_order(order_request)

    @pytest.mark.asyncio
    async def test_cancel_order(self, base_exchange):
        """Test order cancellation."""
        with self.mock_dependencies(base_exchange):
            await base_exchange.start()

            # First place an order
            order_request = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("45000.00")
            )

            response = await base_exchange.place_order(order_request)
            order_id = response.order_id

            # Then cancel it
            cancel_response = await base_exchange.cancel_order("BTCUSDT", order_id)
            assert isinstance(cancel_response, OrderResponse)
            assert cancel_response.order_id == order_id
            assert cancel_response.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, base_exchange):
        """Test cancelling non-existent order."""
        with self.mock_dependencies(base_exchange):
            await base_exchange.start()

            with pytest.raises(OrderRejectionError):
                await base_exchange.cancel_order("BTCUSDT", "non_existent_order")

    @pytest.mark.asyncio
    async def test_get_order_status(self, base_exchange):
        """Test getting order status."""
        with self.mock_dependencies(base_exchange):
            await base_exchange.start()

            # First place an order
            order_request = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("45000.00")
            )

            response = await base_exchange.place_order(order_request)
            order_id = response.order_id

            # Get order status
            status_response = await base_exchange.get_order_status("BTCUSDT", order_id)
            assert isinstance(status_response, OrderResponse)
            assert status_response.order_id == order_id

    @pytest.mark.asyncio
    async def test_get_open_orders(self, base_exchange):
        """Test getting open orders."""
        with self.mock_dependencies(base_exchange):
            await base_exchange.start()

            # Get all open orders
            open_orders = await base_exchange.get_open_orders()
            assert isinstance(open_orders, list)

            # Get orders for specific symbol
            btc_orders = await base_exchange.get_open_orders("BTCUSDT")
            assert isinstance(btc_orders, list)

    @pytest.mark.asyncio
    async def test_get_account_balance(self, base_exchange):
        """Test getting account balance."""
        with self.mock_dependencies(base_exchange):
            await base_exchange.start()

            balances = await base_exchange.get_account_balance()
            assert isinstance(balances, dict)
            assert "USDT" in balances
            assert "BTC" in balances
            assert isinstance(balances["USDT"], Decimal)
            assert isinstance(balances["BTC"], Decimal)
            assert balances["USDT"] > Decimal("0")

    @pytest.mark.asyncio
    async def test_get_positions(self, base_exchange):
        """Test getting positions."""
        with self.mock_dependencies(base_exchange):
            await base_exchange.start()

            positions = await base_exchange.get_positions()
            assert isinstance(positions, list)
            # Mock exchange returns empty positions for spot trading

    @pytest.mark.asyncio
    async def test_health_check(self, base_exchange):
        """Test health check."""
        # Should be unhealthy when not connected
        health = await base_exchange.health_check()
        assert health.status.value == "unhealthy"

        # Should be healthy when connected
        with self.mock_dependencies(base_exchange):
            await base_exchange.start()
            health = await base_exchange.health_check()
            assert health.status.value == "healthy"
            assert "exchange" in health.details
            assert health.details["exchange"] == "test"

    def test_validation_helpers(self, base_exchange):
        """Test validation helper methods."""
        base_exchange._trading_symbols = ["BTCUSDT", "ETHUSDT"]

        # Test valid symbol
        base_exchange._validate_symbol("BTCUSDT")  # Should not raise

        # Test invalid symbol
        with pytest.raises(ValidationError):
            base_exchange._validate_symbol("INVALID")

        # Test empty symbol
        with pytest.raises(ValidationError):
            base_exchange._validate_symbol("")

        # Test price validation
        base_exchange._validate_price(Decimal("100.0"))  # Should not raise

        with pytest.raises(ValidationError):
            base_exchange._validate_price(Decimal("0"))

        with pytest.raises(ValidationError):
            base_exchange._validate_price(Decimal("-10"))

        with pytest.raises(ValidationError):
            base_exchange._validate_price(100.0)  # Not Decimal

        # Test quantity validation
        base_exchange._validate_quantity(Decimal("1.0"))  # Should not raise

        with pytest.raises(ValidationError):
            base_exchange._validate_quantity(Decimal("0"))

        with pytest.raises(ValidationError):
            base_exchange._validate_quantity(1.0)  # Not Decimal

    def test_utility_methods(self, base_exchange):
        """Test utility methods."""
        # Test exchange info getter
        assert base_exchange.get_exchange_info() is None  # Not loaded yet

        # Test trading symbols getter
        assert base_exchange.get_trading_symbols() is None  # Not loaded yet

        # Test symbol support check
        assert base_exchange.is_symbol_supported("BTCUSDT") is False  # Not loaded yet

    @pytest.mark.asyncio
    async def test_connection_required_operations(self, base_exchange):
        """Test operations that require connection."""
        operations = [
            lambda: base_exchange.get_ticker("BTCUSDT"),
            lambda: base_exchange.get_order_book("BTCUSDT"),
            lambda: base_exchange.get_recent_trades("BTCUSDT"),
            lambda: base_exchange.get_account_balance(),
            lambda: base_exchange.place_order(OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.001")
            )),
            lambda: base_exchange.cancel_order("BTCUSDT", "order_id"),
            lambda: base_exchange.get_order_status("BTCUSDT", "order_id"),
            lambda: base_exchange.get_open_orders(),
            lambda: base_exchange.get_positions(),
        ]

        # All should fail when not connected
        for operation in operations:
            with pytest.raises((ExchangeConnectionError, ValidationError)):
                await operation()

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, base_exchange):
        """Test concurrent exchange operations."""
        with self.mock_dependencies(base_exchange):
            await base_exchange.start()

            # Run multiple operations concurrently
            tasks = [
                base_exchange.get_ticker("BTCUSDT"),
                base_exchange.get_order_book("BTCUSDT"),
                base_exchange.get_recent_trades("BTCUSDT"),
                base_exchange.get_account_balance(),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed or have acceptable exceptions
            assert len(results) == 4
            for result in results:
                if isinstance(result, Exception):
                    # Should only be acceptable exceptions
                    assert isinstance(result, (ValidationError, ExchangeConnectionError))


class TestBaseMockExchangeSpecific:
    """Test cases specific to BaseMockExchange implementation."""

    @pytest.fixture
    def mock_exchange(self):
        """Create BaseMockExchange instance."""
        return BaseMockExchange("mock", {"test": "config"})
        
    def mock_dependencies(self, exchange):
        """Helper to mock DI dependencies."""
        return patch.object(exchange, 'resolve_dependency', side_effect=lambda name: AsyncMock())

    def test_mock_initialization(self, mock_exchange):
        """Test mock exchange initialization."""
        assert mock_exchange.exchange_name == "mock"
        assert isinstance(mock_exchange._mock_balances, dict)
        assert isinstance(mock_exchange._mock_orders, dict)
        assert "USDT" in mock_exchange._mock_balances
        assert "BTC" in mock_exchange._mock_balances

    @pytest.mark.asyncio
    async def test_mock_order_simulation(self, mock_exchange):
        """Test mock order simulation."""
        with self.mock_dependencies(mock_exchange):
            await mock_exchange.start()

            # Place order
            order_request = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("45000.00")
            )

            response = await mock_exchange.place_order(order_request)

            # Check order was stored
            assert response.order_id in mock_exchange._mock_orders
            stored_order = mock_exchange._mock_orders[response.order_id]
            assert stored_order.symbol == "BTCUSDT"
            assert stored_order.status in [OrderStatus.FILLED, OrderStatus.NEW]

    @pytest.mark.asyncio
    async def test_mock_price_simulation(self, mock_exchange):
        """Test mock price simulation."""
        with self.mock_dependencies(mock_exchange):
            await mock_exchange.start()

            # Get ticker multiple times to see price changes
            ticker1 = await mock_exchange.get_ticker("BTCUSDT")
            ticker2 = await mock_exchange.get_ticker("BTCUSDT")

            # Prices might be different due to simulation
            assert isinstance(ticker1.last_price, Decimal)
            assert isinstance(ticker2.last_price, Decimal)
            assert ticker1.last_price > Decimal("0")
            assert ticker2.last_price > Decimal("0")

    def test_mock_configuration(self, mock_exchange):
        """Test mock exchange internal state access."""
        # Test that we can access mock balances directly
        assert "USDT" in mock_exchange._mock_balances
        assert "BTC" in mock_exchange._mock_balances
        assert isinstance(mock_exchange._mock_balances["USDT"], Decimal)
        assert isinstance(mock_exchange._mock_balances["BTC"], Decimal)
        
        # Test that we can modify balances directly (for testing purposes)
        original_btc = mock_exchange._mock_balances["BTC"]
        mock_exchange._mock_balances["BTC"] = Decimal("2.0")
        assert mock_exchange._mock_balances["BTC"] == Decimal("2.0")
        
        # Restore original value
        mock_exchange._mock_balances["BTC"] = original_btc


if __name__ == "__main__":
    pytest.main([__file__])
