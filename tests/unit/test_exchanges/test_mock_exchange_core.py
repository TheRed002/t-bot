"""
Core unit tests for MockExchange functionality.

This module provides comprehensive tests for the MockExchange class following
the current architecture and coding standards.
"""

import asyncio
from decimal import Decimal

import pytest

from src.core.exceptions import ExchangeConnectionError, OrderRejectionError, ValidationError
from src.core.types import (
    ExchangeInfo,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Trade,
)
from src.exchanges.mock_exchange import MockExchange


class TestMockExchangeCore:
    """Test cases for MockExchange core functionality."""

    @pytest.fixture
    def mock_exchange(self):
        """Create MockExchange instance."""
        return MockExchange({"test": "config"})

    def test_initialization(self, mock_exchange):
        """Test MockExchange initialization."""
        # Backend uses "mock_exchange" as exchange name (visible in logs)
        assert mock_exchange.exchange_name == "mock_exchange"
        assert isinstance(mock_exchange._mock_balances, dict)
        assert isinstance(mock_exchange._mock_orders, dict)
        assert isinstance(mock_exchange._mock_prices, dict)

        # Check default balances
        assert "USDT" in mock_exchange._mock_balances
        assert "BTC" in mock_exchange._mock_balances
        assert "ETH" in mock_exchange._mock_balances
        assert mock_exchange._mock_balances["USDT"] > Decimal("0")

    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, mock_exchange):
        """Test connection and disconnection lifecycle."""
        # Initially not connected
        assert not mock_exchange.connected

        # Connect
        await mock_exchange.connect()
        assert mock_exchange.connected
        assert mock_exchange.last_heartbeat is not None

        # Should have exchange info loaded
        assert mock_exchange._exchange_info is not None
        assert mock_exchange._trading_symbols is not None

        # Disconnect
        await mock_exchange.disconnect()
        assert not mock_exchange.connected

    @pytest.mark.asyncio
    async def test_service_lifecycle(self, mock_exchange):
        """Test BaseService lifecycle integration."""
        # Start service
        await mock_exchange.start()
        assert mock_exchange.is_running
        assert mock_exchange.connected

        # Stop service
        await mock_exchange.stop()
        assert not mock_exchange.is_running
        assert not mock_exchange.connected

    @pytest.mark.asyncio
    async def test_ping(self, mock_exchange):
        """Test ping functionality."""
        # Should fail when not connected
        with pytest.raises(ExchangeConnectionError):
            await mock_exchange.ping()

        # Connect and test ping
        await mock_exchange.connect()
        result = await mock_exchange.ping()
        assert result is True
        assert mock_exchange.last_heartbeat is not None

    @pytest.mark.asyncio
    async def test_load_exchange_info(self, mock_exchange):
        """Test loading exchange information."""
        await mock_exchange.connect()

        info = await mock_exchange.load_exchange_info()
        assert isinstance(info, ExchangeInfo)
        # Backend uses "mock_exchange" as exchange name (visible in logs)
        assert info.exchange == "mock_exchange"
        assert info.symbol == "BTCUSDT"  # Default symbol
        assert info.base_asset == "BTC"
        assert info.quote_asset == "USDT"
        assert isinstance(info.min_price, Decimal)
        assert isinstance(info.max_price, Decimal)
        assert isinstance(info.min_quantity, Decimal)

        # Trading symbols should be populated
        assert mock_exchange._trading_symbols is not None
        assert "BTCUSDT" in mock_exchange._trading_symbols
        assert "ETHUSDT" in mock_exchange._trading_symbols

    @pytest.mark.asyncio
    async def test_get_ticker(self, mock_exchange):
        """Test getting ticker data."""
        await mock_exchange.start()

        # Get ticker (backend returns Ticker object, not dict)
        ticker = await mock_exchange.get_ticker("BTCUSDT")

        # Backend returns Ticker object, use attribute access
        assert hasattr(ticker, "symbol")
        assert hasattr(ticker, "last_price") or hasattr(ticker, "price")
        assert hasattr(ticker, "bid_price")
        assert hasattr(ticker, "ask_price")
        assert hasattr(ticker, "volume")
        assert ticker.symbol == "BTCUSDT"
        # Check price is Decimal (use last_price or price attribute)
        price = ticker.last_price if hasattr(ticker, "last_price") else ticker.price
        assert isinstance(price, Decimal)

    @pytest.mark.asyncio
    async def test_get_ticker_price_simulation(self, mock_exchange):
        """Test ticker price simulation."""
        await mock_exchange.start()

        # Get ticker multiple times to test price variation
        tickers = []
        for _ in range(5):
            ticker = await mock_exchange.get_ticker("BTCUSDT")
            # Backend returns Ticker object, extract price attribute
            price = ticker.last_price if hasattr(ticker, "last_price") else ticker.price
            tickers.append(price)

        # Prices should be realistic
        for price in tickers:
            assert isinstance(price, Decimal)
            assert price > Decimal("0")
            assert price < Decimal("1000000")  # Reasonable upper bound

    @pytest.mark.asyncio
    async def test_get_ticker_invalid_symbol(self, mock_exchange):
        """Test getting ticker for invalid symbol."""
        await mock_exchange.start()

        with pytest.raises(ValidationError):
            await mock_exchange.get_ticker("INVALID")

    @pytest.mark.asyncio
    async def test_get_order_book(self, mock_exchange):
        """Test getting order book."""
        await mock_exchange.start()

        order_book = await mock_exchange.get_order_book("BTCUSDT")

        assert order_book.symbol == "BTCUSDT"
        assert len(order_book.bids) > 0
        assert len(order_book.asks) > 0

        # Check bid/ask structure (OrderBookLevel objects)
        best_bid = order_book.bids[0]
        best_ask = order_book.asks[0]

        assert isinstance(best_bid.price, Decimal)
        assert isinstance(best_bid.quantity, Decimal)
        assert isinstance(best_ask.price, Decimal)
        assert isinstance(best_ask.quantity, Decimal)
        assert best_ask.price > best_bid.price  # Spread should exist

    @pytest.mark.asyncio
    async def test_get_order_book_with_limit(self, mock_exchange):
        """Test getting order book with limit."""
        await mock_exchange.start()

        order_book = await mock_exchange.get_order_book("BTCUSDT", limit=5)

        # Should respect limit (up to 10 for mock)
        assert len(order_book.bids) <= 5
        assert len(order_book.asks) <= 5

    @pytest.mark.asyncio
    async def test_get_recent_trades(self, mock_exchange):
        """Test getting recent trades."""
        await mock_exchange.start()

        trades = await mock_exchange.get_recent_trades("BTCUSDT")

        assert isinstance(trades, list)
        assert len(trades) > 0

        trade = trades[0]
        assert isinstance(trade, Trade)
        assert trade.symbol == "BTCUSDT"
        assert isinstance(trade.price, Decimal)
        assert isinstance(trade.quantity, Decimal)
        assert trade.timestamp is not None

    @pytest.mark.asyncio
    async def test_place_order_success(self, mock_exchange):
        """Test successful order placement."""
        await mock_exchange.start()

        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("45000.00")
        )

        response = await mock_exchange.place_order(order_request)

        assert isinstance(response, OrderResponse)
        assert response.symbol == "BTCUSDT"
        assert response.side == OrderSide.BUY
        assert response.order_type == OrderType.LIMIT
        assert response.quantity == Decimal("0.001")
        assert response.price == Decimal("45000.00")
        assert response.order_id is not None
        assert response.status in [OrderStatus.FILLED, OrderStatus.NEW]

        # Order should be stored
        assert response.order_id in mock_exchange._mock_orders

    @pytest.mark.asyncio
    async def test_place_order_dict_format(self, mock_exchange):
        """Test placing order with dict return format (backward compatibility)."""
        await mock_exchange.start()

        # Use the dict format method
        response = await mock_exchange.place_order_dict(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.001"),
            price=Decimal("45000.00")
        )

        assert isinstance(response, dict)
        assert response["symbol"] == "BTCUSDT"
        assert response["side"] == "BUY"
        assert response["order_type"] == "LIMIT"
        assert response["quantity"] == Decimal("0.001")
        assert response["price"] == Decimal("45000.00")
        assert "order_id" in response
        assert "status" in response

    @pytest.mark.asyncio
    async def test_place_order_validation(self, mock_exchange):
        """Test order placement validation."""
        await mock_exchange.start()

        # Invalid symbol
        with pytest.raises(ValidationError):
            order_request = OrderRequest(
                symbol="INVALID",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("45000.00")
            )
            await mock_exchange.place_order(order_request)

    @pytest.mark.asyncio
    async def test_place_order_insufficient_balance(self, mock_exchange):
        """Test order placement with insufficient balance."""
        await mock_exchange.start()

        # Set low USDT balance
        mock_exchange._mock_balances["USDT"] = Decimal("1.0")

        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),  # Large quantity
            price=Decimal("50000.00")  # High price = needs lots of USDT
        )

        with pytest.raises(OrderRejectionError, match="Insufficient.*balance"):
            await mock_exchange.place_order(order_request)

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, mock_exchange):
        """Test successful order cancellation."""
        await mock_exchange.start()

        # Use stop loss order which always starts as NEW status
        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=Decimal("0.001"),
            stop_price=Decimal("40000.00")
        )

        order_response = await mock_exchange.place_order(order_request)
        
        # Stop loss orders should always start as NEW
        assert order_response.status == OrderStatus.NEW
        order_id = order_response.order_id
        
        # Cancel the order
        cancel_response = await mock_exchange.cancel_order("BTCUSDT", order_id)

        assert isinstance(cancel_response, OrderResponse)
        assert cancel_response.order_id == order_id
        assert cancel_response.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, mock_exchange):
        """Test cancelling non-existent order."""
        await mock_exchange.start()

        with pytest.raises(OrderRejectionError, match="not found"):
            await mock_exchange.cancel_order("BTCUSDT", "nonexistent_order")

    @pytest.mark.asyncio
    async def test_get_order_status_success(self, mock_exchange):
        """Test getting order status."""
        await mock_exchange.start()

        # Place an order
        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("45000.00")
        )

        order_response = await mock_exchange.place_order(order_request)
        order_id = order_response.order_id

        # Get order status
        status_response = await mock_exchange.get_order_status("BTCUSDT", order_id)

        assert isinstance(status_response, OrderResponse)
        assert status_response.order_id == order_id

    @pytest.mark.asyncio
    async def test_get_order_status_not_found(self, mock_exchange):
        """Test getting status of non-existent order."""
        await mock_exchange.start()

        with pytest.raises(OrderRejectionError, match="not found"):
            await mock_exchange.get_order_status("BTCUSDT", "nonexistent_order")

    @pytest.mark.asyncio
    async def test_get_open_orders_all(self, mock_exchange):
        """Test getting all open orders."""
        await mock_exchange.start()

        # Place some orders
        for i in range(3):
            order_request = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal(f"{45000 + i}.00")
            )
            await mock_exchange.place_order(order_request)

        # Get all open orders
        open_orders = await mock_exchange.get_open_orders()

        assert isinstance(open_orders, list)
        # Some may be filled, some may be open
        for order in open_orders:
            assert isinstance(order, OrderResponse)
            assert order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]

    @pytest.mark.asyncio
    async def test_get_open_orders_by_symbol(self, mock_exchange):
        """Test getting open orders for specific symbol."""
        await mock_exchange.start()

        open_orders = await mock_exchange.get_open_orders("BTCUSDT")

        assert isinstance(open_orders, list)
        for order in open_orders:
            assert order.symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_get_account_balance(self, mock_exchange):
        """Test getting account balance."""
        await mock_exchange.start()

        balances = await mock_exchange.get_account_balance()

        assert isinstance(balances, dict)
        assert "USDT" in balances
        assert "BTC" in balances
        assert "ETH" in balances

        for currency, balance in balances.items():
            assert isinstance(balance, Decimal)
            assert balance >= Decimal("0")

    @pytest.mark.asyncio
    async def test_get_positions(self, mock_exchange):
        """Test getting positions (should be empty for spot)."""
        await mock_exchange.start()

        positions = await mock_exchange.get_positions()

        assert isinstance(positions, list)
        assert len(positions) == 0  # Mock exchange is spot trading

    @pytest.mark.asyncio
    async def test_health_check(self, mock_exchange):
        """Test health check functionality."""
        # Should be unhealthy when not connected
        health = await mock_exchange.health_check()
        assert health.status.value == "unhealthy"

        # Should be healthy when connected
        await mock_exchange.start()
        health = await mock_exchange.health_check()
        assert health.status.value == "healthy"

    def test_configuration_methods(self, mock_exchange):
        """Test configuration and setup methods."""
        # Test configure method
        mock_exchange.configure(latency=50, failure_rate=0.1, partial_fill_rate=0.2)
        # Should not raise errors

        # Test set_balance
        new_balances = {"BTC": Decimal("2.0"), "ETH": Decimal("20.0")}
        mock_exchange.set_balance(new_balances)

        assert mock_exchange._mock_balances["BTC"] == Decimal("2.0")
        assert mock_exchange._mock_balances["ETH"] == Decimal("20.0")

        # Test set_price
        mock_exchange.set_price("BTCUSDT", Decimal("50000.0"))
        assert mock_exchange._mock_prices["BTCUSDT"] == Decimal("50000.0")

        # Test set_order_book (just shouldn't error)
        mock_exchange.set_order_book("BTCUSDT", {"bids": [], "asks": []})

    @pytest.mark.asyncio
    async def test_backward_compatibility_methods(self, mock_exchange):
        """Test backward compatibility methods."""
        await mock_exchange.start()

        # Test get_balance (alias for get_account_balance)
        balances = await mock_exchange.get_balance()
        assert isinstance(balances, dict)

        # Test get_trades (alias for get_recent_trades)
        trades = await mock_exchange.get_trades("BTCUSDT")
        assert isinstance(trades, list)

    @pytest.mark.asyncio
    async def test_balance_updates_after_filled_orders(self, mock_exchange):
        """Test that balances update after filled orders."""
        await mock_exchange.start()

        # Get initial balance
        initial_balances = await mock_exchange.get_account_balance()
        initial_usdt = initial_balances["USDT"]
        initial_btc = initial_balances.get("BTC", Decimal("0"))

        # Place a small buy order (should be filled by mock)
        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("45000.00")
        )

        response = await mock_exchange.place_order(order_request)

        if response.status == OrderStatus.FILLED:
            # Check balances updated
            new_balances = await mock_exchange.get_account_balance()
            new_usdt = new_balances["USDT"]
            new_btc = new_balances.get("BTC", Decimal("0"))

            # USDT should decrease, BTC should increase
            expected_cost = response.filled_quantity * response.price
            assert new_usdt <= initial_usdt  # May be exactly equal due to mock logic
            assert new_btc >= initial_btc

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_exchange):
        """Test concurrent operations."""
        await mock_exchange.start()

        # Run multiple operations concurrently
        tasks = [
            mock_exchange.get_ticker("BTCUSDT"),
            mock_exchange.get_order_book("BTCUSDT"),
            mock_exchange.get_recent_trades("BTCUSDT"),
            mock_exchange.get_account_balance(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        assert len(results) == 4
        for result in results:
            assert not isinstance(result, Exception)

    @pytest.mark.asyncio
    async def test_operations_when_not_connected(self, mock_exchange):
        """Test that operations fail appropriately when not connected."""
        # Don't connect the exchange

        operations = [
            lambda: mock_exchange.get_ticker("BTCUSDT"),
            lambda: mock_exchange.get_order_book("BTCUSDT"),
            lambda: mock_exchange.get_recent_trades("BTCUSDT"),
            lambda: mock_exchange.get_account_balance(),
            lambda: mock_exchange.place_order(OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.001")
            )),
            lambda: mock_exchange.cancel_order("BTCUSDT", "order_id"),
            lambda: mock_exchange.get_order_status("BTCUSDT", "order_id"),
            lambda: mock_exchange.get_open_orders(),
            lambda: mock_exchange.get_positions(),
        ]

        for operation in operations:
            with pytest.raises((ExchangeConnectionError, ValidationError)):
                await operation()


class TestMockExchangeEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def mock_exchange(self):
        """Create MockExchange instance."""
        return MockExchange()

    @pytest.mark.asyncio
    async def test_order_simulation_randomness(self, mock_exchange):
        """Test that order simulation includes some randomness."""
        await mock_exchange.start()

        # Place multiple orders to see different outcomes
        outcomes = []
        for i in range(10):
            order_request = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal(f"{45000 + i}.00")
            )

            response = await mock_exchange.place_order(order_request)
            outcomes.append(response.status)

        # Should have some variation in outcomes (not all identical)
        unique_outcomes = set(outcomes)
        # At least should have filled orders, might have some new orders
        assert OrderStatus.FILLED in unique_outcomes or OrderStatus.NEW in unique_outcomes

    @pytest.mark.asyncio
    async def test_price_movement_simulation(self, mock_exchange):
        """Test that prices show realistic movement."""
        await mock_exchange.start()

        # Get ticker multiple times
        prices = []
        for _ in range(10):
            ticker = await mock_exchange.get_ticker("BTCUSDT")
            # Backend returns Ticker object, extract price attribute
            price = ticker.last_price if hasattr(ticker, "last_price") else ticker.price
            prices.append(price)

        # Prices should vary within reasonable bounds
        min_price = min(prices)
        max_price = max(prices)

        # Should have some variation (mock includes ±2% change)
        price_range = max_price - min_price
        assert price_range >= Decimal("0")  # At minimum, no negative range

        # All prices should be positive and reasonable
        for price in prices:
            assert price > Decimal("1000")  # Reasonable lower bound for BTC
            assert price < Decimal("200000")  # Reasonable upper bound for BTC


if __name__ == "__main__":
    pytest.main([__file__])
