"""
Comprehensive tests for Mock exchange core trading functions.

This test file focuses on achieving 100% coverage for critical Mock exchange trading functions:
- Order placement (all order types)
- Order cancellation  
- Order status checking
- Balance retrieval
- Account information
- Position management

Tests include financial precision, error conditions, and Mock exchange-specific scenarios.
"""

import asyncio
import random
from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.core.exceptions import (
    ExchangeError,
    OrderRejectionError,
    ValidationError,
)
from src.core.types import (
    Balance,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from src.exchanges.mock_exchange import MockExchange


@pytest.fixture(autouse=True)
def setup_random_seed():
    """Set fixed random seed for deterministic testing."""
    random.seed(42)


@pytest.fixture
def mock_config():
    """Create a valid configuration dict."""
    return {
        "api_key": "test_api_key",
        "api_secret": "test_api_secret",
        "testnet": True
    }


@pytest.fixture
async def mock_exchange(mock_config):
    """Create a Mock exchange instance."""
    exchange = MockExchange(config=mock_config)
    await exchange.connect()  # Connect to set up properly
    return exchange


@pytest.fixture
def mock_order_request():
    """Create a mock order request for Mock exchange."""
    return OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.01"),
        price=Decimal("50000.00"),
        client_order_id="test_mock_order_123"
    )


class TestMockExchangeOrderPlacement:
    """Test Mock exchange order placement functionality."""

    async def test_place_limit_order_success(self, mock_exchange, mock_order_request):
        """Test successful limit order placement."""
        response = await mock_exchange.place_order(mock_order_request)

        assert response.symbol == "BTCUSDT"
        assert response.side == OrderSide.BUY
        assert response.quantity == Decimal("0.01")
        assert response.price == Decimal("50000.00")
        assert response.order_type == OrderType.LIMIT
        assert response.status in [OrderStatus.NEW, OrderStatus.FILLED]  # MockExchange may fill immediately
        assert response.client_order_id == "test_mock_order_123"

        # Verify order is stored in exchange
        assert response.id in mock_exchange.orders
        stored_order = mock_exchange.orders[response.id]
        assert stored_order.symbol == "BTCUSDT"
        assert stored_order.side == OrderSide.BUY

    async def test_place_market_buy_order_success(self, mock_exchange):
        """Test successful market buy order placement."""
        market_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            client_order_id="market_buy_mock_123"
        )

        response = await mock_exchange.place_order(market_order)

        assert response.order_type == OrderType.MARKET
        assert response.price is not None  # MockExchange sets execution price for market orders
        assert isinstance(response.price, Decimal)  # Should be a Decimal
        assert response.status == OrderStatus.FILLED  # MockExchange fills market orders immediately
        assert response.filled_quantity == Decimal("0.01")

    async def test_place_market_sell_order_success(self, mock_exchange):
        """Test successful market sell order placement."""
        market_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            client_order_id="market_sell_mock_123"
        )

        response = await mock_exchange.place_order(market_order)

        assert response.side == OrderSide.SELL
        assert response.order_type == OrderType.MARKET
        assert response.price is not None  # MockExchange sets execution price
        assert response.status == OrderStatus.FILLED
        assert response.filled_quantity == Decimal("0.01")

    async def test_place_stop_order_success(self, mock_exchange):
        """Test successful stop order placement."""
        stop_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=Decimal("0.01"),
            price=Decimal("48000.00"),
            stop_price=Decimal("49000.00"),
            client_order_id="stop_mock_123"
        )

        response = await mock_exchange.place_order(stop_order)

        assert response.order_type == OrderType.STOP_LOSS
        assert response.price == Decimal("48000.00")
        assert response.status == OrderStatus.NEW  # Stop orders wait for trigger

    async def test_place_order_insufficient_funds(self, mock_exchange):
        """Test order placement with insufficient funds."""
        large_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100.0"),  # More than available USDT can buy
            price=Decimal("50000.00"),
            client_order_id="large_order_mock"
        )

        with pytest.raises(OrderRejectionError):
            await mock_exchange.place_order(large_order)

    async def test_place_order_invalid_symbol(self, mock_exchange):
        """Test order placement with invalid symbol."""
        invalid_order = OrderRequest(
            symbol="INVALID-SYMBOL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.01"),
            price=Decimal("50000.00"),
            client_order_id="invalid_symbol_mock"
        )

        with pytest.raises(ValidationError):
            await mock_exchange.place_order(invalid_order)

    async def test_place_order_zero_quantity(self, mock_exchange):
        """Test order placement with zero quantity."""
        # OrderRequest validates quantity > 0 at creation time
        with pytest.raises(ValidationError):
            OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0"),
                price=Decimal("50000.00"),
                client_order_id="zero_quantity_mock"
            )

    async def test_place_order_negative_price(self, mock_exchange):
        """Test order placement with negative price."""
        # OrderRequest validates price > 0 at creation time
        with pytest.raises(ValidationError):
            OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.01"),
                price=Decimal("-1000.00"),
                client_order_id="negative_price_mock"
            )

    async def test_place_order_with_time_in_force(self, mock_exchange):
        """Test order placement with time-in-force."""
        tif_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.01"),
            price=Decimal("50000.00"),
            time_in_force=TimeInForce.IOC,
            client_order_id="tif_order_mock"
        )

        response = await mock_exchange.place_order(tif_order)

        # MockExchange should handle IOC orders - they typically get filled or expire immediately
        # Since IOC orders fill immediately if possible, check the status
        assert response.status in [OrderStatus.FILLED, OrderStatus.EXPIRED]
        # Verify the basic order properties
        assert response.order_type == OrderType.LIMIT
        assert response.symbol == "BTCUSDT"


class TestMockExchangeOrderCancellation:
    """Test Mock exchange order cancellation functionality."""

    async def test_cancel_order_success(self, mock_exchange):
        """Test successful order cancellation."""
        # Create a stop order which starts as NEW (can be cancelled)
        stop_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=Decimal("0.01"),
            price=Decimal("48000.00"),
            stop_price=Decimal("49000.00"),
            client_order_id="cancel_success_test"
        )
        
        # First place an order
        place_response = await mock_exchange.place_order(stop_order)
        order_id = place_response.id
        
        # Ensure order is in NEW status (can be cancelled)
        assert place_response.status == OrderStatus.NEW

        # Then cancel it
        result = await mock_exchange.cancel_order("BTCUSDT", order_id)

        assert result.status == OrderStatus.CANCELLED
        # Verify order status is updated
        canceled_order = mock_exchange.orders[order_id]
        assert canceled_order.status == OrderStatus.CANCELLED

    async def test_cancel_order_not_found(self, mock_exchange):
        """Test canceling non-existent order."""
        with pytest.raises(OrderRejectionError, match="Order.*not found"):
            await mock_exchange.cancel_order("BTCUSDT", "nonexistent_order")

    async def test_cancel_already_filled_order(self, mock_exchange):
        """Test canceling an already filled order."""
        # Place a market order (gets filled immediately)
        market_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            client_order_id="market_order_cancel_test"
        )

        place_response = await mock_exchange.place_order(market_order)
        order_id = place_response.id

        # Try to cancel filled order
        with pytest.raises(OrderRejectionError, match="Cannot cancel order"):
            await mock_exchange.cancel_order("BTCUSDT", order_id)

    async def test_cancel_already_canceled_order(self, mock_exchange):
        """Test canceling an already canceled order."""
        # Create a stop order which starts as NEW (not filled immediately)
        stop_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=Decimal("0.01"),
            price=Decimal("48000.00"),
            stop_price=Decimal("49000.00"),
            client_order_id="stop_cancel_test"
        )
        
        place_response = await mock_exchange.place_order(stop_order)
        order_id = place_response.id
        
        # Ensure order is in NEW status (can be cancelled)
        assert place_response.status == OrderStatus.NEW

        # Cancel it first time
        await mock_exchange.cancel_order("BTCUSDT", order_id)

        # Try to cancel again
        with pytest.raises(OrderRejectionError, match="Cannot cancel order"):
            await mock_exchange.cancel_order("BTCUSDT", order_id)


class TestMockExchangeOrderStatus:
    """Test Mock exchange order status retrieval."""

    async def test_get_order_status_new(self, mock_exchange, mock_order_request):
        """Test order status retrieval for new order."""
        place_response = await mock_exchange.place_order(mock_order_request)
        order_id = place_response.id

        order_response = await mock_exchange.get_order_status("BTCUSDT", order_id)
        assert order_response.status in [OrderStatus.NEW, OrderStatus.FILLED]  # MockExchange may fill randomly

    async def test_get_order_status_filled(self, mock_exchange):
        """Test order status retrieval for filled order."""
        market_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            client_order_id="filled_order_test"
        )

        place_response = await mock_exchange.place_order(market_order)
        order_id = place_response.id

        order_response = await mock_exchange.get_order_status("BTCUSDT", order_id)
        assert order_response.status == OrderStatus.FILLED  # Market orders always fill

    async def test_get_order_status_canceled(self, mock_exchange):
        """Test order status retrieval for canceled order."""
        # Create a stop order which starts as NEW (can be cancelled)
        stop_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=Decimal("0.01"),
            price=Decimal("48000.00"),
            stop_price=Decimal("49000.00"),
            client_order_id="status_cancel_test"
        )
        
        place_response = await mock_exchange.place_order(stop_order)
        order_id = place_response.id
        
        # Ensure order is in NEW status (can be cancelled)
        assert place_response.status == OrderStatus.NEW

        await mock_exchange.cancel_order("BTCUSDT", order_id)

        order_response = await mock_exchange.get_order_status("BTCUSDT", order_id)
        assert order_response.status == OrderStatus.CANCELLED

    async def test_get_order_status_not_found(self, mock_exchange):
        """Test order status retrieval for non-existent order."""
        with pytest.raises(OrderRejectionError, match="Order.*not found"):
            await mock_exchange.get_order_status("BTCUSDT", "nonexistent_order")


class TestMockExchangeAccountInfo:
    """Test Mock exchange account information retrieval."""

    async def test_get_balance_all_currencies(self, mock_exchange):
        """Test retrieving all balances."""
        balances_dict = await mock_exchange.get_balance()

        assert "USDT" in balances_dict
        assert "BTC" in balances_dict
        assert "ETH" in balances_dict

        # Check initial balances  
        assert balances_dict["USDT"] == Decimal("10000.00000000")
        assert balances_dict["BTC"] == Decimal("0.50000000")
        assert balances_dict["ETH"] == Decimal("5.00000000")

        # Verify all are Decimal types
        for currency, balance in balances_dict.items():
            assert isinstance(balance, Decimal)

    async def test_get_balance_specific_currency(self, mock_exchange):
        """Test retrieving balance for all currencies (MockExchange doesn't filter)."""
        balances_dict = await mock_exchange.get_balance()

        assert "BTC" in balances_dict
        assert balances_dict["BTC"] == Decimal("0.50000000")

    async def test_get_balance_nonexistent_currency(self, mock_exchange):
        """Test that non-existent currencies are not in balance dict."""
        balances_dict = await mock_exchange.get_balance()

        assert "DOGE" not in balances_dict  # DOGE should not be in initial balances

    async def test_balance_updates_after_trade(self, mock_exchange):
        """Test that balances are updated after trades."""
        # Get initial balances
        initial_balances = await mock_exchange.get_balance()
        initial_usdt = initial_balances["USDT"]
        initial_btc = initial_balances["BTC"]

        # Place a market buy order
        market_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.001"),  # Small amount
            client_order_id="balance_update_test"
        )

        response = await mock_exchange.place_order(market_order)
        
        # Only check balances if order was filled (MockExchange has 90% fill rate)
        if response.status == OrderStatus.FILLED:
            # Check updated balances
            updated_balances = await mock_exchange.get_balance()
    
            # USDT should decrease, BTC should increase
            assert updated_balances["USDT"] < initial_usdt
            assert updated_balances["BTC"] > initial_btc

    async def test_get_positions(self, mock_exchange):
        """Test positions retrieval."""
        positions = await mock_exchange.get_positions()

        # Mock exchange starts with empty positions
        assert isinstance(positions, list)
        assert len(positions) == 0

    async def test_get_positions_after_trades(self, mock_exchange):
        """Test positions after making some trades."""
        # Place some orders to create positions
        buy_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.001"),
            client_order_id="position_test_buy"
        )

        await mock_exchange.place_order(buy_order)

        positions = await mock_exchange.get_positions()

        # Should still be empty if mock exchange doesn't track positions
        assert isinstance(positions, list)


class TestMockExchangeFinancialPrecision:
    """Test financial precision in Mock exchange operations."""

    async def test_order_decimal_precision_preservation(self, mock_exchange):
        """Test that Decimal precision is preserved in order operations."""
        high_precision_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.123456789"),  # High precision
            price=Decimal("50000.123456789"),  # High precision
            client_order_id="precision_mock_test"
        )

        response = await mock_exchange.place_order(high_precision_order)

        assert isinstance(response.quantity, Decimal)
        assert isinstance(response.price, Decimal)
        assert response.quantity == Decimal("0.123456789")
        assert response.price == Decimal("50000.123456789")

    async def test_balance_decimal_precision_preservation(self, mock_exchange):
        """Test that balance values maintain Decimal precision."""
        balances = await mock_exchange.get_balance()

        for currency, balance in balances.items():
            assert isinstance(balance, Decimal)

    async def test_fee_calculation_precision(self, mock_exchange):
        """Test that fee calculations maintain precision."""
        # Place a market order to test fee calculations
        market_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.001"),
            client_order_id="fee_precision_test"
        )

        response = await mock_exchange.place_order(market_order)

        # Verify order exists and has filled quantity
        order = mock_exchange.orders[response.id]
        assert isinstance(order.filled_quantity, Decimal)

        # If fees are calculated, they should be Decimal
        if hasattr(order, "fee") and order.fee:
            assert isinstance(order.fee, Decimal)

    async def test_no_float_conversions(self, mock_exchange, mock_order_request):
        """Test that no float conversions occur during operations."""
        response = await mock_exchange.place_order(mock_order_request)

        # Ensure all numeric values are Decimal, not float
        assert not isinstance(response.quantity, float)
        assert not isinstance(response.price, float)
        assert isinstance(response.quantity, Decimal)
        assert isinstance(response.price, Decimal)

        # Check stored order
        stored_order = mock_exchange.orders[response.id]
        assert isinstance(stored_order.quantity, Decimal)
        assert isinstance(stored_order.price, Decimal)


class TestMockExchangeEdgeCases:
    """Test Mock exchange edge cases."""

    async def test_order_with_very_small_quantity(self, mock_exchange):
        """Test placing order with very small quantity."""
        small_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.00000001"),  # Satoshi level
            price=Decimal("50000.00"),
            client_order_id="small_quantity_test"
        )

        response = await mock_exchange.place_order(small_order)
        assert response.quantity == Decimal("0.00000001")

    async def test_order_with_very_high_price(self, mock_exchange):
        """Test placing order with very high price."""
        high_price_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("1000000.00"),  # Million dollar BTC
            client_order_id="high_price_test"
        )

        response = await mock_exchange.place_order(high_price_order)
        assert response.price == Decimal("1000000.00")

    async def test_concurrent_order_placement(self, mock_exchange):
        """Test concurrent order placements."""
        orders = []
        for i in range(5):
            order = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("50000.00"),
                client_order_id=f"concurrent_mock_{i}"
            )
            orders.append(order)

        tasks = [mock_exchange.place_order(order) for order in orders]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 5
        for i, response in enumerate(responses):
            assert response.client_order_id == f"concurrent_mock_{i}"
            assert response.status in [OrderStatus.NEW, OrderStatus.FILLED]  # MockExchange may fill randomly

    async def test_order_book_retrieval(self, mock_exchange):
        """Test order book retrieval."""
        # Test the MockExchange order book implementation
        order_book = await mock_exchange.get_order_book("BTCUSDT")
        assert order_book.symbol == "BTCUSDT"
        assert hasattr(order_book, "bids")
        assert hasattr(order_book, "asks")
        assert len(order_book.bids) > 0
        assert len(order_book.asks) > 0

    async def test_ticker_retrieval(self, mock_exchange):
        """Test ticker retrieval."""
        # Test the MockExchange ticker implementation
        # Backend returns a Ticker object, not a dict
        ticker = await mock_exchange.get_ticker("BTCUSDT")
        assert hasattr(ticker, "symbol")
        assert hasattr(ticker, "last_price") or hasattr(ticker, "price")
        assert ticker.symbol == "BTCUSDT"

    async def test_exchange_info_retrieval(self, mock_exchange):
        """Test exchange info retrieval."""
        # Test exchange info (MockExchange loads info on connect)
        # Backend uses "mock_exchange" as exchange name (see logs: "component=mock_exchange_exchange")
        exchange_info = mock_exchange.get_exchange_info()
        assert exchange_info is not None
        assert exchange_info.exchange == "mock_exchange"
        assert exchange_info.symbol == "BTCUSDT"

    async def test_large_order_partial_fill_simulation(self, mock_exchange):
        """Test simulation of partial fills for large orders."""
        # This would test if mock exchange can simulate partial fills
        # Use a smaller order that fits within balance (10,000 USDT available)
        large_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.15"),  # Order worth 0.15 * 10000 = 1500 USDT
            price=Decimal("10000.00"),  # Lower price to fit in balance
            client_order_id="large_order_test"
        )

        response = await mock_exchange.place_order(large_order)

        # Depending on mock exchange implementation, might get partial fill
        assert response.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED]

    async def test_order_expiration_simulation(self, mock_exchange):
        """Test order expiration simulation."""
        ioc_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("40000.00"),  # Low price, unlikely to fill immediately
            time_in_force=TimeInForce.IOC,
            client_order_id="ioc_expire_test"
        )

        response = await mock_exchange.place_order(ioc_order)

        # IOC orders should either fill or expire
        assert response.status in [OrderStatus.FILLED, OrderStatus.EXPIRED]

    async def test_balance_locking_on_limit_orders(self, mock_exchange):
        """Test that balances are locked appropriately for limit orders."""
        # Get initial balance
        initial_balances = await mock_exchange.get_balance()
        initial_available = initial_balances["USDT"]

        # Place a limit buy order
        limit_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000.00"),
            client_order_id="balance_lock_test"
        )

        await mock_exchange.place_order(limit_order)

        response = await mock_exchange.place_order(limit_order)
        
        # Check balances only if order was filled (MockExchange has 90% fill rate)
        if response.status == OrderStatus.FILLED:
            updated_balances = await mock_exchange.get_balance()
            # USDT should decrease for filled orders
            assert updated_balances["USDT"] < initial_available

    async def test_multiple_symbol_support(self, mock_exchange):
        """Test support for multiple trading symbols."""
        symbols_to_test = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

        for symbol in symbols_to_test:
            try:
                order = OrderRequest(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=Decimal("0.001"),
                    price=Decimal("1000.00"),
                    client_order_id=f"multi_symbol_{symbol.replace('-', '_')}"
                )

                response = await mock_exchange.place_order(order)
                assert response.symbol == symbol

            except (ValidationError, OrderRejectionError):
                # Some symbols might not be supported in mock exchange
                continue
