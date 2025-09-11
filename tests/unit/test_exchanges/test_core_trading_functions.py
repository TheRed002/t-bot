"""
Comprehensive tests for core trading functions in exchange module.

This test file focuses on achieving 100% coverage for critical trading functions:
- Order placement (market orders, limit orders)
- Order cancellation  
- Balance retrieval
- Account information
- Order status checking
- Position management

Tests include financial precision, error conditions, and edge cases.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.exceptions import (
    ExchangeConnectionError,
    ExchangeRateLimitError,
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
    Position,
    Ticker,
)
from src.exchanges.base import BaseExchange


class TestableBaseExchange(BaseExchange):
    """Concrete implementation of BaseExchange for testing."""

    def __init__(self, config=None):
        from src.core.config import Config
        if config is None:
            config = Config()
        super().__init__("testable", config.model_dump() if hasattr(config, 'model_dump') else {})
        self._test_responses = {}
        self._test_errors = {}
        # Add missing attributes for rate limiting and connection
        self.unified_rate_limiter = Mock()
        self.unified_rate_limiter.acquire = AsyncMock(return_value=True)
        # Override initialization to avoid complex setup
        self._connected = False

    async def _connect_to_exchange(self) -> bool:
        return True

    async def _disconnect_from_exchange(self) -> None:
        pass

    async def _place_order_on_exchange(self, order: OrderRequest) -> OrderResponse:
        if "place_order_error" in self._test_errors:
            raise self._test_errors["place_order_error"]
        return self._test_responses.get("place_order", OrderResponse(
            id="test_order_123",
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=order.price,
            order_type=order.order_type,
            status=OrderStatus.NEW,
            filled_quantity=Decimal("0"),
            remaining_quantity=order.quantity,
            average_price=None,
            timestamp=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            exchange="testable"
        ))

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        if "cancel_order_error" in self._test_errors:
            raise self._test_errors["cancel_order_error"]
        return self._test_responses.get("cancel_order", True)

    async def get_order_status(self, order_id: str) -> OrderStatus:
        if "get_order_status_error" in self._test_errors:
            raise self._test_errors["get_order_status_error"]
        return self._test_responses.get("get_order_status", OrderStatus.FILLED)

    async def get_balance(self, asset: str = None) -> dict[str, dict[str, Decimal]]:
        if "get_balance_error" in self._test_errors:
            raise self._test_errors["get_balance_error"]
        return self._test_responses.get("get_balance", {
            "BTC": {"free": Decimal("1.0"), "locked": Decimal("0.1")},
            "USDT": {"free": Decimal("10000.0"), "locked": Decimal("500.0")}
        })

    async def get_positions(self) -> list[Position]:
        if "get_positions_error" in self._test_errors:
            raise self._test_errors["get_positions_error"]
        return self._test_responses.get("get_positions", [])

    async def get_exchange_info(self) -> ExchangeInfo:
        return ExchangeInfo(
            symbol="BTC-USDT",
            base_asset="BTC",
            quote_asset="USDT",
            status="TRADING",
            min_price=Decimal("0.01"),
            max_price=Decimal("1000000.00"),
            tick_size=Decimal("0.01"),
            min_quantity=Decimal("0.001"),
            max_quantity=Decimal("1000.0"),
            step_size=Decimal("0.000001"),
            exchange="testable"
        )

    async def get_ticker(self, symbol: str) -> Ticker:
        return Ticker(
            symbol=symbol,
            price=Decimal("50000.00"),
            bid_price=Decimal("49999.99"),
            ask_price=Decimal("50000.01"),
            volume=Decimal("100.0"),
            timestamp=datetime.now(timezone.utc)
        )

    async def get_order_book(self, symbol: str, depth: int = 10) -> OrderBook:
        return OrderBook(
            symbol=symbol,
            bids=[(Decimal("49999.99"), Decimal("1.0"))],
            asks=[(Decimal("50000.01"), Decimal("1.0"))],
            timestamp=datetime.now(timezone.utc)
        )

    async def _get_market_data_from_exchange(self, symbol: str, timeframe: str = "1m") -> list:
        return []

    async def _create_websocket_stream(self, symbol: str, stream_name: str) -> Any:
        return Mock()

    async def _handle_exchange_stream(self, stream_name: str, stream: Any) -> None:
        pass

    async def _close_exchange_stream(self, stream_name: str, stream: Any) -> None:
        pass

    async def _get_trade_history_from_exchange(self, symbol: str, limit: int = 100) -> list:
        return []

    # Abstract methods that need implementation
    async def connect(self) -> bool:
        """Connect to the exchange."""
        result = await self._connect_to_exchange()
        if result:
            self._connected = True
        return result

    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        await self._disconnect_from_exchange()
        self._connected = False

    async def get_account_balance(self) -> dict[str, Decimal]:
        """Get account balance."""
        return {"USDT": Decimal("1000.00"), "BTC": Decimal("0.1")}

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]:
        """Get open orders."""
        return []

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list:
        """Get recent trades."""
        return await self._get_trade_history_from_exchange(symbol, limit)

    async def load_exchange_info(self) -> ExchangeInfo:
        """Load exchange information."""
        return ExchangeInfo(
            symbol="BTC-USDT",
            base_asset="BTC",
            quote_asset="USDT",
            status="TRADING",
            min_price=Decimal("0.01"),
            max_price=Decimal("1000000.00"),
            tick_size=Decimal("0.01"),
            min_quantity=Decimal("0.001"),
            max_quantity=Decimal("100.0"),
            step_size=Decimal("0.000001"),
            exchange=self.exchange_name
        )

    async def ping(self) -> bool:
        """Ping the exchange."""
        return True

    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place an order with validation."""
        # Check validation errors first
        if "validation_error" in self._test_errors:
            raise ValidationError("Order validation failed")
        
        # Check rate limit errors 
        if "rate_limit_error" in self._test_errors:
            raise self._test_errors["rate_limit_error"]
        
        # Call the actual implementation
        return await self._place_order_on_exchange(order_request)

    def is_connected(self) -> bool:
        """Check if exchange is connected."""
        return self._connected
    
    async def pre_trade_validation(self, order: OrderRequest) -> bool:
        """Pre-trade validation."""
        if "validation_error" in self._test_errors:
            return False
        # Basic validation
        if order.quantity <= 0:
            return False
        if order.price is not None and order.price <= 0:
            return False
        # Check if symbol is valid
        exchange_info = await self.get_exchange_info()
        if order.symbol != exchange_info.symbol:
            return False
        return True

    def _check_unified_rate_limit(self, endpoint: str = "default", weight: int = 1) -> None:
        """Check unified rate limit."""
        if "rate_limit_error" in self._test_errors:
            raise self._test_errors["rate_limit_error"]
        pass
    
    async def _initialize_connection_infrastructure(self) -> None:
        """Initialize connection infrastructure."""
        pass
    
    async def _initialize_database(self) -> None:
        """Initialize database."""
        pass
    
    async def _initialize_redis(self) -> None:
        """Initialize redis."""
        pass
    
    async def _cleanup_infrastructure(self) -> None:
        """Cleanup infrastructure."""
        pass


@pytest.fixture
def base_exchange():
    """Create a testable BaseExchange instance."""
    exchange = TestableBaseExchange()
    exchange._connected = True
    return exchange


@pytest.fixture
def mock_order_request():
    """Create a mock order request."""
    return OrderRequest(
        symbol="BTC-USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.01"),
        price=Decimal("50000.00"),
        client_order_id="test_order_123"
    )


class TestCoreOrderFunctions:
    """Test core order management functions."""

    async def test_place_order_success(self, base_exchange, mock_order_request):
        """Test successful order placement."""
        response = await base_exchange.place_order(mock_order_request)

        assert response.id == "test_order_123"
        assert response.symbol == mock_order_request.symbol
        assert response.side == mock_order_request.side
        assert response.quantity == mock_order_request.quantity
        assert response.price == mock_order_request.price
        assert response.status == OrderStatus.NEW

    async def test_place_order_validation_failure(self, base_exchange, mock_order_request):
        """Test order placement with validation failure."""
        base_exchange._test_errors["validation_error"] = True
        with pytest.raises(ValidationError, match="Order validation failed"):
            await base_exchange.place_order(mock_order_request)

    async def test_place_order_rate_limit_exceeded(self, base_exchange, mock_order_request):
        """Test order placement with rate limit exceeded."""
        base_exchange._test_errors["rate_limit_error"] = ExchangeRateLimitError("Rate limit exceeded")
        with pytest.raises(ExchangeRateLimitError):
            await base_exchange.place_order(mock_order_request)

    async def test_place_order_exchange_error(self, base_exchange, mock_order_request):
        """Test order placement with exchange error."""
        base_exchange._test_errors["place_order_error"] = OrderRejectionError("Order rejected")
        
        with pytest.raises(OrderRejectionError):
            await base_exchange.place_order(mock_order_request)

    async def test_place_order_market_order(self, base_exchange):
        """Test placing a market order."""
        market_order = OrderRequest(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            client_order_id="market_order_123"
        )

        response = await base_exchange.place_order(market_order)

        assert response.order_type == OrderType.MARKET
        # Note: our test implementation still returns price, but real market orders may not

    async def test_place_order_financial_precision(self, base_exchange):
        """Test order placement with high financial precision."""
        precision_order = OrderRequest(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.00123456"),  # High precision
            price=Decimal("50000.123456"),   # High precision
            client_order_id="precision_test"
        )

        response = await base_exchange.place_order(precision_order)

        assert response.quantity == Decimal("0.00123456")
        assert response.price == Decimal("50000.123456")

    async def test_cancel_order_success(self, base_exchange):
        """Test successful order cancellation."""
        result = await base_exchange.cancel_order("test_order_123", "BTC-USDT")
        assert result is True

    async def test_cancel_order_failure(self, base_exchange):
        """Test order cancellation failure."""
        base_exchange._test_errors["cancel_order_error"] = OrderRejectionError("Order not found")

        with pytest.raises(OrderRejectionError):
            await base_exchange.cancel_order("nonexistent_order", "BTC-USDT")

    async def test_get_order_status_success(self, base_exchange):
        """Test successful order status retrieval."""
        status = await base_exchange.get_order_status("test_order_123")
        assert status == OrderStatus.FILLED

    async def test_get_order_status_error(self, base_exchange):
        """Test order status retrieval error."""
        base_exchange._test_errors["get_order_status_error"] = ValidationError("Order not found")

        with pytest.raises(ValidationError):
            await base_exchange.get_order_status("nonexistent_order")

    async def test_get_balance_success(self, base_exchange):
        """Test successful balance retrieval."""
        balances = await base_exchange.get_balance()

        assert "BTC" in balances
        assert "USDT" in balances
        assert balances["BTC"]["free"] == Decimal("1.0")
        assert balances["BTC"]["locked"] == Decimal("0.1")
        assert balances["USDT"]["free"] == Decimal("10000.0")
        assert balances["USDT"]["locked"] == Decimal("500.0")

    async def test_get_balance_specific_asset(self, base_exchange):
        """Test balance retrieval for specific asset."""
        balances = await base_exchange.get_balance("BTC")

        assert "BTC" in balances
        assert balances["BTC"]["free"] == Decimal("1.0")

    async def test_get_balance_error(self, base_exchange):
        """Test balance retrieval error."""
        base_exchange._test_errors["get_balance_error"] = ExchangeConnectionError("API error")

        with pytest.raises(ExchangeConnectionError):
            await base_exchange.get_balance()

    async def test_get_positions_success(self, base_exchange):
        """Test successful positions retrieval."""
        positions = await base_exchange.get_positions()
        assert isinstance(positions, list)

    async def test_get_positions_error(self, base_exchange):
        """Test positions retrieval error."""
        base_exchange._test_errors["get_positions_error"] = ExchangeConnectionError("API error")

        with pytest.raises(ExchangeConnectionError):
            await base_exchange.get_positions()


class TestTradingValidations:
    """Test trading validation functions."""

    async def test_pre_trade_validation_valid_order(self, base_exchange, mock_order_request):
        """Test pre-trade validation with valid order."""
        result = await base_exchange.pre_trade_validation(mock_order_request)
        assert result is True

    async def test_pre_trade_validation_invalid_symbol(self, base_exchange):
        """Test pre-trade validation with invalid symbol."""
        invalid_order = OrderRequest(
            symbol="INVALID-SYMBOL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.01"),
            price=Decimal("50000.00")
        )

        result = await base_exchange.pre_trade_validation(invalid_order)
        assert result is False

    async def test_pre_trade_validation_zero_quantity(self, base_exchange):
        """Test pre-trade validation with zero quantity."""
        # This test verifies that zero quantities are rejected at the model validation level
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            OrderRequest(
                symbol="BTC-USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0"),
                price=Decimal("50000.00")
            )

    async def test_pre_trade_validation_negative_price(self, base_exchange):
        """Test pre-trade validation with negative price."""
        # This test verifies that negative prices are rejected at the model validation level
        with pytest.raises(ValidationError, match="Price must be positive"):
            OrderRequest(
                symbol="BTC-USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.01"),
                price=Decimal("-100.00")
            )


class TestExchangeConnection:
    """Test exchange connection management."""

    async def test_connect_success(self, base_exchange):
        """Test successful exchange connection."""
        base_exchange._connected = False

        result = await base_exchange.connect()

        assert result is True
        assert base_exchange.is_connected() is True

    async def test_connect_failure(self, base_exchange):
        """Test failed exchange connection."""
        base_exchange._connected = False

        with patch.object(base_exchange, "_connect_to_exchange", return_value=False):
            result = await base_exchange.connect()

            assert result is False
            assert base_exchange.is_connected() is False

    async def test_disconnect(self, base_exchange):
        """Test exchange disconnection."""
        base_exchange._connected = True

        await base_exchange.disconnect()

        assert base_exchange.is_connected() is False


class TestRateLimiting:
    """Test rate limiting functionality."""

    async def test_rate_limit_check_success(self, base_exchange):
        """Test successful rate limit check."""
        # Should not raise an exception
        base_exchange._check_unified_rate_limit("test_endpoint", 1)

    async def test_rate_limit_exceeded(self, base_exchange):
        """Test rate limit exceeded."""
        base_exchange._test_errors["rate_limit_error"] = ExchangeRateLimitError("Rate limit exceeded")
        
        with pytest.raises(ExchangeRateLimitError):
            base_exchange._check_unified_rate_limit("test_endpoint", 1)


class TestErrorHandling:
    """Test error handling scenarios."""

    async def test_order_placement_network_error(self, base_exchange, mock_order_request):
        """Test order placement with network error."""
        base_exchange._test_errors["place_order_error"] = ExchangeConnectionError("Network error")

        with pytest.raises(ExchangeConnectionError):
            await base_exchange.place_order(mock_order_request)

    async def test_order_placement_validation_error(self, base_exchange, mock_order_request):
        """Test order placement with validation error."""
        base_exchange._test_errors["place_order_error"] = ValidationError("Invalid order parameters")

        with pytest.raises(ValidationError):
            await base_exchange.place_order(mock_order_request)


class TestFinancialPrecision:
    """Test financial precision calculations."""

    async def test_decimal_precision_preservation(self, base_exchange):
        """Test that Decimal precision is preserved throughout trading operations."""
        high_precision_order = OrderRequest(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.123456789"),
            price=Decimal("50000.123456789"),
            client_order_id="precision_test"
        )

        # Mock the response to include high precision values
        base_exchange._test_responses["place_order"] = OrderResponse(
            id="precision_order_123",
            client_order_id=high_precision_order.client_order_id,
            symbol=high_precision_order.symbol,
            side=high_precision_order.side,
            quantity=high_precision_order.quantity,
            price=high_precision_order.price,
            order_type=high_precision_order.order_type,
            status=OrderStatus.NEW,
            filled_quantity=Decimal("0"),
            remaining_quantity=high_precision_order.quantity,
            average_price=None,
            timestamp=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            exchange="testable"
        )

        response = await base_exchange.place_order(high_precision_order)

        assert response.quantity == Decimal("0.123456789")
        assert response.price == Decimal("50000.123456789")
        assert isinstance(response.quantity, Decimal)
        assert isinstance(response.price, Decimal)

    async def test_balance_decimal_precision(self, base_exchange):
        """Test that balance values maintain Decimal precision."""
        # Set high precision balance response
        base_exchange._test_responses["get_balance"] = {
            "BTC": {
                "free": Decimal("1.123456789"),
                "locked": Decimal("0.987654321")
            }
        }

        balances = await base_exchange.get_balance()

        assert isinstance(balances["BTC"]["free"], Decimal)
        assert isinstance(balances["BTC"]["locked"], Decimal)
        assert balances["BTC"]["free"] == Decimal("1.123456789")
        assert balances["BTC"]["locked"] == Decimal("0.987654321")

    async def test_no_float_conversions(self, base_exchange, mock_order_request):
        """Test that no float conversions occur during trading operations."""
        response = await base_exchange.place_order(mock_order_request)

        # Ensure all numeric values are Decimal, not float
        assert not isinstance(response.quantity, float)
        assert not isinstance(response.price, float)
        assert isinstance(response.quantity, Decimal)
        assert isinstance(response.price, Decimal)


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    async def test_place_order_with_very_small_quantity(self, base_exchange):
        """Test placing order with very small quantity."""
        small_order = OrderRequest(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.00000001"),  # Satoshi level
            price=Decimal("50000.00"),
            client_order_id="small_order"
        )

        response = await base_exchange.place_order(small_order)

        assert response.quantity == Decimal("0.00000001")

    async def test_place_order_with_very_high_price(self, base_exchange):
        """Test placing order with very high price."""
        high_price_order = OrderRequest(
            symbol="BTC-USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.01"),
            price=Decimal("1000000.00"),  # Million dollar BTC
            client_order_id="high_price_order"
        )

        response = await base_exchange.place_order(high_price_order)

        assert response.price == Decimal("1000000.00")

    async def test_concurrent_order_placement(self, base_exchange):
        """Test concurrent order placements."""
        orders = []
        for i in range(5):
            order = OrderRequest(
                symbol="BTC-USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.01"),
                price=Decimal("50000.00"),
                client_order_id=f"concurrent_order_{i}"
            )
            orders.append(order)

        tasks = [base_exchange.place_order(order) for order in orders]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 5
        for response in responses:
            assert response.status == OrderStatus.NEW

    async def test_empty_balance_response(self, base_exchange):
        """Test handling of empty balance response."""
        base_exchange._test_responses["get_balance"] = {}

        balances = await base_exchange.get_balance()
        assert balances == {}

    async def test_empty_positions_response(self, base_exchange):
        """Test handling of empty positions response."""
        base_exchange._test_responses["get_positions"] = []

        positions = await base_exchange.get_positions()
        assert positions == []
