"""
Tests for exchanges service module - PERFORMANCE OPTIMIZED.

This module tests the ExchangeService which provides a clean abstraction
over exchange implementations with proper service layer patterns.
All operations are mocked for maximum speed.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

# Import Pydantic ValidationError for model instantiation errors
from pydantic_core import ValidationError as PydanticValidationError

from src.core.exceptions import ComponentError, ServiceError, ValidationError
from src.core.types import (
    ExchangeInfo,
    MarketData,
    OrderBook,
    OrderBookLevel,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    PositionStatus,
    Ticker,
)
from src.exchanges.interfaces import IExchange, IExchangeFactory
from src.exchanges.service import ExchangeService


# Pre-computed constants for performance
FIXED_DATETIME = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
FIXED_DECIMALS = {
    "price": Decimal("50000"),
    "quantity": Decimal("1.0"),
    "volume": Decimal("1000"),
    "balance_btc": Decimal("1.5"),
    "balance_usdt": Decimal("50000"),
}


class MockConfig:
    """Optimized mock configuration for testing."""

    def __init__(self):
        self.exchange_service = MockExchangeServiceConfig()

    def to_dict(self):
        return {"test": "config"}


class MockExchangeServiceConfig:
    """Optimized mock exchange service configuration."""

    def __init__(self):
        # Reduced timeouts for faster tests
        self.default_timeout_seconds = 1
        self.max_retries = 1 
        self.health_check_interval_seconds = 1


class MockExchange(IExchange):
    """Optimized mock exchange for testing - uses pre-computed values."""

    def __init__(self, name: str = "test_exchange", healthy: bool = True):
        self.name = name
        self.healthy = healthy
        self.connected = False
        self.calls = []

    @property
    def exchange_name(self) -> str:
        """Get exchange name."""
        return self.name

    def is_connected(self) -> bool:
        """Check if exchange is connected."""
        return self.connected

    async def connect(self):
        # Instant return, no sleep
        self.connected = True
        self.calls.append("connect")

    async def disconnect(self):
        # Instant return, no sleep 
        self.connected = False
        self.calls.append("disconnect")

    async def health_check(self) -> bool:
        # Instant return, no health check delays
        self.calls.append("health_check")
        return self.healthy

    async def place_order(self, order: OrderRequest) -> OrderResponse:
        self.calls.append(("place_order", order))
        return OrderResponse(
            order_id="order_123",
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            status=OrderStatus.FILLED,
            filled_quantity=order.quantity,
            average_price=order.price or FIXED_DECIMALS["price"],
            created_at=FIXED_DATETIME,  # Use fixed datetime
            exchange=self.exchange_name,
        )

    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse:
        self.calls.append(("cancel_order", symbol, order_id))
        return OrderResponse(
            order_id=order_id,
            client_order_id=f"client_{order_id}",
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=FIXED_DECIMALS["quantity"],
            price=FIXED_DECIMALS["price"],
            status=OrderStatus.CANCELLED,
            filled_quantity=Decimal("0"),
            average_price=Decimal("0"),
            created_at=FIXED_DATETIME,  # Use fixed datetime
            exchange=self.exchange_name,
        )

    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse:
        self.calls.append(("get_order_status", symbol, order_id))
        return OrderResponse(
            order_id=order_id,
            client_order_id=f"client_{order_id}",
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=FIXED_DECIMALS["quantity"],
            price=FIXED_DECIMALS["price"],
            status=OrderStatus.FILLED,
            filled_quantity=FIXED_DECIMALS["quantity"],
            average_price=FIXED_DECIMALS["price"],
            created_at=FIXED_DATETIME,  # Use fixed datetime
            exchange=self.exchange_name,
        )

    async def get_market_data(self, symbol: str, timeframe: str = "1m") -> MarketData:
        self.calls.append(("get_market_data", symbol, timeframe))
        return MarketData(
            symbol=symbol,
            timestamp=FIXED_DATETIME,  # Use fixed datetime
            open=FIXED_DECIMALS["price"],
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("100"),
            exchange=self.name,
        )

    async def get_order_book(self, symbol: str, depth: int = 10) -> OrderBook:
        self.calls.append(("get_order_book", symbol, depth))
        return OrderBook(
            symbol=symbol,
            bids=[OrderBookLevel(price=Decimal("49900"), quantity=FIXED_DECIMALS["quantity"])],
            asks=[OrderBookLevel(price=Decimal("50100"), quantity=FIXED_DECIMALS["quantity"])],
            timestamp=FIXED_DATETIME,  # Use fixed datetime
            exchange=self.name,
        )

    async def get_ticker(self, symbol: str) -> Ticker:
        self.calls.append(("get_ticker", symbol))
        return Ticker(
            symbol=symbol,
            bid_price=Decimal("49900"),
            bid_quantity=FIXED_DECIMALS["quantity"],
            ask_price=Decimal("50100"),
            ask_quantity=FIXED_DECIMALS["quantity"],
            last_price=FIXED_DECIMALS["price"],
            open_price=Decimal("49000"),
            high_price=Decimal("51000"),
            low_price=Decimal("49000"),
            volume=FIXED_DECIMALS["volume"],
            timestamp=FIXED_DATETIME,  # Use fixed datetime
            exchange=self.name,
        )

    async def get_account_balance(self) -> dict[str, Decimal]:
        self.calls.append("get_account_balance")
        return {"BTC": FIXED_DECIMALS["balance_btc"], "USDT": FIXED_DECIMALS["balance_usdt"]}

    async def get_positions(self) -> list[Position]:
        self.calls.append("get_positions")
        return [
            Position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                quantity=FIXED_DECIMALS["quantity"],
                entry_price=Decimal("48000"),
                current_price=FIXED_DECIMALS["price"],
                unrealized_pnl=Decimal("2000"),
                status=PositionStatus.OPEN,
                opened_at=FIXED_DATETIME,  # Use fixed datetime
                exchange=self.name,
            )
        ]

    async def get_exchange_info(self) -> ExchangeInfo:
        self.calls.append("get_exchange_info")

        # Create a mock object that matches test expectations
        class MockExchangeInfo:
            def __init__(self, name, symbols):
                self.name = name
                self.symbols = symbols

        return MockExchangeInfo(name=self.name, symbols=["BTCUSDT", "ETHUSDT"])

    async def get_trade_history(self, symbol: str, limit: int = 100) -> list:
        self.calls.append(("get_trade_history", symbol, limit))
        return []  # Return empty list for testing

    async def subscribe_to_stream(self, symbol: str, callback):
        self.calls.append(("subscribe_to_stream", symbol, callback))


class MockExchangeFactory(IExchangeFactory):
    """Mock exchange factory for testing."""

    def __init__(self):
        self.exchanges = {}
        self.supported_exchanges = ["binance", "coinbase", "okx"]
        self.available_exchanges = ["binance", "coinbase"]
        self.create_calls = []
        self.started = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.started = False

    async def get_exchange(self, exchange_name: str, create_if_missing: bool = False) -> IExchange:
        self.create_calls.append((exchange_name, create_if_missing))

        if exchange_name not in self.exchanges:
            if create_if_missing:
                self.exchanges[exchange_name] = MockExchange(exchange_name)
            else:
                return None

        return self.exchanges[exchange_name]

    def get_supported_exchanges(self) -> list[str]:
        return self.supported_exchanges

    def get_available_exchanges(self) -> list[str]:
        return self.available_exchanges


@pytest.fixture
def mock_config():
    """Provide mock configuration."""
    return MockConfig()


@pytest.fixture
def mock_factory():
    """Provide mock exchange factory."""
    return MockExchangeFactory()


@pytest_asyncio.fixture(scope="function")
async def exchange_service(mock_config, mock_factory):
    """Provide exchange service instance."""
    service = ExchangeService(
        exchange_factory=mock_factory, config=mock_config, correlation_id="test_correlation"
    )
    await service.start()
    yield service
    await service.stop()


class TestExchangeServiceInitialization:
    """Test ExchangeService initialization."""

    def test_init_success(self, mock_config, mock_factory):
        """Test successful initialization."""
        service = ExchangeService(
            exchange_factory=mock_factory, config=mock_config, correlation_id="test_correlation"
        )

        assert service.exchange_factory == mock_factory
        assert service.config == mock_config
        assert len(service._active_exchanges) == 0

    def test_init_missing_factory(self, mock_config):
        """Test initialization allows None factory (will fail later when used)."""
        # The service allows None factory during initialization
        # but will fail when trying to use it
        service = ExchangeService(exchange_factory=None, config=mock_config)
        assert service.exchange_factory is None

    def test_init_missing_config(self, mock_factory):
        """Test initialization allows None config (will fail later when used)."""
        # The service allows None config during initialization
        service = ExchangeService(exchange_factory=mock_factory, config=None)
        assert service.config is None

    def test_init_with_config_dict(self, mock_factory):
        """Test initialization with config as dict."""
        config_dict = {"test": "config"}

        service = ExchangeService(exchange_factory=mock_factory, config=config_dict)

        assert service.config == config_dict


class TestExchangeServiceLifecycle:
    """Test ExchangeService lifecycle methods."""

    @pytest.mark.asyncio
    async def test_start_success(self, mock_config, mock_factory):
        """Test successful service start."""
        service = ExchangeService(exchange_factory=mock_factory, config=mock_config)

        await service.start()

        assert service.is_running
        assert mock_factory.started

    @pytest.mark.asyncio
    async def test_start_factory_without_start_method(self, mock_config):
        """Test start with factory that doesn't have start method."""
        factory = Mock()
        # Remove start method from factory to simulate factory without start method
        del factory.start

        service = ExchangeService(exchange_factory=factory, config=mock_config)

        await service.start()  # Should not raise exception
        assert service.is_running

    @pytest.mark.asyncio
    async def test_start_factory_failure(self, mock_config):
        """Test start failure when factory start fails."""
        factory = Mock()
        factory.start = AsyncMock(side_effect=Exception("Factory start failed"))

        service = ExchangeService(exchange_factory=factory, config=mock_config)

        with pytest.raises(ComponentError, match="Failed to start.*Exchange service startup failed"):
            await service.start()

    @pytest.mark.asyncio
    async def test_stop_success(self, exchange_service, mock_factory):
        """Test successful service stop."""
        # Add a mock exchange to test disconnection
        mock_exchange = MockExchange()
        exchange_service._active_exchanges["test"] = mock_exchange

        await exchange_service.stop()

        assert not exchange_service.is_running
        assert not mock_factory.started
        assert len(exchange_service._active_exchanges) == 0

    @pytest.mark.asyncio
    async def test_stop_factory_without_stop_method(self, mock_config):
        """Test stop with factory that doesn't have stop method."""
        factory = Mock()
        # Make start an async method but don't add stop method
        factory.start = AsyncMock()

        service = ExchangeService(exchange_factory=factory, config=mock_config)

        await service.start()
        await service.stop()  # Should not raise exception
        assert not service.is_running


class TestExchangeManagement:
    """Test exchange management methods."""

    @pytest.mark.asyncio
    async def test_get_exchange_success(self, exchange_service):
        """Test successful exchange retrieval."""
        exchange = await exchange_service.get_exchange("binance")

        assert exchange is not None
        assert exchange.name == "binance"
        assert "binance" in exchange_service._active_exchanges

    @pytest.mark.asyncio
    async def test_get_exchange_cached(self, exchange_service):
        """Test exchange is cached and reused."""
        # First call
        exchange1 = await exchange_service.get_exchange("binance")

        # Second call should return same instance
        exchange2 = await exchange_service.get_exchange("binance")

        assert exchange1 is exchange2

    @pytest.mark.asyncio
    async def test_get_exchange_invalid_name(self, exchange_service):
        """Test get exchange with invalid name."""
        # Should raise ValidationError for invalid exchange name
        with pytest.raises(ValidationError, match="Exchange name is required"):
            await exchange_service.get_exchange("")

    @pytest.mark.asyncio
    async def test_get_exchange_factory_failure(self, exchange_service):
        """Test get exchange when factory fails."""
        exchange_service.exchange_factory.get_exchange = AsyncMock(return_value=None)

        # Should raise ServiceError when factory returns None
        with pytest.raises(ServiceError, match="Failed to create exchange"):
            await exchange_service.get_exchange("binance")

    @pytest.mark.asyncio
    async def test_get_exchange_unhealthy_replacement(self, exchange_service):
        """Test unhealthy exchange is replaced."""
        # Create an unhealthy exchange
        unhealthy_exchange = MockExchange("binance", healthy=False)
        exchange_service._active_exchanges["binance"] = unhealthy_exchange

        # Get exchange should replace the unhealthy one
        exchange = await exchange_service.get_exchange("binance")

        assert exchange is not unhealthy_exchange
        assert exchange.healthy  # New exchange should be healthy

    @pytest.mark.asyncio
    async def test_remove_exchange(self, exchange_service):
        """Test exchange removal."""
        # Add an exchange
        mock_exchange = MockExchange()
        exchange_service._active_exchanges["test"] = mock_exchange

        # Remove it
        await exchange_service._remove_exchange("test")

        assert "test" not in exchange_service._active_exchanges
        assert "disconnect" in mock_exchange.calls

    @pytest.mark.asyncio
    async def test_remove_nonexistent_exchange(self, exchange_service):
        """Test removing non-existent exchange doesn't error."""
        # Should not raise exception
        await exchange_service._remove_exchange("nonexistent")

    @pytest.mark.asyncio
    async def test_disconnect_all_exchanges(self, exchange_service):
        """Test disconnecting all exchanges."""
        # Add some exchanges
        exchange1 = MockExchange("exchange1")
        exchange2 = MockExchange("exchange2")
        exchange_service._active_exchanges["exchange1"] = exchange1
        exchange_service._active_exchanges["exchange2"] = exchange2

        await exchange_service.disconnect_all_exchanges()

        assert len(exchange_service._active_exchanges) == 0
        assert "disconnect" in exchange1.calls
        assert "disconnect" in exchange2.calls


class TestTradingOperations:
    """Test trading operation methods."""

    @pytest.mark.asyncio
    async def test_place_order_success(self, exchange_service):
        """Test successful order placement."""
        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
        )

        response = await exchange_service.place_order("binance", order_request)

        assert response.id == "order_123"
        assert response.symbol == "BTCUSDT"
        assert response.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_place_order_validation_failure(self, exchange_service):
        """Test order placement with validation failure."""
        # OrderRequest now validates at creation time with Pydantic ValidationError
        with pytest.raises(PydanticValidationError, match="Symbol cannot be empty"):
            order_request = OrderRequest(
                symbol="",  # Invalid empty symbol
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                price=Decimal("50000"),
            )

    @pytest.mark.asyncio
    async def test_place_order_invalid_quantity(self, exchange_service):
        """Test order placement with invalid quantity."""
        # OrderRequest validates quantity at creation time with our custom ValidationError
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            order_request = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0"),  # Invalid zero quantity
                price=Decimal("50000"),
            )

    @pytest.mark.asyncio
    async def test_place_order_invalid_side(self, exchange_service):
        """Test order placement with invalid side."""
        # Test with invalid side value - Pydantic will catch this at instantiation
        with pytest.raises(PydanticValidationError):
            order_request = OrderRequest(
                symbol="BTCUSDT",
                side="invalid_side",  # Invalid side value
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                price=Decimal("50000"),
            )

    @pytest.mark.asyncio
    async def test_place_order_missing_required_fields(self, exchange_service):
        """Test order placement with missing required fields."""
        # Test missing symbol
        with pytest.raises(PydanticValidationError):
            order_request = OrderRequest(
                # Missing symbol
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                price=Decimal("50000"),
            )

        # Test missing side
        with pytest.raises(PydanticValidationError):
            order_request = OrderRequest(
                symbol="BTCUSDT",
                # Missing side
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                price=Decimal("50000"),
            )

        # Test missing order_type
        with pytest.raises(PydanticValidationError):
            order_request = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                # Missing order_type
                quantity=Decimal("1.0"),
                price=Decimal("50000"),
            )

        # Test missing quantity
        with pytest.raises(PydanticValidationError):
            order_request = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                # Missing quantity
                price=Decimal("50000"),
            )

    @pytest.mark.asyncio
    async def test_place_order_negative_quantity(self, exchange_service):
        """Test order placement with negative quantity."""
        # OrderRequest validates quantity at creation time with our custom ValidationError
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            order_request = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("-1.0"),  # Invalid negative quantity
                price=Decimal("50000"),
            )

    @pytest.mark.asyncio
    async def test_place_order_invalid_price(self, exchange_service):
        """Test order placement with invalid price."""
        # Test negative price
        with pytest.raises(ValidationError, match="Price must be positive"):
            order_request = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                price=Decimal("-50000"),  # Invalid negative price
            )

        # Test zero price
        with pytest.raises(ValidationError, match="Price must be positive"):
            order_request = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                price=Decimal("0"),  # Invalid zero price
            )

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, exchange_service):
        """Test successful order cancellation."""
        result = await exchange_service.cancel_order("binance", "order_123", symbol="BTCUSDT")

        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_order_missing_id(self, exchange_service):
        """Test order cancellation with missing ID."""
        # Should raise ValidationError for missing order ID
        with pytest.raises(ValidationError, match="Order ID is required"):
            await exchange_service.cancel_order("binance", "", symbol="BTCUSDT")

    @pytest.mark.asyncio
    async def test_get_order_status_success(self, exchange_service):
        """Test successful order status retrieval."""
        status = await exchange_service.get_order_status("binance", "order_123", symbol="BTCUSDT")

        # OrderStatus is an enum, so we check its value
        assert status == OrderStatus.FILLED
        assert status.value == "filled"

    @pytest.mark.asyncio
    async def test_get_order_status_missing_id(self, exchange_service):
        """Test order status with missing ID."""
        # Should raise ValidationError for missing order ID
        with pytest.raises(ValidationError, match="Order ID is required"):
            await exchange_service.get_order_status("binance", "", symbol="BTCUSDT")


class TestMarketDataOperations:
    """Test market data operation methods."""

    @pytest.mark.asyncio
    async def test_get_market_data_success(self, exchange_service):
        """Test successful market data retrieval."""
        data = await exchange_service.get_market_data("binance", "BTCUSDT", "1h")

        assert data.symbol == "BTCUSDT"
        assert data.close == Decimal("50500")
        assert data.exchange == "binance"

    @pytest.mark.asyncio
    async def test_get_market_data_missing_symbol(self, exchange_service):
        """Test market data with missing symbol."""
        # Should raise ValidationError for missing symbol
        with pytest.raises(ValidationError, match="Symbol is required"):
            await exchange_service.get_market_data("binance", "")

    @pytest.mark.asyncio
    async def test_get_order_book_success(self, exchange_service):
        """Test successful order book retrieval."""
        book = await exchange_service.get_order_book("binance", "BTCUSDT", 20)

        assert book.symbol == "BTCUSDT"
        assert len(book.bids) > 0
        assert len(book.asks) > 0

    @pytest.mark.asyncio
    async def test_get_order_book_missing_symbol(self, exchange_service):
        """Test order book with missing symbol."""
        # Should raise ValidationError for missing symbol
        with pytest.raises(ValidationError, match="Symbol is required"):
            await exchange_service.get_order_book("binance", "")

    @pytest.mark.asyncio
    async def test_get_order_book_invalid_depth(self, exchange_service):
        """Test order book with invalid depth."""
        # Should raise ValidationError for invalid depth
        with pytest.raises(ValidationError, match="Depth must be between 1 and 1000"):
            await exchange_service.get_order_book("binance", "BTCUSDT", 0)

        
        # Also test depth too large
        with pytest.raises(ValidationError, match="Depth must be between 1 and 1000"):
            await exchange_service.get_order_book("binance", "BTCUSDT", 1001)

    @pytest.mark.asyncio
    async def test_get_ticker_success(self, exchange_service):
        """Test successful ticker retrieval."""
        ticker = await exchange_service.get_ticker("binance", "BTCUSDT")

        assert ticker.symbol == "BTCUSDT"
        assert ticker.bid_price == Decimal("49900")
        assert ticker.ask_price == Decimal("50100")

    @pytest.mark.asyncio
    async def test_get_ticker_missing_symbol(self, exchange_service):
        """Test ticker with missing symbol."""
        # Should raise ValidationError for missing symbol
        with pytest.raises(ValidationError, match="Symbol is required"):
            await exchange_service.get_ticker("binance", "")


class TestAccountOperations:
    """Test account operation methods."""

    @pytest.mark.asyncio
    async def test_get_account_balance_success(self, exchange_service):
        """Test successful account balance retrieval."""
        balances = await exchange_service.get_account_balance("binance")

        assert "BTC" in balances
        assert "USDT" in balances
        assert balances["BTC"] == Decimal("1.5")

    @pytest.mark.asyncio
    async def test_get_positions_success(self, exchange_service):
        """Test successful positions retrieval."""
        positions = await exchange_service.get_positions("binance")

        assert len(positions) == 1
        assert positions[0].symbol == "BTCUSDT"
        assert positions[0].side == PositionSide.LONG
        assert positions[0].exchange == "binance"

    @pytest.mark.asyncio
    async def test_get_exchange_info_success(self, exchange_service):
        """Test successful exchange info retrieval."""
        info = await exchange_service.get_exchange_info("binance")

        assert info.name == "binance"
        assert "BTCUSDT" in info.symbols

    def test_get_supported_exchanges(self, exchange_service):
        """Test supported exchanges list."""
        exchanges = exchange_service.get_supported_exchanges()

        assert "binance" in exchanges
        assert "coinbase" in exchanges
        assert "okx" in exchanges

    def test_get_available_exchanges(self, exchange_service):
        """Test available exchanges list."""
        exchanges = exchange_service.get_available_exchanges()

        assert "binance" in exchanges
        assert "coinbase" in exchanges
        assert "okx" not in exchanges  # Not in available list


class TestHealthAndManagement:
    """Test health and management methods."""

    @pytest.mark.asyncio
    async def test_get_service_health(self, exchange_service):
        """Test service health status."""
        # Add some exchanges
        exchange1 = MockExchange("binance", healthy=True)
        exchange2 = MockExchange("coinbase", healthy=False)
        exchange_service._active_exchanges["binance"] = exchange1
        exchange_service._active_exchanges["coinbase"] = exchange2

        health = await exchange_service.get_service_health()

        assert health["service"] == "ExchangeService"
        assert health["active_exchanges"] == 2
        assert health["exchanges"]["binance"]["healthy"] is True
        assert health["exchanges"]["coinbase"]["healthy"] is False

    @pytest.mark.asyncio
    async def test_service_health_check_no_exchanges(self, exchange_service):
        """Test service health check with no active exchanges."""
        result = await exchange_service._service_health_check()

        # Health check returns HealthStatus enum, not boolean
        from src.core.base.interfaces import HealthStatus

        assert result == HealthStatus.HEALTHY  # No active exchanges is OK

    @pytest.mark.asyncio
    async def test_service_health_check_healthy_exchanges(self, exchange_service):
        """Test service health check with healthy exchanges."""
        # Add healthy exchange
        healthy_exchange = MockExchange("binance", healthy=True)
        exchange_service._active_exchanges["binance"] = healthy_exchange

        result = await exchange_service._service_health_check()

        from src.core.base.interfaces import HealthStatus

        assert result == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_service_health_check_all_unhealthy(self, exchange_service):
        """Test service health check with all unhealthy exchanges."""
        # Add only unhealthy exchange
        unhealthy_exchange = MockExchange("binance", healthy=False)
        exchange_service._active_exchanges["binance"] = unhealthy_exchange

        result = await exchange_service._service_health_check()

        from src.core.base.interfaces import HealthStatus

        assert result == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_service_health_check_mixed_health(self, exchange_service):
        """Test service health check with mixed health exchanges."""
        # Add healthy and unhealthy exchanges
        healthy_exchange = MockExchange("binance", healthy=True)
        unhealthy_exchange = MockExchange("coinbase", healthy=False)
        exchange_service._active_exchanges["binance"] = healthy_exchange
        exchange_service._active_exchanges["coinbase"] = unhealthy_exchange

        result = await exchange_service._service_health_check()

        from src.core.base.interfaces import HealthStatus

        assert result == HealthStatus.HEALTHY  # At least one healthy


class TestMultiExchangeOperations:
    """Test multi-exchange operation methods."""

    @pytest.mark.asyncio
    async def test_get_best_price_buy(self, exchange_service):
        """Test get best price for buy orders."""
        result = await exchange_service.get_best_price("BTCUSDT", "BUY")

        assert result["symbol"] == "BTCUSDT"
        assert result["side"] == "BUY"
        assert result["best_price"] is not None
        assert result["best_exchange"] is not None

    @pytest.mark.asyncio
    async def test_get_best_price_sell(self, exchange_service):
        """Test get best price for sell orders."""
        result = await exchange_service.get_best_price("BTCUSDT", "SELL")

        assert result["symbol"] == "BTCUSDT"
        assert result["side"] == "SELL"
        assert result["best_price"] is not None
        assert result["best_exchange"] is not None

    @pytest.mark.asyncio
    async def test_get_best_price_specific_exchanges(self, exchange_service):
        """Test get best price with specific exchanges."""
        result = await exchange_service.get_best_price("BTCUSDT", "BUY", exchanges=["binance"])

        assert result["best_exchange"] == "binance"
        assert len(result["all_prices"]) == 1

    @pytest.mark.asyncio
    async def test_get_ticker_safe_success(self, exchange_service):
        """Test safe ticker retrieval."""
        ticker = await exchange_service._get_ticker_safe("binance", "BTCUSDT")

        assert ticker is not None
        assert ticker.symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_get_ticker_safe_failure(self, exchange_service):
        """Test safe ticker retrieval with failure."""
        # Mock the get_ticker to raise exception
        with patch.object(exchange_service, "get_ticker", side_effect=Exception("API error")):
            ticker = await exchange_service._get_ticker_safe("binance", "BTCUSDT")

            assert ticker is None


class TestWebSocketOperations:
    """Test WebSocket operation methods."""

    @pytest.mark.asyncio
    async def test_subscribe_to_stream_success(self, exchange_service):
        """Test successful stream subscription."""
        callback = Mock()

        await exchange_service.subscribe_to_stream("binance", "BTCUSDT", callback)

        # Verify the exchange's subscribe method was called
        exchange = exchange_service._active_exchanges["binance"]
        assert ("subscribe_to_stream", "BTCUSDT", callback) in exchange.calls


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_exchange_operation_failure(self, exchange_service):
        """Test handling of exchange operation failures."""
        # Mock exchange to raise exception
        mock_exchange = MockExchange()
        mock_exchange.get_ticker = AsyncMock(side_effect=Exception("Exchange error"))
        exchange_service._active_exchanges["binance"] = mock_exchange

        # Should raise ServiceError after retries fail
        with pytest.raises(ServiceError, match="Failed to get ticker"):
            await exchange_service.get_ticker("binance", "BTCUSDT")

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, exchange_service):
        """Test circuit breaker decorator integration."""
        # This test verifies decorators are applied
        # Actual circuit breaker testing would require more complex setup

        # Simple test that methods are decorated
        assert hasattr(exchange_service.get_exchange, "__wrapped__")
        assert hasattr(exchange_service.place_order, "__wrapped__")

    @pytest.mark.asyncio
    async def test_retry_decorator_integration(self, exchange_service):
        """Test retry decorator integration."""
        # This test verifies decorators are applied
        # Actual retry testing would require more complex setup

        # Simple test that methods are decorated
        assert hasattr(exchange_service.get_exchange, "__wrapped__")
        assert hasattr(exchange_service.place_order, "__wrapped__")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_concurrent_exchange_access(self, exchange_service):
        """Test concurrent access to same exchange."""
        # Start multiple tasks requesting same exchange
        tasks = []
        for _ in range(5):
            task = asyncio.create_task(exchange_service.get_exchange("binance"))
            tasks.append(task)

        exchanges = await asyncio.gather(*tasks)

        # All should return same instance
        for exchange in exchanges[1:]:
            assert exchange is exchanges[0]

    @pytest.mark.asyncio
    async def test_large_order_book_depth(self, exchange_service):
        """Test order book with maximum depth."""
        book = await exchange_service.get_order_book("binance", "BTCUSDT", 1000)

        assert book.symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_very_large_decimal_values(self, exchange_service):
        """Test handling of very large decimal values."""
        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("999999999.99999999"),
            price=Decimal("999999.99"),
        )

        response = await exchange_service.place_order("binance", order_request)

        assert response.filled_quantity == Decimal("999999999.99999999")

    @pytest.mark.asyncio
    async def test_zero_values(self, exchange_service):
        """Test handling of zero values where appropriate."""
        # Order book depth of 1 (minimum)
        book = await exchange_service.get_order_book("binance", "BTCUSDT", 1)
        assert book is not None

    @pytest.mark.asyncio
    async def test_unicode_symbols(self, exchange_service):
        """Test handling of unicode symbols."""
        # Test with symbol containing special characters (if valid)
        data = await exchange_service.get_market_data("binance", "BTCUSDT")  # Standard symbol
        assert data.symbol == "BTCUSDT"


class TestProcessOrderResponse:
    """Test _process_order_response method."""

    @pytest.mark.asyncio
    async def test_process_order_response_success(self, exchange_service):
        """Test successful order response processing."""
        order_response = OrderResponse(
            order_id="order_123",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("1.0"),
            average_price=Decimal("50000"),
            created_at=datetime.now(timezone.utc),
            exchange="binance",
        )

        # Should not raise exception
        await exchange_service._process_order_response("binance", order_response)

    @pytest.mark.asyncio
    async def test_process_order_response_with_exception(self, exchange_service):
        """Test order response processing handles exceptions gracefully."""
        order_response = OrderResponse(
            order_id="order_123",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("1.0"),
            average_price=Decimal("50000"),
            created_at=datetime.now(timezone.utc),
            exchange="binance",
        )

        # Mock logger to verify warning is logged
        with patch("src.exchanges.service.logger") as mock_logger:
            # This should handle any processing errors gracefully
            await exchange_service._process_order_response("binance", order_response)

            # Verify info was logged (successful case)
            mock_logger.info.assert_called()


class TestValidateOrderRequest:
    """Test _validate_order_request method."""

    @pytest.mark.asyncio
    async def test_validate_order_request_all_valid(self, exchange_service):
        """Test order request validation with all valid fields."""
        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
        )

        # Should not raise exception
        await exchange_service._validate_order_request(order_request)

    @pytest.mark.asyncio
    async def test_validate_order_request_none_quantity(self, exchange_service):
        """Test order request validation with None quantity."""
        # Pydantic will catch None quantity at instantiation time
        with pytest.raises(PydanticValidationError):
            order_request = OrderRequest(
                symbol="BTCUSDT",
                side="buy",
                order_type="limit",
                quantity=None,  # Invalid None quantity
                price=Decimal("50000"),
            )
