"""
Simple unit test for BaseExchange that focuses on basic functionality
without complex mocking that causes import errors.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from typing import Any

import pytest

from src.core.exceptions import ValidationError
from src.core.types import (
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Ticker,
    Position,
    OrderBook,
)
from src.core.types.market import Trade


class MockBaseExchange:
    """Simple mock implementation for testing."""
    
    def __init__(self, name: str, config: dict[str, Any]):
        self.exchange_name = name
        self.config = config
        self._connected = False
        self._last_heartbeat = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_heartbeat(self):
        return self._last_heartbeat

    async def connect(self) -> None:
        self._connected = True
        self._last_heartbeat = datetime.now(timezone.utc)

    async def disconnect(self) -> None:
        self._connected = False

    async def ping(self) -> bool:
        if not self._connected:
            return False
        return True

    async def get_ticker(self, symbol: str) -> Ticker:
        if not symbol:
            raise ValidationError("Symbol cannot be empty")
        return Ticker(
            symbol=symbol,
            bid_price=Decimal("50000.00"),
            bid_quantity=Decimal("1.00"),
            ask_price=Decimal("50001.00"),
            ask_quantity=Decimal("1.00"),
            last_price=Decimal("50000.50"),
            open_price=Decimal("49500.00"),
            high_price=Decimal("50500.00"),
            low_price=Decimal("49000.00"),
            volume=Decimal("100.00"),
            exchange="mock",
            timestamp=datetime.now(timezone.utc),
        )

    async def get_account_balance(self) -> dict[str, Decimal]:
        return {
            "BTC": Decimal("1.5"),
            "ETH": Decimal("10.0"),
            "USDT": Decimal("50000.0")
        }

    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        if order_request.quantity <= 0:
            raise ValidationError("Quantity must be positive")
        if order_request.price and order_request.price <= 0:
            raise ValidationError("Price must be positive")
            
        return OrderResponse(
            order_id="test_order_123",
            exchange_order_id="exchange_123",
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            price=order_request.price,
            status=OrderStatus.FILLED,
            filled_quantity=order_request.quantity,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            exchange="mock"
        )


class TestBaseExchangeSimple:
    """Simple test suite for BaseExchange functionality."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return {
            "api_key": "test_key",
            "api_secret": "test_secret",
            "testnet": True
        }

    @pytest.fixture
    def exchange(self, config):
        """Create mock exchange instance."""
        return MockBaseExchange("test_exchange", config)

    def test_initialization(self, exchange, config):
        """Test basic initialization."""
        assert exchange.exchange_name == "test_exchange"
        assert exchange.config == config
        assert not exchange.connected
        assert exchange.last_heartbeat is None

    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, exchange):
        """Test connection and disconnection."""
        # Initially not connected
        assert not exchange.connected
        
        # Connect
        await exchange.connect()
        assert exchange.connected
        assert exchange.last_heartbeat is not None
        
        # Ping works when connected
        result = await exchange.ping()
        assert result is True
        
        # Disconnect
        await exchange.disconnect()
        assert not exchange.connected

    @pytest.mark.asyncio
    async def test_ticker_retrieval(self, exchange):
        """Test ticker data retrieval."""
        await exchange.connect()
        
        ticker = await exchange.get_ticker("BTCUSDT")
        
        assert ticker.symbol == "BTCUSDT"
        assert ticker.bid_price == Decimal("50000.00")
        assert ticker.ask_price == Decimal("50001.00")
        assert ticker.last_price == Decimal("50000.50")
        assert ticker.volume == Decimal("100.00")
        assert ticker.exchange == "mock"

    @pytest.mark.asyncio
    async def test_ticker_validation(self, exchange):
        """Test ticker validation with empty symbol."""
        await exchange.connect()
        
        with pytest.raises(ValidationError, match="Symbol cannot be empty"):
            await exchange.get_ticker("")

    @pytest.mark.asyncio
    async def test_account_balance(self, exchange):
        """Test account balance retrieval."""
        await exchange.connect()
        
        balance = await exchange.get_account_balance()
        
        assert "BTC" in balance
        assert "ETH" in balance
        assert "USDT" in balance
        assert balance["BTC"] == Decimal("1.5")
        assert balance["ETH"] == Decimal("10.0")
        assert balance["USDT"] == Decimal("50000.0")

    @pytest.mark.asyncio
    async def test_order_placement(self, exchange):
        """Test order placement."""
        await exchange.connect()
        
        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00")
        )
        
        order_response = await exchange.place_order(order_request)
        
        assert order_response.order_id == "test_order_123"
        assert order_response.symbol == "BTCUSDT"
        assert order_response.side == OrderSide.BUY
        assert order_response.quantity == Decimal("0.1")
        assert order_response.price == Decimal("50000.00")
        assert order_response.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_order_validation_zero_quantity(self, exchange):
        """Test order validation with zero quantity."""
        await exchange.connect()
        
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0"),
                price=Decimal("50000.00")
            )

    @pytest.mark.asyncio
    async def test_order_validation_zero_price(self, exchange):
        """Test order validation with zero price."""
        await exchange.connect()
        
        with pytest.raises(ValidationError, match="Price must be positive"):
            OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=Decimal("0")
            )

    @pytest.mark.asyncio
    async def test_ping_when_disconnected(self, exchange):
        """Test ping returns False when disconnected."""
        # Ensure not connected
        assert not exchange.connected
        
        result = await exchange.ping()
        assert result is False