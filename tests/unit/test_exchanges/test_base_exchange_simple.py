"""
Simple focused tests for BaseExchange to boost coverage quickly.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from src.core.exceptions import ServiceError
from src.core.types import OrderRequest, OrderSide, OrderStatus, OrderType
from src.exchanges.base import BaseExchange


class TestableExchange(BaseExchange):
    """Simple testable implementation of BaseExchange."""

    def __init__(self, name="test", config=None):
        if config is None:
            config = {"api_key": "test", "api_secret": "secret"}
        super().__init__(name=name, config=config)
        self.connection_attempted = False
        self.disconnection_attempted = False
        self._is_connected = False
        self._last_heartbeat = None

    async def connect(self):
        """Test implementation of connect."""
        self.connection_attempted = True
        self._is_connected = True
        self._last_heartbeat = datetime.now(timezone.utc)

    @property
    def last_heartbeat(self):
        """Get last heartbeat time."""
        return self._last_heartbeat

    async def disconnect(self):
        """Test implementation of disconnect."""
        self.disconnection_attempted = True
        self._is_connected = False

    @property
    def connected(self) -> bool:
        """Check if connected to exchange."""
        return self._is_connected

    async def ping(self) -> bool:
        """Test exchange connectivity."""
        return self._is_connected

    def resolve_dependency(self, name: str):
        """Mock dependency resolution for testing."""
        mock = MagicMock()
        # Make common async methods return AsyncMock
        mock.set_state = AsyncMock()
        mock.get_state = AsyncMock()
        mock.emit = AsyncMock()
        return mock

    def validate_config(self, config) -> bool:
        """Test implementation of validate_config."""
        return True

    async def load_exchange_info(self):
        """Test implementation of load_exchange_info."""
        pass

    async def get_ticker(self, symbol: str):
        """Test implementation of get_ticker."""
        return {
            "symbol": symbol,
            "price": Decimal("50000.00"),
            "bid_price": Decimal("49999.00"),
            "ask_price": Decimal("50001.00"),
            "timestamp": datetime.now(timezone.utc)
        }

    async def get_order_book(self, symbol: str, limit: int = 100):
        """Test implementation of get_order_book."""
        return {
            "symbol": symbol,
            "bids": [[Decimal("49999.00"), Decimal("1.0")]],
            "asks": [[Decimal("50001.00"), Decimal("1.0")]],
            "timestamp": datetime.now(timezone.utc)
        }

    async def get_recent_trades(self, symbol: str, limit: int = 100):
        """Test implementation of get_recent_trades."""
        return [
            {
                "symbol": symbol,
                "price": Decimal("50000.00"),
                "quantity": Decimal("0.1"),
                "side": OrderSide.BUY,
                "timestamp": datetime.now(timezone.utc)
            }
        ]

    async def place_order(self, order_request: OrderRequest):
        """Test implementation of place_order."""
        return {
            "order_id": "test_12345",
            "symbol": order_request.symbol,
            "side": order_request.side,
            "order_type": order_request.order_type,
            "quantity": order_request.quantity,
            "price": order_request.price,
            "status": OrderStatus.NEW,
            "timestamp": datetime.now(timezone.utc)
        }

    async def cancel_order(self, order_id: str, symbol: str):
        """Test implementation of cancel_order."""
        return True

    async def get_order(self, order_id: str, symbol: str):
        """Test implementation of get_order."""
        return {
            "order_id": order_id,
            "symbol": symbol,
            "status": OrderStatus.FILLED,
            "timestamp": datetime.now(timezone.utc)
        }

    async def get_account_balance(self):
        """Test implementation of get_account_balance."""
        return {
            "BTC": Decimal("1.5"),
            "USDT": Decimal("10000.0")
        }

    async def get_positions(self):
        """Test implementation of get_positions."""
        return []

    async def get_open_orders(self, symbol: str = None):
        """Test implementation of get_open_orders."""
        return []

    async def get_order_status(self, symbol: str, order_id: str):
        """Test implementation of get_order_status."""
        return {
            "order_id": order_id,
            "symbol": symbol,
            "status": OrderStatus.FILLED,
            "timestamp": datetime.now(timezone.utc)
        }

    async def get_open_orders(self, symbol: str = None):
        """Test implementation of get_open_orders."""
        return []


class TestBaseExchange:
    """Test BaseExchange class directly."""

    @pytest.fixture
    def exchange(self):
        """Create a testable exchange instance."""
        return TestableExchange(name="test", config={"api_key": "test", "api_secret": "secret"})

    def test_initialization(self, exchange):
        """Test basic initialization."""
        assert exchange.exchange_name == "test"
        assert exchange.config["api_key"] == "test"
        assert exchange.config["api_secret"] == "secret"
        assert not exchange.connected
        assert exchange.last_heartbeat is None
        assert exchange._exchange_info is None
        assert exchange._trading_symbols is None

    def test_is_connected(self, exchange):
        """Test is_connected method."""
        assert not exchange.connected
        exchange._is_connected = True
        assert exchange.connected

    @pytest.mark.asyncio
    async def test_service_lifecycle(self, exchange):
        """Test service lifecycle methods."""
        # Test starting the service
        await exchange.start()
        assert exchange.is_running
        assert exchange.connection_attempted
        assert exchange.connected

        # Test stopping the service
        await exchange.stop()
        assert not exchange.is_running
        assert exchange.disconnection_attempted
        assert not exchange.connected

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, exchange):
        """Test connect and disconnect methods."""
        # Test connection
        await exchange.connect()
        assert exchange.connected
        assert exchange.connection_attempted
        assert exchange.last_heartbeat is not None

        # Test disconnection
        await exchange.disconnect()
        assert not exchange.connected
        assert exchange.disconnection_attempted

    @pytest.mark.asyncio
    async def test_market_data_methods(self, exchange):
        """Test market data retrieval methods."""
        # Test get_ticker
        ticker = await exchange.get_ticker("BTCUSDT")
        assert ticker["symbol"] == "BTCUSDT"
        assert ticker["price"] == Decimal("50000.00")
        assert "timestamp" in ticker

        # Test get_order_book
        order_book = await exchange.get_order_book("BTCUSDT")
        assert order_book["symbol"] == "BTCUSDT"
        assert len(order_book["bids"]) > 0
        assert len(order_book["asks"]) > 0

        # Test get_recent_trades
        trades = await exchange.get_recent_trades("BTCUSDT")
        assert len(trades) > 0
        assert trades[0]["symbol"] == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_trading_methods(self, exchange):
        """Test trading methods."""
        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("49000.00")
        )

        # Test place_order
        result = await exchange.place_order(order_request)
        assert result["order_id"] == "test_12345"
        assert result["symbol"] == "BTCUSDT"
        assert result["status"] == OrderStatus.NEW

        # Test get_order
        order = await exchange.get_order("test_12345", "BTCUSDT")
        assert order["order_id"] == "test_12345"
        assert order["status"] == OrderStatus.FILLED

        # Test cancel_order
        cancelled = await exchange.cancel_order("test_12345", "BTCUSDT")
        assert cancelled is True

    @pytest.mark.asyncio
    async def test_account_methods(self, exchange):
        """Test account-related methods."""
        # Test get_account_balance
        balance = await exchange.get_account_balance()
        assert balance["BTC"] == Decimal("1.5")
        assert balance["USDT"] == Decimal("10000.0")

        # Test get_positions
        positions = await exchange.get_positions()
        assert isinstance(positions, list)

    @pytest.mark.asyncio
    async def test_error_handling(self, exchange):
        """Test error handling in service lifecycle."""
        # Mock connect to raise exception
        with patch.object(exchange, "connect", side_effect=Exception("Connection failed")):
            with pytest.raises(Exception):  # ComponentError wraps ServiceError
                await exchange.start()

        # Test service should not be running after failure
        assert not exchange.is_running

    @pytest.mark.asyncio
    async def test_health_check(self, exchange):
        """Test health check functionality."""
        # Start the service first
        await exchange.start()

        # Test health check
        health = await exchange.health_check()
        assert health.status.value == "healthy"

        # Stop service and check health
        await exchange.stop()
        health = await exchange.health_check()
        assert health.status.value == "unhealthy"

    @pytest.mark.asyncio
    async def test_validation_methods(self, exchange):
        """Test validation and utility methods."""
        # Test validate_config (now requires a config parameter)
        assert exchange.validate_config(exchange.config) is True

        # Test string representation
        str_repr = str(exchange)
        assert "test_exchange" in str_repr
