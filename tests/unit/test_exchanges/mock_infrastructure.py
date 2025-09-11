"""
Complete Mock Infrastructure for Exchange Testing

This module provides comprehensive mock implementations of all protocols,
adapters, and base classes needed to properly test the exchanges module.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

from src.core.logging import get_logger

logger = get_logger(__name__)


class MockConfig:
    """Complete mock configuration that passes all validations."""

    def __init__(self):
        self._data = {
            "exchange.rate_limit": 10,
            "exchange.timeout": 30,
            "exchange.max_retries": 3,
            "exchange.api_key": "mock_api_key",
            "exchange.api_secret": "mock_api_secret",
            "exchange.passphrase": "mock_passphrase",
            "exchange.testnet": True,
            "exchange.base_url": "https://api.mock.exchange",
            "exchange.websocket_url": "wss://stream.mock.exchange",
            "exchange.order_timeout": 60,
            "exchange.heartbeat_interval": 30,
            "exchange.reconnect_delay": 5,
            "exchange.max_reconnect_attempts": 5,
            "risk.max_position_size": 0.02,
            "risk.max_total_exposure": 1.0,
            "risk.stop_loss_percentage": 0.05,
            "monitoring.enabled": True,
            "monitoring.metrics_interval": 60,
            "database.url": "postgresql://test@localhost/test",
            "redis.url": "redis://localhost:6379",
        }
        self.exchange = MagicMock()
        self.exchange.rate_limit = 10
        self.exchange.timeout = 30
        self.exchange.max_retries = 3
        self.exchange.api_key = "mock_api_key"
        self.exchange.api_secret = "mock_api_secret"
        self.exchange.passphrase = "mock_passphrase"
        self.exchange.testnet = True
        self.exchange.base_url = "https://api.mock.exchange"
        self.exchange.websocket_url = "wss://stream.mock.exchange"

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._data.get(key, default)

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Get nested configuration value."""
        key = ".".join(keys)
        return self._data.get(key, default)

    def validate(self) -> bool:
        """Validate configuration."""
        return True


class MockAdapter:
    """Complete mock adapter implementing all required methods."""

    def __init__(self):
        self.connected = False
        self.orders = {}
        self.balances = {
            "BTC": Decimal("1.5"),
            "ETH": Decimal("10.0"),
            "USDT": Decimal("10000.0")
        }
        self.positions = []
        self.market_data = {}
        self.websocket_connected = False
        self.subscriptions = set()

    async def connect(self) -> bool:
        """Mock connection."""
        self.connected = True
        return True

    async def disconnect(self) -> None:
        """Mock disconnection."""
        self.connected = False

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """Mock order placement."""
        order_id = f"mock_{len(self.orders) + 1}"
        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "price": price,
            "status": "NEW",
            "filled": Decimal("0"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        self.orders[order_id] = order
        return order

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]:
        """Mock order cancellation."""
        if order_id in self.orders:
            self.orders[order_id]["status"] = "CANCELED"
            return self.orders[order_id]
        return {"error": "Order not found"}

    async def get_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]:
        """Mock get order."""
        return self.orders.get(order_id, {"error": "Order not found"})

    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Mock get open orders."""
        open_orders = [
            order for order in self.orders.values()
            if order["status"] in ["NEW", "PARTIALLY_FILLED"]
            and (symbol is None or order["symbol"] == symbol)
        ]
        return open_orders

    async def get_balance(self, asset: str | None = None) -> dict[str, Any]:
        """Mock get balance."""
        if asset:
            return {
                "asset": asset,
                "free": str(self.balances.get(asset, Decimal("0"))),
                "locked": "0"
            }
        return {
            asset: {"free": str(balance), "locked": "0"}
            for asset, balance in self.balances.items()
        }

    async def get_positions(self) -> list[dict[str, Any]]:
        """Mock get positions."""
        return self.positions

    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        """Mock get ticker."""
        return {
            "symbol": symbol,
            "bid": "50000.00",
            "ask": "50001.00",
            "last": "50000.50",
            "volume": "1000.0",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def get_order_book(self, symbol: str, depth: int = 10) -> dict[str, Any]:
        """Mock get order book."""
        return {
            "symbol": symbol,
            "bids": [[f"{50000 - i * 10}", f"{1.0 + i * 0.1}"] for i in range(depth)],
            "asks": [[f"{50001 + i * 10}", f"{1.0 + i * 0.1}"] for i in range(depth)],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        """Mock get recent trades."""
        return [
            {
                "id": str(i),
                "price": f"{50000 + i}",
                "quantity": f"{0.1 * i}",
                "side": "BUY" if i % 2 == 0 else "SELL",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            for i in range(min(limit, 10))
        ]

    async def subscribe_to_channel(self, channel: str, symbol: str | None = None) -> bool:
        """Mock channel subscription."""
        subscription = f"{channel}:{symbol}" if symbol else channel
        self.subscriptions.add(subscription)
        return True

    async def unsubscribe_from_channel(self, channel: str, symbol: str | None = None) -> bool:
        """Mock channel unsubscription."""
        subscription = f"{channel}:{symbol}" if symbol else channel
        self.subscriptions.discard(subscription)
        return True

    async def start_websocket(self) -> bool:
        """Mock WebSocket start."""
        self.websocket_connected = True
        return True

    async def stop_websocket(self) -> None:
        """Mock WebSocket stop."""
        self.websocket_connected = False

    def is_connected(self) -> bool:
        """Check if connected."""
        return self.connected

    def is_websocket_connected(self) -> bool:
        """Check if WebSocket connected."""
        return self.websocket_connected


class MockOrderManager:
    """Mock order manager with complete functionality."""

    def __init__(self):
        self.orders = {}
        self.order_history = []

    async def create_order(self, order_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new order."""
        order_id = f"order_{len(self.orders) + 1}"
        order = {
            "id": order_id,
            "status": "NEW",
            "created_at": datetime.now(timezone.utc),
            **order_data
        }
        self.orders[order_id] = order
        self.order_history.append(order)
        return order

    async def update_order(self, order_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        """Update an existing order."""
        if order_id in self.orders:
            self.orders[order_id].update(updates)
            return self.orders[order_id]
        raise ValueError(f"Order {order_id} not found")

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Get order by ID."""
        if order_id in self.orders:
            return self.orders[order_id]
        raise ValueError(f"Order {order_id} not found")

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an order."""
        if order_id in self.orders:
            self.orders[order_id]["status"] = "CANCELED"
            return self.orders[order_id]
        raise ValueError(f"Order {order_id} not found")

    def get_open_orders(self) -> list[dict[str, Any]]:
        """Get all open orders."""
        return [
            order for order in self.orders.values()
            if order["status"] in ["NEW", "PARTIALLY_FILLED"]
        ]


class MockMarketDataManager:
    """Mock market data manager."""

    def __init__(self):
        self.tickers = {}
        self.order_books = {}
        self.trades = {}

    async def update_ticker(self, symbol: str, ticker: dict[str, Any]) -> None:
        """Update ticker data."""
        self.tickers[symbol] = ticker

    async def update_order_book(self, symbol: str, order_book: dict[str, Any]) -> None:
        """Update order book."""
        self.order_books[symbol] = order_book

    async def add_trade(self, symbol: str, trade: dict[str, Any]) -> None:
        """Add a new trade."""
        if symbol not in self.trades:
            self.trades[symbol] = []
        self.trades[symbol].append(trade)

    def get_ticker(self, symbol: str) -> dict[str, Any] | None:
        """Get ticker for symbol."""
        return self.tickers.get(symbol)

    def get_order_book(self, symbol: str) -> dict[str, Any] | None:
        """Get order book for symbol."""
        return self.order_books.get(symbol)

    def get_recent_trades(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent trades for symbol."""
        trades = self.trades.get(symbol, [])
        return trades[-limit:] if trades else []


class MockWebSocketManager:
    """Mock WebSocket manager."""

    def __init__(self):
        self.connected = False
        self.subscriptions = set()
        self.message_queue = asyncio.Queue()
        self.callbacks = {}

    async def connect(self) -> bool:
        """Connect WebSocket."""
        self.connected = True
        return True

    async def disconnect(self) -> None:
        """Disconnect WebSocket."""
        self.connected = False
        self.subscriptions.clear()

    async def subscribe(self, channel: str, callback=None) -> bool:
        """Subscribe to channel."""
        self.subscriptions.add(channel)
        if callback:
            self.callbacks[channel] = callback
        return True

    async def unsubscribe(self, channel: str) -> bool:
        """Unsubscribe from channel."""
        self.subscriptions.discard(channel)
        self.callbacks.pop(channel, None)
        return True

    async def send_message(self, message: dict[str, Any]) -> None:
        """Send message via WebSocket."""
        await self.message_queue.put(message)

    async def receive_message(self) -> dict[str, Any]:
        """Receive message from WebSocket."""
        return await self.message_queue.get()

    def is_connected(self) -> bool:
        """Check if connected."""
        return self.connected

    async def simulate_message(self, channel: str, data: dict[str, Any]) -> None:
        """Simulate incoming WebSocket message."""
        message = {
            "channel": channel,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await self.message_queue.put(message)

        # Trigger callback if registered
        if channel in self.callbacks:
            await self.callbacks[channel](message)


class MockRateLimiter:
    """Mock rate limiter."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_count = 0
        self.blocked = False

    async def acquire(self) -> bool:
        """Acquire rate limit token."""
        if self.blocked:
            return False
        self.request_count += 1
        return True

    async def release(self) -> None:
        """Release rate limit token."""
        if self.request_count > 0:
            self.request_count -= 1

    def reset(self) -> None:
        """Reset rate limiter."""
        self.request_count = 0
        self.blocked = False

    def block(self) -> None:
        """Block all requests."""
        self.blocked = True

    def unblock(self) -> None:
        """Unblock requests."""
        self.blocked = False


class MockConnectionPool:
    """Mock connection pool."""

    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections = []
        self.available = asyncio.Queue(maxsize=max_connections)
        self.in_use = set()

    async def initialize(self) -> None:
        """Initialize connection pool."""
        for i in range(self.max_connections):
            conn = MockConnection(f"conn_{i}")
            self.connections.append(conn)
            await self.available.put(conn)

    async def acquire(self) -> "MockConnection":
        """Acquire a connection from pool."""
        conn = await self.available.get()
        self.in_use.add(conn)
        return conn

    async def release(self, conn: "MockConnection") -> None:
        """Release connection back to pool."""
        if conn in self.in_use:
            self.in_use.remove(conn)
            await self.available.put(conn)

    async def close(self) -> None:
        """Close all connections."""
        for conn in self.connections:
            await conn.close()
        self.connections.clear()
        self.in_use.clear()


class MockConnection:
    """Mock connection object."""

    def __init__(self, conn_id: str):
        self.id = conn_id
        self.connected = False
        self.request_count = 0

    async def connect(self) -> bool:
        """Connect."""
        self.connected = True
        return True

    async def close(self) -> None:
        """Close connection."""
        self.connected = False

    async def request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        """Make a request."""
        self.request_count += 1
        return {
            "method": method,
            "endpoint": endpoint,
            "response": "mock_response",
            "request_id": self.request_count
        }

    def is_connected(self) -> bool:
        """Check if connected."""
        return self.connected


class MockHealthMonitor:
    """Mock health monitor."""

    def __init__(self):
        self.health_status = "healthy"
        self.metrics = {
            "uptime": 0,
            "request_count": 0,
            "error_count": 0,
            "latency_ms": 10
        }
        self.alerts = []

    async def check_health(self) -> dict[str, Any]:
        """Check system health."""
        return {
            "status": self.health_status,
            "metrics": self.metrics,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def record_metric(self, metric: str, value: Any) -> None:
        """Record a metric."""
        if metric in self.metrics:
            if isinstance(self.metrics[metric], (int, float)):
                self.metrics[metric] += value
            else:
                self.metrics[metric] = value

    async def trigger_alert(self, alert: str, severity: str = "warning") -> None:
        """Trigger an alert."""
        self.alerts.append({
            "alert": alert,
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def set_health_status(self, status: str) -> None:
        """Set health status."""
        self.health_status = status

    def get_alerts(self) -> list[dict[str, Any]]:
        """Get all alerts."""
        return self.alerts


class MockEventBus:
    """Mock event bus for testing event-driven architecture."""

    def __init__(self):
        self.subscribers = {}
        self.event_history = []

    async def publish(self, event_type: str, data: dict[str, Any]) -> None:
        """Publish an event."""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.event_history.append(event)

        # Notify subscribers
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                await callback(event)

    def subscribe(self, event_type: str, callback) -> None:
        """Subscribe to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback) -> None:
        """Unsubscribe from an event type."""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(callback)

    def get_event_history(self, event_type: str | None = None) -> list[dict[str, Any]]:
        """Get event history."""
        if event_type:
            return [e for e in self.event_history if e["type"] == event_type]
        return self.event_history


class MockExchangeFactory:
    """Mock exchange factory for testing."""

    @staticmethod
    def create_exchange(exchange_type: str, config: MockConfig | None = None):
        """Create a mock exchange instance."""
        if config is None:
            config = MockConfig()

        # Create all dependencies
        adapter = MockAdapter()
        order_manager = MockOrderManager()
        market_data_manager = MockMarketDataManager()
        websocket_manager = MockWebSocketManager()
        rate_limiter = MockRateLimiter()
        connection_pool = MockConnectionPool()
        health_monitor = MockHealthMonitor()
        event_bus = MockEventBus()

        # Create mock exchange object
        exchange = MagicMock()
        exchange.config = config
        exchange.adapter = adapter
        exchange.order_manager = order_manager
        exchange.market_data_manager = market_data_manager
        exchange.websocket_manager = websocket_manager
        exchange.rate_limiter = rate_limiter
        exchange.connection_pool = connection_pool
        exchange.health_monitor = health_monitor
        exchange.event_bus = event_bus
        exchange.exchange_type = exchange_type
        exchange.connected = False

        # Add common methods
        exchange.connect = AsyncMock(side_effect=lambda: setattr(exchange, "connected", True))
        exchange.disconnect = AsyncMock(side_effect=lambda: setattr(exchange, "connected", False))
        exchange.place_order = AsyncMock(side_effect=adapter.place_order)
        exchange.cancel_order = AsyncMock(side_effect=adapter.cancel_order)
        exchange.get_order = AsyncMock(side_effect=adapter.get_order)
        exchange.get_balance = AsyncMock(side_effect=adapter.get_balance)
        exchange.get_ticker = AsyncMock(side_effect=adapter.get_ticker)
        exchange.get_order_book = AsyncMock(side_effect=adapter.get_order_book)
        exchange.is_connected = Mock(return_value=lambda: exchange.connected)

        return exchange


# Export all mock classes
__all__ = [
    "MockAdapter",
    "MockConfig",
    "MockConnection",
    "MockConnectionPool",
    "MockEventBus",
    "MockExchangeFactory",
    "MockHealthMonitor",
    "MockMarketDataManager",
    "MockOrderManager",
    "MockRateLimiter",
    "MockWebSocketManager"
]
