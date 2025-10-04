"""
Integration test configuration and fixtures.

Provides common test fixtures, configuration, and utilities for integration tests.
"""

import pytest
import asyncio
import logging
import tempfile
import os
import json
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from tests.integration.base_integration import (
    MockExchangeFactory, MockStrategyFactory, PerformanceMonitor
)
from src.core.types import (
    MarketData, Order, OrderSide, OrderType, OrderStatus, Position,
    Signal, SignalDirection, BotStatus
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    
    yield loop
    
    # Clean up
    loop.close()


@pytest.fixture(scope="session")
def integration_test_config():
    """Integration test configuration."""
    return {
        "database": {
            "url": "sqlite:///:memory:",  # In-memory database for testing
            "pool_size": 5,
            "max_overflow": 10
        },
        "exchanges": {
            "binance": {
                "enabled": True,
                "testnet": True,
                "api_key": "test_binance_key",
                "api_secret": "test_binance_secret",
                "rate_limit": 1200
            },
            "coinbase": {
                "enabled": True,
                "sandbox": True,
                "api_key": "test_coinbase_key",
                "api_secret": "test_coinbase_secret",
                "passphrase": "test_passphrase",
                "rate_limit": 600
            },
            "okx": {
                "enabled": True,
                "demo": True,
                "api_key": "test_okx_key",
                "api_secret": "test_okx_secret",
                "passphrase": "test_okx_passphrase",
                "rate_limit": 600
            }
        },
        "risk": {
            "max_position_size": 50000.0,
            "max_daily_loss": 5000.0,
            "max_drawdown": 0.15,
            "circuit_breakers_enabled": True,
            "position_sizing_method": "FIXED"
        },
        "strategies": {
            "momentum": {
                "enabled": True,
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "timeframe": "1m",
                "parameters": {
                    "lookback_period": 14,
                    "threshold": 0.02
                }
            },
            "mean_reversion": {
                "enabled": True,
                "symbols": ["BTC/USDT", "ETH/USDT", "ADA/USDT"],
                "timeframe": "5m",
                "parameters": {
                    "bollinger_period": 20,
                    "std_dev": 2.0
                }
            }
        },
        "websocket": {
            "reconnect_attempts": 5,
            "reconnect_delay": 5.0,
            "heartbeat_interval": 30.0,
            "message_timeout": 10.0
        },
        "api": {
            "host": "localhost",
            "port": 8000,
            "cors_origins": ["http://localhost:3000"],
            "jwt_secret": "test_jwt_secret_key_for_integration_tests",
            "jwt_expiry_minutes": 15
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }


@pytest.fixture
async def temp_directory():
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="tbot_integration_test_")
    
    yield Path(temp_dir)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def mock_exchanges():
    """Create mock exchanges for testing."""
    exchanges = {
        "binance": MockExchangeFactory.create_binance_mock(
            initial_balance={"USDT": Decimal("100000.0"), "BTC": Decimal("0.0"), "ETH": Decimal("0.0")},
            market_prices={"BTC/USDT": Decimal("50000.0"), "ETH/USDT": Decimal("3000.0")}
        ),
        "coinbase": MockExchangeFactory.create_coinbase_mock(
            initial_balance={"USDT": Decimal("50000.0"), "BTC": Decimal("0.0"), "ETH": Decimal("0.0")},
            market_prices={"BTC/USDT": Decimal("50020.0"), "ETH/USDT": Decimal("3005.0")}
        ),
        "okx": MockExchangeFactory.create_binance_mock(  # Reuse binance factory
            initial_balance={"USDT": Decimal("75000.0"), "BTC": Decimal("0.0"), "ETH": Decimal("0.0")},
            market_prices={"BTC/USDT": Decimal("49995.0"), "ETH/USDT": Decimal("2998.0")}
        )
    }
    exchanges["okx"].name = "okx"
    
    yield exchanges
    
    # Cleanup connections
    for exchange in exchanges.values():
        if hasattr(exchange, 'close') and callable(exchange.close):
            try:
                await exchange.close()
            except Exception:
                pass


@pytest.fixture
async def mock_strategies():
    """Create mock strategies for testing."""
    strategies = {
        "momentum": MockStrategyFactory.create_momentum_strategy(
            symbol="BTC/USDT",
            signal_confidence=0.8,
            signal_frequency=0.4
        ),
        "mean_reversion": MockStrategyFactory.create_mean_reversion_strategy(
            symbol="BTC/USDT",
            signal_confidence=0.7,
            contrarian=True
        )
    }
    
    yield strategies


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"]
    base_prices = {
        "BTC/USDT": Decimal("50000.0"),
        "ETH/USDT": Decimal("3000.0"),
        "ADA/USDT": Decimal("0.5"),
        "DOT/USDT": Decimal("7.0"),
        "LINK/USDT": Decimal("15.0")
    }
    
    market_data = []
    for symbol in symbols:
        base_price = base_prices[symbol]
        spread = base_price * Decimal("0.001")  # 0.1% spread
        
        data = MarketData(
            symbol=symbol,
            price=base_price,
            bid=base_price - spread/2,
            ask=base_price + spread/2,
            volume=Decimal("1000.0"),
            timestamp=datetime.now(timezone.utc),
            open_price=base_price * Decimal("0.995"),  # -0.5% from open
            high_price=base_price * Decimal("1.02"),   # +2% high
            low_price=base_price * Decimal("0.98")     # -2% low
        )
        market_data.append(data)
    
    return market_data


@pytest.fixture
def sample_positions():
    """Generate sample positions for testing."""
    positions = [
        Position(
            symbol="BTC/USDT",
            side=OrderSide.LONG,
            quantity=Decimal("1.0"),
            average_price=Decimal("49000.0"),
            current_price=Decimal("50000.0"),
            unrealized_pnl=Decimal("1000.0"),
            realized_pnl=Decimal("0.0"),
            timestamp=datetime.now(timezone.utc) - timedelta(hours=2)
        ),
        Position(
            symbol="ETH/USDT",
            side=OrderSide.LONG,
            quantity=Decimal("10.0"),
            average_price=Decimal("2900.0"),
            current_price=Decimal("3000.0"),
            unrealized_pnl=Decimal("1000.0"),
            realized_pnl=Decimal("0.0"),
            timestamp=datetime.now(timezone.utc) - timedelta(hours=1)
        ),
        Position(
            symbol="ADA/USDT",
            side=OrderSide.SHORT,
            quantity=Decimal("2000.0"),
            average_price=Decimal("0.52"),
            current_price=Decimal("0.50"),
            unrealized_pnl=Decimal("40.0"),
            realized_pnl=Decimal("0.0"),
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=30)
        )
    ]
    
    return positions


@pytest.fixture
def sample_orders():
    """Generate sample orders for testing."""
    orders = [
        Order(
            id="order_001",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("1.0"),
            remaining_quantity=Decimal("0.0"),
            average_price=Decimal("49000.0"),
            timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
            exchange="binance"
        ),
        Order(
            id="order_002",
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10.0"),
            price=Decimal("2900.0"),
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("10.0"),
            remaining_quantity=Decimal("0.0"),
            average_price=Decimal("2900.0"),
            timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
            exchange="binance"
        ),
        Order(
            id="order_003",
            symbol="ADA/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("2000.0"),
            price=Decimal("0.52"),
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=Decimal("1200.0"),
            remaining_quantity=Decimal("800.0"),
            average_price=Decimal("0.52"),
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=30),
            exchange="coinbase"
        )
    ]
    
    return orders


@pytest.fixture
def sample_signals():
    """Generate sample trading signals for testing."""
    signals = [
        Signal(
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            confidence=0.85,
            price=Decimal("50000.0"),
            timestamp=datetime.now(timezone.utc),
            strategy_name="momentum",
            metadata={
                "rsi": 35.2,
                "macd": 1.5,
                "volume_ratio": 1.8,
                "trend": "bullish"
            }
        ),
        Signal(
            symbol="ETH/USDT",
            direction=SignalDirection.SELL,
            confidence=0.72,
            price=Decimal("3000.0"),
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=5),
            strategy_name="mean_reversion",
            metadata={
                "bollinger_position": 0.95,
                "rsi": 78.5,
                "mean_reversion_strength": 0.7
            }
        ),
        Signal(
            symbol="ADA/USDT",
            direction=SignalDirection.HOLD,
            confidence=0.45,
            price=Decimal("0.50"),
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=10),
            strategy_name="momentum",
            metadata={
                "rsi": 52.1,
                "macd": -0.1,
                "volume_ratio": 0.9,
                "trend": "sideways"
            }
        )
    ]
    
    return signals


@pytest.fixture
async def performance_monitor():
    """Create performance monitor for testing."""
    monitor = PerformanceMonitor()
    monitor.start()
    
    yield monitor
    
    monitor.stop()


@pytest.fixture
async def mock_database():
    """Create mock database connection for testing."""
    
    class MockDatabase:
        def __init__(self):
            self.connected = True
            self.queries_executed = 0
            self.transactions = 0
            self.data_store = {}  # Simple in-memory storage
            
        async def execute(self, query: str, *args) -> int:
            """Execute a query."""
            self.queries_executed += 1
            
            # Simple mock query processing
            if "INSERT" in query.upper():
                # Mock insert operation
                return 1  # 1 row affected
            elif "UPDATE" in query.upper():
                # Mock update operation
                return 1  # 1 row affected
            elif "DELETE" in query.upper():
                # Mock delete operation
                return 1  # 1 row affected
            
            return 0
            
        async def fetch_all(self, query: str, *args) -> List[Dict[str, Any]]:
            """Fetch all results from query."""
            self.queries_executed += 1
            
            # Return mock data based on query
            if "positions" in query.lower():
                return [
                    {
                        "id": 1,
                        "symbol": "BTC/USDT",
                        "side": "LONG",
                        "quantity": "1.0",
                        "average_price": "49000.0"
                    }
                ]
            elif "orders" in query.lower():
                return [
                    {
                        "id": "order_001",
                        "symbol": "BTC/USDT",
                        "side": "BUY",
                        "status": "FILLED"
                    }
                ]
            
            return []
            
        async def fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
            """Fetch one result from query."""
            results = await self.fetch_all(query, *args)
            return results[0] if results else None
            
        async def begin_transaction(self):
            """Begin database transaction."""
            self.transactions += 1
            return MockTransaction()
            
        async def close(self):
            """Close database connection."""
            self.connected = False
            
        def get_stats(self) -> Dict[str, Any]:
            """Get database statistics."""
            return {
                "connected": self.connected,
                "queries_executed": self.queries_executed,
                "transactions": self.transactions
            }
    
    class MockTransaction:
        def __init__(self):
            self.committed = False
            self.rolled_back = False
            
        async def commit(self):
            """Commit transaction."""
            self.committed = True
            
        async def rollback(self):
            """Rollback transaction."""
            self.rolled_back = True
            
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None and not self.rolled_back:
                await self.rollback()
            elif not self.committed and not self.rolled_back:
                await self.commit()
    
    db = MockDatabase()
    
    yield db
    
    await db.close()


@pytest.fixture
def mock_jwt_handler():
    """Create mock JWT handler for authentication testing."""
    
    class MockJWTHandler:
        def __init__(self):
            self.secret_key = "test_jwt_secret_key"
            self.algorithm = "HS256"
            self.tokens_issued = 0
            self.tokens_validated = 0
            
        def generate_token(self, user_id: str, permissions: List[str] = None) -> str:
            """Generate JWT token."""
            if permissions is None:
                permissions = []
                
            self.tokens_issued += 1
            
            # Return a mock token (in real implementation would use jwt.encode)
            return f"mock_jwt_token_{user_id}_{self.tokens_issued}"
            
        def validate_token(self, token: str) -> Dict[str, Any]:
            """Validate JWT token."""
            self.tokens_validated += 1
            
            if token.startswith("mock_jwt_token_"):
                # Extract user_id from mock token
                parts = token.split("_")
                if len(parts) >= 4:
                    user_id = parts[3]
                    return {
                        "sub": user_id,
                        "permissions": ["read", "write"],
                        "exp": datetime.now(timezone.utc) + timedelta(hours=1)
                    }
            
            raise Exception("Invalid token")
            
        def get_stats(self) -> Dict[str, int]:
            """Get JWT handler statistics."""
            return {
                "tokens_issued": self.tokens_issued,
                "tokens_validated": self.tokens_validated
            }
    
    return MockJWTHandler()


@pytest.fixture
async def mock_websocket_manager():
    """Create mock WebSocket manager for testing."""
    
    class MockWebSocketManager:
        def __init__(self):
            self.connections = {}
            self.message_queue = []
            self.connection_count = 0
            
        async def connect(self, exchange_name: str, symbols: List[str]) -> str:
            """Connect to exchange WebSocket."""
            connection_id = f"ws_{exchange_name}_{self.connection_count}"
            self.connection_count += 1
            
            self.connections[connection_id] = {
                "exchange": exchange_name,
                "symbols": symbols,
                "connected": True,
                "message_count": 0
            }
            
            # Start message simulation
            asyncio.create_task(self._simulate_messages(connection_id))
            
            return connection_id
            
        async def disconnect(self, connection_id: str):
            """Disconnect WebSocket."""
            if connection_id in self.connections:
                self.connections[connection_id]["connected"] = False
                
        async def _simulate_messages(self, connection_id: str):
            """Simulate incoming WebSocket messages."""
            connection = self.connections.get(connection_id)
            if not connection:
                return
                
            while connection["connected"]:
                # Generate mock market data message
                import random
                symbol = random.choice(connection["symbols"])
                
                message = {
                    "type": "ticker",
                    "symbol": symbol,
                    "price": str(50000 + random.uniform(-1000, 1000)),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "exchange": connection["exchange"]
                }
                
                self.message_queue.append(message)
                connection["message_count"] += 1
                
                await asyncio.sleep(0.5)  # 2 messages per second
                
        def get_messages(self) -> List[Dict[str, Any]]:
            """Get queued messages."""
            messages = self.message_queue.copy()
            self.message_queue.clear()
            return messages
            
        def get_connection_stats(self) -> Dict[str, Any]:
            """Get connection statistics."""
            active_connections = sum(1 for conn in self.connections.values() if conn["connected"])
            total_messages = sum(conn["message_count"] for conn in self.connections.values())
            
            return {
                "total_connections": len(self.connections),
                "active_connections": active_connections,
                "total_messages": total_messages
            }
    
    manager = MockWebSocketManager()
    
    yield manager
    
    # Cleanup all connections
    for connection_id in list(manager.connections.keys()):
        await manager.disconnect(connection_id)


def register_all_services_for_testing():
    """
    Register ALL services in correct dependency order for integration tests.

    This comprehensive DI registration ensures all modules have their dependencies
    properly configured without circular dependency issues.

    Returns:
        Configured DependencyInjector instance
    """
    from src.core.dependency_injection import DependencyInjector
    from src.core.di_master_registration import register_all_services
    from src.core.config import Config

    # Create test configuration
    config = Config()

    # Use the master registration to register all services in order
    injector = register_all_services(config=config)

    logger.info("Registered all services for integration testing")
    return injector


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Setup logging for integration tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ]
    )

    # Reduce noise from external libraries
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.info("Integration test logging configured")


@pytest.fixture
def test_data_generator():
    """Generate various test data for integration tests."""
    
    class TestDataGenerator:
        def __init__(self):
            self.symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"]
            self.exchanges = ["binance", "coinbase", "okx"]
            
        def generate_market_data(self, count: int = 100) -> List[MarketData]:
            """Generate market data for testing."""
            import random
            
            market_data = []
            base_prices = {
                "BTC/USDT": 50000,
                "ETH/USDT": 3000,
                "ADA/USDT": 0.5,
                "DOT/USDT": 7.0,
                "LINK/USDT": 15.0
            }
            
            for i in range(count):
                symbol = random.choice(self.symbols)
                base_price = base_prices[symbol]
                
                # Add some price variation
                price_variation = random.uniform(-0.05, 0.05)  # ±5%
                price = Decimal(str(base_price * (1 + price_variation)))
                
                data = MarketData(
                    symbol=symbol,
                    price=price,
                    bid=price * Decimal("0.999"),
                    ask=price * Decimal("1.001"),
                    volume=Decimal(str(random.uniform(100, 2000))),
                    timestamp=datetime.now(timezone.utc) - timedelta(seconds=i),
                    open_price=price * Decimal("0.995"),
                    high_price=price * Decimal("1.02"),
                    low_price=price * Decimal("0.98")
                )
                market_data.append(data)
                
            return market_data
            
        def generate_orders(self, count: int = 50) -> List[Order]:
            """Generate orders for testing."""
            import random
            
            orders = []
            statuses = [OrderStatus.NEW, OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCELLED]
            
            for i in range(count):
                order = Order(
                    id=f"test_order_{i:04d}",
                    symbol=random.choice(self.symbols),
                    side=random.choice([OrderSide.BUY, OrderSide.SELL]),
                    order_type=random.choice([OrderType.MARKET, OrderType.LIMIT]),
                    quantity=Decimal(str(random.uniform(0.1, 10.0))),
                    price=Decimal(str(random.uniform(100, 60000))) if random.random() > 0.3 else None,
                    status=random.choice(statuses),
                    filled_quantity=Decimal("0"),
                    remaining_quantity=Decimal("0"),
                    average_price=None,
                    timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
                    exchange=random.choice(self.exchanges)
                )
                
                # Set filled quantities based on status
                if order.status == OrderStatus.FILLED:
                    order.filled_quantity = order.quantity
                    order.remaining_quantity = Decimal("0")
                    order.average_price = order.price or Decimal(str(random.uniform(100, 60000)))
                elif order.status == OrderStatus.PARTIALLY_FILLED:
                    fill_ratio = Decimal(str(random.uniform(0.3, 0.8)))
                    order.filled_quantity = order.quantity * fill_ratio
                    order.remaining_quantity = order.quantity - order.filled_quantity
                    order.average_price = order.price or Decimal(str(random.uniform(100, 60000)))
                
                orders.append(order)
                
            return orders
            
        def generate_signals(self, count: int = 20) -> List[Signal]:
            """Generate trading signals for testing."""
            import random
            
            signals = []
            strategies = ["momentum", "mean_reversion", "breakout", "ml_model"]
            
            for i in range(count):
                signal = Signal(
                    symbol=random.choice(self.symbols),
                    direction=random.choice([SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD]),
                    confidence=random.uniform(0.3, 0.95),
                    price=Decimal(str(random.uniform(100, 60000))),
                    timestamp=datetime.now(timezone.utc) - timedelta(minutes=i * 5),
                    strategy_name=random.choice(strategies),
                    metadata={
                        "rsi": random.uniform(20, 80),
                        "macd": random.uniform(-2, 2),
                        "volume_ratio": random.uniform(0.5, 3.0),
                        "signal_strength": random.uniform(0.1, 1.0)
                    }
                )
                signals.append(signal)
                
            return signals
    
    return TestDataGenerator()


@pytest.fixture
async def load_test_config():
    """Configuration specific to load testing."""
    return {
        "performance_thresholds": {
            "max_response_time_ms": 100,
            "max_p95_response_time_ms": 200,
            "min_success_rate": 0.95,
            "max_error_rate": 0.05,
            "min_throughput_rps": 100
        },
        "load_test_scenarios": {
            "burst_orders": {
                "concurrent_orders": 100,
                "burst_duration_seconds": 10,
                "expected_success_rate": 0.98
            },
            "sustained_load": {
                "rps": 50,
                "duration_seconds": 60,
                "ramp_up_seconds": 10
            },
            "stress_test": {
                "max_rps": 500,
                "duration_seconds": 30,
                "acceptable_degradation": 0.2
            }
        },
        "resource_limits": {
            "max_memory_mb": 1000,
            "max_cpu_percent": 80,
            "max_connections": 100,
            "max_websocket_connections": 50
        },
        "database_performance": {
            "max_query_time_ms": 50,
            "max_transaction_time_ms": 100,
            "connection_pool_size": 10,
            "max_concurrent_queries": 20
        }
    }


@pytest.fixture
async def system_monitor():
    """System resource monitor for performance testing."""
    import psutil
    import asyncio
    from typing import List, Tuple
    
    class SystemMonitor:
        def __init__(self):
            self.monitoring = False
            self.metrics: List[Tuple[float, Dict[str, float]]] = []
            self.monitor_task = None
            
        async def start_monitoring(self, interval: float = 1.0):
            """Start system monitoring."""
            self.monitoring = True
            self.metrics.clear()
            self.monitor_task = asyncio.create_task(self._monitor_loop(interval))
            
        async def stop_monitoring(self):
            """Stop system monitoring."""
            self.monitoring = False
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
                    
        async def _monitor_loop(self, interval: float):
            """Monitor system resources in a loop."""
            while self.monitoring:
                try:
                    process = psutil.Process()
                    system = psutil
                    
                    metrics = {
                        "cpu_percent": process.cpu_percent(),
                        "memory_mb": process.memory_info().rss / 1024 / 1024,
                        "memory_percent": process.memory_percent(),
                        "num_threads": process.num_threads(),
                        "num_connections": len(process.connections()),
                        "system_cpu": system.cpu_percent(),
                        "system_memory": system.virtual_memory().percent,
                        "disk_io_read": system.disk_io_counters().read_bytes if system.disk_io_counters() else 0,
                        "disk_io_write": system.disk_io_counters().write_bytes if system.disk_io_counters() else 0,
                        "network_sent": system.net_io_counters().bytes_sent if system.net_io_counters() else 0,
                        "network_recv": system.net_io_counters().bytes_recv if system.net_io_counters() else 0
                    }
                    
                    self.metrics.append((asyncio.get_event_loop().time(), metrics))
                    
                    await asyncio.sleep(interval)
                except Exception:
                    # Continue monitoring even if individual measurement fails
                    await asyncio.sleep(interval)
                    
        def get_metrics_summary(self) -> Dict[str, Any]:
            """Get summary of collected metrics."""
            if not self.metrics:
                return {}
                
            metric_keys = self.metrics[0][1].keys()
            summary = {}
            
            for key in metric_keys:
                values = [metric[1][key] for metric in self.metrics]
                summary[key] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "samples": len(values)
                }
                
            return summary
            
        def get_peak_usage(self) -> Dict[str, float]:
            """Get peak resource usage."""
            if not self.metrics:
                return {}
                
            peak_usage = {}
            for _, metrics in self.metrics:
                for key, value in metrics.items():
                    if key not in peak_usage or value > peak_usage[key]:
                        peak_usage[key] = value
                        
            return peak_usage
    
    monitor = SystemMonitor()
    yield monitor
    await monitor.stop_monitoring()


@pytest.fixture
async def concurrent_test_executor():
    """Executor for running concurrent performance tests."""
    import concurrent.futures
    import asyncio
    from typing import Callable, List, Any
    
    class ConcurrentTestExecutor:
        def __init__(self):
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=50)
            
        async def run_concurrent_tasks(
            self, 
            task_func: Callable, 
            task_args: List[Any], 
            max_concurrent: int = 10
        ) -> List[Any]:
            """Run tasks concurrently with controlled concurrency."""
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def limited_task(args):
                async with semaphore:
                    if asyncio.iscoroutinefunction(task_func):
                        return await task_func(args)
                    else:
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(self.executor, task_func, args)
            
            tasks = [limited_task(args) for args in task_args]
            return await asyncio.gather(*tasks, return_exceptions=True)
            
        async def run_load_pattern(
            self,
            task_func: Callable,
            rps: int,
            duration_seconds: int,
            ramp_up_seconds: int = 0
        ) -> List[Any]:
            """Run tasks following a specific load pattern."""
            results = []
            start_time = asyncio.get_event_loop().time()
            end_time = start_time + duration_seconds
            
            # Calculate ramp-up
            if ramp_up_seconds > 0:
                ramp_up_end = start_time + ramp_up_seconds
                max_rps = rps
            else:
                ramp_up_end = start_time
                max_rps = rps
            
            task_id = 0
            while asyncio.get_event_loop().time() < end_time:
                current_time = asyncio.get_event_loop().time()
                
                # Calculate current RPS based on ramp-up
                if current_time < ramp_up_end:
                    progress = (current_time - start_time) / ramp_up_seconds
                    current_rps = max_rps * progress
                else:
                    current_rps = max_rps
                    
                if current_rps > 0:
                    interval = 1.0 / current_rps
                    
                    # Execute task
                    task_id += 1
                    try:
                        if asyncio.iscoroutinefunction(task_func):
                            result = await task_func(task_id)
                        else:
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(self.executor, task_func, task_id)
                        results.append(result)
                    except Exception as e:
                        results.append(e)
                    
                    # Wait for next task
                    await asyncio.sleep(interval)
                else:
                    await asyncio.sleep(0.1)  # Small delay during ramp-up
                    
            return results
            
        def cleanup(self):
            """Cleanup resources."""
            self.executor.shutdown(wait=True)
    
    executor = ConcurrentTestExecutor()
    yield executor
    executor.cleanup()


# Pytest configuration - pytest_plugins moved to root conftest.py