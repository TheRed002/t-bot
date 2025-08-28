# Common Patterns - T-Bot Trading System

## Overview
This document describes the common design patterns and architectural approaches used throughout the T-Bot Trading System. These patterns are consistently applied across the codebase.

## 1. Service Layer Pattern

### Pattern Description
Services encapsulate business logic and orchestrate operations between repositories and external systems.

### Implementation Example
```python
from src.core.base.service import BaseService

class ExecutionService(BaseService):
    def __init__(self):
        super().__init__(name="ExecutionService")
        self.add_dependency("OrderRepository")
        self.add_dependency("DatabaseService")
        self.add_dependency("RiskManager")
    
    async def _do_start(self):
        """Resolve dependencies on service start."""
        self.order_repo = self.resolve_dependency("OrderRepository")
        self.db_service = self.resolve_dependency("DatabaseService")
        self.risk_manager = self.resolve_dependency("RiskManager")
    
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Business logic for order placement."""
        # Risk checks
        risk_check = await self.risk_manager.check_order(order_request)
        if not risk_check.approved:
            raise RiskManagementError(risk_check.reason)
        
        # Create order
        order = await self.order_repo.create(order_request.to_model())
        return OrderResponse.from_model(order)
```

### Usage Locations
- `src/execution/service.py`
- `src/ml/service.py`
- `src/data/services/`
- `src/state/services/`

## 2. Repository Pattern

### Pattern Description
Repositories provide an abstraction layer over database operations, implementing CRUD operations for domain entities.

### Implementation Example
```python
from src.database.repository.base import BaseRepository

class OrderRepository(BaseRepository[Order]):
    def __init__(self, session: AsyncSession):
        super().__init__(Order, session)
    
    async def get_active_orders(self, bot_id: str) -> list[Order]:
        """Domain-specific query method."""
        filters = {
            "bot_id": bot_id,
            "status": ["OPEN", "PENDING", "PARTIALLY_FILLED"]
        }
        return await self.get_all(filters=filters)
    
    async def cancel_order(self, order_id: str) -> Order:
        """Business operation on entity."""
        order = await self.get(order_id)
        if not order:
            raise OrderNotFoundError(f"Order {order_id} not found")
        
        order.status = "CANCELLED"
        order.cancelled_at = datetime.now(timezone.utc)
        return await self.update(order)
```

### Usage Locations
- `src/database/repository/`
- All `*_repository.py` files

## 3. Dependency Injection Pattern

### Pattern Description
Dependencies are injected rather than created directly, enabling loose coupling and testability.

### Implementation Example
```python
from src.core.dependency_injection import DependencyContainer

# Registration
container = DependencyContainer()
container.register_class(
    "DatabaseService", 
    DatabaseService, 
    config,
    singleton=True
)
container.register(
    "OrderRepository",
    lambda: OrderRepository(container.get("DatabaseService").session),
    singleton=False
)

# Resolution
class TradingBot:
    def __init__(self, container: DependencyContainer):
        self.order_repo = container.get("OrderRepository")
        self.risk_manager = container.get("RiskManager")
```

### Usage Locations
- `src/core/dependency_injection.py`
- `src/exchanges/factory.py` (ServiceContainer)
- Service initialization code

## 4. Factory Pattern

### Pattern Description
Factories create objects without specifying exact classes, useful for creating exchange connections and strategies.

### Implementation Example
```python
class ExchangeFactory:
    _exchanges: dict[str, type[BaseExchange]] = {
        "binance": BinanceExchange,
        "coinbase": CoinbaseExchange,
        "okx": OKXExchange,
    }
    
    @classmethod
    def create(cls, exchange_name: str, config: Config) -> BaseExchange:
        """Create exchange instance based on name."""
        exchange_class = cls._exchanges.get(exchange_name.lower())
        if not exchange_class:
            raise ValueError(f"Unknown exchange: {exchange_name}")
        
        return exchange_class(config)
```

### Usage Locations
- `src/exchanges/factory.py`
- `src/strategies/factory.py`
- `src/data/factory.py`

## 5. Decorator Pattern for Error Handling

### Pattern Description
Decorators wrap functions to add error handling, retry logic, and circuit breakers without modifying the function itself.

### Implementation Example
```python
from src.error_handling.decorators import with_retry, with_circuit_breaker

@with_retry(
    max_attempts=3,
    delay=1.0,
    backoff=2.0,
    exceptions=(NetworkError, TimeoutError)
)
@with_circuit_breaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=ExchangeError
)
async def fetch_market_data(self, symbol: str) -> MarketData:
    """Fetch with automatic retry and circuit breaker."""
    async with self.session.get(f"/api/ticker/{symbol}") as response:
        return MarketData.from_json(await response.json())
```

### Usage Locations
- Throughout exchange implementations
- Service layer methods
- External API calls

## 6. Unit of Work Pattern

### Pattern Description
Manages database transactions ensuring all operations complete or all rollback.

### Implementation Example
```python
from src.database.uow import UnitOfWork

class OrderService:
    async def create_order_with_position(self, order_data: dict) -> Order:
        async with UnitOfWork() as uow:
            # All operations in same transaction
            order = await uow.orders.create(order_data)
            position = await uow.positions.create({
                "order_id": order.id,
                "quantity": order.quantity
            })
            
            # Commit all or rollback all
            await uow.commit()
            return order
```

### Usage Locations
- `src/database/uow.py`
- Service methods requiring transactions

## 7. Observer Pattern (Event System)

### Pattern Description
Components emit events that other components can subscribe to, enabling loose coupling.

### Implementation Example
```python
from src.core.base.events import EventEmitter

class TradingEngine(EventEmitter):
    async def execute_trade(self, trade: Trade):
        # Execute trade logic
        result = await self._execute(trade)
        
        # Emit event for other components
        await self.emit("trade_executed", {
            "trade_id": trade.id,
            "result": result,
            "timestamp": datetime.now(timezone.utc)
        })

# Subscriber
class RiskMonitor:
    def __init__(self, trading_engine: TradingEngine):
        trading_engine.on("trade_executed", self.on_trade_executed)
    
    async def on_trade_executed(self, data: dict):
        """React to trade execution."""
        await self.update_exposure(data["trade_id"])
```

### Usage Locations
- WebSocket managers
- Trading engines
- State change notifications

## 8. Strategy Pattern

### Pattern Description
Encapsulates algorithms in separate classes allowing runtime algorithm selection.

### Implementation Example
```python
from abc import ABC, abstractmethod

class TradingStrategy(ABC):
    @abstractmethod
    async def generate_signals(self, market_data: MarketData) -> list[Signal]:
        """Generate trading signals from market data."""
        pass
    
    @abstractmethod
    async def calculate_position_size(self, signal: Signal, capital: Decimal) -> Decimal:
        """Calculate position size for signal."""
        pass

class MomentumStrategy(TradingStrategy):
    async def generate_signals(self, market_data: MarketData) -> list[Signal]:
        # Momentum-specific logic
        pass

class MeanReversionStrategy(TradingStrategy):
    async def generate_signals(self, market_data: MarketData) -> list[Signal]:
        # Mean reversion-specific logic
        pass
```

### Usage Locations
- `src/strategies/base.py`
- All strategy implementations

## 9. Singleton Pattern (via Dependency Injection)

### Pattern Description
Ensures only one instance of a class exists throughout application lifetime.

### Implementation Example
```python
class CacheManager:
    _instance: "CacheManager | None" = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.cache = {}
            self.initialized = True

# Or via DI Container
container.register("CacheManager", CacheManager(), singleton=True)
```

### Usage Locations
- `src/core/caching/cache_manager.py`
- Database connections
- Configuration objects

## 10. Circuit Breaker Pattern

### Pattern Description
Prevents cascading failures by stopping calls to failing services.

### Implementation Example
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise ServiceUnavailableError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

### Usage Locations
- Exchange connections
- External API calls
- Database operations

## 11. Adapter Pattern

### Pattern Description
Converts interface of a class into another interface clients expect.

### Implementation Example
```python
class ExchangeAdapter(ABC):
    """Adapts exchange-specific APIs to common interface."""
    
    @abstractmethod
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Common interface for all exchanges."""
        pass

class BinanceAdapter(ExchangeAdapter):
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Adapt to Binance-specific API."""
        binance_order = {
            "symbol": order.symbol.replace("/", ""),
            "side": order.side.upper(),
            "type": self._map_order_type(order.type),
            "quantity": str(order.quantity)
        }
        response = await self.client.create_order(**binance_order)
        return self._map_response(response)
```

### Usage Locations
- `src/data/sources/adapter.py`
- Exchange implementations

## 12. Health Check Pattern

### Pattern Description
Components provide health status for monitoring and automatic recovery.

### Implementation Example
```python
from src.core.base.interfaces import HealthStatus

class DatabaseService(BaseComponent):
    async def health_check(self) -> HealthStatus:
        """Check database health."""
        try:
            # Test connection
            await self.session.execute("SELECT 1")
            
            # Check connection pool
            pool_stats = self.get_pool_stats()
            if pool_stats["active"] > pool_stats["max_size"] * 0.9:
                return HealthStatus.DEGRADED
            
            return HealthStatus.HEALTHY
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return HealthStatus.UNHEALTHY
```

### Usage Locations
- All BaseComponent implementations
- Service health endpoints

## 13. Audit Trail Pattern

### Pattern Description
Tracks all significant operations for compliance and debugging.

### Implementation Example
```python
class AuditMixin:
    created_by = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_by = Column(String(255))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def create_audit_log(cls, operation: str, entity_id: str, details: dict):
        return AuditLog(
            entity_type=cls.__name__,
            entity_id=entity_id,
            operation=operation,
            details=details,
            timestamp=datetime.now(timezone.utc)
        )
```

### Usage Locations
- `src/database/models/base.py`
- All auditable models

## 14. Caching Pattern

### Pattern Description
Stores frequently accessed data in memory to reduce database load.

### Implementation Example
```python
from src.core.caching.cache_manager import CacheManager

class MarketDataService:
    def __init__(self):
        self.cache = CacheManager.get_instance()
    
    async def get_ticker(self, symbol: str) -> Ticker:
        # Check cache first
        cache_key = f"ticker:{symbol}"
        cached = await self.cache.get(cache_key)
        if cached:
            return Ticker.from_dict(cached)
        
        # Fetch from source
        ticker = await self._fetch_ticker(symbol)
        
        # Cache with TTL
        await self.cache.set(cache_key, ticker.to_dict(), ttl=5)
        return ticker
```

### Usage Locations
- Market data services
- Configuration caching
- Session management

## 15. Async Context Manager Pattern

### Pattern Description
Ensures proper resource initialization and cleanup in async contexts.

### Implementation Example
```python
class WebSocketConnection:
    async def __aenter__(self):
        """Initialize connection."""
        self.ws = await aiohttp.ClientSession().ws_connect(self.url)
        await self._authenticate()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup connection."""
        if self.ws:
            await self.ws.close()
        await self.session.close()

# Usage
async with WebSocketConnection(url) as conn:
    async for message in conn:
        await process_message(message)
```

### Usage Locations
- WebSocket connections
- Database sessions
- File operations

## Summary
These patterns form the architectural foundation of the T-Bot Trading System. They ensure consistency, maintainability, and reliability across the entire codebase. When implementing new features, identify which patterns apply and follow the established implementations.