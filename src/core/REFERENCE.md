# Core Module Reference

This document provides a comprehensive API reference for the `src/core` module of the T-Bot Trading System. The core module contains foundational classes, interfaces, types, configuration management, and utilities that serve as the backbone for the entire trading system.

## Table of Contents

1. [Overview](#overview)
2. [Base Components](#base-components)
3. [Interfaces and Protocols](#interfaces-and-protocols)
4. [Type System](#type-system)
5. [Configuration Management](#configuration-management)
6. [Dependency Injection](#dependency-injection)
7. [Caching System](#caching-system)
8. [Performance Monitoring](#performance-monitoring)
9. [Resource Management](#resource-management)
10. [Integration Framework](#integration-framework)
11. [Utilities](#utilities)

## Overview

The core module provides:
- **Base classes** with lifecycle management, health checks, and dependency injection
- **Type system** with comprehensive validation and serialization support
- **Configuration management** with environment-aware settings
- **Caching layer** with distributed locking and warming strategies
- **Performance monitoring** with metrics collection and optimization
- **Resource management** with proper cleanup and leak prevention
- **Integration framework** for environment-aware service orchestration

## Base Components

### BaseComponent

**File:** `src/core/base/component.py`

Enhanced base component with complete lifecycle management.

```python
class BaseComponent(Lifecycle, HealthCheckable, Injectable, Loggable, Monitorable, Configurable)
```

#### Key Features:
- Async lifecycle management (start/stop/restart)
- Health check framework
- Dependency injection support
- Structured logging with correlation IDs
- Metrics collection
- Configuration management
- Resource cleanup
- Error handling with context

#### Methods:

```python
async def start(self) -> None
    """Start the component and initialize resources."""

async def stop(self) -> None
    """Stop the component and cleanup resources."""

async def restart(self) -> None
    """Restart the component."""

async def health_check(self) -> HealthCheckResult
    """Perform comprehensive health check."""

async def ready_check(self) -> HealthCheckResult
    """Check if component is ready to serve requests."""

async def live_check(self) -> HealthCheckResult
    """Check if component is alive and responsive."""

def configure(self, config: ConfigDict) -> None
    """Configure component with provided settings."""

def get_config(self) -> ConfigDict
    """Get current component configuration."""

def get_metrics(self) -> dict[str, Any]
    """Get current component metrics."""
```

#### Properties:

```python
@property
def name(self) -> str
    """Get component name."""

@property
def is_running(self) -> bool
    """Check if component is currently running."""

@property
def uptime(self) -> float
    """Get component uptime in seconds."""
```

### BaseService

**File:** `src/core/base/service.py`

Base service class implementing the service layer pattern.

```python
class BaseService(BaseComponent, ServiceComponent)
```

#### Key Features:
- Business logic organization
- Transaction management
- Service-to-service communication
- Circuit breaker patterns
- Retry mechanisms
- Operation monitoring

#### Methods:

```python
async def execute_with_monitoring(
    self, operation_name: str, operation_func: Any, *args, **kwargs
) -> Any
    """Execute service operation with monitoring and validation."""

def resolve_dependency(self, dependency_name: str) -> Any
    """Resolve a dependency from the DI container."""

def configure_circuit_breaker(
    self, enabled: bool = True, threshold: int = 5, timeout: int = 60
) -> None
    """Configure circuit breaker settings."""

def configure_retry(
    self, enabled: bool = True, max_retries: int = 3, 
    delay: float = 1.0, backoff: float = 2.0
) -> None
    """Configure retry settings."""
```

## Interfaces and Protocols

### Core Protocols

**File:** `src/core/base/interfaces.py`

Defines contracts that all base classes must follow.

#### Lifecycle Protocol

```python
class Lifecycle(Protocol):
    async def start(self) -> None
    async def stop(self) -> None
    async def restart(self) -> None
    @property
    def is_running(self) -> bool
```

#### HealthCheckable Protocol

```python
class HealthCheckable(Protocol):
    async def health_check(self) -> HealthCheckResult
    async def ready_check(self) -> HealthCheckResult
    async def live_check(self) -> HealthCheckResult
```

#### Injectable Protocol

```python
class Injectable(Protocol):
    def configure_dependencies(self, container: Any) -> None
    def get_dependencies(self) -> list[str]
```

#### Repository Protocol

```python
class Repository(Protocol):
    async def create(self, entity: Any) -> Any
    async def get_by_id(self, entity_id: Any) -> Any | None
    async def update(self, entity: Any) -> Any
    async def delete(self, entity_id: Any) -> bool
    async def list(
        self, limit: int | None = None, 
        offset: int | None = None, 
        filters: dict[str, Any] | None = None
    ) -> list[Any]
```

### Health Check Types

```python
class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class HealthCheckResult:
    def __init__(
        self, status: HealthStatus, 
        details: dict[str, Any] | None = None,
        message: str | None = None,
        check_time: datetime | None = None
    )
```

## Type System

### Base Types

**File:** `src/core/types/base.py`

#### Core Enumerations

```python
class TradingMode(Enum):
    """Trading mode enumeration for different execution environments."""
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    SIMULATION = "simulation"

    def is_real_money(self) -> bool
    def allows_testing(self) -> bool
    @classmethod
    def from_string(cls, value: str) -> "TradingMode"

class ExchangeType(Enum):
    """Exchange types for API integration."""
    BINANCE = "binance"
    OKX = "okx"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BYBIT = "bybit"

    def get_rate_limit(self) -> int
    def supports_websocket(self) -> bool
    def get_base_url(self) -> str

class ValidationLevel(Enum):
    """Data validation severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    def should_halt_system(self) -> bool
    def requires_immediate_attention(self) -> bool
    def get_numeric_value(self) -> int
```

#### Base Model Classes

```python
class BaseValidatedModel(BaseModel):
    """Enhanced base model with comprehensive validation."""
    created_at: datetime
    updated_at: datetime | None
    metadata: dict[str, Any]

    def mark_updated(self) -> None
    def to_dict(self) -> dict[str, Any]
    def to_json(self) -> str
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseValidatedModel"
    def add_metadata(self, key: str, value: Any) -> None

class FinancialBaseModel(BaseValidatedModel):
    """Base model for financial data with Decimal precision handling."""
    
    def to_dict_with_decimals(self) -> dict[str, Any]
    def validate_financial_precision(self) -> bool
```

### Trading Types

**File:** `src/core/types/trading.py`

#### Trading Enumerations

```python
class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"
    LONG = "long"
    SHORT = "short"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderStatus(Enum):
    NEW = "new"
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"

class TimeInForce(Enum):
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
```

#### Trading Models

```python
class Signal(BaseModel):
    """Trading signal with direction and metadata."""
    symbol: str
    direction: SignalDirection
    strength: Decimal
    timestamp: datetime
    source: str
    metadata: dict[str, Any]

class OrderRequest(BaseModel):
    """Request to create an order."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Decimal | None
    time_in_force: TimeInForce

class Order(BaseModel):
    """Complete order information."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    status: OrderStatus
    created_at: datetime
    exchange: str

    def is_filled(self) -> bool
    def is_active(self) -> bool

class Position(BaseModel):
    """Trading position information."""
    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    unrealized_pnl: Decimal | None
    status: PositionStatus
    opened_at: datetime
    exchange: str

    def is_open(self) -> bool
    def calculate_pnl(self, current_price: Decimal) -> Decimal

class Balance(BaseModel):
    """Account balance information."""
    currency: str
    available: Decimal
    locked: Decimal
    total: Decimal
    exchange: str
    updated_at: datetime
```

### Market Data Types

**File:** `src/core/types/market.py` (referenced in type exports)

```python
class MarketData(BaseModel):
    """Real-time market data."""
    symbol: str
    price: Decimal
    volume: Decimal
    timestamp: datetime

class OrderBook(BaseModel):
    """Market depth/order book data."""
    symbol: str
    bids: list[tuple[Decimal, Decimal]]
    asks: list[tuple[Decimal, Decimal]]
    timestamp: datetime

class Ticker(BaseModel):
    """Price ticker information."""
    symbol: str
    last_price: Decimal
    bid_price: Decimal
    ask_price: Decimal
    timestamp: datetime
```

## Configuration Management

### Main Configuration

**File:** `src/core/config/main.py`

```python
class Config:
    """Main configuration aggregator with backward compatibility."""
    
    def __init__(self, config_file: str | None = None, env_file: str | None = ".env")
    
    def load_from_file(self, config_file: str) -> None
    def save_to_file(self, config_file: str) -> None
    def validate(self) -> None
    def get_exchange_config(self, exchange: str) -> dict[str, Any]
    def switch_environment(self, environment: str, exchange: str = None) -> bool
    def is_production_mode(self, exchange: str = None) -> bool
    def get_current_environment_status(self) -> dict[str, Any]
```

#### Configuration Properties:
- `database: DatabaseConfig`
- `environment: EnvironmentConfig`
- `exchange: ExchangeConfig`
- `execution: ExecutionConfig`
- `strategy: StrategyConfig`
- `risk: RiskConfig`
- `security: SecurityConfig`

### Global Configuration Access

```python
def get_config(config_file: str | None = None, reload: bool = False) -> Config:
    """Get or create the global configuration instance."""
```

## Dependency Injection

**File:** `src/core/dependency_injection.py`

### DependencyContainer

```python
class DependencyContainer:
    """Lightweight dependency injection container."""
    
    def register(self, interface: type, implementation: type | Any, singleton: bool = False) -> None
    def register_factory(self, name: str, factory_func: Callable, singleton: bool = False) -> None
    def resolve(self, interface: type) -> Any
    def is_registered(self, interface: type) -> bool
    def clear(self) -> None
```

### Decorators

```python
def injectable(singleton: bool = False) -> Callable
    """Mark a class as injectable with optional singleton behavior."""

def inject(dependency_name: str) -> Callable
    """Inject a dependency into a method parameter."""
```

### Global Container Access

```python
def get_container() -> DependencyContainer:
    """Get the global dependency container."""
```

## Caching System

### CacheManager

**File:** `src/core/caching/cache_manager.py`

```python
class CacheManager(BaseComponent):
    """Advanced cache manager with distributed locking and warming strategies."""
    
    async def get(
        self, key: str, namespace: str = "cache",
        fallback: Callable | None = None
    ) -> Any
    
    async def set(
        self, key: str, value: Any, namespace: str = "cache",
        ttl: int | None = None
    ) -> bool
    
    async def delete(self, key: str, namespace: str = "cache") -> bool
    async def exists(self, key: str, namespace: str = "cache") -> bool
    async def get_many(self, keys: list[str], namespace: str = "cache") -> dict[str, Any]
    async def set_many(
        self, mapping: dict[str, Any], namespace: str = "cache"
    ) -> bool
```

#### Distributed Locking

```python
async def acquire_lock(
    self, resource: str, timeout: int | None = None
) -> str | None

async def release_lock(self, resource: str, lock_value: str) -> bool

async def with_lock(
    self, resource: str, func: Callable, *args, **kwargs
):
    """Execute function with distributed lock."""
```

#### Cache Warming

```python
async def warm_cache(
    self, warming_functions: dict[str, Callable],
    batch_size: int = 10
):
    """Warm cache with preloaded data."""
```

### Cache Utilities

```python
def get_cache_manager() -> CacheManager:
    """Get or create global cache manager instance."""
```

## Performance Monitoring

### Performance Monitor

**File:** `src/core/performance/performance_monitor.py` (referenced in glob)

Features:
- Real-time performance metrics collection
- Memory usage tracking
- CPU utilization monitoring
- I/O performance analysis
- Custom metric registration

### Trading Profiler

**File:** `src/core/performance/trading_profiler.py` (referenced in glob)

Features:
- Trading operation profiling
- Latency measurement
- Throughput analysis
- Performance bottleneck identification

## Resource Management

### ResourceManager

**File:** `src/core/resource_manager.py`

```python
class ResourceManager:
    """Comprehensive resource lifecycle manager."""
    
    def register_resource(
        self, resource: Any, resource_type: ResourceType,
        cleanup_callback: Callable | None = None
    ) -> str
    
    def unregister_resource(self, resource_id: str) -> bool
    def cleanup_all(self) -> None
    def get_resource_stats(self) -> dict[str, Any]
```

#### Resource Types

```python
class ResourceType(Enum):
    DATABASE_CONNECTION = "database_connection"
    WEBSOCKET_CONNECTION = "websocket_connection"
    HTTP_CLIENT = "http_client"
    FILE_HANDLE = "file_handle"
    THREAD_POOL = "thread_pool"
    CACHE_CONNECTION = "cache_connection"
```

### WebSocketManager

**File:** `src/core/websocket_manager.py`

```python
class WebSocketManager:
    """Async WebSocket connection manager with proper resource cleanup."""
    
    @asynccontextmanager
    async def connection(self) -> AsyncGenerator["WebSocketManager", None]:
        """Async context manager for WebSocket connection."""
    
    async def send_message(self, message: dict)
    def set_message_callback(self, callback: Callable[[dict], None])
    def set_error_callback(self, callback: Callable[[Exception], None])
    
    @property
    def is_connected(self) -> bool
    
    def get_stats(self) -> dict[str, Any]
```

## Integration Framework

### Environment-Aware Service

**File:** `src/core/integration/environment_aware_service.py` (referenced in glob)

Features:
- Environment-specific service configuration
- Dynamic environment switching
- Service isolation per environment
- Configuration validation

### Environment Orchestrator

**File:** `src/core/integration/environment_orchestrator.py` (referenced in glob)

Features:
- Multi-environment service coordination
- Environment state management
- Service dependency resolution
- Environment health monitoring

## Utilities

### Validation Registry

**File:** `src/core/validator_registry.py`

```python
class ValidatorRegistry:
    """Central registry for all validators."""
    
    def register_validator(self, name: str, validator: ValidatorInterface) -> None
    def register_rule(self, data_type: str, rule: Callable) -> None
    def validate(self, data_type: str, data: Any, **kwargs) -> bool
    def get_validator(self, name: str) -> ValidatorInterface
    def create_composite_validator(self, validator_names: list[str]) -> CompositeValidator
```

#### Built-in Validators

```python
class RangeValidator(ValidatorInterface):
    """Validator for numeric ranges."""

class LengthValidator(ValidatorInterface):
    """Validator for string/collection length."""

class PatternValidator(ValidatorInterface):
    """Validator for regex patterns."""

class TypeValidator(ValidatorInterface):
    """Validator for type checking."""
```

### Global Validator Functions

```python
def register_validator(name: str, validator: ValidatorInterface) -> None
def validate(data_type: str, data: Any, **kwargs) -> bool
def get_validator(name: str) -> ValidatorInterface
```

### Service Manager

**File:** `src/core/service_manager.py`

Centralized service lifecycle management with dependency resolution and health monitoring.

### Task Manager

**File:** `src/core/task_manager.py`

Async task management with proper cancellation, timeout handling, and resource cleanup.

### Memory Manager

**File:** `src/core/memory_manager.py`

Memory usage monitoring, leak detection, and automatic cleanup mechanisms.

## Exported Symbols

### From `src/core/__init__.py`

```python
__all__ = [
    # Base classes
    "BaseComponent", "BaseService",
    # Dependency injection
    "DependencyContainer", "injectable", "inject", "get_container",
    # Configuration
    "Config", "get_config",
    # Types (re-exported from types submodules)
    "TradingMode", "ExchangeType", "ValidationLevel",
    "OrderSide", "OrderType", "OrderStatus", "TimeInForce",
    "Signal", "Order", "Position", "Balance",
    # Exceptions
    "ValidationError", "ServiceError", "ComponentError",
    # Resource management
    "ResourceManager", "WebSocketManager",
    # Caching
    "CacheManager", "get_cache_manager",
    # Validation
    "ValidatorRegistry", "validate", "get_validator"
]
```

## Usage Examples

### Basic Component

```python
from src.core import BaseComponent

class MyService(BaseComponent):
    async def _do_start(self):
        # Custom startup logic
        await self.initialize_resources()
    
    async def _do_stop(self):
        # Custom cleanup logic
        await self.cleanup_resources()

# Usage
service = MyService(name="MyService")
async with service.lifecycle_context():
    # Service is automatically started and stopped
    await service.some_operation()
```

### Configuration Usage

```python
from src.core import get_config

# Get global configuration
config = get_config()

# Access exchange settings
exchange_config = config.get_exchange_config("binance")

# Switch environments
config.switch_environment("sandbox", "binance")
```

### Dependency Injection

```python
from src.core import injectable, get_container

@injectable(singleton=True)
class MyRepository:
    def __init__(self):
        pass

@injectable()
class MyService:
    def __init__(self, repo: MyRepository):
        self.repo = repo

# Register and resolve
container = get_container()
container.register(MyRepository, MyRepository, singleton=True)
container.register(MyService, MyService)

service = container.resolve(MyService)
```

### Caching

```python
from src.core import get_cache_manager

cache = get_cache_manager()

# Basic caching
await cache.set("user:123", user_data, ttl=300)
user = await cache.get("user:123")

# With fallback
user = await cache.get(
    "user:123", 
    fallback=lambda: fetch_user_from_db(123)
)

# Distributed locking
async with cache.with_lock("critical_operation"):
    # Critical section code
    await perform_critical_operation()
```

This reference provides comprehensive documentation for all major components and utilities in the core module. Each section includes class definitions, method signatures, and usage examples to help developers understand and effectively use the T-Bot core framework.