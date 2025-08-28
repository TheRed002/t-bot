# Coding Standards - T-Bot Trading System

## Overview
This document defines the coding standards used throughout the T-Bot Trading System codebase. These standards are derived from the existing implementation and should be followed to maintain consistency.

## 1. Module Structure

### 1.1 Module Documentation
Every module starts with a comprehensive docstring:
```python
"""
Module name and purpose.

Detailed description including:
- Key features
- Integration points
- Critical requirements
- Version information (if applicable)
"""
```

### 1.2 Import Organization
```python
# Standard library imports
import asyncio
import time
from datetime import datetime, timezone
from typing import Any, TypeVar

# Third-party imports
import aiohttp
from sqlalchemy import select

# Local application imports - Core first
from src.core.base import BaseComponent
from src.core.config import Config
from src.core.exceptions import ValidationError

# Local application imports - Module specific
from src.module.submodule import Component
```

## 2. Type Annotations

### 2.1 Always Use Type Hints
```python
def calculate_position_size(
    capital: Decimal, 
    risk_percentage: Decimal, 
    stop_loss: Decimal
) -> Decimal:
    """Calculate position size based on risk."""
    pass
```

### 2.2 Type Variables for Generics
```python
T = TypeVar("T")
class Repository(Generic[T]):
    async def get(self, id: Any) -> T | None:
        pass
```

### 2.3 Using Mapped for SQLAlchemy Models
```python
price: Mapped[Decimal | None] = Column(DECIMAL(20, 8))
quantity: Mapped[Decimal] = Column(DECIMAL(20, 8), nullable=False)
```

## 3. Service Layer Pattern

### 3.1 Base Service Implementation
All services inherit from `BaseService` or `TransactionalService`:
```python
class TradingService(BaseService):
    def __init__(self):
        super().__init__(name="TradingService")
        self.add_dependency("OrderRepository")
        self.add_dependency("RiskManager")
```

### 3.2 Service-to-Service Communication
Services communicate through dependency injection:
```python
async def _do_start(self):
    self.order_repo = self.resolve_dependency("OrderRepository")
    self.risk_manager = self.resolve_dependency("RiskManager")
```

### 3.3 No Direct Database Access in Services
Services use repositories, never direct database queries:
```python
# ❌ WRONG
result = await db.session.execute(select(Order))

# ✅ CORRECT
result = await self.order_repository.get_all(filters={"status": "OPEN"})
```

## 4. Repository Pattern

### 4.1 Repository Interface
All repositories implement the standard interface:
```python
class RepositoryInterface(ABC, Generic[T]):
    async def get(self, id: Any) -> T | None
    async def get_all(self, filters: dict | None = None) -> list[T]
    async def create(self, entity: T) -> T
    async def update(self, entity: T) -> T
    async def delete(self, id: Any) -> bool
```

## 5. Database Models

### 5.1 Financial Precision
Always use DECIMAL for financial values:
```python
# Crypto: 8 decimal places
price: Mapped[Decimal] = Column(DECIMAL(20, 8))

# Forex: 4 decimal places  
fx_rate: Mapped[Decimal] = Column(DECIMAL(20, 4))

# Stocks: 2 decimal places
stock_price: Mapped[Decimal] = Column(DECIMAL(20, 2))
```

### 5.2 Model Relationships
Always define both sides with back_populates:
```python
# Parent side
orders = relationship("Order", back_populates="bot", cascade="all, delete-orphan")

# Child side
bot = relationship("Bot", back_populates="orders")
```

### 5.3 Database Constraints
Add business rule constraints:
```python
__table_args__ = (
    CheckConstraint("quantity > 0", name="check_quantity_positive"),
    CheckConstraint("filled_quantity <= quantity", name="check_filled_max"),
    UniqueConstraint("exchange", "exchange_order_id", name="uq_exchange_order"),
)
```

## 6. Error Handling

### 6.1 Use Specific Exceptions
Import from `src.core.exceptions`:
```python
from src.core.exceptions import (
    ValidationError,
    ServiceError,
    ExchangeError,
)

# Use specific exceptions
if not order.is_valid():
    raise ValidationError("Invalid order parameters")
```

### 6.2 Error Decorators
Use decorators instead of repetitive try-catch:
```python
@with_retry(max_attempts=3, delay=1.0)
@with_circuit_breaker(failure_threshold=5)
async def place_order(self, order: OrderRequest) -> OrderResponse:
    pass
```

## 7. Async/Await Patterns

### 7.1 Always Await Async Operations
```python
# ❌ WRONG
result = self.async_method()

# ✅ CORRECT
result = await self.async_method()
```

### 7.2 Concurrent Operations
Use asyncio.gather for parallel operations:
```python
results = await asyncio.gather(
    self.fetch_ticker(symbol),
    self.fetch_order_book(symbol),
    self.fetch_trades(symbol),
    return_exceptions=True
)
```

## 8. Logging

### 8.1 Use Module Logger
```python
from src.core.logging import get_logger
logger = get_logger(__name__)
```

### 8.2 Structured Logging
```python
logger.info(
    "Order placed successfully",
    extra={
        "order_id": order.id,
        "symbol": order.symbol,
        "quantity": str(order.quantity),
        "price": str(order.price)
    }
)
```

## 9. Component Lifecycle

### 9.1 BaseComponent Pattern
All major components inherit from BaseComponent:
```python
class MyComponent(BaseComponent):
    async def _do_start(self) -> None:
        """Initialize resources."""
        pass
    
    async def _do_stop(self) -> None:
        """Cleanup resources."""
        pass
    
    async def health_check(self) -> HealthStatus:
        """Check component health."""
        pass
```

## 10. Configuration

### 10.1 Use Config Classes
```python
from src.core.config import Config

class TradingService:
    def __init__(self, config: Config):
        self.config = config
        self.max_position_size = config.risk.max_position_size
```

## 11. Testing

### 11.1 Test File Naming
```
src/module/component.py → tests/unit/test_module/test_component.py
```

### 11.2 Async Test Pattern
```python
import pytest

@pytest.mark.asyncio
async def test_async_operation():
    service = TradingService()
    result = await service.async_method()
    assert result is not None
```

## 12. Constants and Enums

### 12.1 Use Enums for Fixed Values
```python
from enum import Enum

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
```

## 13. Dependency Injection

### 13.1 Service Registration
```python
container = DependencyContainer()
container.register("TradingService", TradingService, singleton=True)
container.register_class("OrderRepository", OrderRepository, session)
```

### 13.2 Dependency Resolution
```python
trading_service = container.get("TradingService")
```

## 14. WebSocket Handling

### 14.1 Use Async Context Managers
```python
async with self.websocket_manager.connect(url) as ws:
    async for message in ws:
        await self.process_message(message)
```

## 15. Performance Considerations

### 15.1 Use Decimal for Calculations
```python
from decimal import Decimal

# ❌ WRONG
price = float(data['price'])

# ✅ CORRECT
price = Decimal(data['price'])
```

### 15.2 Batch Database Operations
```python
# Instead of multiple single inserts
async with self.session.begin():
    self.session.add_all(entities)
```

## 16. Security

### 16.1 Never Log Sensitive Data
```python
# ❌ WRONG
logger.info(f"API key: {api_key}")

# ✅ CORRECT
logger.info(f"API key: {'*' * 8}{api_key[-4:]}")
```

### 16.2 Validate All External Input
```python
if not self.validator.validate_order(order_data):
    raise ValidationError("Invalid order data")
```

## Summary
These standards reflect the actual patterns used in the codebase. Following them ensures consistency and maintainability across the entire trading system.