# Common Code Patterns and Standards

## Overview
This document defines mandatory code patterns that ALL prompts must follow to ensure consistency and avoid duplication. These patterns must be referenced and used throughout the implementation.

## 1. Error Handling Patterns

### Standard Exception Usage
```python
# MANDATORY: All prompts must use these exact patterns
from src.core.exceptions import ExchangeError, RiskManagementError, ValidationError

# Correct pattern - use existing exceptions
try:
    result = await exchange.place_order(order)
except ExchangeError as e:
    logger.error("Exchange API failed", error=str(e), order_id=order.id)
    raise

# WRONG: Don't create new exception types
# class BinanceError(Exception):  # DON'T DO THIS
```

### Retry Pattern with Exponential Backoff
```python
# MANDATORY: Use this exact pattern for retries
import asyncio
from typing import TypeVar, Callable, Any
from src.utils.decorators import retry

T = TypeVar('T')

@retry(max_attempts=3, backoff_base=2, exceptions=(ExchangeError,))
async def api_call_with_retry(func: Callable[..., T], *args, **kwargs) -> T:
    """Standard retry pattern for API calls"""
    return await func(*args, **kwargs)
```

## 2. Configuration Loading Patterns

### Standard Configuration Class
```python
# MANDATORY: All config classes must follow this pattern
from pydantic import BaseSettings, Field
from typing import Optional

class ComponentConfig(BaseSettings):
    """Component-specific configuration"""
    
    # Environment variable with default
    api_key: str = Field(..., env="COMPONENT_API_KEY")
    timeout: int = Field(30, env="COMPONENT_TIMEOUT")
    max_retries: int = Field(3, env="COMPONENT_MAX_RETRIES")
    
    # Optional settings
    debug_mode: bool = Field(False, env="DEBUG_MODE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
```

### Configuration Usage Pattern
```python
# MANDATORY: Load config this way in all components
from src.core.config import Config

class ComponentClass:
    def __init__(self):
        self.config = Config()
        self.component_config = self.config.component  # Access sub-config
```

## 3. Async Context Manager Pattern

### Standard Resource Management
```python
# MANDATORY: Use this pattern for all resource management
from contextlib import asynccontextmanager
from typing import AsyncGenerator

class ResourceManager:
    @asynccontextmanager
    async def managed_resource(self) -> AsyncGenerator[ResourceType, None]:
        """Standard resource management pattern"""
        resource = None
        try:
            # Acquire resource
            resource = await self.acquire_resource()
            yield resource
        except Exception as e:
            logger.error("Resource operation failed", error=str(e))
            raise
        finally:
            # Always cleanup
            if resource:
                await self.release_resource(resource)
```

## 4. Logging Patterns

### Structured Logging
```python
# MANDATORY: Use structured logging everywhere
from src.core.logging import get_logger

# Correct logging pattern
logger = get_logger(__name__)

async def process_order(order: Order) -> OrderResult:
    """Example of proper logging"""
    logger.info(
        "Processing order",
        order_id=order.id,
        symbol=order.symbol,
        quantity=float(order.quantity),
        side=order.side
    )
    
    try:
        result = await self.execute_order(order)
        logger.info(
            "Order processed successfully",
            order_id=order.id,
            result_id=result.id,
            execution_time_ms=result.execution_time
        )
        return result
    except Exception as e:
        logger.error(
            "Order processing failed",
            order_id=order.id,
            error=str(e),
            error_type=type(e).__name__
        )
        raise
```

### Performance Logging
```python
# MANDATORY: Use this decorator for performance monitoring
from src.utils.decorators import time_execution

@time_execution
async def expensive_operation(data: Any) -> Result:
    """Operations that need performance monitoring"""
    # Implementation here
    pass
```

## 5. Database Patterns

### Standard Model Usage
```python
# MANDATORY: Import existing models, don't create new ones
from src.database.models import Trade, Position, BotInstance
from src.database.connection import get_session

async def create_trade_record(trade_data: TradeData) -> Trade:
    """Standard database operation pattern"""
    async with get_session() as session:
        try:
            trade = Trade(
                bot_id=trade_data.bot_id,
                symbol=trade_data.symbol,
                side=trade_data.side,
                quantity=trade_data.quantity,
                price=trade_data.price
            )
            session.add(trade)
            await session.commit()
            await session.refresh(trade)
            return trade
        except Exception as e:
            await session.rollback()
            logger.error("Database operation failed", error=str(e))
            raise
```

### Query Pattern
```python
# MANDATORY: Use this pattern for database queries
from sqlalchemy import select
from typing import List, Optional

async def get_active_positions(bot_id: str) -> List[Position]:
    """Standard query pattern"""
    async with get_session() as session:
        stmt = select(Position).where(
            Position.bot_id == bot_id,
            Position.is_active == True
        )
        result = await session.execute(stmt)
        return result.scalars().all()
```

## 6. Validation Patterns

### Input Validation
```python
# MANDATORY: Use these validation patterns
from src.utils.validators import validate_price, validate_quantity, validate_symbol
from src.core.exceptions import ValidationError

def validate_order_request(order: OrderRequest) -> None:
    """Standard validation pattern"""
    try:
        validate_symbol(order.symbol)
        validate_price(order.price, order.symbol)
        validate_quantity(order.quantity, order.symbol)
    except ValidationError as e:
        logger.error("Order validation failed", order=order.dict(), error=str(e))
        raise
```

### Type Validation
```python
# MANDATORY: Use type checking patterns
from typing import Union, Any
from src.core.types import Signal, MarketData

def process_signal(signal: Union[Signal, dict]) -> Signal:
    """Convert and validate signal data"""
    if isinstance(signal, dict):
        try:
            signal = Signal(**signal)
        except Exception as e:
            raise ValidationError(f"Invalid signal format: {e}")
    
    if not isinstance(signal, Signal):
        raise ValidationError(f"Expected Signal, got {type(signal)}")
    
    return signal
```

## 7. API Response Patterns

### Standard API Response
```python
# MANDATORY: Use this response pattern for all APIs
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from src.utils.formatters import format_api_response
from typing import Any, Dict

async def api_endpoint_handler(request_data: Any) -> JSONResponse:
    """Standard API response pattern"""
    try:
        # Process request
        result = await process_request(request_data)
        
        # Format response
        response_data = format_api_response(
            success=True,
            data=result,
            message="Operation completed successfully"
        )
        
        return JSONResponse(content=response_data, status_code=200)
        
    except ValidationError as e:
        return JSONResponse(
            content=format_api_response(
                success=False,
                error=str(e),
                error_code="VALIDATION_ERROR"
            ),
            status_code=400
        )
    except Exception as e:
        logger.error("API endpoint failed", error=str(e))
        return JSONResponse(
            content=format_api_response(
                success=False,
                error="Internal server error",
                error_code="INTERNAL_ERROR"
            ),
            status_code=500
        )
```

## 8. Strategy Base Implementation Pattern

### Standard Strategy Structure
```python
# MANDATORY: All strategies must follow this pattern
from abc import ABC, abstractmethod
from src.strategies.base import BaseStrategy
from src.core.types import Signal, MarketData, Position
from src.utils.decorators import time_execution, retry
from typing import List, Optional

class ConcreteStrategy(BaseStrategy):
    """Standard strategy implementation pattern"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "strategy_name"
        self.version = "1.0.0"
    
    @time_execution
    @retry(max_attempts=2)
    async def generate_signals(self, data: MarketData) -> List[Signal]:
        """MANDATORY: Implement signal generation logic"""
        try:
            # Validation
            if not data or not data.price:
                return []
            
            # Signal generation logic
            signals = await self._analyze_market_data(data)
            
            # Validate signals
            validated_signals = []
            for signal in signals:
                if await self.validate_signal(signal):
                    validated_signals.append(signal)
            
            return validated_signals
            
        except Exception as e:
            logger.error("Signal generation failed", strategy=self.name, error=str(e))
            return []  # Fail gracefully
    
    async def validate_signal(self, signal: Signal) -> bool:
        """MANDATORY: Implement signal validation"""
        # Standard validation logic
        return (
            signal.confidence > self.config.min_confidence and
            signal.direction in ['buy', 'sell'] and
            signal.timestamp is not None
        )
```

## 9. Exchange Integration Pattern

### Standard Exchange Method
```python
# MANDATORY: All exchange methods must follow this pattern
from src.exchanges.base import BaseExchange
from src.core.types import OrderRequest, OrderResponse
from src.utils.decorators import time_execution, circuit_breaker

class ConcreteExchange(BaseExchange):
    """Standard exchange implementation pattern"""
    
    @time_execution
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """MANDATORY: Standard order placement pattern"""
        try:
            # Pre-validation
            await self._validate_order_request(order)
            
            # Rate limiting check
            await self._check_rate_limits()
            
            # API call with retry
            response = await self._make_api_call('POST', '/order', order.dict())
            
            # Response validation
            order_response = self._parse_order_response(response)
            
            # Logging
            logger.info(
                "Order placed successfully",
                order_id=order_response.id,
                symbol=order.symbol,
                exchange=self.name
            )
            
            return order_response
            
        except Exception as e:
            logger.error(
                "Order placement failed",
                symbol=order.symbol,
                error=str(e),
                exchange=self.name
            )
            raise ExchangeError(f"Failed to place order: {e}")
```

## 10. Required Imports Pattern

### Standard Import Structure
```python
# MANDATORY: Use this import structure in all files

# Standard library imports
import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union

# Third-party imports  
import structlog
from pydantic import BaseModel, Field

# Core application imports (ALWAYS FIRST)
from src.core.config import Config
from src.core.types import Signal, MarketData, Position, OrderRequest, OrderResponse
from src.core.exceptions import (
    TradingBotError, ExchangeError, RiskManagementError, 
    ValidationError, ExecutionError
)

# Component-specific imports
from src.utils.decorators import time_execution, retry, circuit_breaker
from src.utils.validators import validate_price, validate_quantity, validate_symbol
from src.utils.formatters import format_currency, format_percentage

# Module-specific imports
# (Add specific imports based on component)
```

## 11. Testing Patterns

### Standard Test Structure
```python
# MANDATORY: Use this test structure
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.component.module import ComponentClass
from src.core.types import Signal, MarketData

class TestComponentClass:
    """Standard test class pattern"""
    
    @pytest.fixture
    def component(self):
        """Standard component fixture"""
        config = {"test_param": "test_value"}
        return ComponentClass(config)
    
    @pytest.fixture
    def sample_market_data(self):
        """Standard market data fixture"""
        return MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000.00"),
            volume=Decimal("1.5"),
            timestamp=datetime.now(timezone.utc)
        )
    
    @pytest.mark.asyncio
    async def test_component_method(self, component, sample_market_data):
        """Standard async test pattern"""
        # Arrange
        expected_result = "expected_value"
        
        # Act
        result = await component.method_under_test(sample_market_data)
        
        # Assert
        assert result == expected_result
        assert isinstance(result, ExpectedType)
```

## 12. Documentation Patterns

### Standard Docstring Format
```python
# MANDATORY: Use Google-style docstrings everywhere
def complex_function(
    param1: str,
    param2: Optional[int] = None,
    param3: Dict[str, Any] = None
) -> Tuple[bool, str]:
    """Brief description of what the function does.
    
    Longer description explaining the function's purpose, algorithm,
    or important implementation details.
    
    Args:
        param1: Description of param1 and its expected format.
        param2: Optional parameter description. Defaults to None.
        param3: Dictionary parameter description. Defaults to None.
    
    Returns:
        Tuple containing:
            - bool: Success status
            - str: Result message or error description
    
    Raises:
        ValidationError: When param1 is invalid format.
        ConfigurationError: When required configuration is missing.
    
    Example:
        >>> success, message = complex_function("test", 42, {"key": "value"})
        >>> print(f"Success: {success}, Message: {message}")
        Success: True, Message: Operation completed
    """
    # Implementation here
    pass
```

## Integration Requirements

### Every Prompt Must:
1. **Import from existing modules** - never recreate types, exceptions, or configs
2. **Use decorators from utils** - apply @time_execution, @retry, @circuit_breaker
3. **Follow logging patterns** - structured logging with context
4. **Use validation patterns** - input validation with proper error handling
5. **Implement error handling** - use existing exception hierarchy
6. **Follow database patterns** - use existing models and session management
7. **Apply testing patterns** - comprehensive test coverage with fixtures

### Reverse Integration Checklist:
- [ ] Check if new types should be added to P-001 types.py
- [ ] Check if new exceptions should be added to P-001 exceptions.py  
- [ ] Check if new config classes should be added to P-001 config.py
- [ ] Check if database models need updates in P-002
- [ ] Check if base interfaces need updates (BaseStrategy, BaseExchange)
- [ ] Check if API endpoints need additions in P-027
- [ ] Check if WebSocket streams need additions in P-028 