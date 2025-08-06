# Coding Standards

## Overview
This document outlines the coding standards and best practices for the trading bot project, ensuring consistency, maintainability, and quality across all code contributions.

## Python Code Standards

### **Code Style**
- **PEP 8 Compliance**: Follow PEP 8 style guide for Python code
- **Line Length**: Maximum 88 characters per line (Black formatter default)
- **Indentation**: 4 spaces (no tabs)
- **File Encoding**: UTF-8 for all Python files
- **Line Endings**: Unix-style (LF) for all files

### **Naming Conventions**
```python
# Variables and functions: snake_case
user_name = "john_doe"
def calculate_position_size():
    pass

# Classes: PascalCase
class RiskManager:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_POSITION_SIZE = 0.02
DEFAULT_TIMEOUT = 30

# Private methods: leading underscore
def _internal_helper():
    pass

# Protected methods: leading underscore
def _protected_method():
    pass
```

### **Import Organization**
```python
# Standard library imports
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Third-party imports
import pandas as pd
import numpy as np
from pydantic import BaseModel

# Local imports
from src.utils.logger import get_logger
from src.risk_management.base import RiskManager
```

### **Type Hints**
```python
from typing import Dict, List, Optional, Union, Tuple

def calculate_risk(
    position_size: float,
    volatility: float,
    confidence_level: Optional[float] = 0.95
) -> Dict[str, float]:
    """Calculate risk metrics for a position."""
    pass

class TradingStrategy:
    def __init__(self, name: str, parameters: Dict[str, Union[int, float, str]]):
        self.name = name
        self.parameters = parameters
    
    def execute(self, market_data: pd.DataFrame) -> Tuple[bool, float]:
        """Execute trading strategy and return signal and confidence."""
        pass
```

### **Documentation**
```python
def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.
    
    Args:
        returns: Series of portfolio returns
        risk_free_rate: Annual risk-free rate (default: 0.02)
        periods_per_year: Number of periods per year (default: 252)
    
    Returns:
        float: Sharpe ratio
    
    Raises:
        ValueError: If returns is empty or contains invalid data
    
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03])
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe Ratio: {sharpe:.2f}")
    """
    if returns.empty:
        raise ValueError("Returns series cannot be empty")
    
    excess_returns = returns - risk_free_rate / periods_per_year
    return excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
```

## Project-Specific Standards

### **Error Handling**
```python
from src.core.logging import get_logger
from typing import Optional
from src.error_handling.error_manager import ErrorManager

logger = get_logger(__name__)

class ExchangeManager:
    def __init__(self):
        self.error_manager = ErrorManager()
    
    def place_order(self, order_data: Dict) -> Optional[Dict]:
        """
        Place order with comprehensive error handling.
        
        Returns:
            Optional[Dict]: Order response or None if failed
        """
        try:
            # TODO: Remove in production - debug logging
            logger.debug(f"Placing order: {order_data}")
            
            response = self._execute_order(order_data)
            return response
            
        except ConnectionError as e:
            logger.error(f"Connection error placing order: {e}")
            self.error_manager.handle_connection_error(e)
            return None
            
        except ValueError as e:
            logger.error(f"Invalid order data: {e}")
            self.error_manager.handle_validation_error(e)
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error placing order: {e}")
            self.error_manager.handle_unexpected_error(e)
            return None
```

### **Configuration Management**
```python
from pydantic import BaseSettings, Field
from typing import Dict, Optional

class DatabaseConfig(BaseSettings):
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    
    class Config:
        env_prefix = "DB_"

class TradingConfig(BaseSettings):
    max_position_size: float = Field(default=0.02, description="Maximum position size")
    default_stop_loss: float = Field(default=0.02, description="Default stop loss")
    risk_free_rate: float = Field(default=0.02, description="Risk-free rate")
    
    class Config:
        env_prefix = "TRADING_"
```

### **Logging Standards**
```python
from src.core.logging import get_logger
from typing import Any, Dict

logger = get_logger()

class StrategyExecutor:
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.logger = logger.bind(strategy=strategy_name)
    
    def execute(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Execute trading strategy with structured logging."""
        self.logger.info(
            "Executing strategy",
            data_points=len(market_data),
            strategy_type=self.strategy_name
        )
        
        try:
            result = self._calculate_signals(market_data)
            
            self.logger.info(
                "Strategy execution completed",
                signals_generated=len(result.get("signals", [])),
                confidence=result.get("confidence", 0.0)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Strategy execution failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
```

### **Testing Standards**
```python
import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

class TestRiskManager:
    """Test suite for RiskManager class."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create RiskManager instance for testing."""
        return RiskManager(
            max_position_size=0.02,
            default_stop_loss=0.02
        )
    
    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for testing."""
        return pd.DataFrame({
            'price': [100, 101, 99, 102],
            'volume': [1000, 1100, 900, 1200],
            'timestamp': pd.date_range('2024-01-01', periods=4, freq='H')
        })
    
    def test_calculate_position_size_valid_input(self, risk_manager, sample_market_data):
        """Test position size calculation with valid input."""
        # Arrange
        account_balance = 10000
        volatility = 0.02
        
        # Act
        position_size = risk_manager.calculate_position_size(
            account_balance, volatility
        )
        
        # Assert
        assert 0 < position_size <= risk_manager.max_position_size
        assert isinstance(position_size, float)
    
    def test_calculate_position_size_invalid_balance(self, risk_manager):
        """Test position size calculation with invalid balance."""
        # Arrange
        invalid_balance = -1000
        
        # Act & Assert
        with pytest.raises(ValueError, match="Account balance must be positive"):
            risk_manager.calculate_position_size(invalid_balance, 0.02)
    
    @patch('src.risk_management.risk_manager.ExchangeAPI')
    def test_risk_manager_with_mock_exchange(self, mock_exchange, risk_manager):
        """Test RiskManager with mocked exchange API."""
        # Arrange
        mock_exchange.return_value.get_balance.return_value = 10000
        
        # Act
        balance = risk_manager.get_account_balance()
        
        # Assert
        assert balance == 10000
        mock_exchange.return_value.get_balance.assert_called_once()
```

## Database Standards

### **SQL Standards**
```sql
-- Use descriptive table and column names
CREATE TABLE trading_positions (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    position_size DECIMAL(10, 4) NOT NULL,
    entry_price DECIMAL(15, 8) NOT NULL,
    entry_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    stop_loss DECIMAL(15, 8),
    take_profit DECIMAL(15, 8),
    status VARCHAR(20) DEFAULT 'open',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Use indexes for performance
CREATE INDEX idx_positions_strategy ON trading_positions(strategy_name);
CREATE INDEX idx_positions_symbol ON trading_positions(symbol);
CREATE INDEX idx_positions_status ON trading_positions(status);

-- Use constraints for data integrity
ALTER TABLE trading_positions 
ADD CONSTRAINT chk_position_size_positive 
CHECK (position_size > 0);

ALTER TABLE trading_positions 
ADD CONSTRAINT chk_status_valid 
CHECK (status IN ('open', 'closed', 'cancelled'));
```

### **ORM Standards**
```python
from sqlalchemy import Column, Integer, String, Numeric, DateTime, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class TradingPosition(Base):
    """Trading position model."""
    
    __tablename__ = 'trading_positions'
    
    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(100), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    position_size = Column(Numeric(10, 4), nullable=False)
    entry_price = Column(Numeric(15, 8), nullable=False)
    entry_timestamp = Column(DateTime(timezone=True), nullable=False)
    stop_loss = Column(Numeric(15, 8))
    take_profit = Column(Numeric(15, 8))
    status = Column(String(20), default='open', index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        CheckConstraint('position_size > 0', name='chk_position_size_positive'),
        CheckConstraint("status IN ('open', 'closed', 'cancelled')", name='chk_status_valid'),
    )
```

## API Standards

### **REST API Design**
```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Trading Bot API", version="1.0.0")

class OrderRequest(BaseModel):
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str = 'market'
    price: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "BTCUSDT",
                "side": "buy",
                "quantity": 0.001,
                "order_type": "limit",
                "price": 50000.0
            }
        }

@app.post("/api/v1/orders", response_model=Dict[str, Any])
async def create_order(
    order: OrderRequest,
    current_user: str = Depends(get_current_user)
):
    """
    Create a new trading order.
    
    Args:
        order: Order request data
        current_user: Authenticated user
    
    Returns:
        Dict containing order details and status
    
    Raises:
        HTTPException: If order creation fails
    """
    try:
        # TODO: Remove in production - debug logging
        logger.debug(f"Creating order for user {current_user}: {order}")
        
        result = await order_service.create_order(order, current_user)
        
        return {
            "order_id": result.order_id,
            "status": result.status,
            "message": "Order created successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    except InsufficientFundsError as e:
        raise HTTPException(status_code=400, detail="Insufficient funds")
        
    except Exception as e:
        logger.error(f"Order creation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

## Configuration Standards

### **YAML Configuration**
```yaml
# config.yaml
environment: "development"

database:
  postgresql:
    host: "localhost"
    port: 5432
    database: "trading_bot"
    username: "${DB_USERNAME}"
    password: "${DB_PASSWORD}"
  
  redis:
    host: "localhost"
    port: 6379
    password: "${REDIS_PASSWORD}"

trading:
  risk_management:
    max_position_size: 0.02
    default_stop_loss: 0.02
    max_daily_loss: 0.05
  
  strategies:
    mean_reversion:
      enabled: true
      lookback_period: 20
      threshold: 2.0
    
    trend_following:
      enabled: true
      short_period: 10
      long_period: 30

monitoring:
  logging:
    level: "INFO"
    format: "json"
  
  alerts:
    email:
      enabled: true
      smtp_server: "smtp.gmail.com"
    discord:
      enabled: true
      webhook_url: "${DISCORD_WEBHOOK}"
```

## Documentation Standards

### **README Files**
```markdown
# Module Name

Brief description of the module's purpose and functionality.

## Features

- Feature 1: Description
- Feature 2: Description
- Feature 3: Description

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.module_name import ClassName

# Example usage
instance = ClassName()
result = instance.method_name()
```

## Configuration

Describe configuration options and provide examples.

## Testing

```bash
pytest tests/module_name/
```

## Contributing

Guidelines for contributing to this module.
```

### **API Documentation**
```python
"""
Trading Bot API Documentation

This module provides the main API endpoints for the trading bot system.

Endpoints:
- POST /api/v1/orders: Create new trading orders
- GET /api/v1/positions: Get current positions
- PUT /api/v1/positions/{id}: Update position
- DELETE /api/v1/positions/{id}: Close position

Authentication:
All endpoints require Bearer token authentication.

Rate Limiting:
- 100 requests per minute per user
- 1000 requests per hour per user

Error Codes:
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error
"""
```

## Security Standards

### **Input Validation**
```python
from pydantic import BaseModel, validator, Field
from typing import Optional

class OrderRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20, regex=r'^[A-Z0-9]+$')
    side: str = Field(..., regex=r'^(buy|sell)$')
    quantity: float = Field(..., gt=0, le=1000)
    order_type: str = Field(default='market', regex=r'^(market|limit)$')
    price: Optional[float] = Field(None, gt=0)
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate trading symbol format."""
        if not v.isalnum():
            raise ValueError('Symbol must contain only alphanumeric characters')
        return v.upper()
    
    @validator('quantity')
    def validate_quantity(cls, v):
        """Validate order quantity."""
        if v <= 0:
            raise ValueError('Quantity must be positive')
        if v > 1000:
            raise ValueError('Quantity exceeds maximum allowed')
        return round(v, 8)  # Round to 8 decimal places
```

### **Authentication and Authorization**
```python
from functools import wraps
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def require_permission(permission: str):
    """Decorator to require specific permission."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check user permissions
            if not has_permission(get_current_user(), permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission '{permission}' required"
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator

@require_permission("trading:create_order")
async def create_order(order: OrderRequest):
    """Create order with permission check."""
    pass
```

## Performance Standards

### **Database Optimization**
```python
# Use connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True
)

# Use bulk operations for large datasets
def bulk_insert_positions(positions: List[Dict]):
    """Bulk insert positions for better performance."""
    with engine.begin() as conn:
        conn.execute(
            TradingPosition.__table__.insert(),
            positions
        )

# Use appropriate indexes
CREATE INDEX CONCURRENTLY idx_positions_symbol_timestamp 
ON trading_positions(symbol, entry_timestamp);
```

### **Caching Standards**
```python
import redis
from functools import wraps
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expire_time: int = 300):
    """Cache decorator for function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            redis_client.setex(
                cache_key,
                expire_time,
                json.dumps(result, default=str)
            )
            
            return result
        return wrapper
    return decorator

@cache_result(expire_time=60)
def get_market_data(symbol: str, timeframe: str):
    """Get market data with caching."""
    pass
```

## Testing Standards

### **Test Organization**
```python
# tests/unit/test_risk_manager.py
import pytest
from unittest.mock import Mock, patch
from src.risk_management.risk_manager import RiskManager

class TestRiskManager:
    """Test suite for RiskManager class."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.risk_manager = RiskManager()
        self.sample_data = pd.DataFrame({
            'price': [100, 101, 99, 102],
            'volume': [1000, 1100, 900, 1200]
        })
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        # Arrange
        account_balance = 10000
        volatility = 0.02
        
        # Act
        position_size = self.risk_manager.calculate_position_size(
            account_balance, volatility
        )
        
        # Assert
        assert 0 < position_size <= self.risk_manager.max_position_size
        assert isinstance(position_size, float)
    
    @pytest.mark.parametrize("balance,expected_error", [
        (-1000, ValueError),
        (0, ValueError),
        ("invalid", TypeError)
    ])
    def test_calculate_position_size_invalid_input(self, balance, expected_error):
        """Test position size calculation with invalid input."""
        with pytest.raises(expected_error):
            self.risk_manager.calculate_position_size(balance, 0.02)
```

### **Integration Tests**
```python
# tests/integration/test_trading_workflow.py
import pytest
from src.trading_workflow import TradingWorkflow

class TestTradingWorkflow:
    """Integration tests for trading workflow."""
    
    @pytest.mark.integration
    def test_complete_trading_cycle(self):
        """Test complete trading cycle from signal to execution."""
        # Arrange
        workflow = TradingWorkflow()
        market_data = self.get_test_market_data()
        
        # Act
        signals = workflow.generate_signals(market_data)
        orders = workflow.create_orders(signals)
        executions = workflow.execute_orders(orders)
        
        # Assert
        assert len(signals) > 0
        assert len(orders) == len(signals)
        assert all(execution.status == 'filled' for execution in executions)
```

## Code Review Standards

### **Review Checklist**
- [ ] Code follows PEP 8 style guidelines
- [ ] Type hints are used for all functions and methods
- [ ] Comprehensive docstrings are provided
- [ ] Error handling is implemented appropriately
- [ ] Tests are written and passing
- [ ] No TODO comments without issue references
- [ ] Security considerations are addressed
- [ ] Performance implications are considered
- [ ] Documentation is updated if needed

### **Review Comments**
```python
# Good review comment
"""
Consider adding input validation for the volatility parameter.
A negative volatility could cause issues in the calculation.
"""

# Bad review comment
"""
This is wrong.
"""
```

## Deployment Standards

### **Environment Configuration**
```python
# config/environments/production.py
import os
from typing import Dict, Any

def get_production_config() -> Dict[str, Any]:
    """Get production configuration."""
    return {
        'environment': 'production',
        'debug': False,
        'database': {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME'),
            'username': os.getenv('DB_USERNAME'),
            'password': os.getenv('DB_PASSWORD'),
        },
        'logging': {
            'level': 'WARNING',
            'format': 'json'
        }
    }
```

### **Docker Standards**
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements/production.txt .
RUN pip install --no-cache-dir -r production.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Monitoring and Logging Standards

### **Structured Logging**
```python
from src.core.logging import get_logger
from typing import Any, Dict

logger = get_logger()

def log_trading_event(
    event_type: str,
    symbol: str,
    quantity: float,
    price: float,
    user_id: str,
    **kwargs
):
    """Log trading events with structured data."""
    logger.info(
        "Trading event occurred",
        event_type=event_type,
        symbol=symbol,
        quantity=quantity,
        price=price,
        user_id=user_id,
        timestamp=datetime.utcnow().isoformat(),
        **kwargs
    )
```

### **Metrics Collection**
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
orders_created = Counter('orders_created_total', 'Total orders created', ['symbol', 'side'])
order_processing_time = Histogram('order_processing_seconds', 'Order processing time')
active_positions = Gauge('active_positions', 'Number of active positions', ['strategy'])

def track_order_creation(symbol: str, side: str):
    """Track order creation metrics."""
    orders_created.labels(symbol=symbol, side=side).inc()

@order_processing_time.time()
def process_order(order_data: Dict):
    """Process order with timing metrics."""
    pass
```

## Summary

These coding standards ensure:
- **Consistency** across the codebase
- **Maintainability** of the code
- **Quality** and reliability
- **Security** best practices
- **Performance** optimization
- **Testability** of all components
- **Documentation** completeness
- **Deployment** readiness

All developers must follow these standards to maintain code quality and project integrity.
