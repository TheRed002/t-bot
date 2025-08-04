# Trading Bot Suite - LLM Implementation Prompts

## Overview
This document contains 36 sequential prompts to build a comprehensive algorithmic trading platform. Each prompt builds upon previous work and references supporting documents to minimize context usage for smaller LLM models.

## Supporting Documents
- **@CODING_STANDARDS.md**: Code quality, style, and patterns
- **@DEPENDENCIES.md**: Technical stack and version requirements  
- **@INTEGRATION_POINTS.md**: Component interaction patterns
- **@COMMON_PATTERNS.md**: Reusable code patterns
- **@SPECIFICATIONS.md**: Complete project requirements (reference specific sections)

## Critical Integration Rules
1. **NEVER DUPLICATE**: Always import existing types, exceptions, and configs - never recreate them
2. **MANDATORY PATTERNS**: Follow ALL patterns from @COMMON_PATTERNS.md exactly
3. **EXACT VERSIONS**: Use only dependency versions from @DEPENDENCIES.md
4. **REVERSE INTEGRATION**: Update previous prompts' work when creating shared components

---

## **Prompt P-001: Core Framework Foundation**

**Title:** Implement core framework with types, configuration, logging, and custom exceptions

### Context
- **Current State:** Empty project structure
- **Target State:** Foundational framework with type definitions, configuration management, structured logging, and exception hierarchy
- **Phase Goal:** Establish the core infrastructure that all other components will depend on

**Technical Context**: Windows 11 WSL Ubuntu, Python 3.11+, virtual environment `.venv`. Follow @CODING_STANDARDS.md patterns.

### Dependencies
**Depends On:** Project initialization
**Enables:** All subsequent components (P-002 through P-036)

### Task Details

Create the core framework components in the following order:

#### 1. Type Definitions (`src/core/types.py`)
Define core data structures used throughout the system:

**CRITICAL**: This file will be updated by subsequent prompts. Use exact types from @COMMON_PATTERNS.md.

**Core Trading Types:**
```python
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field

class TradingMode(Enum):
    LIVE = "live"
    PAPER = "paper" 
    BACKTEST = "backtest"

class SignalDirection(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class Signal(BaseModel):
    direction: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    symbol: str
    strategy_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MarketData(BaseModel):
    symbol: str
    price: Decimal
    volume: Decimal
    timestamp: datetime
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    open_price: Optional[Decimal] = None
    high_price: Optional[Decimal] = None
    low_price: Optional[Decimal] = None

class OrderRequest(BaseModel):
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "GTC"
    client_order_id: Optional[str] = None

class OrderResponse(BaseModel):
    id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal]
    filled_quantity: Decimal = Decimal("0")
    status: str
    timestamp: datetime

class Position(BaseModel):
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    side: OrderSide
    timestamp: datetime
```

**REVERSE INTEGRATION POINTS**: Future prompts will add:
- P-003: Exchange-specific types (ExchangeInfo, Ticker, OrderBook)
- P-008: Risk types (RiskMetrics, PositionLimits)  
- P-011: Strategy types (StrategyConfig, StrategyStatus)
- P-017: ML types (ModelPrediction, ModelMetadata)

#### 2. Configuration Management (`src/core/config.py`)
Implement comprehensive Pydantic-based configuration system:

**CRITICAL**: This file will be extended by ALL subsequent prompts. Use exact patterns from @COMMON_PATTERNS.md.

**Base Configuration Structure:**
```python
from pydantic import BaseSettings, Field, validator
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

class BaseConfig(BaseSettings):
    """Base configuration class with common patterns"""
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        validate_assignment = True

class DatabaseConfig(BaseConfig):
    # PostgreSQL
    postgresql_host: str = Field(..., env="DB_HOST")
    postgresql_port: int = Field(5432, env="DB_PORT")
    postgresql_database: str = Field(..., env="DB_NAME")
    postgresql_username: str = Field(..., env="DB_USERNAME")
    postgresql_password: str = Field(..., env="DB_PASSWORD")
    postgresql_pool_size: int = Field(10, env="DB_POOL_SIZE")
    
    # Redis
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    redis_db: int = Field(0, env="REDIS_DB")
    
    # InfluxDB
    influxdb_host: str = Field("localhost", env="INFLUXDB_HOST")
    influxdb_port: int = Field(8086, env="INFLUXDB_PORT")
    influxdb_token: str = Field(..., env="INFLUXDB_TOKEN")
    influxdb_org: str = Field(..., env="INFLUXDB_ORG")
    influxdb_bucket: str = Field(..., env="INFLUXDB_BUCKET")

class SecurityConfig(BaseConfig):
    secret_key: str = Field(..., env="SECRET_KEY")
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(30, env="JWT_EXPIRE_MINUTES")
    encryption_key: str = Field(..., env="ENCRYPTION_KEY")
    
class Config(BaseConfig):
    """Master configuration class"""
    
    # Environment
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    
    # Application
    app_name: str = Field("trading-bot-suite", env="APP_NAME")
    app_version: str = Field("2.0.0", env="APP_VERSION")
    
    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    security: SecurityConfig = SecurityConfig()
    
    @validator('environment')
    def validate_environment(cls, v):
        allowed = ['development', 'staging', 'production']
        if v not in allowed:
            raise ValueError(f'Environment must be one of: {allowed}')
        return v
    
    def generate_schema(self) -> None:
        """Generate JSON schema for configuration validation"""
        schema = self.schema()
        schema_path = Path("config/config.schema.json")
        schema_path.parent.mkdir(exist_ok=True)
        with open(schema_path, 'w') as f:
            json.dump(schema, f, indent=2)
```

**REVERSE INTEGRATION POINTS**: Future prompts will add these config classes:
- P-002: Extended DatabaseConfig with connection pooling settings
- P-003: ExchangeConfig for API credentials and rate limits
- P-008: RiskConfig for risk management parameters
- P-010A: CapitalConfig for fund management settings
- P-011: StrategyConfig for strategy parameters
- P-017: MLConfig for model registry and training
- P-026: WebConfig for FastAPI and security settings

#### 3. Exception Hierarchy (`src/core/exceptions.py`)
Create comprehensive exception hierarchy that ALL components must use:

**CRITICAL**: Never create duplicate exceptions. All prompts must import and use these exact classes.

**Complete Exception Hierarchy:**
```python
class TradingBotError(Exception):
    """Base exception for all trading bot errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)

# Exchange-related exceptions
class ExchangeError(TradingBotError):
    """Exchange API and connection errors"""
    pass

class ExchangeConnectionError(ExchangeError):
    """Exchange connection failures"""
    pass

class ExchangeRateLimitError(ExchangeError):
    """Exchange rate limit violations"""
    pass

class ExchangeInsufficientFundsError(ExchangeError):
    """Insufficient funds for order"""
    pass

# Risk management exceptions
class RiskManagementError(TradingBotError):
    """Risk management violations"""
    pass

class PositionLimitError(RiskManagementError):
    """Position size limit violations"""
    pass

class DrawdownLimitError(RiskManagementError):
    """Drawdown limit violations"""
    pass

# Data-related exceptions
class DataError(TradingBotError):
    """Data quality and processing errors"""
    pass

class DataValidationError(DataError):
    """Data validation failures"""
    pass

class DataSourceError(DataError):
    """External data source failures"""
    pass

# ML model exceptions
class ModelError(TradingBotError):
    """Machine learning model errors"""
    pass

class ModelLoadError(ModelError):
    """Model loading failures"""
    pass

class ModelInferenceError(ModelError):
    """Model inference failures"""
    pass

class ModelDriftError(ModelError):
    """Model drift detection"""
    pass

# Validation exceptions
class ValidationError(TradingBotError):
    """Input and schema validation errors"""
    pass

class ConfigurationError(ValidationError):
    """Configuration validation errors"""
    pass

# Execution exceptions
class ExecutionError(TradingBotError):
    """Order execution errors"""
    pass

class OrderRejectionError(ExecutionError):
    """Order rejection by exchange"""
    pass

class SlippageError(ExecutionError):
    """Excessive slippage detected"""
    pass

# State consistency exceptions
class StateConsistencyError(TradingBotError):
    """State synchronization problems"""
    pass

class StateCorruptionError(StateConsistencyError):
    """State data corruption detected"""
    pass

# Security exceptions
class SecurityError(TradingBotError):
    """Security and authentication errors"""
    pass

class AuthenticationError(SecurityError):
    """Authentication failures"""
    pass

class AuthorizationError(SecurityError):
    """Authorization failures"""
    pass
```

**USAGE RULE**: ALL prompts must import these exceptions:
```python
from src.core.exceptions import (
    TradingBotError, ExchangeError, RiskManagementError,
    ValidationError, ExecutionError, ModelError, DataError,
    StateConsistencyError, SecurityError
)
```

**REVERSE INTEGRATION**: Future prompts may add specific sub-exceptions but must extend these base classes, never replace them.

#### 4. Structured Logging (`src/core/logging.py`)
Set up structured logging with correlation tracking:
- Configure `structlog` with JSON formatting for production
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Context managers for operation correlation IDs
- Performance logging decorators for latency tracking
- Secure logging (no sensitive data in logs)
- Log rotation and retention policies

#### 5. Application Entry Point (`src/main.py`)
Create main application entry point:
- Configuration loading with validation
- Database connection initialization
- Component registry setup
- Graceful shutdown handling
- Health check endpoint setup
- Error handling and logging configuration

### Directory Structure to Create
```
src/
├── __init__.py
├── main.py
└── core/
    ├── __init__.py
    ├── types.py
    ├── config.py
    ├── exceptions.py
    └── logging.py
```

### Acceptance Criteria

#### Code Quality
- All files created with proper imports following @COMMON_PATTERNS.md
- Type hints with `mypy --strict` compliance
- Docstrings following Google style
- Error handling following @CODING_STANDARDS.md patterns
- TODO comments for debug functionality marked "TODO: Remove in production"

#### Functionality  
- Configuration loads from environment variables and YAML files
- Exception hierarchy provides specific error types for all failure modes
- Logging outputs structured JSON with correlation IDs
- Type definitions support financial precision with Decimal types
- All components properly validated with Pydantic

#### Performance
- Configuration loading <100ms
- Logger initialization <50ms
- Type validation overhead <1ms per operation

### Integration Points
- Database connection string from `DatabaseConfig`
- Exchange API credentials from `ExchangeConfig`  
- Risk parameters from `TradingConfig`
- Model registry settings from `MLConfig`

### Security Considerations
- Environment variables for sensitive data (API keys, passwords)
- Credential encryption at rest
- Audit logging for all configuration changes
- Input sanitization in configuration validation

### Reverse Integration Required
**CRITICAL**: Future prompts MUST update P-001 files when adding shared components:

#### Update src/core/types.py when implementing:
- **P-003 (Exchanges)**: Add `ExchangeInfo`, `Ticker`, `OrderBook` types
- **P-008 (Risk)**: Add `RiskMetrics`, `PositionLimits`, `RiskLevel` types  
- **P-011 (Strategies)**: Add `StrategyConfig`, `StrategyStatus`, `StrategyMetrics` types
- **P-017 (ML)**: Add `ModelPrediction`, `ModelMetadata`, `ModelStatus` types
- **P-020 (Execution)**: Add `TradeResult`, `ExecutionReport` types

#### Update src/core/config.py when implementing:
- **P-003 (Exchanges)**: Add `ExchangeConfig` class with API keys and rate limits
- **P-008 (Risk)**: Add `RiskConfig` class with position limits and drawdown settings
- **P-010A (Capital)**: Add `CapitalConfig` class with allocation and fund management
- **P-011 (Strategies)**: Add `StrategyConfig` class with strategy parameters
- **P-017 (ML)**: Add `MLConfig` class with model registry and training settings
- **P-026 (Web)**: Add `WebConfig` class with FastAPI and security settings

#### Update src/core/exceptions.py when implementing:
- **P-002A (Error Handling)**: Add specific error recovery types
- **P-010A (Capital)**: Add `CapitalManagementError` and sub-types
- **P-013A (Arbitrage)**: Add `ArbitrageError` sub-type if needed
- **P-013B (Market Making)**: Add `MarketMakingError` sub-type if needed

### Integration Validation
After P-001 completion, verify:
```bash
# Test configuration loading
python -c "from src.core.config import Config; c = Config(); print('Config loaded successfully')"

# Test type imports  
python -c "from src.core.types import Signal, MarketData, Position; print('Types imported successfully')"

# Test exception hierarchy
python -c "from src.core.exceptions import ExchangeError, ValidationError; print('Exceptions imported successfully')"
```

---

## **Prompt P-002: Database Models and Connection Management**

**Title:** Implement SQLAlchemy models, database connections, and migration system

### Context
- **Current State:** Core framework established (P-001)
- **Target State:** Complete database layer with models, connections, and migrations
- **Phase Goal:** Establish persistent storage foundation for all trading data

**Technical Context**: PostgreSQL 17+, SQLAlchemy 2.0+ with async support, Alembic for migrations. Reference @SPECIFICATIONS.md Section 12 "Database Schema".

### Dependencies
**Depends On:** P-001 (core framework, configuration, exceptions)
**Enables:** P-003 (exchange integration), P-008+ (all components requiring persistence)

### Task Details

#### 1. Database Models (`src/database/models.py`)
Implement SQLAlchemy models matching @SPECIFICATIONS.md Section 12.1:

**Core Tables:**
- `User`: id, username, email, password_hash, is_active, timestamps
- `BotInstance`: id, name, user_id, strategy_type, exchange, status, config, timestamps  
- `Trade`: id, bot_id, exchange, symbol, side, order_type, quantity, price, executed_price, fee, status, pnl, timestamps
- `Position`: id, bot_id, exchange, symbol, side, quantity, entry_price, current_price, unrealized_pnl, stop_loss_price, timestamps
- `BalanceSnapshot`: id, user_id, exchange, currency, free_balance, locked_balance, total_balance, btc_value, usd_value, timestamp
- `StrategyConfig`: id, name, strategy_type, parameters, risk_parameters, is_active, version, timestamps
- `MLModel`: id, name, model_type, version, file_path, metrics, parameters, training_data_range, is_active, timestamps
- `PerformanceMetrics`: id, bot_id, metric_date, trade_counts, pnl_values, ratios, timestamps
- `Alert`: id, user_id, bot_id, alert_type, severity, title, message, metadata, is_read, timestamp
- `AuditLog`: id, user_id, action, resource_type, resource_id, old_value, new_value, ip_address, user_agent, timestamp

**Required Features:**
- UUID primary keys using `gen_random_uuid()`
- Proper foreign key relationships with cascade options
- JSON columns for flexible configuration storage
- Indexes for performance optimization
- Check constraints for data integrity
- Timestamps with timezone support

#### 2. Database Connection (`src/database/connection.py`)
Implement async database connection management:
- SQLAlchemy async engine with connection pooling
- Session factory with proper context management
- Connection health monitoring and automatic reconnection
- Transaction management with rollback on errors
- Connection pool sizing (min: 5, max: 20 connections)
- Query performance monitoring and slow query logging

#### 3. Migration System (`src/database/migrations/`)
Set up Alembic migration system:
- Initialize Alembic configuration
- Create initial migration with all tables
- Migration versioning and dependency tracking
- Rollback capabilities for each migration
- Data migration utilities for schema changes
- Environment-specific migration configurations

#### 4. Database Utilities (`src/database/queries.py`)
Implement common database operations:
- Generic CRUD operations with type safety
- Bulk insert/update operations for performance
- Query builders for complex filtering
- Pagination utilities for large datasets
- Data export/import functions
- Database health check functions

#### 5. Redis Integration (`src/database/redis_client.py`)
Implement Redis client for real-time state:
- Async Redis client with connection pooling
- Key namespacing for different data types
- Serialization/deserialization utilities
- TTL management for cached data
- Pub/sub support for real-time updates
- Redis health monitoring and failover

#### 6. InfluxDB Integration (`src/database/influxdb_client.py`)
Implement InfluxDB client for time series data:
- InfluxDB v2 client with async support
- Time series data writing with batch optimization
- Measurement schema for market data, trades, and metrics
- Data retention policies and downsampling
- Query interface for historical analysis
- Performance metrics storage and retrieval

### Directory Structure to Create
```
src/database/
├── __init__.py
├── models.py
├── connection.py
├── redis_client.py
├── influxdb_client.py
├── queries.py
└── migrations/
    ├── env.py
    ├── script.py.mako
    └── versions/
        └── 001_initial_schema.py
```

### Configuration Schema
Add to existing config classes from P-001:
```yaml
database:
  postgresql:
    host: "${DB_HOST:localhost}"
    port: "${DB_PORT:5432}"
    database: "${DB_NAME:trading_bot}"
    username: "${DB_USERNAME}"
    password: "${DB_PASSWORD}"
    pool_size: 10
    max_overflow: 20
    
  redis:
    host: "${REDIS_HOST:localhost}"
    port: "${REDIS_PORT:6379}"
    password: "${REDIS_PASSWORD}"
    db: 0
    max_connections: 100
  
  influxdb:
    host: "${INFLUXDB_HOST:localhost}"
    port: "${INFLUXDB_PORT:8086}"
    token: "${INFLUXDB_TOKEN}"
    org: "${INFLUXDB_ORG:trading-org}"
    bucket: "${INFLUXDB_BUCKET:trading-metrics}"
    batch_size: 1000
    flush_interval: 10  # seconds
```

### Acceptance Criteria

#### Code Quality
- All models inherit from declarative base with proper type hints
- Foreign key relationships properly defined with cascade rules
- Async patterns following @COMMON_PATTERNS.md
- Exception handling using custom exceptions from P-001
- Connection management with proper resource cleanup

#### Functionality
- All tables created with proper constraints and indexes
- Migrations run successfully forward and backward
- Connection pooling handles concurrent access
- Redis client supports all required operations
- CRUD operations work with all model types

#### Performance
- Database connection establishment <500ms
- CRUD operations <10ms for simple queries
- Bulk operations handle 1000+ records efficiently
- Connection pool prevents resource exhaustion
- Query performance monitored and logged

### Integration Points
- Configuration loading from P-001 `DatabaseConfig`
- Exception handling using P-001 custom exceptions
- Logging integration with structured logging from P-001
- Health checks exposed to main application

---

## **Prompt P-002A: Comprehensive Error Handling Framework**

**Title:** Implement comprehensive error handling, recovery, and resilience framework

### Context
- **Current State:** Database layer and core exceptions established (P-001, P-002)
- **Target State:** Robust error handling with automatic recovery capabilities
- **Phase Goal:** Production-ready error resilience before exchange integrations

**Technical Context**: Reference @SPECIFICATIONS.md Section 11 "Error Handling & Recovery". Implement specific recovery scenarios.

### Dependencies
**Depends On:** P-001 (core exceptions), P-002 (database for error logging)
**Enables:** P-003+ (exchanges with resilient error handling), all subsequent components

### Task Details

#### 1. Error Handler Framework (`src/error_handling/error_handler.py`)
Implement comprehensive error handling system:
- Error categorization and severity classification (Critical, High, Medium, Low)
- Retry policies with exponential backoff for transient errors
- Circuit breaker integration to prevent cascading failures
- Error context preservation for debugging and recovery
- Error aggregation and pattern detection
- Automatic escalation for repeated failures

#### 2. Recovery Scenarios (`src/error_handling/recovery_scenarios.py`)
Implement specific recovery procedures:
- **Partial Order Fill Recovery**: Handle incomplete order executions
- **Network Disconnection Recovery**: Automatic reconnection with state sync
- **Exchange Maintenance Recovery**: Graceful handling of scheduled downtime
- **Data Feed Interruption Recovery**: Fallback data sources and conservative mode
- **Order Rejection Recovery**: Intelligent retry with parameter adjustment
- **API Rate Limit Recovery**: Automatic throttling and request queuing

#### 3. Connection Resilience Manager (`src/error_handling/connection_manager.py`)
Implement connection reliability:
- Automatic reconnection with exponential backoff
- Connection pooling with health monitoring
- Heartbeat detection for connection validation
- Message queuing during brief disconnections
- Connection state synchronization across components
- Failover to backup connections when available

#### 4. State Consistency Monitor (`src/error_handling/state_monitor.py`)
Implement state integrity validation:
- Cross-system state consistency checking
- Automatic state reconciliation procedures
- State corruption detection and recovery
- Transaction rollback capabilities
- State audit trail for forensic analysis
- Real-time state validation alerts

#### 5. Error Pattern Analytics (`src/error_handling/pattern_analytics.py`)
Implement error pattern detection:
- Error frequency analysis and trending
- Root cause analysis automation
- Predictive error detection based on patterns
- Error correlation analysis across components
- Performance impact assessment
- Automated error reporting and escalation

### Directory Structure to Create
```
src/error_handling/
├── __init__.py
├── error_handler.py
├── recovery_scenarios.py
├── connection_manager.py
├── state_monitor.py
└── pattern_analytics.py
```

### Error Handling Configuration
```yaml
error_handling:
  # Error severity levels
  severity_levels:
    critical: 
      description: "System failure, data corruption, security breach"
      auto_escalate: true
      notify_channels: ["sms", "discord", "email"]
    high:
      description: "Trading halted, model failure, risk limit breach"
      auto_escalate: false
      notify_channels: ["discord", "email"]
    medium:
      description: "Performance degradation, data quality issues"
      auto_escalate: false
      notify_channels: ["discord"]
    low:
      description: "Configuration warnings, minor validation errors"
      auto_escalate: false
      notify_channels: ["logs"]
  
  # Retry policies
  retry_policies:
    network_errors:
      max_attempts: 5
      backoff_strategy: "exponential"
      base_delay: 1  # seconds
      max_delay: 60  # seconds
      jitter: true
    
    api_rate_limits:
      max_attempts: 3
      backoff_strategy: "linear"
      base_delay: 5  # seconds
      respect_retry_after: true
    
    database_errors:
      max_attempts: 3
      backoff_strategy: "exponential"
      base_delay: 0.5  # seconds
      max_delay: 10  # seconds
  
  # Circuit breaker settings
  circuit_breakers:
    api_calls:
      failure_threshold: 5  # failures before opening
      recovery_timeout: 30  # seconds before half-open
      success_threshold: 3  # successes to close
    
    database_connections:
      failure_threshold: 3
      recovery_timeout: 15
      success_threshold: 2
  
  # Recovery scenarios
  recovery_scenarios:
    partial_order_fill:
      min_fill_percentage: 0.5  # 50% minimum to accept
      cancel_remainder: true
      log_details: true
    
    network_disconnection:
      max_offline_duration: 300  # 5 minutes
      sync_on_reconnect: true
      conservative_mode: true
    
    exchange_maintenance:
      detect_maintenance: true
      redistribute_capital: true
      pause_new_orders: true
    
    data_feed_interruption:
      max_staleness: 30  # seconds
      fallback_sources: ["backup_feed", "static_data"]
      conservative_trading: true
    
    order_rejection:
      analyze_rejection_reason: true
      adjust_parameters: true
      max_retry_attempts: 2
  
  # State monitoring
  state_monitoring:
    validation_frequency: 60  # seconds
    consistency_checks:
      - "portfolio_balance_sync"
      - "position_quantity_sync"  
      - "order_status_sync"
      - "risk_limit_compliance"
    
    reconciliation:
      auto_reconcile: true
      max_discrepancy: 0.01  # 1% tolerance
      force_sync_threshold: 0.05  # 5% force sync
  
  # Error analytics
  error_analytics:
    pattern_detection: true
    correlation_analysis: true
    predictive_alerts: true
    
    reporting:
      daily_summary: true
      weekly_analysis: true
      trend_alerts: true
      
    retention:
      error_logs: 90  # days
      pattern_data: 30  # days
      analytics_reports: 365  # days
```

### Specific Error Recovery Implementations

#### Partial Order Fill Recovery
```python
async def handle_partial_fill(self, order: Order, filled_quantity: float):
    """Handle partially filled orders with intelligent recovery"""
    fill_percentage = filled_quantity / order.quantity
    
    if fill_percentage < self.config.min_fill_percentage:
        # Cancel remainder and re-evaluate signal
        await self.cancel_order(order.id)
        await self.log_partial_fill(order, filled_quantity)
        return await self.reevaluate_signal(order.signal)
    else:
        # Accept partial fill and adjust position tracking
        await self.update_position(order, filled_quantity)
        await self.adjust_stop_loss(order, filled_quantity)
```

#### Network Disconnection Recovery
```python
async def handle_network_disconnection(self):
    """Comprehensive network disconnection recovery"""
    # 1. Switch to offline mode
    self.mode = TradingMode.OFFLINE
    
    # 2. Persist pending operations
    await self.persist_pending_operations()
    
    # 3. Attempt reconnection with exponential backoff
    for attempt in range(self.max_reconnect_attempts):
        if await self.try_reconnect():
            # 4. Reconcile state with exchange
            await self.reconcile_positions()
            await self.reconcile_orders()
            await self.verify_balances()
            self.mode = TradingMode.ONLINE
            return
        await asyncio.sleep(2 ** attempt)
    
    # 5. Enter safe mode if reconnection fails
    await self.enter_safe_mode()
```

### Acceptance Criteria
- Error recovery scenarios handle all specified failure modes
- Circuit breakers prevent cascading system failures
- State consistency maintained across all components
- Error patterns detected and analyzed automatically
- Recovery procedures complete within specified timeframes

### Integration Points
- Exception hierarchy from P-001 for consistent error handling
- Database logging via P-002 for error persistence
- State validation with all subsequent components
- Monitoring integration with P-030+ (monitoring system)
- Alert integration with P-031+ (notification system)

### Reverse Integration Required
- **Update P-003+ (Exchanges)**: Integrate error recovery scenarios for API failures
- **Update P-008+ (Risk Management)**: Add error handling for risk calculation failures
- **Update P-011+ (Strategies)**: Implement error handling decorators from utils module
- **Update P-020 (Execution)**: Integrate partial fill recovery and network disconnection handling

---

## **Prompt P-003: Base Exchange Interface and Factory**

**Title:** Implement unified exchange interface and factory pattern for multi-exchange support

### Context
- **Current State:** Core framework (P-001) and database layer (P-002) established
- **Target State:** Unified interface for multiple exchanges with factory instantiation
- **Phase Goal:** Abstract exchange differences to enable consistent trading logic across all supported exchanges

**Technical Context**: Reference @SPECIFICATIONS.md Section 1 "Exchange Integration Module". Support for Binance, OKX, Coinbase Pro with future extensibility.

### Dependencies
**Depends On:** P-001 (types, exceptions, config), P-002 (database models for trade logging), P-002A (error handling)
**Enables:** P-004 (Binance), P-005 (OKX), P-006 (Coinbase), P-007 (rate limiting)

### Mandatory Integration Requirements
**CRITICAL**: This prompt MUST integrate with existing P-001 components:

#### Required Imports from P-001:
```python
# MANDATORY: Import existing types, don't create new ones
from src.core.types import (
    OrderRequest, OrderResponse, MarketData, Position,
    Signal, TradingMode, OrderSide, OrderType
)
from src.core.exceptions import (
    ExchangeError, ExchangeConnectionError, ExchangeRateLimitError,
    ValidationError, ExecutionError
)
from src.core.config import Config
```

#### Required Patterns from @COMMON_PATTERNS.md:
- Use standard exception handling patterns
- Apply @time_execution decorators to all API methods
- Use structured logging with context
- Follow async context manager patterns

### Task Details

#### 1. Base Exchange Interface (`src/exchanges/base.py`)
Create abstract base class defining unified exchange operations:

**Required Abstract Methods:**
- `async def get_account_balance() -> Dict[str, Decimal]`: Get all asset balances
- `async def place_order(order: OrderRequest) -> OrderResponse`: Execute trade order
- `async def cancel_order(order_id: str) -> bool`: Cancel existing order
- `async def get_order_status(order_id: str) -> OrderStatus`: Check order status
- `async def get_market_data(symbol: str, timeframe: str) -> MarketData`: Get OHLCV data
- `async def subscribe_to_stream(symbol: str, callback: Callable) -> None`: Real-time data stream
- `async def get_order_book(symbol: str, depth: int = 10) -> OrderBook`: Get market depth
- `async def get_trade_history(symbol: str, limit: int = 100) -> List[Trade]`: Historical trades

**Base Implementation:**
- Exchange configuration validation
- Rate limiting interface integration
- Connection health monitoring
- Error handling and retry logic
- Logging with structured data
- Metrics collection for monitoring

#### 2. Exchange Factory (`src/exchanges/factory.py`)
Implement factory pattern for exchange instantiation:
- Registry of supported exchanges
- Dynamic exchange creation from configuration
- Validation of exchange credentials and connectivity
- Exchange capability detection (spot, futures, margin)
- Connection pooling across exchanges
- Hot-swapping of exchange instances

#### 3. Common Exchange Types (`src/exchanges/types.py`)
Define exchange-specific data structures:
- `OrderRequest`: symbol, side, order_type, quantity, price, time_in_force
- `OrderResponse`: order_id, status, filled_quantity, average_price, fees
- `OrderBook`: bids, asks, timestamp, symbol
- `Ticker`: symbol, bid, ask, last_price, volume, timestamp
- `Trade`: id, symbol, side, quantity, price, timestamp, fee
- `ExchangeInfo`: exchange_name, supported_symbols, rate_limits, features

#### 4. Rate Limiting Framework (`src/exchanges/rate_limiter.py`)
Implement token bucket rate limiting:
- Exchange-specific rate limit configurations
- Request queuing and throttling
- Burst capacity handling
- Rate limit monitoring and alerting
- Graceful degradation when limits approached
- Automatic retry with exponential backoff

#### 5. Connection Manager (`src/exchanges/connection_manager.py`)
Manage exchange connections and WebSocket streams:
- Connection pooling for REST APIs
- WebSocket connection management
- Automatic reconnection on failures
- Heartbeat monitoring for connection health
- Message queuing during disconnections
- Connection state synchronization

### Directory Structure to Create
```
src/exchanges/
├── __init__.py
├── base.py
├── factory.py
├── types.py
├── rate_limiter.py
└── connection_manager.py
```

### Exchange Configuration Schema
Add to config from P-001:
```yaml
exchanges:
  supported: ["binance", "okx", "coinbase"]
  default_timeout: 30
  max_retries: 3
  
  rate_limits:
    binance:
      requests_per_minute: 1200
      orders_per_second: 10
      websocket_connections: 5
    okx:
      requests_per_minute: 600
      orders_per_second: 20
      websocket_connections: 3
    coinbase:
      requests_per_minute: 600
      orders_per_second: 15
      websocket_connections: 4
```

### Acceptance Criteria

#### Code Quality
- Abstract base class properly defines interface contract
- Factory pattern enables dynamic exchange selection
- Type hints and protocols for exchange capabilities
- Error handling using exceptions from P-001
- Async patterns throughout with proper resource management

#### Functionality
- Base exchange class validates common operations
- Factory creates exchanges from configuration
- Rate limiter prevents API violations
- Connection manager handles failures gracefully
- All abstract methods have proper signatures and documentation

#### Performance
- Exchange instantiation <1s including connectivity check
- Rate limiter adds <1ms overhead per request
- Connection manager maintains <100ms average latency
- WebSocket connections handle 1000+ messages/second
- Memory usage <100MB per exchange connection

### Integration Points
- Configuration from P-001 `ExchangeConfig`
- Database logging to P-002 models (`Trade`, `BalanceSnapshot`)
- Exception hierarchy from P-001 for error handling
- Metrics export for monitoring system (P-030+)

### Security Considerations
- API credentials encrypted in configuration
- Request signing validation before transmission
- Rate limit compliance to prevent account suspension
- Connection security with TLS verification
- Audit logging of all exchange interactions

### Reverse Integration Required
**CRITICAL**: This prompt MUST update P-001 files with exchange-specific components:

#### Update P-001 src/core/types.py:
Add these exchange-specific types to the existing file:
```python
# Add these to src/core/types.py after existing types
from typing import List

class ExchangeInfo(BaseModel):
    name: str
    supported_symbols: List[str]
    rate_limits: Dict[str, int]
    features: List[str]
    api_version: str

class Ticker(BaseModel):
    symbol: str
    bid: Decimal
    ask: Decimal
    last_price: Decimal
    volume_24h: Decimal
    price_change_24h: Decimal
    timestamp: datetime

class OrderBook(BaseModel):
    symbol: str
    bids: List[List[Decimal]]  # [[price, quantity], ...]
    asks: List[List[Decimal]]  # [[price, quantity], ...]
    timestamp: datetime

class ExchangeStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
```

#### Update P-001 src/core/config.py:
Add ExchangeConfig class to the existing Config class:
```python
# Add this to src/core/config.py
class ExchangeConfig(BaseConfig):
    # Default settings
    default_timeout: int = Field(30, env="EXCHANGE_TIMEOUT")
    max_retries: int = Field(3, env="EXCHANGE_MAX_RETRIES")
    
    # Binance
    binance_api_key: str = Field(..., env="BINANCE_API_KEY")
    binance_api_secret: str = Field(..., env="BINANCE_API_SECRET")
    binance_testnet: bool = Field(True, env="BINANCE_TESTNET")
    
    # OKX  
    okx_api_key: str = Field(..., env="OKX_API_KEY")
    okx_api_secret: str = Field(..., env="OKX_API_SECRET")
    okx_passphrase: str = Field(..., env="OKX_PASSPHRASE")
    okx_sandbox: bool = Field(True, env="OKX_SANDBOX")
    
    # Coinbase
    coinbase_api_key: str = Field(..., env="COINBASE_API_KEY")
    coinbase_api_secret: str = Field(..., env="COINBASE_API_SECRET")
    coinbase_sandbox: bool = Field(True, env="COINBASE_SANDBOX")

# Update the main Config class to include exchanges:
class Config(BaseConfig):
    # ... existing fields ...
    exchanges: ExchangeConfig = ExchangeConfig()
```

### Integration Validation
After P-003 completion, verify:
```bash
# Test exchange config loading
python -c "from src.core.config import Config; c = Config(); print(f'Exchanges configured: {len(c.exchanges.__dict__)}')"

# Test exchange types
python -c "from src.core.types import ExchangeInfo, Ticker, OrderBook; print('Exchange types imported successfully')"

# Test base exchange
python -c "from src.exchanges.base import BaseExchange; print('BaseExchange imported successfully')"
```

---

## **Prompt P-004: Binance Exchange Implementation**

**Title:** Implement Binance-specific exchange client with full API integration

### Context
- **Current State:** Base exchange interface established (P-003)
- **Target State:** Complete Binance integration supporting spot trading, WebSocket streams, and rate limiting
- **Phase Goal:** First concrete exchange implementation demonstrating the unified interface

**Technical Context**: Binance API v3, CCXT 4.1.64+, python-binance 1.0.19+. Reference @SPECIFICATIONS.md Section 1.4 "Exchange-Specific Rate Limits".

### Dependencies
**Depends On:** P-003 (base exchange, factory), P-001 (config, exceptions), P-002 (trade logging)
**Enables:** P-007 (rate limiting), P-011+ (strategy implementations)

### Task Details

#### 1. Binance Exchange Class (`src/exchanges/binance.py`)
Implement `BinanceExchange` inheriting from `BaseExchange`:

**API Integration:**
- REST API client using `python-binance` with async support
- WebSocket stream manager for real-time data
- Order placement with all supported order types (market, limit, stop)
- Balance retrieval with asset conversion to USD equivalent
- Historical data fetching with pagination support
- Symbol information and trading rules validation

**Rate Limiting Implementation:**
- Weight-based rate limiting (1200 requests/minute)
- Order rate limiting (50 orders/10 seconds, 160k/24 hours)
- WebSocket connection limits (5 messages/second, 300 connections/5 minutes)
- Intelligent request prioritization (critical orders first)
- Rate limit monitoring with proactive throttling

**Error Handling:**
- Binance-specific error code mapping
- Network error recovery with exponential backoff
- Invalid symbol/parameter validation
- Insufficient balance detection and reporting
- API key permission validation

#### 2. Binance WebSocket Handler (`src/exchanges/binance_websocket.py`)
Implement real-time data streaming:
- Ticker price streams for all configured symbols
- Order book depth streams with configurable levels
- Trade execution stream for portfolio tracking
- User data stream for account balance updates
- Connection heartbeat and automatic reconnection
- Message queuing during brief disconnections

#### 3. Binance Order Management (`src/exchanges/binance_orders.py`)
Specialized order handling for Binance:
- Order type conversion (market, limit, stop-loss, OCO)
- Time-in-force parameter handling (GTC, IOC, FOK)
- Order status tracking and fill monitoring
- Partial fill handling and notification
- Order cancellation with confirmation
- Fee calculation and reporting

### Directory Structure to Create
```
src/exchanges/
├── binance.py
├── binance_websocket.py
└── binance_orders.py
```

### Binance Configuration
```yaml
exchanges:
  binance:
    api_key: "${BINANCE_API_KEY}"
    api_secret: "${BINANCE_API_SECRET}"
    testnet: true
    base_url: "https://testnet.binance.vision"
    ws_url: "wss://testnet-dex.binance.org/ws"
    timeout: 30
    max_retries: 3
    rate_limits:
      weight_per_minute: 1200
      orders_per_10_seconds: 50
      orders_per_24_hours: 160000
```

### Acceptance Criteria
- All `BaseExchange` methods implemented and tested
- Rate limiting prevents API violations
- WebSocket streams provide real-time data with <100ms latency
- Order placement success rate >99% under normal conditions
- Error recovery handles network issues gracefully

---

## **Prompt P-005: OKX Exchange Implementation**

**Title:** Implement OKX exchange client with unified API interface

### Context
- **Current State:** Binance implementation complete (P-004)
- **Target State:** OKX exchange fully integrated with the unified interface
- **Phase Goal:** Demonstrate interface consistency across different exchange APIs

**Technical Context**: OKX API v5, different rate limiting structure from Binance. Reference @SPECIFICATIONS.md Section 1.4.

### Dependencies
**Depends On:** P-003 (base exchange), P-004 (reference implementation)
**Enables:** P-006 (Coinbase), P-011+ (multi-exchange strategies)

### Task Details

#### 1. OKX Exchange Class (`src/exchanges/okx.py`)
Implement `OKXExchange` following Binance patterns:

**API Differences from Binance:**
- Different authentication (API key + secret + passphrase)
- Unified account model (trading/funding accounts)
- Different order types and parameters
- Alternative rate limiting structure (60 requests/2 seconds per endpoint)
- Different WebSocket authentication requirements

**Specific Implementations:**
- Account balance aggregation across trading/funding accounts
- Order placement with OKX-specific parameters
- Position size calculations for spot trading
- Symbol mapping and normalization
- Fee structure handling (maker/taker differences)

### Directory Structure to Create
```
src/exchanges/
├── okx.py
├── okx_websocket.py
└── okx_orders.py
```

---

## **Prompt P-006: Coinbase Pro Exchange Implementation**

**Title:** Implement Coinbase Pro exchange client completing multi-exchange support

### Context
- **Current State:** Binance (P-004) and OKX (P-005) implementations complete
- **Target State:** Three major exchanges supported with unified interface
- **Phase Goal:** Complete initial multi-exchange capability

**Technical Context**: Coinbase Pro API, point-based rate limiting system. Reference @SPECIFICATIONS.md Section 1.4.

### Dependencies
**Depends On:** P-003 (base), P-004 (Binance), P-005 (OKX)
**Enables:** P-007 (advanced rate limiting), P-008+ (risk management)

### Task Details

#### 1. Coinbase Exchange Class (`src/exchanges/coinbase.py`)
Implement `CoinbaseExchange` with Coinbase-specific features:

**Coinbase Specifics:**
- Point-based rate limiting (8000 points/minute)
- Different order lifecycle management
- USD-centric trading pairs
- Unique fee structure
- Different market data format

### Directory Structure to Create
```
src/exchanges/
├── coinbase.py
├── coinbase_websocket.py
└── coinbase_orders.py
```

---

## **Prompt P-007: Advanced Rate Limiting and Connection Management**

**Title:** Implement sophisticated rate limiting with multi-exchange coordination

### Context
- **Current State:** All three exchanges implemented (P-004, P-005, P-006)
- **Target State:** Advanced rate limiting preventing violations across all exchanges
- **Phase Goal:** Robust connection management for production reliability

**Technical Context**: Coordinate rate limits across Binance (weight-based), OKX (endpoint-based), Coinbase (point-based).

### Dependencies
**Depends On:** P-004, P-005, P-006 (all exchange implementations)
**Enables:** P-008+ (risk management can assume reliable exchange connectivity)

---

## **Prompt P-008: Risk Management Framework Foundation**

**Title:** Implement base risk management system with position sizing and portfolio limits

### Context
- **Current State:** All exchanges integrated with reliable connections (P-007)
- **Target State:** Core risk management protecting against excessive losses
- **Phase Goal:** Essential safety mechanisms before strategy implementation

**Technical Context**: Reference @SPECIFICATIONS.md Section 2 "Risk Management Module". Support static, dynamic, and AI-powered risk modes.

### Dependencies
**Depends On:** P-001 (types, exceptions, config), P-002 (position tracking), P-002A (error handling), P-003+ (exchange integrations), P-016A (utils)
**Enables:** P-009 (circuit breakers), P-010 (dynamic risk), P-010A (capital management), P-011+ (strategy implementations)

### Mandatory Integration Requirements
**CRITICAL**: This prompt MUST integrate with existing components and update P-001:

#### Required Imports from Previous Prompts:
```python
# From P-001 - MANDATORY: Use existing types and exceptions
from src.core.types import (
    Position, MarketData, Signal, OrderRequest,
    RiskMetrics, PositionLimits, RiskLevel  # Will be added to P-001 by this prompt
)
from src.core.exceptions import (
    RiskManagementError, PositionLimitError, DrawdownLimitError,
    ValidationError
)
from src.core.config import Config

# From P-002A - MANDATORY: Use error handling patterns
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import RecoveryScenario

# From P-016A - MANDATORY: Use decorators and validators
from src.utils.decorators import time_execution, retry, circuit_breaker
from src.utils.validators import validate_price, validate_quantity, validate_position
from src.utils.formatters import format_percentage, format_currency

# From P-003+ - MANDATORY: Use exchange interfaces
from src.exchanges.base import BaseExchange
```

#### Required Patterns from @COMMON_PATTERNS.md:
- MANDATORY: Use standard exception handling patterns for risk violations
- MANDATORY: Apply @time_execution decorator to all risk calculation methods
- MANDATORY: Use structured logging with risk context
- MANDATORY: Implement validation patterns for all risk parameters

### Task Details

#### 1. Base Risk Manager (`src/risk_management/base.py`)
Implement abstract risk management interface:
- Position size calculation with multiple algorithms
- Portfolio exposure monitoring
- Risk limit validation before trade execution
- Real-time risk metric calculation (VaR, drawdown)
- Risk parameter configuration and validation

#### 2. Position Sizing (`src/risk_management/position_sizing.py`)
Implement multiple position sizing methods:
- Fixed percentage (default 2% of portfolio)
- Kelly Criterion for optimal sizing
- Volatility-adjusted sizing using ATR
- Confidence-weighted sizing for ML strategies
- Portfolio heat mapping for correlation adjustment

#### 3. Portfolio Limits (`src/risk_management/portfolio_limits.py`)
Enforce portfolio-level risk controls:
- Maximum positions per strategy/symbol
- Correlation limits between positions
- Sector/asset class exposure limits
- Leverage limits for margin trading
- Concentration risk monitoring

### Directory Structure to Create
```
src/risk_management/
├── __init__.py
├── base.py
├── position_sizing.py
├── portfolio_limits.py
└── risk_metrics.py
```

### Reverse Integration Required
**CRITICAL**: This prompt MUST update P-001 files with risk-specific components:

#### Update P-001 src/core/types.py:
Add these risk-specific types to the existing file:
```python
# Add these to src/core/types.py after existing types
from enum import Enum

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PositionSizeMethod(Enum):
    FIXED_PCT = "fixed_percentage"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    CONFIDENCE_WEIGHTED = "confidence_weighted"

class RiskMetrics(BaseModel):
    var_1d: Decimal  # 1-day Value at Risk
    var_5d: Decimal  # 5-day Value at Risk
    expected_shortfall: Decimal
    max_drawdown: Decimal
    sharpe_ratio: Optional[Decimal] = None
    current_drawdown: Decimal = Decimal("0")
    risk_level: RiskLevel
    timestamp: datetime

class PositionLimits(BaseModel):
    max_position_size: Decimal
    max_positions_per_symbol: int = 1
    max_total_positions: int = 10
    max_portfolio_exposure: Decimal = Decimal("0.95")  # 95% max exposure
    max_sector_exposure: Decimal = Decimal("0.25")     # 25% per sector
    max_correlation_exposure: Decimal = Decimal("0.5")  # Max correlated exposure
    max_leverage: Decimal = Decimal("1.0")             # No leverage by default
```

#### Update P-001 src/core/config.py:
Add RiskConfig class to the main Config:
```python
# Add this to src/core/config.py
class RiskConfig(BaseConfig):
    # Position sizing
    default_position_size_method: str = Field("fixed_percentage", env="RISK_POSITION_SIZE_METHOD")
    default_position_size_pct: float = Field(0.02, env="RISK_POSITION_SIZE_PCT")  # 2%
    max_position_size_pct: float = Field(0.1, env="RISK_MAX_POSITION_SIZE_PCT")   # 10%
    
    # Portfolio limits
    max_total_positions: int = Field(10, env="RISK_MAX_POSITIONS")
    max_portfolio_exposure: float = Field(0.95, env="RISK_MAX_PORTFOLIO_EXPOSURE")
    max_sector_exposure: float = Field(0.25, env="RISK_MAX_SECTOR_EXPOSURE")
    max_correlation_exposure: float = Field(0.5, env="RISK_MAX_CORRELATION_EXPOSURE")
    
    # Risk thresholds
    max_daily_loss_pct: float = Field(0.05, env="RISK_MAX_DAILY_LOSS")      # 5%
    max_drawdown_pct: float = Field(0.15, env="RISK_MAX_DRAWDOWN")          # 15%
    var_confidence_level: float = Field(0.95, env="RISK_VAR_CONFIDENCE")     # 95%
    
    # Kelly Criterion settings
    kelly_lookback_days: int = Field(30, env="RISK_KELLY_LOOKBACK")
    kelly_max_fraction: float = Field(0.25, env="RISK_KELLY_MAX_FRACTION")   # Max 25% Kelly
    
    # Volatility adjustment
    volatility_window: int = Field(20, env="RISK_VOLATILITY_WINDOW")
    volatility_target: float = Field(0.02, env="RISK_VOLATILITY_TARGET")     # 2% daily vol target

# Update the main Config class:
class Config(BaseConfig):
    # ... existing fields ...
    risk: RiskConfig = RiskConfig()
```

### Integration Validation
After P-008 completion, verify:
```bash
# Test risk configuration loading
python -c "from src.core.config import Config; c = Config(); print(f'Risk config: {c.risk.default_position_size_pct}')"

# Test risk types
python -c "from src.core.types import RiskMetrics, PositionLimits, RiskLevel; print('Risk types imported successfully')"

# Test base risk manager
python -c "from src.risk_management.base import BaseRiskManager; print('BaseRiskManager imported successfully')"
```

---

## **Prompt P-009: Circuit Breakers and Emergency Controls**

**Title:** Implement circuit breakers and emergency stop mechanisms

### Context
- **Current State:** Base risk management established (P-008)
- **Target State:** Automated trading halts on excessive risk or system failures
- **Phase Goal:** Safety mechanisms to prevent catastrophic losses

**Technical Context**: Reference @SPECIFICATIONS.md Section 2.1.4 "Circuit Breaker Configuration".

### Dependencies
**Depends On:** P-008 (risk management base), P-001 (types, exceptions), P-002A (error handling), P-016A (utils)
**Enables:** P-010 (dynamic risk), P-010A (capital management), P-011+ (strategies with safety guarantees)

### Mandatory Integration Requirements
**CRITICAL**: This prompt MUST integrate with existing P-008 risk management:

#### Required Imports from Previous Prompts:
```python
# From P-008 - MANDATORY: Use existing risk management
from src.risk_management.base import BaseRiskManager
from src.risk_management.risk_metrics import RiskCalculator

# From P-001 - MANDATORY: Use existing types and exceptions
from src.core.types import RiskMetrics, RiskLevel, Position
from src.core.exceptions import RiskManagementError, DrawdownLimitError
from src.core.config import Config

# From P-002A - MANDATORY: Use error handling patterns
from src.error_handling.error_handler import ErrorHandler

# From P-016A - MANDATORY: Use decorators
from src.utils.decorators import time_execution, circuit_breaker
```

#### Required Patterns from @COMMON_PATTERNS.md:
- MANDATORY: Use standard exception handling patterns for circuit breaker triggers
- MANDATORY: Apply @time_execution decorator to all monitoring methods
- MANDATORY: Use structured logging for all circuit breaker events
- MANDATORY: Integrate with existing risk management framework

### Task Details

#### 1. Circuit Breaker System (`src/risk_management/circuit_breakers.py`)
Implement multiple circuit breaker types:
- Daily loss limit breaker (default 5%)
- Portfolio drawdown breaker (default 10%)
- Volatility spike detection
- Model confidence degradation
- System error rate monitoring

#### 2. Emergency Controls (`src/risk_management/emergency_controls.py`)
Implement emergency trading controls:
- Immediate position closure
- New order blocking
- Exchange-specific emergency procedures
- Manual override capabilities
- Recovery procedures and validation

### Directory Structure to Create
```
src/risk_management/
├── circuit_breakers.py
└── emergency_controls.py
```

---

## **Prompt P-010: Dynamic Risk Management**

**Title:** Implement dynamic and adaptive risk management with market regime detection

### Context
- **Current State:** Static risk management and circuit breakers active (P-008, P-009)
- **Target State:** Risk parameters that adapt to changing market conditions
- **Phase Goal:** More sophisticated risk management before ML integration

**Technical Context**: Reference @SPECIFICATIONS.md Section 2.1.2 "Dynamic Risk Management".

### Dependencies
**Depends On:** P-008 (base risk), P-009 (circuit breakers), P-001 (types), P-002A (error handling), P-016A (utils)
**Enables:** P-010A (capital management), P-011+ (advanced strategies), P-017+ (ML integration)

### Mandatory Integration Requirements
**CRITICAL**: This prompt MUST integrate with existing P-008/P-009 risk management:

#### Required Imports from Previous Prompts:
```python
# From P-008/P-009 - MANDATORY: Use existing risk management
from src.risk_management.base import BaseRiskManager
from src.risk_management.circuit_breakers import CircuitBreakerManager
from src.risk_management.risk_metrics import RiskCalculator

# From P-001 - MANDATORY: Use existing types
from src.core.types import RiskMetrics, RiskLevel, MarketData
from src.core.exceptions import RiskManagementError
from src.core.config import Config

# From P-002A - MANDATORY: Use error handling
from src.error_handling.error_handler import ErrorHandler

# From P-016A - MANDATORY: Use decorators and validators
from src.utils.decorators import time_execution, retry
from src.utils.validators import validate_price
```

#### Required Patterns from @COMMON_PATTERNS.md:
- MANDATORY: Extend existing BaseRiskManager, don't create new base class
- MANDATORY: Use standard exception handling for regime detection failures
- MANDATORY: Apply performance decorators to regime detection algorithms
- MANDATORY: Integrate with existing circuit breaker system

### Task Details

#### 1. Market Regime Detection (`src/risk_management/regime_detection.py`)
Implement market regime classification:
- Volatility regime detection (low/medium/high)
- Trend regime identification (trending/ranging)
- Correlation regime monitoring
- Regime change detection and alerting

#### 2. Adaptive Risk Parameters (`src/risk_management/adaptive_risk.py`)
Implement dynamic risk parameter adjustment:
- Volatility-adjusted position sizing
- Correlation-based portfolio limits
- Momentum-based stop loss adjustment
- Market stress testing and scenario analysis

### Directory Structure to Create
```
src/risk_management/
├── regime_detection.py
└── adaptive_risk.py
```

---

## **Prompt P-010A: Capital Management System Implementation**

**Title:** Implement comprehensive capital allocation and fund management system

### Context
- **Current State:** Dynamic risk management operational (P-010)
- **Target State:** Sophisticated capital allocation across strategies and exchanges
- **Phase Goal:** Optimal capital utilization and protection mechanisms

**Technical Context**: Reference @SPECIFICATIONS.md Section 3.5 "Capital Management". Support multi-strategy and multi-exchange allocation.

### Dependencies
**Depends On:** P-008+ (risk management), P-003+ (multi-exchange support), P-002 (portfolio tracking), P-001 (types, config), P-002A (error handling), P-016A (utils)
**Enables:** P-011 (strategies with capital allocation), P-013+ (strategy capital requirements)

### Mandatory Integration Requirements
**CRITICAL**: This prompt MUST integrate with existing risk management and update P-001:

#### Required Imports from Previous Prompts:
```python
# From P-008+ - MANDATORY: Use existing risk management
from src.risk_management.base import BaseRiskManager
from src.risk_management.position_sizing import PositionSizer

# From P-003+ - MANDATORY: Use existing exchange interfaces
from src.exchanges.base import BaseExchange

# From P-001 - MANDATORY: Use existing types and will add capital types
from src.core.types import Position, MarketData, CapitalAllocation  # Will be added
from src.core.exceptions import RiskManagementError, ValidationError
from src.core.config import Config

# From P-002A - MANDATORY: Use error handling
from src.error_handling.error_handler import ErrorHandler

# From P-016A - MANDATORY: Use decorators and validators
from src.utils.decorators import time_execution, retry
from src.utils.validators import validate_quantity, validate_percentage
from src.utils.formatters import format_currency
```

#### Required Patterns from @COMMON_PATTERNS.md:
- MANDATORY: Integrate with existing risk management framework
- MANDATORY: Use standard exception handling for capital allocation failures
- MANDATORY: Apply performance decorators to allocation algorithms
- MANDATORY: Use structured logging for all capital movements

### Task Details

#### 1. Capital Allocator (`src/capital_management/capital_allocator.py`)
Implement dynamic capital allocation framework:
- Total capital management with emergency reserves (10% default)
- Performance-based strategy allocation adjustments
- Risk-adjusted capital distribution using Sharpe ratios
- Dynamic rebalancing based on strategy performance
- Kelly Criterion integration for optimal sizing
- Capital scaling based on account growth

#### 2. Multi-Exchange Distribution (`src/capital_management/exchange_distributor.py`)
Implement capital distribution across exchanges:
- Dynamic distribution based on liquidity scores
- Fee structure optimization for capital efficiency
- API reliability weighting for exchange selection
- Historical slippage analysis for allocation decisions
- Rebalancing frequency management (daily default)
- Cross-exchange hedging and currency management

#### 3. Currency Manager (`src/capital_management/currency_manager.py`)
Implement multi-currency capital management:
- Base currency standardization (USDT default)
- Cross-currency exposure monitoring and hedging
- Currency conversion optimization
- Exchange rate risk management
- Multi-asset portfolio currency hedging
- Currency exposure limits and controls

#### 4. Fund Flow Manager (`src/capital_management/fund_flow_manager.py`)
Implement deposit/withdrawal management:
- Minimum capital requirements per strategy validation
- Withdrawal rules enforcement (profit-only, minimum maintenance)
- Auto-compounding of profits (weekly default)
- Performance-based withdrawal permissions
- Emergency capital preservation procedures
- Capital flow audit trail and reporting

#### 5. Position Sizer (`src/capital_management/position_sizer.py`)
Implement advanced position sizing algorithms:
- Kelly Criterion optimal position sizing
- Risk-parity position allocation
- Volatility-adjusted position sizing using ATR
- Confidence-weighted sizing for ML strategies
- Portfolio heat mapping for correlation adjustment
- Maximum position size enforcement across strategies

### Directory Structure to Create
```
src/capital_management/
├── __init__.py
├── capital_allocator.py
├── exchange_distributor.py
├── currency_manager.py
├── fund_flow_manager.py
└── position_sizer.py
```

### Capital Management Configuration
```yaml
capital_management:
  # Initial capital requirements
  minimum_capital:
    testing: 1000  # USD for paper trading
    production_starter: 10000  # USD minimum recommended
    production_optimal: 50000  # USD for full strategy deployment
  
  # Per-strategy minimum allocations
  per_strategy_minimum:
    mean_reversion: 5000
    trend_following: 10000
    ml_strategy: 15000  # Higher due to complexity
    arbitrage: 20000  # Needs liquidity
    market_making: 25000  # Inventory requirements
  
  # Exchange allocation
  exchange_allocation:
    distribution_mode: "dynamic"  # or "fixed", "proportional"
    
    fixed_distribution:
      binance: 0.5
      okx: 0.3
      coinbase: 0.2
    
    dynamic_factors:
      - liquidity_score
      - fee_structure
      - historical_slippage
      - api_reliability
    
    rebalancing:
      frequency: "daily"
      threshold: 0.05  # 5% deviation triggers rebalance
      method: "gradual"  # Avoid market impact
  
  # Currency management
  currency_management:
    base_currency: "USDT"
    supported_currencies: ["USDT", "BUSD", "USDC", "BTC", "ETH"]
    hedging_enabled: true
    hedging_threshold: 0.1  # 10% exposure
    hedge_ratio: 0.8  # 80% hedge coverage
  
  # Cash flow management
  cash_flow_management:
    deposits:
      min_amount: 1000
      processing_time: "immediate"
      allocation_strategy: "proportional"
    
    withdrawals:
      min_amount: 100
      max_percentage: 0.2  # Max 20% of total capital
      cooldown_period: 24  # hours
      
      rules:
        - name: "profit_only"
          description: "Only withdraw realized profits"
          enabled: true
        - name: "maintain_minimum"
          description: "Keep minimum capital for each strategy"
          enabled: true
        - name: "performance_based"
          description: "Allow withdrawals only if performance > threshold"
          threshold: 0.05  # 5% profit
          enabled: false
    
    auto_compound:
      enabled: true
      frequency: "weekly"
      profit_threshold: 100  # Minimum profit to compound
  
  # Capital protection
  capital_protection:
    emergency_reserve: 0.1  # 10% always in reserve
    
    drawdown_limits:
      daily: 0.05  # 5% max daily loss
      weekly: 0.10  # 10% max weekly loss
      monthly: 0.15  # 15% max monthly loss
    
    recovery_rules:
      - trigger: "daily_limit_hit"
        action: "pause_trading"
        duration: 24  # hours
      - trigger: "weekly_limit_hit"
        action: "reduce_position_sizes"
        reduction: 0.5  # 50% reduction
      - trigger: "monthly_limit_hit"
        action: "strategy_review"
        require_manual_restart: true
    
    capital_locks:
      profit_lock: 0.5  # Lock 50% of profits
      unlock_schedule: "quarterly"
      emergency_unlock: true  # Allow with 48h notice
```

### Acceptance Criteria
- Capital allocation adjusts dynamically based on performance
- Multi-exchange distribution optimizes for fees and liquidity
- Currency hedging maintains exposure within limits
- Withdrawal rules enforce capital protection
- Position sizing integrates Kelly Criterion and risk metrics

### Integration Points
- Portfolio data from P-002 (database models)
- Risk parameters from P-008+ (risk management)
- Exchange balances from P-003+ (exchange integrations)
- Strategy performance from P-011+ (strategy framework)
- Monitoring alerts via P-030+ (monitoring system)

### Reverse Integration Required
**CRITICAL**: This prompt MUST update P-001 and P-008 with capital management components:

#### Update P-001 src/core/types.py:
Add these capital management types:
```python
# Add these to src/core/types.py after existing types
class AllocationStrategy(Enum):
    EQUAL_WEIGHT = "equal_weight"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    VOLATILITY_WEIGHTED = "volatility_weighted"
    RISK_PARITY = "risk_parity"
    DYNAMIC = "dynamic"

class CapitalAllocation(BaseModel):
    strategy_id: str
    exchange: str
    allocated_amount: Decimal
    utilized_amount: Decimal = Decimal("0")
    available_amount: Decimal
    allocation_percentage: float
    last_rebalance: datetime
    
class FundFlow(BaseModel):
    from_strategy: Optional[str] = None
    to_strategy: Optional[str] = None
    from_exchange: Optional[str] = None
    to_exchange: Optional[str] = None
    amount: Decimal
    reason: str
    timestamp: datetime
    
class CapitalMetrics(BaseModel):
    total_capital: Decimal
    allocated_capital: Decimal
    available_capital: Decimal
    utilization_rate: float
    allocation_efficiency: float
    rebalance_frequency_hours: int
```

#### Update P-001 src/core/config.py:
Add capital management configuration:
```python
# Add this to src/core/config.py
class CapitalManagementConfig(BaseConfig):
    # Base settings
    base_currency: str = Field("USDT", env="CAPITAL_BASE_CURRENCY")
    total_capital: float = Field(10000.0, env="CAPITAL_TOTAL")
    emergency_reserve_pct: float = Field(0.1, env="CAPITAL_EMERGENCY_RESERVE")  # 10%
    
    # Allocation strategy
    allocation_strategy: str = Field("risk_parity", env="CAPITAL_ALLOCATION_STRATEGY")
    rebalance_frequency_hours: int = Field(24, env="CAPITAL_REBALANCE_FREQUENCY")
    min_allocation_pct: float = Field(0.01, env="CAPITAL_MIN_ALLOCATION")     # 1%
    max_allocation_pct: float = Field(0.25, env="CAPITAL_MAX_ALLOCATION")     # 25%
    
    # Exchange distribution
    max_exchange_allocation_pct: float = Field(0.6, env="CAPITAL_MAX_EXCHANGE_PCT")  # 60%
    min_exchange_balance: float = Field(100.0, env="CAPITAL_MIN_EXCHANGE_BALANCE")
    
    # Fund flow controls
    max_daily_reallocation_pct: float = Field(0.2, env="CAPITAL_MAX_DAILY_REALLOCATION")
    fund_flow_cooldown_minutes: int = Field(60, env="CAPITAL_FLOW_COOLDOWN")

# Update the main Config class:
class Config(BaseConfig):
    # ... existing fields ...
    capital_management: CapitalManagementConfig = CapitalManagementConfig()
```

#### Update P-008 Integration:
Update position sizing to use capital management:
```python
# P-008 position_sizing.py must be updated to integrate with capital allocation
# Import from capital management:
from src.capital_management.capital_allocator import CapitalAllocator
```

### Integration Validation
After P-010A completion, verify:
```bash
# Test capital management config
python -c "from src.core.config import Config; c = Config(); print(f'Capital config: {c.capital_management.base_currency}')"

# Test capital types
python -c "from src.core.types import CapitalAllocation, FundFlow; print('Capital types imported successfully')"

# Test capital management system
python -c "from src.capital_management.capital_allocator import CapitalAllocator; print('CapitalAllocator ready')"
```

---

## **Prompt P-011: Strategy Framework Foundation**

**Title:** Implement base strategy interface, factory, and configuration system

### Context
- **Current State:** Risk management system operational (P-008, P-009, P-010)
- **Target State:** Foundation for all trading strategies with unified interface
- **Phase Goal:** Enable strategy development and hot-swapping

**Technical Context**: Reference @SPECIFICATIONS.md Section 3 "Strategy Framework". Support static, dynamic, and AI strategies.

### Dependencies
**Depends On:** P-001 (types), P-008+ (risk management), P-003+ (exchanges), P-002A (error handling), P-016A (utils)
**Enables:** P-012 (static strategies), P-013 (dynamic strategies), P-013A-E (all strategy types), P-019 (AI strategies)

### Mandatory Integration Requirements
**CRITICAL**: This prompt MUST integrate with existing components and follow patterns:

#### Required Imports from Previous Prompts:
```python
# From P-001 - MANDATORY: Use existing types
from src.core.types import (
    Signal, MarketData, Position, OrderRequest, OrderResponse,
    StrategyConfig, StrategyStatus, StrategyMetrics, TradingMode
)
from src.core.exceptions import (
    ValidationError, RiskManagementError, ExecutionError
)
from src.core.config import Config

# From P-002A - MANDATORY: Use error handling patterns
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import RecoveryScenario

# From P-016A - MANDATORY: Use decorators and validators
from src.utils.decorators import time_execution, retry, circuit_breaker
from src.utils.validators import validate_signal, validate_position
from src.utils.formatters import format_percentage

# From P-008+ - MANDATORY: Use risk management
from src.risk_management.base import BaseRiskManager

# From P-003+ - MANDATORY: Use exchange interfaces
from src.exchanges.base import BaseExchange
```

#### Required Patterns from @COMMON_PATTERNS.md:
- MANDATORY: Use standard strategy implementation pattern
- MANDATORY: Apply @time_execution decorator to generate_signals()
- MANDATORY: Use structured logging with strategy context
- MANDATORY: Implement error handling with graceful degradation

### Task Details

#### 1. Base Strategy Interface (`src/strategies/base.py`)
Create comprehensive abstract strategy interface that ALL strategies must inherit from:

**CRITICAL**: This interface will be extended by P-013A-E and P-019. Design for extensibility.

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from decimal import Decimal
from datetime import datetime

# MANDATORY: Import from P-001
from src.core.types import (
    Signal, MarketData, Position, StrategyConfig, 
    StrategyStatus, StrategyMetrics
)
from src.core.exceptions import ValidationError, RiskManagementError

# MANDATORY: Import from P-016A  
from src.utils.decorators import time_execution, retry
from src.utils.validators import validate_signal

class BaseStrategy(ABC):
    """Base strategy interface that ALL strategies must inherit from"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = StrategyConfig(**config)
        self.name: str = self.__class__.__name__
        self.version: str = "1.0.0"
        self.status: StrategyStatus = StrategyStatus.STOPPED
        self.metrics: StrategyMetrics = StrategyMetrics()
        self._risk_manager: Optional[BaseRiskManager] = None
    
    @abstractmethod
    @time_execution
    async def generate_signals(self, data: MarketData) -> List[Signal]:
        """Generate trading signals from market data
        
        MANDATORY: All implementations must:
        1. Validate input data
        2. Return empty list on errors (graceful degradation)
        3. Apply confidence thresholds
        4. Log signal generation events
        """
        pass
    
    @abstractmethod
    async def validate_signal(self, signal: Signal) -> bool:
        """Validate signal before execution
        
        MANDATORY: Check signal confidence, direction, timestamp
        """
        pass
    
    @abstractmethod
    def get_position_size(self, signal: Signal) -> Decimal:
        """Calculate position size for signal
        
        MANDATORY: Integrate with risk management
        """
        pass
    
    @abstractmethod
    def should_exit(self, position: Position, data: MarketData) -> bool:
        """Determine if position should be closed
        
        MANDATORY: Check stop loss, take profit, time exits
        """
        pass
    
    # Standard methods that can be overridden
    async def pre_trade_validation(self, signal: Signal) -> bool:
        """Pre-trade validation hook"""
        if not await self.validate_signal(signal):
            return False
        
        if self._risk_manager:
            return await self._risk_manager.validate_signal(signal)
        
        return True
    
    async def post_trade_processing(self, trade_result: Any) -> None:
        """Post-trade processing hook"""
        # Update metrics
        self.metrics.total_trades += 1
        # Log trade result
        pass
    
    def set_risk_manager(self, risk_manager: BaseRiskManager) -> None:
        """Set risk manager for strategy"""
        self._risk_manager = risk_manager
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status.value,
            "config": self.config.dict(),
            "metrics": self.metrics.dict()
        }
```

**CRITICAL**: Future strategy prompts (P-012, P-013A-E, P-019) MUST inherit from this exact interface.

#### 2. Strategy Factory (`src/strategies/factory.py`)
Implement strategy instantiation and management:
- Dynamic strategy creation from configuration
- Strategy registration and discovery
- Hot-swapping capabilities without downtime
- Strategy versioning and rollback
- Resource management and cleanup

#### 3. Strategy Configuration (`src/strategies/config.py`)
Unified configuration system for all strategies:
- Parameter validation with type checking
- Default parameter management
- Environment-specific configurations
- Configuration schema validation
- Real-time parameter updates

### Directory Structure to Create
```
src/strategies/
├── __init__.py
├── base.py
├── factory.py
├── config.py
└── static/
    └── __init__.py
```

### Reverse Integration Required
**CRITICAL**: This prompt MUST update P-001 files with strategy-specific components:

#### Update P-001 src/core/types.py:
Add these strategy-specific types to the existing file:
```python
# Add these to src/core/types.py after existing types
from enum import Enum

class StrategyType(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic" 
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    EVOLUTIONARY = "evolutionary"
    HYBRID = "hybrid"
    AI_ML = "ai_ml"

class StrategyStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

class StrategyConfig(BaseModel):
    name: str
    strategy_type: StrategyType
    enabled: bool = True
    symbols: List[str]
    timeframe: str = "1h"
    min_confidence: float = Field(0.6, ge=0.0, le=1.0)
    max_positions: int = Field(5, ge=1)
    position_size_pct: float = Field(0.02, ge=0.001, le=0.1)
    stop_loss_pct: float = Field(0.02, ge=0.001, le=0.1)
    take_profit_pct: float = Field(0.04, ge=0.001, le=0.2)
    parameters: Dict[str, Any] = Field(default_factory=dict)

class StrategyMetrics(BaseModel):
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    win_rate: float = 0.0
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

#### Update P-001 src/core/config.py:
Add StrategyConfig class to the main Config:
```python
# Add this to src/core/config.py
class StrategyManagementConfig(BaseConfig):
    # Global strategy settings
    max_concurrent_strategies: int = Field(10, env="MAX_STRATEGIES")
    strategy_restart_delay: int = Field(60, env="STRATEGY_RESTART_DELAY")
    performance_window_days: int = Field(30, env="PERFORMANCE_WINDOW")
    
    # Default strategy parameters
    default_min_confidence: float = Field(0.6, env="DEFAULT_MIN_CONFIDENCE")
    default_position_size: float = Field(0.02, env="DEFAULT_POSITION_SIZE")
    default_stop_loss: float = Field(0.02, env="DEFAULT_STOP_LOSS")
    
    # Hot reloading
    enable_hot_reload: bool = Field(True, env="ENABLE_HOT_RELOAD")
    config_check_interval: int = Field(30, env="CONFIG_CHECK_INTERVAL")

# Update the main Config class:
class Config(BaseConfig):
    # ... existing fields ...
    strategies: StrategyManagementConfig = StrategyManagementConfig()
```

### Integration Validation
After P-011 completion, verify:
```bash
# Test strategy base interface
python -c "from src.strategies.base import BaseStrategy; print('BaseStrategy interface ready')"

# Test strategy factory
python -c "from src.strategies.factory import StrategyFactory; print('StrategyFactory ready')"

# Test strategy types in core
python -c "from src.core.types import StrategyConfig, StrategyStatus; print('Strategy types ready')"

# Test strategy config
python -c "from src.core.config import Config; c = Config(); print(f'Strategy config loaded: {c.strategies.max_concurrent_strategies}')"
```

---

## **Prompt P-012: Static Trading Strategies Implementation**

**Title:** Implement mean reversion, trend following, and breakout strategies

### Context
- **Current State:** Strategy framework established (P-011)
- **Target State:** Core static strategies operational and profitable
- **Phase Goal:** Proven trading strategies before adding complexity

**Technical Context**: Reference @SPECIFICATIONS.md Section 3.3 strategy configurations. Include comprehensive backtesting.

### Dependencies
**Depends On:** P-011 (strategy base), P-008+ (risk management), P-001 (types), P-002A (error handling), P-016A (utils)
**Enables:** P-013 (dynamic strategies), P-013A-E (all strategy types), P-020+ (execution engine)

### Mandatory Integration Requirements
**CRITICAL**: This prompt MUST use the exact BaseStrategy interface from P-011:

#### Required Imports (MANDATORY):
```python
# From P-011 - NEVER recreate the base strategy
from src.strategies.base import BaseStrategy

# From P-001 - Use existing types  
from src.core.types import (
    Signal, MarketData, Position, SignalDirection,
    StrategyConfig, StrategyType
)
from src.core.exceptions import ValidationError

# From P-016A - Use decorators and validators
from src.utils.decorators import time_execution, retry
from src.utils.validators import validate_price, validate_quantity

# From P-008+ - Use risk management
from src.risk_management.base import BaseRiskManager
```

#### Implementation Pattern (MANDATORY):
ALL strategy classes MUST follow this exact pattern from @COMMON_PATTERNS.md:
```python
class MeanReversionStrategy(BaseStrategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "mean_reversion"
        self.strategy_type = StrategyType.STATIC
    
    @time_execution
    async def generate_signals(self, data: MarketData) -> List[Signal]:
        # MANDATORY: Use graceful error handling
        try:
            # MANDATORY: Input validation
            if not data or not data.price:
                return []
            
            # Strategy logic here
            signals = await self._calculate_signals(data)
            
            # MANDATORY: Signal validation
            validated_signals = []
            for signal in signals:
                if await self.validate_signal(signal):
                    validated_signals.append(signal)
            
            return validated_signals
        except Exception as e:
            logger.error("Signal generation failed", strategy=self.name, error=str(e))
            return []  # MANDATORY: Graceful degradation
```

### Task Details

#### 1. Mean Reversion Strategy (`src/strategies/static/mean_reversion.py`)
Implement statistical arbitrage strategy:
- Z-score calculation with configurable lookback (default 20 periods)
- Entry threshold configuration (default ±2.0 standard deviations)
- Exit threshold (default ±0.5 standard deviations)
- ATR-based stop loss and take profit
- Volume and volatility filters
- Multi-timeframe confirmation

#### 2. Trend Following Strategy (`src/strategies/static/trend_following.py`)
Implement momentum-based strategy:
- Moving average crossover signals (20/50 default)
- RSI momentum confirmation (60/40 thresholds)
- Volume confirmation requirements
- Pyramiding support (max 3 levels)
- Trailing stop implementation
- Time-based exit rules

#### 3. Breakout Strategy (`src/strategies/static/breakout.py`)
Implement support/resistance breakout strategy:
- Support/resistance level detection
- Breakout confirmation with volume
- False breakout filtering
- Consolidation period requirements
- Dynamic stop loss placement
- Target calculation based on range

### Directory Structure to Create
```
src/strategies/static/
├── mean_reversion.py
├── trend_following.py
└── breakout.py
```

### Strategy Configurations
```yaml
strategies:
  mean_reversion:
    enabled: true
    symbols: ["BTCUSDT", "ETHUSDT"]
    timeframe: "5m"
    lookback_period: 20
    entry_threshold: 2.0
    exit_threshold: 0.5
    
  trend_following:
    enabled: true
    symbols: ["BTCUSDT", "ETHUSDT"]
    timeframe: "1h"
    fast_ma: 20
    slow_ma: 50
    rsi_period: 14
```

---

## **Prompt P-013: Dynamic and Adaptive Strategies**

**Title:** Implement adaptive strategies with market regime awareness

### Context
- **Current State:** Static strategies operational (P-012)
- **Target State:** Adaptive strategies that adjust to market conditions
- **Phase Goal:** More sophisticated strategies before ML integration

**Technical Context**: Reference @SPECIFICATIONS.md Section 3.1.2 "Dynamic Strategies".

### Dependencies
**Depends On:** P-012 (static strategies), P-010 (regime detection), P-011 (strategy base), P-001 (types), P-002A (error handling), P-016A (utils)
**Enables:** P-013A (arbitrage), P-013B (market making), P-013C (backtesting), P-013D (evolutionary), P-013E (hybrid), P-019 (AI strategies)

### Mandatory Integration Requirements
**CRITICAL**: This prompt MUST inherit from exact P-011 BaseStrategy and integrate with existing components:

#### Required Imports from Previous Prompts:
```python
# From P-011 - MANDATORY: Use exact BaseStrategy interface
from src.strategies.base import BaseStrategy

# From P-010 - MANDATORY: Use regime detection
from src.risk_management.regime_detection import RegimeDetector
from src.risk_management.adaptive_risk import AdaptiveRiskManager

# From P-001 - MANDATORY: Use existing types
from src.core.types import (
    Signal, MarketData, Position, StrategyType, 
    StrategyConfig, StrategyStatus
)
from src.core.exceptions import ValidationError, RiskManagementError

# From P-002A - MANDATORY: Use error handling
from src.error_handling.error_handler import ErrorHandler

# From P-016A - MANDATORY: Use decorators and validators
from src.utils.decorators import time_execution, retry
from src.utils.validators import validate_signal
```

#### Required Patterns from @COMMON_PATTERNS.md:
- MANDATORY: ALL strategies must inherit from P-011 BaseStrategy exactly
- MANDATORY: Use standard strategy implementation pattern
- MANDATORY: Apply performance decorators to signal generation
- MANDATORY: Implement graceful error handling with empty signal list fallback

### Task Details

#### 1. Adaptive Momentum Strategy (`src/strategies/dynamic/adaptive_momentum.py`)
Implement regime-aware momentum strategy:
- Market regime detection integration
- Parameter adjustment based on volatility
- Correlation-adjusted position sizing
- Dynamic stop loss based on market stress
- Multi-asset momentum scoring

#### 2. Volatility Breakout Strategy (`src/strategies/dynamic/volatility_breakout.py`)
Implement volatility-based breakout strategy:
- ATR-based breakout thresholds
- Volatility regime filtering
- Dynamic position sizing
- Time-decay adjustments
- Cross-asset volatility monitoring

### Directory Structure to Create
```
src/strategies/dynamic/
├── __init__.py
├── adaptive_momentum.py
└── volatility_breakout.py
```

---

## **Prompt P-013A: Arbitrage Strategy Implementation**

**Title:** Implement cross-exchange and triangular arbitrage strategies

### Context
- **Current State:** Dynamic strategies operational (P-013)
- **Target State:** Advanced arbitrage strategies for cross-exchange and triangular opportunities
- **Phase Goal:** High-frequency arbitrage capabilities for profit generation

**Technical Context**: Reference @SPECIFICATIONS.md Section 3.3.4 "Arbitrage Strategy". Requires ultra-low latency execution.

### Dependencies
**Depends On:** P-013 (dynamic strategies), P-003+ (multi-exchange support), P-020 (execution engine)
**Enables:** P-013B (market making), P-014 (data pipeline for latency monitoring)

### Task Details

#### 1. Cross-Exchange Arbitrage (`src/strategies/static/cross_exchange_arbitrage.py`)
Implement cross-exchange price differential strategy:
- Real-time price monitoring across Binance, OKX, Coinbase
- Spread detection with configurable minimum profit threshold (0.1% after fees)
- Simultaneous order placement across exchanges
- Position balancing and currency conversion handling
- Latency monitoring with 100ms threshold alerting
- Execution shortfall analysis and optimization

#### 2. Triangular Arbitrage (`src/strategies/static/triangular_arbitrage.py`)
Implement triangular arbitrage opportunities:
- Three-pair arbitrage detection (e.g., BTC/USDT → ETH/BTC → ETH/USDT)
- Path optimization for maximum profit extraction
- Rapid execution sequencing with partial fill handling
- Currency conversion chain validation
- Slippage impact analysis across the arbitrage chain
- Maximum execution time enforcement (500ms)

#### 3. Arbitrage Opportunity Scanner (`src/strategies/static/arbitrage_scanner.py`)
Implement opportunity detection engine:
- Real-time spread calculation across all supported pairs
- Order book depth analysis for position sizing
- Fee calculation across different exchanges
- Network latency compensation
- Opportunity prioritization by profit potential
- Risk assessment for each arbitrage trade

### Directory Structure to Create
```
src/strategies/static/
├── cross_exchange_arbitrage.py
├── triangular_arbitrage.py
└── arbitrage_scanner.py

config/strategies/
└── arbitrage.yaml
```

### Strategy Configuration
```yaml
strategies:
  arbitrage:
    enabled: true
    type: "triangular"  # or "cross_exchange"
    
    # Exchanges for cross-exchange arbitrage
    exchanges: ["binance", "okx", "coinbase"]
    
    # Pairs for triangular arbitrage
    triangular_paths:
      - ["BTCUSDT", "ETHBTC", "ETHUSDT"]
      - ["BTCUSDT", "BNBBTC", "BNBUSDT"]
    
    # Thresholds
    min_profit_threshold: 0.001  # 0.1% after fees
    max_execution_time: 500  # milliseconds
    
    # Risk limits
    max_position_size: 0.1  # 10% of portfolio
    max_open_arbitrages: 5
    
    # Execution
    order_type: "market"  # Speed is critical
    partial_fill_timeout: 1000  # ms
    
    # Monitoring
    latency_threshold: 100  # ms
    slippage_limit: 0.0005  # 0.05%
```

### Acceptance Criteria
- Cross-exchange arbitrage detection <50ms
- Triangular arbitrage execution <500ms total
- Profit calculation accuracy >99.9%
- Risk limits enforced before execution
- Latency monitoring and alerting operational

### Integration Points
- Real-time market data from P-014 (data pipeline)
- Risk validation from P-008+ (risk management)
- Multi-exchange execution via P-003+ (exchanges)
- Performance monitoring via P-030+ (monitoring)

---

## **Prompt P-013B: Market Making Strategy Implementation**

**Title:** Implement sophisticated market making strategy with inventory management

### Context
- **Current State:** Arbitrage strategies operational (P-013A)
- **Target State:** Active market making with spread management and inventory control
- **Phase Goal:** Liquidity provision and spread capture capabilities

**Technical Context**: Reference @SPECIFICATIONS.md Section 3.3.5 "Market Making Strategy". Requires advanced order management.

### Dependencies
**Depends On:** P-013A (arbitrage), P-020 (execution engine), P-008+ (risk management)
**Enables:** P-014 (data pipeline), P-019 (AI strategies)

### Task Details

#### 1. Market Making Engine (`src/strategies/static/market_making.py`)
Implement core market making logic:
- Dual-sided order placement with configurable spreads
- Order level management (5 levels default)
- Spread adjustment based on volatility and competition
- Inventory skew implementation for risk management
- Order refresh management (30-second default)
- Competitive quote monitoring and adjustment

#### 2. Inventory Manager (`src/strategies/static/inventory_manager.py`)
Implement inventory risk management:
- Target inventory maintenance (50% of max position)
- Inventory risk aversion adjustments
- Position rebalancing triggers
- Currency hedging for multi-asset market making
- Inventory-based spread skewing
- Emergency inventory liquidation procedures

#### 3. Spread Optimizer (`src/strategies/static/spread_optimizer.py`)
Implement dynamic spread optimization:
- Volatility-based spread adjustment (2x multiplier default)
- Order book imbalance detection
- Competitor spread analysis
- Market impact assessment
- Optimal bid-ask spread calculation
- Adaptive spread widening during high volatility

### Directory Structure to Create
```
src/strategies/static/
├── market_making.py
├── inventory_manager.py
└── spread_optimizer.py

config/strategies/
└── market_making.yaml
```

### Strategy Configuration
```yaml
strategies:
  market_making:
    enabled: true
    symbols: ["ETHUSDT", "BNBUSDT"]
    
    # Spread configuration
    base_spread: 0.001  # 0.1%
    spread_adjustment:
      volatility_multiplier: 2.0
      inventory_skew: true
      competitive_quotes: true
    
    # Order management
    order_levels: 5
    order_size_distribution: "exponential"  # or "linear", "constant"
    base_order_size: 0.01  # BTC
    size_multiplier: 1.5  # For each level
    
    # Inventory management
    target_inventory: 0.5  # 50% of max position
    max_inventory: 1.0  # BTC
    inventory_risk_aversion: 0.1
    
    # Risk parameters
    max_position_value: 10000  # USD
    stop_loss_inventory: 2.0  # BTC
    daily_loss_limit: 100  # USD
    
    # Smart features
    order_refresh_time: 30  # seconds
    adaptive_spreads: true
    competition_monitoring: true
    min_profit_per_trade: 0.00001  # BTC
```

### Acceptance Criteria
- Dual-sided quotes maintained >95% uptime
- Inventory within target range >90% of time
- Spread adjustments respond to volatility <10s
- Order refresh cycles execute reliably
- Competitive positioning maintained automatically

### Integration Points
- Order book data from P-014 (data pipeline)
- Risk limits from P-008+ (risk management)
- Order execution via P-020 (execution engine)
- Performance tracking via P-030+ (monitoring)

---

## **Prompt P-013C: Backtesting Framework Implementation**

**Title:** Implement comprehensive backtesting framework with historical market replay

### Context
- **Current State:** All core strategies implemented (P-012, P-013, P-013A, P-013B)
- **Target State:** Robust backtesting framework for strategy validation and optimization
- **Phase Goal:** Historical testing capabilities before live deployment

**Technical Context**: Reference @SPECIFICATIONS.md Section 3.4 "Backtesting Framework". Support walk-forward analysis and Monte Carlo simulation.

### Dependencies
**Depends On:** P-012+ (all strategies), P-008+ (risk management), P-002 (database for historical data)
**Enables:** P-013D (evolutionary strategies), P-013E (hybrid strategies), P-014 (data pipeline)

### Task Details

#### 1. Historical Market Replay Engine (`src/backtesting/market_replay.py`)
Implement historical data replay system:
- OHLCV data replay with configurable timeframes
- Order book depth simulation for realistic fills
- Trade execution simulation with slippage modeling
- Sentiment data integration for comprehensive backtests
- Multi-exchange data synchronization for arbitrage testing
- Real-time feed simulation for latency testing

#### 2. Walk-Forward Analysis (`src/backtesting/walk_forward.py`)
Implement rolling window out-of-sample testing:
- Training/validation/test period management
- Rolling window optimization with configurable periods
- Parameter stability analysis across time periods
- Out-of-sample performance tracking
- Overfitting detection and prevention
- Strategy parameter optimization over time

#### 3. Monte Carlo Simulation (`src/backtesting/monte_carlo.py`)
Implement probabilistic scenario testing:
- Random price path generation based on historical statistics
- Multiple scenario stress testing (1000+ iterations)
- Confidence interval calculation for performance metrics
- Tail risk analysis and worst-case scenario planning
- Parameter sensitivity analysis
- Drawdown probability distributions

#### 4. Strategy Comparison Dashboard (`src/backtesting/comparison_engine.py`)
Implement multi-strategy comparison system:
- Normalized performance metrics across strategies
- Risk-adjusted return comparisons (Sharpe, Sortino, Calmar)
- Correlation analysis between strategy performances
- Portfolio optimization with multiple strategies
- Performance attribution analysis
- Strategy selection recommendations

#### 5. Backtesting Metrics Calculator (`src/backtesting/metrics.py`)
Implement comprehensive performance metrics:
- Sharpe Ratio, Sortino Ratio, Calmar Ratio calculations
- Maximum Drawdown analysis with duration
- Win Rate, Profit Factor, Average Win/Loss calculations
- Value at Risk (VaR) and Expected Shortfall
- Beta and correlation analysis
- Trade-level performance analytics

### Directory Structure to Create
```
src/backtesting/
├── __init__.py
├── market_replay.py
├── walk_forward.py
├── monte_carlo.py
├── comparison_engine.py
└── metrics.py
```

### Backtesting Configuration
```yaml
backtesting:
  # Data settings
  historical_data:
    start_date: "2020-01-01"
    end_date: "2024-01-01"
    timeframe: "1m"  # 1m, 5m, 1h, 1d
    exchanges: ["binance", "okx", "coinbase"]
  
  # Walk-forward analysis
  walk_forward:
    training_period: 365  # days
    validation_period: 30  # days
    test_period: 30  # days
    step_size: 7  # days between windows
    min_trades: 50  # minimum trades per period
  
  # Monte Carlo settings
  monte_carlo:
    iterations: 1000
    confidence_levels: [0.95, 0.99]
    random_seed: 42
    scenario_types: ["normal", "stress", "extreme"]
  
  # Performance metrics
  metrics:
    benchmark: "buy_and_hold_btc"
    risk_free_rate: 0.02  # 2% annual
    required_sharpe: 1.0
    max_drawdown_limit: 0.15  # 15%
  
  # Execution simulation
  execution:
    slippage_model: "square_root"
    commission_rate: 0.001  # 0.1%
    market_impact: true
    partial_fills: true
```

### Acceptance Criteria
- Historical replay matches actual market conditions >99% accuracy
- Walk-forward analysis supports 2+ years of data
- Monte Carlo simulations complete <5 minutes for 1000 iterations
- Strategy comparison dashboard shows normalized metrics
- All financial metrics calculated correctly and validated

### Integration Points
- Historical data from P-002 (database) and P-014 (data pipeline)
- Strategy testing via P-012+ (all strategy implementations)
- Risk validation through P-008+ (risk management)
- Web interface integration via P-027+ (API endpoints)
- Performance monitoring via P-030+ (monitoring system)

---

## **Prompt P-013D: Evolutionary Trading Strategies Implementation**

**Title:** Implement evolutionary algorithms for adaptive strategy optimization

### Context
- **Current State:** Backtesting framework operational (P-013C)
- **Target State:** Self-optimizing strategies using evolutionary algorithms
- **Phase Goal:** Advanced AI strategies that evolve and improve over time

**Technical Context**: Reference @SPECIFICATIONS.md Section 3.1.4 "Evolutionary Strategies". Implement genetic algorithms and neuroevolution.

### Dependencies
**Depends On:** P-013C (backtesting), P-012+ (all base strategies), P-017+ (ML infrastructure)
**Enables:** P-013E (hybrid strategies), P-019 (AI strategies)

### Task Details

#### 1. Genetic Algorithm Strategy (`src/strategies/evolutionary/genetic_strategy.py`)
Implement genetic algorithm-based strategy evolution:
- Population-based strategy rule evolution
- Fitness evaluation using backtesting results
- Crossover and mutation operators for strategy parameters
- Multi-objective optimization (profit vs risk)
- Adaptive population size based on market conditions
- Elitism preservation for best-performing strategies

#### 2. Neuroevolution Engine (`src/strategies/evolutionary/neuroevolution.py`)
Implement neural network evolution:
- Neural network topology evolution (NEAT algorithm)
- Weight and bias evolution for decision networks
- Activation function optimization
- Network complexity control and regularization
- Performance-based selection and reproduction
- Real-time adaptation to market regime changes

#### 3. Reinforcement-Evolved Policies (`src/strategies/evolutionary/rl_evolution.py`)
Implement reinforcement learning policy evolution:
- Policy gradient evolution with genetic operators
- Multi-agent evolutionary strategies
- Exploration vs exploitation balance optimization
- Reward function adaptation based on performance
- Environment adaptation for different market conditions
- Meta-learning for strategy transfer across assets

#### 4. Evolution Controller (`src/strategies/evolutionary/evolution_controller.py`)
Implement evolutionary process management:
- Population lifecycle management
- Evolution scheduling and resource allocation
- Performance tracking and fitness evaluation
- Strategy versioning and rollback capabilities
- Distributed evolution across multiple instances
- Evolution history and lineage tracking

### Directory Structure to Create
```
src/strategies/evolutionary/
├── __init__.py
├── genetic_strategy.py
├── neuroevolution.py
├── rl_evolution.py
└── evolution_controller.py
```

### Evolutionary Configuration
```yaml
evolutionary_strategies:
  genetic_algorithm:
    enabled: true
    population_size: 50
    mutation_rate: 0.1
    crossover_rate: 0.8
    elitism_count: 5
    max_generations: 100
    
    fitness_metrics:
      - sharpe_ratio: 0.4
      - total_return: 0.3
      - max_drawdown: 0.2
      - win_rate: 0.1
    
    parameter_ranges:
      lookback_period: [10, 100]
      threshold_multiplier: [1.0, 3.0]
      stop_loss_pct: [0.01, 0.05]
  
  neuroevolution:
    enabled: true
    network_config:
      input_size: 20  # Number of features
      hidden_layers: [64, 32]
      output_size: 3  # Buy, sell, hold
      activation: "relu"
    
    evolution_params:
      population_size: 30
      mutation_rate: 0.15
      structure_mutation_rate: 0.05
      weight_mutation_std: 0.1
      max_generations: 50
  
  reinforcement_evolution:
    enabled: true
    policy_type: "actor_critic"
    population_size: 20
    learning_rate: 0.001
    discount_factor: 0.99
    entropy_coefficient: 0.01
    
    evolution_schedule:
      generations_per_epoch: 10
      evaluation_episodes: 100
      selection_pressure: 0.5
```

### Acceptance Criteria
- Genetic algorithms optimize strategy parameters over multiple generations
- Neuroevolution produces neural networks with improving performance
- RL policy evolution shows adaptation to market conditions
- Evolution controller manages resource allocation efficiently
- All evolutionary strategies integrate with backtesting framework

### Integration Points
- Backtesting framework from P-013C for fitness evaluation
- ML infrastructure from P-017+ for neural network support
- Performance monitoring via P-030+ for evolution tracking
- Risk management from P-008+ for constraint enforcement

---

## **Prompt P-013E: Hybrid Strategy Implementation**

**Title:** Implement hybrid strategies combining multiple approaches with intelligent fallback

### Context
- **Current State:** Evolutionary strategies operational (P-013D)
- **Target State:** Sophisticated hybrid strategies with intelligent mode switching
- **Phase Goal:** Production-ready adaptive strategies before execution engine

**Technical Context**: Reference @SPECIFICATIONS.md Section 3.1.5 "Hybrid Strategies". Implement adaptive ensembles and fallback mechanisms.

### Dependencies
**Depends On:** P-013D (evolutionary), P-012+ (all strategy types), P-010 (regime detection)
**Enables:** P-014 (data pipeline), P-019 (AI strategies), P-020 (execution engine)

### Task Details

#### 1. Rule-Based AI Strategy (`src/strategies/hybrid/rule_based_ai.py`)
Implement combination of traditional rules with AI predictions:
- Traditional technical analysis rule engine
- AI prediction integration with confidence weighting
- Rule validation against AI predictions
- Conflict resolution between rules and AI
- Performance attribution between rule-based and AI decisions
- Dynamic rule weight adjustment based on performance

#### 2. Adaptive Ensemble Strategy (`src/strategies/hybrid/adaptive_ensemble.py`)
Implement dynamic weighting of multiple strategy types:
- Multi-strategy ensemble with dynamic weights
- Performance-based weight adjustment
- Correlation analysis for strategy selection
- Market regime-based strategy allocation
- Real-time strategy performance monitoring
- Automatic strategy inclusion/exclusion based on performance

#### 3. Fallback Handler (`src/strategies/hybrid/fallback_handler.py`)
Implement intelligent fallback mechanisms:
- AI failure detection and automatic fallback to static rules
- Confidence threshold monitoring for mode switching
- Gradual model re-integration after recovery
- Performance comparison between modes
- Emergency static mode activation
- Fallback decision audit trail

#### 4. Performance Blender (`src/strategies/hybrid/performance_blender.py`)
Implement real-time strategy allocation:
- Recent performance analysis and trending
- Risk-adjusted performance measurement
- Strategy capacity and position size allocation
- Dynamic rebalancing based on performance
- Market condition-based strategy selection
- Performance prediction for strategy allocation

#### 5. Hybrid Controller (`src/strategies/hybrid/hybrid_controller.py`)
Implement central hybrid strategy coordination:
- Mode switching logic and decision engine
- Strategy coordination and conflict resolution
- Resource allocation across strategy types
- Performance aggregation and reporting
- Risk overlay for all strategy modes
- Configuration management for hybrid parameters

### Directory Structure to Create
```
src/strategies/hybrid/
├── __init__.py
├── rule_based_ai.py
├── adaptive_ensemble.py
├── fallback_handler.py
├── performance_blender.py
└── hybrid_controller.py
```

### Hybrid Strategy Configuration
```yaml
hybrid_strategies:
  rule_based_ai:
    enabled: true
    rule_weight: 0.6
    ai_weight: 0.4
    
    confidence_thresholds:
      high_confidence: 0.8  # Use AI prediction
      medium_confidence: 0.6  # Blend with rules
      low_confidence: 0.4  # Use rules primarily
    
    rule_types:
      - "moving_average_crossover"
      - "rsi_divergence"
      - "support_resistance"
      - "volume_confirmation"
  
  adaptive_ensemble:
    enabled: true
    strategies:
      - name: "mean_reversion"
        initial_weight: 0.25
        min_weight: 0.1
        max_weight: 0.5
      - name: "trend_following"
        initial_weight: 0.25
        min_weight: 0.1
        max_weight: 0.5
      - name: "arbitrage"
        initial_weight: 0.25
        min_weight: 0.0
        max_weight: 0.4
      - name: "ml_strategy"
        initial_weight: 0.25
        min_weight: 0.1
        max_weight: 0.6
    
    rebalancing:
      frequency: "daily"
      lookback_period: 30  # days
      performance_threshold: 0.02  # 2% minimum change to rebalance
  
  fallback_mechanisms:
    enabled: true
    
    triggers:
      - name: "model_timeout"
        threshold: 5  # seconds
        action: "fallback_to_static"
      - name: "low_confidence"
        threshold: 0.4
        consecutive_count: 5
        action: "increase_static_weight"
      - name: "performance_degradation"
        threshold: -0.05  # -5% relative performance
        lookback_days: 7
        action: "disable_ai_temporarily"
    
    recovery:
      gradual_reintegration: true
      performance_validation_period: 24  # hours
      minimum_confidence_for_recovery: 0.6
  
  performance_blending:
    enabled: true
    
    allocation_method: "sharpe_ratio_weighted"
    rebalancing_frequency: "6h"
    min_allocation_per_strategy: 0.05  # 5%
    max_allocation_per_strategy: 0.60  # 60%
    
    performance_metrics:
      - sharpe_ratio: 0.5
      - sortino_ratio: 0.3
      - calmar_ratio: 0.2
```

### Acceptance Criteria
- Rule-based AI strategies show improved performance over pure rule-based
- Adaptive ensembles automatically adjust weights based on performance
- Fallback mechanisms activate within specified thresholds
- Performance blending optimizes capital allocation dynamically
- Hybrid controller coordinates all modes seamlessly

### Integration Points
- Strategy performance data from P-013C (backtesting)
- Risk management validation from P-008+ (risk framework)
- Regime detection from P-010 (dynamic risk management)
- Execution coordination with P-020 (execution engine)
- Performance monitoring via P-030+ (monitoring system)

---

## **Prompt P-014: Data Pipeline and Sources Integration**

**Title:** Implement comprehensive data ingestion pipeline with validation

### Context
- **Current State:** Strategies require market data input
- **Target State:** Reliable, validated data pipeline feeding all strategies
- **Phase Goal:** High-quality data foundation for ML and strategies

**Technical Context**: Reference @SPECIFICATIONS.md Section 4 "Data Management System".

### Dependencies
**Depends On:** P-002 (database), P-003+ (exchanges for market data), P-001 (types), P-002A (error handling), P-016A (utils)
**Enables:** P-015 (feature engineering), P-016 (data quality), P-017+ (ML models)

### Mandatory Integration Requirements
**CRITICAL**: This prompt MUST integrate with existing components and use P-001 data types:

#### Required Imports from Previous Prompts:
```python
# From P-001 - MANDATORY: Use existing MarketData types
from src.core.types import MarketData, Ticker, OrderBook
from src.core.exceptions import DataError, DataValidationError, DataSourceError
from src.core.config import Config

# From P-003+ - MANDATORY: Use existing exchange interfaces
from src.exchanges.base import BaseExchange

# From P-002 - MANDATORY: Use existing database models
from src.database.models import MarketDataRecord, Trade
from src.database.connection import get_session

# From P-002A - MANDATORY: Use error handling patterns
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import RecoveryScenario

# From P-016A - MANDATORY: Use decorators and validators
from src.utils.decorators import time_execution, retry, circuit_breaker
from src.utils.validators import validate_price, validate_quantity
from src.utils.formatters import format_currency
```

#### Required Patterns from @COMMON_PATTERNS.md:
- MANDATORY: Use standard exception handling for data source failures
- MANDATORY: Apply @retry decorators for external data source calls
- MANDATORY: Use structured logging for all data pipeline events
- MANDATORY: Follow async context manager patterns for data connections

### Task Details

#### 1. Data Sources (`src/data/sources/`)
Implement multiple data source integrations:
- Market data from exchanges (OHLCV, order book, trades)
- News sentiment from NewsAPI
- Social media sentiment (Twitter/Reddit APIs)
- Economic indicators (FRED API)
- Alternative data (weather, satellite data)

#### 2. Data Ingestion Pipeline (`src/data/pipeline/ingestion.py`)
Implement robust data ingestion:
- Real-time stream processing
- Batch data collection for historical data
- Data normalization across sources
- Timestamp synchronization and alignment
- Error handling and retry logic

#### 3. Data Validation (`src/data/pipeline/validation.py`)
Implement comprehensive data validation:
- Schema validation for all data types
- Range and sanity checks
- Missing data detection and handling
- Outlier detection using statistical methods
- Data quality scoring and monitoring

### Directory Structure to Create
```
src/data/
├── __init__.py
├── sources/
│   ├── __init__.py
│   ├── market_data.py
│   ├── news_data.py
│   ├── social_media.py
│   └── alternative_data.py
└── pipeline/
    ├── __init__.py
    ├── ingestion.py
    ├── processing.py
    ├── validation.py
    └── storage.py
```

---

## **Prompt P-015: Feature Engineering Framework**

**Title:** Implement comprehensive feature engineering pipeline with 100+ indicators

### Context
- **Current State:** Data pipeline operational (P-014)
- **Target State:** Rich feature set for ML models and advanced strategies
- **Phase Goal:** Comprehensive technical and alternative features

**Technical Context**: Reference @SPECIFICATIONS.md Section 4.3 "Feature Engineering". Use TA-Lib for technical indicators.

### Dependencies
**Depends On:** P-014 (data pipeline), P-001 (types), P-002A (error handling), P-016A (utils)
**Enables:** P-016 (data quality), P-017+ (ML models), P-019 (AI strategies)

### Mandatory Integration Requirements
**CRITICAL**: This prompt MUST integrate with existing data pipeline and use P-001 types:

#### Required Imports from Previous Prompts:
```python
# From P-014 - MANDATORY: Use existing data pipeline
from src.data.pipeline.processor import DataProcessor
from src.data.sources.base import BaseDataSource

# From P-001 - MANDATORY: Use existing MarketData types
from src.core.types import MarketData, Signal
from src.core.exceptions import DataError, ValidationError
from src.core.config import Config

# From P-002A - MANDATORY: Use error handling patterns
from src.error_handling.error_handler import ErrorHandler

# From P-016A - MANDATORY: Use decorators and validators
from src.utils.decorators import time_execution, retry, cache_result
from src.utils.validators import validate_price, validate_quantity
from src.utils.formatters import format_percentage

# Required for technical analysis
import talib
import pandas as pd
import numpy as np
```

#### Required Patterns from @COMMON_PATTERNS.md:
- MANDATORY: Use standard exception handling for feature calculation failures
- MANDATORY: Apply @cache_result decorators for expensive feature calculations
- MANDATORY: Use structured logging for feature engineering pipeline events
- MANDATORY: Implement graceful degradation when features cannot be calculated

### Task Details

#### 1. Technical Indicators (`src/data/features/technical_indicators.py`)
Implement 50+ technical indicators:
- **Price-based**: SMA, EMA, VWAP, Bollinger Bands, Pivot Points
- **Momentum**: RSI, MACD, Stochastic, Williams %R, CCI
- **Volume**: OBV, Volume Profile, MFI, A/D Line, VWAP
- **Volatility**: ATR, Historical Volatility, GARCH estimates
- **Market Structure**: Support/Resistance, Fibonacci levels

#### 2. Statistical Features (`src/data/features/statistical_features.py`)
Implement statistical feature extraction:
- Rolling statistics (mean, std, skewness, kurtosis)
- Autocorrelation features
- Cross-correlation between assets
- Regime indicators (trending vs ranging)
- Seasonality and cyclical features

#### 3. Alternative Features (`src/data/features/alternative_features.py`)
Implement alternative data features:
- News sentiment scores
- Social media sentiment and trend analysis
- Economic indicator derivatives
- Market microstructure features
- Cross-asset correlation features

### Directory Structure to Create
```
src/data/features/
├── __init__.py
├── technical_indicators.py
├── statistical_features.py
└── alternative_features.py
```

---

## **Prompt P-016: Data Quality Management System**

**Title:** Implement comprehensive data quality monitoring and validation

### Context
- **Current State:** Feature engineering pipeline complete (P-015)
- **Target State:** Robust data quality assurance for ML models
- **Phase Goal:** Ensure high-quality features for ML training

**Technical Context**: Reference @SPECIFICATIONS.md Section 4.4 "Data Quality Management".

### Dependencies
**Depends On:** P-014 (data pipeline), P-015 (feature engineering)
**Enables:** P-017+ (reliable ML training data)

### Task Details

#### 1. Real-time Validation (`src/data/quality/validation.py`)
Implement comprehensive validation:
- Schema validation for incoming data
- Range checks and business rule validation
- Statistical outlier detection
- Data freshness monitoring
- Cross-source consistency checks

#### 2. Data Cleaning (`src/data/quality/cleaning.py`)
Implement data cleaning pipeline:
- Missing data imputation strategies
- Outlier handling (remove vs adjust)
- Data smoothing for noisy signals
- Duplicate detection and removal
- Data normalization and standardization

#### 3. Quality Monitoring (`src/data/quality/monitoring.py`)
Implement ongoing quality monitoring:
- Data drift detection using statistical tests
- Feature distribution monitoring
- Quality score calculation and trending
- Automated alerting on quality degradation
- Quality reports and dashboards

### Directory Structure to Create
```
src/data/quality/
├── __init__.py
├── validation.py
├── cleaning.py
└── monitoring.py
```

---

## **Prompt P-016A: Utility Framework and Helper Functions**

**Title:** Implement comprehensive utility framework with decorators, helpers, validators, and formatters

### Context
- **Current State:** Data quality management operational (P-016)
- **Target State:** Reusable utility functions available across all components
- **Phase Goal:** Common utilities before ML infrastructure implementation

**Technical Context**: Reference @SPECIFICATIONS.md Section 15 project structure utils module. Essential for code reusability.

### Dependencies
**Depends On:** P-001 (core types), P-002A (error handling)
**Enables:** P-017+ (ML models), all components benefit from utilities

### Task Details

#### 1. Common Decorators (`src/utils/decorators.py`)
Implement reusable decorators for cross-cutting concerns:
- **Performance Monitoring**: `@time_execution`, `@memory_usage`, `@cpu_usage`
- **Error Handling**: `@retry`, `@circuit_breaker`, `@timeout`
- **Caching**: `@cache_result`, `@redis_cache`, `@ttl_cache`
- **Logging**: `@log_calls`, `@log_performance`, `@log_errors`
- **Validation**: `@validate_input`, `@validate_output`, `@type_check`
- **Rate Limiting**: `@rate_limit`, `@api_throttle`

#### 2. Helper Functions (`src/utils/helpers.py`)
Implement common utility functions:
- **Mathematical Utilities**: statistical calculations, financial metrics
- **Date/Time Utilities**: timezone handling, trading session detection
- **Data Conversion**: unit conversions, currency conversions
- **File Operations**: safe file I/O, configuration loading
- **Network Utilities**: connection testing, latency measurement
- **String Utilities**: parsing, formatting, sanitization

#### 3. Validation Utilities (`src/utils/validators.py`)
Implement comprehensive validation functions:
- **Financial Data Validation**: price ranges, volume checks, symbol validation
- **Configuration Validation**: parameter bounds, required fields
- **API Input Validation**: request payload validation, security checks
- **Data Type Validation**: type checking, schema validation
- **Business Rule Validation**: trading rules, risk limit validation
- **Exchange Data Validation**: order validation, balance verification

#### 4. Data Formatters (`src/utils/formatters.py`)
Implement data formatting and transformation utilities:
- **Financial Formatting**: currency formatting, percentage display
- **API Response Formatting**: JSON standardization, error formatting
- **Log Formatting**: structured log formatting, correlation IDs
- **Chart Data Formatting**: OHLCV formatting, indicator data
- **Report Formatting**: performance reports, risk reports
- **Export Formatting**: CSV, Excel, PDF export utilities

#### 5. Constants and Enums (`src/utils/constants.py`)
Define system-wide constants and enumerations:
- **Trading Constants**: market hours, settlement times, precision
- **API Constants**: endpoints, rate limits, timeouts
- **Financial Constants**: fee structures, minimum amounts
- **Configuration Constants**: default values, limits
- **Error Constants**: error codes, message templates
- **Market Constants**: symbol mappings, exchange specifications

### Directory Structure to Create
```
src/utils/
├── __init__.py
├── decorators.py
├── helpers.py
├── validators.py
├── formatters.py
└── constants.py
```

### Key Utility Implementations

#### Performance Monitoring Decorator
```python
import time
import functools
from typing import Callable, Any
import structlog

logger = structlog.get_logger()

def time_execution(func: Callable) -> Callable:
    """Decorator to measure and log execution time"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            logger.info(
                "Function executed",
                function=func.__name__,
                execution_time_ms=round(execution_time * 1000, 2),
                success=True
            )
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(
                "Function failed",
                function=func.__name__,
                execution_time_ms=round(execution_time * 1000, 2),
                error=str(e),
                success=False
            )
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            logger.info(
                "Function executed",
                function=func.__name__,
                execution_time_ms=round(execution_time * 1000, 2),
                success=True
            )
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(
                "Function failed",
                function=func.__name__,
                execution_time_ms=round(execution_time * 1000, 2),
                error=str(e),
                success=False
            )
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
```

#### Financial Validation Functions
```python
from decimal import Decimal
from typing import Optional
from src.core.exceptions import ValidationError

def validate_price(price: float, symbol: str) -> Decimal:
    """Validate and normalize price values"""
    if price <= 0:
        raise ValidationError(f"Price must be positive for {symbol}, got {price}")
    
    if price > 1_000_000:  # Sanity check for extremely high prices
        raise ValidationError(f"Price {price} for {symbol} exceeds maximum allowed")
    
    # Convert to Decimal for financial precision
    return Decimal(str(price))

def validate_quantity(quantity: float, symbol: str, min_qty: Optional[float] = None) -> Decimal:
    """Validate trading quantity"""
    if quantity <= 0:
        raise ValidationError(f"Quantity must be positive for {symbol}, got {quantity}")
    
    if min_qty and quantity < min_qty:
        raise ValidationError(f"Quantity {quantity} below minimum {min_qty} for {symbol}")
    
    return Decimal(str(quantity))

def validate_symbol(symbol: str) -> str:
    """Validate trading symbol format"""
    if not symbol or len(symbol) < 3:
        raise ValidationError(f"Invalid symbol format: {symbol}")
    
    if not symbol.replace('/', '').replace('-', '').isalnum():
        raise ValidationError(f"Symbol contains invalid characters: {symbol}")
    
    return symbol.upper()
```

#### Financial Formatting Utilities
```python
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

def format_currency(amount: float, currency: str = "USD", precision: int = 2) -> str:
    """Format amount as currency string"""
    if currency.upper() in ["BTC", "ETH"]:
        precision = 8  # Crypto precision
    elif currency.upper() in ["USDT", "USDC", "USD"]:
        precision = 2  # Fiat precision
    
    formatted = f"{amount:,.{precision}f}"
    return f"{formatted} {currency.upper()}"

def format_percentage(value: float, precision: int = 2) -> str:
    """Format value as percentage"""
    percentage = value * 100
    return f"{percentage:+.{precision}f}%"

def format_pnl(pnl: float, currency: str = "USD") -> str:
    """Format P&L with appropriate color coding info"""
    formatted = format_currency(pnl, currency)
    color = "green" if pnl >= 0 else "red"
    symbol = "+" if pnl >= 0 else ""
    return f"{symbol}{formatted}", color
```

### Acceptance Criteria
- All decorators work with both sync and async functions
- Validators handle edge cases and provide clear error messages
- Formatters support multiple currencies and precision levels
- Helpers provide consistent functionality across components
- Performance overhead <1ms for decorator applications

### Integration Points
- Error handling via P-002A (error handling framework)
- Logging integration with P-001 (structured logging)
- Type definitions from P-001 (core types)
- Used by P-017+ (ML models) and all subsequent components
- Performance monitoring via P-030+ (monitoring system)

### Reverse Integration Required
- **Update P-003+ (Exchanges)**: Apply performance monitoring decorators to all API calls
- **Update P-008+ (Risk Management)**: Use validation utilities for risk parameter validation
- **Update P-011+ (Strategies)**: Apply performance decorators and validation to all strategy methods
- **Update P-014+ (Data Pipeline)**: Use formatting utilities for data display and validation
- **Update P-017+ (ML Models)**: Apply caching decorators and validation utilities
- **Update P-026+ (Web Interface)**: Use formatters for API responses and financial data display

---

## **Prompt P-017: Machine Learning Model Registry and Base Classes**

**Title:** Implement ML model registry, base classes, and model management system

### Context
- **Current State:** High-quality feature pipeline established (P-016)
- **Target State:** ML infrastructure ready for model training and deployment
- **Phase Goal:** Foundation for AI-powered trading strategies

**Technical Context**: Reference @SPECIFICATIONS.md Section 5 "Machine Learning Infrastructure". MLflow integration.

### Dependencies
**Depends On:** P-015 (features), P-016 (data quality), P-001 (types, config), P-002A (error handling), P-016A (utils)
**Enables:** P-018 (training pipeline), P-019 (AI strategies)

### Mandatory Integration Requirements
**CRITICAL**: This prompt MUST integrate with existing components and update P-001 with ML types:

#### Required Imports from Previous Prompts:
```python
# From P-015/P-016 - MANDATORY: Use existing feature pipeline
from src.data.features.calculator import FeatureCalculator
from src.data.quality.validator import DataQualityValidator

# From P-001 - MANDATORY: Use existing types and will add ML types
from src.core.types import (
    MarketData, Signal, Position,
    ModelPrediction, ModelMetadata, ModelStatus  # Will be added to P-001
)
from src.core.exceptions import ModelError, ModelLoadError, ModelInferenceError
from src.core.config import Config

# From P-002A - MANDATORY: Use error handling patterns
from src.error_handling.error_handler import ErrorHandler

# From P-016A - MANDATORY: Use decorators and validators
from src.utils.decorators import time_execution, retry, cache_result
from src.utils.validators import validate_model_input
from src.utils.formatters import format_percentage

# ML Dependencies from @DEPENDENCIES.md
import mlflow
import tensorflow as tf
import torch
import scikit-learn
import xgboost
import optuna
```

#### Required Patterns from @COMMON_PATTERNS.md:
- MANDATORY: Use standard exception handling for model operations
- MANDATORY: Apply @cache_result decorators for model inference
- MANDATORY: Use structured logging for model lifecycle events
- MANDATORY: Implement graceful fallback when models fail

### Task Details

#### 1. Base ML Model Classes (`src/ml/models/base.py`)
Implement abstract model interfaces:
- `BaseMLModel` with fit, predict, validate methods
- Model versioning and serialization
- Performance tracking and metrics collection
- Feature importance and explainability
- Model health monitoring

#### 2. Model Registry (`src/ml/registry/model_store.py`)
Implement MLflow-based model registry:
- Model registration and versioning
- Model deployment and rollback
- Model metadata management
- A/B testing framework
- Model performance comparison

#### 3. Specific Model Types (`src/ml/models/`)
Implement model type implementations:
- Price prediction models (regression)
- Direction classification models
- Volatility forecasting models
- Regime detection models
- Ensemble model frameworks

#### 4. Inference Engine (`src/ml/inference/predictor.py`)
Implement real-time model inference:
- Real-time prediction API with <100ms latency
- Model loading and caching for fast inference
- Feature preprocessing pipeline integration
- Prediction confidence scoring and thresholding
- Batch inference for historical analysis
- Model fallback and error handling

#### 5. Batch Inference System (`src/ml/inference/batch_inference.py`)
Implement batch processing for large-scale predictions:
- Batch prediction scheduling and management
- Parallel processing for multiple models
- Result aggregation and storage
- Performance monitoring and optimization
- Resource management for batch jobs
- Integration with data pipeline for automated processing

### Directory Structure to Create
```
src/ml/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── price_prediction.py
│   ├── direction_classification.py
│   └── volatility_forecasting.py
├── inference/
│   ├── __init__.py
│   ├── predictor.py
│   └── batch_inference.py
└── registry/
    ├── __init__.py
    ├── model_store.py
    └── version_manager.py
```

### Reverse Integration Required
**CRITICAL**: This prompt MUST update P-001 files with ML-specific components:

#### Update P-001 src/core/types.py:
Add these ML-specific types to the existing file:
```python
# Add these to src/core/types.py after existing types
from enum import Enum

class ModelType(Enum):
    PRICE_PREDICTION = "price_prediction"
    DIRECTION_CLASSIFICATION = "direction_classification"
    VOLATILITY_FORECASTING = "volatility_forecasting"
    REGIME_DETECTION = "regime_detection"
    RISK_ASSESSMENT = "risk_assessment"

class ModelStatus(Enum):
    TRAINING = "training"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    TESTING = "testing"

class ModelPrediction(BaseModel):
    model_id: str
    model_version: str
    prediction_type: ModelType
    prediction_value: Union[float, str, Dict[str, float]]
    confidence: float = Field(ge=0.0, le=1.0)
    features_used: List[str]
    timestamp: datetime
    symbol: Optional[str] = None

class ModelMetadata(BaseModel):
    model_id: str
    model_name: str
    model_type: ModelType
    version: str
    status: ModelStatus
    created_at: datetime
    last_updated: datetime
    training_data_period: str
    performance_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    hyperparameters: Dict[str, Any]
    deployment_config: Dict[str, Any]

class ModelPerformance(BaseModel):
    model_id: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    evaluation_period: str
    last_evaluated: datetime
```

#### Update P-001 src/core/config.py:
Add ML configuration:
```python
# Add this to src/core/config.py
class MLConfig(BaseConfig):
    # Model registry
    mlflow_tracking_uri: str = Field("http://localhost:5000", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field("trading-bot", env="MLFLOW_EXPERIMENT_NAME")
    model_registry_backend: str = Field("mlflow", env="MODEL_REGISTRY_BACKEND")
    
    # Training settings
    default_train_test_split: float = Field(0.8, env="ML_TRAIN_TEST_SPLIT")
    cross_validation_folds: int = Field(5, env="ML_CV_FOLDS")
    hyperparameter_trials: int = Field(100, env="ML_HYPERPARAM_TRIALS")
    early_stopping_patience: int = Field(10, env="ML_EARLY_STOPPING_PATIENCE")
    
    # Model deployment
    model_serving_host: str = Field("localhost", env="MODEL_SERVING_HOST")
    model_serving_port: int = Field(8001, env="MODEL_SERVING_PORT")
    inference_timeout_seconds: int = Field(5, env="ML_INFERENCE_TIMEOUT")
    
    # Performance thresholds
    min_accuracy_threshold: float = Field(0.6, env="ML_MIN_ACCURACY")
    min_sharpe_threshold: float = Field(1.0, env="ML_MIN_SHARPE")
    max_drawdown_threshold: float = Field(0.2, env="ML_MAX_DRAWDOWN")
    
    # Feature engineering
    feature_selection_method: str = Field("importance", env="ML_FEATURE_SELECTION")
    max_features: int = Field(100, env="ML_MAX_FEATURES")
    feature_cache_hours: int = Field(24, env="ML_FEATURE_CACHE_HOURS")

# Update the main Config class:
class Config(BaseConfig):
    # ... existing fields ...
    ml: MLConfig = MLConfig()
```

### Integration Validation
After P-017 completion, verify:
```bash
# Test ML configuration loading
python -c "from src.core.config import Config; c = Config(); print(f'ML config: {c.ml.mlflow_tracking_uri}')"

# Test ML types
python -c "from src.core.types import ModelPrediction, ModelMetadata, ModelType; print('ML types imported successfully')"

# Test model registry
python -c "from src.ml.registry.model_store import ModelRegistry; print('ModelRegistry ready')"
```

---

## **Prompt P-018: ML Training Pipeline and Model Management**

**Title:** Implement automated model training, validation, and deployment pipeline

### Context
- **Current State:** ML model registry and base classes ready (P-017)
- **Target State:** Automated ML pipeline for continuous model improvement
- **Phase Goal:** Production-ready ML model lifecycle management

**Technical Context**: Reference @SPECIFICATIONS.md Section 5.2-5.4 ML specifications. Automated retraining and validation.

### Dependencies
**Depends On:** P-017 (model registry), P-015+ (feature pipeline)
**Enables:** P-019 (AI strategies), production ML deployment

### Task Details

#### 1. Training Pipeline (`src/ml/training/trainer.py`)
Implement automated training pipeline:
- Data preparation and feature selection
- Cross-validation for financial time series
- Hyperparameter optimization with Optuna
- Model training with multiple algorithms
- Performance evaluation and validation

#### 2. Model Validation (`src/ml/training/validation.py`)
Implement comprehensive model validation:
- Walk-forward analysis for time series
- Out-of-sample testing
- Statistical significance testing
- Overfitting detection
- Model stability analysis

#### 3. Deployment Pipeline (`src/ml/training/deployment.py`)
Implement model deployment automation:
- Model packaging and containerization
- A/B testing framework
- Gradual rollout capabilities
- Performance monitoring in production
- Automatic rollback on degradation

### Directory Structure to Create
```
src/ml/training/
├── __init__.py
├── trainer.py
├── hyperparameter_tuning.py
├── validation.py
└── deployment.py
```

---

## **Prompt P-019: AI-Powered Trading Strategies Implementation**

**Title:** Implement ML ensemble strategy with real-time inference and fallback mechanisms

### Context
- **Current State:** ML training pipeline operational (P-018)
- **Target State:** Production-ready AI strategies with fallback to static rules
- **Phase Goal:** Advanced AI decision-making with reliability guarantees

**Technical Context**: Reference @SPECIFICATIONS.md Section 3.1.3 "AI/ML Strategies" and Section 5.6 "Model Fallback Strategy".

### Dependencies
**Depends On:** P-017 (ML models), P-018 (training), P-012 (static strategies for fallback)
**Enables:** P-020+ (execution engine), production AI trading

### Task Details

#### 1. ML Ensemble Strategy (`src/strategies/ai/ml_ensemble_strategy.py`)
Implement sophisticated ensemble strategy:
- Multiple model integration (XGBoost, LSTM, Random Forest)
- Confidence-weighted ensemble voting
- Real-time feature calculation and caching
- Prediction confidence thresholding
- Dynamic model weight adjustment

#### 2. Fallback Handler (`src/strategies/ai/fallback_handler.py`)
Implement automatic fallback mechanisms:
- Model failure detection (timeout, low confidence, errors)
- Automatic switch to static mean reversion strategy
- Gradual model re-integration after recovery
- Fallback performance monitoring
- Manual override capabilities

#### 3. Feature Pipeline Integration (`src/strategies/ai/feature_pipeline.py`)
Implement real-time feature engineering:
- Feature calculation with <50ms latency
- Redis-based feature caching (5-minute TTL)
- Missing data handling and interpolation
- Feature drift monitoring
- Dynamic feature selection

### Directory Structure to Create
```
src/strategies/ai/
├── __init__.py
├── ml_ensemble_strategy.py
├── feature_pipeline.py
├── model_ensemble.py
└── fallback_handler.py
```

---

## **Prompt P-020: Order Management and Execution Engine**

**Title:** Implement sophisticated order management and execution algorithms

### Context
- **Current State:** All strategy types implemented (P-012, P-013, P-019)
- **Target State:** Efficient order execution with minimal slippage
- **Phase Goal:** Bridge strategy signals to actual market execution

**Technical Context**: Reference @SPECIFICATIONS.md Section 6 "Execution Engine". Support multiple execution algorithms.

### Dependencies
**Depends On:** P-011+ (strategies), P-003+ (exchanges), P-008+ (risk management), P-001 (types), P-002A (error handling), P-016A (utils)
**Enables:** P-021 (bot instances), P-022+ (bot orchestration), production trading

### Mandatory Integration Requirements
**CRITICAL**: This prompt MUST integrate with all existing components for seamless execution:

#### Required Imports from Previous Prompts:
```python
# From P-011+ - MANDATORY: Use existing strategy interfaces
from src.strategies.base import BaseStrategy
from src.strategies.factory import StrategyFactory

# From P-003+ - MANDATORY: Use existing exchange interfaces
from src.exchanges.base import BaseExchange
from src.exchanges.factory import ExchangeFactory

# From P-008+ - MANDATORY: Use existing risk management
from src.risk_management.base import BaseRiskManager
from src.risk_management.position_sizing import PositionSizer
from src.risk_management.circuit_breakers import CircuitBreakerManager

# From P-010A - MANDATORY: Use capital management
from src.capital_management.capital_allocator import CapitalAllocator

# From P-001 - MANDATORY: Use existing types
from src.core.types import (
    OrderRequest, OrderResponse, Signal, Position,
    TradeResult, ExecutionReport  # Will be added to P-001
)
from src.core.exceptions import (
    ExecutionError, OrderRejectionError, SlippageError,
    RiskManagementError
)
from src.core.config import Config

# From P-002A - MANDATORY: Use error handling and recovery
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import (
    PartialFillRecovery, NetworkDisconnectionRecovery
)

# From P-016A - MANDATORY: Use decorators and validators
from src.utils.decorators import time_execution, retry, circuit_breaker
from src.utils.validators import validate_order, validate_position
from src.utils.formatters import format_currency
```

#### Required Patterns from @COMMON_PATTERNS.md:
- MANDATORY: Use standard exception handling for all execution failures
- MANDATORY: Apply @circuit_breaker decorator to exchange order placement
- MANDATORY: Integrate with P-002A partial fill recovery scenarios
- MANDATORY: Use structured logging for all execution events with trade context

### Task Details

#### 1. Order Manager (`src/execution/order_manager.py`)
Implement comprehensive order management:
- Order lifecycle tracking (pending → submitted → filled → settled)
- Partial fill handling and aggregation
- Order modification and cancellation
- Fill reporting and confirmation
- Order book impact analysis

#### 2. Execution Algorithms (`src/execution/execution_engine.py`)
Implement multiple execution strategies:
- Market orders for immediate execution
- Limit orders with price improvement
- TWAP (Time-Weighted Average Price) algorithm
- VWAP (Volume-Weighted Average Price) algorithm
- Implementation shortfall minimization

#### 3. Slippage Optimizer (`src/execution/slippage_optimizer.py`)
Implement slippage minimization:
- Order size optimization
- Timing optimization based on volatility
- Market impact estimation
- Smart order routing across exchanges
- Adaptive execution based on market conditions

### Directory Structure to Create
```
src/execution/
├── __init__.py
├── order_manager.py
├── execution_engine.py
├── slippage_optimizer.py
└── trade_tracker.py
```

---

## **Prompt P-021: Bot Instance Management System**

**Title:** Implement individual bot instance lifecycle management and isolation

### Context
- **Current State:** Order execution engine operational (P-020)
- **Target State:** Independent bot instances with resource isolation
- **Phase Goal:** Multi-bot support with individual lifecycle management

**Technical Context**: Reference @SPECIFICATIONS.md Section 7 "Bot Orchestration System". Each bot as separate process.

### Dependencies
**Depends On:** P-020 (execution), P-011+ (strategies), P-008+ (risk management)
**Enables:** P-022 (multi-bot coordination), P-026+ (web interface)

### Task Details

#### 1. Bot Instance Class (`src/bot_management/bot_instance.py`)
Implement individual bot management:
- Bot lifecycle (start, stop, pause, resume)
- Configuration loading and validation
- Strategy instantiation and management
- State persistence and recovery
- Resource monitoring and limits
- Performance tracking and reporting

#### 2. Bot State Manager (`src/bot_management/state_manager.py`)
Implement bot state management:
- State serialization and persistence
- Recovery from unexpected shutdowns
- State validation and consistency checks
- Hot configuration updates
- State backup and restore

#### 3. Resource Manager (`src/bot_management/resource_manager.py`)
Implement resource allocation and monitoring:
- Memory allocation per bot (1-8GB based on strategy)
- CPU core assignment and scheduling
- Network bandwidth allocation
- Storage quota management
- Resource usage monitoring and alerting

### Directory Structure to Create
```
src/bot_management/
├── __init__.py
├── bot_instance.py
├── state_manager.py
└── resource_manager.py
```

---

## **Prompt P-022: Multi-Bot Orchestration and Coordination**

**Title:** Implement bot orchestrator for managing multiple bot instances

### Context
- **Current State:** Individual bot instances functional (P-021)
- **Target State:** Coordinated multi-bot system with resource sharing
- **Phase Goal:** Production-ready bot orchestration

**Technical Context**: Reference @SPECIFICATIONS.md Section 7.2-7.3 for multi-bot coordination and resource management.

### Dependencies
**Depends On:** P-021 (bot instances)
**Enables:** P-024+ (state management), production deployment

### Task Details

#### 1. Bot Orchestrator (`src/bot_management/orchestrator.py`)
Implement central bot coordination:
- Bot discovery and registration
- Health monitoring and restart capabilities
- Load balancing across available resources
- Inter-bot communication and coordination
- System-wide emergency controls

#### 2. Bot Coordinator (`src/bot_management/coordinator.py`)
Implement bot interaction management:
- Position conflict resolution
- Shared resource allocation
- Risk aggregation across bots
- Performance comparison and optimization
- Portfolio rebalancing coordination

### Directory Structure to Create
```
src/bot_management/
├── orchestrator.py
└── coordinator.py
```

---

## **Prompt P-023: Trade Lifecycle and Quality Controls**

**Title:** Implement comprehensive trade lifecycle management and quality monitoring

### Context
- **Current State:** Bot orchestration operational (P-022)
- **Target State:** Complete trade tracking with quality assurance
- **Phase Goal:** Production-ready trade management

**Technical Context**: Reference @SPECIFICATIONS.md Section 6.4 "Trade Quality Controls".

### Dependencies
**Depends On:** P-020 (execution), P-022 (orchestration)
**Enables:** P-030+ (monitoring), production trading

### Task Details

#### 1. Trade Manager (`src/execution/trade_manager.py`)
Implement comprehensive trade management:
- Pre-trade validation and risk checks
- Trade execution monitoring and tracking
- Post-trade analysis and reporting
- Exception handling and recovery
- Audit trail maintenance

#### 2. Quality Controller (`src/execution/quality_controller.py`)
Implement trade quality monitoring:
- Slippage analysis and reporting
- Execution timing quality assessment
- Market impact measurement
- Fill rate optimization
- Performance attribution analysis

### Directory Structure to Create
```
src/execution/
├── trade_manager.py
└── quality_controller.py
```

---

## **Prompt P-024: State Persistence and Recovery System**

**Title:** Implement comprehensive state management with recovery capabilities

### Context
- **Current State:** Trade quality controls established (P-023)
- **Target State:** Robust state persistence with fast recovery
- **Phase Goal:** Production reliability and fault tolerance

**Technical Context**: Reference @SPECIFICATIONS.md Section 8 "State Management & Persistence".

### Dependencies
**Depends On:** P-002 (database), P-021+ (bot management)
**Enables:** P-025 (real-time state), production reliability

### Task Details

#### 1. State Manager (`src/state/state_manager.py`)
Implement comprehensive state management:
- State serialization and versioning
- Atomic state updates with ACID guarantees
- State validation and consistency checks
- State compression and optimization
- State archival and cleanup

#### 2. Recovery System (`src/state/recovery.py`)
Implement automatic recovery mechanisms:
- Crash detection and automatic restart
- State rollback to last consistent state
- Partial state recovery for corrupted data
- Recovery validation and verification
- Recovery performance optimization

#### 3. State Synchronization (`src/state/synchronization.py`)
Implement multi-database state sync:
- PostgreSQL ↔ Redis synchronization
- InfluxDB time series coordination
- Conflict resolution strategies
- Eventual consistency management
- Cross-system state validation

### Directory Structure to Create
```
src/state/
├── __init__.py
├── state_manager.py
├── recovery.py
└── synchronization.py
```

---

## **Prompt P-025: Real-Time State Synchronization**

**Title:** Implement real-time state updates with WebSocket broadcasting

### Context
- **Current State:** State persistence system operational (P-024)
- **Target State:** Real-time state updates for web interface
- **Phase Goal:** Live data feeds for monitoring and control

**Technical Context**: Reference @SPECIFICATIONS.md Section 8.1 state architecture with Redis real-time state.

### Dependencies
**Depends On:** P-024 (state management)
**Enables:** P-026+ (web interface), P-030+ (monitoring)

### Task Details

#### 1. Real-Time State Publisher (`src/state/real_time_publisher.py`)
Implement state change broadcasting:
- State change detection and filtering
- WebSocket message publishing
- Client subscription management
- Message queuing for offline clients
- Bandwidth optimization and compression

#### 2. State Event Manager (`src/state/event_manager.py`)
Implement event-driven state updates:
- Event sourcing for state changes
- Event replay capabilities
- Event filtering and routing
- Event persistence and recovery
- Event-driven workflow triggers

### Directory Structure to Create
```
src/state/
├── real_time_publisher.py
└── event_manager.py
```

---

## **Prompt P-026: FastAPI Backend Foundation**

**Title:** Implement FastAPI backend with authentication and core API structure

### Context
- **Current State:** Real-time state system operational (P-025)
- **Target State:** Web API foundation for frontend interaction
- **Phase Goal:** Secure API access to trading system

**Technical Context**: Reference @SPECIFICATIONS.md Section 9.2 "Backend API" and Section 9.4 "Security Features".

### Dependencies
**Depends On:** P-001 (config), P-002 (database), P-025 (real-time state), P-002A (error handling), P-016A (utils)
**Enables:** P-027 (API endpoints), P-028 (WebSocket), P-029 (frontend)

### Mandatory Integration Requirements
**CRITICAL**: This prompt MUST integrate with existing components and update P-001 with web config:

#### Required Imports from Previous Prompts:
```python
# From P-001 - MANDATORY: Use existing types and config
from src.core.types import Position, Signal, MarketData
from src.core.exceptions import SecurityError, AuthenticationError, ValidationError
from src.core.config import Config

# From P-002 - MANDATORY: Use existing database models
from src.database.models import User, BotInstance, Trade
from src.database.connection import get_session

# From P-025 - MANDATORY: Use real-time state system
from src.state_management.real_time_state import StateManager

# From P-002A - MANDATORY: Use error handling patterns
from src.error_handling.error_handler import ErrorHandler

# From P-016A - MANDATORY: Use formatters and validators
from src.utils.formatters import format_api_response, format_currency
from src.utils.validators import validate_user_input
from src.utils.decorators import time_execution

# FastAPI dependencies from @DEPENDENCIES.md
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
```

#### Required Patterns from @COMMON_PATTERNS.md:
- MANDATORY: Use standard API response patterns for all endpoints
- MANDATORY: Apply authentication middleware from security patterns
- MANDATORY: Use P-016A formatters for all financial data in responses
- MANDATORY: Implement standard exception handling for API errors

### Task Details

#### 1. FastAPI Application (`src/web_interface/app.py`)
Implement main FastAPI application:
- Application initialization and configuration
- Middleware setup (CORS, authentication, logging)
- Exception handling and error responses
- Health check endpoints
- API documentation with OpenAPI

#### 2. Authentication System (`src/web_interface/security/auth.py`)
Implement comprehensive authentication:
- JWT-based authentication with secure token generation
- Multi-Factor Authentication (2FA) with TOTP support
- Token validation, refresh, and revocation
- Role-based access control (Admin, Trader, Viewer)
- Session management with secure timeout
- Password hashing with bcrypt and salt
- Account lockout protection and audit logging

#### 3. Encryption Manager (`src/web_interface/security/encryption.py`)
Implement data encryption and key management:
- AES-256 encryption for sensitive data at rest
- TLS 1.3 enforcement for all data in transit
- Secure key generation, storage, and rotation
- API key encryption and management
- Database field-level encryption for credentials
- HSM integration for production environments

#### 4. Security Middleware (`src/web_interface/security/permissions.py`)
Implement comprehensive security controls:
- Advanced API rate limiting with sliding windows
- Input validation and sanitization (XSS, SQL injection prevention)
- CSRF protection with secure token generation
- Security headers (HSTS, CSP, X-Frame-Options)
- IP whitelisting and geographic restrictions
- Request/response logging with sensitive data masking

#### 5. Network Security (`src/web_interface/security/network.py`)
Implement network-level security:
- DDoS protection with rate limiting and traffic analysis
- VPN integration for secure remote access
- Firewall rules management and validation
- Connection monitoring and anomaly detection
- SSL/TLS certificate management and auto-renewal
- Network access control and port security

### Directory Structure to Create
```
src/web_interface/
├── __init__.py
├── app.py
├── models/
│   ├── __init__.py
│   ├── requests.py
│   └── responses.py
└── security/
    ├── __init__.py
    ├── auth.py
    ├── encryption.py
    ├── permissions.py
    └── network.py
```

### Reverse Integration Required
**CRITICAL**: This prompt MUST update P-001 with web interface configuration:

#### Update P-001 src/core/config.py:
Add web interface configuration:
```python
# Add this to src/core/config.py
class WebConfig(BaseConfig):
    # Server settings
    host: str = Field("0.0.0.0", env="WEB_HOST")
    port: int = Field(8000, env="WEB_PORT")
    debug: bool = Field(False, env="WEB_DEBUG")
    reload: bool = Field(False, env="WEB_RELOAD")
    
    # CORS settings
    cors_origins: List[str] = Field(["http://localhost:3000"], env="WEB_CORS_ORIGINS")
    cors_credentials: bool = Field(True, env="WEB_CORS_CREDENTIALS")
    cors_methods: List[str] = Field(["GET", "POST", "PUT", "DELETE"], env="WEB_CORS_METHODS")
    
    # Authentication
    auth_secret_key: str = Field(..., env="WEB_AUTH_SECRET_KEY")
    auth_algorithm: str = Field("HS256", env="WEB_AUTH_ALGORITHM")
    auth_expire_minutes: int = Field(30, env="WEB_AUTH_EXPIRE_MINUTES")
    
    # API settings
    api_prefix: str = Field("/api/v1", env="WEB_API_PREFIX")
    docs_url: str = Field("/docs", env="WEB_DOCS_URL")
    redoc_url: str = Field("/redoc", env="WEB_REDOC_URL")
    openapi_url: str = Field("/openapi.json", env="WEB_OPENAPI_URL")
    
    # Rate limiting
    rate_limit_requests: int = Field(100, env="WEB_RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(60, env="WEB_RATE_LIMIT_WINDOW")  # seconds
    
    # Security
    enable_https: bool = Field(True, env="WEB_ENABLE_HTTPS")
    ssl_cert_file: Optional[str] = Field(None, env="WEB_SSL_CERT_FILE")
    ssl_key_file: Optional[str] = Field(None, env="WEB_SSL_KEY_FILE")

# Update the main Config class:
class Config(BaseConfig):
    # ... existing fields ...
    web: WebConfig = WebConfig()
```

### Integration Validation
After P-026 completion, verify:
```bash
# Test web configuration loading
python -c "from src.core.config import Config; c = Config(); print(f'Web config: {c.web.host}:{c.web.port}')"

# Test FastAPI app creation
python -c "from src.web_interface.app import create_app; app = create_app(); print('FastAPI app created successfully')"

# Test authentication system
python -c "from src.web_interface.security.auth import AuthManager; print('AuthManager ready')"
```

---

## **Prompt P-027: Core API Endpoints Implementation**

**Title:** Implement REST API endpoints for bot management, portfolio, and trading

### Context
- **Current State:** FastAPI backend foundation ready (P-026)
- **Target State:** Complete REST API for web interface interaction
- **Phase Goal:** Full API coverage for trading operations

**Technical Context**: Reference @SPECIFICATIONS.md Section 17.1 "REST API Endpoints".

### Dependencies
**Depends On:** P-026 (FastAPI backend), P-021+ (bot management), P-011+ (strategies), P-008+ (risk), P-017+ (ML), P-001 (types), P-016A (utils)
**Enables:** P-028 (WebSocket), P-029 (frontend integration)

### Mandatory Integration Requirements
**CRITICAL**: This prompt MUST integrate with all existing components to provide complete API coverage:

#### Required Imports from Previous Prompts:
```python
# From P-026 - MANDATORY: Use existing FastAPI foundation
from src.web_interface.app import app
from src.web_interface.security.auth import get_current_user, AuthManager
from src.web_interface.middleware.auth import authentication_middleware

# From P-021+ - MANDATORY: Use bot management
from src.orchestration.bot_manager import BotManager
from src.orchestration.bot_instance import BotInstance

# From P-011+ - MANDATORY: Use strategy management
from src.strategies.factory import StrategyFactory
from src.strategies.base import BaseStrategy

# From P-008+ - MANDATORY: Use risk management
from src.risk_management.base import BaseRiskManager
from src.risk_management.circuit_breakers import CircuitBreakerManager

# From P-017+ - MANDATORY: Use ML model management
from src.ml.registry.model_store import ModelRegistry
from src.ml.models.base import BaseModel

# From P-010A - MANDATORY: Use capital management
from src.capital_management.capital_allocator import CapitalAllocator

# From P-001 - MANDATORY: Use existing types
from src.core.types import (
    BotConfig, Position, Signal, MarketData,
    ModelMetadata, RiskMetrics, CapitalAllocation
)
from src.core.exceptions import ValidationError, SecurityError

# From P-016A - MANDATORY: Use formatters for API responses
from src.utils.formatters import (
    format_api_response, format_currency, format_percentage,
    format_financial_data
)
from src.utils.validators import validate_user_input, validate_bot_config

# FastAPI dependencies
from fastapi import APIRouter, HTTPException, Depends, Query, Path
from pydantic import BaseModel as PydanticModel
```

#### Required Patterns from @COMMON_PATTERNS.md:
- MANDATORY: Use standard API response patterns for ALL endpoints
- MANDATORY: Apply authentication decorators to protected endpoints
- MANDATORY: Use P-016A formatters for all financial data in responses
- MANDATORY: Implement comprehensive input validation using P-016A validators

### Task Details

#### 1. Bot Management API (`src/web_interface/routers/bots.py`)
Implement bot control endpoints:
- GET `/api/v1/bots` - List all bots with status
- POST `/api/v1/bots` - Create new bot instance
- PUT `/api/v1/bots/{bot_id}/start` - Start bot
- PUT `/api/v1/bots/{bot_id}/stop` - Stop bot
- PUT `/api/v1/bots/{bot_id}/pause` - Pause bot execution
- PUT `/api/v1/bots/{bot_id}/resume` - Resume paused bot
- PUT `/api/v1/bots/{bot_id}/config` - Update configuration
- DELETE `/api/v1/bots/{bot_id}` - Delete bot instance

#### 2. Portfolio API (`src/web_interface/routers/portfolio.py`)
Implement portfolio management endpoints:
- GET `/api/v1/portfolio/overview` - Portfolio summary
- GET `/api/v1/portfolio/positions` - Current positions
- GET `/api/v1/portfolio/history` - Historical performance
- GET `/api/v1/portfolio/balances` - Account balances
- POST `/api/v1/portfolio/orders` - Manual order placement

#### 3. Strategy API (`src/web_interface/routers/strategies.py`)
Implement strategy management endpoints:
- GET `/api/v1/strategies` - Available strategies
- GET `/api/v1/strategies/{strategy_name}/config` - Strategy configuration
- POST `/api/v1/strategies/{strategy_name}/backtest` - Run backtest
- GET `/api/v1/strategies/{strategy_name}/performance` - Strategy metrics

#### 4. Risk Management API (`src/web_interface/routers/risk.py`)
Implement risk monitoring endpoints:
- GET `/api/v1/risk/overview` - Risk metrics summary
- GET `/api/v1/risk/limits` - Risk limits and usage
- POST `/api/v1/risk/limits` - Update risk limits
- GET `/api/v1/risk/alerts` - Active risk alerts

#### 5. Model Management API (`src/web_interface/routers/models.py`)
Implement ML model management endpoints:
- GET `/api/v1/models` - List all models with metadata
- GET `/api/v1/models/{model_id}` - Get specific model details
- POST `/api/v1/models/{model_id}/retrain` - Trigger model retraining
- GET `/api/v1/models/{model_id}/performance` - Model performance metrics
- PUT `/api/v1/models/{model_id}/activate` - Activate model version
- PUT `/api/v1/models/{model_id}/deactivate` - Deactivate model
- GET `/api/v1/models/{model_id}/predictions` - Recent predictions
- DELETE `/api/v1/models/{model_id}/versions/{version}` - Delete model version

### Directory Structure to Create
```
src/web_interface/routers/
├── __init__.py
├── auth.py
├── bots.py
├── portfolio.py
├── strategies.py
├── risk.py
└── models.py
```

---

## **Prompt P-028: WebSocket Real-Time Updates**

**Title:** Implement WebSocket handlers for real-time data streaming

### Context
- **Current State:** REST API endpoints complete (P-027)
- **Target State:** Real-time data feeds for live monitoring
- **Phase Goal:** Live updates for web interface

**Technical Context**: Reference @SPECIFICATIONS.md Section 17.2 "WebSocket API".

### Dependencies
**Depends On:** P-025 (real-time state), P-027 (API endpoints)
**Enables:** P-029 (frontend real-time features)

### Task Details

#### 1. Market Data WebSocket (`src/web_interface/websockets/market_data.py`)
Implement real-time market data streaming:
- Symbol subscription management
- Price and volume updates
- Order book depth streaming
- Trade execution notifications
- Connection management and heartbeat

#### 2. Bot Status WebSocket (`src/web_interface/websockets/bot_status.py`)
Implement bot status streaming:
- Bot state change notifications
- Performance metric updates
- Alert and notification streaming
- Log message streaming
- Health status updates

#### 3. Portfolio WebSocket (`src/web_interface/websockets/portfolio.py`)
Implement portfolio update streaming:
- Position change notifications
- P&L updates
- Balance change notifications
- Trade execution confirmations
- Risk metric updates

### Directory Structure to Create
```
src/web_interface/websockets/
├── __init__.py
├── market_data.py
├── bot_status.py
└── portfolio.py
```

---

## **Prompt P-029: React Frontend Application**

**Title:** Implement React frontend with real-time dashboard and bot management

### Context
- **Current State:** Backend API and WebSocket ready (P-027, P-028)
- **Target State:** Complete web interface for trading system control
- **Phase Goal:** User-friendly interface for system management

**Technical Context**: Reference @SPECIFICATIONS.md Section 9.1 "Frontend Architecture" and Section 9.3 "Core Pages".

### Dependencies
**Depends On:** P-027 (API), P-028 (WebSocket)
**Enables:** Complete web interface, production readiness

### Task Details

#### 1. React Application Setup (`frontend/src/`)
Set up React application structure:
- TypeScript configuration with strict mode
- Material-UI component library integration
- Redux store for state management
- WebSocket client integration
- Chart.js for data visualization

#### 2. Core Pages Implementation
Implement main application pages:
- **Dashboard**: Overview metrics, recent activity, quick actions
- **Bot Management**: Bot list, configuration, start/stop controls
- **Portfolio**: Positions, balances, performance charts
- **Strategy Center**: Available strategies, backtesting interface
- **Risk Dashboard**: Risk metrics, alerts, circuit breaker status

#### 3. Real-Time Components
Implement WebSocket-driven components:
- Live price charts with candlestick visualization
- Real-time P&L tracking
- Live order book display
- Bot status indicators
- Alert notification system

### Directory Structure to Create
```
frontend/
├── package.json
├── tsconfig.json
├── webpack.config.js
├── src/
│   ├── index.tsx
│   ├── App.tsx
│   ├── components/
│   │   ├── Dashboard/
│   │   ├── BotManagement/
│   │   ├── Portfolio/
│   │   ├── Charts/
│   │   └── Common/
│   ├── pages/
│   │   ├── Dashboard.tsx
│   │   ├── BotManagement.tsx
│   │   ├── Portfolio.tsx
│   │   └── Strategies.tsx
│   ├── services/
│   │   ├── api.ts
│   │   ├── websocket.ts
│   │   └── auth.ts
│   └── store/
│       ├── index.ts
│       └── slices/
└── public/
    ├── index.html
    └── manifest.json
```

---

## **Prompt P-030: Monitoring Infrastructure Setup**

**Title:** Implement Prometheus metrics collection and Grafana dashboards

### Context
- **Current State:** Web interface complete (P-029)
- **Target State:** Comprehensive monitoring and alerting system
- **Phase Goal:** Production monitoring and observability

**Technical Context**: Reference @SPECIFICATIONS.md Section 10 "Monitoring & Alerting System".

### Dependencies
**Depends On:** All core components (P-001 through P-029)
**Enables:** P-031 (alerting), production operations

### Task Details

#### 1. Metrics Collection (`src/monitoring/metrics.py`)
Implement Prometheus metrics:
- Trading metrics (trades, P&L, slippage)
- System metrics (CPU, memory, latency)
- Risk metrics (VaR, drawdown, exposure)
- Model performance metrics (accuracy, drift)
- API performance metrics (response time, error rate)

#### 2. Health Monitoring (`src/monitoring/health_check.py`)
Implement health check system:
- Component health validation
- Dependency health checks
- Performance benchmark validation
- Resource utilization monitoring
- Service availability tracking

#### 3. Performance Monitoring (`src/monitoring/performance.py`)
Implement performance tracking:
- Latency measurement and tracking
- Throughput monitoring
- Resource efficiency metrics
- Bottleneck identification
- Performance trend analysis

### Directory Structure to Create
```
src/monitoring/
├── __init__.py
├── metrics.py
├── health_check.py
├── performance.py
└── dashboards/
    ├── trading_dashboard.json
    ├── system_dashboard.json
    └── risk_dashboard.json
```

---

## **Prompt P-031: Alerting and Notification System**

**Title:** Implement comprehensive alerting with multiple notification channels

### Context
- **Current State:** Monitoring infrastructure operational (P-030)
- **Target State:** Proactive alerting system with multi-channel notifications
- **Phase Goal:** Early warning system for all operational issues

**Technical Context**: Reference @SPECIFICATIONS.md Section 10.3-10.4 alerting rules and notification channels.

### Dependencies
**Depends On:** P-030 (monitoring)
**Enables:** P-032+ (complete monitoring), production operations

### Task Details

#### 1. Alert Manager (`src/monitoring/alerts.py`)
Implement alert management system:
- Trading alerts (large losses, risk breaches)
- System alerts (failures, resource exhaustion)
- Performance alerts (latency, degradation)
- Security alerts (unauthorized access, API issues)
- Model drift alerts (prediction accuracy decline)

#### 2. Notification Channels (`src/monitoring/notifications/`)
Implement multiple notification methods:
- Discord integration for real-time alerts
- Telegram bot for critical notifications
- Email alerts for weekly reports
- SMS for emergency situations
- Webhook integration for external systems

### Directory Structure to Create
```
src/monitoring/
├── alerts.py
└── notifications/
    ├── __init__.py
    ├── discord.py
    ├── telegram.py
    ├── email.py
    └── webhooks.py
```

---

## **Prompt P-032: Performance Optimization and Monitoring**

**Title:** Implement performance optimization and continuous monitoring

### Context
- **Current State:** Alerting system operational (P-031)
- **Target State:** Optimized system with continuous performance monitoring
- **Phase Goal:** Production-ready performance and scalability

**Technical Context**: Reference @SPECIFICATIONS.md Section 18 "Performance Requirements".

### Dependencies
**Depends On:** P-030 (monitoring), P-031 (alerting)
**Enables:** P-033+ (testing), production scalability

### Task Details

#### 1. Performance Optimizer (`src/monitoring/optimizer.py`)
Implement automatic performance optimization:
- Database query optimization
- Memory usage optimization
- CPU utilization balancing
- Network bandwidth management
- Cache optimization strategies

#### 2. Scalability Monitor (`src/monitoring/scalability.py`)
Implement scalability monitoring:
- Load testing and capacity planning
- Resource utilization tracking
- Bottleneck identification and resolution
- Auto-scaling recommendations
- Performance regression detection

### Directory Structure to Create
```
src/monitoring/
├── optimizer.py
└── scalability.py
```

---

## **Prompt P-033: Comprehensive Testing Framework**

**Title:** Implement complete testing suite with unit, integration, and performance tests

### Context
- **Current State:** Monitoring and optimization complete (P-032)
- **Target State:** Comprehensive test coverage ensuring system reliability
- **Phase Goal:** Production-ready quality assurance

**Technical Context**: Reference @SPECIFICATIONS.md Section 13 "Testing Strategy".

### Dependencies
**Depends On:** All core components (P-001 through P-032), @COMMON_PATTERNS.md, @DEPENDENCIES.md
**Enables:** P-034+ (deployment), production confidence

### Mandatory Integration Requirements
**CRITICAL**: This prompt MUST test integration between ALL components systematically:

#### Required Test Imports from All Previous Prompts:
```python
# Testing framework from @DEPENDENCIES.md
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pytest_benchmark import benchmark
import factory_boy

# Core component imports for testing
from src.core.types import Signal, MarketData, Position, OrderRequest
from src.core.exceptions import ExchangeError, RiskManagementError, ValidationError
from src.core.config import Config

# Component imports for integration testing
from src.strategies.base import BaseStrategy
from src.strategies.factory import StrategyFactory
from src.exchanges.base import BaseExchange
from src.risk_management.base import BaseRiskManager
from src.execution.order_manager import OrderManager
from src.ml.models.base import BaseModel
from src.web_interface.app import create_app
from src.orchestration.bot_manager import BotManager

# Test utilities and fixtures
from src.utils.validators import validate_signal, validate_order
from src.utils.formatters import format_currency
```

#### Required Integration Test Patterns:
- MANDATORY: Test all component interfaces work together (strategy → risk → execution → exchange)
- MANDATORY: Test database integration across all components
- MANDATORY: Test API endpoints integration with backend services
- MANDATORY: Test ML model integration with strategies
- MANDATORY: Test error handling propagation through entire system
- MANDATORY: Test configuration loading for all components
- MANDATORY: Validate reverse integration points are working (P-001 types used everywhere)

### Task Details

#### 1. Unit Test Suite (`tests/unit/`)
Implement comprehensive unit tests:
- Strategy logic testing with 90%+ coverage
- Risk management function testing
- Data processing pipeline testing
- ML model inference testing
- API endpoint testing

#### 2. Integration Test Suite (`tests/integration/`)
Implement integration tests:
- Exchange API integration testing
- Database integration testing
- WebSocket communication testing
- End-to-end trading workflow testing
- Multi-component interaction testing

#### 3. Performance Test Suite (`tests/performance/`)
Implement performance benchmarks:
- Latency testing for all critical paths
- Throughput testing under load
- Memory leak detection
- Stress testing for system limits
- Load testing for concurrent operations

### Directory Structure to Create
```
tests/
├── __init__.py
├── conftest.py
├── unit/
│   ├── test_strategies.py
│   ├── test_risk_management.py
│   ├── test_data_pipeline.py
│   ├── test_ml_models.py
│   └── test_exchanges.py
├── integration/
│   ├── test_trading_workflow.py
│   ├── test_database_integration.py
│   ├── test_api_integration.py
│   └── test_websocket_integration.py
├── performance/
│   ├── test_latency.py
│   ├── test_throughput.py
│   ├── test_memory_usage.py
│   └── test_load_capacity.py
└── fixtures/
    ├── market_data.json
    ├── trade_data.json
    └── config_data.yaml
```

---

## **Prompt P-034: Docker Containerization and Multi-Environment Support**

**Title:** Implement Docker containerization with development, staging, and production environments

### Context
- **Current State:** Testing framework complete (P-033)
- **Target State:** Containerized application with multi-environment support
- **Phase Goal:** Deployment-ready containerization

**Technical Context**: Reference @SPECIFICATIONS.md Section 14 "Docker Containerization".

### Dependencies
**Depends On:** Complete application (P-001 through P-033)
**Enables:** P-035 (CI/CD), P-036 (production deployment)

### Task Details

#### 1. Application Containers (`docker/`)
Create multi-stage Docker containers:
- Main application container with Python 3.11
- Web interface container with Node.js build
- Database containers (PostgreSQL, Redis, InfluxDB)
- Monitoring containers (Prometheus, Grafana)
- ML service containers (MLflow, Jupyter)

#### 2. Docker Compose Configurations
Implement environment-specific compositions:
- `docker-compose.dev.yml` for development
- `docker-compose.staging.yml` for staging
- `docker-compose.prod.yml` for production
- Service orchestration and networking
- Volume management and persistence

#### 3. Container Optimization
Implement production optimizations:
- Multi-stage builds for minimal image size
- Security hardening and non-root users
- Health checks and restart policies
- Resource limits and reservations
- Secrets management and environment variables

### Directory Structure to Create
```
docker/
├── Dockerfile
├── Dockerfile.dev
├── Dockerfile.web
├── docker-compose.yml
├── docker-compose.dev.yml
├── docker-compose.staging.yml
├── docker-compose.prod.yml
├── services/
│   ├── postgresql/
│   ├── redis/
│   ├── prometheus/
│   └── grafana/
└── scripts/
    ├── build.sh
    ├── deploy.sh
    └── health-check.sh
```

---

## **Prompt P-035: CI/CD Pipeline Implementation**

**Title:** Implement GitHub Actions CI/CD pipeline with automated testing and deployment

### Context
- **Current State:** Containerized application ready (P-034)
- **Target State:** Automated CI/CD pipeline for continuous integration and deployment
- **Phase Goal:** Production-ready deployment automation

**Technical Context**: Reference @SPECIFICATIONS.md Section 20.2 "CI/CD Pipeline".

### Dependencies
**Depends On:** P-033 (testing), P-034 (containerization)
**Enables:** P-036 (production deployment), continuous delivery

### Task Details

#### 1. GitHub Actions Workflows (`.github/workflows/`)
Implement CI/CD automation:
- Code quality checks (linting, type checking)
- Automated testing (unit, integration, performance)
- Security scanning (dependency vulnerabilities, code analysis)
- Container building and registry publishing
- Automated deployment to staging and production

#### 2. Quality Gates and Validation
Implement deployment safeguards:
- Test coverage requirements (>90%)
- Performance benchmark validation
- Security scan requirements
- Manual approval for production
- Rollback capabilities

#### 3. Environment Management
Implement environment-specific deployments:
- Feature branch deployments to development
- Pull request validation environments
- Automated staging deployments
- Production deployment with approval gates
- Environment health validation

### Directory Structure to Create
```
.github/
├── workflows/
│   ├── ci.yml
│   ├── cd.yml
│   ├── security.yml
│   └── performance.yml
├── ISSUE_TEMPLATE/
├── PULL_REQUEST_TEMPLATE.md
└── CODE_OF_CONDUCT.md
```

---

## **Prompt P-036: Production Deployment and Operations**

**Title:** Implement production deployment with monitoring, backup, and disaster recovery

### Context
- **Current State:** CI/CD pipeline operational (P-035)
- **Target State:** Production-ready deployment with full operational capabilities
- **Phase Goal:** Complete production trading system

**Technical Context**: Reference @SPECIFICATIONS.md Section 20.1 "Environment Setup" production configuration.

### Dependencies
**Depends On:** P-035 (CI/CD), all previous components
**Enables:** Live trading operations

### Task Details

#### 1. Production Configuration (`config/environments/production.yaml`)
Implement production-optimized configuration:
- Security hardening (encryption, authentication)
- Performance optimization (connection pooling, caching)
- Monitoring and alerting configuration
- Backup and disaster recovery settings
- Resource allocation and scaling parameters

#### 2. Operational Scripts (`scripts/`)
Implement operational automation:
- Deployment and rollback scripts
- Backup and restore procedures
- Health check and monitoring scripts
- Log management and rotation
- Database maintenance and optimization

#### 3. Disaster Recovery (`scripts/disaster_recovery/`)
Implement comprehensive disaster recovery:
- Data backup strategies (automated daily backups)
- System recovery procedures
- Failover mechanisms for critical components
- Data integrity validation
- Recovery time optimization

#### 4. Production Monitoring Setup
Implement production monitoring:
- Application performance monitoring (APM)
- Infrastructure monitoring and alerting
- Security monitoring and audit logging
- Business metrics tracking
- SLA monitoring and reporting

### Directory Structure to Create
```
scripts/
├── deployment/
│   ├── deploy.sh
│   ├── rollback.sh
│   └── health_check.sh
├── backup/
│   ├── backup_database.sh
│   ├── backup_models.sh
│   └── restore.sh
├── disaster_recovery/
│   ├── failover.sh
│   ├── recovery_plan.md
│   └── test_recovery.sh
└── monitoring/
    ├── setup_monitoring.sh
    ├── dashboard_setup.sh
    └── alert_config.sh
```

### Production Deployment Checklist
- [ ] Security configuration validated
- [ ] Performance benchmarks met
- [ ] Monitoring and alerting operational
- [ ] Backup procedures tested
- [ ] Disaster recovery plan validated
- [ ] Load testing completed
- [ ] Security audit passed
- [ ] Documentation complete
- [ ] Team training completed
- [ ] Go-live approval obtained

---

## **Prompt P-037: Documentation and Operational Scripts**

**Title:** Implement comprehensive documentation structure and operational scripts

### Context
- **Current State:** Production deployment ready (P-036)
- **Target State:** Complete documentation and operational scripts for maintenance
- **Phase Goal:** Production-ready documentation and automation scripts

**Technical Context**: Reference @SPECIFICATIONS.md Section 15 project structure documentation and scripts directories.

### Dependencies
**Depends On:** P-036 (production deployment)
**Enables:** Complete production-ready system

### Task Details

#### 1. Core Documentation (`docs/`)
Implement essential project documentation:
- **README.md**: Project overview, quick start, key features
- **INSTALLATION.md**: Step-by-step installation instructions
- **CONFIGURATION.md**: Complete configuration guide
- **API.md**: REST API and WebSocket documentation
- **DEPLOYMENT.md**: Production deployment guide

#### 2. Architecture Documentation (`docs/architecture/`)
Implement technical architecture documentation:
- **system_design.md**: System architecture and component relationships
- **data_flow.md**: Data flow diagrams and processing pipelines
- **security.md**: Security architecture and best practices

#### 3. User Documentation (`docs/user_guide/`)
Implement user-facing documentation:
- **getting_started.md**: User onboarding and first steps
- **web_interface.md**: Web interface usage guide
- **troubleshooting.md**: Common issues and solutions

#### 4. Developer Documentation (`docs/developer_guide/`)
Implement developer resources:
- **contributing.md**: Contribution guidelines and code standards
- **coding_standards.md**: Code style and patterns
- **testing_guide.md**: Testing procedures and standards

#### 5. Setup Scripts (`scripts/setup/`)
Implement installation and setup automation:
- **install_dependencies.sh**: Automated dependency installation
- **setup_database.sh**: Database initialization and migration
- **configure_exchanges.sh**: Exchange API setup and validation

#### 6. Maintenance Scripts (`scripts/maintenance/`)
Implement operational maintenance automation:
- **backup_data.sh**: Automated data backup procedures
- **cleanup_old_data.sh**: Data retention and cleanup
- **update_models.sh**: ML model update and deployment

#### 7. Development Scripts (`scripts/development/`)
Implement development workflow automation:
- **run_tests.sh**: Comprehensive test execution
- **lint_code.sh**: Code quality validation
- **generate_docs.sh**: Documentation generation automation

### Directory Structure to Create
```
docs/
├── README.md
├── INSTALLATION.md
├── CONFIGURATION.md
├── API.md
├── DEPLOYMENT.md
├── architecture/
│   ├── system_design.md
│   ├── data_flow.md
│   └── security.md
├── user_guide/
│   ├── getting_started.md
│   ├── web_interface.md
│   └── troubleshooting.md
└── developer_guide/
    ├── contributing.md
    ├── coding_standards.md
    └── testing_guide.md

scripts/
├── setup/
│   ├── install_dependencies.sh
│   ├── setup_database.sh
│   └── configure_exchanges.sh
├── maintenance/
│   ├── backup_data.sh
│   ├── cleanup_old_data.sh
│   └── update_models.sh
└── development/
    ├── run_tests.sh
    ├── lint_code.sh
    └── generate_docs.sh
```

### Key Documentation Content

#### README.md Overview
```markdown
# Trading Bot Suite v2.0

## Overview
Comprehensive Python-based algorithmic trading platform with AI/ML capabilities.

## Key Features
- Multi-exchange support (Binance, OKX, Coinbase Pro)
- Advanced risk management with circuit breakers
- AI-powered trading strategies with fallback mechanisms
- Real-time web interface with monitoring
- Production-ready with Docker containerization

## Quick Start
1. Install dependencies: `./scripts/setup/install_dependencies.sh`
2. Configure environment: `cp .env.example .env`
3. Setup database: `./scripts/setup/setup_database.sh`
4. Run development: `docker-compose -f docker-compose.dev.yml up`

## Documentation
- [Installation Guide](docs/INSTALLATION.md)
- [Configuration](docs/CONFIGURATION.md)
- [API Reference](docs/API.md)
- [User Guide](docs/user_guide/)
- [Developer Guide](docs/developer_guide/)
```

#### Key Script Examples

##### setup_database.sh
```bash
#!/bin/bash
set -e

echo "Setting up trading bot database..."

# Check if PostgreSQL is running
if ! pg_isready -h localhost -p 5432; then
    echo "PostgreSQL is not running. Please start PostgreSQL first."
    exit 1
fi

# Create database if it doesn't exist
createdb trading_bot 2>/dev/null || echo "Database already exists"

# Run migrations
echo "Running database migrations..."
cd "$(dirname "$0")/../.."
python -m alembic upgrade head

# Seed initial data
echo "Seeding initial data..."
python -m src.database.seed_initial_data

echo "Database setup completed successfully!"
```

##### run_tests.sh
```bash
#!/bin/bash
set -e

echo "Running comprehensive test suite..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run linting
echo "Running code quality checks..."
black --check src/
isort --check-only src/
flake8 src/
mypy src/

# Run security scan
echo "Running security scan..."
bandit -r src/

# Run tests with coverage
echo "Running tests..."
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

echo "All tests completed successfully!"
```

### Acceptance Criteria
- All documentation follows consistent structure and format
- Installation scripts work on clean Ubuntu systems
- Maintenance scripts handle edge cases gracefully
- Development scripts support full CI/CD workflow
- API documentation matches actual implementation

### Integration Points
- Documentation references all prompts P-001 through P-036
- Scripts integrate with Docker containers from P-034
- Deployment procedures align with P-036 production setup
- Testing scripts validate work from P-033 testing framework

---

## **Summary and Next Steps**

### Prompt Execution Order
Execute prompts **P-001 through P-037** in sequential order (including all sub-prompts: P-002A, P-010A, P-013A, P-013B, P-013C, P-013D, P-013E, P-016A). Each prompt builds upon previous work and enables subsequent components.

### Critical Dependencies
- **Foundation**: P-001 (core) → P-002 (database) → P-002A (error handling) → P-003 (exchanges)
- **Risk Management**: P-008 → P-009 → P-010 → P-010A (capital management)
- **Strategies**: P-011 → P-012 → P-013 → P-013A (arbitrage) → P-013B (market making) → P-013C (backtesting) → P-019
- **Data Pipeline**: P-014 → P-015 → P-016 → P-016A (utils)
- **ML Pipeline**: P-017 → P-018 → P-019
- **Web Interface**: P-026 → P-027 → P-028 → P-029
- **Production**: P-030 → P-031 → P-032 → P-033 → P-034 → P-035 → P-036 → P-037

### Quality Assurance
- Follow @CODING_STANDARDS.md throughout
- Reference @COMMON_PATTERNS.md for consistent implementation
- Use @DEPENDENCIES.md for exact versions
- Refer to @INTEGRATION_POINTS.md for component interaction
- Validate against @SPECIFICATIONS.md sections as referenced

### Post-Implementation Validation
After completing all prompts:
1. Run complete test suite (P-033)
2. Validate performance requirements (P-032)
3. Verify security implementation (P-026, P-035)
4. Test disaster recovery procedures (P-036)
5. Conduct end-to-end trading workflow test
6. Validate all monitoring and alerting systems

### Production Readiness Criteria
- [ ] All 42 prompts completed successfully (P-001 through P-037 + sub-prompts)
- [ ] Test coverage >90% achieved
- [ ] Performance benchmarks met
- [ ] Security audit passed
- [ ] Monitoring and alerting operational
- [ ] Documentation complete
- [ ] Disaster recovery tested
- [ ] Team training completed
