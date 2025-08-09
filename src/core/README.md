# Core Module

The **Core Module** provides the fundamental framework foundation for the entire trading bot system, serving as the backbone that all other modules depend on and integrate with.

## Module Overview

### What This Module Does
- **Type System**: Defines comprehensive type hierarchy for all trading operations and data structures
- **Configuration Management**: Provides centralized, validated configuration system with environment support
- **Exception Framework**: Establishes unified exception hierarchy for consistent error handling across the system
- **Structured Logging**: Implements correlation-aware, performance-monitoring logging infrastructure
- **Foundation Framework**: Serves as the single source of truth for all core data structures and patterns

### What This Module Does NOT Do
- **Business Logic**: Does not implement trading strategies, risk management, or execution logic
- **External Integrations**: Does not connect to exchanges, databases, or external APIs
- **Data Processing**: Does not perform market data analysis or feature engineering
- **State Management**: Does not maintain application state or session information
- **Network Operations**: Does not handle network communications or protocols

---

## Submodules

The core module consists of individual files rather than submodules, each serving a specific foundational purpose:

### 1. **types.py** - Core Type System
Comprehensive type definitions and data models used throughout the entire system.

### 2. **config.py** - Configuration Management
Centralized configuration system with Pydantic validation and environment support.

### 3. **exceptions.py** - Exception Framework
Complete exception hierarchy for unified error handling across all modules.

### 4. **logging.py** - Structured Logging
Advanced logging system with correlation tracking, performance monitoring, and security features.

---

## File Reference

### **types.py**
Core type definitions and data structures used throughout the system.

#### **Trading Core Enums:**
- `TradingMode(Enum)` - Trading execution environments (LIVE, PAPER, BACKTEST)
- `SignalDirection(Enum)` - Signal directions (BUY, SELL, HOLD)
- `OrderSide(Enum)` - Order sides (BUY, SELL)
- `OrderType(Enum)` - Order types (MARKET, LIMIT, STOP_LOSS, TAKE_PROFIT)
- `OrderStatus(Enum)` - Order status tracking (PENDING, PARTIALLY_FILLED, FILLED, CANCELLED, REJECTED, EXPIRED, UNKNOWN)

#### **Exchange Integration Enums:**
- `ExchangeType(Enum)` - Supported exchanges (BINANCE, OKX, COINBASE)
- `ExchangeStatus(Enum)` - Exchange connection states (ONLINE, OFFLINE, MAINTENANCE)
- `RequestType(Enum)` - API request types (MARKET_DATA, ORDER_PLACEMENT, ORDER_CANCELLATION, BALANCE_QUERY, POSITION_QUERY, HISTORICAL_DATA, WEBSOCKET_CONNECTION)
- `ConnectionType(Enum)` - Connection stream types (TICKER, ORDERBOOK, TRADES, USER_DATA, MARKET_DATA)

#### **Risk Management Enums:**
- `RiskLevel(Enum)` - Risk levels (LOW, MEDIUM, HIGH, CRITICAL)
- `PositionSizeMethod(Enum)` - Position sizing methods (FIXED_PCT, KELLY_CRITERION, VOLATILITY_ADJUSTED, CONFIDENCE_WEIGHTED)
- `CircuitBreakerStatus(Enum)` - Circuit breaker states (CLOSED, OPEN, HALF_OPEN)
- `CircuitBreakerType(Enum)` - Circuit breaker triggers (DAILY_LOSS_LIMIT, DRAWDOWN_LIMIT, VOLATILITY_SPIKE, MODEL_CONFIDENCE, SYSTEM_ERROR_RATE, MANUAL_TRIGGER)
- `MarketRegime(Enum)` - Market regimes (LOW_VOLATILITY, MEDIUM_VOLATILITY, HIGH_VOLATILITY, TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_CORRELATION, LOW_CORRELATION, CRISIS)

#### **Strategy & Capital Management Enums:**
- `AllocationStrategy(Enum)` - Capital allocation strategies (EQUAL_WEIGHT, PERFORMANCE_WEIGHTED, VOLATILITY_WEIGHTED, RISK_PARITY, DYNAMIC)
- `StrategyType(Enum)` - Strategy types (STATIC, DYNAMIC, ARBITRAGE, MARKET_MAKING, EVOLUTIONARY, HYBRID, AI_ML)
- `StrategyStatus(Enum)` - Strategy states (STOPPED, STARTING, RUNNING, PAUSED, ERROR)

#### **Data Quality & Pipeline Enums:**
- `ValidationLevel(Enum)` - Validation severity levels (CRITICAL, HIGH, MEDIUM, LOW)
- `ValidationResult(Enum)` - Validation results (PASS, FAIL, WARNING)
- `QualityLevel(Enum)` - Data quality levels (EXCELLENT, GOOD, FAIR, POOR, CRITICAL)
- `DriftType(Enum)` - Data drift types (CONCEPT_DRIFT, COVARIATE_DRIFT, LABEL_DRIFT, DISTRIBUTION_DRIFT)
- `IngestionMode(Enum)` - Data ingestion modes (REAL_TIME, BATCH, HYBRID)
- `PipelineStatus(Enum)` - Pipeline states (STOPPED, STARTING, RUNNING, PAUSED, ERROR, STOPPING)
- `ProcessingStep(Enum)` - Processing steps (NORMALIZE, ENRICH, AGGREGATE, TRANSFORM, VALIDATE, FILTER)
- `StorageMode(Enum)` - Storage modes (REAL_TIME, BATCH, BUFFER)

#### **Alternative Data Enums:**
- `NewsSentiment(Enum)` - News sentiment levels (VERY_POSITIVE, POSITIVE, NEUTRAL, NEGATIVE, VERY_NEGATIVE)
- `SocialSentiment(Enum)` - Social media sentiment (VERY_BULLISH, BULLISH, NEUTRAL, BEARISH, VERY_BEARISH)

#### **Core Data Models:**

**Trading Models:**
- `Signal(BaseModel)` - Trading signal with direction, confidence, timestamp, symbol, strategy_name, metadata
- `MarketData(BaseModel)` - Market data with symbol, price, volume, timestamp, bid, ask, OHLC data
- `OrderRequest(BaseModel)` - Order request with symbol, side, order_type, quantity, price, stop_price, time_in_force, client_order_id
- `OrderResponse(BaseModel)` - Order response with id, client_order_id, symbol, side, order_type, quantity, price, filled_quantity, status, timestamp
- `Position(BaseModel)` - Position with symbol, quantity, entry_price, current_price, unrealized_pnl, side, timestamp, metadata
- `Trade(BaseModel)` - Trade execution record with id, symbol, side, quantity, price, timestamp, fee, fee_currency

**Exchange Models:**
- `ExchangeInfo(BaseModel)` - Exchange information with name, supported_symbols, rate_limits, features, api_version
- `Ticker(BaseModel)` - Real-time ticker with symbol, bid, ask, last_price, volume_24h, price_change_24h, timestamp
- `OrderBook(BaseModel)` - Order book with symbol, bids, asks, timestamp

**Risk & Capital Models:**
- `RiskMetrics(BaseModel)` - Risk metrics with var_1d, var_5d, expected_shortfall, max_drawdown, sharpe_ratio, current_drawdown, risk_level, timestamp
- `PositionLimits(BaseModel)` - Position limits with max_position_size, max_positions_per_symbol, max_total_positions, max_portfolio_exposure, max_sector_exposure, max_correlation_exposure, max_leverage
- `CircuitBreakerEvent(BaseModel)` - Circuit breaker event with trigger_type, threshold, actual_value, timestamp, description, metadata
- `RegimeChangeEvent(BaseModel)` - Regime change event with from_regime, to_regime, confidence, timestamp, trigger_metrics, description
- `CapitalAllocation(BaseModel)` - Capital allocation with strategy_id, exchange, allocated_amount, utilized_amount, available_amount, allocation_percentage, last_rebalance
- `FundFlow(BaseModel)` - Fund flow tracking with from_strategy, to_strategy, from_exchange, to_exchange, amount, currency, reason, timestamp, converted_amount, exchange_rate
- `CapitalMetrics(BaseModel)` - Capital metrics with total_capital, allocated_capital, available_capital, utilization_rate, allocation_efficiency, rebalance_frequency_hours, emergency_reserve, last_updated, allocation_count

**Strategy Models:**
- `StrategyConfig(BaseModel)` - Strategy configuration with name, strategy_type, enabled, symbols, timeframe, min_confidence, max_positions, position_size_pct, stop_loss_pct, take_profit_pct, parameters
- `StrategyMetrics(BaseModel)` - Strategy metrics with total_trades, winning_trades, losing_trades, total_pnl, win_rate, sharpe_ratio, max_drawdown, last_updated

---

### **config.py**
Configuration management with Pydantic-based settings and validation.

#### **Configuration Classes:**

**DatabaseConfig(BaseConfig):**
- `validate_ports(cls, v: int) -> int` - Validate port numbers are within valid range (1-65535)
- `validate_pool_size(cls, v: int) -> int` - Validate database pool size (1-100)

**SecurityConfig(BaseConfig):**
- `validate_jwt_expire(cls, v: int) -> int` - Validate JWT expiration time (1-1440 minutes)
- `validate_key_length(cls, v: str) -> str` - Validate key lengths for security (min 32 characters)

**ErrorHandlingConfig(BaseConfig):**
- `validate_positive_integers(cls, v: int) -> int` - Validate positive integer values
- `validate_positive_floats(cls, v: float) -> float` - Validate positive float values

**ExchangeConfig(BaseConfig):**
- `validate_api_credentials(cls, v: str) -> str` - Validate API credentials (empty for testing or min 16 chars)

**RiskConfig(BaseConfig):**
- `validate_percentage_fields(cls, v: float) -> float` - Validate percentage fields are between 0 and 1
- `validate_positive_integers(cls, v: int) -> int` - Validate positive integer fields

**CapitalManagementConfig(BaseConfig):**
- `validate_percentage_fields(cls, v: float) -> float` - Validate percentage fields are between 0 and 1
- `validate_positive_integers(cls, v: int) -> int` - Validate positive integer fields
- `validate_positive_floats(cls, v: float) -> float` - Validate positive float fields

**StrategyManagementConfig(BaseConfig):**
- `validate_positive_integers(cls, v: int) -> int` - Validate positive integer fields
- `validate_percentage_fields(cls, v: float) -> float` - Validate percentage fields are between 0 and 1

**Config(BaseConfig) - Master Configuration Class:**
- `validate_environment(cls, v: str) -> str` - Validate environment setting (development, staging, production)
- `generate_schema(self) -> None` - Generate JSON schema for configuration validation
- `from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config'` - Load configuration from YAML file
- `from_yaml_with_env_override(cls, yaml_path: Union[str, Path]) -> 'Config'` - Load configuration with environment variable overrides
- `to_yaml(self, yaml_path: Union[str, Path]) -> None` - Save current configuration to YAML file
- `get_database_url(self) -> str` - Generate PostgreSQL database URL
- `get_async_database_url(self) -> str` - Generate async PostgreSQL database URL
- `get_redis_url(self) -> str` - Generate Redis connection URL with optional password
- `is_production(self) -> bool` - Check if running in production environment
- `is_development(self) -> bool` - Check if running in development environment
- `validate_yaml_config(self, yaml_path: Union[str, Path]) -> bool` - Validate YAML configuration file without loading

---

### **exceptions.py**
Comprehensive exception hierarchy for unified error handling across the system.

#### **Base Exception:**
- `TradingBotError(Exception)` - Base exception for all trading bot errors
  - `__init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None)` - Initialize with message, optional error code and details
  - `__str__(self) -> str` - Return formatted error message with code and details

#### **Exception Categories:**

**Exchange Exceptions:**
- `ExchangeError(TradingBotError)`, `ExchangeConnectionError(ExchangeError)`, `ExchangeRateLimitError(ExchangeError)`, `ExchangeInsufficientFundsError(ExchangeError)`, `ExchangeOrderError(ExchangeError)`, `ExchangeAuthenticationError(ExchangeError)`

**Risk Management Exceptions:**
- `RiskManagementError(TradingBotError)`, `PositionLimitError(RiskManagementError)`, `DrawdownLimitError(RiskManagementError)`, `RiskCalculationError(RiskManagementError)`, `CapitalAllocationError(RiskManagementError)`, `CircuitBreakerTriggeredError(RiskManagementError)`, `EmergencyStopError(RiskManagementError)`

**Data Exceptions:**
- `DataError(TradingBotError)`, `DataValidationError(DataError)`, `DataSourceError(DataError)`, `DataProcessingError(DataError)`, `DataCorruptionError(DataError)`

**Model Exceptions:**
- `ModelError(TradingBotError)`, `ModelLoadError(ModelError)`, `ModelInferenceError(ModelError)`, `ModelDriftError(ModelError)`, `ModelTrainingError(ModelError)`, `ModelValidationError(ModelError)`

**Validation Exceptions:**
- `ValidationError(TradingBotError)`, `ConfigurationError(ValidationError)`, `SchemaValidationError(ValidationError)`, `InputValidationError(ValidationError)`

**Execution Exceptions:**
- `ExecutionError(TradingBotError)`, `OrderRejectionError(ExecutionError)`, `SlippageError(ExecutionError)`, `ExecutionTimeoutError(ExecutionError)`, `ExecutionPartialFillError(ExecutionError)`

**Strategy & Other Exceptions:**
- `StrategyError(TradingBotError)`, `ArbitrageError(StrategyError)`, `DatabaseError(TradingBotError)`, `NetworkError(TradingBotError)`, `CapitalManagementError(TradingBotError)`, `SecurityError(TradingBotError)`, `StateConsistencyError(TradingBotError)`

---

### **logging.py**
Structured logging system with correlation tracking, performance monitoring, and security features.

#### **Classes:**

**CorrelationContext:**
- `__init__(self)` - Initialize correlation context with contextvars support
- `set_correlation_id(self, correlation_id: str) -> None` - Set correlation ID for current context
- `get_correlation_id(self) -> Optional[str]` - Get current correlation ID from context
- `generate_correlation_id(self) -> str` - Generate a new UUID-based correlation ID
- `correlation_context(self, correlation_id: Optional[str] = None)` - Context manager for correlation ID tracking

**SecureLogger:**
- `__init__(self, logger: structlog.BoundLogger)` - Initialize with underlying structured logger
- `info(self, message: str, **kwargs) -> None` - Log info message with sanitized data
- `warning(self, message: str, **kwargs) -> None` - Log warning message with sanitized data
- `error(self, message: str, **kwargs) -> None` - Log error message with sanitized data
- `critical(self, message: str, **kwargs) -> None` - Log critical message with sanitized data

#### **Functions:**

**Setup & Logger Functions:**
- `setup_logging(environment: str = "development", log_level: str = "INFO", log_file: Optional[str] = None, max_bytes: int = 10*1024*1024, backup_count: int = 5, retention_days: int = 30) -> None` - Setup structured logging configuration with rotation and retention
- `get_logger(name: str) -> structlog.BoundLogger` - Get a structured logger instance with correlation ID support
- `get_secure_logger(name: str) -> SecureLogger` - Get a secure logger instance that sanitizes sensitive data

**Performance Decorators:**
- `log_performance(func: Callable) -> Callable` - Decorator to log function performance metrics including execution time and success/failure
- `log_async_performance(func: Callable) -> Callable` - Decorator to log async function performance metrics

---

## Module Dependencies

### Local Modules This Core Module Depends On
**None** - The core module is the foundation and has no dependencies on other local modules.

### External Dependencies
- **Pydantic**: For configuration validation and data models
- **structlog**: For structured logging implementation
- **PyYAML**: For YAML configuration file support
- **contextvars**: For correlation context management
- **Python Standard Library**: datetime, typing, pathlib, enum, functools, etc.

---

## Modules That Should Import From This Core Module

### **All Local Modules** - Every module in the system depends on core:

1. **`src.utils`** - Imports types, exceptions, and logging for utility functions
2. **`src.error_handling`** - Imports exceptions, types, config, and logging for error management
3. **`src.database`** - Imports config, exceptions, types, and logging for database operations
4. **`src.data`** - Imports types, exceptions, config, and logging for data management
5. **`src.exchanges`** - Imports types, exceptions, config, and logging for exchange integration
6. **`src.risk_management`** - Imports types, exceptions, config, and logging for risk management
7. **`src.capital_management`** - Imports types, exceptions, config, and logging for capital management
8. **`src.strategies`** - Imports types, exceptions, config, and logging for strategy implementation

### **Standard Import Pattern:**
```python
from src.core.types import (
    MarketData, Signal, Position, OrderRequest, OrderResponse,
    TradingMode, SignalDirection, OrderSide, OrderType
)
from src.core.exceptions import (
    TradingBotError, ExchangeError, ValidationError, DataError
)
from src.core.config import Config
from src.core.logging import get_logger
```

---

## Integration Notes

- **Central Type System**: All data structures flow through core types ensuring consistency
- **Unified Exception Handling**: All modules use core exceptions for consistent error handling
- **Configuration Cascade**: All modules receive configuration through the core Config class
- **Logging Standards**: All modules use core logging for consistent log formatting and correlation
- **Validation Patterns**: All Pydantic models follow core validation patterns
- **No Circular Dependencies**: Core is the foundation with no dependencies on other local modules

**The core module serves as the unshakeable foundation that enables consistent, type-safe, and well-structured development across the entire trading system.**
