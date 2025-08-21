# Core Module Documentation

## Overview

The Core module provides the foundational components for the T-Bot trading system, including configuration management, error handling, logging, dependency injection, and type definitions. This module is production-ready and serves as the backbone for all other modules in the system.

## Module Structure

```
src/core/
├── __init__.py           # Module exports and initialization
├── config.py             # Configuration management system
├── dependency_injection.py # Dependency injection framework
├── exceptions.py         # Unified exception hierarchy
├── global_error_handler.py # Global error handling
├── logging.py            # Structured logging system
├── memory_manager.py     # Memory optimization utilities
├── service_manager.py    # Service lifecycle management
├── validator_registry.py # Validation framework
├── base/                 # Base component classes
├── caching/              # Caching framework
├── config/               # Configuration submodules
├── performance/          # Performance optimization
└── types/                # Type definitions
```

## Files and Components

### 1. `__init__.py`

Main module initialization file that exports core functionality.

**Exports:**
- Configuration classes
- Exception classes
- Logging utilities
- Type definitions

### 2. `config.py`

Comprehensive configuration management system using Pydantic.

#### Classes

##### `BaseConfig`
Base configuration class with common patterns.

**Configuration:**
- `model_config`: Dict with settings for Pydantic
  - `env_file`: ".env"
  - `case_sensitive`: False
  - `validate_assignment`: True
  - `extra`: "ignore"
  - `populate_by_name`: True

##### `DatabaseConfig(BaseConfig)`
Database configuration for PostgreSQL, Redis, and InfluxDB.

**Parameters:**
- `postgresql_host` (str): PostgreSQL host (default: "localhost")
- `postgresql_port` (int): PostgreSQL port (default: 5432)
- `postgresql_database` (str): Database name (default: "tbot_dev")
- `postgresql_username` (str): Username (default: "tbot")
- `postgresql_password` (str): Password (min 8 chars)
- `postgresql_pool_size` (int): Connection pool size (default: 20)
- `redis_host` (str): Redis host (default: "localhost")
- `redis_port` (int): Redis port (default: 6379)
- `redis_password` (str | None): Redis password (optional)
- `redis_db` (int): Redis database number (default: 0)
- `influxdb_host` (str): InfluxDB host (default: "localhost")
- `influxdb_port` (int): InfluxDB port (default: 8086)
- `influxdb_token` (str): InfluxDB access token
- `influxdb_org` (str): InfluxDB organization (default: "tbot")
- `influxdb_bucket` (str): InfluxDB bucket (default: "trading_data")

**Properties:**
- `postgresql_url` → str: PostgreSQL connection URL
- `redis_url` → str: Redis connection URL

##### `SecurityConfig(BaseConfig)`
Security configuration for authentication and encryption.

**Parameters:**
- `secret_key` (str): JWT secret key (min 32 chars, from env)
- `jwt_algorithm` (str): JWT algorithm (default: "HS256")
- `jwt_expire_minutes` (int): JWT expiration (default: 30)
- `encryption_key` (str): Encryption key (min 32 chars, from env)

##### `ErrorHandlingConfig(BaseConfig)`
Error handling configuration for P-002A framework.

**Parameters:**
- `circuit_breaker_failure_threshold` (int): Failure threshold (default: 5)
- `circuit_breaker_recovery_timeout` (int): Recovery timeout seconds (default: 30)
- `max_retry_attempts` (int): Max retry attempts (default: 3)
- `retry_backoff_factor` (float): Backoff factor (default: 2.0)
- `retry_max_delay` (int): Max retry delay seconds (default: 60)
- `pattern_detection_enabled` (bool): Enable pattern detection (default: True)
- `correlation_analysis_enabled` (bool): Enable correlation analysis (default: True)
- `predictive_alerts_enabled` (bool): Enable predictive alerts (default: True)

##### `ExchangeConfig(BaseConfig)`
Exchange configuration for API credentials and rate limits.

**Parameters:**
- `default_timeout` (int): Default API timeout (default: 30)
- `max_retries` (int): Max retry attempts (default: 3)
- `binance_api_key` (str): Binance API key
- `binance_api_secret` (str): Binance API secret
- `binance_testnet` (bool): Use testnet (default: True)
- `okx_api_key` (str): OKX API key
- `okx_api_secret` (str): OKX API secret
- `okx_passphrase` (str): OKX passphrase
- `okx_sandbox` (bool): Use sandbox (default: True)
- `coinbase_api_key` (str): Coinbase API key
- `coinbase_api_secret` (str): Coinbase API secret
- `coinbase_sandbox` (bool): Use sandbox (default: True)
- `rate_limits` (dict): Exchange-specific rate limits
- `supported_exchanges` (list[str]): List of supported exchanges

**Methods:**
- `get_exchange_credentials(exchange: str)` → dict: Get credentials for exchange
- `get_websocket_config(exchange: str)` → dict: Get WebSocket configuration

##### `RiskConfig(BaseConfig)`
Risk management configuration.

**Parameters:**
- `default_position_size_method` (str): Position sizing method (default: "fixed_percentage")
- `default_position_size_pct` (float): Position size percentage (default: 0.02)
- `max_position_size_pct` (float): Max position size (default: 0.1)
- `max_total_positions` (int): Max total positions (default: 10)
- `max_positions_per_symbol` (int): Max per symbol (default: 1)
- `max_portfolio_exposure` (float): Max exposure (default: 0.95)
- `max_daily_loss_pct` (float): Max daily loss (default: 0.05)
- `max_drawdown_pct` (float): Max drawdown (default: 0.15)
- `var_confidence_level` (float): VaR confidence (default: 0.95)

##### `Config(BaseConfig)`
Master configuration class aggregating all configurations.

**Parameters:**
- `environment` (str): Application environment (default: "development")
- `debug` (bool): Debug mode (default: False)
- `app_name` (str): Application name (default: "trading-bot-suite")
- `app_version` (str): Application version (default: "1.0.0")
- `database` (DatabaseConfig): Database configuration
- `security` (SecurityConfig): Security configuration
- `error_handling` (ErrorHandlingConfig): Error handling configuration
- `exchanges` (ExchangeConfig): Exchange configuration
- `risk` (RiskConfig): Risk management configuration

**Methods:**
- `from_yaml(yaml_path: str | Path)` → Config: Load from YAML file
- `to_yaml(yaml_path: str | Path)` → None: Save to YAML file
- `get_database_url()` → str: Get PostgreSQL URL
- `get_async_database_url()` → str: Get async PostgreSQL URL
- `get_redis_url()` → str: Get Redis URL
- `is_production()` → bool: Check if production environment
- `is_development()` → bool: Check if development environment

### 3. `dependency_injection.py`

Dependency injection framework to eliminate circular dependencies.

#### Classes

##### `DependencyContainer`
Container for managing dependencies.

**Methods:**
- `register(name: str, service: Any | Callable, singleton: bool = False)` → None
- `register_class(name: str, cls: type[T], *args, singleton: bool = False, **kwargs)` → None
- `get(name: str)` → Any: Get service by name
- `has(name: str)` → bool: Check if service registered
- `clear()` → None: Clear all services

##### `DependencyInjector`
Singleton dependency injector for automatic resolution.

**Methods:**
- `register(name: str | None = None, singleton: bool = False)`: Decorator to register service
- `inject(func: Callable)` → Callable: Decorator to inject dependencies
- `resolve(name: str)` → Any: Resolve dependency by name
- `register_service(name: str, service: Any, singleton: bool = False)` → None
- `register_factory(name: str, factory: Callable, singleton: bool = False)` → None
- `has_service(name: str)` → bool: Check if service registered
- `clear()` → None: Clear all services
- `get_instance()` → DependencyInjector: Get singleton instance

##### `ServiceLocator`
Service locator for easy access to services.

**Methods:**
- `__getattr__(name: str)` → Any: Get service by attribute access

#### Global Instances
- `injector`: Global DependencyInjector instance
- `services`: Global ServiceLocator instance

#### Decorators
- `@injectable(name: str | None = None, singleton: bool = False)`: Mark class as injectable
- `@inject`: Inject dependencies into function

### 4. `exceptions.py`

Comprehensive unified exception hierarchy for the entire system.

#### Base Classes

##### `TradingBotError(Exception)`
Base exception for all trading bot errors.

**Parameters:**
- `message` (str): Human-readable error message
- `error_code` (str | None): Standardized error code
- `category` (ErrorCategory): Error category for handling
- `severity` (ErrorSeverity): Error severity level
- `details` (dict | None): Additional context data
- `retryable` (bool): Whether error can be retried
- `retry_after` (int | None): Suggested retry delay
- `suggested_action` (str | None): Recommended resolution
- `context` (dict | None): Additional contextual info

**Methods:**
- `to_dict()` → dict: Convert to dictionary
- `__str__()` → str: Formatted error message
- `__repr__()` → str: Detailed representation

#### Enums

##### `ErrorCategory`
- `RETRYABLE`: Can be retried automatically
- `FATAL`: Cannot be retried
- `VALIDATION`: Input validation errors
- `CONFIGURATION`: Configuration errors
- `PERMISSION`: Auth errors
- `RATE_LIMIT`: Rate limiting errors
- `NETWORK`: Network errors
- `DATA_QUALITY`: Data quality issues
- `BUSINESS_LOGIC`: Business rule violations
- `SYSTEM`: System/infrastructure errors

##### `ErrorSeverity`
- `LOW`: Low severity
- `MEDIUM`: Medium severity
- `HIGH`: High severity
- `CRITICAL`: Critical severity

#### Exception Categories

##### Exchange Exceptions
- `ExchangeError`: Base exchange error
- `ExchangeConnectionError`: Network connection failures
- `ExchangeRateLimitError`: Rate limit violations
- `ExchangeInsufficientFundsError`: Insufficient balance
- `ExchangeOrderError`: Order-related errors
- `ExchangeAuthenticationError`: Auth failures
- `InvalidOrderError`: Invalid order parameters

##### Risk Management Exceptions
- `RiskManagementError`: Base risk error
- `PositionLimitError`: Position limit violations
- `DrawdownLimitError`: Drawdown limit violations
- `RiskCalculationError`: Risk metric failures
- `CapitalAllocationError`: Capital allocation violations
- `CircuitBreakerTriggeredError`: Circuit breaker activation
- `EmergencyStopError`: Emergency stop failures

##### Data Exceptions
- `DataError`: Base data error
- `DataValidationError`: Data validation failures
- `DataSourceError`: Data source connectivity issues
- `DataProcessingError`: Processing pipeline failures
- `DataCorruptionError`: Data integrity issues
- `DataQualityError`: Data quality problems

##### Model Exceptions
- `ModelError`: Base ML model error
- `ModelLoadError`: Model loading failures
- `ModelInferenceError`: Prediction failures
- `ModelDriftError`: Performance drift detection
- `ModelTrainingError`: Training failures
- `ModelValidationError`: Validation failures

##### Validation Exceptions
- `ValidationError`: Base validation error
- `ConfigurationError`: Configuration validation
- `SchemaValidationError`: Schema compliance failures
- `InputValidationError`: Input parameter validation
- `BusinessRuleValidationError`: Business rule violations

##### Execution Exceptions
- `ExecutionError`: Base execution error
- `OrderRejectionError`: Order rejected
- `SlippageError`: Excessive slippage
- `ExecutionTimeoutError`: Execution timeout
- `ExecutionPartialFillError`: Partial fill handling

##### Network Exceptions
- `NetworkError`: Base network error
- `ConnectionError`: Connection failures
- `TimeoutError`: Operation timeouts
- `WebSocketError`: WebSocket errors

##### State Management Exceptions
- `StateConsistencyError`: Base state error
- `StateError`: General state errors
- `StateCorruptionError`: State corruption
- `StateLockError`: Lock acquisition failures
- `SynchronizationError`: Sync errors
- `ConflictError`: State conflicts

##### Security Exceptions
- `SecurityError`: Base security error
- `AuthenticationError`: Auth failures
- `AuthorizationError`: Permission failures
- `EncryptionError`: Encryption failures
- `TokenValidationError`: Token validation failures

##### Component Exceptions
- `ComponentError`: Base component error
- `ServiceError`: Service layer errors
- `RepositoryError`: Repository errors
- `FactoryError`: Factory pattern errors
- `DependencyError`: Dependency injection errors
- `HealthCheckError`: Health check errors
- `EventError`: Event system errors

##### Performance Exceptions
- `PerformanceError`: Base performance error
- `CacheError`: Cache operation errors
- `MemoryOptimizationError`: Memory optimization errors
- `DatabaseOptimizationError`: DB performance errors
- `ConnectionPoolError`: Connection pool errors
- `ProfilingError`: Profiling errors

#### Utilities

##### `ExchangeErrorMapper`
Maps exchange-specific errors to standardized exceptions.

**Methods:**
- `map_error(exchange: str, error_data: dict)` → TradingBotError
- `map_binance_error(error_data: dict)` → TradingBotError
- `map_coinbase_error(error_data: dict)` → TradingBotError
- `map_okx_error(error_data: dict)` → TradingBotError

##### Helper Functions
- `create_error_from_dict(error_dict: dict)` → TradingBotError
- `is_retryable_error(error: Exception)` → bool
- `get_retry_delay(error: Exception)` → int | None

### 5. `logging.py`

Structured logging system with correlation tracking and performance monitoring.

#### Classes

##### `CorrelationContext`
Context manager for correlation ID tracking.

**Methods:**
- `set_correlation_id(correlation_id: str)` → None
- `get_correlation_id()` → str | None
- `generate_correlation_id()` → str
- `correlation_context(correlation_id: str | None = None)`: Context manager

##### `SecureLogger`
Logger wrapper preventing sensitive data exposure.

**Parameters:**
- `logger` (structlog.BoundLogger): Underlying logger

**Methods:**
- `info(message: str, **kwargs)` → None
- `warning(message: str, **kwargs)` → None
- `error(message: str, **kwargs)` → None
- `critical(message: str, **kwargs)` → None
- `debug(message: str, **kwargs)` → None

##### `PerformanceMonitor`
Performance monitoring context manager.

**Parameters:**
- `operation_name` (str): Operation to monitor

**Context Manager:**
- Tracks execution time
- Logs performance metrics
- Handles exceptions

#### Functions

##### `setup_logging`
Setup structured logging configuration.

**Parameters:**
- `environment` (str): Environment (default: "development")
- `log_level` (str): Log level (default: "INFO")
- `log_file` (str | None): Log file path (None for stdout)
- `max_bytes` (int): Max bytes per file (default: 10MB)
- `backup_count` (int): Backup files to keep (default: 5)
- `retention_days` (int): Days to retain logs (default: 30)

##### `get_logger`
Get a structured logger instance.

**Parameters:**
- `name` (str): Logger name (usually __name__)

**Returns:**
- `structlog.BoundLogger`: Configured logger

##### `get_secure_logger`
Get a secure logger instance.

**Parameters:**
- `name` (str): Logger name

**Returns:**
- `SecureLogger`: Secure logger instance

##### Decorators

###### `@log_performance`
Decorator for function performance logging.

**Parameters:**
- `func` (Callable): Function to decorate

**Returns:**
- `Callable`: Decorated function

###### `@log_async_performance`
Decorator for async function performance logging.

**Parameters:**
- `func` (Callable): Async function to decorate

**Returns:**
- `Callable`: Decorated async function

#### Global Instances
- `correlation_context`: Global CorrelationContext instance

### 6. `global_error_handler.py`

Global error handling utilities.

#### Classes

##### `GlobalErrorHandler`
Centralized error handling system.

**Methods:**
- `handle_error(error: Exception, context: dict | None = None)` → None
- `register_handler(error_type: type, handler: Callable)` → None
- `set_default_handler(handler: Callable)` → None

### 7. `memory_manager.py`

Memory optimization and management utilities.

#### Classes

##### `MemoryManager`
Memory usage monitoring and optimization.

**Methods:**
- `get_memory_usage()` → dict: Current memory usage
- `optimize_memory()` → None: Run garbage collection
- `set_memory_limit(limit_mb: float)` → None
- `check_memory_threshold()` → bool

### 8. `service_manager.py`

Service lifecycle management.

#### Classes

##### `ServiceManager`
Manages service initialization and shutdown.

**Methods:**
- `register_service(name: str, service: Any)` → None
- `start_service(name: str)` → None
- `stop_service(name: str)` → None
- `restart_service(name: str)` → None
- `get_service_status(name: str)` → dict

### 9. `validator_registry.py`

Validation framework for data and configuration.

#### Classes

##### `ValidatorRegistry`
Registry for validation functions.

**Methods:**
- `register_validator(name: str, validator: Callable)` → None
- `validate(name: str, data: Any)` → ValidationResult
- `get_validator(name: str)` → Callable | None

## Submodules

### base/
Base component classes for the framework.

- `component.py`: BaseComponent class
- `events.py`: Event emitter system
- `factory.py`: Factory pattern implementation
- `health.py`: Health check framework
- `interfaces.py`: Interface definitions
- `repository.py`: Repository pattern
- `service.py`: Service layer base

### caching/
Caching framework implementation.

- `cache_decorators.py`: Caching decorators
- `cache_keys.py`: Cache key generation
- `cache_manager.py`: Cache management
- `cache_metrics.py`: Cache performance metrics
- `cache_monitoring.py`: Cache monitoring
- `cache_warming.py`: Cache warming strategies
- `unified_cache_layer.py`: Unified caching interface

### config/
Configuration submodules.

- `base.py`: Base configuration classes
- `capital.py`: Capital management config
- `database.py`: Database configuration
- `exchange.py`: Exchange configuration
- `main.py`: Main configuration
- `migration.py`: Migration configuration
- `risk.py`: Risk management config
- `security.py`: Security configuration
- `service.py`: Service configuration
- `strategy.py`: Strategy configuration

### performance/
Performance optimization utilities.

- `memory_optimizer.py`: Memory optimization
- `performance_monitor.py`: Performance monitoring
- `performance_optimizer.py`: Performance optimization
- `trading_profiler.py`: Trading-specific profiling

### types/
Type definitions for the system.

- `base.py`: Base type definitions
- `bot.py`: Bot-related types
- `data.py`: Data processing types
- `execution.py`: Order execution types
- `market.py`: Market data types
- `risk.py`: Risk management types
- `strategy.py`: Strategy types
- `trading.py`: Trading types

## Usage Examples

### Configuration

```python
from src.core.config import Config

# Load configuration from environment
config = Config()

# Load from YAML file
config = Config.from_yaml("config/trading.yaml")

# Access configuration values
db_url = config.get_database_url()
is_prod = config.is_production()
```

### Logging

```python
from src.core.logging import get_logger, PerformanceMonitor, correlation_context

# Get logger
logger = get_logger(__name__)

# Log with correlation
with correlation_context.correlation_context() as correlation_id:
    logger.info("Processing trade", trade_id="123", correlation_id=correlation_id)

# Monitor performance
with PerformanceMonitor("trade_execution"):
    execute_trade()
```

### Exception Handling

```python
from src.core.exceptions import ExchangeError, RiskManagementError

try:
    place_order()
except ExchangeRateLimitError as e:
    if e.retryable:
        time.sleep(e.retry_after)
        retry_order()
except RiskManagementError as e:
    logger.error("Risk violation", error=e.to_dict())
    emergency_stop()
```

### Dependency Injection

```python
from src.core.dependency_injection import injectable, inject, injector

@injectable(singleton=True)
class TradingService:
    def execute_trade(self):
        pass

@inject
def process_order(trading_service: TradingService):
    trading_service.execute_trade()

# Register services
injector.register_service("db", database_connection)
```

## Testing

The core module has comprehensive test coverage:

```bash
# Run all core module tests
pytest tests/unit/test_core/ -v

# Run specific test file
pytest tests/unit/test_core/test_core_config.py -v

# Run with coverage
pytest tests/unit/test_core/ --cov=src/core --cov-report=html
```

## Production Deployment

### Environment Variables

Required environment variables for production:

```bash
# Security (Required)
export SECRET_KEY="your-32-character-minimum-jwt-secret-key"
export ENCRYPTION_KEY="your-32-character-minimum-encryption-key"

# Database
export DB_HOST="postgresql-server"
export DB_PORT="5432"
export DB_NAME="trading_prod"
export DB_USER="trading_user"
export DB_PASSWORD="secure-password"

# Redis
export REDIS_HOST="redis-server"
export REDIS_PORT="6379"
export REDIS_PASSWORD="redis-password"

# Environment
export ENVIRONMENT="production"
```

### Initialization

```python
from src.core.logging import setup_production_logging
from src.core.config import Config

# Initialize logging first
setup_production_logging(log_dir="/var/log/trading-bot", app_name="t-bot")

# Load configuration
config = Config()

# Verify production settings
assert config.is_production()
assert len(config.security.secret_key) >= 32
```

## Best Practices

1. **Always use the unified exception hierarchy** - Never create new exception types outside of `exceptions.py`
2. **Use dependency injection** - Avoid direct imports that create circular dependencies
3. **Log with correlation IDs** - Always include correlation IDs for request tracing
4. **Secure sensitive data** - Use SecureLogger for any logs that might contain sensitive information
5. **Validate configuration** - Always validate configuration on startup
6. **Handle errors properly** - Check if errors are retryable and respect retry delays
7. **Monitor performance** - Use PerformanceMonitor for critical operations
8. **Clean up resources** - Use context managers and proper cleanup in error handlers

## Migration Notes

The core module has been fully refactored and cleaned:
- All TODO comments have been removed
- Duplicate code has been eliminated
- Import errors have been fixed
- Security issues have been addressed
- All tests are passing (156/156)

## Version History

- **1.0.0** - Initial production-ready release
  - Unified exception hierarchy
  - Comprehensive configuration management
  - Structured logging with correlation
  - Dependency injection framework
  - Full test coverage