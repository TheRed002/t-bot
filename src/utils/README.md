# Utils Module

The **Utils Module** provides comprehensive utility functions, decorators, validators, formatters, and constants that serve as the essential toolbox for all other modules in the trading bot system.

## Module Overview

### What This Module Does
- **Performance Monitoring**: Provides decorators for execution time, memory usage, and CPU monitoring across all operations
- **Error Handling**: Implements retry mechanisms, circuit breakers, timeouts, and resilient operation patterns
- **Data Validation**: Offers comprehensive validation for financial data, API inputs, configurations, and business rules
- **Data Formatting**: Standardizes output formatting for financial data, API responses, reports, and exports
- **Mathematical Utilities**: Provides financial calculations, statistical analysis, and risk metrics computation
- **Caching System**: Implements in-memory, Redis, and TTL caching for performance optimization
- **System Constants**: Centralizes all system-wide constants, limits, thresholds, and configuration values
- **Utility Functions**: Offers helper functions for common operations across the entire system

### What This Module Does NOT Do
- **Business Logic**: Does not implement trading strategies, risk management decisions, or execution logic
- **External Integrations**: Does not directly connect to exchanges, databases, or external APIs
- **State Management**: Does not maintain application state or manage sessions
- **Data Storage**: Does not handle data persistence or database operations
- **Network Communications**: Does not manage network protocols or communication channels

---

## File Structure

The utils module consists of individual utility files, each serving a specific cross-cutting concern:

### 1. **decorators.py** - Cross-Cutting Decorators
Performance monitoring, error handling, caching, logging, validation, and rate limiting decorators.

### 2. **helpers.py** - Common Operations
Mathematical utilities, date/time handling, data conversion, file operations, and string processing.

### 3. **validators.py** - Data Integrity
Comprehensive validation for financial data, configurations, API inputs, and business rules.

### 4. **formatters.py** - Output Formatting
Consistent data formatting for financial data, API responses, reports, and exports.

### 5. **constants.py** - System Constants
Centralized constants, limits, thresholds, and configuration values for the entire system.

---

## File Reference

### **decorators.py**
Cross-cutting decorators for performance monitoring, error handling, caching, and validation.

#### **Performance Monitoring Decorators:**
- `time_execution(func: Callable) -> Callable` - Decorator to measure and log function execution time with detailed performance metrics
- `memory_usage(func: Callable) -> Callable` - Decorator to monitor memory usage during function execution with before/after snapshots
- `cpu_usage(func: Callable) -> Callable` - Decorator to monitor CPU usage during function execution with performance analysis

#### **Error Handling Decorators:**
- `retry(max_attempts: int = 3, delay: float = 1.0, backoff_factor: float = 2.0, exceptions: tuple = (Exception,)) -> Callable` - Decorator to retry function execution with exponential backoff on failure
- `circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60.0, expected_exception: type = Exception) -> Callable` - Decorator implementing circuit breaker pattern for fault tolerance
- `timeout(seconds: float) -> Callable` - Decorator to enforce execution timeout limits on functions

#### **Caching Decorators:**
- `cache_result(ttl_seconds: float = 300) -> Callable` - Decorator to cache function results in memory with TTL expiration
- `redis_cache(ttl_seconds: float = 300, prefix: str = "cache") -> Callable` - Decorator to cache function results in Redis with distributed caching
- `ttl_cache(ttl_seconds: float = 300) -> Callable` - Alias for cache_result with TTL-based expiration

#### **Logging Decorators:**
- `log_calls(func: Callable) -> Callable` - Decorator to log all function calls with parameter information and success/failure status
- `log_performance(func: Callable) -> Callable` - Decorator to log detailed performance metrics including execution time and resource usage
- `log_errors(func: Callable) -> Callable` - Decorator to log function errors with context and stack traces

#### **Validation Decorators:**
- `validate_input(validation_func: Callable) -> Callable` - Decorator to validate function input parameters using custom validation function
- `validate_output(validation_func: Callable) -> Callable` - Decorator to validate function output using custom validation function
- `type_check(func: Callable) -> Callable` - Decorator to perform runtime type checking of function parameters and return values

#### **Rate Limiting Decorators:**
- `rate_limit(max_calls: int, time_window: float) -> Callable` - Decorator to enforce rate limits on function calls within time windows
- `api_throttle(requests_per_second: float) -> Callable` - Decorator to throttle API calls to prevent rate limit violations

---

### **helpers.py**
Helper functions for mathematical calculations, date/time handling, data conversion, and common operations.

#### **Mathematical Utilities:**
- `calculate_percentage_change(old_value: float, new_value: float) -> float` - Calculate percentage change between two values with division by zero protection
- `calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02, frequency: str = "daily") -> float` - Calculate Sharpe ratio for performance evaluation
- `calculate_max_drawdown(values: List[float]) -> Tuple[float, int, int]` - Calculate maximum drawdown with start and end indices
- `calculate_var(returns: List[float], confidence_level: float = 0.95) -> float` - Calculate Value at Risk (VaR) at specified confidence level
- `calculate_volatility(returns: List[float], frequency: str = "daily") -> float` - Calculate annualized volatility from returns
- `calculate_correlation(x: List[float], y: List[float]) -> float` - Calculate Pearson correlation coefficient between two datasets

#### **Date/Time Utilities:**
- `get_trading_session(dt: datetime) -> str` - Determine trading session (asian, european, american) for given datetime
- `is_market_open(symbol: str, dt: datetime = None) -> bool` - Check if market is open for trading symbol at specified time
- `convert_timezone(dt: datetime, target_tz: str) -> datetime` - Convert datetime to target timezone with proper handling
- `parse_datetime(dt_str: str, format_str: Optional[str] = None) -> datetime` - Parse datetime string with multiple format support

#### **Data Conversion Utilities:**
- `convert_currency(amount: float, from_currency: str, to_currency: str, exchange_rate: float) -> float` - Convert amount between currencies with precision handling
- `normalize_price(price: float, symbol: str) -> Decimal` - Normalize price to appropriate precision for trading symbol
- `round_to_precision(value: float, precision: int) -> Decimal` - Round value to specified decimal precision with banker's rounding

#### **File Operations:**
- `safe_read_file(file_path: str, encoding: str = "utf-8") -> str` - Safely read file contents with error handling and validation
- `safe_write_file(file_path: str, content: str, encoding: str = "utf-8") -> bool` - Safely write content to file with atomic operations
- `load_config_file(file_path: str) -> Dict[str, Any]` - Load configuration file with support for JSON and YAML formats

#### **Network Utilities:**
- `test_connection(host: str, port: int, timeout: float = 5.0) -> bool` - Test network connection to host and port with timeout
- `measure_latency(host: str, port: int = 80, timeout: float = 5.0) -> float` - Measure network latency to host in milliseconds
- `ping_host(host: str, timeout: float = 5.0) -> Tuple[bool, float]` - Ping host and return status with latency measurement

#### **String Utilities:**
- `sanitize_symbol(symbol: str) -> str` - Sanitize trading symbol removing invalid characters and normalizing format
- `parse_trading_pair(pair: str) -> Tuple[str, str]` - Parse trading pair string into base and quote currencies
- `format_timestamp(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str` - Format datetime as string with customizable format

#### **Technical Analysis Utilities:**
- `calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float` - Calculate Average True Range for volatility measurement

---

### **validators.py**
Comprehensive validation functions for data integrity across all system components.

#### **Financial Data Validation:**
- `validate_price(price: Union[float, Decimal], symbol: str = None) -> bool` - Validate price values with range checks and precision validation
- `validate_quantity(quantity: Union[float, Decimal], symbol: str = None) -> bool` - Validate order quantities with minimum amount checks
- `validate_symbol(symbol: str) -> bool` - Validate trading symbol format and supported exchanges
- `validate_order_request(order: OrderRequest) -> bool` - Validate complete order request with all business rules
- `validate_market_data(data: MarketData) -> bool` - Validate market data completeness and ranges

#### **Configuration Validation:**
- `validate_config(config: Union[Dict[str, Any], Config], required_fields: List[str] = None) -> bool` - Validate configuration objects with required field checking
- `validate_risk_parameters(params: Union[Dict[str, Any], RiskConfig]) -> bool` - Validate risk management parameters and limits
- `validate_strategy_config(config: Union[Dict[str, Any], StrategyConfig]) -> bool` - Validate strategy configuration parameters

#### **API Input Validation:**
- `validate_api_request(request_data: Dict[str, Any], required_fields: List[str] = None) -> bool` - Validate API request payloads
- `validate_webhook_payload(payload: Dict[str, Any], signature: str = None) -> bool` - Validate webhook payloads with signature verification
- `sanitize_user_input(input_data: Any) -> Any` - Sanitize user input to prevent injection attacks

#### **Data Type Validation:**
- `validate_decimal(value: Any, precision: int = None) -> bool` - Validate decimal values with precision constraints
- `validate_positive_number(value: Union[int, float, Decimal]) -> bool` - Validate positive numeric values
- `validate_percentage(value: float) -> bool` - Validate percentage values (0-1 range)

#### **Business Rule Validation:**
- `validate_trading_rules(symbol: str, order_type: str, quantity: float, price: float = None) -> bool` - Validate trading business rules
- `validate_risk_limits(position_size: float, portfolio_value: float, max_exposure: float) -> bool` - Validate risk limit compliance
- `validate_position_limits(positions: List[Position], limits: Dict[str, Any]) -> bool` - Validate position limits across portfolio

#### **Exchange Data Validation:**
- `validate_order_response(response: OrderResponse) -> bool` - Validate exchange order responses
- `validate_balance_data(balance: Dict[str, Any]) -> bool` - Validate account balance data from exchanges
- `validate_trade_data(trade: Dict[str, Any]) -> bool` - Validate trade execution data

#### **Core Object Validation:**
- `validate_position(position: Position) -> bool` - Validate position objects with all business rules
- `validate_signal(signal: Signal) -> bool` - Validate trading signals with confidence and direction checks
- `validate_order(order: OrderRequest) -> bool` - Validate order requests with comprehensive checks

---

### **formatters.py**
Data formatting utilities for consistent output across all system components.

#### **Financial Formatting:**
- `format_currency(amount: float, currency: str = "USD", precision: int = 2) -> str` - Format amounts as currency strings with appropriate precision
- `format_percentage(value: float, precision: int = 2) -> str` - Format values as percentage strings with % symbol
- `format_pnl(pnl: float, currency: str = "USD") -> Tuple[str, str]` - Format P&L with color coding (positive/negative) and currency
- `format_quantity(quantity: float, symbol: str) -> str` - Format trading quantities with symbol-appropriate precision
- `format_price(price: float, symbol: str) -> str` - Format prices with symbol-appropriate precision and thousands separators

#### **API Response Formatting:**
- `format_api_response(data: Any, success: bool = True, message: str = None) -> Dict[str, Any]` - Format standardized API responses with consistent structure
- `format_error_response(error: Exception, code: int = 400) -> Dict[str, Any]` - Format error responses with proper error codes and messages
- `format_success_response(data: Any, message: str = "Success") -> Dict[str, Any]` - Format successful API responses with data payload

#### **Log Formatting:**
- `format_log_entry(level: str, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]` - Format structured log entries with context
- `format_correlation_id(correlation_id: str) -> str` - Format correlation IDs for request tracing
- `format_structured_log(event: str, **kwargs) -> Dict[str, Any]` - Format structured logs with consistent field naming

#### **Chart Data Formatting:**
- `format_ohlcv_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]` - Format OHLCV data for charting libraries
- `format_indicator_data(indicator: str, data: List[float], timestamps: List[datetime]) -> Dict[str, Any]` - Format technical indicator data
- `format_chart_data(data: Dict[str, Any], chart_type: str = "candlestick") -> Dict[str, Any]` - Format complete chart data packages

#### **Report Formatting:**
- `format_performance_report(metrics: Dict[str, Any]) -> Dict[str, Any]` - Format performance reports with standardized metrics
- `format_risk_report(risk_data: Dict[str, Any]) -> Dict[str, Any]` - Format risk assessment reports with color coding
- `format_trade_report(trades: List[Dict[str, Any]]) -> Dict[str, Any]` - Format trade execution reports with summaries

#### **Export Formatting:**
- `format_csv_data(data: List[Dict[str, Any]], headers: List[str] = None) -> str` - Format data for CSV export with proper escaping
- `format_excel_data(data: List[Dict[str, Any]], sheet_name: str = "Data") -> bytes` - Format data for Excel export with formatting
- `format_json_data(data: Any, indent: int = 2) -> str` - Format data as pretty-printed JSON strings

---

### **constants.py**
System-wide constants and configuration values used across all modules.

#### **Trading Constants:**
- `MARKET_HOURS` - Market operating hours for different exchanges and timezones
- `SETTLEMENT_TIMES` - Settlement periods for different asset types (T+0, T+1, T+2)
- `PRECISION_LEVELS` - Decimal precision levels for different currencies and calculations
- `TRADING_SESSIONS` - Global trading session definitions (Asian, European, American)

#### **API Constants:**
- `API_ENDPOINTS` - Base URLs and endpoints for different exchanges (production and testnet)
- `RATE_LIMITS` - Rate limiting thresholds for different exchanges and request types
- `TIMEOUTS` - Timeout values for different operation types (short, long, websocket)
- `HTTP_STATUS_CODES` - Standard HTTP status codes with descriptive names

#### **Financial Constants:**
- `FEE_STRUCTURES` - Fee schedules for different exchanges (maker/taker fees)
- `GLOBAL_FEE_STRUCTURE` - Default fee structure for calculations
- `MINIMUM_AMOUNTS` - Minimum order sizes and notional values by currency
- `GLOBAL_MINIMUM_AMOUNTS` - System-wide minimum amounts for operations
- `MAXIMUM_AMOUNTS` - Maximum limits for orders, positions, and risk exposure
- `SLIPPAGE_TOLERANCE` - Acceptable slippage levels (low, medium, high, max)

#### **Configuration Constants:**
- `DEFAULT_VALUES` - Default configuration values for strategies and risk management
- `LIMITS` - System resource limits (memory, CPU, connections, file sizes)
- `THRESHOLDS` - Performance and quality thresholds for monitoring and alerting

#### **Error Constants:**
- `ERROR_CODES` - Standardized error codes for different error categories
- `ERROR_MESSAGES` - Error message templates with placeholder support
- `ERROR_SEVERITY` - Error severity levels for prioritization
- `ERROR_RECOVERY_STRATEGIES` - Available recovery strategies for different error types

#### **Market Constants:**
- `SYMBOL_MAPPINGS` - Symbol format mappings between different exchanges
- `EXCHANGE_SPECIFICATIONS` - Exchange capabilities and supported features
- `TRADING_PAIRS` - Trading pair specifications with precision and limits
- `MARKET_DATA_INTERVALS` - Standard timeframe intervals for market data
- `ORDER_BOOK_DEPTHS` - Standard order book depth levels
- `TIME_IN_FORCE_OPTIONS` - Order time-in-force options
- `ORDER_STATUS_VALUES` - Standard order status values
- `POSITION_SIDE_VALUES` - Position side indicators
- `SIGNAL_DIRECTION_VALUES` - Signal direction mappings
- `RISK_LEVEL_VALUES` - Risk level classifications

---

## Module Dependencies

### Local Modules This Utils Module Depends On
1. **`src.core`** - Core types, exceptions, configuration, and logging framework
2. **`src.error_handling`** - Error handling framework for resilient operations

### External Dependencies
- **numpy**: For mathematical calculations and statistical analysis
- **pandas**: For data manipulation and analysis
- **psutil**: For system resource monitoring (CPU, memory)
- **aiohttp**: For asynchronous HTTP operations
- **pytz**: For timezone handling and conversions
- **PyYAML**: For YAML configuration file support
- **hashlib**: For hash generation and data integrity
- **Python Standard Library**: typing, decimal, datetime, pathlib, json, csv, etc.

---

## Modules That Should Import From This Utils Module

### **All Local Modules** - Every module benefits from utils functionality:

1. **`src.data`** - Uses decorators, validators, formatters, and helpers for data processing
2. **`src.exchanges`** - Uses decorators, validators, and formatters for API operations
3. **`src.risk_management`** - Uses mathematical helpers, validators, and constants for risk calculations
4. **`src.capital_management`** - Uses financial helpers, validators, and formatters for capital operations
5. **`src.strategies`** - Uses decorators, validators, and mathematical helpers for strategy implementation
6. **`src.database`** - Uses decorators, validators, and helpers for database operations
7. **`src.error_handling`** - Uses decorators and helpers for error management

### **Standard Import Pattern:**
```python
# Decorators for cross-cutting concerns
from src.utils.decorators import time_execution, retry, cache_result

# Validators for data integrity
from src.utils.validators import validate_price, validate_quantity, validate_order

# Formatters for consistent output
from src.utils.formatters import format_currency, format_percentage, format_pnl

# Helpers for common operations
from src.utils.helpers import calculate_sharpe_ratio, calculate_max_drawdown

# Constants for system-wide values
from src.utils.constants import PRECISION_LEVELS, RATE_LIMITS, DEFAULT_VALUES
```

---

## Integration Notes

- **Performance Optimization**: Decorators provide comprehensive performance monitoring and caching
- **Data Integrity**: Validators ensure data quality and business rule compliance throughout the system
- **Consistent Output**: Formatters standardize all data presentation across modules
- **Error Resilience**: Retry mechanisms and circuit breakers provide fault tolerance
- **System Configuration**: Constants centralize all configurable values for easy maintenance
- **Mathematical Foundation**: Helper functions provide reliable financial calculations
- **Type Safety**: Validation decorators ensure runtime type checking and data integrity

**The utils module serves as the essential toolbox that empowers all other modules with reliable, performant, and consistent utility functions.**
