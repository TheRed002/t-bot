# Utils Module Reference

## Overview
The Utils module provides comprehensive utility functions, decorators, validators, and formatters for the T-Bot trading system. All utilities are production-ready with proper error handling, type hints, and integration with the core module.

## Module Structure

```
src/utils/
├── __init__.py             # Main module exports
├── calculations/           # Financial calculations
│   ├── __init__.py
│   └── financial.py        # Financial metrics and calculations
├── validation/             # Validation framework
│   ├── __init__.py
│   ├── core.py            # Core validation framework
│   └── service.py         # Validation service with DI
├── constants.py           # System-wide constants
├── data_utils.py          # Data manipulation utilities
├── datetime_utils.py      # Date/time utilities
├── decimal_utils.py       # Decimal precision utilities
├── decorators.py          # Function decorators
├── file_utils.py          # File operations
├── formatters.py          # Data formatting utilities
├── gpu_utils.py           # GPU acceleration utilities
├── helpers.py             # Consolidated helper functions
├── math_utils.py          # Mathematical utilities
├── network_utils.py       # Network operations
├── string_utils.py        # String manipulation
└── validators.py          # Enhanced validation framework
```

## Module Reference

### constants.py
System-wide constants and configurations.

**Constants:**
- `API_ENDPOINTS: dict[str, dict[str, str]]` - API endpoint configurations
- `RATE_LIMITS: dict[str, dict[str, int]]` - Rate limiting configurations
- `TIMEOUTS: dict[str, int]` - Timeout configurations
- `FEE_STRUCTURES: dict[str, dict[str, float]]` - Exchange fee structures
- `MINIMUM_AMOUNTS: dict[str, dict[str, float]]` - Minimum trading amounts
- `MAXIMUM_AMOUNTS: dict[str, dict[str, float]]` - Maximum trading amounts
- `PRECISION_LEVELS: dict[str, dict[str, int]]` - Price/quantity precision
- `MARKET_HOURS: dict[str, dict[str, str]]` - Market trading hours
- `SETTLEMENT_TIMES: dict[str, int]` - Settlement time configurations
- `ERROR_CODES: dict[str, int]` - System error codes
- `ERROR_MESSAGES: dict[int, str]` - Error message mappings
- `THRESHOLDS: dict[str, float]` - System thresholds
- `LIMITS: dict[str, int]` - System limits
- `DEFAULT_VALUES: dict[str, Any]` - Default configurations
- `SYMBOL_MAPPINGS: dict[str, dict[str, str]]` - Symbol mappings
- `EXCHANGE_SPECIFICATIONS: dict[str, dict[str, Any]]` - Exchange specs
- `TRADING_PAIRS: dict[str, list[str]]` - Trading pair configurations

### data_utils.py
Data manipulation and transformation utilities.

**Functions:**
- `dict_to_dataframe(data: dict[str, list[Any]]) -> pd.DataFrame`
  - Converts dictionary to pandas DataFrame
  - Returns: DataFrame object

- `dataframe_to_dict(df: pd.DataFrame, orient: str = "records") -> dict[str, Any]`
  - Converts DataFrame to dictionary
  - Parameters:
    - `orient`: Output format ("records", "list", "series", "split", "index")
  - Returns: Dictionary representation

- `aggregate_ohlcv(data: list[dict[str, Any]], timeframe: str = "1h") -> list[dict[str, Any]]`
  - Aggregates OHLCV data to specified timeframe
  - Parameters:
    - `data`: List of OHLCV candles
    - `timeframe`: Target timeframe
  - Returns: Aggregated OHLCV data

- `filter_outliers(data: list[float], std_multiplier: float = 3.0) -> list[float]`
  - Filters outliers using standard deviation
  - Parameters:
    - `data`: Input data
    - `std_multiplier`: Standard deviation multiplier
  - Returns: Filtered data

- `normalize_data(data: list[float], method: str = "minmax") -> list[float]`
  - Normalizes data using specified method
  - Parameters:
    - `data`: Input data
    - `method`: "minmax" or "zscore"
  - Returns: Normalized data

- `resample_timeseries(data: list[dict[str, Any]], interval: str = "1h") -> list[dict[str, Any]]`
  - Resamples time series data
  - Parameters:
    - `data`: Time series data
    - `interval`: Resampling interval
  - Returns: Resampled data

- `calculate_returns(prices: list[float], method: str = "simple") -> list[float]`
  - Calculates price returns
  - Parameters:
    - `prices`: Price series
    - `method`: "simple" or "log"
  - Returns: Returns series

- `merge_dataframes(dfs: list[pd.DataFrame], on: str | list[str], how: str = "inner") -> pd.DataFrame`
  - Merges multiple DataFrames
  - Parameters:
    - `dfs`: List of DataFrames
    - `on`: Column(s) to merge on
    - `how`: Merge type
  - Returns: Merged DataFrame

- `pivot_data(df: pd.DataFrame, index: str, columns: str, values: str) -> pd.DataFrame`
  - Creates pivot table
  - Parameters:
    - `df`: Input DataFrame
    - `index`: Index column
    - `columns`: Column to pivot
    - `values`: Values column
  - Returns: Pivot table

- `rolling_window(data: list[float], window_size: int, func: Callable) -> list[float]`
  - Applies rolling window function
  - Parameters:
    - `data`: Input data
    - `window_size`: Window size
    - `func`: Function to apply
  - Returns: Rolling window results

- `interpolate_missing(data: list[float | None], method: str = "linear") -> list[float]`
  - Interpolates missing values
  - Parameters:
    - `data`: Data with missing values
    - `method`: Interpolation method
  - Returns: Interpolated data

### datetime_utils.py
Date and time manipulation utilities.

**Functions:**
- `get_current_utc() -> datetime`
  - Returns current UTC datetime

- `convert_to_utc(dt: datetime) -> datetime`
  - Converts datetime to UTC
  - Returns: UTC datetime

- `get_market_open_close(exchange: str = "binance", date: datetime | None = None) -> tuple[datetime, datetime]`
  - Gets market open/close times
  - Returns: (open_time, close_time)

- `is_trading_hours(dt: datetime, exchange: str = "binance") -> bool`
  - Checks if within trading hours
  - Returns: True if trading hours

- `get_next_trading_day(dt: datetime, exchange: str = "binance") -> datetime`
  - Gets next trading day
  - Returns: Next trading day datetime

- `calculate_time_until_close(exchange: str = "binance") -> timedelta`
  - Calculates time until market close
  - Returns: Time remaining

- `format_duration(seconds: float) -> str`
  - Formats duration as human-readable string
  - Returns: Formatted duration

### decimal_utils.py
Decimal precision and financial calculation utilities.

**Constants:**
- `ZERO = Decimal("0")` - Zero constant
- `ONE = Decimal("1")` - One constant
- `EPSILON = Decimal("1e-10")` - Epsilon for comparisons

**Functions:**
- `to_decimal(value: Any) -> Decimal`
  - Converts value to Decimal
  - Returns: Decimal value

- `round_decimal(value: Decimal, precision: int) -> Decimal`
  - Rounds Decimal to precision
  - Returns: Rounded Decimal

- `is_zero(value: Decimal, epsilon: Decimal = EPSILON) -> bool`
  - Checks if value is zero
  - Returns: True if zero

- `safe_divide(numerator: Decimal, denominator: Decimal, default: Decimal = ZERO) -> Decimal`
  - Safe division with zero check
  - Returns: Division result or default

- `calculate_percentage(value: Decimal, total: Decimal) -> Decimal`
  - Calculates percentage
  - Returns: Percentage as Decimal

### decorators.py
Function decorators for cross-cutting concerns.

**Classes:**
- `UnifiedDecorator` - Unified decorator with multiple features
  - Methods:
    - `__init__(cache: bool, validate: bool, monitor: bool, log: bool)`
    - `__call__(func: Callable) -> Callable`

**Functions:**
- `@retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0)`
  - Retries function on failure
  
- `@cache_result(ttl: int = 3600)`
  - Caches function results with TTL
  
- `@validate_input(**validators)`
  - Validates function inputs
  
- `@validate_output(validator: Callable)`
  - Validates function output
  
- `@log_performance(threshold: float = 1.0)`
  - Logs performance metrics
  
- `@rate_limit(calls: int = 10, period: float = 60)`
  - Rate limits function calls
  
- `@timeout(seconds: float = 30)`
  - Adds timeout to function
  
- `@circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60)`
  - Circuit breaker pattern
  
- `@time_execution`
  - Times function execution
  
- `@log_calls`
  - Logs function calls
  
- `@log_errors`
  - Logs function errors

### file_utils.py
File I/O operations.

**Functions:**
- `safe_read_file(file_path: str, encoding: str = "utf-8") -> str`
  - Safely reads file content
  - Returns: File content

- `safe_write_file(file_path: str, content: str, encoding: str = "utf-8") -> None`
  - Safely writes file content
  
- `async ensure_directory_exists(directory_path: str | Path) -> None`
  - Ensures directory exists
  
- `load_config_file(file_path: str) -> dict[str, Any]`
  - Loads configuration from YAML/JSON
  - Returns: Configuration dictionary
  
- `save_json(data: Any, file_path: str, indent: int = 2) -> None`
  - Saves data as JSON
  
- `load_json(file_path: str) -> Any`
  - Loads JSON file
  - Returns: Parsed JSON data
  
- `save_yaml(data: dict[str, Any], file_path: str) -> None`
  - Saves data as YAML
  
- `load_yaml(file_path: str) -> dict[str, Any]`
  - Loads YAML file
  - Returns: Parsed YAML data

### formatters.py
Data formatting utilities.

**Functions:**
- `format_price(price: float, precision: int = 8) -> str`
  - Formats price for display
  - Returns: Formatted price string

- `format_quantity(quantity: float, precision: int = 8) -> str`
  - Formats quantity for display
  - Returns: Formatted quantity string

- `format_percentage(value: float, decimals: int = 2) -> str`
  - Formats percentage (0.05 → "5.00%")
  - Returns: Formatted percentage

- `format_currency(amount: float, symbol: str = "$", decimals: int = 2) -> str`
  - Formats currency amount
  - Returns: Formatted currency string

- `format_pnl(pnl: float, include_sign: bool = True) -> str`
  - Formats profit/loss
  - Returns: Formatted P&L string

- `format_ohlcv_data(data: list[dict]) -> pd.DataFrame`
  - Formats OHLCV data
  - Returns: Formatted DataFrame

- `format_api_response(success: bool, data: Any = None, error: str = None) -> dict`
  - Formats API response
  - Returns: Response dictionary

- `format_error_response(error: Exception, request_id: str = None) -> dict`
  - Formats error response
  - Returns: Error response dictionary

- `format_success_response(data: Any, message: str = None) -> dict`
  - Formats success response
  - Returns: Success response dictionary

- `format_log_entry(level: str, message: str, **kwargs) -> dict`
  - Formats log entry
  - Returns: Log entry dictionary

### gpu_utils.py
GPU acceleration utilities (optional).

**Classes:**
- `GPUManager` - Manages GPU resources
  - Methods:
    - `__init__()`
    - `get_memory_info(device_id: int = 0) -> dict[str, float]`
    - `clear_cache() -> None`
    - `to_gpu(data: Any, dtype: str = None) -> Any`
    - `to_cpu(data: Any) -> Any`
    - `accelerate_computation(func: Callable, *args, **kwargs) -> Any`

**Functions:**
- `get_optimal_batch_size(data_size: int, memory_limit_gb: float = 4.0) -> int`
  - Calculates optimal batch size
  - Returns: Batch size

- `parallel_apply(df: pd.DataFrame, func: Callable, axis: int = 0, use_gpu: bool = True) -> pd.DataFrame`
  - Applies function with GPU acceleration
  - Returns: Processed DataFrame

- `gpu_accelerated_correlation(data: np.ndarray) -> np.ndarray`
  - GPU-accelerated correlation
  - Returns: Correlation matrix

- `gpu_accelerated_rolling_window(data: np.ndarray, window_size: int, func: Callable) -> np.ndarray`
  - GPU-accelerated rolling window
  - Returns: Results array

### helpers.py
Consolidated helper functions (imports from specialized modules).

Re-exports functions from:
- math_utils
- datetime_utils
- data_utils
- file_utils
- network_utils
- string_utils
- Technical analysis functions

### math_utils.py
Mathematical and statistical utilities.

**Functions:**
- `calculate_percentage_change(old_value: float, new_value: float) -> float`
  - Calculates percentage change
  - Returns: Percentage change

- `calculate_sharpe_ratio(returns: list[float], risk_free_rate: float = 0.02, frequency: str = "daily") -> float`
  - Calculates Sharpe ratio
  - Returns: Sharpe ratio

- `calculate_max_drawdown(equity_curve: list[float]) -> tuple[float, int, int]`
  - Calculates maximum drawdown
  - Returns: (max_drawdown, start_idx, end_idx)

- `calculate_var(returns: list[float], confidence_level: float = 0.95) -> float`
  - Calculates Value at Risk
  - Returns: VaR value

- `calculate_volatility(returns: list[float], window: int = None) -> float`
  - Calculates volatility
  - Returns: Volatility

- `calculate_correlation(series1: list[float], series2: list[float]) -> float`
  - Calculates correlation
  - Returns: Correlation coefficient

- `calculate_beta(asset_returns: list[float], market_returns: list[float]) -> float`
  - Calculates beta
  - Returns: Beta coefficient

- `calculate_sortino_ratio(returns: list[float], risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float`
  - Calculates Sortino ratio
  - Returns: Sortino ratio

### network_utils.py
Network operations and connectivity.

**Functions:**
- `async test_connection(host: str, port: int, timeout: float = 5.0) -> bool`
  - Tests network connection
  - Returns: True if connected

- `async measure_latency(host: str, port: int, timeout: float = 5.0) -> float`
  - Measures network latency
  - Returns: Latency in milliseconds

- `async ping_host(host: str, count: int = 3, port: int = 80) -> dict[str, Any]`
  - Pings host and returns statistics
  - Returns: Ping statistics

- `async check_multiple_hosts(hosts: list[tuple[str, int]], timeout: float = 5.0) -> dict[str, bool]`
  - Checks multiple hosts
  - Returns: Host status dictionary

- `parse_url(url: str) -> dict[str, Any]`
  - Parses URL components
  - Returns: URL components

- `async wait_for_service(host: str, port: int, max_wait: float = 30.0, check_interval: float = 1.0) -> bool`
  - Waits for service availability
  - Returns: True if available

### string_utils.py
String manipulation utilities.

**Functions:**
- `normalize_symbol(symbol: str) -> str`
  - Normalizes trading symbol
  - Returns: Normalized symbol

- `format_price(price: float, precision: int = 8) -> str`
  - Formats price string
  - Returns: Formatted price

- `parse_trading_pair(pair: str) -> tuple[str, str]`
  - Parses trading pair
  - Returns: (base, quote) tuple

- `generate_hash(data: str) -> str`
  - Generates SHA-256 hash
  - Returns: Hash string

- `validate_email(email: str) -> bool`
  - Validates email format
  - Returns: True if valid

- `extract_numbers(text: str) -> list[float]`
  - Extracts numbers from text
  - Returns: List of numbers

- `camel_to_snake(name: str) -> str`
  - Converts camelCase to snake_case
  - Returns: snake_case string

- `snake_to_camel(name: str) -> str`
  - Converts snake_case to camelCase
  - Returns: camelCase string

- `truncate(text: str, max_length: int, suffix: str = "...") -> str`
  - Truncates text
  - Returns: Truncated string

- `format_percentage(value: float, decimals: int = 2) -> str`
  - Formats percentage
  - Returns: Percentage string

### validation/core.py
Core validation framework.

**Class: ValidationFramework**
Single source of truth for validation logic.

**Methods:**
- `validate_order(order: dict[str, Any]) -> bool`
  - Validates order data
  - Raises: ValueError if invalid

- `validate_strategy_params(params: dict[str, Any]) -> bool`
  - Validates strategy parameters
  - Raises: ValueError if invalid

- `validate_price(price: Any, max_price: float = 1_000_000) -> bool`
  - Validates price
  - Raises: ValueError if invalid

- `validate_quantity(quantity: Any, min_qty: float = 0.00000001) -> bool`
  - Validates quantity
  - Raises: ValueError if invalid

- `validate_symbol(symbol: str) -> bool`
  - Validates trading symbol
  - Raises: ValueError if invalid

- `validate_exchange_credentials(credentials: dict[str, Any]) -> bool`
  - Validates exchange credentials
  - Raises: ValueError if invalid

- `validate_risk_parameters(params: dict[str, Any]) -> bool`
  - Validates risk parameters
  - Raises: ValueError if invalid

- `validate_timeframe(timeframe: str) -> str`
  - Validates and normalizes timeframe
  - Returns: Normalized timeframe
  - Raises: ValueError if invalid

- `validate_batch(validations: list[tuple[str, Callable, Any]]) -> dict[str, Any]`
  - Batch validation
  - Returns: Validation results

### validation/service.py
Validation service with dependency injection and caching.

**Class: ValidationService**
Enhanced validation service with async support.

**Methods:**
- `async validate_order(order_data: dict) -> ValidationResult`
- `async validate_position(position_data: dict) -> ValidationResult`
- `async validate_strategy(strategy_data: dict) -> ValidationResult`
- `async validate_risk_params(risk_data: dict) -> ValidationResult`

**Class: ValidationResult**
- Properties:
  - `is_valid: bool`
  - `errors: list[ValidationDetail]`
  - `warnings: list[ValidationDetail]`
  - `metadata: dict[str, Any]`

## Usage Examples

### Basic Validation
```python
from src.utils.validation import validate_order, validate_price

# Validate order
order = {
    "symbol": "BTCUSDT",
    "side": "BUY",
    "type": "LIMIT",
    "price": 50000.0,
    "quantity": 0.1
}
validate_order(order)  # Returns True or raises ValueError

# Validate price
validate_price(50000.0)  # Returns True or raises ValueError
```

### Using Decorators
```python
from src.utils.decorators import retry, cache_result, validate_input

@retry(max_attempts=3, delay=1.0)
@cache_result(ttl=3600)
@validate_input(price=lambda x: x > 0)
def fetch_market_data(symbol: str, price: float):
    # Function implementation
    pass
```

### Data Formatting
```python
from src.utils.formatters import format_price, format_percentage, format_currency

price_str = format_price(50000.12345678, precision=2)  # "50000.12"
pct_str = format_percentage(0.0525)  # "5.25%"
currency_str = format_currency(1234.56, symbol="$")  # "$1,234.56"
```

### File Operations
```python
from src.utils.file_utils import load_config_file, save_json

# Load configuration
config = load_config_file("config.yaml")

# Save JSON data
data = {"key": "value"}
save_json(data, "output.json")
```

### Mathematical Calculations
```python
from src.utils.math_utils import calculate_sharpe_ratio, calculate_max_drawdown

returns = [0.01, 0.02, -0.01, 0.03, 0.01]
sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

equity_curve = [100, 105, 103, 110, 108, 115]
max_dd, start_idx, end_idx = calculate_max_drawdown(equity_curve)
```

## Error Handling

All utility functions use consistent error handling:
- Validation errors raise `ValidationError` from `src.core.exceptions`
- Functions that can fail return `None` or a default value
- Async functions properly handle asyncio exceptions
- All errors are logged with appropriate context

## Performance Considerations

- GPU utilities automatically fall back to CPU if GPU not available
- Caching decorators reduce redundant computations
- Batch operations available for bulk processing
- Async functions for I/O-bound operations
- Epsilon comparisons for float operations (EPSILON = 1e-10)

## Testing

The utils module has comprehensive test coverage:
- Unit tests: `tests/unit/test_utils/`
- Integration tests: `tests/integration/`
- Performance tests: `tests/performance/`

Run tests:
```bash
pytest tests/unit/test_utils/ -v
```

## Dependencies

- Core module (`src.core`)
- NumPy for numerical operations
- Pandas for data manipulation
- Optional: GPU libraries (torch, tensorflow, cupy) for acceleration
- Optional: TA-Lib for technical analysis (fallback implementations available)

## Version History

- v1.0.0: Initial production release
  - Consolidated validation framework
  - Removed code duplication
  - Fixed float comparison issues
  - Made GPU libraries optional
  - Standardized error handling