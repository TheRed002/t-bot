"""
Utility Framework and Helper Functions

This module provides comprehensive utility functions, decorators, validators, and formatters
that are used across all components of the trading bot system.

Key Components:
- Services: ConfigService, ValidationService (modern dependency injection approach)
- Decorators: Performance monitoring, error handling, caching, logging, validation
- Helpers: Mathematical utilities, date/time handling, data conversion, file operations
- Validators: Financial data validation, configuration validation, API input validation
- Formatters: Financial formatting, API response formatting, log formatting
- Constants: System-wide constants and enumerations

Modern Usage (Recommended):
    ```python
    # Use service-based approach with dependency injection
    def __init__(self, validation_service: ValidationService):
        self.validation_service = validation_service
    ```

Legacy Usage (Backward Compatibility):
    ```python
    # Direct imports still work
    from src.utils import validate_order
    ```

Dependencies:
- P-001: Core types, exceptions, config, logging
- P-002A: Error handling framework

Enables:
- P-017+: ML models and all subsequent components
"""

from .constants import (
    # API constants
    API_ENDPOINTS,
    # Configuration constants
    DEFAULT_VALUES,
    # Error constants
    ERROR_CODES,
    ERROR_MESSAGES,
    EXCHANGE_SPECIFICATIONS,
    # Financial constants
    FEE_STRUCTURES,
    GLOBAL_FEE_STRUCTURE,
    GLOBAL_MINIMUM_AMOUNTS,
    LIMITS,
    # Trading constants
    MARKET_HOURS,
    MAXIMUM_AMOUNTS,
    MINIMUM_AMOUNTS,
    PRECISION_LEVELS,
    RATE_LIMITS,
    SETTLEMENT_TIMES,
    # Market constants
    SYMBOL_MAPPINGS,
    THRESHOLDS,
    TIMEOUTS,
    TRADING_PAIRS,
)
from .decimal_utils import (
    ONE,
    ZERO,
    clamp_decimal,
    format_decimal,
    safe_divide,
    to_decimal,
)
from .decorators import (
    UnifiedDecorator,
    api_throttle,
    # Backward compatibility imports
    cache_result,
    cached,
    circuit_breaker,
    cpu_usage,
    dec,
    log_calls,
    log_errors,
    log_performance,
    logged,
    memory_usage,
    monitored,
    rate_limit,
    redis_cache,
    retry,
    robust,
    time_execution,
    timeout,
    ttl_cache,
    type_check,
    validate_input,
    validate_output,
    validated,
)
from .formatters import (
    # API response formatting
    format_api_response,
    format_chart_data,
    format_correlation_id,
    # Export formatting
    format_csv_data,
    # Financial formatting
    format_currency,
    format_error_response,
    format_excel_data,
    format_indicator_data,
    format_json_data,
    # Log formatting
    format_log_entry,
    # Chart data formatting
    format_ohlcv_data,
    format_percentage,
    # Report formatting
    format_performance_report,
    format_pnl,
    format_price,
    format_quantity,
    format_risk_report,
    format_structured_log,
    format_success_response,
    format_trade_report,
)
from .helpers import (
    calculate_correlation,
    calculate_max_drawdown,
    # Mathematical utilities
    calculate_sharpe_ratio,
    calculate_var,
    calculate_volatility,
    # Data conversion utilities
    convert_currency,
    convert_timezone,
    format_timestamp,
    # Date/Time utilities
    get_trading_session,
    is_market_open,
    load_config_file,
    measure_latency,
    normalize_price,
    parse_datetime,
    parse_trading_pair,
    ping_host,
    round_to_precision,
    round_to_precision_decimal,
    # File operations
    safe_read_file,
    safe_write_file,
    # String utilities
    sanitize_symbol,
    # Network utilities
    test_connection,
)
from .interfaces import (
    BaseUtilityService,
    ValidationServiceInterface,
)
from .validation import (
    # Framework and service exports
    ValidationFramework,
    ValidationService,
    validate_batch,
    validate_exchange_credentials,
    # Core validation exports
    validate_order,
    validate_price,
    validate_quantity,
    validate_risk_parameters,
    validate_strategy_params,
    validate_symbol,
    validate_timeframe,
    validator,
)
from .validators import (
    # Standalone validation functions for backward compatibility
    validate_decimal,
    validate_positive_number,
)

__all__ = [
    "API_ENDPOINTS",
    "BaseUtilityService",
    "DEFAULT_VALUES",
    "ERROR_CODES",
    "ERROR_MESSAGES",
    "EXCHANGE_SPECIFICATIONS",
    "FEE_STRUCTURES",
    "GLOBAL_FEE_STRUCTURE",
    "GLOBAL_MINIMUM_AMOUNTS",
    "LIMITS",
    # Constants
    "MARKET_HOURS",
    "MAXIMUM_AMOUNTS",
    "MINIMUM_AMOUNTS",
    "ONE",
    "PRECISION_LEVELS",
    "RATE_LIMITS",
    "SETTLEMENT_TIMES",
    "SYMBOL_MAPPINGS",
    "THRESHOLDS",
    "TIMEOUTS",
    "TRADING_PAIRS",
    "ZERO",
    "UnifiedDecorator",
    # Interfaces
    "ValidationServiceInterface",
    # Validation framework
    "ValidationFramework",
    "ValidationService",
    "api_throttle",
    "cache_result",
    "cached",
    "calculate_correlation",
    "calculate_max_drawdown",
    # Helpers
    "calculate_sharpe_ratio",
    "calculate_var",
    "calculate_volatility",
    "circuit_breaker",
    "clamp_decimal",
    "convert_currency",
    "convert_timezone",
    "cpu_usage",
    "dec",
    "format_api_response",
    "format_chart_data",
    "format_correlation_id",
    "format_csv_data",
    # Formatters
    "format_currency",
    "format_decimal",
    "format_error_response",
    "format_excel_data",
    "format_indicator_data",
    "format_json_data",
    "format_log_entry",
    "format_ohlcv_data",
    "format_percentage",
    "format_performance_report",
    "format_pnl",
    "format_price",
    "format_quantity",
    "format_risk_report",
    "format_structured_log",
    "format_success_response",
    "format_timestamp",
    "format_trade_report",
    "get_trading_session",
    "is_market_open",
    "load_config_file",
    "log_calls",
    "log_errors",
    "log_performance",
    "logged",
    "measure_latency",
    "memory_usage",
    "monitored",
    "normalize_price",
    "parse_datetime",
    "parse_trading_pair",
    "ping_host",
    "rate_limit",
    "redis_cache",
    "retry",
    "robust",
    "round_to_precision",
    "round_to_precision_decimal",
    "safe_divide",
    "safe_read_file",
    "safe_write_file",
    "sanitize_symbol",
    "test_connection",
    # Decorators
    "time_execution",
    "timeout",
    "to_decimal",
    "ttl_cache",
    "type_check",
    "validate_batch",
    "validate_decimal",
    "validate_exchange_credentials",
    "validate_input",
    # Validators - core validation exports
    "validate_order",
    "validate_output",
    "validate_positive_number",
    "validate_price",
    "validate_quantity",
    "validate_risk_parameters",
    "validate_strategy_params",
    "validate_symbol",
    "validate_timeframe",
    "validated",
    "validator",
]

__version__ = "1.0.0"
__author__ = "Trading Bot Team"
# Service registry available for explicit registration
# from .service_registry import register_util_services

__description__ = "Comprehensive utility framework for trading bot system"
