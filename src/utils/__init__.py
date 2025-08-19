"""
Utility Framework and Helper Functions

This module provides comprehensive utility functions, decorators, validators, and formatters
that are used across all components of the trading bot system.

Key Components:
- Decorators: Performance monitoring, error handling, caching, logging, validation
- Helpers: Mathematical utilities, date/time handling, data conversion, file operations
- Validators: Financial data validation, configuration validation, API input validation
- Formatters: Financial formatting, API response formatting, log formatting
- Constants: System-wide constants and enumerations

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
from .decorators import (
    UnifiedDecorator,
    dec,
    Decorators,
    decorators,
)

# Backward compatibility for old decorator names
retry = decorators.retry_on_failure
cache_result = decorators.cached
validate_input = decorators.validated
log_calls = decorators.logged
log_errors = decorators.logged
log_performance = decorators.monitored
api_throttle = decorators.api_call
circuit_breaker = decorators.robust
rate_limit = decorators.api_call
redis_cache = decorators.cached
time_execution = decorators.monitored
timeout = lambda t: dec.enhance(timeout=t)
ttl_cache = decorators.cached
type_check = decorators.validated
validate_output = decorators.validated
cpu_usage = decorators.monitored
memory_usage = decorators.monitored
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
    # File operations
    safe_read_file,
    safe_write_file,
    # String utilities
    sanitize_symbol,
    # Network utilities
    test_connection,
)
from .validators import (
    sanitize_user_input,
    # API input validation
    validate_api_request,
    validate_balance_data,
    # Configuration validation
    validate_config,
    # Data type validation
    validate_decimal,
    validate_market_data,
    validate_order,
    validate_order_request,
    # Exchange data validation
    validate_order_response,
    validate_percentage,
    # Additional validation functions
    validate_position,
    validate_position_limits,
    validate_positive_number,
    # Financial data validation
    validate_price,
    validate_quantity,
    validate_risk_limits,
    validate_risk_parameters,
    validate_signal,
    validate_strategy_config,
    validate_symbol,
    validate_trade_data,
    # Business rule validation
    validate_trading_rules,
    validate_webhook_payload,
)

__all__ = [
    "API_ENDPOINTS",
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
    "PRECISION_LEVELS",
    "RATE_LIMITS",
    "SETTLEMENT_TIMES",
    "SYMBOL_MAPPINGS",
    "THRESHOLDS",
    "TIMEOUTS",
    "TRADING_PAIRS",
    "api_throttle",
    "cache_result",
    "calculate_correlation",
    "calculate_max_drawdown",
    # Helpers
    "calculate_sharpe_ratio",
    "calculate_var",
    "calculate_volatility",
    "circuit_breaker",
    "convert_currency",
    "convert_timezone",
    "cpu_usage",
    "format_api_response",
    "format_chart_data",
    "format_correlation_id",
    "format_csv_data",
    # Formatters
    "format_currency",
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
    "measure_latency",
    "memory_usage",
    "normalize_price",
    "parse_datetime",
    "parse_trading_pair",
    "ping_host",
    "rate_limit",
    "redis_cache",
    "retry",
    "round_to_precision",
    "safe_read_file",
    "safe_write_file",
    "sanitize_symbol",
    "sanitize_user_input",
    "test_connection",
    # Decorators
    "time_execution",
    "timeout",
    "ttl_cache",
    "type_check",
    "validate_api_request",
    "validate_balance_data",
    "validate_config",
    "validate_decimal",
    "validate_input",
    "validate_market_data",
    "validate_order",
    "validate_order_request",
    "validate_order_response",
    "validate_output",
    "validate_percentage",
    "validate_position",
    "validate_position_limits",
    "validate_positive_number",
    # Validators
    "validate_price",
    "validate_quantity",
    "validate_risk_limits",
    "validate_risk_parameters",
    "validate_signal",
    "validate_strategy_config",
    "validate_symbol",
    "validate_trade_data",
    "validate_trading_rules",
    "validate_webhook_payload",
]

__version__ = "1.0.0"
__author__ = "Trading Bot Team"
__description__ = "Comprehensive utility framework for trading bot system"
