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

from .decorators import (
    time_execution,
    memory_usage,
    cpu_usage,
    retry,
    circuit_breaker,
    timeout,
    cache_result,
    redis_cache,
    ttl_cache,
    log_calls,
    log_performance,
    log_errors,
    validate_input,
    validate_output,
    type_check,
    rate_limit,
    api_throttle
)

from .helpers import (
    # Mathematical utilities
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_var,
    calculate_volatility,
    calculate_correlation,
    
    # Date/Time utilities
    get_trading_session,
    is_market_open,
    convert_timezone,
    parse_datetime,
    
    # Data conversion utilities
    convert_currency,
    normalize_price,
    round_to_precision,
    
    # File operations
    safe_read_file,
    safe_write_file,
    load_config_file,
    
    # Network utilities
    test_connection,
    measure_latency,
    ping_host,
    
    # String utilities
    sanitize_symbol,
    parse_trading_pair,
    format_timestamp
)

from .validators import (
    # Financial data validation
    validate_price,
    validate_quantity,
    validate_symbol,
    validate_order_request,
    validate_market_data,
    
    # Configuration validation
    validate_config,
    validate_risk_parameters,
    validate_strategy_config,
    
    # API input validation
    validate_api_request,
    validate_webhook_payload,
    sanitize_user_input,
    
    # Data type validation
    validate_decimal,
    validate_positive_number,
    validate_percentage,
    
    # Business rule validation
    validate_trading_rules,
    validate_risk_limits,
    validate_position_limits,
    
    # Exchange data validation
    validate_order_response,
    validate_balance_data,
    validate_trade_data
)

from .formatters import (
    # Financial formatting
    format_currency,
    format_percentage,
    format_pnl,
    format_quantity,
    format_price,
    
    # API response formatting
    format_api_response,
    format_error_response,
    format_success_response,
    
    # Log formatting
    format_log_entry,
    format_correlation_id,
    format_structured_log,
    
    # Chart data formatting
    format_ohlcv_data,
    format_indicator_data,
    format_chart_data,
    
    # Report formatting
    format_performance_report,
    format_risk_report,
    format_trade_report,
    
    # Export formatting
    format_csv_data,
    format_excel_data,
    format_json_data
)

from .constants import (
    # Trading constants
    MARKET_HOURS,
    SETTLEMENT_TIMES,
    PRECISION_LEVELS,
    
    # API constants
    API_ENDPOINTS,
    RATE_LIMITS,
    TIMEOUTS,
    
    # Financial constants
    FEE_STRUCTURES,
    MINIMUM_AMOUNTS,
    MAXIMUM_AMOUNTS,
    
    # Configuration constants
    DEFAULT_VALUES,
    LIMITS,
    THRESHOLDS,
    
    # Error constants
    ERROR_CODES,
    ERROR_MESSAGES,
    
    # Market constants
    SYMBOL_MAPPINGS,
    EXCHANGE_SPECIFICATIONS,
    TRADING_PAIRS
)

__all__ = [
    # Decorators
    'time_execution', 'memory_usage', 'cpu_usage', 'retry', 'circuit_breaker',
    'timeout', 'cache_result', 'redis_cache', 'ttl_cache', 'log_calls',
    'log_performance', 'log_errors', 'validate_input', 'validate_output',
    'type_check', 'rate_limit', 'api_throttle',
    
    # Helpers
    'calculate_sharpe_ratio', 'calculate_max_drawdown', 'calculate_var',
    'calculate_volatility', 'calculate_correlation', 'get_trading_session',
    'is_market_open', 'convert_timezone', 'parse_datetime', 'convert_currency',
    'normalize_price', 'round_to_precision', 'safe_read_file', 'safe_write_file',
    'load_config_file', 'test_connection', 'measure_latency', 'ping_host',
    'sanitize_symbol', 'parse_trading_pair', 'format_timestamp',
    
    # Validators
    'validate_price', 'validate_quantity', 'validate_symbol', 'validate_order_request',
    'validate_market_data', 'validate_config', 'validate_risk_parameters',
    'validate_strategy_config', 'validate_api_request', 'validate_webhook_payload',
    'sanitize_user_input', 'validate_decimal', 'validate_positive_number',
    'validate_percentage', 'validate_trading_rules', 'validate_risk_limits',
    'validate_position_limits', 'validate_order_response', 'validate_balance_data',
    'validate_trade_data',
    
    # Formatters
    'format_currency', 'format_percentage', 'format_pnl', 'format_quantity',
    'format_price', 'format_api_response', 'format_error_response',
    'format_success_response', 'format_log_entry', 'format_correlation_id',
    'format_structured_log', 'format_ohlcv_data', 'format_indicator_data',
    'format_chart_data', 'format_performance_report', 'format_risk_report',
    'format_trade_report', 'format_csv_data', 'format_excel_data', 'format_json_data',
    
    # Constants
    'MARKET_HOURS', 'SETTLEMENT_TIMES', 'PRECISION_LEVELS', 'API_ENDPOINTS',
    'RATE_LIMITS', 'TIMEOUTS', 'FEE_STRUCTURES', 'MINIMUM_AMOUNTS',
    'MAXIMUM_AMOUNTS', 'DEFAULT_VALUES', 'LIMITS', 'THRESHOLDS',
    'ERROR_CODES', 'ERROR_MESSAGES', 'SYMBOL_MAPPINGS',
    'EXCHANGE_SPECIFICATIONS', 'TRADING_PAIRS'
]

__version__ = "1.0.0"
__author__ = "Trading Bot Team"
__description__ = "Comprehensive utility framework for trading bot system" 