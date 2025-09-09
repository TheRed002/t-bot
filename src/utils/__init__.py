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
- Exchange Utilities: Shared utilities for exchange implementations to eliminate duplication

Modern Usage (Recommended):
    ```python
    # Use service-based approach with dependency injection
    def __init__(self, validation_service: ValidationService):
        self.validation_service = validation_service


    # Use shared exchange utilities to eliminate duplication
    from src.utils.exchange_order_utils import OrderManagementUtils
    from src.utils.exchange_validation_utils import ExchangeValidationUtils
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
    round_to_precision,
    safe_decimal_conversion,
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

# Execution utilities (new) - shared utilities to eliminate duplication in execution module
from .execution_utils import (
    calculate_order_value,
    calculate_price_deviation_bps,
    calculate_slippage_bps,
    calculate_trade_risk_ratio,
    convert_order_to_request,
    extract_order_details,
    is_order_within_price_bounds,
    validate_order_basic,
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
    # Network utilities
    check_connection,
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
    # REMOVED: round_to_precision, round_to_precision_decimal (use decimal_utils)
    # File operations
    safe_read_file,
    safe_write_file,
    # String utilities
    sanitize_symbol,
)
from .interfaces import (
    BaseUtilityService,
    ValidationServiceInterface,
)
from .monitoring_helpers import (
    AsyncTaskManager,
    HTTPSessionManager,
    MetricValueProcessor,
    SystemMetricsCollector,
    cleanup_http_sessions,
    create_error_context,
    generate_correlation_id,
    generate_fingerprint,
    get_http_session,
    handle_error_with_fallback,
    http_request_with_retry,
    log_unusual_values,
    safe_duration_parse,
    validate_monitoring_parameter,
)

# State constants (new)
from .state_constants import (
    BOT_STATE_REQUIRED_FIELDS,
    CHECKPOINT_FILE_EXTENSION,
    DEFAULT_CACHE_TTL,
    DEFAULT_COMPRESSION_THRESHOLD,
    DEFAULT_MAX_CHECKPOINTS,
    STATE_TYPES,
    VALIDATION_ERROR_TYPES,
)

# State management utilities (new)
from .state_utils import (
    calculate_state_checksum,
    create_state_change_record,
    create_state_metadata,
    deserialize_state_data,
    detect_state_changes,
    ensure_directory_exists,
    format_cache_key,
    get_from_redis_with_timeout,
    serialize_state_data,
    store_in_redis_with_timeout,
)

# State validation utilities (new)
from .state_validation_utils import (
    validate_bot_id_format,
    validate_bot_status,
    validate_capital_allocation,
    validate_cash_balance,
    validate_decimal_field_with_details,
    validate_order_price_logic,
    validate_positive_value_with_details,
    validate_required_fields_with_details,
    validate_strategy_params,
    validate_symbol_format,
    validate_var_limits,
)
from .validation import (
    # Framework and service exports
    ValidationFramework,
    ValidationService,
    get_validator,
    validate_batch,
    validate_exchange_credentials,
    # Core validation exports
    validate_order,
    validate_price,
    validate_quantity,
    validate_risk_parameters,
    validate_symbol,
    validate_timeframe,
)
from .web_interface_utils import (
    create_error_response,
    extract_error_details,
    handle_api_error,
    handle_not_found,
    safe_format_currency,
    safe_format_percentage,
    safe_get_api_facade,
)
from .websocket_manager_utils import (
    BaseWebSocketManager,
    BotStatusWebSocketManager,
    MarketDataWebSocketManager,
    authenticate_websocket,
    handle_websocket_disconnect,
)

# Validation functions are available through ValidationService for type safety and consistency

__all__ = [
    "API_ENDPOINTS",
    "AsyncTaskManager",
    "DEFAULT_VALUES",
    "ERROR_CODES",
    "ERROR_MESSAGES",
    "EXCHANGE_SPECIFICATIONS",
    "FEE_STRUCTURES",
    "GLOBAL_FEE_STRUCTURE",
    "GLOBAL_MINIMUM_AMOUNTS",
    "HTTPSessionManager",
    "LIMITS",
    # Constants
    "MARKET_HOURS",
    "MAXIMUM_AMOUNTS",
    "MINIMUM_AMOUNTS",
    "MetricValueProcessor",
    "ONE",
    "PRECISION_LEVELS",
    "RATE_LIMITS",
    "SETTLEMENT_TIMES",
    "SYMBOL_MAPPINGS",
    "SystemMetricsCollector",
    "THRESHOLDS",
    "TIMEOUTS",
    "TRADING_PAIRS",
    "ZERO",
    "BaseUtilityService",
    "UnifiedDecorator",
    # Validation framework
    "ValidationFramework",
    "ValidationService",
    # Interfaces
    "ValidationServiceInterface",
    "api_throttle",
    "cache_result",
    "cached",
    "calculate_correlation",
    "calculate_max_drawdown",
    # Helpers
    "calculate_sharpe_ratio",
    "calculate_var",
    "calculate_volatility",
    "check_connection",
    "circuit_breaker",
    "clamp_decimal",
    "cleanup_http_sessions",
    "convert_currency",
    "create_error_context",
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
    "generate_correlation_id",
    "generate_fingerprint",
    "get_http_session",
    "get_trading_session",
    "get_validator",
    "handle_error_with_fallback",
    "http_request_with_retry",
    "is_market_open",
    "load_config_file",
    "log_calls",
    "log_errors",
    "log_performance",
    "log_unusual_values",
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
    "safe_divide",
    "safe_duration_parse",
    "safe_read_file",
    "safe_write_file",
    "sanitize_symbol",
    # Decorators
    "time_execution",
    "timeout",
    "to_decimal",
    "ttl_cache",
    "type_check",
    "validate_batch",
    # Core validation functions available through ValidationService
    "validate_exchange_credentials",
    "validate_input",
    "validate_monitoring_parameter",
    # Validators - core validation exports
    "validate_order",
    "validate_output",
    "validate_price",
    "validate_quantity",
    "validate_risk_parameters",
    "validate_symbol",
    "validate_timeframe",
    "validated",
    # Execution utilities
    "calculate_order_value",
    "calculate_price_deviation_bps",
    "calculate_slippage_bps",
    "calculate_trade_risk_ratio",
    "convert_order_to_request",
    "extract_order_details",
    "is_order_within_price_bounds",
    "safe_decimal_conversion",
    "validate_order_basic",
    # Web interface utilities
    "handle_api_error",
    "safe_format_currency",
    "safe_format_percentage",
    "safe_get_api_facade",
    "create_error_response",
    "handle_not_found",
    "extract_error_details",
    # WebSocket utilities
    "BaseWebSocketManager",
    "MarketDataWebSocketManager",
    "BotStatusWebSocketManager",
    "authenticate_websocket",
    "handle_websocket_disconnect",
    # State constants
    "BOT_STATE_REQUIRED_FIELDS",
    "CHECKPOINT_FILE_EXTENSION",
    "DEFAULT_CACHE_TTL",
    "DEFAULT_COMPRESSION_THRESHOLD",
    "DEFAULT_MAX_CHECKPOINTS",
    "STATE_TYPES",
    "VALIDATION_ERROR_TYPES",
    # State utilities
    "calculate_state_checksum",
    "create_state_change_record",
    "create_state_metadata",
    "deserialize_state_data",
    "detect_state_changes",
    "ensure_directory_exists",
    "format_cache_key",
    "get_from_redis_with_timeout",
    "serialize_state_data",
    "store_in_redis_with_timeout",
    # State validation utilities
    "validate_bot_id_format",
    "validate_bot_status",
    "validate_capital_allocation",
    "validate_cash_balance",
    "validate_decimal_field_with_details",
    "validate_order_price_logic",
    "validate_positive_value_with_details",
    "validate_required_fields_with_details",
    "validate_strategy_params",
    "validate_symbol_format",
    "validate_var_limits",
]

__version__ = "1.0.0"
__author__ = "Trading Bot Team"
# Service registry available for explicit registration
# from .service_registry import register_util_services

__description__ = "Comprehensive utility framework for trading bot system"
