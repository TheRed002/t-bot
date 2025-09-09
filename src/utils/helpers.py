"""
Helper Functions for Common Operations

This module provides a centralized interface to utility functions from specialized modules.
Instead of duplicating functionality, this module imports and re-exports functions from:
- math_utils: Mathematical calculations and statistical metrics
- datetime_utils: Date/time handling and timezone operations
- data_utils: Data conversion and normalization
- file_utils: File operations and configuration loading
- network_utils: Network connectivity and latency testing
- string_utils: String processing and parsing

Key Functions:
- Mathematical Utilities: statistical calculations, financial metrics
- Date/Time Utilities: timezone handling, trading session detection
- Data Conversion: unit conversions, currency conversions
- File Operations: safe file I/O, configuration loading
- Network Utilities: connection testing, latency measurement
- String Utilities: parsing, formatting, sanitization

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
"""

# Import from P-001 core components first
# Import specialized utility modules

# Import formatting functions needed later

import src.utils.data_utils as data_utils
import src.utils.datetime_utils as datetime_utils
import src.utils.file_utils as file_utils
import src.utils.math_utils as math_utils
import src.utils.network_utils as network_utils
import src.utils.string_utils as string_utils
from src.core.logging import get_logger

# Import new utility modules for backtesting
from src.utils.attribution_structures import (
    create_attribution_summary,
    create_empty_attribution_structure,
    create_empty_service_attribution_structure,
    create_symbol_attribution_summary,
)
from src.utils.config_conversion import convert_config_to_dict
from src.utils.decimal_utils import format_decimal
from src.utils.synthetic_data_generator import generate_synthetic_ohlcv_data, validate_ohlcv_data
from src.utils.timezone_utils import ensure_timezone_aware, ensure_utc_timezone

logger = get_logger(__name__)


# =============================================================================
# Mathematical Utilities - Re-exported from MathUtils
# =============================================================================

# Re-export mathematical functions from math_utils
calculate_percentage_change = math_utils.calculate_percentage_change
calculate_sharpe_ratio = math_utils.calculate_sharpe_ratio
calculate_max_drawdown = math_utils.calculate_max_drawdown
calculate_var = math_utils.calculate_var
calculate_volatility = math_utils.calculate_volatility
calculate_correlation = math_utils.calculate_correlation
calculate_beta = math_utils.calculate_beta
calculate_sortino_ratio = math_utils.calculate_sortino_ratio
safe_min = math_utils.safe_min
safe_max = math_utils.safe_max
safe_percentage = math_utils.safe_percentage


# =============================================================================
# Date/Time Utilities - Re-exported from DateTimeUtils
# =============================================================================

# Re-export datetime functions from datetime_utils
get_trading_session = datetime_utils.get_trading_session
is_market_open = datetime_utils.is_market_open
convert_timezone = datetime_utils.convert_timezone
parse_datetime = datetime_utils.parse_datetime
parse_timeframe = datetime_utils.parse_timeframe
format_timestamp = datetime_utils.to_timestamp
get_redis_key_ttl = datetime_utils.get_redis_key_ttl


# =============================================================================
# Data Conversion Utilities - Re-exported from DataUtils
# =============================================================================

# Re-export data conversion functions from data_utils
convert_currency = data_utils.convert_currency
normalize_price = data_utils.normalize_price
normalize_array = data_utils.normalize_array
dict_to_dataframe = data_utils.dict_to_dataframe
flatten_dict = data_utils.flatten_dict
unflatten_dict = data_utils.unflatten_dict
merge_dicts = data_utils.merge_dicts
filter_none_values = data_utils.filter_none_values
chunk_list = data_utils.chunk_list
format_decimal = format_decimal  # Already imported at top


# =============================================================================
# File Operations - Re-exported from FileUtils
# =============================================================================

# Re-export file operations from file_utils
safe_read_file = file_utils.safe_read_file
safe_write_file = file_utils.safe_write_file
ensure_directory_exists = file_utils.ensure_directory_exists
load_config_file = file_utils.load_config_file
save_config_file = file_utils.save_config_file
delete_file = file_utils.delete_file
get_file_size = file_utils.get_file_size
list_files = file_utils.list_files


# =============================================================================
# Network Utilities - Re-exported from NetworkUtils
# =============================================================================

# Re-export network functions from network_utils
check_connection = network_utils.check_connection
measure_latency = network_utils.measure_latency
ping_host = network_utils.ping_host
check_multiple_hosts = network_utils.check_multiple_hosts
parse_url = network_utils.parse_url
wait_for_service = network_utils.wait_for_service


# =============================================================================
# String Utilities - Re-exported from StringUtils
# =============================================================================

# Re-export string functions from string_utils
sanitize_symbol = string_utils.normalize_symbol  # Alias for compatibility
normalize_symbol = string_utils.normalize_symbol
parse_trading_pair = string_utils.parse_trading_pair
generate_hash = string_utils.generate_hash
validate_email = string_utils.validate_email
extract_numbers = string_utils.extract_numbers
camel_to_snake = string_utils.camel_to_snake
snake_to_camel = string_utils.snake_to_camel
truncate = string_utils.truncate


# =============================================================================
# Technical Analysis Utilities - Legacy Support Removed
# =============================================================================
# All deprecated functions have been removed for production deployment.
# Use decimal_utils.to_decimal() for financial data to maintain precision.


# Technical indicator helper functions moved to dedicated module
# Advanced mathematical and technical analysis functions are available in dedicated modules:
# - math_utils.py: Statistical calculations, z-score, moving averages
# - Technical analysis functions: Available through specialized TA modules


# =============================================================================
# Backtesting Utilities - New utility functions
# =============================================================================

# Re-export backtesting utility functions
convert_config_to_dict = convert_config_to_dict
ensure_timezone_aware = ensure_timezone_aware
ensure_utc_timezone = ensure_utc_timezone
generate_synthetic_ohlcv_data = generate_synthetic_ohlcv_data
validate_ohlcv_data = validate_ohlcv_data

# Attribution structure utilities
create_attribution_summary = create_attribution_summary
create_empty_attribution_structure = create_empty_attribution_structure
create_empty_service_attribution_structure = create_empty_service_attribution_structure
create_symbol_attribution_summary = create_symbol_attribution_summary
