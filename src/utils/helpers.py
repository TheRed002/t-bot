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
from src.core.exceptions import ValidationError
from src.core.logging import get_logger

# Import specialized utility modules
import src.utils.data_utils as data_utils
import src.utils.datetime_utils as datetime_utils
import src.utils.file_utils as file_utils
import src.utils.math_utils as math_utils
import src.utils.network_utils as network_utils
import src.utils.string_utils as string_utils

# Import formatting functions needed later
from src.utils.formatters import format_percentage, format_price

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


# =============================================================================
# Data Conversion Utilities - Re-exported from DataUtils
# =============================================================================

# Re-export data conversion functions from data_utils
convert_currency = data_utils.convert_currency
normalize_price = data_utils.normalize_price
round_to_precision = data_utils.round_to_precision
round_to_precision_decimal = data_utils.round_to_precision_decimal
normalize_array = data_utils.normalize_array
dict_to_dataframe = data_utils.dict_to_dataframe
flatten_dict = data_utils.flatten_dict
unflatten_dict = data_utils.unflatten_dict
merge_dicts = data_utils.merge_dicts
filter_none_values = data_utils.filter_none_values
chunk_list = data_utils.chunk_list


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
test_connection = network_utils.test_connection
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
# Technical Analysis Utilities - Legacy Support
# =============================================================================


def calculate_atr(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> float:
    """
    Calculate Average True Range (ATR) using ta-lib.

    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        period: ATR period (default 14)

    Returns:
        ATR value as float or None if insufficient data

    Raises:
        ValidationError: If data is insufficient or invalid
    """
    try:
        import numpy as np
        try:
            import talib
            HAS_TALIB = True
        except ImportError:
            HAS_TALIB = False

        if HAS_TALIB:
            # Convert to numpy arrays
            high_array = np.array(highs, dtype=np.float64)
            low_array = np.array(lows, dtype=np.float64)
            close_array = np.array(closes, dtype=np.float64)

            # Calculate ATR using ta-lib
            atr = talib.ATR(high_array, low_array, close_array, timeperiod=period)

            # Find the last non-NaN value
            for i in range(len(atr) - 1, -1, -1):
                if not np.isnan(atr[i]):
                    return float(atr[i])
            return None
        else:
            raise ImportError("ta-lib not available")

    except ImportError:
        logger.warning("ta-lib not available, falling back to manual ATR calculation")
        # Fallback to manual calculation
        if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
            raise ValidationError(
                f"Need at least {period + 1} data points for ATR calculation"
            ) from None

        if len(highs) != len(lows) or len(highs) != len(closes):
            raise ValidationError("High, low, and close arrays must have the same length") from None

        # Calculate True Range
        true_ranges = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close_prev = abs(highs[i] - closes[i - 1])
            low_close_prev = abs(lows[i] - closes[i - 1])

            true_range = max(high_low, high_close_prev, low_close_prev)
            true_ranges.append(true_range)

        # Calculate ATR using simple moving average
        if len(true_ranges) < period:
            raise ValidationError(f"Not enough true range data for {period}-period ATR") from None

        # Use the last 'period' true ranges
        recent_true_ranges = true_ranges[-period:]
        atr = sum(recent_true_ranges) / period

        return float(atr)
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        raise ValidationError(f"Failed to calculate ATR: {e}") from e


def calculate_zscore(values: list[float], lookback_period: int = 20) -> float:
    """
    Calculate Z-score for mean reversion analysis.

    Args:
        values: List of values (e.g., prices)
        lookback_period: Number of periods to look back (default 20)

    Returns:
        Z-score as float or None if insufficient data

    Raises:
        ValidationError: If insufficient data or invalid parameters
    """
    try:
        import numpy as np
        try:
            import talib
            HAS_TALIB = True
        except ImportError:
            HAS_TALIB = False

        if HAS_TALIB:
            # Convert to numpy array
            price_array = np.array(values, dtype=np.float64)

            # Calculate Simple Moving Average using ta-lib
            sma = talib.SMA(price_array, timeperiod=lookback_period)

            # Calculate Standard Deviation manually for the lookback period
            if len(sma) > 0 and not np.isnan(sma[-1]):
                mean = sma[-1]
                recent_prices = price_array[-lookback_period:]
                std_dev = np.std(recent_prices)

                if std_dev > 0:
                    current_price = price_array[-1]
                    z_score = (current_price - mean) / std_dev
                    return float(z_score)

            return None
        else:
            raise ImportError("ta-lib not available")

    except ImportError:
        logger.warning("ta-lib not available, falling back to manual Z-score calculation")
        # Fallback to manual calculation
        if len(values) < lookback_period:
            raise ValidationError(
                f"Need at least {lookback_period} values for Z-score calculation"
            ) from None

        if lookback_period <= 0:
            raise ValidationError("Lookback period must be positive") from None

        # Get the recent values for calculation
        recent_values = values[-lookback_period:]

        # Calculate mean and standard deviation
        mean = sum(recent_values) / len(recent_values)

        # Calculate variance using unbiased estimator (n-1 divisor)
        n = len(recent_values)
        variance = sum((x - mean) ** 2 for x in recent_values) / (n - 1) if n > 1 else 0
        std_dev = variance**0.5

        # Avoid division by zero
        if std_dev == 0:
            # If all values are the same, return 0 (no deviation from mean)
            return 0.0

        # Calculate Z-score for the current value
        current_value = values[-1]
        zscore = (current_value - mean) / std_dev

        return float(zscore)
    except Exception as e:
        logger.error(f"Error calculating Z-score: {e}")
        raise ValidationError(f"Failed to calculate Z-score: {e}") from e


def calculate_rsi(prices: list[float], period: int = 14) -> float:
    """
    Calculate Relative Strength Index (RSI) using ta-lib.

    Args:
        prices: List of prices
        period: RSI period (default 14)

    Returns:
        RSI value as float (0-100) or None if insufficient data

    Raises:
        ValidationError: If insufficient data or invalid parameters
    """
    try:
        import numpy as np
        try:
            import talib
            HAS_TALIB = True
        except ImportError:
            HAS_TALIB = False

        if HAS_TALIB:
            # Convert to numpy array
            price_array = np.array(prices, dtype=np.float64)

            # Calculate RSI using ta-lib
            rsi = talib.RSI(price_array, timeperiod=period)

            # Return the last non-NaN value
            if len(rsi) > 0 and not np.isnan(rsi[-1]):
                return float(rsi[-1])
            return None
        else:
            raise ImportError("ta-lib not available")

    except ImportError:
        logger.warning("ta-lib not available, falling back to manual RSI calculation")
        # Fallback to manual calculation
        if len(prices) < period + 1:
            raise ValidationError(
                f"Need at least {period + 1} prices for RSI calculation"
            ) from None

        if period <= 0:
            raise ValidationError("RSI period must be positive") from None

        # Calculate price changes
        changes = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            changes.append(change)

        if len(changes) < period:
            raise ValidationError(f"Not enough price changes for {period}-period RSI") from None

        # Get recent changes
        recent_changes = changes[-period:]

        # Separate gains and losses
        gains = [change if change > 0 else 0 for change in recent_changes]
        losses = [-change if change < 0 else 0 for change in recent_changes]

        # Calculate average gain and loss
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        # Avoid division by zero
        if avg_loss == 0:
            return 100.0

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        raise ValidationError(f"Failed to calculate RSI: {e}") from e


def calculate_moving_average(prices: list[float], period: int, ma_type: str = "sma") -> float:
    """
    Calculate moving average using ta-lib.

    Args:
        prices: List of prices
        period: Moving average period
        ma_type: Type of moving average ("sma", "ema", "wma")

    Returns:
        Moving average value as float or None if insufficient data

    Raises:
        ValidationError: If insufficient data or invalid parameters
    """
    try:
        import numpy as np
        try:
            import talib
            HAS_TALIB = True
        except ImportError:
            HAS_TALIB = False

        if HAS_TALIB:
            # Convert to numpy array
            price_array = np.array(prices, dtype=np.float64)

            if ma_type.lower() == "sma":
                ma = talib.SMA(price_array, timeperiod=period)
            elif ma_type.lower() == "ema":
                ma = talib.EMA(price_array, timeperiod=period)
            elif ma_type.lower() == "wma":
                ma = talib.WMA(price_array, timeperiod=period)
            else:
                logger.error(f"Unsupported moving average type: {ma_type}")
                return None

            # Return the last non-NaN value
            if len(ma) > 0 and not np.isnan(ma[-1]):
                return float(ma[-1])
            return None
        else:
            raise ImportError("ta-lib not available")

    except ImportError:
        logger.warning("ta-lib not available, falling back to manual MA calculation")
        # Fallback to manual calculation
        if len(prices) < period:
            raise ValidationError(
                f"Need at least {period} prices for {period}-period moving average"
            ) from None

        if period <= 0:
            raise ValidationError("Moving average period must be positive") from None

        if ma_type.lower() == "sma":
            # Simple Moving Average
            recent_prices = prices[-period:]
            sma = sum(recent_prices) / period
            return float(sma)

        elif ma_type.lower() == "ema":
            # Exponential Moving Average
            if len(prices) < period:
                raise ValidationError(
                    f"Need at least {period} prices for EMA calculation"
                ) from None

            # Use SMA as initial EMA value
            initial_ema = sum(prices[:period]) / period

            # Calculate multiplier
            multiplier = 2 / (period + 1)

            # Calculate EMA
            ema = initial_ema
            for price in prices[period:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))

            return float(ema)

        else:
            raise ValidationError(f"Unsupported moving average type: {ma_type}") from None
    except Exception as e:
        logger.error(f"Error calculating moving average: {e}")
        raise ValidationError(f"Failed to calculate moving average: {e}") from e


def calculate_support_resistance(
    prices: list[float], lookback_period: int = 20
) -> tuple[float, float]:
    """
    Calculate support and resistance levels.

    Args:
        prices: List of prices
        lookback_period: Number of periods to analyze (default 20)

    Returns:
        Tuple of (support_level, resistance_level) or (None, None) if insufficient data

    Raises:
        ValidationError: If insufficient data
    """

    try:
        import numpy as np

        # Convert to numpy array
        price_array = np.array(prices, dtype=np.float64)

        if len(price_array) < lookback_period:
            return None, None

        # Get recent prices for analysis
        recent_prices = price_array[-lookback_period:]

        # Find local minima and maxima
        support_level = float(np.min(recent_prices))
        resistance_level = float(np.max(recent_prices))

        return support_level, resistance_level

    except Exception as e:
        logger.error(f"Error calculating support/resistance: {e}")
        raise ValidationError(f"Failed to calculate support/resistance: {e}") from e
