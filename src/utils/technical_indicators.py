"""
Shared Technical Indicators Utilities

This module consolidates duplicate technical indicator calculations from
vectorized_processor.py and features/technical_indicators.py into a single
shared location to eliminate code duplication and ensure consistency.

All calculations use Decimal for financial precision and include proper
error handling.
"""

from decimal import Decimal, getcontext
from typing import Any

import numpy as np

from src.utils.decimal_utils import to_decimal

talib: Any | None = None
try:
    import talib as _talib_module  # type: ignore[import]

    talib = _talib_module
except ImportError:
    pass

try:
    from numba import jit, prange
except ImportError:
    # Create dummy decorators if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range

from src.core.exceptions import ValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)


def _check_talib():
    """Check if talib is available and raise error if not."""
    if talib is None:
        raise ValidationError(
            "TA-Lib not installed. Please install TA-Lib library for technical indicators."
        )


# Vectorized calculations using Numba for high-performance processing
@jit(nopython=True, parallel=True, cache=True)
def calculate_sma_vectorized(prices: np.ndarray, period: int) -> np.ndarray:
    """Vectorized Simple Moving Average calculation."""
    n = len(prices)
    sma = np.empty(n, dtype=np.float64)

    # Fill initial values with NaN
    for i in prange(period - 1):
        sma[i] = np.nan

    # Calculate SMA for valid periods
    for i in prange(period - 1, n):
        sma[i] = np.mean(prices[i - period + 1 : i + 1])

    return sma


@jit(nopython=True, parallel=True, cache=True)
def calculate_ema_vectorized(prices: np.ndarray, period: int) -> np.ndarray:
    """High-performance Exponential Moving Average calculation."""
    n = len(prices)
    alpha = 2.0 / (period + 1.0)
    ema = np.empty(n, dtype=np.float64)

    # Initialize first EMA value
    ema[0] = prices[0]

    # Calculate EMA using vectorized operations
    for i in prange(1, n):
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1]

    return ema


@jit(nopython=True, parallel=True, cache=True)
def calculate_rsi_vectorized(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Vectorized RSI calculation with SIMD optimizations."""
    n = len(prices)
    rsi = np.empty(n, dtype=np.float64)

    # Calculate price changes
    deltas = np.diff(prices)

    # Separate gains and losses
    gains = np.maximum(deltas, 0.0)
    losses = -np.minimum(deltas, 0.0)

    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Fill initial values
    for i in prange(period):
        rsi[i] = np.nan

    # Calculate RSI using Wilder's smoothing
    for i in prange(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@jit(nopython=True, parallel=True, cache=True)
def calculate_bollinger_bands_vectorized(
    prices: np.ndarray, period: int = 20, std_dev: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Bollinger Bands calculation."""
    n = len(prices)
    middle_band = np.empty(n, dtype=np.float64)
    upper_band = np.empty(n, dtype=np.float64)
    lower_band = np.empty(n, dtype=np.float64)

    # Fill initial values with NaN
    for i in prange(period - 1):
        middle_band[i] = np.nan
        upper_band[i] = np.nan
        lower_band[i] = np.nan

    # Calculate bands
    for i in prange(period - 1, n):
        window = prices[i - period + 1 : i + 1]
        mean_val = np.mean(window)
        std_val = np.std(window)

        middle_band[i] = mean_val
        upper_band[i] = mean_val + (std_dev * std_val)
        lower_band[i] = mean_val - (std_dev * std_val)

    return upper_band, middle_band, lower_band


@jit(nopython=True, parallel=True, cache=True)
def calculate_macd_vectorized(
    prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized MACD calculation."""
    ema_fast = calculate_ema_vectorized(prices, fast_period)
    ema_slow = calculate_ema_vectorized(prices, slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema_vectorized(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# TALib-based calculations with Decimal precision for financial accuracy
def calculate_sma_talib(prices: np.ndarray, period: int) -> Decimal | None:
    """Calculate Simple Moving Average using TALib with Decimal precision."""
    _check_talib()
    try:
        if len(prices) < period:
            return None

        sma_values = talib.SMA(prices, timeperiod=period)
        if np.isnan(sma_values[-1]):
            return None

        # Convert to Decimal with proper financial precision (8 decimal places for crypto)
        getcontext().prec = 16
        return to_decimal(str(sma_values[-1])).quantize(Decimal("0.00000001"))
    except Exception as e:
        logger.error(f"SMA calculation failed: {e}")
        return None


def calculate_ema_talib(prices: np.ndarray, period: int) -> Decimal | None:
    """Calculate Exponential Moving Average using TALib with Decimal precision."""
    _check_talib()
    try:
        if len(prices) < period:
            return None

        ema_values = talib.EMA(prices, timeperiod=period)
        if np.isnan(ema_values[-1]):
            return None

        # Convert to Decimal with proper financial precision (8 decimal places for crypto)
        getcontext().prec = 16
        return to_decimal(str(ema_values[-1])).quantize(Decimal("0.00000001"))
    except Exception as e:
        logger.error(f"EMA calculation failed: {e}")
        return None


def calculate_rsi_talib(prices: np.ndarray, period: int = 14) -> Decimal | None:
    """Calculate Relative Strength Index using TALib with Decimal precision."""
    try:
        if len(prices) < period + 1:
            return None

        rsi_values = talib.RSI(prices, timeperiod=period)
        if np.isnan(rsi_values[-1]):
            return None

        # RSI is a percentage, use 4 decimal places for precision
        getcontext().prec = 16
        return to_decimal(str(rsi_values[-1])).quantize(Decimal("0.0001"))
    except Exception as e:
        logger.error(f"RSI calculation failed: {e}")
        return None


def calculate_macd_talib(
    prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> dict[str, Decimal] | None:
    """Calculate MACD using TALib with Decimal precision."""
    try:
        if len(prices) < slow + signal:
            return None

        macd_line, macd_signal, macd_histogram = talib.MACD(
            prices, fastperiod=fast, slowperiod=slow, signalperiod=signal
        )

        if not (
            np.isnan(macd_line[-1]) or np.isnan(macd_signal[-1]) or np.isnan(macd_histogram[-1])
        ):
            # Convert to Decimal with proper financial precision (8 decimal places for crypto)
            getcontext().prec = 16
            return {
                "macd": to_decimal(str(macd_line[-1])).quantize(Decimal("0.00000001")),
                "signal": to_decimal(str(macd_signal[-1])).quantize(Decimal("0.00000001")),
                "histogram": to_decimal(str(macd_histogram[-1])).quantize(Decimal("0.00000001")),
            }
        return None
    except Exception as e:
        logger.error(f"MACD calculation failed: {e}")
        return None


def calculate_bollinger_bands_talib(
    prices: np.ndarray, period: int = 20, std_dev: Decimal = Decimal("2")
) -> dict[str, Decimal] | None:
    """Calculate Bollinger Bands using TALib with Decimal precision."""
    try:
        if len(prices) < period:
            return None

        # Convert std_dev to float for talib compatibility
        std_dev_float = float(std_dev) if isinstance(std_dev, Decimal) else std_dev
        upper, middle, lower = talib.BBANDS(
            prices, timeperiod=period, nbdevup=std_dev_float, nbdevdn=std_dev_float
        )

        if not (np.isnan(upper[-1]) or np.isnan(middle[-1]) or np.isnan(lower[-1])):
            # Convert to Decimal with proper financial precision (8 decimal places for crypto)
            getcontext().prec = 16
            upper_decimal = to_decimal(str(upper[-1])).quantize(Decimal("0.00000001"))
            middle_decimal = to_decimal(str(middle[-1])).quantize(Decimal("0.00000001"))
            lower_decimal = to_decimal(str(lower[-1])).quantize(Decimal("0.00000001"))
            current_price_decimal = to_decimal(str(prices[-1])).quantize(Decimal("0.00000001"))

            width = upper_decimal - lower_decimal
            position = (
                (current_price_decimal - lower_decimal) / width
                if width > Decimal("0")
                else Decimal("0")
            )

            return {
                "upper": upper_decimal,
                "middle": middle_decimal,
                "lower": lower_decimal,
                "width": width,
                "position": position.quantize(Decimal("0.0001")),
            }
        return None
    except Exception as e:
        logger.error(f"Bollinger Bands calculation failed: {e}")
        return None


def calculate_atr_talib(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> Decimal | None:
    """Calculate Average True Range using TALib with Decimal precision."""
    try:
        if len(high) < period or len(low) < period or len(close) < period:
            return None

        atr_values = talib.ATR(high, low, close, timeperiod=period)
        if np.isnan(atr_values[-1]):
            return None

        # Convert to Decimal with proper financial precision (8 decimal places for crypto)
        getcontext().prec = 16
        return to_decimal(str(atr_values[-1])).quantize(Decimal("0.00000001"))
    except Exception as e:
        logger.error(f"ATR calculation failed: {e}")
        return None
