
"""
Synthetic Data Generation Utilities for Backtesting.

Provides consistent synthetic market data generation for testing purposes
using geometric Brownian motion and proper OHLCV relationships.
"""

from datetime import datetime

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: str = "1h",
    initial_price: float = 100.0,
    annual_drift: float = 0.1,
    annual_volatility: float = 0.2,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data using geometric Brownian motion.

    Args:
        symbol: Trading symbol (used for consistent random seed)
        start_date: Start date for data generation
        end_date: End date for data generation
        timeframe: Data timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        initial_price: Starting price for the series
        annual_drift: Annual return drift (mu)
        annual_volatility: Annual volatility (sigma)
        seed: Random seed override (uses hash of symbol if None)

    Returns:
        DataFrame with OHLCV columns indexed by timestamp
    """
    # Parse timeframe mapping
    freq_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
    }

    freq = freq_map.get(timeframe, "1h")
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    if len(dates) == 0:
        return pd.DataFrame()

    # Set consistent random seed based on symbol
    if seed is None:
        seed = hash(symbol) % 2**32
    np.random.seed(seed)

    # Generate price series using geometric Brownian motion
    n = len(dates)
    dt = 1 / 252  # Daily time step

    # Generate returns
    returns = np.random.normal(annual_drift * dt, annual_volatility * np.sqrt(dt), n)
    prices = initial_price * np.exp(np.cumsum(returns))

    # Ensure prices is always an array
    if np.isscalar(prices):
        prices = np.array([prices])
    elif not isinstance(prices, np.ndarray):
        prices = np.array(prices)

    # Generate OHLC from prices with proper relationships
    df = pd.DataFrame(index=dates)
    df["close"] = prices

    # Open price: close of previous period plus small noise
    open_noise = np.random.normal(0, 0.001, n)
    if np.isscalar(open_noise):
        open_noise = np.array([open_noise])
    elif not isinstance(open_noise, np.ndarray):
        open_noise = np.array(open_noise)
    df["open"] = prices * (1 + open_noise)

    # High and low with proper OHLC relationships
    high_offsets = np.abs(np.random.normal(0, 0.005, n))
    low_offsets = np.abs(np.random.normal(0, 0.005, n))
    if np.isscalar(high_offsets):
        high_offsets = np.array([high_offsets])
    if np.isscalar(low_offsets):
        low_offsets = np.array([low_offsets])

    # High is maximum of open, close, and upward price movement
    df["high"] = np.maximum(np.maximum(df["open"], df["close"]), prices * (1 + high_offsets))

    # Low is minimum of open, close, and downward price movement
    df["low"] = np.minimum(np.minimum(df["open"], df["close"]), prices * (1 - low_offsets))

    # Volume with some randomness
    df["volume"] = np.random.uniform(1000, 10000, n)

    return df


def generate_timeframe_mapping() -> dict[str, str]:
    """
    Get timeframe to pandas frequency mapping.

    Returns:
        Dictionary mapping trading timeframes to pandas frequency strings
    """
    return {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
    }


def validate_ohlcv_data(data: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate OHLCV data for proper relationships and values.

    Args:
        data: DataFrame with OHLCV columns

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check required columns
    required_columns = ["open", "high", "low", "close", "volume"]
    missing = set(required_columns) - set(data.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return False, errors

    # Check for NaN values
    nan_counts = data[required_columns].isna().sum()
    if nan_counts.any():
        errors.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")

    # Check OHLC relationships
    invalid_high_low = (data["high"] < data["low"]).sum()
    if invalid_high_low > 0:
        errors.append(f"Invalid high < low relationships: {invalid_high_low} rows")

    invalid_high_open = (data["high"] < data["open"]).sum()
    if invalid_high_open > 0:
        errors.append(f"Invalid high < open relationships: {invalid_high_open} rows")

    invalid_high_close = (data["high"] < data["close"]).sum()
    if invalid_high_close > 0:
        errors.append(f"Invalid high < close relationships: {invalid_high_close} rows")

    invalid_low_open = (data["low"] > data["open"]).sum()
    if invalid_low_open > 0:
        errors.append(f"Invalid low > open relationships: {invalid_low_open} rows")

    invalid_low_close = (data["low"] > data["close"]).sum()
    if invalid_low_close > 0:
        errors.append(f"Invalid low > close relationships: {invalid_low_close} rows")

    # Check for negative values
    negative_counts = (data[required_columns] < 0).sum()
    if negative_counts.any():
        errors.append(f"Negative values found: {negative_counts[negative_counts > 0].to_dict()}")

    is_valid = len(errors) == 0
    return is_valid, errors
