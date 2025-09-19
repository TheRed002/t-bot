"""
Shared financial calculation utilities for backtesting module.

This module contains common financial calculations used across backtesting components
to eliminate duplication and ensure consistency.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.decimal_utils import to_decimal
from src.utils.financial_constants import TRADING_DAYS_PER_YEAR, DEFAULT_RISK_FREE_RATE


def calculate_daily_returns(equity_curve: List[Dict[str, Any]]) -> List[float]:
    """
    Calculate daily returns from equity curve.

    Args:
        equity_curve: List of equity data points with timestamp and equity

    Returns:
        List of daily returns as floats
    """
    if not equity_curve:
        return []

    df = pd.DataFrame(equity_curve)
    if df.empty:
        return []

    df.set_index("timestamp", inplace=True)
    daily_equity = df.resample("D")["equity"].last()
    daily_returns = daily_equity.pct_change().dropna().tolist()

    return daily_returns


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: Decimal = Decimal(str(DEFAULT_RISK_FREE_RATE))
) -> Decimal:
    """
    Calculate Sharpe ratio from returns.

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate for excess return calculation

    Returns:
        Sharpe ratio as Decimal
    """
    if len(returns) <= 1:
        return to_decimal("0")

    # Remove NaN values
    returns_clean = returns[~np.isnan(returns)]
    if len(returns_clean) == 0:
        return to_decimal("0")

    annual_factor = to_decimal(str(np.sqrt(TRADING_DAYS_PER_YEAR)))

    mean_return = to_decimal(str(np.mean(returns_clean))) * to_decimal(str(TRADING_DAYS_PER_YEAR))
    volatility = to_decimal(str(np.std(returns_clean))) * annual_factor

    excess_return = mean_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else to_decimal("0")

    return sharpe_ratio


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: Decimal = Decimal(str(DEFAULT_RISK_FREE_RATE))
) -> Decimal:
    """
    Calculate Sortino ratio from returns (downside deviation).

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate for excess return calculation

    Returns:
        Sortino ratio as Decimal
    """
    if len(returns) <= 1:
        return to_decimal("0")

    # Remove NaN values
    returns_clean = returns[~np.isnan(returns)]
    if len(returns_clean) == 0:
        return to_decimal("0")

    annual_factor = to_decimal(str(np.sqrt(TRADING_DAYS_PER_YEAR)))

    mean_return = to_decimal(str(np.mean(returns_clean))) * to_decimal(str(TRADING_DAYS_PER_YEAR))
    excess_return = mean_return - risk_free_rate

    # Calculate downside deviation
    downside_returns = returns_clean[returns_clean < 0]
    if len(downside_returns) > 0:
        downside_std = to_decimal(str(np.std(downside_returns))) * annual_factor
        sortino_ratio = excess_return / downside_std if downside_std > 0 else to_decimal("0")
    else:
        # No downside, use Sharpe ratio instead
        volatility = to_decimal(str(np.std(returns_clean))) * annual_factor
        sortino_ratio = excess_return / volatility if volatility > 0 else to_decimal("0")

    return sortino_ratio


def calculate_max_drawdown(equity_curve: List[Dict[str, Any]]) -> tuple[Decimal, Dict[str, Any]]:
    """
    Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: List of equity data points

    Returns:
        Tuple of (max_drawdown, drawdown_info)
    """
    if not equity_curve:
        return to_decimal("0"), {}

    max_drawdown = to_decimal("0")
    peak_value = to_decimal("0")
    peak_timestamp = None
    trough_value = to_decimal("0")
    trough_timestamp = None
    recovery_timestamp = None

    for point in equity_curve:
        equity = to_decimal(str(point.get("equity", 0)))
        timestamp = point.get("timestamp")

        # Track peak
        if equity > peak_value:
            peak_value = equity
            peak_timestamp = timestamp
            recovery_timestamp = timestamp  # Recovery starts at new peak

        # Calculate current drawdown
        if peak_value > 0:
            drawdown = (peak_value - equity) / peak_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                trough_value = equity
                trough_timestamp = timestamp
                recovery_timestamp = None  # Recovery not yet found

    drawdown_info = {
        "max_drawdown_pct": max_drawdown * to_decimal("100"),
        "peak_value": peak_value,
        "peak_timestamp": peak_timestamp,
        "trough_value": trough_value,
        "trough_timestamp": trough_timestamp,
        "recovery_timestamp": recovery_timestamp,
    }

    return max_drawdown, drawdown_info


def calculate_volatility(returns: np.ndarray) -> Decimal:
    """
    Calculate annualized volatility from returns.

    Args:
        returns: Array of returns

    Returns:
        Annualized volatility as Decimal
    """
    if len(returns) <= 1:
        return to_decimal("0")

    # Remove NaN values
    returns_clean = returns[~np.isnan(returns)]
    if len(returns_clean) == 0:
        return to_decimal("0")

    annual_factor = to_decimal(str(np.sqrt(TRADING_DAYS_PER_YEAR)))
    volatility = to_decimal(str(np.std(returns_clean))) * annual_factor

    return volatility


def calculate_var_cvar(
    returns: np.ndarray,
    confidence_level: float = 0.95
) -> tuple[Decimal, Decimal]:
    """
    Calculate Value at Risk and Conditional Value at Risk.

    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)

    Returns:
        Tuple of (VaR, CVaR) as Decimals
    """
    if len(returns) == 0:
        return to_decimal("0"), to_decimal("0")

    # Remove NaN values
    returns_clean = returns[~np.isnan(returns)]
    if len(returns_clean) == 0:
        return to_decimal("0"), to_decimal("0")

    # Calculate VaR (quantile of losses)
    var_quantile = 1 - confidence_level
    var = to_decimal(str(np.percentile(returns_clean, var_quantile * 100)))

    # Calculate CVaR (average of returns below VaR)
    var_float = float(var)
    tail_returns = returns_clean[returns_clean <= var_float]

    if len(tail_returns) > 0:
        cvar = to_decimal(str(np.mean(tail_returns)))
    else:
        cvar = var

    return abs(var), abs(cvar)


def calculate_market_impact(order_size: Decimal, volume: Decimal) -> Decimal:
    """
    Calculate simple market impact based on order size relative to volume.

    Args:
        order_size: Size of the order
        volume: Market volume

    Returns:
        Market impact as a percentage (Decimal)
    """
    if volume <= 0:
        return to_decimal("0.001")  # Default small impact

    # Simple square-root impact model
    participation_rate = order_size / volume

    # Cap participation rate at 100%
    participation_rate = min(participation_rate, to_decimal("1.0"))

    # Square-root relationship with a scaling factor
    impact = participation_rate ** to_decimal("0.5") * to_decimal("0.001")

    return impact


def calculate_simulation_metrics(
    returns: List[Decimal],
    initial_capital: Decimal
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a single simulation run.

    Args:
        returns: List of returns as Decimals
        initial_capital: Initial capital for the simulation

    Returns:
        Dictionary containing calculated metrics
    """
    if not returns:
        return {}

    # Calculate equity curve
    equity = initial_capital
    equity_curve_data = [{"timestamp": datetime.now(timezone.utc), "equity": equity}]

    for ret in returns:
        equity *= to_decimal("1") + ret
        equity_curve_data.append({
            "timestamp": datetime.now(timezone.utc),
            "equity": equity
        })

    # Calculate drawdown
    max_drawdown, _ = calculate_max_drawdown(equity_curve_data)

    # Convert to numpy array for statistical calculations
    returns_array = np.array([float(r) for r in returns])

    # Calculate metrics
    sharpe = calculate_sharpe_ratio(returns_array)
    sortino = calculate_sortino_ratio(returns_array)
    volatility = calculate_volatility(returns_array)
    var_95, cvar_95 = calculate_var_cvar(returns_array)

    return {
        "total_return": (equity - initial_capital) / initial_capital * to_decimal("100"),
        "max_drawdown": max_drawdown * to_decimal("100"),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "volatility": volatility * to_decimal("100"),
        "var_95": var_95 * to_decimal("100"),
        "cvar_95": cvar_95 * to_decimal("100"),
        "final_equity": equity,
    }


def calculate_max_drawdown_simple(values: List[Decimal]) -> tuple[Decimal, int, int]:
    """
    Calculate maximum drawdown from simple value list.

    Args:
        values: List of portfolio values as Decimals

    Returns:
        Tuple of (max_drawdown, peak_index, trough_index)
    """
    if not values or len(values) < 2:
        return to_decimal("0"), 0, 0

    max_drawdown = to_decimal("0")
    peak_value = values[0]
    current_peak_idx = 0
    max_dd_peak_idx = 0
    max_dd_trough_idx = 0

    for i, value in enumerate(values):
        # Track new peaks
        if value > peak_value:
            peak_value = value
            current_peak_idx = i

        # Calculate current drawdown
        if peak_value > 0:
            drawdown = (peak_value - value) / peak_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_dd_peak_idx = current_peak_idx
                max_dd_trough_idx = i

    return max_drawdown, max_dd_peak_idx, max_dd_trough_idx


def calculate_std(values: List[Decimal]) -> Decimal:
    """
    Calculate standard deviation of decimal values.

    This is a shared utility function extracted from optimization module
    to eliminate code duplication.

    Args:
        values: List of Decimal values

    Returns:
        Standard deviation as Decimal
    """
    if len(values) < 2:
        return Decimal("0")

    mean = sum(values) / Decimal(str(len(values)))
    variance = sum((v - mean) ** 2 for v in values) / Decimal(str(len(values) - 1))

    return variance.sqrt()