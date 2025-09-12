
"""
Centralized risk calculation utilities to eliminate code duplication.

This module provides a single source of truth for all risk calculations,
eliminating duplication across risk_metrics.py, core/calculator.py, and services.
"""

from collections.abc import Callable
from decimal import Decimal

import numpy as np

from src.core.logging import get_logger
from src.core.types.risk import RiskLevel
from src.utils.decimal_utils import ONE, ZERO, safe_divide, to_decimal

logger = get_logger(__name__)


def calculate_var(
    returns: list[Decimal], confidence_level: Decimal = to_decimal("0.95"), time_horizon: int = 1
) -> Decimal:
    """
    Calculate Value at Risk (VaR) using percentile method.

    Args:
        returns: Historical returns as Decimal list
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        time_horizon: Time horizon in days

    Returns:
        VaR value as Decimal
    """
    if not returns or len(returns) < 10:
        logger.warning(f"Insufficient data for VaR: {len(returns)} returns")
        return ZERO

    try:
        # Convert to numpy array for calculations
        returns_array = np.array([float(ret) for ret in returns])

        # Calculate VaR using percentile method with Decimal precision
        var_percentile = float((ONE - confidence_level) * to_decimal("100"))
        daily_var = np.percentile(returns_array, var_percentile)

        # Scale for time horizon
        scaling_factor = np.sqrt(time_horizon)
        scaled_var = daily_var * scaling_factor

        return to_decimal(str(abs(scaled_var)))

    except Exception as e:
        logger.error(f"VaR calculation failed: {e}")
        return ZERO


def calculate_expected_shortfall(
    returns: list[Decimal], confidence_level: Decimal = to_decimal("0.95")
) -> Decimal:
    """
    Calculate Expected Shortfall (Conditional VaR).

    Args:
        returns: Historical returns as Decimal list
        confidence_level: Confidence level

    Returns:
        Expected shortfall value as Decimal
    """
    if not returns or len(returns) < 10:
        return ZERO

    try:
        returns_array = np.array([float(ret) for ret in returns])
        var_percentile = float((ONE - confidence_level) * to_decimal("100"))
        var_threshold = np.percentile(returns_array, var_percentile)

        # Calculate mean of returns below VaR threshold
        tail_returns = returns_array[returns_array <= var_threshold]

        if len(tail_returns) == 0:
            return to_decimal(str(abs(var_threshold)))

        # Convert result back to Decimal with precision
        mean_tail = np.mean(tail_returns)
        return to_decimal(str(abs(mean_tail)))

    except Exception as e:
        logger.error(f"Expected shortfall calculation failed: {e}")
        return ZERO


def calculate_sharpe_ratio(
    returns: list[Decimal], risk_free_rate: Decimal = to_decimal("0.02")
) -> Decimal | None:
    """
    Calculate Sharpe ratio with proper Decimal precision.

    Args:
        returns: Historical returns as Decimal list
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Sharpe ratio as Decimal or None if insufficient data
    """
    if not returns or len(returns) < 30:
        return None

    try:
        returns_array = np.array([float(ret) for ret in returns])

        # Annualized metrics with Decimal precision
        mean_return_float = np.mean(returns_array) * 252
        volatility_float = np.std(returns_array) * np.sqrt(252)

        mean_return = to_decimal(str(mean_return_float))
        volatility = to_decimal(str(volatility_float))

        if volatility == ZERO:
            return None

        # Sharpe ratio = (return - risk_free_rate) / volatility
        sharpe_ratio = safe_divide(mean_return - risk_free_rate, volatility, ZERO)
        return sharpe_ratio

    except Exception as e:
        logger.error(f"Sharpe ratio calculation failed: {e}")
        return None


def calculate_max_drawdown(values: list[Decimal]) -> tuple[Decimal, int, int]:
    """
    Calculate maximum drawdown with peak and trough indices.

    Args:
        values: Portfolio values as Decimal list

    Returns:
        Tuple of (max_drawdown, peak_index, trough_index)
    """
    if not values or len(values) < 2:
        return ZERO, 0, 0

    try:
        values_array = np.array([float(val) for val in values])
        running_max = np.maximum.accumulate(values_array)
        drawdowns = (running_max - values_array) / running_max

        max_dd_idx = np.argmax(drawdowns)
        max_drawdown_float = drawdowns[max_dd_idx]

        # Find peak before max drawdown
        peak_idx = np.argmax(values_array[: max_dd_idx + 1]) if max_dd_idx > 0 else 0

        # Convert result back to Decimal with precision
        max_drawdown = to_decimal(str(max_drawdown_float))
        return max_drawdown, int(peak_idx), int(max_dd_idx)

    except Exception as e:
        logger.error(f"Max drawdown calculation failed: {e}")
        return ZERO, 0, 0


def calculate_current_drawdown(current_value: Decimal, historical_values: list[Decimal]) -> Decimal:
    """
    Calculate current drawdown from peak.

    Args:
        current_value: Current portfolio value
        historical_values: Historical portfolio values

    Returns:
        Current drawdown as Decimal
    """
    if not historical_values:
        return ZERO

    try:
        peak_value = max(historical_values)
        if peak_value <= ZERO:
            return ZERO

        current_drawdown = safe_divide(peak_value - current_value, peak_value, ZERO)
        return max(ZERO, current_drawdown)

    except Exception:
        # Avoid using logger in exception path to prevent issues with mocked functions
        return ZERO


def calculate_portfolio_value(positions, market_data) -> Decimal:
    """
    Calculate portfolio value from positions and market data.

    Args:
        positions: List of Position objects
        market_data: List of MarketData objects

    Returns:
        Total portfolio value as Decimal
    """
    portfolio_value = ZERO

    try:
        # Create price lookup
        price_lookup = {data.symbol: data.close for data in market_data}

        for position in positions:
            current_price = price_lookup.get(position.symbol, position.current_price)
            if current_price and current_price > ZERO:
                position_value = position.quantity * current_price
                portfolio_value += position_value

        return portfolio_value

    except Exception as e:
        logger.error(f"Portfolio value calculation failed: {e}")
        return ZERO


def calculate_sortino_ratio(
    returns: list[Decimal],
    risk_free_rate: Decimal = to_decimal("0.02"),
    target_return: Decimal = ZERO,
) -> Decimal:
    """
    Calculate Sortino ratio.

    Args:
        returns: Historical returns
        risk_free_rate: Risk-free rate
        target_return: Target return for downside deviation

    Returns:
        Sortino ratio as Decimal
    """
    if not returns or len(returns) < 10:
        return ZERO

    try:
        returns_array = np.array([float(ret) for ret in returns])
        excess_returns = returns_array - float(risk_free_rate)

        # Calculate downside deviation
        downside_returns = np.minimum(returns_array - float(target_return), 0)
        downside_deviation = np.std(downside_returns)

        if downside_deviation == 0:
            return ZERO

        # Convert result back to Decimal with precision
        sortino_float = np.mean(excess_returns) / downside_deviation
        return to_decimal(str(sortino_float))

    except Exception as e:
        logger.error(f"Sortino ratio calculation failed: {e}")
        return ZERO


def calculate_calmar_ratio(returns: list[Decimal], period_years: Decimal = ONE) -> Decimal:
    """
    Calculate Calmar ratio.

    Args:
        returns: Historical returns
        period_years: Period in years

    Returns:
        Calmar ratio as Decimal
    """
    if not returns:
        return ZERO

    try:
        returns_array = np.array([float(ret) for ret in returns])
        annual_return_float = np.mean(returns_array) * 252
        annual_return = to_decimal(str(annual_return_float))

        max_dd, _, _ = calculate_max_drawdown(returns)

        if max_dd == ZERO:
            return ZERO

        return safe_divide(annual_return, max_dd, ZERO)

    except Exception as e:
        logger.error(f"Calmar ratio calculation failed: {e}")
        return ZERO


def determine_risk_level(
    var_1d: Decimal,
    current_drawdown: Decimal,
    sharpe_ratio: Decimal | None,
    portfolio_value: Decimal,
) -> RiskLevel:
    """
    Determine risk level based on multiple metrics.

    Args:
        var_1d: 1-day Value at Risk
        current_drawdown: Current drawdown
        sharpe_ratio: Sharpe ratio (can be None)
        portfolio_value: Current portfolio value

    Returns:
        Risk level classification
    """
    try:
        # Calculate VaR as percentage of portfolio
        var_pct = safe_divide(var_1d, portfolio_value, ZERO) if portfolio_value > ZERO else ZERO

        # Risk scoring system
        risk_score = 0

        # VaR scoring
        if var_pct > to_decimal("0.10"):  # > 10%
            risk_score += 3
        elif var_pct > to_decimal("0.05"):  # > 5%
            risk_score += 2
        elif var_pct > to_decimal("0.02"):  # > 2%
            risk_score += 1

        # Drawdown scoring
        if current_drawdown > to_decimal("0.20"):  # > 20%
            risk_score += 3
        elif current_drawdown > to_decimal("0.10"):  # > 10%
            risk_score += 2
        elif current_drawdown > to_decimal("0.05"):  # > 5%
            risk_score += 1

        # Sharpe ratio scoring (inverse)
        if sharpe_ratio is not None:
            if sharpe_ratio < to_decimal("-1.0"):
                risk_score += 3
            elif sharpe_ratio < ZERO:
                risk_score += 2
            elif sharpe_ratio < to_decimal("0.5"):
                risk_score += 1

        # Map score to risk level
        if risk_score >= 7:
            return RiskLevel.CRITICAL
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    except Exception as e:
        logger.error(f"Risk level determination failed: {e}")
        return RiskLevel.MEDIUM  # Safe default


def update_returns_history(
    current_values: list[Decimal],
    update_callback: Callable[[list[Decimal]], None] | None = None,
    max_history: int = 252,
) -> list[Decimal]:
    """
    Update returns history with proper memory management.

    Args:
        current_values: Current portfolio values
        update_callback: Optional callback for updates
        max_history: Maximum history length

    Returns:
        Calculated returns list
    """
    if len(current_values) < 2:
        return []

    try:
        returns = []
        for i in range(1, len(current_values)):
            if current_values[i - 1] > ZERO:
                return_val = safe_divide(
                    current_values[i] - current_values[i - 1], current_values[i - 1], ZERO
                )
                returns.append(return_val)

        # Keep only recent history
        if len(returns) > max_history:
            returns = returns[-max_history:]

        if update_callback:
            update_callback(returns)

        return returns

    except Exception as e:
        logger.error(f"Returns history update failed: {e}")
        return []


def validate_risk_inputs(
    portfolio_value: Decimal,
    positions: list,
    market_data: list,
    min_portfolio_value: Decimal = to_decimal("100"),
) -> bool:
    """
    Validate risk calculation inputs.

    Args:
        portfolio_value: Portfolio value to validate
        positions: Positions list
        market_data: Market data list
        min_portfolio_value: Minimum required portfolio value

    Returns:
        True if inputs are valid
    """
    try:
        # Check portfolio value
        if portfolio_value < min_portfolio_value:
            logger.warning(f"Portfolio value {portfolio_value} below minimum {min_portfolio_value}")
            return False

        # Check data alignment
        if positions and market_data:
            if len(positions) != len(market_data):
                logger.warning("Position and market data count mismatch")
                return False

        # Validate individual positions
        for position in positions:
            if not hasattr(position, "symbol") or not position.symbol:
                logger.warning("Position missing symbol")
                return False
            if not hasattr(position, "quantity") or position.quantity <= ZERO:
                logger.warning(f"Invalid position quantity for {position.symbol}")
                return False

        return True

    except Exception:
        # Avoid using logger in exception path to prevent issues with mocked functions
        return False
