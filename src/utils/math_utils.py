"""Mathematical and statistical utilities for the T-Bot trading system."""

from decimal import Decimal

from src.core.exceptions import ValidationError
from src.utils.decimal_utils import ZERO, safe_divide, to_decimal


def calculate_percentage_change(old_value: Decimal, new_value: Decimal) -> Decimal:
    """
    Calculate percentage change between two values.

    Args:
        old_value: Original value as Decimal
        new_value: New value as Decimal

    Returns:
        Percentage change as Decimal (e.g., 0.05 for 5% increase)

    Raises:
        ValidationError: If old_value is zero
    """
    old_decimal = to_decimal(old_value) if not isinstance(old_value, Decimal) else old_value
    new_decimal = to_decimal(new_value) if not isinstance(new_value, Decimal) else new_value

    if old_decimal == ZERO:
        raise ValidationError("Cannot calculate percentage change with zero old value")

    percentage_change = (new_decimal - old_decimal) / old_decimal
    return percentage_change


def calculate_sharpe_ratio(
    returns: list[Decimal], risk_free_rate: Decimal = Decimal("0.02"), frequency: str = "daily"
) -> Decimal:
    """
    Calculate the Sharpe ratio for a series of returns with frequency adjustment.

    Args:
        returns: List of return values as Decimal (e.g., 0.05 for 5%)
        risk_free_rate: Annual risk-free rate as Decimal (default 2%)
        frequency: Data frequency ("daily", "weekly", "monthly", "yearly")

    Returns:
        Sharpe ratio as Decimal

    Raises:
        ValidationError: If returns list is empty or contains invalid values
    """
    import numpy as np
    from src.utils.financial_calculations import calculate_sharpe_ratio as calc_sharpe

    if not returns:
        raise ValidationError("Returns list cannot be empty")

    if len(returns) < 2:
        raise ValidationError("Need at least 2 returns to calculate Sharpe ratio")

    # Validate frequency
    valid_frequencies = {"daily": 252, "weekly": 52, "monthly": 12, "yearly": 1}
    if frequency not in valid_frequencies:
        raise ValidationError(
            f"Invalid frequency: {frequency}. Must be one of {list(valid_frequencies.keys())}"
        )

    # For non-daily frequencies, we need to adjust the underlying calculation
    # The centralized function assumes daily returns (252 trading days)
    frequency_factor = Decimal(str(valid_frequencies[frequency])) / Decimal("252")

    # Convert to numpy array and calculate using centralized function
    returns_array = np.array([float(r) for r in returns])
    base_sharpe = calc_sharpe(returns_array, risk_free_rate)

    # Adjust for frequency if not daily
    if frequency != "daily":
        return base_sharpe * frequency_factor.sqrt()

    return base_sharpe


def calculate_max_drawdown(equity_curve: list[Decimal]) -> tuple[Decimal, int, int]:
    """
    Calculate the maximum drawdown from an equity curve.

    Args:
        equity_curve: List of equity values over time as Decimal

    Returns:
        Tuple of (max_drawdown, start_index, end_index)

    Raises:
        ValidationError: If equity curve is empty or contains invalid values
    """
    from src.utils.financial_calculations import calculate_max_drawdown_simple

    if not equity_curve:
        raise ValidationError("Equity curve cannot be empty")

    if len(equity_curve) < 2:
        # For single point, return no drawdown
        return ZERO, 0, 0

    # Convert to Decimal for precise calculations
    decimal_equity = [to_decimal(e) if not isinstance(e, Decimal) else e for e in equity_curve]

    return calculate_max_drawdown_simple(decimal_equity)


def calculate_var(returns: list[Decimal], confidence_level: Decimal = Decimal("0.95")) -> Decimal:
    """
    Calculate Value at Risk (VaR) for a series of returns.

    Args:
        returns: List of return values as Decimal
        confidence_level: Confidence level for VaR calculation as Decimal (default 95%)

    Returns:
        VaR as Decimal (negative value represents loss)

    Raises:
        ValidationError: If returns list is empty or confidence level is invalid
    """
    if not returns:
        raise ValidationError("Returns list cannot be empty")

    conf_level = (
        to_decimal(confidence_level)
        if not isinstance(confidence_level, Decimal)
        else confidence_level
    )
    if not (ZERO < conf_level < Decimal("1")):
        raise ValidationError("Confidence level must be between 0 and 1")

    # Convert to Decimal for precise calculations
    decimal_returns = [to_decimal(r) if not isinstance(r, Decimal) else r for r in returns]
    decimal_returns.sort()

    # Calculate VaR using historical simulation with Decimal precision
    var_percentile = (Decimal("1") - conf_level) * Decimal("100")
    percentile_index = int((var_percentile / Decimal("100")) * Decimal(len(decimal_returns)))

    # Ensure index is within bounds
    percentile_index = max(0, min(percentile_index, len(decimal_returns) - 1))

    var = decimal_returns[percentile_index]
    return var


def calculate_volatility(returns: list[Decimal], window: int | None = None) -> Decimal:
    """
    Calculate volatility (standard deviation) of returns.

    Args:
        returns: List of Decimal return values
        window: Rolling window size (None for full series)

    Returns:
        Volatility as a Decimal

    Raises:
        ValidationError: If returns list is empty or window is invalid
    """
    if not returns:
        raise ValidationError("Returns list cannot be empty")

    if window is not None and (window <= 0 or window > len(returns)):
        raise ValidationError(f"Invalid window size: {window}")

    # Convert to Decimal for precise calculations
    returns_decimal = [to_decimal(r) if not isinstance(r, Decimal) else r for r in returns]

    if window is None:
        # Calculate volatility for entire series
        selected_returns = returns_decimal
    else:
        # Calculate rolling volatility
        if len(returns_decimal) < window:
            raise ValidationError(f"Not enough data for window size {window}")

        # Use the last window elements
        selected_returns = returns_decimal[-window:]

    # Calculate standard deviation with Decimal precision
    n = Decimal(len(selected_returns))
    if n <= Decimal("1"):
        return ZERO

    mean = sum(selected_returns) / n
    variance = sum((r - mean) ** 2 for r in selected_returns) / (n - Decimal("1"))

    return variance.sqrt()


def calculate_correlation(series1: list[Decimal], series2: list[Decimal]) -> Decimal:
    """
    Calculate correlation coefficient between two series.

    Args:
        series1: First series of Decimal values
        series2: Second series of Decimal values

    Returns:
        Correlation coefficient as a Decimal

    Raises:
        ValidationError: If series are empty or have different lengths
    """
    if not series1 or not series2:
        raise ValidationError("Both series must not be empty")

    if len(series1) != len(series2):
        raise ValidationError("Series must have the same length")

    if len(series1) < 2:
        raise ValidationError("Need at least 2 points to calculate correlation")

    # Convert to Decimal for precise calculations
    series1_decimal = [to_decimal(s) if not isinstance(s, Decimal) else s for s in series1]
    series2_decimal = [to_decimal(s) if not isinstance(s, Decimal) else s for s in series2]

    # Filter out invalid values (None or infinite)
    valid_pairs = []
    for s1, s2 in zip(series1_decimal, series2_decimal, strict=False):
        if s1.is_finite() and s2.is_finite():
            valid_pairs.append((s1, s2))

    if len(valid_pairs) < 2:
        raise ValidationError("Not enough valid data points after removing invalid values")

    # Calculate correlation with Decimal precision
    n = Decimal(len(valid_pairs))

    # Calculate means
    mean1 = sum(pair[0] for pair in valid_pairs) / n
    mean2 = sum(pair[1] for pair in valid_pairs) / n

    # Calculate correlation components
    numerator = sum((pair[0] - mean1) * (pair[1] - mean2) for pair in valid_pairs)
    sum_sq1 = sum((pair[0] - mean1) ** 2 for pair in valid_pairs)
    sum_sq2 = sum((pair[1] - mean2) ** 2 for pair in valid_pairs)

    # Ensure we're working with Decimals
    sum_sq1_decimal = to_decimal(sum_sq1) if sum_sq1 != ZERO else ZERO
    sum_sq2_decimal = to_decimal(sum_sq2) if sum_sq2 != ZERO else ZERO
    numerator_decimal = to_decimal(numerator) if numerator != ZERO else ZERO

    denominator_squared = sum_sq1_decimal * sum_sq2_decimal
    if denominator_squared <= ZERO:
        return ZERO

    denominator = denominator_squared.sqrt()
    return safe_divide(numerator_decimal, denominator)


def calculate_beta(asset_returns: list[Decimal], market_returns: list[Decimal]) -> Decimal:
    """
    Calculate beta coefficient for an asset relative to market.

    Args:
        asset_returns: Asset return series as Decimal
        market_returns: Market return series as Decimal

    Returns:
        Beta coefficient as Decimal

    Raises:
        ValidationError: If series are invalid
    """
    if not asset_returns or not market_returns:
        raise ValidationError("Return series cannot be empty")

    if len(asset_returns) != len(market_returns):
        raise ValidationError("Return series must have the same length")

    # Convert to Decimal for precise calculations
    asset_decimal = [to_decimal(r) if not isinstance(r, Decimal) else r for r in asset_returns]
    market_decimal = [to_decimal(r) if not isinstance(r, Decimal) else r for r in market_returns]

    n = Decimal(len(asset_decimal))
    if n <= Decimal("1"):
        return ZERO

    # Calculate means
    asset_mean = sum(asset_decimal) / n
    market_mean = sum(market_decimal) / n

    # Calculate covariance and market variance
    covariance = sum(
        (a - asset_mean) * (m - market_mean)
        for a, m in zip(asset_decimal, market_decimal, strict=False)
    ) / (n - Decimal("1"))
    market_variance = sum((m - market_mean) ** 2 for m in market_decimal) / (n - Decimal("1"))

    if market_variance <= ZERO:
        return ZERO

    return safe_divide(covariance, market_variance, ZERO)


def calculate_sortino_ratio(
    returns: list[Decimal], risk_free_rate: Decimal = Decimal("0.02"), periods_per_year: int = 252
) -> Decimal:
    """
    Calculate Sortino ratio with period adjustment.

    Args:
        returns: List of Decimal returns
        risk_free_rate: Annual risk-free rate as Decimal
        periods_per_year: Number of periods in a year

    Returns:
        Sortino ratio as Decimal
    """
    import numpy as np
    from src.utils.financial_calculations import calculate_sortino_ratio as calc_sortino

    if len(returns) < 2:
        return ZERO

    # The centralized function assumes 252 periods per year
    # If different periods are requested, adjust the calculation
    periods_factor = Decimal(str(periods_per_year)) / Decimal("252")

    # Convert to numpy array and calculate using centralized function
    returns_array = np.array([float(r) for r in returns])
    base_sortino = calc_sortino(returns_array, risk_free_rate)

    # Adjust for periods if not 252
    if periods_per_year != 252:
        return base_sortino * periods_factor.sqrt()

    return base_sortino


def safe_min(*args: Decimal, default: Decimal | None = None) -> Decimal:
    """
    Safely calculate minimum value using Decimal precision, handling None and invalid inputs.

    Args:
        *args: Decimal values to compare
        default: Default Decimal value if all inputs are None/invalid

    Returns:
        Minimum Decimal value or default

    Raises:
        ValidationError: If no valid values and no default provided
    """
    valid_values = []
    for arg in args:
        if arg is not None:
            try:
                val = to_decimal(arg) if not isinstance(arg, Decimal) else arg
                if val.is_finite():  # Check for NaN and infinity
                    valid_values.append(val)
            except (TypeError, ValueError):
                continue

    if not valid_values:
        if default is not None:
            return default
        raise ValidationError("No valid values provided and no default specified")

    return min(valid_values)


def safe_max(*args: Decimal, default: Decimal | None = None) -> Decimal:
    """
    Safely calculate maximum value using Decimal precision, handling None and invalid inputs.

    Args:
        *args: Decimal values to compare
        default: Default Decimal value if all inputs are None/invalid

    Returns:
        Maximum Decimal value or default

    Raises:
        ValidationError: If no valid values and no default provided
    """
    valid_values = []
    for arg in args:
        if arg is not None:
            try:
                val = to_decimal(arg) if not isinstance(arg, Decimal) else arg
                if val.is_finite():  # Check for NaN and infinity
                    valid_values.append(val)
            except (TypeError, ValueError):
                continue

    if not valid_values:
        if default is not None:
            return default
        raise ValidationError("No valid values provided and no default specified")

    return max(valid_values)


def safe_percentage(value: Decimal, total: Decimal, default: Decimal = ZERO) -> Decimal:
    """
    Safely calculate percentage using Decimal precision, handling zero division and invalid inputs.

    Args:
        value: Numerator value as Decimal
        total: Denominator value as Decimal
        default: Default Decimal value if calculation fails

    Returns:
        Percentage as Decimal (e.g., 15 for 15%)
    """
    from src.utils.decimal_utils import safe_divide

    try:
        # Handle None values
        if value is None or total is None:
            return default

        # Convert to Decimal for precision
        value_decimal = to_decimal(value) if not isinstance(value, Decimal) else value
        total_decimal = to_decimal(total) if not isinstance(total, Decimal) else total
        default_decimal = to_decimal(default) if not isinstance(default, Decimal) else default

        # Use safe_divide from decimal_utils for proper handling and multiply by 100 for percentage
        from src.utils.decimal_utils import HUNDRED

        ratio = safe_divide(value_decimal, total_decimal, default_decimal)
        return ratio * HUNDRED

    except (TypeError, ValueError):
        # Return default on any conversion error
        return to_decimal(default) if not isinstance(default, Decimal) else default
