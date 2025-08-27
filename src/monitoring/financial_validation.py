"""
Financial validation utilities for trading system monitoring.

This module provides comprehensive validation functions for financial metrics
used throughout the T-Bot trading system monitoring components.
"""

import logging
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)

# Financial precision constants
CRYPTO_DECIMAL_PLACES = 8  # Standard precision for crypto calculations
FIAT_DECIMAL_PLACES = 2  # Standard precision for fiat currency
BPS_DECIMAL_PLACES = 2  # Basis points precision
PERCENTAGE_DECIMAL_PLACES = 4  # Percentage precision

# Financial bounds for validation
MAX_TRADE_VALUE_USD = 10_000_000  # $10M max per trade
MAX_PORTFOLIO_VALUE_USD = 1_000_000_000  # $1B max portfolio
MAX_SLIPPAGE_BPS = 10_000  # 100% max slippage
MAX_EXECUTION_TIME_SECONDS = 3600  # 1 hour max execution
MAX_VAR_USD = 100_000_000  # $100M max VaR
MAX_DRAWDOWN_PERCENT = 100  # 100% max drawdown
MAX_SHARPE_RATIO = 10  # Practical limit for Sharpe ratio


def validate_price(price: float | int | Decimal, symbol: str = "UNKNOWN") -> Decimal:
    """
    Validate and normalize a price value.

    Args:
        price: Price value to validate
        symbol: Trading symbol for context in error messages

    Returns:
        Validated price as Decimal with appropriate precision

    Raises:
        ValueError: If price is invalid
    """
    if price is None:
        raise ValueError(f"Price cannot be None for {symbol}")

    try:
        price_decimal = Decimal(str(price))
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"Invalid price format for {symbol}: {price}") from e

    if price_decimal <= 0:
        raise ValueError(f"Price must be positive for {symbol}: {price_decimal}")

    if price_decimal > Decimal("1000000"):  # $1M per unit (sanity check)
        logger.warning(f"Unusually high price for {symbol}: ${price_decimal}")

    # Round to crypto precision
    return price_decimal.quantize(Decimal("0.00000001"))


def validate_quantity(quantity: float | int | Decimal, symbol: str = "UNKNOWN") -> Decimal:
    """
    Validate and normalize a quantity value.

    Args:
        quantity: Quantity value to validate
        symbol: Trading symbol for context

    Returns:
        Validated quantity as Decimal with appropriate precision

    Raises:
        ValueError: If quantity is invalid
    """
    if quantity is None:
        raise ValueError(f"Quantity cannot be None for {symbol}")

    try:
        quantity_decimal = Decimal(str(quantity))
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"Invalid quantity format for {symbol}: {quantity}") from e

    if quantity_decimal <= 0:
        raise ValueError(f"Quantity must be positive for {symbol}: {quantity_decimal}")

    if quantity_decimal > Decimal("1000000000"):  # 1B units (sanity check)
        logger.warning(f"Unusually large quantity for {symbol}: {quantity_decimal}")

    # Round to crypto precision
    return quantity_decimal.quantize(Decimal("0.00000001"))


def validate_pnl_usd(pnl_usd: float | int, context: str = "") -> float:
    """
    Validate P&L in USD.

    Args:
        pnl_usd: P&L value in USD
        context: Context for error messages

    Returns:
        Validated and rounded P&L value

    Raises:
        ValueError: If P&L value is invalid
    """
    if not isinstance(pnl_usd, int | float):
        raise ValueError(f"P&L must be numeric {context}")

    if abs(pnl_usd) > MAX_TRADE_VALUE_USD:
        logger.warning(f"Large P&L value {context}: ${pnl_usd:,.2f}")

    return round(pnl_usd, FIAT_DECIMAL_PLACES)


def validate_volume_usd(volume_usd: float | int, context: str = "") -> float:
    """
    Validate trading volume in USD.

    Args:
        volume_usd: Volume value in USD
        context: Context for error messages

    Returns:
        Validated and rounded volume value

    Raises:
        ValueError: If volume is invalid
    """
    if not isinstance(volume_usd, int | float):
        raise ValueError(f"Volume must be numeric {context}")

    if volume_usd < 0:
        raise ValueError(f"Volume cannot be negative {context}")

    if volume_usd > MAX_TRADE_VALUE_USD:
        logger.warning(f"Large volume {context}: ${volume_usd:,.2f}")

    return round(volume_usd, FIAT_DECIMAL_PLACES)


def validate_slippage_bps(slippage_bps: float | int, context: str = "") -> float:
    """
    Validate slippage in basis points.

    Args:
        slippage_bps: Slippage in basis points
        context: Context for error messages

    Returns:
        Validated and rounded slippage value

    Raises:
        ValueError: If slippage is invalid
    """
    if not isinstance(slippage_bps, int | float):
        raise ValueError(f"Slippage must be numeric {context}")

    if abs(slippage_bps) > MAX_SLIPPAGE_BPS:
        raise ValueError(f"Invalid slippage {context}: {slippage_bps} bps (exceeds 100%)")

    if abs(slippage_bps) > 1000:  # > 10% slippage is unusual
        logger.warning(f"High slippage {context}: {slippage_bps:.2f} bps")

    return round(slippage_bps, BPS_DECIMAL_PLACES)


def validate_execution_time(execution_time: float | int, context: str = "") -> float:
    """
    Validate order execution time.

    Args:
        execution_time: Execution time in seconds
        context: Context for error messages

    Returns:
        Validated execution time

    Raises:
        ValueError: If execution time is invalid
    """
    if not isinstance(execution_time, int | float):
        raise ValueError(f"Execution time must be numeric {context}")

    if execution_time < 0:
        raise ValueError(f"Execution time cannot be negative {context}")

    if execution_time > MAX_EXECUTION_TIME_SECONDS:
        logger.warning(f"Long execution time {context}: {execution_time:.2f}s")

    return round(execution_time, 6)  # Microsecond precision


def validate_var(var_value: float | int, confidence_level: float, context: str = "") -> float:
    """
    Validate Value at Risk.

    Args:
        var_value: VaR value in USD
        confidence_level: VaR confidence level
        context: Context for error messages

    Returns:
        Validated VaR value

    Raises:
        ValueError: If VaR parameters are invalid
    """
    if not isinstance(var_value, int | float):
        raise ValueError(f"VaR value must be numeric {context}")

    if not 0.5 <= confidence_level <= 0.999:
        raise ValueError(f"Invalid VaR confidence level {context}: {confidence_level}")

    if var_value < 0:
        logger.warning(f"Negative VaR {context}: ${var_value:,.2f}")

    if var_value > MAX_VAR_USD:
        logger.warning(f"Extremely high VaR {context}: ${var_value:,.2f}")

    return round(var_value, FIAT_DECIMAL_PLACES)


def validate_drawdown_percent(drawdown_pct: float | int, context: str = "") -> float:
    """
    Validate drawdown percentage.

    Args:
        drawdown_pct: Drawdown percentage
        context: Context for error messages

    Returns:
        Validated drawdown percentage

    Raises:
        ValueError: If drawdown is invalid
    """
    if not isinstance(drawdown_pct, int | float):
        raise ValueError(f"Drawdown must be numeric {context}")

    if drawdown_pct < 0:
        raise ValueError(f"Drawdown must be positive (represents loss) {context}")

    if drawdown_pct > MAX_DRAWDOWN_PERCENT:
        raise ValueError(f"Invalid drawdown {context}: {drawdown_pct}%")

    if drawdown_pct > 20:  # > 20% drawdown is concerning
        logger.warning(f"High drawdown {context}: {drawdown_pct:.2f}%")

    return round(drawdown_pct, PERCENTAGE_DECIMAL_PLACES)


def validate_sharpe_ratio(sharpe_ratio: float | int, context: str = "") -> float:
    """
    Validate Sharpe ratio.

    Args:
        sharpe_ratio: Sharpe ratio value
        context: Context for error messages

    Returns:
        Validated Sharpe ratio

    Raises:
        ValueError: If Sharpe ratio is invalid
    """
    if not isinstance(sharpe_ratio, int | float):
        raise ValueError(f"Sharpe ratio must be numeric {context}")

    if abs(sharpe_ratio) > MAX_SHARPE_RATIO:
        raise ValueError(f"Unrealistic Sharpe ratio {context}: {sharpe_ratio}")

    if sharpe_ratio > 2:
        logger.info(f"Excellent Sharpe ratio {context}: {sharpe_ratio:.3f}")
    elif sharpe_ratio < -1:
        logger.warning(f"Poor Sharpe ratio {context}: {sharpe_ratio:.3f}")

    return round(sharpe_ratio, PERCENTAGE_DECIMAL_PLACES)


def validate_portfolio_value(value_usd: float | int, context: str = "") -> float:
    """
    Validate portfolio value.

    Args:
        value_usd: Portfolio value in USD
        context: Context for error messages

    Returns:
        Validated portfolio value

    Raises:
        ValueError: If portfolio value is invalid
    """
    if not isinstance(value_usd, int | float):
        raise ValueError(f"Portfolio value must be numeric {context}")

    if value_usd < 0:
        raise ValueError(f"Portfolio value cannot be negative {context}")

    if value_usd > MAX_PORTFOLIO_VALUE_USD:
        logger.warning(f"Large portfolio value {context}: ${value_usd:,.2f}")

    return round(value_usd, FIAT_DECIMAL_PLACES)


def validate_timeframe(timeframe: str) -> str:
    """
    Validate trading timeframe.

    Args:
        timeframe: Timeframe string

    Returns:
        Validated timeframe

    Raises:
        ValueError: If timeframe is invalid
    """
    valid_timeframes = [
        "1s",
        "5s",
        "15s",
        "30s",
        "1m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "12h",
        "1d",
        "1w",
        "1M",
        "3M",
        "1y",
    ]

    if timeframe not in valid_timeframes:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}")

    return timeframe


def calculate_pnl_percentage(pnl_usd: float, portfolio_value: float) -> float:
    """
    Calculate P&L percentage with validation.

    Args:
        pnl_usd: P&L in USD
        portfolio_value: Portfolio value in USD

    Returns:
        P&L percentage

    Raises:
        ValueError: If calculation is invalid
    """
    if portfolio_value <= 0:
        raise ValueError("Cannot calculate P&L percentage with zero or negative portfolio value")

    pnl_percentage = (pnl_usd / portfolio_value) * 100

    if abs(pnl_percentage) > 100:
        logger.warning(f"Extreme P&L percentage: {pnl_percentage:.2f}%")

    return round(pnl_percentage, PERCENTAGE_DECIMAL_PLACES)


def validate_position_size_usd(size_usd: float | int, context: str = "") -> float:
    """
    Validate position size in USD.

    Args:
        size_usd: Position size in USD
        context: Context for error messages

    Returns:
        Validated position size

    Raises:
        ValueError: If position size is invalid
    """
    if not isinstance(size_usd, int | float):
        raise ValueError(f"Position size must be numeric {context}")

    if size_usd < 0:
        raise ValueError(f"Position size cannot be negative {context}")

    if size_usd > MAX_TRADE_VALUE_USD:
        logger.warning(f"Large position size {context}: ${size_usd:,.2f}")

    return round(size_usd, FIAT_DECIMAL_PLACES)
