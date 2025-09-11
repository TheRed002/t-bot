"""
Financial validation utilities for trading system monitoring.

This module provides comprehensive validation functions for financial metrics
used throughout the T-Bot trading system monitoring components.
"""

import logging
from decimal import Decimal, InvalidOperation

from src.monitoring.financial_precision import _FINANCIAL_DECIMAL_CONTEXT

# Unused imports removed for production cleanup

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


def validate_pnl_usd(pnl_usd: float | int | Decimal, context: str = "") -> Decimal:
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
    if not isinstance(pnl_usd, (int, float, Decimal)):
        raise ValueError(f"P&L must be numeric {context}")

    # Convert to Decimal for precise calculations
    pnl_decimal = Decimal(str(pnl_usd))

    if abs(pnl_decimal) > Decimal(str(MAX_TRADE_VALUE_USD)):
        logger.warning(f"Large P&L value {context}: ${pnl_decimal}")

    return pnl_decimal.quantize(Decimal("0.01"))  # FIAT precision


def validate_volume_usd(volume_usd: float | int | Decimal, context: str = "") -> Decimal:
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
    if not isinstance(volume_usd, (int, float, Decimal)):
        raise ValueError(f"Volume must be numeric {context}")

    # Convert to Decimal for precise calculations
    volume_decimal = Decimal(str(volume_usd))

    if volume_decimal < 0:
        raise ValueError(f"Volume cannot be negative {context}")

    if volume_decimal > Decimal(str(MAX_TRADE_VALUE_USD)):
        logger.warning(f"Large volume {context}: ${volume_decimal}")

    return volume_decimal.quantize(Decimal("0.01"))  # FIAT precision


def validate_slippage_bps(slippage_bps: float | int | Decimal, context: str = "") -> Decimal:
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
    if not isinstance(slippage_bps, (int, float, Decimal)):
        raise ValueError(f"Slippage must be numeric {context}")

    # Convert to Decimal for precise calculations
    slippage_decimal = Decimal(str(slippage_bps))

    if abs(slippage_decimal) > Decimal(str(MAX_SLIPPAGE_BPS)):
        raise ValueError(f"Invalid slippage {context}: {slippage_decimal} bps (exceeds 100%)")

    if abs(slippage_decimal) > 1000:  # > 10% slippage is unusual
        logger.warning(f"High slippage {context}: {slippage_decimal} bps")

    return slippage_decimal.quantize(Decimal("0.01"))  # BPS precision


def validate_execution_time(execution_time: float | int | Decimal, context: str = "") -> Decimal:
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
    if not isinstance(execution_time, (int, float, Decimal)):
        raise ValueError(f"Execution time must be numeric {context}")

    # Convert to Decimal for precise calculations
    time_decimal = Decimal(str(execution_time))

    if time_decimal < 0:
        raise ValueError(f"Execution time cannot be negative {context}")

    if time_decimal > Decimal(str(MAX_EXECUTION_TIME_SECONDS)):
        logger.warning(f"Long execution time {context}: {time_decimal}s")

    return time_decimal.quantize(Decimal("0.000001"))  # Microsecond precision


def validate_var(
    var_value: float | int | Decimal, confidence_level: float | Decimal, context: str = ""
) -> Decimal:
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
    if not isinstance(var_value, (int, float, Decimal)):
        raise ValueError(f"VaR value must be numeric {context}")

    # Convert to Decimal for precise calculations
    var_decimal = Decimal(str(var_value))
    confidence_decimal = Decimal(str(confidence_level))

    if not (Decimal("0.5") <= confidence_decimal <= Decimal("0.999")):
        raise ValueError(f"Invalid VaR confidence level {context}: {confidence_decimal}")

    if var_decimal < 0:
        logger.warning(f"Negative VaR {context}: ${var_decimal}")

    if var_decimal > Decimal(str(MAX_VAR_USD)):
        logger.warning(f"Extremely high VaR {context}: ${var_decimal}")

    return var_decimal.quantize(Decimal("0.01"))  # FIAT precision


def validate_drawdown_percent(drawdown_pct: float | int | Decimal, context: str = "") -> Decimal:
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
    if not isinstance(drawdown_pct, (int, float, Decimal)):
        raise ValueError(f"Drawdown must be numeric {context}")

    # Convert to Decimal for precise calculations
    drawdown_decimal = Decimal(str(drawdown_pct))

    if drawdown_decimal < 0:
        raise ValueError(f"Drawdown must be positive (represents loss) {context}")

    if drawdown_decimal > Decimal(str(MAX_DRAWDOWN_PERCENT)):
        raise ValueError(f"Invalid drawdown {context}: {drawdown_decimal}%")

    if drawdown_decimal > 20:  # > 20% drawdown is concerning
        logger.warning(f"High drawdown {context}: {drawdown_decimal}%")

    return drawdown_decimal.quantize(Decimal("0.0001"))  # PERCENTAGE precision


def validate_sharpe_ratio(sharpe_ratio: float | int | Decimal, context: str = "") -> Decimal:
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
    if not isinstance(sharpe_ratio, (int, float, Decimal)):
        raise ValueError(f"Sharpe ratio must be numeric {context}")

    # Convert to Decimal for precise calculations
    sharpe_decimal = Decimal(str(sharpe_ratio))

    if abs(sharpe_decimal) > Decimal(str(MAX_SHARPE_RATIO)):
        raise ValueError(f"Unrealistic Sharpe ratio {context}: {sharpe_decimal}")

    if sharpe_decimal > 2:
        logger.info(f"Excellent Sharpe ratio {context}: {sharpe_decimal}")
    elif sharpe_decimal < -1:
        logger.warning(f"Poor Sharpe ratio {context}: {sharpe_decimal}")

    return sharpe_decimal.quantize(Decimal("0.0001"))  # PERCENTAGE precision


def validate_portfolio_value(value_usd: float | int | Decimal, context: str = "") -> Decimal:
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
    if not isinstance(value_usd, (int, float, Decimal)):
        raise ValueError(f"Portfolio value must be numeric {context}")

    # Convert to Decimal for precise calculations
    value_decimal = Decimal(str(value_usd))

    if value_decimal < 0:
        raise ValueError(f"Portfolio value cannot be negative {context}")

    if value_decimal > Decimal(str(MAX_PORTFOLIO_VALUE_USD)):
        logger.warning(f"Large portfolio value {context}: ${value_decimal}")

    return value_decimal.quantize(Decimal("0.01"))  # FIAT precision


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


def calculate_pnl_percentage(pnl_usd: float | Decimal, portfolio_value: float | Decimal) -> Decimal:
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
    # Convert inputs to Decimal for precise calculations
    pnl_decimal = Decimal(str(pnl_usd))
    portfolio_decimal = Decimal(str(portfolio_value))

    if portfolio_decimal <= 0:
        raise ValueError("Cannot calculate P&L percentage with zero or negative portfolio value")

    # Use Decimal arithmetic for financial precision
    from decimal import localcontext

    with localcontext(_FINANCIAL_DECIMAL_CONTEXT):
        pnl_percentage_decimal = (pnl_decimal / portfolio_decimal) * Decimal("100")

    if abs(pnl_percentage_decimal) > 100:
        logger.warning(f"Extreme P&L percentage: {pnl_percentage_decimal}%")

    return pnl_percentage_decimal.quantize(Decimal("0.0001"))  # PERCENTAGE precision


def validate_position_size_usd(size_usd: float | int | Decimal, context: str = "") -> Decimal:
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
    if not isinstance(size_usd, (int, float, Decimal)):
        raise ValueError(f"Position size must be numeric {context}")

    # Convert to Decimal for precise calculations
    size_decimal = Decimal(str(size_usd))

    if size_decimal < 0:
        raise ValueError(f"Position size cannot be negative {context}")

    if size_decimal > Decimal(str(MAX_TRADE_VALUE_USD)):
        logger.warning(f"Large position size {context}: ${size_decimal}")

    return size_decimal.quantize(Decimal("0.01"))  # FIAT precision
