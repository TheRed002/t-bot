"""
Decimal Precision Utilities for Financial Calculations.

This module provides centralized Decimal handling to ensure precision
in all financial calculations throughout the trading system.

CRITICAL: All monetary values, prices, quantities, and percentages
must use Decimal to prevent floating-point errors that could lead
to financial losses.
"""

import warnings
from decimal import ROUND_DOWN, ROUND_HALF_UP, Context, Decimal, setcontext
from typing import Any

from src.core.exceptions import ValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)

# Set global decimal context for financial precision
# 28 significant digits should be sufficient for all cryptocurrency calculations
FINANCIAL_CONTEXT = Context(
    prec=28,  # Precision for significant digits
    rounding=ROUND_HALF_UP,  # Standard financial rounding
    Emin=-999999,  # Minimum exponent
    Emax=999999,  # Maximum exponent
    capitals=1,  # Use E notation
    clamp=0,  # No clamping
    flags=[],  # Clear all flags
    traps=[],  # Don't trap any conditions
)

# Set as default context
setcontext(FINANCIAL_CONTEXT)

# Common decimal constants
ZERO = Decimal("0")
ONE = Decimal("1")
TWO = Decimal("2")
TEN = Decimal("10")
HUNDRED = Decimal("100")
THOUSAND = Decimal("1000")

# Percentage constants
ONE_PERCENT = Decimal("0.01")
FIVE_PERCENT = Decimal("0.05")
TEN_PERCENT = Decimal("0.10")
TWENTY_FIVE_PERCENT = Decimal("0.25")
FIFTY_PERCENT = Decimal("0.50")
SEVENTY_FIVE_PERCENT = Decimal("0.75")
ONE_HUNDRED_PERCENT = Decimal("1.00")

# Basis point constants (1 bp = 0.01% = 0.0001)
ONE_BP = Decimal("0.0001")
TEN_BPS = Decimal("0.001")
HUNDRED_BPS = Decimal("0.01")

# Common crypto precision levels
SATOSHI = Decimal("0.00000001")  # Bitcoin precision
WEI = Decimal("0.000000000000000001")  # Ethereum precision
USDT_PRECISION = Decimal("0.000001")  # USDT typical precision


def to_decimal(value: str | int | float | Decimal, context: Context | None = None) -> Decimal:
    """
    Safely convert a value to Decimal with proper precision.

    CRITICAL: This function should be used for ALL numeric conversions
    in financial calculations to maintain precision.

    Args:
        value: Value to convert to Decimal
        context: Optional decimal context (defaults to FINANCIAL_CONTEXT)

    Returns:
        Decimal representation of the value

    Raises:
        ValidationError: If value cannot be converted to Decimal
    """
    if value is None:
        raise ValidationError("Cannot convert None to Decimal")

    # Already a Decimal
    if isinstance(value, Decimal):
        return value

    # Use specified context or default
    ctx = context or FINANCIAL_CONTEXT

    try:
        # For floats, convert to string first to avoid precision loss
        if isinstance(value, float):
            # Check for special values
            if value != value:  # NaN check
                raise ValidationError("Cannot convert NaN to Decimal")
            if value == float("inf") or value == float("-inf"):
                raise ValidationError("Cannot convert infinity to Decimal")

            # Convert via string to preserve precision
            # Use repr() for exact float representation
            result = ctx.create_decimal(repr(value))
        else:
            # Direct conversion for strings and integers
            result = ctx.create_decimal(str(value))

        return result

    except Exception as e:
        logger.error(f"Failed to convert {type(value).__name__} to Decimal: {value}")
        raise ValidationError(f"Invalid Decimal conversion: {e}")


def decimal_to_str(value: Decimal, precision: int | None = None) -> str:
    """
    Convert Decimal to string with optional precision.

    Args:
        value: Decimal value to convert
        precision: Number of decimal places (None for full precision)

    Returns:
        String representation of the Decimal
    """
    if precision is not None:
        # Quantize to specified precision
        quantizer = Decimal(10) ** -precision
        value = value.quantize(quantizer, rounding=ROUND_HALF_UP)

    # Remove trailing zeros and decimal point if not needed
    return str(value).rstrip("0").rstrip(".")


def round_price(price: Decimal, tick_size: Decimal) -> Decimal:
    """
    Round price to the nearest tick size (exchange price precision).

    Args:
        price: Price to round
        tick_size: Minimum price movement (e.g., 0.01 for 2 decimals)

    Returns:
        Rounded price
    """
    if tick_size <= ZERO:
        raise ValidationError("Tick size must be positive")

    return (price / tick_size).quantize(ONE, rounding=ROUND_HALF_UP) * tick_size


def round_quantity(quantity: Decimal, lot_size: Decimal) -> Decimal:
    """
    Round quantity to the nearest lot size (exchange quantity precision).

    Args:
        quantity: Quantity to round
        lot_size: Minimum quantity increment

    Returns:
        Rounded quantity
    """
    if lot_size <= ZERO:
        raise ValidationError("Lot size must be positive")

    # Round down for quantities to avoid exceeding available balance
    return (quantity / lot_size).quantize(ONE, rounding=ROUND_DOWN) * lot_size


def calculate_percentage(value: Decimal, percentage: Decimal) -> Decimal:
    """
    Calculate a percentage of a value with full precision.

    Args:
        value: Base value
        percentage: Percentage as decimal (e.g., 0.05 for 5%)

    Returns:
        Calculated percentage value
    """
    return value * percentage


def calculate_basis_points(value: Decimal, bps: Decimal) -> Decimal:
    """
    Calculate basis points of a value.

    Args:
        value: Base value
        bps: Basis points (1 bp = 0.01%)

    Returns:
        Calculated basis point value
    """
    return value * (bps * ONE_BP)


def safe_divide(numerator: Decimal, denominator: Decimal, default: Decimal = ZERO) -> Decimal:
    """
    Safely divide two Decimals, returning default on division by zero.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if denominator is zero

    Returns:
        Result of division or default value
    """
    if denominator == ZERO:
        logger.warning(f"Division by zero avoided: {numerator} / 0")
        return default

    return numerator / denominator


def validate_positive(value: Decimal, name: str = "value") -> None:
    """
    Validate that a Decimal value is positive.

    Args:
        value: Value to validate
        name: Name of the value for error messages

    Raises:
        ValidationError: If value is not positive
    """
    if value <= ZERO:
        raise ValidationError(f"{name} must be positive, got {value}")


def validate_non_negative(value: Decimal, name: str = "value") -> None:
    """
    Validate that a Decimal value is non-negative.

    Args:
        value: Value to validate
        name: Name of the value for error messages

    Raises:
        ValidationError: If value is negative
    """
    if value < ZERO:
        raise ValidationError(f"{name} must be non-negative, got {value}")


def validate_percentage(value: Decimal, name: str = "percentage") -> None:
    """
    Validate that a Decimal value is a valid percentage (0-1).

    Args:
        value: Value to validate
        name: Name of the value for error messages

    Raises:
        ValidationError: If value is not between 0 and 1
    """
    if value < ZERO or value > ONE:
        raise ValidationError(f"{name} must be between 0 and 1, got {value}")


def compare_decimals(a: Decimal, b: Decimal, tolerance: Decimal = SATOSHI) -> int:
    """
    Compare two Decimals with tolerance for rounding differences.

    Args:
        a: First value
        b: Second value
        tolerance: Maximum difference to consider equal

    Returns:
        -1 if a < b, 0 if equal (within tolerance), 1 if a > b
    """
    diff = abs(a - b)
    if diff <= tolerance:
        return 0
    return -1 if a < b else 1


def format_decimal(value: Decimal, decimals: int = 8, thousands_sep: bool = True) -> str:
    """
    Format a Decimal for display with proper precision and separators.

    Args:
        value: Decimal value to format
        decimals: Number of decimal places
        thousands_sep: Whether to include thousands separators

    Returns:
        Formatted string representation
    """
    # Quantize to specified decimals
    quantizer = Decimal(10) ** -decimals
    value = value.quantize(quantizer, rounding=ROUND_HALF_UP)

    # Convert to string
    result = str(value)

    if thousands_sep and "." in result:
        # Add thousands separators
        integer_part, decimal_part = result.split(".")
        integer_part = f"{int(integer_part):,}"
        result = f"{integer_part}.{decimal_part}"
    elif thousands_sep:
        result = f"{int(value):,}"

    return result


def sum_decimals(values: list[Decimal]) -> Decimal:
    """
    Sum a list of Decimal values safely.

    Args:
        values: List of Decimal values

    Returns:
        Sum of all values
    """
    return sum(values, ZERO)


def avg_decimals(values: list[Decimal]) -> Decimal:
    """
    Calculate average of Decimal values safely.

    Args:
        values: List of Decimal values

    Returns:
        Average of values or ZERO if empty
    """
    if not values:
        return ZERO

    return sum_decimals(values) / Decimal(len(values))


def min_decimal(*values: Decimal) -> Decimal:
    """
    Return minimum of Decimal values.

    Args:
        values: Variable number of Decimal values

    Returns:
        Minimum value
    """
    return min(values)


def max_decimal(*values: Decimal) -> Decimal:
    """
    Return maximum of Decimal values.

    Args:
        values: Variable number of Decimal values

    Returns:
        Maximum value
    """
    return max(values)


def clamp_decimal(value: Decimal, min_val: Decimal, max_val: Decimal) -> Decimal:
    """
    Clamp a Decimal value between min and max.

    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


def decimal_to_float(value: Decimal) -> float:
    """
    Convert Decimal to float for compatibility with libraries that require float.

    WARNING: This conversion may result in precision loss and should only be used
    when interfacing with libraries that don't support Decimal.

    Args:
        value: Decimal value to convert

    Returns:
        Float representation of the Decimal
    """
    FloatDeprecationWarning.warn_float_usage("decimal_to_float conversion")
    return float(value)


def float_to_decimal(value: float) -> Decimal:
    """
    Convert float to Decimal safely.

    Args:
        value: Float value to convert

    Returns:
        Decimal representation of the float
    """
    return to_decimal(value)


class DecimalEncoder:
    """JSON encoder that handles Decimal values."""

    @staticmethod
    def encode(obj: Any) -> Any:
        """
        Encode Decimal values for JSON serialization.

        Args:
            obj: Object to encode

        Returns:
            JSON-serializable representation
        """
        if isinstance(obj, Decimal):
            # Convert to string to preserve precision
            return str(obj)
        elif isinstance(obj, dict):
            return {k: DecimalEncoder.encode(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DecimalEncoder.encode(item) for item in obj]
        else:
            return obj


class FloatDeprecationWarning:
    """
    Context manager to detect and warn about float usage in financial code.
    """

    def __enter__(self):
        """Enable float deprecation warnings."""
        warnings.filterwarnings("always", category=DeprecationWarning)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Disable float deprecation warnings."""
        warnings.filterwarnings("default", category=DeprecationWarning)

    @staticmethod
    def warn_float_usage(context: str):
        """
        Issue a deprecation warning for float usage.

        Args:
            context: Context where float was used
        """
        warnings.warn(
            f"Float used in financial calculation: {context}. "
            "Use Decimal for all financial calculations to prevent precision loss.",
            DeprecationWarning,
            stacklevel=2,
        )


# Module initialization - Only log in debug mode to avoid spam
logger.debug(
    "Decimal utilities initialized",
    precision=FINANCIAL_CONTEXT.prec,
    rounding=FINANCIAL_CONTEXT.rounding,
)
