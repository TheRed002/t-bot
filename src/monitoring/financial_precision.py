"""
Financial Precision Utilities for Monitoring.

This module provides utilities to safely convert Decimal financial values
to float for Prometheus metrics while preserving maximum precision and
logging any potential precision loss.
"""

import warnings
from decimal import ROUND_HALF_UP, Context, Decimal, localcontext

from src.core import get_logger
from src.core.exceptions import ValidationError

logger = get_logger(__name__)

# Financial precision context for critical calculations
_FINANCIAL_DECIMAL_CONTEXT = Context(
    prec=28,  # 28 significant digits (matches most financial systems)
    rounding=ROUND_HALF_UP,
    Emin=-999999,
    Emax=999999,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[],
)

# Configuration dictionary expected by tests
FINANCIAL_CONTEXT = {
    "max_precision": 12,
    "default_precision": 8,
    "warn_on_precision_loss": True,
    "strict_mode": False,
}


class FinancialPrecisionWarning(UserWarning):
    """Warning raised when precision loss is detected in financial calculations."""

    pass


def safe_decimal_to_float(
    value: Decimal | float | int,
    metric_name: str,
    precision_digits: int = 8,
    warn_on_loss: bool = True,
) -> float:
    """
    Safely convert financial Decimal values to float for metrics.

    Args:
        value: The financial value to convert
        metric_name: Name of the metric (for logging)
        precision_digits: Number of decimal places to preserve
        warn_on_loss: Whether to warn on precision loss

    Returns:
        Float value suitable for Prometheus metrics

    Raises:
        ValueError: If value is invalid or None
    """
    if value is None:
        raise ValueError(f"Cannot convert None to float for metric {metric_name}")

    # If already float or int, just validate and return
    if isinstance(value, (float, int)):
        if isinstance(value, float) and not float("-inf") < value < float("inf"):
            raise ValueError(f"Invalid float value for metric {metric_name}: {value}")
        return round(float(value), precision_digits)

    # Handle Decimal conversion
    if not isinstance(value, Decimal):
        raise TypeError(
            f"Expected Decimal, float, or int for metric {metric_name}, got {type(value)}"
        )

    # Check for special values
    if not value.is_finite():
        raise ValueError(f"Non-finite Decimal value for metric {metric_name}: {value}")

    # Perform conversion with precision tracking
    original_str = str(value)
    float_value = float(value)

    # Check for precision loss
    if warn_on_loss:
        # Convert back to Decimal to check precision loss
        back_to_decimal = Decimal(str(float_value))

        # Use financial context for comparison
        with localcontext(_FINANCIAL_DECIMAL_CONTEXT):
            difference = abs(value - back_to_decimal)
            relative_error = difference / abs(value) if value != 0 else Decimal(0)

            # Warn if relative error > 0.00001% (0.1 basis point of a basis point)
            if relative_error > Decimal("0.0000001"):
                warning_msg = (
                    f"Precision loss detected for metric {metric_name}: "
                    f"Original={original_str}, Float={float_value}, "
                    f"RelativeError={float(relative_error * 100):.8f}%"
                )
                warnings.warn(warning_msg, FinancialPrecisionWarning, stacklevel=2)
                logger.warning(warning_msg)

    return round(float_value, precision_digits)


def convert_financial_batch(
    values: dict[str, Decimal | float | int],
    metric_prefix: str,
    precision_map: dict[str, int] | None = None,
) -> dict[str, float]:
    """
    Convert a batch of financial values to floats with precision tracking.

    Args:
        values: Dictionary of metric_name -> value
        metric_prefix: Prefix for metric names in logging
        precision_map: Optional map of metric_name -> precision_digits

    Returns:
        Dictionary of metric_name -> float value
    """
    precision_map = precision_map or {}
    results = {}

    for name, value in values.items():
        precision = precision_map.get(name, 8)  # Default to 8 decimal places
        metric_full_name = f"{metric_prefix}.{name}" if metric_prefix else name

        try:
            results[name] = safe_decimal_to_float(
                value, metric_full_name, precision_digits=precision
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert {metric_full_name}: {e}")
            raise

    return results


def validate_financial_range(
    value: Decimal | float,
    min_value: Decimal | float | None = None,
    max_value: Decimal | float | None = None,
    metric_name: str | None = None,
) -> None:
    """
    Validate that a financial value is within expected bounds.

    Args:
        value: The value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        metric_name: Name of the metric (for error messages)

    Raises:
        ValidationError: If value is outside allowed range
    """
    if min_value is not None and value < min_value:
        metric_info = f" for {metric_name}" if metric_name else ""
        raise ValidationError(f"Financial value{metric_info} below minimum: {value} < {min_value}")

    if max_value is not None and value > max_value:
        metric_info = f" for {metric_name}" if metric_name else ""
        raise ValidationError(f"Financial value{metric_info} above maximum: {value} > {max_value}")


def detect_precision_requirements(value: Decimal, metric_name: str) -> tuple[int, bool]:
    """
    Detect the precision requirements for a Decimal value.

    Args:
        value: The Decimal value to analyze
        metric_name: Name of the metric (for logging)

    Returns:
        Tuple of (required_decimal_places, is_high_precision)
    """
    # Convert to string and analyze decimal places
    value_str = str(value.normalize())

    # Handle scientific notation
    if "E" in value_str or "e" in value_str:
        # High precision required for very small/large numbers
        return (12, True)

    # Count decimal places
    if "." in value_str:
        decimal_part = value_str.split(".")[-1]
        decimal_places = len(decimal_part)

        # Cryptocurrency typically needs 8 decimal places
        # Traditional finance typically needs 2-4
        is_high_precision = decimal_places > 4

        return (min(decimal_places, 12), is_high_precision)

    return (0, False)


# Precision recommendations for different financial metrics
METRIC_PRECISION_MAP = {
    # Prices and values - 8 decimals for crypto compatibility
    "price": 8,
    "value_usd": 8,
    "pnl_usd": 8,
    "volume_usd": 8,
    # Percentages and ratios - 4 decimals (0.01% precision)
    "percent": 4,
    "ratio": 4,
    "rate": 4,
    "apy": 4,
    "sharpe_ratio": 4,
    # Basis points - 2 decimals
    "bps": 2,
    "slippage_bps": 2,
    # Counts and integers - 0 decimals
    "count": 0,
    "total": 0,
    # Time measurements - 6 decimals (microseconds)
    "duration_seconds": 6,
    "latency_seconds": 6,
}


def get_recommended_precision(metric_name: str) -> int:
    """
    Get recommended decimal precision for a metric based on its name.

    Args:
        metric_name: Name of the metric

    Returns:
        Recommended number of decimal places
    """
    metric_lower = metric_name.lower()

    # Check exact matches first
    if metric_lower in METRIC_PRECISION_MAP:
        return METRIC_PRECISION_MAP[metric_lower]

    # Check suffixes
    for suffix, precision in METRIC_PRECISION_MAP.items():
        if metric_lower.endswith(suffix):
            return precision

    # Default to 8 decimals for financial metrics
    return 8
