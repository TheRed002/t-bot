"""
Enhanced Validation Framework for the T-Bot Trading System

This module provides a centralized validation framework that leverages the comprehensive
validation functions from the validation/core.py module while providing a simplified
interface for common validation tasks.

The ValidationFramework class serves as a high-level API that delegates to the more
detailed validation functions in validation/core.py, eliminating code duplication
while maintaining ease of use.

Dependencies:
- validation/core.py: Comprehensive validation functions
- P-001: Core types, exceptions, logging
"""

import math
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any

from src.core.exceptions import ValidationError
from src.core.logging import get_logger

# Import the actual ValidationFramework from core module
from src.utils.validation.core import ValidationFramework as CoreValidationFramework

logger = get_logger(__name__)


def validate_decimal_precision(value: Decimal | float | str, places: int = 8) -> bool:
    """
    Validate decimal precision for financial data with enhanced parameter validation.

    Args:
        value: The value to check
        places: Maximum number of decimal places allowed

    Returns:
        bool: True if precision is within limits

    Raises:
        ValidationError: If parameters are invalid
    """
    # Validate places parameter
    if not isinstance(places, int) or places < 0:
        raise ValidationError(f"places must be a non-negative integer, got: {places}")

    if places > 28:  # Maximum practical decimal precision
        raise ValidationError(f"places cannot exceed 28 decimal places, got: {places}")

    try:
        if value is None:
            return False

        decimal_value = Decimal(str(value))
        # Check if it's a special value (NaN, Infinity, etc.)
        if not decimal_value.is_finite():
            return False

        # Get the number of decimal places
        sign, digits, exponent = decimal_value.as_tuple()
        # For finite decimals, exponent is always an int
        assert isinstance(exponent, int), "Exponent should be int for finite decimals"

        if exponent >= 0:
            return True
        decimal_places = -exponent
        return decimal_places <= places
    except (ValueError, InvalidOperation, AssertionError):
        return False


def validate_ttl(ttl: int | float | None, max_ttl: int = 86400) -> int:
    """
    Validate TTL (Time To Live) parameter with comprehensive checks.

    Args:
        ttl: TTL value in seconds
        max_ttl: Maximum allowed TTL (default: 24 hours)

    Returns:
        int: Validated TTL value

    Raises:
        ValidationError: If TTL is invalid
    """
    if ttl is None:
        raise ValidationError("TTL cannot be None")

    try:
        ttl_int = int(ttl)
    except (ValueError, TypeError, OverflowError) as e:
        raise ValidationError(
            f"TTL must be convertible to integer, got: {type(ttl).__name__}"
        ) from e

    if ttl_int < 0:
        raise ValidationError(f"TTL must be non-negative, got: {ttl_int}")

    if ttl_int == 0:
        raise ValidationError("TTL cannot be zero (would expire immediately)")

    if ttl_int > max_ttl:
        raise ValidationError(f"TTL exceeds maximum allowed value of {max_ttl}s, got: {ttl_int}")

    return ttl_int


def validate_precision_range(
    precision: int, min_precision: int = 0, max_precision: int = 28
) -> int:
    """
    Validate precision parameter is within acceptable range.

    Args:
        precision: Precision value to validate
        min_precision: Minimum allowed precision
        max_precision: Maximum allowed precision

    Returns:
        int: Validated precision value

    Raises:
        ValidationError: If precision is out of range
    """
    if not isinstance(precision, int):
        raise ValidationError(f"Precision must be an integer, got: {type(precision).__name__}")

    if precision < min_precision:
        raise ValidationError(f"Precision below minimum: {precision} < {min_precision}")

    if precision > max_precision:
        raise ValidationError(f"Precision above maximum: {precision} > {max_precision}")

    return precision


def validate_financial_range(
    value: Decimal | float,
    min_value: Decimal | float | None = None,
    max_value: Decimal | float | None = None,
    field_name: str = "value",
) -> Decimal:
    """
    Validate financial value is within acceptable range with dynamic bounds.

    Args:
        value: Financial value to validate
        min_value: Minimum allowed value (adaptive)
        max_value: Maximum allowed value (adaptive)
        field_name: Name of field for error messages

    Returns:
        Decimal: Validated value as Decimal

    Raises:
        ValidationError: If value is out of range or invalid
    """
    if value is None:
        raise ValidationError(f"{field_name} cannot be None")

    try:
        if isinstance(value, Decimal):
            decimal_value = value
        else:
            decimal_value = Decimal(str(value))
    except (ValueError, InvalidOperation, OverflowError) as e:
        raise ValidationError(f"Invalid {field_name}: cannot convert to Decimal") from e

    if not decimal_value.is_finite():
        raise ValidationError(f"{field_name} must be finite, got: {decimal_value}")

    # Dynamic range validation
    if min_value is not None:
        min_decimal = Decimal(str(min_value)) if not isinstance(min_value, Decimal) else min_value
        if decimal_value < min_decimal:
            raise ValidationError(f"{field_name} below minimum: {decimal_value} < {min_decimal}")

    if max_value is not None:
        max_decimal = Decimal(str(max_value)) if not isinstance(max_value, Decimal) else max_value
        if decimal_value > max_decimal:
            raise ValidationError(f"{field_name} above maximum: {decimal_value} > {max_decimal}")

    return decimal_value


def validate_null_handling(value: Any, allow_null: bool = False, field_name: str = "value") -> Any:
    """
    Comprehensive null/None value handling with explicit policies.

    Args:
        value: Value to check for null
        allow_null: Whether null values are allowed
        field_name: Name of field for error messages

    Returns:
        Any: The original value if valid

    Raises:
        ValidationError: If null handling policy is violated
    """
    if value is None:
        if allow_null:
            return None
        else:
            raise ValidationError(f"{field_name} cannot be None")

    # Check for other null-like values
    if isinstance(value, str) and value.strip() == "":
        if allow_null:
            return None
        else:
            raise ValidationError(f"{field_name} cannot be empty string")

    # Check for NaN values
    if isinstance(value, float) and math.isnan(value):
        if allow_null:
            return None
        else:
            raise ValidationError(f"{field_name} cannot be NaN")

    return value


def validate_type_conversion(
    value: Any, target_type: type, field_name: str = "value", strict: bool = True
) -> Any:
    """
    Validate type conversion with comprehensive error handling.

    Args:
        value: Value to convert
        target_type: Target type for conversion
        field_name: Name of field for error messages
        strict: Whether to enforce strict type conversion

    Returns:
        Any: Converted value

    Raises:
        ValidationError: If conversion fails
    """
    if value is None:
        raise ValidationError(f"Cannot convert None {field_name} to {target_type.__name__}")

    try:
        if target_type == Decimal:
            if isinstance(value, Decimal):
                return value
            elif isinstance(value, int | float):
                if isinstance(value, float):
                    if not math.isfinite(value):
                        raise ValidationError(
                            f"Cannot convert non-finite float {field_name} to Decimal"
                        )
                return Decimal(str(value))
            elif isinstance(value, str):
                return Decimal(value.strip())
            else:
                raise ValidationError(f"Cannot convert {type(value).__name__} to Decimal")

        elif target_type is float:
            # FINANCIAL PRECISION WARNING: Float conversion for financial data
            # This should be avoided for price, quantity, amount fields
            logger.warning(
                f"Converting {field_name} to float may cause precision loss for financial data"
            )

            if isinstance(value, float):
                if not math.isfinite(value):
                    raise ValidationError(f"Invalid float {field_name}: {value}")
                return value
            elif isinstance(value, int | Decimal):
                # Use safe conversion with precision warnings for financial data
                if field_name.lower() in [
                    "price",
                    "quantity",
                    "amount",
                    "value",
                    "balance",
                    "fee",
                    "cost",
                ]:
                    logger.error(
                        f"CRITICAL: Converting financial field '{field_name}' from Decimal to float - use Decimal throughout"
                    )
                try:
                    from src.monitoring.financial_precision import safe_decimal_to_float

                    result = safe_decimal_to_float(
                        value, f"validation_{field_name}", warn_on_loss=True
                    )
                    if not math.isfinite(result):
                        raise ValidationError(
                            f"Conversion of {field_name} to float resulted in non-finite value"
                        )
                    return result
                except ImportError:
                    # Fallback if monitoring module not available
                    result = float(value)
                    if not math.isfinite(result):
                        raise ValidationError(
                            f"Conversion of {field_name} to float resulted in non-finite value"
                        )
                    return result
            else:
                # Use safe conversion for unknown types
                try:
                    from src.monitoring.financial_precision import safe_decimal_to_float

                    return safe_decimal_to_float(
                        value, f"validation_{field_name}", warn_on_loss=True
                    )
                except ImportError:
                    # Fallback conversion
                    return float(value)

        elif target_type is int:
            if isinstance(value, int):
                return value
            elif isinstance(value, float | Decimal):
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValidationError(f"Cannot convert non-finite float {field_name} to int")
                result = int(value)
                if not strict:
                    return result
                # Strict mode: check for precision loss using Decimal comparison
                if Decimal(str(result)) != Decimal(str(value)):
                    raise ValidationError(
                        f"Precision loss converting {field_name} to int: {value} -> {result}"
                    )
                return result
            else:
                return int(value)  # Let Python handle the conversion

        else:
            return target_type(value)

    except (ValueError, TypeError, OverflowError, InvalidOperation) as e:
        raise ValidationError(
            f"Failed to convert {field_name} to {target_type.__name__}: {e}"
        ) from e


def validate_market_conditions(
    price: Decimal | float,
    volume: Decimal | float | None = None,
    spread: Decimal | float | None = None,
    symbol: str = "unknown",
) -> dict[str, Decimal]:
    """
    Dynamic range validation based on market conditions.

    NOTE: This function is deprecated. Use ValidationService.validate_market_data() instead.
    """
    # Simplified fallback - delegate to service layer or use basic validation
    fallback_result: dict[str, Decimal] = {}
    fallback_result["price"] = validate_financial_range(
        price,
        min_value=Decimal("0.00000001"),
        max_value=Decimal("10000000"),
        field_name=f"price for {symbol}",
    )
    if volume is not None:
        fallback_result["volume"] = validate_financial_range(
            volume,
            min_value=Decimal("0"),
            max_value=Decimal("1000000000"),
            field_name=f"volume for {symbol}",
        )
    if spread is not None:
        fallback_result["spread"] = validate_financial_range(
            spread,
            min_value=Decimal("0"),
            max_value=fallback_result["price"] * Decimal("0.1"),
            field_name=f"spread for {symbol}",
        )
    return fallback_result


def _validate_symbol(data: dict[str, Any]) -> None:
    """Validate symbol field in market data."""
    if not data.get("symbol"):
        raise ValidationError("MarketData symbol is required")

    symbol = data.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValidationError("MarketData symbol must be a non-empty string")


def _validate_numeric_field(data: dict[str, Any], field: str) -> None:
    """Validate a numeric field in market data."""
    if field in data and data[field] is not None:
        try:
            field_val = float(data[field])
            if field_val < 0:
                raise ValidationError(f"MarketData {field} cannot be negative")
        except (ValueError, TypeError) as e:
            # Use core exception types for consistent error propagation
            raise ValidationError(
                f"Invalid MarketData {field}: {e}",
                error_code="DATA_001",
                details={"field": field, "value": data.get(field), "original_error": str(e)},
            ) from e


def _validate_bid_ask_relationship(data: dict[str, Any]) -> None:
    """Validate bid/ask price relationship."""
    if "bid" in data and "ask" in data and data["bid"] is not None and data["ask"] is not None:
        # Use Decimal for precise comparison and consistent error types
        from decimal import Decimal

        try:
            bid_decimal = Decimal(str(data["bid"]))
            ask_decimal = Decimal(str(data["ask"]))
            if bid_decimal > ask_decimal:
                raise ValidationError(
                    "MarketData bid cannot be greater than ask",
                    error_code="DATA_001",
                    details={"bid": str(bid_decimal), "ask": str(ask_decimal)},
                )
        except (ValueError, TypeError) as e:
            raise ValidationError(
                "Invalid bid/ask price format",
                error_code="DATA_001",
                details={"bid": data["bid"], "ask": data["ask"], "original_error": str(e)},
            ) from e


def _validate_timestamp(data: dict[str, Any]) -> None:
    """Validate timestamp field."""
    if "timestamp" in data and data["timestamp"] is not None:
        if not isinstance(data["timestamp"], int | float | str):
            raise ValidationError("MarketData timestamp must be numeric or string")


def validate_market_data(data: dict[str, Any]) -> bool:
    """
    Comprehensive validation for market data.

    NOTE: This function is deprecated. Use ValidationService.validate_market_data() instead.
    """
    # Simplified fallback validation
    _validate_symbol(data)
    numeric_fields = ["price", "volume", "bid", "ask"]
    for field in numeric_fields:
        _validate_numeric_field(data, field)
    _validate_bid_ask_relationship(data)
    _validate_timestamp(data)
    return True


class ValidationFramework:
    """
    Centralized validation framework that provides simplified interfaces
    to the comprehensive validation functions in validation/core.py

    NOTE: This class is deprecated. Use ValidationService instead.
    This class now delegates to the service layer to avoid business logic duplication.
    """

    @staticmethod
    def validate_order(order: dict[str, Any]) -> bool:
        """
        Single source of truth for order validation.

        NOTE: This method is deprecated. Use ValidationService.validate_order() instead.
        This method now delegates to the service layer.

        Args:
            order: Order dictionary to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        try:
            # Fallback to core ValidationFramework for backward compatibility
            return CoreValidationFramework.validate_order(order)

        except ValidationError:
            # Re-raise ValidationError as is
            raise

    @staticmethod
    def validate_strategy_params(params: dict[str, Any]) -> bool:
        """
        Single source for strategy parameter validation.
        Delegates to the core ValidationFramework.

        Args:
            params: Strategy parameters to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        try:
            # Delegate to core ValidationFramework
            return CoreValidationFramework.validate_strategy_params(params)

        except ValidationError:
            # Re-raise ValidationError as is
            raise

    @staticmethod
    def validate_price(price: Any, max_price: Decimal = Decimal("1000000")) -> Decimal:
        """
        Validate and normalize price using core validation with Decimal precision.

        Args:
            price: Price to validate
            max_price: Maximum allowed price as Decimal

        Returns:
            Normalized price value as Decimal (8 decimal precision)

        Raises:
            ValidationError: If price is invalid
        """
        try:
            # Delegate to core ValidationFramework
            return CoreValidationFramework.validate_price(price, max_price)

        except ValidationError:
            raise

    @staticmethod
    def validate_quantity(quantity: Any, min_qty: Decimal = Decimal("0.00000001")) -> Decimal:
        """
        Validate and normalize quantity using core validation with Decimal precision.

        Args:
            quantity: Quantity to validate
            min_qty: Minimum allowed quantity as Decimal

        Returns:
            Normalized quantity value as Decimal (8 decimal precision)

        Raises:
            ValidationError: If quantity is invalid
        """
        try:
            # Delegate to core ValidationFramework
            return CoreValidationFramework.validate_quantity(quantity, min_qty)

        except ValidationError:
            raise

    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """
        Validate and normalize trading symbol using core validation.

        Args:
            symbol: Trading symbol to validate

        Returns:
            Normalized symbol string

        Raises:
            ValidationError: If symbol is invalid
        """
        try:
            # Delegate to core ValidationFramework
            return CoreValidationFramework.validate_symbol(symbol)

        except ValidationError:
            raise

    @staticmethod
    def validate_exchange_credentials(credentials: dict[str, Any]) -> bool:
        """
        Validate exchange API credentials.

        Args:
            credentials: Credentials dictionary

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        try:
            # Delegate to core ValidationFramework
            return CoreValidationFramework.validate_exchange_credentials(credentials)
        except ValidationError:
            raise

    @staticmethod
    def validate_risk_params(params: dict[str, Any]) -> bool:
        """
        Validate risk management parameters using core validation.

        Args:
            params: Risk parameters to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        try:
            # Delegate to core ValidationFramework
            return CoreValidationFramework.validate_risk_params(params)

        except ValidationError:
            raise

    @staticmethod
    def validate_risk_parameters(params: dict[str, Any]) -> bool:
        """
        Validate risk management parameters (alias for validate_risk_params).

        Args:
            params: Risk parameters to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        return ValidationFramework.validate_risk_params(params)

    @staticmethod
    def validate_timeframe(timeframe: str) -> str:
        """
        Validate and normalize timeframe.

        Args:
            timeframe: Timeframe string

        Returns:
            Normalized timeframe

        Raises:
            ValidationError: If timeframe is invalid
        """
        try:
            # Delegate to core ValidationFramework
            return CoreValidationFramework.validate_timeframe(timeframe)
        except ValidationError:
            raise

    @staticmethod
    def validate_batch(validations: list[tuple[str, Callable[[Any], Any], Any]]) -> dict[str, Any]:
        """
        Run multiple validations and collect results.

        Args:
            validations: List of (name, validator_func, data) tuples

        Returns:
            Dictionary with validation results
        """
        try:
            # Delegate to core ValidationFramework
            return CoreValidationFramework.validate_batch(validations)
        except Exception as e:
            # If the batch validation fails entirely, return error for all
            return {name: {"status": "error", "error": str(e)} for name, _, _ in validations}


# Re-export validation framework class for backward compatibility
ValidationUtilities = ValidationFramework


# Validation functions are available through ValidationService for improved type safety and consistency.
# Use ValidationService or ValidationFramework for all validation operations.
