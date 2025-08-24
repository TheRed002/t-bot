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

from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any

from src.core.exceptions import ValidationError

# Import the actual ValidationFramework from core module
from src.utils.validation.core import ValidationFramework as CoreValidationFramework


def validate_decimal_precision(value: Decimal | float | str, places: int = 8) -> bool:
    """
    Validate decimal precision for financial data.

    Args:
        value: The value to check
        places: Maximum number of decimal places allowed

    Returns:
        bool: True if precision is within limits
    """
    try:
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


def validate_market_data(data: dict[str, Any]) -> bool:
    """
    Comprehensive validation for market data.

    Args:
        data: Market data dictionary to validate

    Returns:
        bool: True if data is valid

    Raises:
        ValidationError: If validation fails
    """
    # Check required fields
    if not data.get("symbol"):
        raise ValidationError("MarketData symbol is required")

    # Validate symbol format
    symbol = data.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValidationError("MarketData symbol must be a non-empty string")

    # Validate price
    if "price" in data and data["price"] is not None:
        try:
            price_val = float(data["price"])
            if price_val < 0:
                raise ValidationError("MarketData price cannot be negative")
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid MarketData price: {e}") from e

    # Validate volume
    if "volume" in data and data["volume"] is not None:
        try:
            volume_val = float(data["volume"])
            if volume_val < 0:
                raise ValidationError("MarketData volume cannot be negative")
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid MarketData volume: {e}") from e

    # Validate bid/ask if present
    if "bid" in data and data["bid"] is not None:
        try:
            bid_val = float(data["bid"])
            if bid_val < 0:
                raise ValidationError("MarketData bid cannot be negative")
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid MarketData bid: {e}") from e

    if "ask" in data and data["ask"] is not None:
        try:
            ask_val = float(data["ask"])
            if ask_val < 0:
                raise ValidationError("MarketData ask cannot be negative")
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid MarketData ask: {e}") from e

    # Validate bid/ask relationship
    if "bid" in data and "ask" in data and data["bid"] is not None and data["ask"] is not None:
        if float(data["bid"]) > float(data["ask"]):
            raise ValidationError("MarketData bid cannot be greater than ask")

    # Validate timestamp if present
    if "timestamp" in data and data["timestamp"] is not None:
        if not isinstance(data["timestamp"], (int, float, str)):
            raise ValidationError("MarketData timestamp must be numeric or string")

    return True


class ValidationFramework:
    """
    Centralized validation framework that provides simplified interfaces
    to the comprehensive validation functions in validation/core.py
    """

    @staticmethod
    def validate_order(order: dict[str, Any]) -> bool:
        """
        Single source of truth for order validation.
        Delegates to the core ValidationFramework.

        Args:
            order: Order dictionary to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        try:
            # Delegate to core ValidationFramework
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
    def validate_price(price: Any, max_price: float = 1_000_000) -> bool:
        """
        Validate and normalize price using core validation.

        Args:
            price: Price to validate
            max_price: Maximum allowed price

        Returns:
            True if valid

        Raises:
            ValidationError: If price is invalid
        """
        try:
            # Delegate to core ValidationFramework
            return CoreValidationFramework.validate_price(price, max_price)

        except ValidationError:
            raise

    @staticmethod
    def validate_quantity(quantity: Any, min_qty: float = 0.00000001) -> bool:
        """
        Validate and normalize quantity using core validation.

        Args:
            quantity: Quantity to validate
            min_qty: Minimum allowed quantity

        Returns:
            True if valid

        Raises:
            ValidationError: If quantity is invalid
        """
        try:
            # Delegate to core ValidationFramework
            return CoreValidationFramework.validate_quantity(quantity, min_qty)

        except ValidationError:
            raise

    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """
        Validate and normalize trading symbol using core validation.

        Args:
            symbol: Trading symbol to validate

        Returns:
            True if valid

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


# Standalone validation functions for backward compatibility
def validate_decimal(value: Any) -> Decimal:
    """
    Validate and convert value to Decimal.

    Args:
        value: Value to validate

    Returns:
        Decimal value

    Raises:
        ValidationError: If value cannot be converted to Decimal
    """

    if isinstance(value, Decimal):
        return value

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as e:
        raise ValidationError(f"Invalid decimal value: {value}") from e


def validate_positive_number(value: Any) -> float:
    """
    Validate that value is a positive number.

    Args:
        value: Value to validate

    Returns:
        Positive float value

    Raises:
        ValidationError: If value is not positive
    """
    try:
        num = float(value)
        if num <= 0:
            raise ValidationError(f"Value must be positive: {value}")
        return num
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Invalid numeric value: {value}") from e


def validate_percentage(value: Any, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """
    Validate that value is a valid percentage.

    Args:
        value: Value to validate
        min_val: Minimum percentage value (default: 0.0)
        max_val: Maximum percentage value (default: 100.0)

    Returns:
        Validated percentage value

    Raises:
        ValidationError: If value is not a valid percentage
    """
    try:
        num = float(value)
        if not min_val <= num <= max_val:
            raise ValidationError(f"Percentage must be between {min_val} and {max_val}: {value}")
        return num
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Invalid percentage value: {value}") from e


def validate_price(value: Any) -> Decimal:
    """
    Validate that value is a valid price.

    Args:
        value: Value to validate

    Returns:
        Validated price as Decimal

    Raises:
        ValidationError: If value is not a valid price
    """
    decimal_value = validate_decimal(value)
    if decimal_value < 0:
        raise ValidationError(f"Price cannot be negative: {value}")
    return decimal_value


def validate_quantity(value: Any) -> Decimal:
    """
    Validate that value is a valid quantity.

    Args:
        value: Value to validate

    Returns:
        Validated quantity as Decimal

    Raises:
        ValidationError: If value is not a valid quantity
    """
    decimal_value = validate_decimal(value)
    if decimal_value <= 0:
        raise ValidationError(f"Quantity must be positive: {value}")
    return decimal_value
