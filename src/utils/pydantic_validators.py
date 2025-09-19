"""
Pydantic Validator Utilities for T-Bot Trading System.

This module provides reusable Pydantic validators to eliminate duplication
across API models in the web_interface module.
"""

from decimal import Decimal
from typing import Any, Callable

from pydantic import validator

from src.utils.validation.core import ValidationFramework


class PydanticValidators:
    """Reusable Pydantic validators to eliminate duplication."""

    @staticmethod
    def amount_validator(field_name: str = "amount") -> Callable[[Any, Any], str]:
        """
        Create a Pydantic validator for positive amount fields.

        Args:
            field_name: Name of the field for error messages

        Returns:
            Pydantic validator function
        """
        def validate_amount(cls, v):
            """Validate amount is a valid positive Decimal."""
            from src.utils.validation.core import ValidationFramework
            try:
                return ValidationFramework.validate_positive_amount(v)
            except Exception as e:
                raise ValueError(str(e)) from e

        validate_amount.__name__ = f"validate_{field_name}"
        return validator(field_name)(validate_amount)

    @staticmethod
    def non_negative_amount_validator(field_name: str = "amount") -> Callable[[Any, Any], str]:
        """
        Create a Pydantic validator for non-negative amount fields.

        Args:
            field_name: Name of the field for error messages

        Returns:
            Pydantic validator function
        """
        def validate_non_negative_amount(cls, v):
            """Validate amount is a valid non-negative Decimal."""
            from src.utils.validation.core import ValidationFramework
            try:
                return ValidationFramework.validate_non_negative_amount(v)
            except Exception as e:
                raise ValueError(str(e)) from e

        validate_non_negative_amount.__name__ = f"validate_{field_name}"
        return validator(field_name)(validate_non_negative_amount)

    @staticmethod
    def price_validator(field_name: str = "price") -> Callable[[Any, Any], str]:
        """
        Create a Pydantic validator for price fields.

        Args:
            field_name: Name of the field for error messages

        Returns:
            Pydantic validator function
        """
        def validate_price(cls, v):
            """Validate price is a valid positive Decimal."""
            if v is None:
                return v
            from src.utils.validation.core import ValidationFramework
            try:
                validated_price = ValidationFramework.validate_price(v)
                return str(validated_price)
            except Exception as e:
                raise ValueError(str(e)) from e

        validate_price.__name__ = f"validate_{field_name}"
        return validator(field_name, allow_reuse=True)(validate_price)

    @staticmethod
    def quantity_validator(field_name: str = "quantity") -> Callable[[Any, Any], str]:
        """
        Create a Pydantic validator for quantity fields.

        Args:
            field_name: Name of the field for error messages

        Returns:
            Pydantic validator function
        """
        def validate_quantity(cls, v):
            """Validate quantity is a valid positive Decimal."""
            if v is None:
                return v
            from src.utils.validation.core import ValidationFramework
            try:
                validated_quantity = ValidationFramework.validate_quantity(v)
                return str(validated_quantity)
            except Exception as e:
                raise ValueError(str(e)) from e

        validate_quantity.__name__ = f"validate_{field_name}"
        return validator(field_name, allow_reuse=True)(validate_quantity)

    @staticmethod
    def symbol_validator(field_name: str = "symbol") -> Callable[[Any, Any], str]:
        """
        Create a Pydantic validator for trading symbol fields.

        Args:
            field_name: Name of the field for error messages

        Returns:
            Pydantic validator function
        """
        def validate_symbol(cls, v):
            """Validate symbol format."""
            from src.utils.validation.core import ValidationFramework
            try:
                return ValidationFramework.validate_symbol(v)
            except Exception as e:
                raise ValueError(str(e)) from e

        validate_symbol.__name__ = f"validate_{field_name}"
        return validator(field_name, allow_reuse=True)(validate_symbol)

    @staticmethod
    def percentage_validator(field_name: str = "percentage") -> Callable[[Any, Any], str]:
        """
        Create a Pydantic validator for percentage fields (0-1 range).

        Args:
            field_name: Name of the field for error messages

        Returns:
            Pydantic validator function
        """
        def validate_percentage(cls, v):
            """Validate percentage is between 0 and 1."""
            try:
                value = Decimal(str(v))
                if value < 0 or value > 1:
                    raise ValueError(f"{field_name} must be between 0 and 1")
                return str(value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid {field_name}: {e}") from e

        validate_percentage.__name__ = f"validate_{field_name}"
        return validator(field_name, allow_reuse=True)(validate_percentage)

    @staticmethod
    def exchange_validator(field_name: str = "exchange") -> Callable[[Any, Any], str]:
        """
        Create a Pydantic validator for exchange name fields.

        Args:
            field_name: Name of the field for error messages

        Returns:
            Pydantic validator function
        """
        def validate_exchange(cls, v):
            """Validate exchange name format."""
            if not v or not isinstance(v, str):
                raise ValueError(f"{field_name} must be a non-empty string")
            # Convert to lowercase for consistency
            return v.lower().strip()

        validate_exchange.__name__ = f"validate_{field_name}"
        return validator(field_name, allow_reuse=True)(validate_exchange)


# Pre-configured common validators for direct use
validate_amount = PydanticValidators.amount_validator("amount")
validate_utilized_amount = PydanticValidators.non_negative_amount_validator("utilized_amount")
validate_allocated_capital = PydanticValidators.amount_validator("allocated_capital")
validate_price = PydanticValidators.price_validator("price")
validate_quantity = PydanticValidators.quantity_validator("quantity")
validate_symbol = PydanticValidators.symbol_validator("symbol")
validate_risk_percentage = PydanticValidators.percentage_validator("risk_percentage")
validate_exchange = PydanticValidators.exchange_validator("exchange")