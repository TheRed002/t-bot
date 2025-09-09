"""
Capital Management Validation Utilities

This module provides shared validation functions for capital management operations
to eliminate code duplication across capital management services.
"""

from decimal import Decimal
from typing import Any

from src.core.exceptions import ValidationError

# Import formatters when needed dynamically
from src.utils.decimal_utils import safe_decimal_conversion as centralized_safe_decimal_conversion


def validate_capital_amount(
    amount: Decimal,
    amount_name: str = "amount",
    min_value: Decimal = Decimal("0"),
    component: str = "CapitalManagement",
) -> None:
    """
    Validate capital amount with consistent error handling.

    Args:
        amount: Amount to validate
        amount_name: Name of the amount field for error messages
        min_value: Minimum allowed value
        component: Component name for error context

    Raises:
        ValidationError: If amount is invalid
    """
    if not isinstance(amount, Decimal):
        raise ValidationError(
            f"Invalid {amount_name} type: expected Decimal, got {type(amount).__name__}",
            error_code="VAL_001",
            details={"component": component, "field": amount_name},
        )

    if amount <= min_value:
        # Lazy import to avoid circular dependency
        from src.utils.formatters import format_currency

        raise ValidationError(
            f"Invalid {amount_name}: {format_currency(amount)} must be greater than {format_currency(min_value)}",
            error_code="VAL_002",
            details={"amount": str(amount), "min_value": str(min_value), "component": component},
        )


def validate_strategy_id(strategy_id: str, component: str = "CapitalManagement") -> None:
    """
    Validate strategy ID with consistent error handling.

    Args:
        strategy_id: Strategy ID to validate
        component: Component name for error context

    Raises:
        ValidationError: If strategy ID is invalid
    """
    if not strategy_id or not strategy_id.strip():
        raise ValidationError(
            "Strategy ID cannot be empty",
            error_code="VAL_003",
            details={"component": component, "field": "strategy_id"},
        )


def validate_exchange_name(exchange: str, component: str = "CapitalManagement") -> None:
    """
    Validate exchange name with consistent error handling.

    Args:
        exchange: Exchange name to validate
        component: Component name for error context

    Raises:
        ValidationError: If exchange name is invalid
    """
    if not exchange or not exchange.strip():
        raise ValidationError(
            "Exchange cannot be empty",
            error_code="VAL_004",
            details={"component": component, "field": "exchange"},
        )


def validate_currency_code(currency: str, component: str = "CapitalManagement") -> None:
    """
    Validate currency code with consistent error handling.

    Args:
        currency: Currency code to validate
        component: Component name for error context

    Raises:
        ValidationError: If currency code is invalid
    """
    if not currency or not currency.strip():
        raise ValidationError(
            "Currency code cannot be empty",
            error_code="VAL_005",
            details={"component": component, "field": "currency"},
        )

    # Basic currency format validation
    if len(currency.strip()) < 2 or len(currency.strip()) > 10:
        raise ValidationError(
            f"Invalid currency code format: {currency}",
            error_code="VAL_006",
            details={"currency": currency, "component": component},
        )


def validate_percentage(
    percentage: Decimal,
    percentage_name: str = "percentage",
    min_value: Decimal = Decimal("0"),
    max_value: Decimal = Decimal("1"),
    component: str = "CapitalManagement",
) -> None:
    """
    Validate percentage value with consistent error handling.

    Args:
        percentage: Percentage to validate (0.0 to 1.0)
        percentage_name: Name of the percentage field for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        component: Component name for error context

    Raises:
        ValidationError: If percentage is invalid
    """
    if not isinstance(percentage, Decimal):
        raise ValidationError(
            f"Invalid {percentage_name} type: expected Decimal, got {type(percentage).__name__}",
            error_code="VAL_007",
            details={"component": component, "field": percentage_name},
        )

    if percentage < min_value or percentage > max_value:
        raise ValidationError(
            f"Invalid {percentage_name}: {percentage:.4f} must be between {min_value:.4f} and {max_value:.4f}",
            error_code="VAL_008",
            details={
                "percentage": str(percentage),
                "min_value": str(min_value),
                "max_value": str(max_value),
                "component": component,
            },
        )


def validate_allocation_request(
    strategy_id: str, exchange: str, amount: Decimal, component: str = "CapitalManagement"
) -> None:
    """
    Validate complete capital allocation request.

    Args:
        strategy_id: Strategy identifier
        exchange: Exchange name
        amount: Amount to allocate
        component: Component name for error context

    Raises:
        ValidationError: If any validation fails
    """
    validate_strategy_id(strategy_id, component)
    validate_exchange_name(exchange, component)
    validate_capital_amount(amount, "allocation amount", component=component)


def validate_withdrawal_request(
    amount: Decimal,
    currency: str = "USDT",
    exchange: str = "binance",
    min_amount: Decimal = Decimal("100"),
    component: str = "CapitalManagement",
) -> None:
    """
    Validate withdrawal request.

    Args:
        amount: Amount to withdraw
        currency: Currency code
        exchange: Exchange name
        min_amount: Minimum withdrawal amount
        component: Component name for error context

    Raises:
        ValidationError: If any validation fails
    """
    validate_capital_amount(amount, "withdrawal amount", min_amount, component)
    validate_currency_code(currency, component)
    validate_exchange_name(exchange, component)


def validate_supported_currencies(
    currency: str, supported_currencies: list[str], component: str = "CapitalManagement"
) -> None:
    """
    Validate that currency is supported.

    Args:
        currency: Currency code to validate
        supported_currencies: List of supported currencies
        component: Component name for error context

    Raises:
        ValidationError: If currency is not supported
    """
    validate_currency_code(currency, component)

    if currency not in supported_currencies:
        raise ValidationError(
            f"Unsupported currency: {currency}",
            error_code="VAL_009",
            details={
                "currency": currency,
                "supported_currencies": supported_currencies,
                "component": component,
            },
        )


def safe_decimal_conversion(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    """
    Safely convert any value to Decimal with consistent handling.

    DEPRECATED: Use centralized_safe_decimal_conversion from src.utils.decimal_utils instead.
    This function is kept for backward compatibility and delegates to the centralized version.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Decimal: Converted value or default
    """
    # Delegate to centralized function with precision=-1 to avoid quantization
    return centralized_safe_decimal_conversion(value, precision=-1, default=default)
