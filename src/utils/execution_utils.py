"""
Execution Utilities - Common functions for order execution and validation.

This module consolidates frequently used execution-related functions to eliminate
code duplication across the execution module. All functions maintain financial
precision using Decimal types.
"""

from decimal import Decimal
from typing import Any

from src.core.types import MarketData, Order, OrderRequest
from src.utils.decimal_utils import ZERO, safe_divide, to_decimal


def calculate_order_value(
    quantity: Decimal,
    price: Decimal | None,
    market_data: MarketData | None = None,
    default_price: Decimal = Decimal("50000"),
) -> Decimal:
    """
    Calculate the total value of an order with proper fallback logic.

    Args:
        quantity: Order quantity
        price: Order price (None for market orders)
        market_data: Current market data for price fallback
        default_price: Default price if no other price available

    Returns:
        Decimal: Total order value
    """
    if price is not None:
        return quantity * price

    if market_data is not None:
        if hasattr(market_data, "price") and market_data.price:
            return quantity * market_data.price
        elif hasattr(market_data, "close") and market_data.close:
            return quantity * market_data.close

    return quantity * default_price


# safe_decimal_conversion function moved to src.utils.decimal_utils for centralization


def calculate_price_deviation_bps(order_price: Decimal, market_price: Decimal) -> Decimal:
    """
    Calculate price deviation in basis points with Decimal precision.

    Args:
        order_price: Order price
        market_price: Current market price

    Returns:
        Decimal: Deviation in basis points
    """
    if market_price <= ZERO:
        return ZERO

    price_diff = abs(order_price - market_price)
    return safe_divide(price_diff * to_decimal("10000"), market_price, ZERO)


def is_order_within_price_bounds(
    order: Order, market_data: MarketData, max_deviation_percent: Decimal = to_decimal("0.10")
) -> bool:
    """
    Check if order price is within acceptable bounds of market price.

    Args:
        order: Order to check
        market_data: Current market data
        max_deviation_percent: Maximum allowed deviation (e.g., 0.10 for 10%)

    Returns:
        bool: True if price is within bounds
    """
    if not order.price:
        return True  # Market order

    current_price = market_data.price if hasattr(market_data, "price") else market_data.close
    if not current_price:
        return False

    one = to_decimal("1")
    lower_bound = current_price * (one - max_deviation_percent)
    upper_bound = current_price * (one + max_deviation_percent)

    return lower_bound <= order.price <= upper_bound


def calculate_trade_risk_ratio(order_value: Decimal, account_value: Decimal) -> Decimal:
    """
    Calculate risk ratio of trade relative to account value with Decimal precision.

    Args:
        order_value: Value of the order
        account_value: Total account value

    Returns:
        Decimal: Risk ratio (0.0 to 1.0+)
    """
    if account_value <= ZERO:
        return to_decimal("inf")  # Return a very large number instead of inf

    return safe_divide(order_value, account_value, ZERO)


def extract_order_details(order: Order) -> dict[str, Any]:
    """
    Extract order details into a standardized dictionary.

    Args:
        order: Order object

    Returns:
        dict: Standardized order details
    """
    return {
        "id": str(order.id),
        "symbol": order.symbol,
        "side": order.side.value,
        "quantity": str(order.quantity),
        "price": str(order.price) if order.price else None,
        "type": order.order_type.value if hasattr(order, "order_type") else order.type.value,
        "exchange": getattr(order, "exchange", None),
        "timestamp": getattr(order, "timestamp", None),
    }


def convert_order_to_request(order: Order) -> OrderRequest:
    """
    Convert Order object to OrderRequest for compatibility.

    Args:
        order: Order object

    Returns:
        OrderRequest: Converted order request
    """
    return OrderRequest(
        symbol=order.symbol,
        side=order.side,
        order_type=order.order_type if hasattr(order, "order_type") else order.type,
        quantity=order.quantity,
        price=order.price,
        time_in_force=getattr(order, "time_in_force", None),
        exchange=getattr(order, "exchange", None),
    )


def validate_order_basic(order: Order) -> list[str]:
    """
    Perform basic order validation and return list of errors.

    Args:
        order: Order to validate

    Returns:
        list[str]: List of validation errors (empty if valid)
    """
    errors = []

    if not order.symbol:
        errors.append("Order symbol is required")

    if order.quantity <= ZERO:
        errors.append("Order quantity must be positive")

    if order.price is not None and order.price <= ZERO:
        errors.append("Order price must be positive if specified")

    return errors


def calculate_slippage_bps(executed_price: Decimal, expected_price: Decimal) -> Decimal:
    """
    Calculate slippage in basis points with Decimal precision.

    Args:
        executed_price: Actual execution price
        expected_price: Expected execution price

    Returns:
        Decimal: Slippage in basis points
    """
    if expected_price <= ZERO:
        return ZERO

    slippage = abs(executed_price - expected_price)
    return safe_divide(slippage * to_decimal("10000"), expected_price, ZERO)
