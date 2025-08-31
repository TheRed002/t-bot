"""
Common state validation utilities.

This module provides shared validation functions for state management
to eliminate duplication across state validation components.
"""

import re
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime, timezone

from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.core.types import BotStatus, OrderSide, OrderType

logger = get_logger(__name__)


def validate_required_fields_with_details(
    data: Dict[str, Any], 
    required_fields: List[str]
) -> Dict[str, Any]:
    """
    Validate required fields and return detailed results.
    
    Args:
        data: Data to validate
        required_fields: List of required field names
        
    Returns:
        Dictionary with validation results
    """
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    
    return {
        "passed": len(missing_fields) == 0,
        "message": f"Missing required fields: {missing_fields}" if missing_fields else "All required fields present",
        "missing_fields": missing_fields
    }


def validate_string_field_with_details(
    data: Dict[str, Any], 
    field_name: str
) -> Dict[str, Any]:
    """
    Validate that a field is a string with detailed results.
    
    Args:
        data: Data to validate
        field_name: Field name to validate
        
    Returns:
        Dictionary with validation results
    """
    value = data.get(field_name)
    is_valid = isinstance(value, str)
    
    return {
        "passed": is_valid,
        "message": f"{field_name} must be a string" if not is_valid else "Valid string",
        "current_value": value,
        "expected_type": "string"
    }


def validate_decimal_field_with_details(
    data: Dict[str, Any], 
    field_name: str, 
    max_places: int = 8
) -> Dict[str, Any]:
    """
    Validate that a field is a valid decimal with detailed results.
    
    Args:
        data: Data to validate
        field_name: Field name to validate
        max_places: Maximum decimal places allowed
        
    Returns:
        Dictionary with validation results
    """
    value = data.get(field_name)
    
    try:
        if value is None:
            return {
                "passed": False, 
                "message": f"{field_name} is required",
                "current_value": value
            }
        
        decimal_value = Decimal(str(value))
        
        # Check decimal places
        if decimal_value.as_tuple().exponent < -max_places:
            return {
                "passed": False,
                "message": f"{field_name} has too many decimal places (max {max_places})",
                "current_value": value,
                "decimal_places": abs(decimal_value.as_tuple().exponent)
            }
        
        return {
            "passed": True, 
            "message": "Valid decimal value", 
            "current_value": value,
            "decimal_value": decimal_value
        }
        
    except (ValueError, TypeError) as e:
        return {
            "passed": False,
            "message": f"{field_name} must be a valid decimal: {e}",
            "current_value": value,
            "error": str(e)
        }


def validate_positive_value_with_details(
    data: Dict[str, Any], 
    field_name: str
) -> Dict[str, Any]:
    """
    Validate that a numeric field is positive with detailed results.
    
    Args:
        data: Data to validate
        field_name: Field name to validate
        
    Returns:
        Dictionary with validation results
    """
    value = data.get(field_name)
    
    try:
        decimal_value = Decimal(str(value)) if value is not None else None
        is_positive = decimal_value is not None and decimal_value > 0
        
        return {
            "passed": is_positive,
            "message": f"{field_name} must be positive" if not is_positive else "Valid positive value",
            "current_value": value,
            "expected_condition": "> 0",
            "decimal_value": decimal_value
        }
        
    except (ValueError, TypeError):
        return {
            "passed": False,
            "message": f"{field_name} must be a valid positive number",
            "current_value": value,
            "expected_type": "positive number"
        }


def validate_non_negative_value_with_details(
    data: Dict[str, Any], 
    field_name: str
) -> Dict[str, Any]:
    """
    Validate that a numeric field is non-negative with detailed results.
    
    Args:
        data: Data to validate
        field_name: Field name to validate
        
    Returns:
        Dictionary with validation results
    """
    value = data.get(field_name)
    
    try:
        decimal_value = Decimal(str(value)) if value is not None else None
        is_non_negative = decimal_value is not None and decimal_value >= 0
        
        return {
            "passed": is_non_negative,
            "message": f"{field_name} cannot be negative" if not is_non_negative else "Valid non-negative value",
            "current_value": value,
            "expected_condition": ">= 0",
            "decimal_value": decimal_value
        }
        
    except (ValueError, TypeError):
        return {
            "passed": False,
            "message": f"{field_name} must be a valid non-negative number",
            "current_value": value,
            "expected_type": "non-negative number"
        }


def validate_list_field_with_details(
    data: Dict[str, Any], 
    field_name: str
) -> Dict[str, Any]:
    """
    Validate that a field is a list with detailed results.
    
    Args:
        data: Data to validate
        field_name: Field name to validate
        
    Returns:
        Dictionary with validation results
    """
    value = data.get(field_name)
    is_list = isinstance(value, list)
    
    return {
        "passed": is_list,
        "message": f"{field_name} must be a list" if not is_list else "Valid list",
        "current_value": value,
        "current_type": type(value).__name__ if value is not None else None,
        "expected_type": "list",
        "list_length": len(value) if is_list else None
    }


def validate_dict_field_with_details(
    data: Dict[str, Any], 
    field_name: str
) -> Dict[str, Any]:
    """
    Validate that a field is a dictionary with detailed results.
    
    Args:
        data: Data to validate
        field_name: Field name to validate
        
    Returns:
        Dictionary with validation results
    """
    value = data.get(field_name)
    is_dict = isinstance(value, dict)
    
    return {
        "passed": is_dict,
        "message": f"{field_name} must be a dictionary" if not is_dict else "Valid dictionary",
        "current_value": value,
        "current_type": type(value).__name__ if value is not None else None,
        "expected_type": "dict",
        "dict_keys": list(value.keys()) if is_dict else None
    }


def validate_bot_id_format(bot_id: str) -> Dict[str, Any]:
    """
    Validate bot ID format with detailed results.
    
    Args:
        bot_id: Bot ID to validate
        
    Returns:
        Dictionary with validation results
    """
    if not bot_id:
        return {
            "passed": False,
            "message": "Bot ID cannot be empty",
            "current_value": bot_id
        }
    
    # Bot ID should be alphanumeric with dashes and underscores
    pattern = r"^[a-zA-Z0-9_-]+$"
    is_valid = re.match(pattern, bot_id) is not None
    
    return {
        "passed": is_valid,
        "message": "Valid bot ID format" if is_valid else "Bot ID must contain only letters, numbers, dashes, and underscores",
        "current_value": bot_id,
        "pattern": pattern,
        "expected_format": "alphanumeric with - and _"
    }


def validate_bot_status(status: Any) -> Dict[str, Any]:
    """
    Validate bot status value with detailed results.
    
    Args:
        status: Status value to validate
        
    Returns:
        Dictionary with validation results
    """
    if isinstance(status, BotStatus):
        return {
            "passed": True, 
            "message": "Valid bot status",
            "current_value": status.value,
            "status_enum": True
        }
    
    if isinstance(status, str):
        valid_statuses = {s.value for s in BotStatus}
        is_valid = status in valid_statuses
        
        return {
            "passed": is_valid,
            "message": "Valid bot status" if is_valid else f"Invalid bot status: {status}",
            "current_value": status,
            "valid_values": list(valid_statuses),
            "status_enum": False
        }
    
    return {
        "passed": False,
        "message": "Bot status must be a BotStatus enum or valid string",
        "current_value": status,
        "current_type": type(status).__name__,
        "expected_type": "BotStatus or string"
    }


def validate_order_side(side: Any) -> Dict[str, Any]:
    """
    Validate order side with detailed results.
    
    Args:
        side: Order side to validate
        
    Returns:
        Dictionary with validation results
    """
    if isinstance(side, OrderSide):
        return {
            "passed": True,
            "message": "Valid order side",
            "current_value": side.value,
            "side_enum": True
        }
    
    if isinstance(side, str):
        valid_sides = {s.value for s in OrderSide}
        is_valid = side.upper() in valid_sides
        
        return {
            "passed": is_valid,
            "message": "Valid order side" if is_valid else f"Invalid order side: {side}",
            "current_value": side,
            "valid_values": list(valid_sides),
            "side_enum": False
        }
    
    return {
        "passed": False,
        "message": "Order side must be an OrderSide enum or valid string",
        "current_value": side,
        "current_type": type(side).__name__,
        "expected_type": "OrderSide or string"
    }


def validate_order_type(order_type: Any) -> Dict[str, Any]:
    """
    Validate order type with detailed results.
    
    Args:
        order_type: Order type to validate
        
    Returns:
        Dictionary with validation results
    """
    if isinstance(order_type, OrderType):
        return {
            "passed": True,
            "message": "Valid order type",
            "current_value": order_type.value,
            "type_enum": True
        }
    
    if isinstance(order_type, str):
        valid_types = {t.value for t in OrderType}
        is_valid = order_type.upper() in valid_types
        
        return {
            "passed": is_valid,
            "message": "Valid order type" if is_valid else f"Invalid order type: {order_type}",
            "current_value": order_type,
            "valid_values": list(valid_types),
            "type_enum": False
        }
    
    return {
        "passed": False,
        "message": "Order type must be an OrderType enum or valid string",
        "current_value": order_type,
        "current_type": type(order_type).__name__,
        "expected_type": "OrderType or string"
    }


def validate_symbol_format(symbol: str) -> Dict[str, Any]:
    """
    Validate trading symbol format with detailed results.
    
    Args:
        symbol: Symbol to validate
        
    Returns:
        Dictionary with validation results
    """
    if not symbol:
        return {
            "passed": False,
            "message": "Symbol cannot be empty",
            "current_value": symbol
        }
    
    # Symbol should be uppercase letters with optional separators
    pattern = r"^[A-Z0-9]+([\/\-][A-Z0-9]+)*$"
    is_valid = re.match(pattern, symbol.upper()) is not None
    
    return {
        "passed": is_valid,
        "message": "Valid symbol format" if is_valid else "Symbol must be in valid format (e.g., BTC/USD, BTCUSD)",
        "current_value": symbol,
        "normalized_value": symbol.upper(),
        "pattern": pattern,
        "expected_format": "Valid trading symbol format"
    }


def validate_capital_allocation(data: Dict[str, Any], max_allocation: Optional[Decimal] = None) -> Dict[str, Any]:
    """
    Validate capital allocation limits with detailed results.
    
    Args:
        data: Data containing allocation
        max_allocation: Maximum allowed allocation
        
    Returns:
        Dictionary with validation results
    """
    allocation = data.get("capital_allocation")
    if allocation is None:
        return {
            "passed": True,
            "message": "No capital allocation specified",
            "current_value": allocation
        }
    
    try:
        allocation_value = Decimal(str(allocation))
        
        # Check if allocation is positive
        if allocation_value <= 0:
            return {
                "passed": False,
                "message": "Capital allocation must be positive",
                "current_value": allocation,
                "decimal_value": allocation_value
            }
        
        # Check against maximum allocation
        max_allocation = max_allocation or Decimal("1000000")  # Default $1M
        if allocation_value > max_allocation:
            return {
                "passed": False,
                "message": f"Capital allocation exceeds maximum of ${max_allocation}",
                "current_value": allocation,
                "decimal_value": allocation_value,
                "max_allowed": max_allocation
            }
        
        return {
            "passed": True,
            "message": "Valid capital allocation",
            "current_value": allocation,
            "decimal_value": allocation_value,
            "max_allowed": max_allocation
        }
        
    except (ValueError, TypeError):
        return {
            "passed": False,
            "message": "Capital allocation must be a valid number",
            "current_value": allocation,
            "expected_type": "positive number"
        }


def validate_order_price_logic(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate order price logic based on order type with detailed results.
    
    Args:
        data: Order data to validate
        
    Returns:
        Dictionary with validation results
    """
    order_type = data.get("type")
    price = data.get("price")
    
    if not order_type:
        return {
            "passed": True,
            "message": "No order type specified",
            "current_value": {"type": order_type, "price": price}
        }
    
    # Market orders shouldn't have price
    if order_type in ["MARKET", "market"]:
        if price is not None:
            return {
                "passed": False,
                "message": "Market orders should not specify price",
                "current_value": price,
                "recommendation": "Remove price for market orders",
                "order_type": order_type
            }
    
    # Limit and stop orders require price
    elif order_type in ["LIMIT", "limit", "STOP", "stop", "STOP_LIMIT", "stop_limit"]:
        if price is None:
            return {
                "passed": False,
                "message": f"{order_type} orders require price",
                "current_value": price,
                "expected_value": "positive price value",
                "order_type": order_type
            }
        
        # Validate price is positive
        try:
            price_value = Decimal(str(price))
            if price_value <= 0:
                return {
                    "passed": False,
                    "message": "Order price must be positive",
                    "current_value": price,
                    "decimal_value": price_value,
                    "order_type": order_type
                }
        except (ValueError, TypeError):
            return {
                "passed": False,
                "message": "Order price must be a valid number",
                "current_value": price,
                "expected_type": "positive number",
                "order_type": order_type
            }
    
    return {
        "passed": True,
        "message": "Valid order price logic",
        "order_type": order_type,
        "price": price
    }


def validate_cash_balance(data: Dict[str, Any], min_cash_ratio: Optional[Decimal] = None) -> Dict[str, Any]:
    """
    Validate cash balance against positions with detailed results.
    
    Args:
        data: Data containing cash balance and positions
        min_cash_ratio: Minimum cash ratio (default 0.1 = 10%)
        
    Returns:
        Dictionary with validation results
    """
    cash_balance = data.get("cash_balance")
    positions = data.get("positions", [])
    min_cash_ratio = min_cash_ratio or Decimal("0.1")  # 10% minimum cash
    
    if cash_balance is None:
        return {
            "passed": True,
            "message": "No cash balance specified",
            "current_value": cash_balance
        }
    
    try:
        cash_value = Decimal(str(cash_balance))
        
        # Calculate total position value (simplified)
        total_position_value = Decimal("0")
        for position in positions:
            if isinstance(position, dict):
                quantity = position.get("quantity", 0)
                price = position.get("current_price", position.get("entry_price", 0))
                if quantity and price:
                    total_position_value += Decimal(str(quantity)) * Decimal(str(price))
        
        total_value = cash_value + total_position_value
        cash_ratio = cash_value / total_value if total_value > 0 else Decimal("1")
        
        if total_value > 0 and cash_ratio < min_cash_ratio:
            return {
                "passed": False,
                "message": f"Cash balance too low: {cash_value} ({cash_ratio * 100:.1f}%)",
                "current_value": str(cash_value),
                "cash_ratio": float(cash_ratio),
                "min_ratio": float(min_cash_ratio),
                "recommendation": f"Maintain at least {min_cash_ratio * 100}% cash",
                "total_value": str(total_value),
                "position_value": str(total_position_value)
            }
        
        return {
            "passed": True,
            "message": "Adequate cash balance",
            "current_value": str(cash_value),
            "cash_ratio": float(cash_ratio),
            "min_ratio": float(min_cash_ratio),
            "total_value": str(total_value),
            "position_value": str(total_position_value)
        }
        
    except (ValueError, TypeError):
        return {
            "passed": False,
            "message": "Cash balance must be a valid number",
            "current_value": cash_balance,
            "expected_type": "positive number"
        }


def validate_var_limits(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate VaR against risk limits with detailed results.
    
    Args:
        data: Data containing VaR and risk limits
        
    Returns:
        Dictionary with validation results
    """
    var_value = data.get("var")
    risk_limits = data.get("risk_limits", {})
    
    if var_value is None:
        return {
            "passed": True,
            "message": "No VaR specified",
            "current_value": var_value
        }
    
    try:
        var_decimal = Decimal(str(var_value))
        max_var = risk_limits.get("max_var", "0.02")  # Default 2%
        max_var_decimal = Decimal(str(max_var))
        
        if var_decimal > max_var_decimal:
            return {
                "passed": False,
                "message": f"VaR {var_decimal} exceeds limit {max_var_decimal}",
                "current_value": str(var_decimal),
                "max_allowed": str(max_var_decimal),
                "var_percentage": float(var_decimal * 100),
                "limit_percentage": float(max_var_decimal * 100)
            }
        
        return {
            "passed": True,
            "message": "VaR within limits",
            "current_value": str(var_decimal),
            "max_allowed": str(max_var_decimal),
            "var_percentage": float(var_decimal * 100),
            "limit_percentage": float(max_var_decimal * 100)
        }
        
    except (ValueError, TypeError):
        return {
            "passed": False,
            "message": "VaR must be a valid number",
            "current_value": var_value,
            "expected_type": "decimal percentage"
        }


def validate_trade_execution(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate trade execution data with detailed results.
    
    Args:
        data: Trade data with execution information
        
    Returns:
        Dictionary with validation results
    """
    execution_data = data.get("execution", {})
    
    if not isinstance(execution_data, dict):
        return {
            "passed": True,
            "message": "No execution data provided",
            "current_value": execution_data
        }
    
    # Check for required execution fields
    required_fields = ["filled_quantity", "average_price"]
    missing_fields = [field for field in required_fields if field not in execution_data]
    
    if missing_fields:
        return {
            "passed": False,
            "message": f"Missing execution fields: {missing_fields}",
            "current_value": list(execution_data.keys()),
            "expected_fields": required_fields,
            "missing_fields": missing_fields
        }
    
    # Validate execution values
    try:
        filled_qty = Decimal(str(execution_data["filled_quantity"]))
        avg_price = Decimal(str(execution_data["average_price"]))
        
        if filled_qty < 0:
            return {
                "passed": False,
                "message": "Filled quantity cannot be negative",
                "current_value": str(filled_qty),
                "field": "filled_quantity"
            }
        
        if avg_price <= 0:
            return {
                "passed": False,
                "message": "Average price must be positive",
                "current_value": str(avg_price),
                "field": "average_price"
            }
        
        return {
            "passed": True,
            "message": "Valid trade execution",
            "filled_quantity": str(filled_qty),
            "average_price": str(avg_price),
            "execution_value": str(filled_qty * avg_price)
        }
        
    except (ValueError, TypeError):
        return {
            "passed": False,
            "message": "Execution data must contain valid numbers",
            "current_value": execution_data,
            "expected_types": {"filled_quantity": "non-negative number", "average_price": "positive number"}
        }


def validate_strategy_params(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate strategy parameters with detailed results.
    
    Args:
        data: Strategy data with parameters
        
    Returns:
        Dictionary with validation results
    """
    params = data.get("params", {})
    
    if not isinstance(params, dict):
        return {
            "passed": False,
            "message": "Strategy parameters must be a dictionary",
            "current_value": params,
            "current_type": type(params).__name__,
            "expected_type": "dict"
        }
    
    # Check for required strategy parameters
    required_params = {"timeframe", "risk_per_trade"}
    missing_params = required_params - set(params.keys())
    
    if missing_params:
        return {
            "passed": False,
            "message": f"Missing required strategy parameters: {missing_params}",
            "current_params": list(params.keys()),
            "required_params": list(required_params),
            "missing_params": list(missing_params)
        }
    
    # Validate risk_per_trade is reasonable
    risk_per_trade = params.get("risk_per_trade")
    if risk_per_trade:
        try:
            risk_value = Decimal(str(risk_per_trade))
            max_risk = Decimal("0.05")  # 5% max risk per trade
            
            if risk_value > max_risk:
                return {
                    "passed": False,
                    "message": f"Risk per trade too high: {risk_value}",
                    "current_value": str(risk_value),
                    "max_allowed": str(max_risk),
                    "risk_percentage": float(risk_value * 100),
                    "max_percentage": float(max_risk * 100)
                }
        except (ValueError, TypeError):
            return {
                "passed": False,
                "message": "Risk per trade must be a valid number",
                "current_value": risk_per_trade,
                "expected_type": "decimal percentage"
            }
    
    return {
        "passed": True,
        "message": "Valid strategy parameters",
        "params": params,
        "risk_per_trade": risk_per_trade
    }