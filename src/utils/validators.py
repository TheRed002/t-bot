"""
Validation Utilities for Data Integrity

This module provides comprehensive validation functions for financial data, configuration,
API input, data types, business rules, and exchange data to ensure data integrity
across all components of the trading bot system.

Key Functions:
- Financial Data Validation: price ranges, volume checks, symbol validation
- Configuration Validation: parameter bounds, required fields
- API Input Validation: request payload validation, security checks
- Data Type Validation: type checking, schema validation
- Business Rule Validation: trading rules, risk limit validation
- Exchange Data Validation: order validation, balance verification

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
"""

import re
from typing import Any, Dict, List, Optional, Union, Tuple
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

# Import from P-001 core components
from src.core.exceptions import ValidationError, DataError
from src.core.types import (
    OrderRequest, OrderResponse, MarketData, Position,
    Signal, SignalDirection, OrderSide, OrderType
)
from src.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Financial Data Validation
# =============================================================================

def validate_price(price: float, symbol: str) -> Decimal:
    """
    Validate and normalize price values.
    
    Args:
        price: Price to validate
        symbol: Trading symbol for context
        
    Returns:
        Normalized price as Decimal
        
    Raises:
        ValidationError: If price is invalid
    """
    if not isinstance(price, (int, float, Decimal)):
        raise ValidationError(f"Price must be a number for {symbol}, got {type(price).__name__}")
    
    if price <= 0:
        raise ValidationError(f"Price must be positive for {symbol}, got {price}")
    
    if price > 1_000_000:  # Sanity check for extremely high prices
        raise ValidationError(f"Price {price} for {symbol} exceeds maximum allowed")
    
    try:
        # Convert to Decimal for financial precision
        decimal_price = Decimal(str(price))
        
        # Determine precision based on symbol
        if "BTC" in symbol.upper():
            precision = 8
        elif "ETH" in symbol.upper():
            precision = 6
        elif "USDT" in symbol.upper() or "USD" in symbol.upper():
            precision = 2
        else:
            precision = 4
        
        # Round to appropriate precision
        normalized_price = decimal_price.quantize(
            Decimal(f"0.{'0' * (precision - 1)}1")
        )
        
        return normalized_price
        
    except (InvalidOperation, ValueError) as e:
        raise ValidationError(f"Invalid price format for {symbol}: {str(e)}")


def validate_quantity(quantity: float, symbol: str, min_qty: Optional[float] = None) -> Decimal:
    """
    Validate trading quantity.
    
    Args:
        quantity: Quantity to validate
        symbol: Trading symbol for context
        min_qty: Minimum quantity allowed
        
    Returns:
        Normalized quantity as Decimal
        
    Raises:
        ValidationError: If quantity is invalid
    """
    if not isinstance(quantity, (int, float, Decimal)):
        raise ValidationError(f"Quantity must be a number for {symbol}, got {type(quantity).__name__}")
    
    if quantity <= 0:
        raise ValidationError(f"Quantity must be positive for {symbol}, got {quantity}")
    
    if min_qty and quantity < min_qty:
        raise ValidationError(f"Quantity {quantity} below minimum {min_qty} for {symbol}")
    
    try:
        # Convert to Decimal for financial precision
        decimal_qty = Decimal(str(quantity))
        
        # Determine precision based on symbol
        if "BTC" in symbol.upper():
            precision = 8
        elif "ETH" in symbol.upper():
            precision = 6
        else:
            precision = 4
        
        # Round to appropriate precision
        normalized_qty = decimal_qty.quantize(
            Decimal(f"0.{'0' * (precision - 1)}1")
        )
        
        return normalized_qty
        
    except (InvalidOperation, ValueError) as e:
        raise ValidationError(f"Invalid quantity format for {symbol}: {str(e)}")


def validate_symbol(symbol: str) -> str:
    """
    Validate trading symbol format.
    
    Args:
        symbol: Symbol to validate
        
    Returns:
        Normalized symbol string
        
    Raises:
        ValidationError: If symbol is invalid
    """
    if not symbol or not isinstance(symbol, str):
        raise ValidationError("Symbol cannot be empty")
    
    if len(symbol) < 3:
        raise ValidationError(f"Symbol too short: {symbol}")
    
    if len(symbol) > 20:
        raise ValidationError(f"Symbol too long: {symbol}")
    
    # Remove whitespace and convert to uppercase
    normalized = symbol.strip().upper()
    
    # Check for valid characters (alphanumeric and common separators)
    if not re.match(r'^[A-Z0-9/_-]+$', normalized):
        raise ValidationError(f"Symbol contains invalid characters: {symbol}")
    
    # Check for common patterns
    if '/' in normalized:
        parts = normalized.split('/')
        if len(parts) != 2:
            raise ValidationError(f"Invalid symbol format: {symbol}")
        if not parts[0] or not parts[1]:
            raise ValidationError(f"Invalid symbol format: {symbol}")
    
    return normalized


def validate_order_request(order: OrderRequest) -> bool:
    """
    Validate order request data.
    
    Args:
        order: OrderRequest object to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If order request is invalid
    """
    try:
        # Validate symbol
        validate_symbol(order.symbol)
        
        # Validate quantity
        validate_quantity(float(order.quantity), order.symbol)
        
        # Validate price for limit orders
        if order.order_type == OrderType.LIMIT:
            if not order.price:
                raise ValidationError("Price is required for limit orders")
            validate_price(float(order.price), order.symbol)
        
        # Validate stop price for stop orders
        if order.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
            if not order.stop_price:
                raise ValidationError("Stop price is required for stop orders")
            validate_price(float(order.stop_price), order.symbol)
        
        # Validate time in force
        valid_tif = ["GTC", "IOC", "FOK"]
        if order.time_in_force not in valid_tif:
            raise ValidationError(f"Invalid time in force: {order.time_in_force}")
        
        return True
        
    except Exception as e:
        raise ValidationError(f"Order request validation failed: {str(e)}")


def validate_market_data(data: MarketData) -> bool:
    """
    Validate market data.
    
    Args:
        data: MarketData object to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If market data is invalid
    """
    try:
        # Validate symbol
        validate_symbol(data.symbol)
        
        # Validate price
        validate_price(float(data.price), data.symbol)
        
        # Validate volume
        if data.volume <= 0:
            raise ValidationError(f"Volume must be positive for {data.symbol}")
        
        # Validate timestamp
        if not isinstance(data.timestamp, datetime):
            raise ValidationError("Timestamp must be a datetime object")
        
        # Validate bid/ask if present
        if data.bid is not None:
            validate_price(float(data.bid), data.symbol)
        if data.ask is not None:
            validate_price(float(data.ask), data.symbol)
        
        # Validate OHLC data if present
        if data.open_price is not None:
            validate_price(float(data.open_price), data.symbol)
        if data.high_price is not None:
            validate_price(float(data.high_price), data.symbol)
        if data.low_price is not None:
            validate_price(float(data.low_price), data.symbol)
        
        return True
        
    except Exception as e:
        raise ValidationError(f"Market data validation failed: {str(e)}")


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_config(config: Dict[str, Any], required_fields: List[str] = None) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        required_fields: List of required field names
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError("Configuration must be a dictionary")
    
    if required_fields:
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Required field missing: {field}")
            if config[field] is None:
                raise ValidationError(f"Required field cannot be None: {field}")
    
    return True


def validate_risk_parameters(params: Dict[str, Any]) -> bool:
    """
    Validate risk management parameters.
    
    Args:
        params: Risk parameters dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If risk parameters are invalid
    """
    required_fields = [
        "max_position_size", "max_daily_loss", "max_drawdown"
    ]
    
    validate_config(params, required_fields)
    
    # Validate numeric ranges
    if params["max_position_size"] <= 0 or params["max_position_size"] > 1:
        raise ValidationError("max_position_size must be between 0 and 1")
    
    if params["max_daily_loss"] <= 0 or params["max_daily_loss"] > 1:
        raise ValidationError("max_daily_loss must be between 0 and 1")
    
    if params["max_drawdown"] <= 0 or params["max_drawdown"] > 1:
        raise ValidationError("max_drawdown must be between 0 and 1")
    
    return True


def validate_signal(signal: Signal) -> bool:
    """
    Validate trading signal.
    
    Args:
        signal: Signal to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If signal is invalid
    """
    if not isinstance(signal, Signal):
        raise ValidationError("Signal must be a Signal instance")
    
    # Validate confidence
    if not 0 <= signal.confidence <= 1:
        raise ValidationError("Signal confidence must be between 0 and 1")
    
    # Validate direction
    valid_directions = ["buy", "sell", "hold"]
    if signal.direction.value not in valid_directions:
        raise ValidationError(f"Invalid signal direction: {signal.direction}")
    
    # Validate symbol
    validate_symbol(signal.symbol)
    
    # Validate timestamp
    if not isinstance(signal.timestamp, datetime):
        raise ValidationError("Signal timestamp must be a datetime instance")
    
    # Validate strategy name
    if not signal.strategy_name or not isinstance(signal.strategy_name, str):
        raise ValidationError("Signal must have a valid strategy name")
    
    return True


def validate_strategy_config(config: Dict[str, Any]) -> bool:
    """
    Validate strategy configuration.
    
    Args:
        config: Strategy configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If strategy configuration is invalid
    """
    required_fields = [
        "name", "strategy_type", "symbols", "timeframe"
    ]
    
    validate_config(config, required_fields)
    
    # Validate strategy type
    valid_types = ["static", "dynamic", "arbitrage", "market_making", "ai_ml"]
    if config["strategy_type"] not in valid_types:
        raise ValidationError(f"Invalid strategy type: {config['strategy_type']}")
    
    # Validate symbols
    if not isinstance(config["symbols"], list) or not config["symbols"]:
        raise ValidationError("Symbols must be a non-empty list")
    
    for symbol in config["symbols"]:
        validate_symbol(symbol)
    
    # Validate timeframe
    valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    if config["timeframe"] not in valid_timeframes:
        raise ValidationError(f"Invalid timeframe: {config['timeframe']}")
    
    # Validate optional parameters
    if "min_confidence" in config:
        if not 0 <= config["min_confidence"] <= 1:
            raise ValidationError("min_confidence must be between 0 and 1")
    
    if "position_size_pct" in config:
        if not 0 < config["position_size_pct"] <= 1:
            raise ValidationError("position_size_pct must be between 0 and 1")
    
    return True


# =============================================================================
# API Input Validation
# =============================================================================

def validate_api_request(request_data: Dict[str, Any], required_fields: List[str] = None) -> bool:
    """
    Validate API request payload.
    
    Args:
        request_data: Request data dictionary
        required_fields: List of required field names
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If API request is invalid
    """
    if not isinstance(request_data, dict):
        raise ValidationError("Request data must be a dictionary")
    
    if required_fields:
        for field in required_fields:
            if field not in request_data:
                raise ValidationError(f"Required field missing: {field}")
    
    return True


def validate_webhook_payload(payload: Dict[str, Any], signature: str = None) -> bool:
    """
    Validate webhook payload with optional signature verification.
    
    Args:
        payload: Webhook payload dictionary
        signature: Optional signature for verification
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If webhook payload is invalid
    """
    if not isinstance(payload, dict):
        raise ValidationError("Webhook payload must be a dictionary")
    
    # Validate required fields for common webhook types
    if "event_type" not in payload:
        raise ValidationError("Webhook payload must contain event_type")
    
    # Validate timestamp if present
    if "timestamp" in payload:
        try:
            timestamp = payload["timestamp"]
            if isinstance(timestamp, str):
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif isinstance(timestamp, (int, float)):
                datetime.fromtimestamp(timestamp)
        except (ValueError, TypeError):
            raise ValidationError("Invalid timestamp format in webhook payload")
    
    # TODO: Implement signature verification if needed
    if signature:
        logger.warning("Signature verification not implemented")
    
    return True


def sanitize_user_input(input_data: str) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        input_data: Raw user input
        
    Returns:
        Sanitized input string
        
    Raises:
        ValidationError: If input contains dangerous content
    """
    if not isinstance(input_data, str):
        raise ValidationError("Input must be a string")
    
    # Remove potentially dangerous characters
    sanitized = input_data.strip()
    
    # Check for SQL injection patterns
    sql_patterns = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)",
        r"(\b(UNION|OR|AND)\b)",
        r"(--|/\*|\*/)",
        r"(\b(EXEC|EXECUTE)\b)"
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            raise ValidationError("Input contains potentially dangerous SQL patterns")
    
    # Check for script injection patterns
    script_patterns = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>"
    ]
    
    for pattern in script_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            raise ValidationError("Input contains potentially dangerous script patterns")
    
    return sanitized


# =============================================================================
# Data Type Validation
# =============================================================================

def validate_decimal(value: Any, min_value: Optional[Decimal] = None, 
                    max_value: Optional[Decimal] = None) -> Decimal:
    """
    Validate and convert value to Decimal.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Decimal value
        
    Raises:
        ValidationError: If value is invalid
    """
    try:
        if isinstance(value, Decimal):
            decimal_value = value
        else:
            decimal_value = Decimal(str(value))
        
        if min_value is not None and decimal_value < min_value:
            raise ValidationError(f"Value {decimal_value} below minimum {min_value}")
        
        if max_value is not None and decimal_value > max_value:
            raise ValidationError(f"Value {decimal_value} above maximum {max_value}")
        
        return decimal_value
        
    except (InvalidOperation, ValueError, TypeError) as e:
        raise ValidationError(f"Cannot convert to Decimal: {str(e)}")


def validate_positive_number(value: Any, field_name: str = "value") -> float:
    """
    Validate that value is a positive number.
    
    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        
    Returns:
        Positive float value
        
    Raises:
        ValidationError: If value is not a positive number
    """
    try:
        if isinstance(value, str):
            num_value = float(value)
        else:
            num_value = float(value)
        
        if num_value <= 0:
            raise ValidationError(f"{field_name} must be positive, got {num_value}")
        
        return num_value
        
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Invalid {field_name}: {str(e)}")


def validate_percentage(value: Any, field_name: str = "percentage") -> float:
    """
    Validate that value is a valid percentage (0-100).
    
    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        
    Returns:
        Percentage as float (0-100)
        
    Raises:
        ValidationError: If value is not a valid percentage
    """
    try:
        if isinstance(value, str):
            pct_value = float(value)
        else:
            pct_value = float(value)
        
        if pct_value < 0 or pct_value > 100:
            raise ValidationError(f"{field_name} must be between 0 and 100, got {pct_value}")
        
        return pct_value
        
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Invalid {field_name}: {str(e)}")


def validate_timestamp(timestamp: Any) -> datetime:
    """
    Validate and convert timestamp to datetime.
    
    Args:
        timestamp: Timestamp to validate
        
    Returns:
        Datetime object
        
    Raises:
        ValidationError: If timestamp is invalid
    """
    if isinstance(timestamp, datetime):
        return timestamp
    
    try:
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        elif isinstance(timestamp, str):
            # Try common formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
            
            raise ValidationError(f"Cannot parse timestamp: {timestamp}")
        else:
            raise ValidationError(f"Invalid timestamp type: {type(timestamp)}")
            
    except Exception as e:
        raise ValidationError(f"Invalid timestamp: {str(e)}")


# =============================================================================
# Business Rule Validation
# =============================================================================

def validate_trading_rules(signal: Signal, current_positions: List[Position]) -> bool:
    """
    Validate trading rules for a signal.
    
    Args:
        signal: Trading signal to validate
        current_positions: List of current positions
        
    Returns:
        True if signal passes trading rules
        
    Raises:
        ValidationError: If signal violates trading rules
    """
    # Validate signal confidence
    if signal.confidence < 0 or signal.confidence > 1:
        raise ValidationError(f"Invalid signal confidence: {signal.confidence}")
    
    # Check for conflicting positions
    for position in current_positions:
        if position.symbol == signal.symbol:
            if position.side != signal.direction:
                # Check if we're trying to reverse position
                if abs(float(position.quantity)) > 0:
                    logger.warning(f"Signal would reverse position for {signal.symbol}")
                    # This might be allowed depending on strategy rules
    
    # Validate signal timestamp
    if signal.timestamp > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise ValidationError("Signal timestamp is in the future")
    
    if signal.timestamp < datetime.now(timezone.utc) - timedelta(hours=1):
        raise ValidationError("Signal timestamp is too old")
    
    return True


def validate_risk_limits(positions: List[Position], risk_config: Dict[str, Any]) -> bool:
    """
    Validate risk limits for current positions.
    
    Args:
        positions: List of current positions
        risk_config: Risk configuration dictionary
        
    Returns:
        True if positions are within risk limits
        
    Raises:
        ValidationError: If risk limits are exceeded
    """
    # Calculate total exposure
    total_exposure = sum(abs(float(pos.quantity) * float(pos.current_price)) 
                        for pos in positions)
    
    # Check maximum portfolio exposure
    max_exposure = risk_config.get("max_portfolio_exposure", 0.95)
    if total_exposure > max_exposure:
        raise ValidationError(f"Total exposure {total_exposure} exceeds limit {max_exposure}")
    
    # Check maximum positions per symbol
    symbol_counts = {}
    for pos in positions:
        symbol_counts[pos.symbol] = symbol_counts.get(pos.symbol, 0) + 1
    
    max_positions_per_symbol = risk_config.get("max_positions_per_symbol", 1)
    for symbol, count in symbol_counts.items():
        if count > max_positions_per_symbol:
            raise ValidationError(f"Too many positions for {symbol}: {count}")
    
    # Check maximum total positions
    max_total_positions = risk_config.get("max_total_positions", 10)
    if len(positions) > max_total_positions:
        raise ValidationError(f"Too many total positions: {len(positions)}")
    
    return True


def validate_position_limits(position: Position, risk_config: Dict[str, Any]) -> bool:
    """
    Validate position size limits.
    
    Args:
        position: Position to validate
        risk_config: Risk configuration dictionary
        
    Returns:
        True if position is within limits
        
    Raises:
        ValidationError: If position exceeds limits
    """
    position_value = abs(float(position.quantity) * float(position.current_price))
    
    # Check maximum position size
    max_position_size = risk_config.get("max_position_size", 0.1)
    if position_value > max_position_size:
        raise ValidationError(f"Position value {position_value} exceeds limit {max_position_size}")
    
    # Check minimum position size
    min_position_size = risk_config.get("min_position_size", 0.001)
    if position_value < min_position_size:
        raise ValidationError(f"Position value {position_value} below minimum {min_position_size}")
    
    return True


# =============================================================================
# Exchange Data Validation
# =============================================================================

def validate_order_response(response: OrderResponse) -> bool:
    """
    Validate order response from exchange.
    
    Args:
        response: OrderResponse object to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If order response is invalid
    """
    try:
        # Validate required fields
        if not response.id:
            raise ValidationError("Order ID is required")
        
        if not response.symbol:
            raise ValidationError("Symbol is required")
        
        # Validate quantity
        if response.quantity <= 0:
            raise ValidationError("Quantity must be positive")
        
        # Validate filled quantity
        if response.filled_quantity < 0:
            raise ValidationError("Filled quantity cannot be negative")
        
        if response.filled_quantity > response.quantity:
            raise ValidationError("Filled quantity cannot exceed total quantity")
        
        # Validate price for limit orders
        if response.order_type == OrderType.LIMIT and response.price:
            validate_price(float(response.price), response.symbol)
        
        # Validate status
        valid_statuses = ["pending", "filled", "cancelled", "rejected", "partial"]
        if response.status not in valid_statuses:
            raise ValidationError(f"Invalid order status: {response.status}")
        
        return True
        
    except Exception as e:
        raise ValidationError(f"Order response validation failed: {str(e)}")


def validate_balance_data(balances: Dict[str, float]) -> bool:
    """
    Validate balance data from exchange.
    
    Args:
        balances: Balance dictionary to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If balance data is invalid
    """
    if not isinstance(balances, dict):
        raise ValidationError("Balances must be a dictionary")
    
    for currency, balance in balances.items():
        if not isinstance(currency, str):
            raise ValidationError("Currency must be a string")
        
        if not isinstance(balance, (int, float, Decimal)):
            raise ValidationError(f"Balance for {currency} must be a number")
        
        if balance < 0:
            raise ValidationError(f"Balance for {currency} cannot be negative")
    
    return True


def validate_trade_data(trade: Dict[str, Any]) -> bool:
    """
    Validate trade data from exchange.
    
    Args:
        trade: Trade dictionary to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If trade data is invalid
    """
    required_fields = ["id", "symbol", "side", "quantity", "price", "timestamp"]
    
    for field in required_fields:
        if field not in trade:
            raise ValidationError(f"Required field missing: {field}")
    
    # Validate symbol
    validate_symbol(trade["symbol"])
    
    # Validate side
    valid_sides = ["buy", "sell"]
    if trade["side"] not in valid_sides:
        raise ValidationError(f"Invalid trade side: {trade['side']}")
    
    # Validate quantity and price
    validate_quantity(trade["quantity"], trade["symbol"])
    validate_price(trade["price"], trade["symbol"])
    
    # Validate timestamp
    validate_timestamp(trade["timestamp"])
    
    # Validate fee if present
    if "fee" in trade:
        if trade["fee"] < 0:
            raise ValidationError("Trade fee cannot be negative")
    
    return True


def validate_exchange_info(exchange_info: Dict[str, Any]) -> bool:
    """
    Validate exchange information.
    
    Args:
        exchange_info: Exchange info dictionary to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If exchange info is invalid
    """
    required_fields = ["name", "supported_symbols", "rate_limits"]
    
    for field in required_fields:
        if field not in exchange_info:
            raise ValidationError(f"Required field missing: {field}")
    
    # Validate exchange name
    if not isinstance(exchange_info["name"], str):
        raise ValidationError("Exchange name must be a string")
    
    # Validate supported symbols
    if not isinstance(exchange_info["supported_symbols"], list):
        raise ValidationError("Supported symbols must be a list")
    
    for symbol in exchange_info["supported_symbols"]:
        validate_symbol(symbol)
    
    # Validate rate limits
    if not isinstance(exchange_info["rate_limits"], dict):
        raise ValidationError("Rate limits must be a dictionary")
    
    return True 