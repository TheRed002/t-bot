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

from typing import Any

from src.core.exceptions import ValidationError
from src.core.types import OrderRequest, OrderSide, OrderType

# Import comprehensive validation functions from core module
# from src.utils.validation.core import ValidationFramework
# TODO: Fix validation imports - functions don't exist as module-level functions


class ValidationFramework:
    """
    Centralized validation framework that provides simplified interfaces
    to the comprehensive validation functions in validation/core.py
    """

    @staticmethod
    def validate_order(order: dict[str, Any]) -> bool:
        """
        Single source of truth for order validation.
        Converts dict to OrderRequest and uses core validation.

        Args:
            order: Order dictionary to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        try:
            # Convert dict to OrderRequest for comprehensive validation
            order_request = OrderRequest(
                symbol=order.get("symbol", ""),
                side=OrderSide(order.get("side", "BUY")),
                order_type=OrderType(order.get("type", "MARKET")),
                quantity=order.get("quantity", 0),
                price=order.get("price"),
                stop_price=order.get("stop_price"),
                time_in_force=order.get("time_in_force", "GTC"),
            )

            # Use comprehensive validation from core
            return validate_order_request(order_request)

        except Exception as e:
            # Convert ValidationError to ValueError for compatibility
            raise ValueError(str(e)) from e

    @staticmethod
    def validate_strategy_params(params: dict[str, Any]) -> bool:
        """
        Single source for strategy parameter validation.
        Uses comprehensive strategy config validation from core.

        Args:
            params: Strategy parameters to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        try:
            # Use comprehensive validation from core
            return validate_strategy_config(params)

        except ValidationError as e:
            # Convert ValidationError to ValueError for compatibility
            raise ValueError(str(e)) from e

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
            ValueError: If price is invalid
        """
        try:
            # Use symbol-aware validation from core (with default symbol)
            validate_price(price, "DEFAULT", "binance")

            # Additional max price check
            price_float = float(price)
            if price_float > max_price:
                raise ValueError(f"Price {price_float} exceeds maximum {max_price}")

            return True

        except ValidationError as e:
            raise ValueError(str(e)) from e

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
            ValueError: If quantity is invalid
        """
        try:
            # Use comprehensive validation from core
            validate_quantity(quantity, "DEFAULT", min_qty)
            return True

        except ValidationError as e:
            raise ValueError(str(e)) from e

    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """
        Validate and normalize trading symbol using core validation.

        Args:
            symbol: Trading symbol to validate

        Returns:
            True if valid

        Raises:
            ValueError: If symbol is invalid
        """
        try:
            # Use comprehensive validation from core
            validate_symbol(symbol)
            return True

        except ValidationError as e:
            raise ValueError(str(e)) from e

    @staticmethod
    def validate_exchange_credentials(credentials: dict[str, Any]) -> bool:
        """
        Validate exchange API credentials.

        Args:
            credentials: Credentials dictionary

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        required_fields = ["api_key", "api_secret"]

        for field in required_fields:
            if field not in credentials:
                raise ValueError(f"{field} is required")
            if not credentials[field] or not isinstance(credentials[field], str):
                raise ValueError(f"{field} must be a non-empty string")

        # Check for test/production mode
        if "testnet" in credentials and not isinstance(credentials["testnet"], bool):
            raise ValueError("testnet must be a boolean")

        return True

    @staticmethod
    def validate_risk_params(params: dict[str, Any]) -> bool:
        """
        Validate risk management parameters using core validation.

        Args:
            params: Risk parameters to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        try:
            # Use comprehensive validation from core
            validate_risk_parameters(params)
            return True

        except ValidationError as e:
            raise ValueError(str(e)) from e

    @staticmethod
    def validate_risk_parameters(params: dict[str, Any]) -> bool:
        """
        Validate risk management parameters (alias for validate_risk_params).

        Args:
            params: Risk parameters to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
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
            ValueError: If timeframe is invalid
        """
        valid_timeframes = {
            "1m": "1m",
            "1min": "1m",
            "1minute": "1m",
            "5m": "5m",
            "5min": "5m",
            "5minutes": "5m",
            "15m": "15m",
            "15min": "15m",
            "15minutes": "15m",
            "30m": "30m",
            "30min": "30m",
            "30minutes": "30m",
            "1h": "1h",
            "1hr": "1h",
            "1hour": "1h",
            "60m": "1h",
            "4h": "4h",
            "4hr": "4h",
            "4hours": "4h",
            "240m": "4h",
            "1d": "1d",
            "1day": "1d",
            "daily": "1d",
            "1w": "1w",
            "1week": "1w",
            "weekly": "1w",
        }

        timeframe_lower = timeframe.lower().strip()

        if timeframe_lower not in valid_timeframes:
            raise ValueError(
                f"Invalid timeframe: {timeframe}. "
                f"Valid options: {list(set(valid_timeframes.values()))}"
            )

        return valid_timeframes[timeframe_lower]

    @staticmethod
    def validate_batch(validations: list[tuple[str, callable, Any]]) -> dict[str, Any]:
        """
        Run multiple validations and collect results.

        Args:
            validations: List of (name, validator_func, data) tuples

        Returns:
            Dictionary with validation results
        """
        results = {}

        for name, validator_func, data in validations:
            try:
                result = validator_func(data)
                results[name] = {"status": "success", "result": result}
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}

        return results

    @staticmethod
    def validate_api_request(
        request_data: dict[str, Any], required_fields: list[str] | None = None
    ) -> bool:
        """
        Validate API request payload using core validation.

        Args:
            request_data: Request data dictionary
            required_fields: List of required field names

        Returns:
            True if valid

        Raises:
            ValueError: If API request is invalid
        """
        try:
            # Use comprehensive validation from core
            validate_api_request(request_data, required_fields)
            return True

        except ValidationError as e:
            raise ValueError(str(e)) from e

    @staticmethod
    def validate_decimal_value(
        value: Any, min_value: float | None = None, max_value: float | None = None
    ) -> bool:
        """
        Validate decimal value using core validation.

        Args:
            value: Value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            True if valid

        Raises:
            ValueError: If value is invalid
        """
        try:
            from decimal import Decimal

            min_dec = Decimal(str(min_value)) if min_value is not None else None
            max_dec = Decimal(str(max_value)) if max_value is not None else None

            # Use comprehensive validation from core
            validate_decimal(value, min_dec, max_dec)
            return True

        except ValidationError as e:
            raise ValueError(str(e)) from e

    @staticmethod
    def validate_positive_value(value: Any, field_name: str = "value") -> bool:
        """
        Validate positive number using core validation.

        Args:
            value: Value to validate
            field_name: Name of the field for error messages

        Returns:
            True if valid

        Raises:
            ValueError: If value is not positive
        """
        try:
            # Use comprehensive validation from core
            validate_positive_number(value, field_name)
            return True

        except ValidationError as e:
            raise ValueError(str(e)) from e

    @staticmethod
    def validate_percentage_value(value: Any, field_name: str = "percentage") -> bool:
        """
        Validate percentage value using core validation.

        Args:
            value: Value to validate (0-100)
            field_name: Name of the field for error messages

        Returns:
            True if valid

        Raises:
            ValueError: If value is not a valid percentage
        """
        try:
            # Use comprehensive validation from core
            validate_percentage(value, field_name)
            return True

        except ValidationError as e:
            raise ValueError(str(e)) from e

    @staticmethod
    def validate_timestamp_value(timestamp: Any) -> bool:
        """
        Validate timestamp using core validation.

        Args:
            timestamp: Timestamp to validate

        Returns:
            True if valid

        Raises:
            ValueError: If timestamp is invalid
        """
        try:
            # Use comprehensive validation from core
            validate_timestamp(timestamp)
            return True

        except ValidationError as e:
            raise ValueError(str(e)) from e


# Re-export validation framework class for backward compatibility
ValidationUtilities = ValidationFramework
