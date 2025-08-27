"""Centralized validation framework for the T-Bot trading system."""

import re
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any

from src.core.exceptions import ValidationError


class ValidationFramework:
    """Centralized validation framework to eliminate duplication."""

    @staticmethod
    def validate_order(order: dict[str, Any]) -> bool:
        """
        Single source of truth for order validation.

        Args:
            order: Order dictionary to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        validators: list[tuple[str, Callable[[Any], bool], str]] = [
            ("price", lambda x: x > 0, "Price must be positive"),
            ("quantity", lambda x: x > 0, "Quantity must be positive"),
            (
                "symbol",
                lambda x: bool(x) and isinstance(x, str),
                "Symbol required and must be string",
            ),
            ("side", lambda x: x in ["BUY", "SELL"], "Side must be BUY or SELL"),
            (
                "type",
                lambda x: x in ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"],
                "Invalid order type",
            ),
        ]

        # Check for missing required fields first (except price for MARKET orders)
        required_fields = ["symbol", "side", "type"]
        for field in required_fields:
            if field not in order:
                if field == "type":
                    raise ValidationError("type: Invalid order type")
                else:
                    raise ValidationError(f"{field} is required")

        # Check if quantity is provided for all order types
        if "quantity" not in order:
            raise ValidationError("quantity is required")

        # Check if price is required based on order type
        if order.get("type") in ["LIMIT", "STOP_LIMIT"] and "price" not in order:
            raise ValidationError("price is required for LIMIT orders")

        # Validate provided fields
        for field, validator_func, error_msg in validators:
            if field in order:
                if not validator_func(order[field]):
                    raise ValidationError(f"{field}: {error_msg}")

        return True

    @staticmethod
    def _validate_common_params(params: dict[str, Any]) -> None:
        """Validate common strategy parameters."""
        if "timeframe" in params:
            valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            if params["timeframe"] not in valid_timeframes:
                raise ValidationError(f"Invalid timeframe. Must be one of {valid_timeframes}")

    @staticmethod
    def _validate_mean_reversion_params(params: dict[str, Any]) -> None:
        """Validate mean reversion strategy parameters."""
        required = ["window_size", "num_std", "entry_threshold"]
        for field in required:
            if field not in params:
                raise ValidationError(f"{field} is required for MEAN_REVERSION strategy")

        if params["window_size"] < 2:
            raise ValidationError("window_size must be at least 2")
        if params["num_std"] <= 0:
            raise ValidationError("num_std must be positive")

    @staticmethod
    def _validate_momentum_params(params: dict[str, Any]) -> None:
        """Validate momentum strategy parameters."""
        required = ["lookback_period", "momentum_threshold"]
        for field in required:
            if field not in params:
                raise ValidationError(f"{field} is required for MOMENTUM strategy")

        if params["lookback_period"] < 1:
            raise ValidationError("lookback_period must be at least 1")

    @staticmethod
    def _validate_market_making_params(params: dict[str, Any]) -> None:
        """Validate market making strategy parameters."""
        if "bid_spread" in params and params["bid_spread"] < 0:
            raise ValidationError("bid_spread must be non-negative")
        if "ask_spread" in params and params["ask_spread"] < 0:
            raise ValidationError("ask_spread must be non-negative")
        if "order_size" in params and params["order_size"] <= 0:
            raise ValidationError("order_size must be positive")

    @staticmethod
    def validate_strategy_params(params: dict[str, Any]) -> bool:
        """
        Single source for strategy parameter validation.

        Args:
            params: Strategy parameters to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if "strategy_type" not in params:
            raise ValidationError("strategy_type is required")

        strategy_type = params["strategy_type"]

        # Common validations
        ValidationFramework._validate_common_params(params)

        # Strategy-specific validations
        if strategy_type == "MEAN_REVERSION":
            ValidationFramework._validate_mean_reversion_params(params)
        elif strategy_type == "MOMENTUM":
            ValidationFramework._validate_momentum_params(params)
        elif strategy_type == "market_making":
            ValidationFramework._validate_market_making_params(params)

        return True

    @staticmethod
    def validate_price(price: Any, max_price: float = 1_000_000) -> float:
        """
        Validate and normalize price.

        Args:
            price: Price to validate
            max_price: Maximum allowed price

        Returns:
            Normalized price value (rounded to 8 decimals)

        Raises:
            ValidationError: If price is invalid
        """
        if price is None:
            raise ValidationError("Price cannot be None")

        try:
            # Use Decimal for precision, then convert to float for comparison
            price_decimal = Decimal(str(price))
            # Check if the Decimal is valid (not NaN or infinite)
            if not price_decimal.is_finite():
                raise ValidationError(f"Price must be a valid number, got {price}")
            price_float = float(price_decimal)
        except (TypeError, ValueError, InvalidOperation) as e:
            raise ValidationError(f"Price must be numeric, got {type(price)}") from e

        if price_float <= 0:
            raise ValidationError("Price must be positive")
        if price_float > max_price:
            raise ValidationError(f"Price {price_float} exceeds maximum {max_price}")
        if price_float == float("inf"):
            raise ValidationError("Price cannot be infinity")

        # Round to 8 decimals for crypto precision
        return round(price_float, 8)

    @staticmethod
    def validate_quantity(quantity: Any, min_qty: float = 0.00000001) -> float:
        """
        Validate and normalize quantity.

        Args:
            quantity: Quantity to validate
            min_qty: Minimum allowed quantity

        Returns:
            Normalized quantity value

        Raises:
            ValidationError: If quantity is invalid
        """
        if quantity is None:
            raise ValidationError("Quantity cannot be None")

        try:
            # Use Decimal for precision, then convert to float for comparison
            qty_decimal = Decimal(str(quantity))
            # Check if the Decimal is valid (not NaN or infinite)
            if not qty_decimal.is_finite():
                raise ValidationError(f"Quantity must be a valid number, got {quantity}")
            qty_float = float(qty_decimal)
        except (TypeError, ValueError, InvalidOperation) as e:
            raise ValidationError(f"Quantity must be numeric, got {type(quantity)}") from e

        if qty_float <= 0:
            raise ValidationError("Quantity must be positive")
        if qty_float < min_qty:
            raise ValidationError(f"Quantity {qty_float} below minimum {min_qty}")
        if qty_float == float("inf"):
            raise ValidationError("Quantity cannot be infinity")

        return qty_float

    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """
        Validate and normalize trading symbol.

        Args:
            symbol: Trading symbol to validate

        Returns:
            Normalized symbol string

        Raises:
            ValidationError: If symbol is invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol must be a non-empty string")

        # Normalize to uppercase
        symbol_norm = symbol.upper().strip()

        # Check format (e.g., BTC/USDT or BTCUSDT)
        if not re.match(r"^[A-Z]+(/|_|-)?[A-Z]+$", symbol_norm):
            raise ValidationError(f"Invalid symbol format: {symbol}")

        return symbol_norm

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
        required_fields = ["api_key", "api_secret"]

        for field in required_fields:
            if field not in credentials:
                raise ValidationError(f"{field} is required")
            if not credentials[field] or not isinstance(credentials[field], str):
                raise ValidationError(f"{field} must be a non-empty string")

        # Check for test/production mode
        if "testnet" in credentials and not isinstance(credentials["testnet"], bool):
            raise ValidationError("testnet must be a boolean")

        return True

    @staticmethod
    def validate_risk_params(params: dict[str, Any]) -> bool:
        """
        Validate risk management parameters.

        Args:
            params: Risk parameters to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        # Check risk_per_trade specifically
        if "risk_per_trade" in params:
            if params["risk_per_trade"] > 0.1:  # 10% max
                raise ValidationError("Risk per trade must be at most 0.1 (10%)")
            if params["risk_per_trade"] <= 0:
                raise ValidationError("Risk per trade must be positive")

        # Check other risk parameters
        if "stop_loss" in params:
            if params["stop_loss"] <= 0 or params["stop_loss"] >= 1:
                raise ValidationError("Stop loss must be between 0 and 1")

        if "take_profit" in params:
            if params["take_profit"] <= 0:
                raise ValidationError("Take profit must be positive")

        if "max_position_size" in params:
            if params["max_position_size"] <= 0:
                raise ValidationError("Max position size must be positive")

        return True

    @staticmethod
    def validate_risk_parameters(params: dict[str, Any]) -> bool:
        """
        Validate risk management parameters.

        Args:
            params: Risk parameters to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        validators: list[tuple[str, Callable[[Any], bool], str]] = [
            (
                "max_position_size",
                lambda x: 0 < x <= 1,
                "Max position size must be between 0 and 1",
            ),
            (
                "stop_loss_pct",
                lambda x: 0 < x < 0.5,
                "Stop loss percentage must be between 0 and 0.5 (50%)",
            ),
            (
                "take_profit_pct",
                lambda x: 0 < x < 10,
                "Take profit percentage must be between 0 and 10",
            ),
            ("max_drawdown", lambda x: 0 < x < 1, "Max drawdown must be between 0 and 1"),
            (
                "risk_per_trade",
                lambda x: 0 < x <= 0.1,
                "Risk per trade must be between 0 and 0.1 (10%)",
            ),
        ]

        for field, validator_func, error_msg in validators:
            if field in params:
                if not validator_func(params[field]):
                    raise ValidationError(f"{field}: {error_msg}")

        return True

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
            valid_options = list(set(valid_timeframes.values()))
            raise ValidationError(f"Invalid timeframe: {timeframe}. Valid options: {valid_options}")

        return valid_timeframes[timeframe_lower]

    @staticmethod
    def validate_batch(validations: list[tuple[str, Callable[[Any], Any], Any]]) -> dict[str, Any]:
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
