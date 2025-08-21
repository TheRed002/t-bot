"""Centralized validation framework for the T-Bot trading system."""

import re
from collections.abc import Callable
from typing import Any


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
            ValueError: If validation fails
        """
        validators = [
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
                    raise ValueError("type: Invalid order type")
                else:
                    raise ValueError(f"{field} is required")

        # Check if quantity is provided for all order types
        if "quantity" not in order:
            raise ValueError("quantity is required")

        # Check if price is required based on order type
        if order.get("type") in ["LIMIT", "STOP_LIMIT"] and "price" not in order:
            raise ValueError("price is required for LIMIT orders")

        # Validate provided fields
        for field, validator_func, error_msg in validators:
            if field in order:
                if not validator_func(order[field]):
                    raise ValueError(f"{field}: {error_msg}")

        return True

    @staticmethod
    def validate_strategy_params(params: dict[str, Any]) -> bool:
        """
        Single source for strategy parameter validation.

        Args:
            params: Strategy parameters to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if "strategy_type" not in params:
            raise ValueError("strategy_type is required")

        strategy_type = params["strategy_type"]

        # Common validations
        if "timeframe" in params:
            valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            if params["timeframe"] not in valid_timeframes:
                raise ValueError(f"Invalid timeframe. Must be one of {valid_timeframes}")

        # Strategy-specific validations
        if strategy_type == "MEAN_REVERSION":
            required = ["window_size", "num_std", "entry_threshold"]
            for field in required:
                if field not in params:
                    raise ValueError(f"{field} is required for MEAN_REVERSION strategy")

            if params["window_size"] < 2:
                raise ValueError("window_size must be at least 2")
            if params["num_std"] <= 0:
                raise ValueError("num_std must be positive")

        elif strategy_type == "MOMENTUM":
            required = ["lookback_period", "momentum_threshold"]
            for field in required:
                if field not in params:
                    raise ValueError(f"{field} is required for MOMENTUM strategy")

            if params["lookback_period"] < 1:
                raise ValueError("lookback_period must be at least 1")

        elif strategy_type == "market_making":
            # Validate market making specific parameters
            if "bid_spread" in params and params["bid_spread"] < 0:
                raise ValueError("bid_spread must be non-negative")
            if "ask_spread" in params and params["ask_spread"] < 0:
                raise ValueError("ask_spread must be non-negative")
            if "order_size" in params and params["order_size"] <= 0:
                raise ValueError("order_size must be positive")

        return True

    @staticmethod
    def validate_price(price: Any, max_price: float = 1_000_000) -> bool:
        """
        Validate and normalize price.

        Args:
            price: Price to validate
            max_price: Maximum allowed price

        Returns:
            True if valid

        Raises:
            ValueError: If price is invalid
        """
        try:
            price_float = float(price)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Price must be numeric, got {type(price)}") from e

        if price_float <= 0:
            raise ValueError("Price must be positive")
        if price_float > max_price:
            raise ValueError(f"Price {price_float} exceeds maximum {max_price}")
        if price_float == float("inf"):
            raise ValueError("Price cannot be infinity")

        return True

    @staticmethod
    def validate_quantity(quantity: Any, min_qty: float = 0.00000001) -> bool:
        """
        Validate and normalize quantity.

        Args:
            quantity: Quantity to validate
            min_qty: Minimum allowed quantity

        Returns:
            True if valid

        Raises:
            ValueError: If quantity is invalid
        """
        try:
            qty_float = float(quantity)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Quantity must be numeric, got {type(quantity)}") from e

        if qty_float <= 0:
            raise ValueError("Quantity must be positive")
        if qty_float < min_qty:
            raise ValueError(f"Quantity {qty_float} below minimum {min_qty}")
        if qty_float == float("inf"):
            raise ValueError("Quantity cannot be infinity")

        return True

    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """
        Validate and normalize trading symbol.

        Args:
            symbol: Trading symbol to validate

        Returns:
            True if valid

        Raises:
            ValueError: If symbol is invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")

        # Normalize to uppercase
        symbol_norm = symbol.upper().strip()

        # Check format (e.g., BTC/USDT or BTCUSDT)
        if not re.match(r"^[A-Z]+(/|_|-)?[A-Z]+$", symbol_norm):
            raise ValueError(f"Invalid symbol format: {symbol}")

        return True

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
        Validate risk management parameters.

        Args:
            params: Risk parameters to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        # Check risk_per_trade specifically
        if "risk_per_trade" in params:
            if params["risk_per_trade"] > 0.1:  # 10% max
                raise ValueError("Risk per trade must be at most 0.1 (10%)")
            if params["risk_per_trade"] <= 0:
                raise ValueError("Risk per trade must be positive")

        # Check other risk parameters
        if "stop_loss" in params:
            if params["stop_loss"] <= 0 or params["stop_loss"] >= 1:
                raise ValueError("Stop loss must be between 0 and 1")

        if "take_profit" in params:
            if params["take_profit"] <= 0:
                raise ValueError("Take profit must be positive")

        if "max_position_size" in params:
            if params["max_position_size"] <= 0:
                raise ValueError("Max position size must be positive")

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
            ValueError: If validation fails
        """
        validators = [
            (
                "max_position_size",
                lambda x: 0 < x <= 1,
                "Max position size must be between 0 and 1",
            ),
            ("stop_loss_pct", lambda x: 0 < x < 1, "Stop loss percentage must be between 0 and 1"),
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
                    raise ValueError(f"{field}: {error_msg}")

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
            valid_options = list(set(valid_timeframes.values()))
            raise ValueError(f"Invalid timeframe: {timeframe}. Valid options: {valid_options}")

        return valid_timeframes[timeframe_lower]

    @staticmethod
    def validate_batch(validations: list[tuple[str, Callable, Any]]) -> dict[str, Any]:
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
