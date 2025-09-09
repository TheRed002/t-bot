"""
Consolidated Market Data Validation Utilities

This module provides a single source of truth for all market data validation logic,
eliminating duplication across the data module validators.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from src.core.exceptions import ValidationError
from src.core.types import MarketData
from src.utils.validation.core import ValidationFramework
from src.utils.validation.validation_types import (
    ValidationCategory,
    ValidationIssue,
    ValidationLevel,
)


class MarketDataValidationUtils:
    """Centralized market data validation utilities."""

    @staticmethod
    def validate_symbol_format(symbol: str) -> bool:
        """
        Validate trading symbol format.

        Args:
            symbol: Trading symbol to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If symbol format is invalid
        """
        return ValidationFramework.validate_symbol(symbol) is not None

    @staticmethod
    def validate_price_value(
        price: int | float | str | Decimal,
        field_name: str = "price",
        min_price: Decimal = Decimal("0"),
        max_price: Decimal = Decimal("1000000"),
    ) -> Decimal:
        """
        Validate and normalize price value.

        Args:
            price: Price value to validate
            field_name: Name of the field for error messages
            min_price: Minimum allowed price
            max_price: Maximum allowed price

        Returns:
            Validated and normalized price as Decimal

        Raises:
            ValidationError: If price is invalid
        """
        if price is None:
            raise ValidationError(f"{field_name} cannot be None")

        try:
            price_decimal = Decimal(str(price))
        except (TypeError, ValueError, InvalidOperation) as e:
            raise ValidationError(f"Invalid {field_name} type: {type(price)}") from e

        if price_decimal < min_price:
            raise ValidationError(f"{field_name} {price_decimal} below minimum {min_price}")

        if price_decimal > max_price:
            raise ValidationError(f"{field_name} {price_decimal} above maximum {max_price}")

        # Round to 8 decimals for crypto precision
        return price_decimal.quantize(Decimal("0.00000001"))

    @staticmethod
    def validate_volume_value(
        volume: int | float | str | Decimal,
        field_name: str = "volume",
        min_volume: Decimal = Decimal("0"),
    ) -> Decimal:
        """
        Validate and normalize volume value.

        Args:
            volume: Volume value to validate
            field_name: Name of the field for error messages
            min_volume: Minimum allowed volume

        Returns:
            Validated and normalized volume as Decimal

        Raises:
            ValidationError: If volume is invalid
        """
        if volume is None:
            raise ValidationError(f"{field_name} cannot be None")

        try:
            volume_decimal = Decimal(str(volume))
        except (TypeError, ValueError, InvalidOperation) as e:
            raise ValidationError(f"Invalid {field_name} type: {type(volume)}") from e

        if volume_decimal < min_volume:
            raise ValidationError(f"{field_name} {volume_decimal} below minimum {min_volume}")

        # Round to 8 decimals for crypto precision
        return volume_decimal.quantize(Decimal("0.00000001"))

    @staticmethod
    def validate_timestamp_value(
        timestamp: datetime,
        field_name: str = "timestamp",
        max_future_seconds: int = 300,  # 5 minutes
        max_age_seconds: int = 3600,  # 1 hour
    ) -> datetime:
        """
        Validate timestamp value.

        Args:
            timestamp: Timestamp to validate
            field_name: Name of the field for error messages
            max_future_seconds: Maximum seconds in the future allowed
            max_age_seconds: Maximum age in seconds allowed

        Returns:
            Validated timestamp

        Raises:
            ValidationError: If timestamp is invalid
        """
        if timestamp is None:
            raise ValidationError(f"{field_name} cannot be None")

        if not isinstance(timestamp, datetime):
            raise ValidationError(f"{field_name} must be a datetime object")

        current_time = datetime.now(timezone.utc)

        # Check if timestamp is too far in the future
        max_future_time = current_time + timedelta(seconds=max_future_seconds)
        if timestamp > max_future_time:
            raise ValidationError(f"{field_name} is too far in the future")

        # Check if timestamp is too old
        min_time = current_time - timedelta(seconds=max_age_seconds)
        if timestamp < min_time:
            raise ValidationError(f"{field_name} is too old")

        return timestamp

    @staticmethod
    def validate_decimal_precision(
        value: int | float | str | Decimal, field_name: str, max_decimal_places: int = 8
    ) -> bool:
        """
        Validate decimal precision for financial data.

        Args:
            value: Value to check precision
            field_name: Name of the field for error messages
            max_decimal_places: Maximum allowed decimal places

        Returns:
            True if precision is valid

        Raises:
            ValidationError: If precision is invalid
        """
        try:
            decimal_value = Decimal(str(value))
            _, digits, exponent = decimal_value.as_tuple()

            # Count decimal places
            decimal_places = -int(exponent) if isinstance(exponent, int) and exponent < 0 else 0

            if decimal_places > max_decimal_places:
                raise ValidationError(
                    f"Too many decimal places in {field_name} ({decimal_places} > {max_decimal_places})"
                )

            return True

        except (ValueError, TypeError, InvalidOperation) as e:
            raise ValidationError(f"Invalid decimal value for {field_name}") from e

    @staticmethod
    def validate_bid_ask_spread(bid: Decimal, ask: Decimal) -> bool:
        """
        Validate bid-ask spread consistency.

        Args:
            bid: Bid price
            ask: Ask price

        Returns:
            True if spread is valid

        Raises:
            ValidationError: If spread is invalid
        """
        if bid is None or ask is None:
            return True  # Allow None values, checked elsewhere

        if bid >= ask:
            raise ValidationError(f"Bid price {bid} must be less than ask price {ask}")

        # Check for extremely wide spreads (>10%)
        spread_pct = ((ask - bid) / ask) * 100
        if spread_pct > 10.0:
            raise ValidationError(f"Abnormally wide bid/ask spread: {spread_pct:.2f}%")

        return True

    @staticmethod
    def validate_price_consistency(data: MarketData) -> bool:
        """
        Validate price consistency across OHLC values.

        Args:
            data: MarketData object to validate

        Returns:
            True if prices are consistent

        Raises:
            ValidationError: If prices are inconsistent
        """
        # High >= Low check
        if data.high and data.low and data.high < data.low:
            raise ValidationError(f"High price {data.high} less than low price {data.low}")

        # Close within High/Low range
        if data.close and data.high and data.close > data.high:
            raise ValidationError(f"Close price {data.close} higher than high price {data.high}")

        if data.close and data.low and data.close < data.low:
            raise ValidationError(f"Close price {data.close} lower than low price {data.low}")

        # Open within High/Low range (if available)
        if data.open and data.high and data.open > data.high:
            raise ValidationError(f"Open price {data.open} higher than high price {data.high}")

        if data.open and data.low and data.open < data.low:
            raise ValidationError(f"Open price {data.open} lower than low price {data.low}")

        return True

    @staticmethod
    def create_validation_issue(
        field: str,
        value: Any,
        expected: Any,
        message: str,
        level: ValidationLevel = ValidationLevel.HIGH,
        category: ValidationCategory = ValidationCategory.SCHEMA,
        source: str = "MarketDataValidator",
        metadata: dict[str, Any] | None = None,
    ) -> ValidationIssue:
        """
        Create a standardized validation issue.

        Args:
            field: Field name that failed validation
            value: Actual value that failed
            expected: Expected value or format
            message: Error message
            level: Validation severity level
            category: Validation category
            source: Source of the validation
            metadata: Additional metadata

        Returns:
            ValidationIssue object
        """
        return ValidationIssue(
            field=field,
            value=value,
            expected=expected,
            message=message,
            level=level,
            source=source,
            metadata=metadata or {},
            category=category,
        )


class MarketDataValidator:
    """
    Consolidated market data validator that replaces multiple duplicate implementations.

    This class provides a single, comprehensive validation interface for market data
    that can be used across all data services.
    """

    def __init__(
        self,
        enable_precision_validation: bool = True,
        enable_consistency_validation: bool = True,
        enable_timestamp_validation: bool = True,
        max_decimal_places: int = 8,
        max_future_seconds: int = 300,
        max_age_seconds: int = 3600,
    ):
        """
        Initialize market data validator.

        Args:
            enable_precision_validation: Enable decimal precision checks
            enable_consistency_validation: Enable price consistency checks
            enable_timestamp_validation: Enable timestamp validation
            max_decimal_places: Maximum decimal places allowed
            max_future_seconds: Maximum seconds in future allowed
            max_age_seconds: Maximum age in seconds allowed
        """
        self.enable_precision_validation = enable_precision_validation
        self.enable_consistency_validation = enable_consistency_validation
        self.enable_timestamp_validation = enable_timestamp_validation
        self.max_decimal_places = max_decimal_places
        self.max_future_seconds = max_future_seconds
        self.max_age_seconds = max_age_seconds
        self._validation_errors: list[str] = []

    def validate_market_data_record(self, data: MarketData) -> bool:
        """
        Validate a single market data record.

        Args:
            data: MarketData object to validate

        Returns:
            True if valid, False otherwise
        """
        self._validation_errors.clear()

        try:
            # Basic schema validation
            self._validate_required_fields(data)

            # Price validations
            if data.close:
                MarketDataValidationUtils.validate_price_value(data.close, "close")

            if data.open:
                MarketDataValidationUtils.validate_price_value(data.open, "open")

            if data.high:
                MarketDataValidationUtils.validate_price_value(data.high, "high")

            if data.low:
                MarketDataValidationUtils.validate_price_value(data.low, "low")

            if data.bid_price:
                MarketDataValidationUtils.validate_price_value(data.bid_price, "bid_price")

            if data.ask_price:
                MarketDataValidationUtils.validate_price_value(data.ask_price, "ask_price")

            # Volume validation
            if data.volume:
                MarketDataValidationUtils.validate_volume_value(data.volume, "volume")

            # Symbol validation
            if data.symbol:
                MarketDataValidationUtils.validate_symbol_format(data.symbol)

            # Timestamp validation
            if self.enable_timestamp_validation and data.timestamp:
                MarketDataValidationUtils.validate_timestamp_value(
                    data.timestamp,
                    max_future_seconds=self.max_future_seconds,
                    max_age_seconds=self.max_age_seconds,
                )

            # Decimal precision validation
            if self.enable_precision_validation:
                for field_name in [
                    "close",
                    "open",
                    "high",
                    "low",
                    "bid_price",
                    "ask_price",
                    "volume",
                ]:
                    value = getattr(data, field_name, None)
                    if value:
                        MarketDataValidationUtils.validate_decimal_precision(
                            value, field_name, self.max_decimal_places
                        )

            # Price consistency validation
            if self.enable_consistency_validation:
                MarketDataValidationUtils.validate_price_consistency(data)

                # Bid-ask spread validation
                if data.bid_price and data.ask_price:
                    MarketDataValidationUtils.validate_bid_ask_spread(
                        data.bid_price, data.ask_price
                    )

            return True

        except ValidationError as e:
            self._validation_errors.append(str(e))
            return False
        except Exception as e:
            self._validation_errors.append(f"Validation error: {e}")
            return False

    def validate_market_data_batch(self, data_list: list[MarketData]) -> list[MarketData]:
        """
        Validate a batch of market data records.

        Args:
            data_list: List of MarketData objects to validate

        Returns:
            List of valid MarketData objects
        """
        valid_data = []

        for i, data in enumerate(data_list):
            if self.validate_market_data_record(data):
                valid_data.append(data)
            else:
                # Add index to error messages
                indexed_errors = [f"Record {i}: {error}" for error in self._validation_errors]
                self._validation_errors.extend(indexed_errors)

        return valid_data

    def _validate_required_fields(self, data: MarketData) -> None:
        """Validate required fields are present."""
        required_fields = ["symbol", "timestamp"]

        for field in required_fields:
            if not getattr(data, field, None):
                raise ValidationError(f"Required field '{field}' is missing or None")

        # At least one price field should be present
        price_fields = ["close", "open", "high", "low", "bid_price", "ask_price"]
        if not any(getattr(data, field, None) for field in price_fields):
            raise ValidationError("At least one price field is required")

    def get_validation_errors(self) -> list[str]:
        """Get validation errors from last validation."""
        return self._validation_errors.copy()

    def reset(self) -> None:
        """Reset validator state."""
        self._validation_errors.clear()
