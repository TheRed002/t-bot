"""
Common data validators refactored to use consolidated validation utilities.

These validators now serve as thin wrappers around the consolidated
MarketDataValidationUtils to eliminate code duplication.
"""

from decimal import Decimal
from typing import Any

from src.core.exceptions import ValidationError
from src.utils.validation.base_validator import BaseRecordValidator
from src.utils.validation.market_data_validation import MarketDataValidationUtils


class PriceValidator(BaseRecordValidator):
    """Validator for price data using consolidated utilities."""

    def __init__(self, min_price: Decimal = Decimal("0"), max_price: Decimal = Decimal("1000000")):
        """
        Initialize price validator.

        Args:
            min_price: Minimum valid price
            max_price: Maximum valid price
        """
        super().__init__()
        self.min_price = min_price
        self.max_price = max_price

    def _validate_record(self, record: dict, index: int | None = None) -> bool:
        """Validate a single record using consolidated utilities."""
        if "price" not in record:
            self._add_error("Missing 'price' field", index)
            return False

        try:
            # Use consolidated validation utilities
            MarketDataValidationUtils.validate_price_value(
                record["price"], "price", self.min_price, self.max_price
            )
            return True
        except ValidationError as e:
            self._add_error(str(e), index)
            return False


class VolumeValidator(BaseRecordValidator):
    """Validator for volume data using consolidated utilities."""

    def __init__(self, min_volume: Decimal = Decimal("0")):
        """
        Initialize volume validator.

        Args:
            min_volume: Minimum valid volume
        """
        super().__init__()
        self.min_volume = min_volume

    def _validate_record(self, record: dict, index: int | None = None) -> bool:
        """Validate a single record using consolidated utilities."""
        if "volume" not in record:
            self._add_error("Missing 'volume' field", index)
            return False

        try:
            # Use consolidated validation utilities
            MarketDataValidationUtils.validate_volume_value(
                record["volume"], "volume", self.min_volume
            )
            return True
        except ValidationError as e:
            self._add_error(str(e), index)
            return False


class TimestampValidator(BaseRecordValidator):
    """Validator for timestamp data using consolidated utilities."""

    def __init__(
        self,
        require_ordered: bool = False,
        max_future_seconds: int = 300,
        max_age_seconds: int = 3600,
    ):
        """
        Initialize timestamp validator.

        Args:
            require_ordered: Whether timestamps must be in order
            max_future_seconds: Maximum seconds in future allowed
            max_age_seconds: Maximum age in seconds allowed
        """
        super().__init__()
        self.require_ordered = require_ordered
        self.max_future_seconds = max_future_seconds
        self.max_age_seconds = max_age_seconds
        self.last_timestamp: int | float | str | None = None

    def validate(self, data: Any) -> bool:
        """Override to reset timestamp tracking."""
        self.last_timestamp = None
        return super().validate(data)

    def _validate_record(self, record: dict, index: int | None = None) -> bool:
        """Validate a single record using consolidated utilities."""
        if "timestamp" not in record:
            self._add_error("Missing 'timestamp' field", index)
            return False

        timestamp = record["timestamp"]

        # Basic type validation
        if not isinstance(timestamp, (int, float, str)):
            self._add_error(f"Invalid timestamp type: {type(timestamp)}", index)
            return False

        # Convert to datetime if needed for consolidated validation
        try:
            from datetime import datetime, timezone

            if isinstance(timestamp, (int, float)):
                timestamp_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            else:
                timestamp_dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))

            # Use consolidated validation utilities
            MarketDataValidationUtils.validate_timestamp_value(
                timestamp_dt, "timestamp", self.max_future_seconds, self.max_age_seconds
            )

        except (ValidationError, ValueError, OSError) as e:
            if isinstance(e, ValidationError):
                self._add_error(str(e), index)
            else:
                self._add_error(f"Invalid timestamp format: {timestamp}", index)
            return False

        # Check ordering if required
        if self.require_ordered and self.last_timestamp is not None:
            try:
                current = float(timestamp) if not isinstance(timestamp, str) else 0
                last = float(self.last_timestamp) if not isinstance(self.last_timestamp, str) else 0

                if current < last:
                    self._add_error(f"Timestamp {current} is before previous {last}", index)
                    return False
            except (TypeError, ValueError):
                # Skip ordering check if conversion fails
                pass

        self.last_timestamp = timestamp
        return True

    def reset(self) -> None:
        """Reset validator state including timestamp tracking."""
        super().reset()
        self.last_timestamp = None


class SchemaValidator(BaseRecordValidator):
    """Validator for data schema."""

    def __init__(self, required_fields: list[str] | None = None, optional_fields: list[str] | None = None):
        """
        Initialize schema validator.

        Args:
            required_fields: List of required field names
            optional_fields: List of optional field names
        """
        super().__init__()
        self.required_fields = set(required_fields)
        self.optional_fields = set(optional_fields) if optional_fields is not None else set()

    def _validate_record(self, record: dict, index: int | None = None) -> bool:
        """Validate a single record."""
        if not isinstance(record, dict):
            self._add_error("Not a dictionary", index)
            return False

        # Check required fields
        missing = self.required_fields - set(record.keys())
        if missing:
            self._add_error(f"Missing required fields: {missing}", index)
            return False

        # Check for unknown fields (optional)
        all_fields = self.required_fields.union(self.optional_fields)
        unknown = set(record.keys()) - all_fields
        if unknown and len(all_fields) > 0:
            # Only warn about unknown fields if we have a defined schema
            self._add_error(f"Unknown fields: {unknown}", index)
            # This is a warning, not a failure

        return len(missing) == 0
