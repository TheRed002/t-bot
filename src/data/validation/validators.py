"""Common data validators."""

from typing import Any

from src.data.interfaces import DataValidatorInterface


class PriceValidator(DataValidatorInterface):
    """Validator for price data."""

    def __init__(self, min_price: float = 0, max_price: float = float("inf")):
        """
        Initialize price validator.

        Args:
            min_price: Minimum valid price
            max_price: Maximum valid price
        """
        self.min_price = min_price
        self.max_price = max_price
        self.errors: list[str] = []

    def validate(self, data: Any) -> bool:
        """Validate price data."""
        self.errors.clear()

        if isinstance(data, dict):
            # Single record
            return self._validate_record(data)
        elif isinstance(data, list):
            # Multiple records
            all_valid = True
            for i, record in enumerate(data):
                if not self._validate_record(record, index=i):
                    all_valid = False
            return all_valid
        else:
            self.errors.append(f"Invalid data type: {type(data)}")
            return False

    def _validate_record(self, record: dict, index: int | None = None) -> bool:
        """Validate a single record."""
        prefix = f"Record {index}: " if index is not None else ""

        if "price" not in record:
            self.errors.append(f"{prefix}Missing 'price' field")
            return False

        price = record["price"]

        try:
            price_float = float(price)
        except (TypeError, ValueError):
            self.errors.append(f"{prefix}Invalid price type: {type(price)}")
            return False

        if price_float < self.min_price:
            self.errors.append(f"{prefix}Price {price_float} below minimum {self.min_price}")
            return False

        if price_float > self.max_price:
            self.errors.append(f"{prefix}Price {price_float} above maximum {self.max_price}")
            return False

        return True

    def get_errors(self) -> list[str]:
        """Get validation errors."""
        return self.errors.copy()

    def reset(self) -> None:
        """Reset validator state."""
        self.errors.clear()


class VolumeValidator(DataValidatorInterface):
    """Validator for volume data."""

    def __init__(self, min_volume: float = 0):
        """
        Initialize volume validator.

        Args:
            min_volume: Minimum valid volume
        """
        self.min_volume = min_volume
        self.errors: list[str] = []

    def validate(self, data: Any) -> bool:
        """Validate volume data."""
        self.errors.clear()

        if isinstance(data, dict):
            return self._validate_record(data)
        elif isinstance(data, list):
            all_valid = True
            for i, record in enumerate(data):
                if not self._validate_record(record, index=i):
                    all_valid = False
            return all_valid
        else:
            self.errors.append(f"Invalid data type: {type(data)}")
            return False

    def _validate_record(self, record: dict, index: int | None = None) -> bool:
        """Validate a single record."""
        prefix = f"Record {index}: " if index is not None else ""

        if "volume" not in record:
            self.errors.append(f"{prefix}Missing 'volume' field")
            return False

        volume = record["volume"]

        try:
            volume_float = float(volume)
        except (TypeError, ValueError):
            self.errors.append(f"{prefix}Invalid volume type: {type(volume)}")
            return False

        if volume_float < self.min_volume:
            self.errors.append(f"{prefix}Volume {volume_float} below minimum {self.min_volume}")
            return False

        return True

    def get_errors(self) -> list[str]:
        """Get validation errors."""
        return self.errors.copy()

    def reset(self) -> None:
        """Reset validator state."""
        self.errors.clear()


class TimestampValidator(DataValidatorInterface):
    """Validator for timestamp data."""

    def __init__(self, require_ordered: bool = False):
        """
        Initialize timestamp validator.

        Args:
            require_ordered: Whether timestamps must be in order
        """
        self.require_ordered = require_ordered
        self.errors: list[str] = []
        self.last_timestamp: int | float | str | None = None

    def validate(self, data: Any) -> bool:
        """Validate timestamp data."""
        self.errors.clear()
        self.last_timestamp = None

        if isinstance(data, dict):
            return self._validate_record(data)
        elif isinstance(data, list):
            all_valid = True
            for i, record in enumerate(data):
                if not self._validate_record(record, index=i):
                    all_valid = False
            return all_valid
        else:
            self.errors.append(f"Invalid data type: {type(data)}")
            return False

    def _validate_record(self, record: dict, index: int | None = None) -> bool:
        """Validate a single record."""
        prefix = f"Record {index}: " if index is not None else ""

        if "timestamp" not in record:
            self.errors.append(f"{prefix}Missing 'timestamp' field")
            return False

        timestamp = record["timestamp"]

        # Validate timestamp format/type
        if not isinstance(timestamp, int | float | str):
            self.errors.append(f"{prefix}Invalid timestamp type: {type(timestamp)}")
            return False

        # Check ordering if required
        if self.require_ordered and self.last_timestamp is not None:
            try:
                current = float(timestamp) if not isinstance(timestamp, str) else 0
                last = float(self.last_timestamp) if not isinstance(self.last_timestamp, str) else 0

                if current < last:
                    self.errors.append(f"{prefix}Timestamp {current} is before previous {last}")
                    return False
            except (TypeError, ValueError):
                # Skip ordering check if conversion fails
                pass

        self.last_timestamp = timestamp
        return True

    def get_errors(self) -> list[str]:
        """Get validation errors."""
        return self.errors.copy()

    def reset(self) -> None:
        """Reset validator state."""
        self.errors.clear()
        self.last_timestamp = None


class SchemaValidator(DataValidatorInterface):
    """Validator for data schema."""

    def __init__(self, required_fields: list[str], optional_fields: list[str] | None = None):
        """
        Initialize schema validator.

        Args:
            required_fields: List of required field names
            optional_fields: List of optional field names
        """
        self.required_fields = set(required_fields)
        self.optional_fields = set(optional_fields or [])
        self.errors: list[str] = []

    def validate(self, data: Any) -> bool:
        """Validate data schema."""
        self.errors.clear()

        if isinstance(data, dict):
            return self._validate_record(data)
        elif isinstance(data, list):
            all_valid = True
            for i, record in enumerate(data):
                if not self._validate_record(record, index=i):
                    all_valid = False
            return all_valid
        else:
            self.errors.append(f"Invalid data type: {type(data)}")
            return False

    def _validate_record(self, record: dict, index: int | None = None) -> bool:
        """Validate a single record."""
        if not isinstance(record, dict):
            self.errors.append(f"Record {index}: Not a dictionary")
            return False

        prefix = f"Record {index}: " if index is not None else ""

        # Check required fields
        missing = self.required_fields - set(record.keys())
        if missing:
            self.errors.append(f"{prefix}Missing required fields: {missing}")
            return False

        # Check for unknown fields (optional)
        all_fields = self.required_fields | self.optional_fields
        unknown = set(record.keys()) - all_fields
        if unknown and len(all_fields) > 0:
            # Only warn about unknown fields if we have a defined schema
            self.errors.append(f"{prefix}Unknown fields: {unknown}")
            # This is a warning, not a failure

        return len(missing) == 0

    def get_errors(self) -> list[str]:
        """Get validation errors."""
        return self.errors.copy()

    def reset(self) -> None:
        """Reset validator state."""
        self.errors.clear()
