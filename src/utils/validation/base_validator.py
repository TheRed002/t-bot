"""Base validator class with common validation patterns."""

from abc import ABC, abstractmethod
from typing import Any


class BaseRecordValidator(ABC):
    """Base class for validators that process both single records and batches."""

    def __init__(self) -> None:
        """Initialize base validator."""
        self.errors: list[str] = []

    def validate(self, data: Any) -> bool:
        """
        Validate data - handles both single records and batches.

        Args:
            data: Data to validate (dict or list of dicts)

        Returns:
            True if all data is valid, False otherwise
        """
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

    @abstractmethod
    def _validate_record(self, record: dict, index: int | None = None) -> bool:
        """
        Validate a single record. Must be implemented by subclasses.

        Args:
            record: Record to validate
            index: Optional index for batch processing

        Returns:
            True if record is valid
        """
        pass

    def get_errors(self) -> list[str]:
        """Get validation errors."""
        return self.errors.copy()

    def reset(self) -> None:
        """Reset validator state."""
        self.errors.clear()

    async def health_check(self) -> dict[str, Any]:
        """Perform validator health check."""
        return {
            "status": "healthy",
            "error_count": len(self.errors),
        }

    def _add_error(self, message: str, index: int | None = None) -> None:
        """Add an error with optional index prefix."""
        prefix = f"Record {index}: " if index is not None else ""
        self.errors.append(f"{prefix}{message}")
