"""Centralized data validation pipeline."""

from typing import Any

from src.base import BaseComponent
from src.data.interfaces import DataValidatorInterface


class DataValidationPipeline(BaseComponent):
    """Centralized validation pipeline for data."""

    def __init__(self):
        """Initialize validation pipeline."""
        super().__init__()  # Initialize BaseComponent
        self.validators: list[DataValidatorInterface] = []

    def add_validator(self, validator: DataValidatorInterface) -> "DataValidationPipeline":
        """
        Add a validator to the pipeline.

        Args:
            validator: Validator to add

        Returns:
            Self for chaining
        """
        self.validators.append(validator)
        self._logger.debug(f"Added validator: {validator.__class__.__name__}")
        return self

    def remove_validator(self, validator_type: type) -> bool:
        """
        Remove validators of a specific type.

        Args:
            validator_type: Type of validator to remove

        Returns:
            True if any were removed
        """
        initial_count = len(self.validators)
        self.validators = [v for v in self.validators if not isinstance(v, validator_type)]
        removed = initial_count - len(self.validators)
        if removed:
            self._logger.debug(f"Removed {removed} validators of type {validator_type.__name__}")
        return removed > 0

    def validate(self, data: Any) -> tuple[bool, list[str]]:
        """
        Run all validators and collect errors.

        Args:
            data: Data to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        for validator in self.validators:
            try:
                # Reset validator state
                validator.reset()

                # Run validation
                if not validator.validate(data):
                    validator_errors = validator.get_errors()
                    if validator_errors:
                        errors.extend(validator_errors)
                    else:
                        errors.append(f"{validator.__class__.__name__} validation failed")

            except Exception as e:
                error_msg = f"{validator.__class__.__name__} raised exception: {e!s}"
                self._logger.error(error_msg)
                errors.append(error_msg)

        is_valid = len(errors) == 0

        if not is_valid:
            self._logger.warning(f"Validation failed with {len(errors)} errors")

        return is_valid, errors

    def validate_batch(self, data_list: list[Any]) -> list[tuple[bool, list[str]]]:
        """
        Validate a batch of data items.

        Args:
            data_list: List of data items to validate

        Returns:
            List of (is_valid, errors) tuples
        """
        results = []
        for data in data_list:
            results.append(self.validate(data))
        return results

    def clear(self) -> None:
        """Clear all validators from the pipeline."""
        self.validators.clear()
        self._logger.debug("Cleared all validators")

    def get_validator_count(self) -> int:
        """Get the number of validators in the pipeline."""
        return len(self.validators)
