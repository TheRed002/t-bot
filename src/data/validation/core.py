"""Centralized data validation pipeline."""

from typing import Any

from src.core import BaseComponent
from src.data.interfaces import DataValidatorInterface


class DataValidationPipeline(BaseComponent):
    """Centralized validation pipeline for data."""

    def __init__(self) -> None:
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
        self.logger.debug(f"Added validator: {validator.__class__.__name__}")
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
            self.logger.debug(f"Removed {removed} validators of type {validator_type.__name__}")
        return removed > 0

    def validate(self, data: Any) -> tuple[bool, list[str]]:
        """
        Run all validators and collect errors with consistent error handling.

        Args:
            data: Data to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Module boundary validation
        if not self.validators:
            self.logger.warning("No validators configured in pipeline")
            return True, []

        if data is None:
            errors.append("Cannot validate None data")
            return False, errors

        for validator in self.validators:
            try:
                # Reset validator state for clean validation
                validator.reset()

                # Run validation with consistent error handling
                if not validator.validate(data):
                    validator_errors = validator.get_errors()
                    if validator_errors:
                        errors.extend(validator_errors)
                    else:
                        errors.append(
                            f"{validator.__class__.__name__} validation failed (no specific errors reported)"
                        )

            except Exception as e:
                # Use consistent error propagation pattern
                error_msg = f"{validator.__class__.__name__} validation error: {e!s}"
                self.logger.error(error_msg)
                errors.append(error_msg)

                # Log the full exception for debugging while maintaining consistent interface
                self.logger.debug(
                    f"Validator exception details: {type(e).__name__}: {e!s}", exc_info=True
                )

        is_valid = len(errors) == 0

        if not is_valid:
            self.logger.warning(
                f"Validation pipeline failed with {len(errors)} errors for data type: {type(data).__name__}"
            )
        else:
            self.logger.debug(f"Validation pipeline passed for {len(self.validators)} validators")

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
        self.logger.debug("Cleared all validators")

    def get_validator_count(self) -> int:
        """Get the number of validators in the pipeline."""
        return len(self.validators)
