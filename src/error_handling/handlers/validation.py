"""Validation error handlers with secure data sanitization."""

from typing import Any

from src.core.exceptions import ValidationError
from src.error_handling.base import ErrorHandlerBase
from src.error_handling.security_validator import SensitivityLevel
from src.utils.error_categorization import detect_data_validation_error
from src.utils.error_handling_utils import (
    create_recovery_response,
    extract_field_from_error,
    get_or_create_sanitizer,
    sanitize_error_with_level,
)


class ValidationErrorHandler(ErrorHandlerBase):
    """Handler for validation errors with secure sanitization."""

    def __init__(self, next_handler=None, sanitizer=None):
        super().__init__(next_handler)
        self.sanitizer = get_or_create_sanitizer(sanitizer)

    def can_handle(self, error: Exception) -> bool:
        """Check if this is a validation error."""
        validation_errors = (
            ValidationError,
            ValueError,
            TypeError,
            AssertionError,
        )

        # Check for validation-related keywords
        error_msg = str(error).lower()
        validation_keywords = [
            "validation",
            "invalid",
            "must be",
            "should be",
            "required",
            "missing",
            "format",
            "type error",
        ]

        return isinstance(error, validation_errors) or any(
            keyword in error_msg for keyword in validation_keywords
        )

    async def handle(
        self, error: Exception, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Handle validation error.

        Validation errors typically should not be retried as they
        indicate incorrect input that needs to be fixed.

        Args:
            error: The validation error
            context: Optional context

        Returns:
            Recovery action dictionary
        """
        sanitized_msg = sanitize_error_with_level(error, SensitivityLevel.LOW, self.sanitizer)
        self._logger.error(f"Validation error: {sanitized_msg}")

        # Extract field information if available
        field = extract_field_from_error(error, context)

        return create_recovery_response(
            action="reject",
            reason="validation_failed",
            error=error,
            level=SensitivityLevel.LOW,
            sanitizer=self.sanitizer,
            field=field,
            recoverable=False,
            user_action_required=True,
        )


class DataValidationErrorHandler(ErrorHandlerBase):
    """Handler for data validation errors with secure sanitization."""

    def __init__(self, next_handler=None, sanitizer=None):
        super().__init__(next_handler)
        self.sanitizer = get_or_create_sanitizer(sanitizer)

    def can_handle(self, error: Exception) -> bool:
        """Check if this is a data validation error."""
        error_msg = str(error).lower()
        return detect_data_validation_error(error_msg)

    async def handle(
        self, error: Exception, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Handle data validation error.

        Args:
            error: The data validation error
            context: Optional context

        Returns:
            Recovery action dictionary
        """
        sanitized_msg = sanitize_error_with_level(error, SensitivityLevel.MEDIUM, self.sanitizer)
        self._logger.error(f"Data validation error: {sanitized_msg}")

        # Check if we can request fresh data
        if context and context.get("can_refresh", False):
            return create_recovery_response(
                action="refresh",
                reason="data_validation_failed",
                error=error,
                level=SensitivityLevel.MEDIUM,
                sanitizer=self.sanitizer,
            )

        return create_recovery_response(
            action="reject",
            reason="invalid_data",
            error=error,
            level=SensitivityLevel.MEDIUM,
            sanitizer=self.sanitizer,
            recoverable=False,
        )
