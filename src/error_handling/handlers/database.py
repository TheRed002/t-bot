"""Database error handlers with secure data sanitization."""

from typing import Any

from src.error_handling.base import ErrorHandlerBase
from src.error_handling.security_sanitizer import (
    SensitivityLevel,
    get_security_sanitizer,
)


class DatabaseErrorHandler(ErrorHandlerBase):
    """Handler for database-related errors with secure sanitization."""

    def __init__(self, next_handler=None):
        super().__init__(next_handler)
        self.sanitizer = get_security_sanitizer()

    def can_handle(self, error: Exception) -> bool:
        """Check if this is a database error."""
        # Check for SQLAlchemy errors
        error_type_name = type(error).__name__
        db_error_types = [
            "OperationalError",
            "IntegrityError",
            "DataError",
            "DatabaseError",
            "InterfaceError",
            "InternalError",
            "ProgrammingError",
            "NotSupportedError",
        ]

        # Check error message for database keywords
        error_msg = str(error).lower()
        db_keywords = [
            "database",
            "connection pool",
            "transaction",
            "deadlock",
            "lock",
            "constraint",
            "duplicate",
            "postgresql",
            "mysql",
            "sqlite",
            "redis",
        ]

        return error_type_name in db_error_types or any(
            keyword in error_msg for keyword in db_keywords
        )

    async def handle(
        self, error: Exception, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Handle database error with appropriate recovery strategy.

        Args:
            error: The database error
            context: Optional context

        Returns:
            Recovery action dictionary
        """
        error_msg = str(error).lower()

        # Handle deadlock with immediate retry
        if "deadlock" in error_msg:
            sanitized_msg = self.sanitizer.sanitize_error_message(
                str(error), SensitivityLevel.MEDIUM
            )
            self._logger.warning(f"Database deadlock detected: {sanitized_msg}")
            return {
                "action": "retry",
                "delay": 0.1,  # Small delay
                "reason": "deadlock",
                "max_retries": 3,
                "sanitized_error": sanitized_msg,
            }

        # Handle connection issues
        if any(word in error_msg for word in ["connection", "pool", "closed"]):
            sanitized_msg = self.sanitizer.sanitize_error_message(str(error), SensitivityLevel.HIGH)
            self._logger.error(f"Database connection error: {sanitized_msg}")
            return {
                "action": "reconnect",
                "delay": 5,
                "reason": "connection_lost",
                "sanitized_error": sanitized_msg,
            }

        # Handle constraint violations
        if any(word in error_msg for word in ["constraint", "duplicate", "unique"]):
            sanitized_msg = self.sanitizer.sanitize_error_message(
                str(error), SensitivityLevel.MEDIUM
            )
            self._logger.error(f"Database constraint violation: {sanitized_msg}")
            return {
                "action": "reject",
                "reason": "constraint_violation",
                "sanitized_error": sanitized_msg,
                "recoverable": False,
            }

        # Handle lock timeouts
        if "lock" in error_msg and "timeout" in error_msg:
            sanitized_msg = self.sanitizer.sanitize_error_message(
                str(error), SensitivityLevel.MEDIUM
            )
            self._logger.warning(f"Database lock timeout: {sanitized_msg}")
            return {
                "action": "retry",
                "delay": 2,
                "reason": "lock_timeout",
                "max_retries": 2,
                "sanitized_error": sanitized_msg,
            }

        # Default database error handling
        sanitized_msg = self.sanitizer.sanitize_error_message(str(error), SensitivityLevel.HIGH)
        self._logger.error(f"Database error: {sanitized_msg}")
        return {
            "action": "fail",
            "reason": "database_error",
            "sanitized_error": sanitized_msg,
            "requires_manual_intervention": True,
        }
