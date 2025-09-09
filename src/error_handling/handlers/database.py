"""Database error handlers with secure data sanitization."""

from typing import Any

from src.core.exceptions import DatabaseConnectionError, DatabaseError, DatabaseQueryError
from src.error_handling.base import ErrorHandlerBase
from src.error_handling.security_validator import SensitivityLevel
from src.utils.error_handling_utils import (
    create_recovery_response,
    get_or_create_sanitizer,
)


class DatabaseErrorHandler(ErrorHandlerBase):
    """Handler for database-related errors with secure sanitization."""

    def __init__(self, next_handler=None, sanitizer=None):
        super().__init__(next_handler)
        self.sanitizer = get_or_create_sanitizer(sanitizer)

    def can_handle(self, error: Exception) -> bool:
        """Check if this is a database error."""
        # Check for specific database exceptions from core.exceptions first
        if isinstance(error, DatabaseError | DatabaseConnectionError | DatabaseQueryError):
            return True

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
            response = create_recovery_response(
                action="retry",
                reason="deadlock",
                error=error,
                level=SensitivityLevel.MEDIUM,
                sanitizer=self.sanitizer,
                delay="0.1",
                max_retries=3,
            )
            self._logger.warning(f"Database deadlock detected: {response['sanitized_error']}")
            return response

        # Handle connection issues
        if any(word in error_msg for word in ["connection", "pool", "closed"]):
            response = create_recovery_response(
                action="reconnect",
                reason="connection_lost",
                error=error,
                level=SensitivityLevel.HIGH,
                sanitizer=self.sanitizer,
                delay="5",
            )
            self._logger.error(f"Database connection error: {response['sanitized_error']}")
            return response

        # Handle constraint violations
        if any(word in error_msg for word in ["constraint", "duplicate", "unique"]):
            response = create_recovery_response(
                action="reject",
                reason="constraint_violation",
                error=error,
                level=SensitivityLevel.MEDIUM,
                sanitizer=self.sanitizer,
                recoverable=False,
            )
            self._logger.error(f"Database constraint violation: {response['sanitized_error']}")
            return response

        # Handle lock timeouts
        if "lock" in error_msg and "timeout" in error_msg:
            response = create_recovery_response(
                action="retry",
                reason="lock_timeout",
                error=error,
                level=SensitivityLevel.MEDIUM,
                sanitizer=self.sanitizer,
                delay="2",
                max_retries=2,
            )
            self._logger.warning(f"Database lock timeout: {response['sanitized_error']}")
            return response

        # Default database error handling
        response = create_recovery_response(
            action="fail",
            reason="database_error",
            error=error,
            level=SensitivityLevel.HIGH,
            sanitizer=self.sanitizer,
            requires_manual_intervention=True,
        )
        self._logger.error(f"Database error: {response['sanitized_error']}")
        return response
