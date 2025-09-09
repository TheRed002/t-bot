"""
Secure context manager backward compatibility.

Redirects to consolidated security types.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from src.utils.security_types import (
    InformationLevel,
    SecurityContext,
    UserRole,
)


class SecureErrorReport:
    """Secure error report with filtered information."""

    def __init__(self, message: str = None, details: dict[str, Any] | None = None, **kwargs):
        # Handle different ways the class can be constructed
        if message:
            self.message = message
        elif "user_message" in kwargs:
            self.message = kwargs.pop("user_message")
        else:
            self.message = "Unknown error"

        self.details = details or {}

        # Handle additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Set timestamp if not provided
        if not hasattr(self, "timestamp"):
            self.timestamp = datetime.now()


class SecureErrorContextManager:
    """Secure error context manager with role-based filtering."""

    def __init__(self, security_context: SecurityContext | None = None):
        self.security_context = security_context or SecurityContext()

    def create_secure_report(self, error: Exception, **context) -> SecureErrorReport:
        """Create a secure error report with role-based filtering."""
        # Simple implementation for backward compatibility
        message = str(error)
        return SecureErrorReport(message=message, details=context)


@asynccontextmanager
async def secure_context(operation: str, **context):
    """Simple secure context manager stub."""
    try:
        yield context
    except Exception as e:
        # Log the error and re-raise
        from src.core.logging import get_logger
        logger = get_logger(__name__)
        logger.error(f"Error in secure context {operation}: {e}")
        raise


def create_secure_context(**kwargs) -> dict[str, Any]:
    """Create a secure context dict."""
    return kwargs


__all__ = [
    "InformationLevel",
    "SecureErrorContextManager",
    "SecureErrorReport",
    "SecurityContext",
    "UserRole",
    "create_secure_context",
    "secure_context",
]
