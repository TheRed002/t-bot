"""
Simple security validation for error handling.

Basic sanitization and validation functions.
"""

import re
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

from .constants import SENSITIVE_KEYS


@dataclass
class RateLimitResult:
    """Rate limit check result."""

    allowed: bool
    reason: str | None = None
    suggested_retry_after: int | None = None


def sanitize_error_data(data: dict[str, Any]) -> dict[str, Any]:
    """Basic sanitization of error data to remove sensitive information."""

    if not isinstance(data, dict):
        return {}

    sanitized: dict[str, Any] = {}
    sensitive_keys = SENSITIVE_KEYS

    for key, value in data.items():
        # Skip sensitive keys
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            sanitized[key] = "[REDACTED]"
            continue

        # Sanitize string values
        if isinstance(value, str):
            sanitized[key] = sanitize_string_value(value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_error_data(value)
        elif isinstance(value, list):
            sanitized[key] = [
                (
                    sanitize_error_data(item)
                    if isinstance(item, dict)
                    else sanitize_string_value(item)
                    if isinstance(item, str)
                    else item
                )
                for item in value
            ]
        else:
            sanitized[key] = value

    return sanitized


def sanitize_string_value(value: str) -> str:
    """Sanitize string values to remove potential sensitive data."""

    if not isinstance(value, str):
        return str(value)

    # Remove potential API keys (look for long alphanumeric strings)
    value = re.sub(r"\b[A-Za-z0-9]{32,}\b", "[REDACTED_KEY]", value)

    # Remove potential tokens
    value = re.sub(r"token[=:]\s*[A-Za-z0-9]+", "token=[REDACTED]", value, flags=re.IGNORECASE)

    # Remove email addresses in error messages
    value = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]", value
    )

    return value


def validate_error_context(context: dict[str, Any]) -> bool:
    """Basic validation of error context."""

    if not isinstance(context, dict):
        return False

    required_fields = {"error_type", "component"}

    # Check required fields exist
    for field in required_fields:
        if field not in context:
            return False

    return True


_sanitizer_instance = None


def get_security_sanitizer() -> "SecuritySanitizer":
    """Get a simple security sanitizer for backward compatibility."""
    global _sanitizer_instance
    if _sanitizer_instance is None:
        _sanitizer_instance = SecuritySanitizer()
    return _sanitizer_instance


class SecuritySanitizer:
    """Simple security sanitizer."""

    def sanitize_context(self, context: dict[str, Any], sensitivity_level=None) -> dict[str, Any]:
        """Sanitize error context."""
        return sanitize_error_data(context)

    def sanitize_error_message(self, message: str, sensitivity_level=None) -> str:
        """Sanitize error message."""
        return sanitize_string_value(message)

    def sanitize_stack_trace(self, stack_trace: str, sensitivity_level=None) -> str:
        """Sanitize stack trace."""
        return sanitize_string_value(stack_trace)

    def validate_context(self, context: dict[str, Any]) -> bool:
        """Validate error context."""
        return validate_error_context(context)


class SecurityRateLimiter:
    """Simple rate limiter stub."""

    def __init__(self):
        self.request_counts = {}

    def is_allowed(self, key: str) -> bool:
        """Always allow for simplified implementation."""
        return True

    def increment(self, key: str) -> None:
        """No-op for simplified implementation."""
        # Simplified rate limiter does not track increments
        return

    async def check_rate_limit(
        self, component: str, operation: str, context: dict[str, Any] | None = None
    ) -> "RateLimitResult":
        """Check rate limit asynchronously - always allow for simplified implementation."""
        return RateLimitResult(allowed=True, reason=None, suggested_retry_after=None)


def get_security_rate_limiter() -> "SecurityRateLimiter":
    """Get a simple rate limiter for backward compatibility."""
    return SecurityRateLimiter()


class SensitivityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorPattern:
    """Simple error pattern for backward compatibility."""

    def __init__(self, pattern_id: str, pattern_type: str = "frequency", **kwargs):
        self.pattern_id = pattern_id
        self.pattern_type = pattern_type
        self.component = kwargs.get("component", "unknown")
        self.error_type = kwargs.get("error_type", "unknown")
        self.frequency = kwargs.get("frequency", 1)
        self.severity = kwargs.get("severity", "medium")
        self.confidence = Decimal(str(kwargs.get("confidence", "0.8")))
        self.is_active = kwargs.get("is_active", True)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "component": self.component,
            "error_type": self.error_type,
            "frequency": self.frequency,
            "severity": self.severity,
            "confidence": self.confidence,
            "is_active": self.is_active,
        }
