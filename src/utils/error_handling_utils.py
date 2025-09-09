"""
Common utilities for error handling to eliminate code duplication.
"""

from typing import Any

from src.error_handling.security_validator import SensitivityLevel


def get_or_create_sanitizer(sanitizer=None):
    """
    Get sanitizer instance or create default one.

    Args:
        sanitizer: Provided sanitizer instance (optional)

    Returns:
        Sanitizer instance
    """
    if sanitizer is None:
        from src.error_handling.security_validator import get_security_sanitizer

        return get_security_sanitizer()
    return sanitizer


def sanitize_error_with_level(error: Exception, level: SensitivityLevel, sanitizer=None) -> str:
    """
    Sanitize error message with specified sensitivity level.

    Args:
        error: Exception to sanitize
        level: Sensitivity level for sanitization
        sanitizer: Optional sanitizer instance

    Returns:
        Sanitized error message
    """
    sanitizer_instance = get_or_create_sanitizer(sanitizer)
    return sanitizer_instance.sanitize_error_message(str(error), level)


def create_recovery_response(
    action: str,
    reason: str,
    error: Exception,
    level: SensitivityLevel = SensitivityLevel.MEDIUM,
    sanitizer=None,
    **kwargs,
) -> dict[str, Any]:
    """
    Create standardized recovery response with sanitized error.

    Args:
        action: Recovery action to take
        reason: Reason for the recovery action
        error: Exception that occurred
        level: Sensitivity level for sanitization
        sanitizer: Optional sanitizer instance
        **kwargs: Additional response fields

    Returns:
        Recovery response dictionary
    """
    response = {
        "action": action,
        "reason": reason,
        "sanitized_error": sanitize_error_with_level(error, level, sanitizer),
        **kwargs,
    }
    return response


def extract_field_from_error(error: Exception, context: dict[str, Any] | None = None) -> str | None:
    """
    Extract field name from validation error or context.

    Args:
        error: Exception to analyze
        context: Optional context dictionary

    Returns:
        Field name if found, None otherwise
    """
    # Check context first
    if context and "field" in context:
        return context["field"]

    # Try to parse from error message
    import re

    error_str = str(error)

    # Look for patterns like "field 'price' is invalid"
    match = re.search(r"field\s+['\"]?(\w+)['\"]?", error_str, re.IGNORECASE)
    if match:
        return match.group(1)

    # Look for "price must be"
    match = re.search(r"(\w+)\s+must\s+be", error_str, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def extract_retry_after_from_error(error: Exception, context: dict[str, Any] | None = None):
    """
    Extract retry-after value from error or context.

    Args:
        error: Exception to analyze
        context: Optional context dictionary

    Returns:
        Decimal retry-after value if found, None otherwise
    """
    from decimal import Decimal

    # Check context first
    if context and "retry_after" in context:
        return Decimal(str(context["retry_after"]))

    # Try to parse from error message
    import re

    error_str = str(error)

    # Look for patterns like "retry after 30 seconds"
    match = re.search(r"retry[\s\-_]?after[\s:]*(\d+)", error_str, re.IGNORECASE)
    if match:
        return Decimal(match.group(1))

    # Look for "429" status with time
    match = re.search(r"429.*?(\d+)\s*(?:seconds?|s)", error_str, re.IGNORECASE)
    if match:
        return Decimal(match.group(1))

    return None


__all__ = [
    "create_recovery_response",
    "extract_field_from_error",
    "extract_retry_after_from_error",
    "get_or_create_sanitizer",
    "sanitize_error_with_level",
]
