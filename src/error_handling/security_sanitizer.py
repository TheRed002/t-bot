"""
Security sanitizer for error handling.

Redirects to simplified security validator.
"""

from .security_validator import (
    SecuritySanitizer,
    SensitivityLevel,
    get_security_sanitizer,
    sanitize_error_data,
    sanitize_string_value,
    validate_error_context,
)

__all__ = [
    "SecuritySanitizer",
    "SensitivityLevel",
    "get_security_sanitizer",
    "sanitize_error_data",
    "sanitize_string_value",
    "validate_error_context",
]
