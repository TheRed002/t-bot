"""
Tests for security sanitizer module.

Testing security sanitization imports and exports.
"""

import pytest

from src.error_handling.security_sanitizer import (
    SecuritySanitizer,
    SensitivityLevel,
    get_security_sanitizer,
    sanitize_error_data,
    sanitize_string_value,
    validate_error_context,
)


class TestSecuritySanitizerImports:
    """Test security sanitizer imports."""

    def test_security_sanitizer_import(self):
        """Test SecuritySanitizer import."""
        sanitizer = SecuritySanitizer()
        assert isinstance(sanitizer, SecuritySanitizer)
        assert hasattr(sanitizer, 'sanitize_context')
        assert hasattr(sanitizer, 'sanitize_error_message')

    def test_sensitivity_level_import(self):
        """Test SensitivityLevel import."""
        assert SensitivityLevel.LOW.value == "low"
        assert SensitivityLevel.MEDIUM.value == "medium"
        assert SensitivityLevel.HIGH.value == "high"
        assert SensitivityLevel.CRITICAL.value == "critical"

    def test_get_security_sanitizer_import(self):
        """Test get_security_sanitizer function import."""
        sanitizer = get_security_sanitizer()
        assert isinstance(sanitizer, SecuritySanitizer)

    def test_sanitize_error_data_import(self):
        """Test sanitize_error_data function import."""
        data = {"test": "value", "password": "secret"}
        result = sanitize_error_data(data)
        
        assert result["test"] == "value"
        assert result["password"] == "[REDACTED]"

    def test_sanitize_string_value_import(self):
        """Test sanitize_string_value function import."""
        result = sanitize_string_value("test@example.com")
        assert "[EMAIL_REDACTED]" in result

    def test_validate_error_context_import(self):
        """Test validate_error_context function import."""
        valid_context = {"error_type": "TestError", "component": "test"}
        invalid_context = {"error_type": "TestError"}  # Missing component
        
        assert validate_error_context(valid_context) is True
        assert validate_error_context(invalid_context) is False


class TestSecuritySanitizerExports:
    """Test security sanitizer exports."""

    def test_all_exports_available(self):
        """Test that all expected exports are available."""
        from src.error_handling.security_sanitizer import __all__
        
        expected_exports = [
            "SecuritySanitizer",
            "SensitivityLevel", 
            "get_security_sanitizer",
            "sanitize_error_data",
            "sanitize_string_value",
            "validate_error_context",
        ]
        
        for export in expected_exports:
            assert export in __all__

    def test_functions_work_as_expected(self):
        """Test that imported functions work as expected."""
        # Test sanitize_error_data
        data = {"api_key": "secret123", "message": "test"}
        result = sanitize_error_data(data)
        assert result["api_key"] == "[REDACTED]"
        assert result["message"] == "test"
        
        # Test sanitize_string_value  
        text = "Error with token=abc123"
        result = sanitize_string_value(text)
        assert "token=[REDACTED]" in result
        
        # Test validate_error_context
        context = {"error_type": "Error", "component": "test"}
        assert validate_error_context(context) is True

    def test_sensitivity_level_enum_values(self):
        """Test SensitivityLevel enum values."""
        levels = [SensitivityLevel.LOW, SensitivityLevel.MEDIUM, 
                 SensitivityLevel.HIGH, SensitivityLevel.CRITICAL]
        values = [level.value for level in levels]
        
        assert "low" in values
        assert "medium" in values
        assert "high" in values
        assert "critical" in values

    def test_get_security_sanitizer_singleton(self):
        """Test get_security_sanitizer returns singleton."""
        sanitizer1 = get_security_sanitizer()
        sanitizer2 = get_security_sanitizer()
        
        assert sanitizer1 is sanitizer2
        assert isinstance(sanitizer1, SecuritySanitizer)