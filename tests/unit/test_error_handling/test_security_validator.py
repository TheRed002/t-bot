"""
Tests for security validator module.

Testing security validation, sanitization, and rate limiting functionality.
"""

from decimal import Decimal
from unittest.mock import patch

import pytest

from src.error_handling.security_validator import (
    ErrorPattern,
    RateLimitResult,
    SecurityRateLimiter,
    SecuritySanitizer,
    SensitivityLevel,
    get_security_rate_limiter,
    get_security_sanitizer,
    sanitize_error_data,
    sanitize_string_value,
    validate_error_context,
)


class TestRateLimitResult:
    """Test rate limit result dataclass."""

    def test_rate_limit_result_allowed(self):
        """Test rate limit result when allowed."""
        result = RateLimitResult(allowed=True)
        
        assert result.allowed is True
        assert result.reason is None
        assert result.suggested_retry_after is None

    def test_rate_limit_result_denied(self):
        """Test rate limit result when denied."""
        result = RateLimitResult(
            allowed=False,
            reason="Rate limit exceeded",
            suggested_retry_after=60
        )
        
        assert result.allowed is False
        assert result.reason == "Rate limit exceeded"
        assert result.suggested_retry_after == 60


class TestSanitizeErrorData:
    """Test error data sanitization."""

    def test_sanitize_error_data_empty_dict(self):
        """Test sanitizing empty dictionary."""
        result = sanitize_error_data({})
        assert result == {}

    def test_sanitize_error_data_non_dict(self):
        """Test sanitizing non-dictionary input."""
        result = sanitize_error_data("not a dict")
        assert result == {}

    def test_sanitize_sensitive_keys(self):
        """Test sanitizing sensitive keys."""
        data = {
            "username": "testuser",
            "password": "secret123",
            "api_key": "abc123def456",
            "token": "xyz789",
            "message": "Error occurred"
        }
        
        result = sanitize_error_data(data)
        
        assert result["username"] == "testuser"
        assert result["password"] == "[REDACTED]"
        assert result["api_key"] == "[REDACTED]"
        assert result["token"] == "[REDACTED]"
        assert result["message"] == "Error occurred"

    def test_sanitize_nested_dict(self):
        """Test sanitizing nested dictionaries."""
        data = {
            "outer": {
                "inner": {
                    "secret": "hidden_value",
                    "public": "visible_value"
                }
            }
        }
        
        result = sanitize_error_data(data)
        
        assert result["outer"]["inner"]["secret"] == "[REDACTED]"
        assert result["outer"]["inner"]["public"] == "visible_value"

    def test_sanitize_list_values(self):
        """Test sanitizing list values."""
        data = {
            "items": [
                "public_string",
                {"secret": "hidden", "public": "visible"},
                "another_string"
            ]
        }
        
        result = sanitize_error_data(data)
        
        assert result["items"][0] == "public_string"
        assert result["items"][1]["secret"] == "[REDACTED]"
        assert result["items"][1]["public"] == "visible"
        assert result["items"][2] == "another_string"

    def test_sanitize_mixed_types(self):
        """Test sanitizing mixed data types."""
        data = {
            "string": "test_string",
            "number": 42,
            "decimal": Decimal("3.14"),
            "boolean": True,
            "none_value": None
        }
        
        result = sanitize_error_data(data)
        
        assert result["string"] == "test_string"
        assert result["number"] == 42
        assert result["decimal"] == Decimal("3.14")
        assert result["boolean"] is True
        assert result["none_value"] is None


class TestSanitizeStringValue:
    """Test string value sanitization."""

    def test_sanitize_string_value_normal(self):
        """Test sanitizing normal string."""
        result = sanitize_string_value("This is a normal string")
        assert result == "This is a normal string"

    def test_sanitize_string_value_non_string(self):
        """Test sanitizing non-string input."""
        result = sanitize_string_value(123)
        assert result == "123"

    def test_sanitize_api_key(self):
        """Test sanitizing long alphanumeric strings (API keys)."""
        text = "API key abc123def456ghi789jkl012mno345pqrs leaked"
        result = sanitize_string_value(text)
        assert "[REDACTED_KEY]" in result
        assert "abc123def456ghi789jkl012mno345pqrs" not in result

    def test_sanitize_token(self):
        """Test sanitizing token patterns."""
        text = "token=abc123def456"
        result = sanitize_string_value(text)
        assert result == "token=[REDACTED]"

    def test_sanitize_email(self):
        """Test sanitizing email addresses."""
        text = "User error: john.doe@example.com failed to login"
        result = sanitize_string_value(text)
        assert "[EMAIL_REDACTED]" in result
        assert "john.doe@example.com" not in result

    def test_sanitize_multiple_patterns(self):
        """Test sanitizing multiple sensitive patterns."""
        text = "Error for user john@test.com with key abc123def456ghi789jkl012mno345pqrs found"
        result = sanitize_string_value(text)
        
        assert "[EMAIL_REDACTED]" in result
        assert "[REDACTED_KEY]" in result
        assert "john@test.com" not in result
        assert "abc123def456ghi789jkl012mno345pqrs" not in result


class TestValidateErrorContext:
    """Test error context validation."""

    def test_validate_error_context_valid(self):
        """Test validating valid error context."""
        context = {
            "error_type": "ValidationError",
            "component": "data_service",
            "operation": "validate_input"
        }
        
        assert validate_error_context(context) is True

    def test_validate_error_context_missing_error_type(self):
        """Test validating context missing error_type."""
        context = {
            "component": "data_service",
            "operation": "validate_input"
        }
        
        assert validate_error_context(context) is False

    def test_validate_error_context_missing_component(self):
        """Test validating context missing component."""
        context = {
            "error_type": "ValidationError",
            "operation": "validate_input"
        }
        
        assert validate_error_context(context) is False

    def test_validate_error_context_non_dict(self):
        """Test validating non-dictionary context."""
        assert validate_error_context("not a dict") is False
        assert validate_error_context(None) is False
        assert validate_error_context([]) is False


class TestSecuritySanitizer:
    """Test security sanitizer class."""

    @pytest.fixture
    def sanitizer(self):
        """Create security sanitizer instance."""
        return SecuritySanitizer()

    def test_sanitize_context(self, sanitizer):
        """Test sanitizing error context."""
        context = {
            "error": "Test error",
            "password": "secret123"
        }
        
        result = sanitizer.sanitize_context(context)
        
        assert result["error"] == "Test error"
        assert result["password"] == "[REDACTED]"

    def test_sanitize_context_with_sensitivity_level(self, sanitizer):
        """Test sanitizing context with sensitivity level (ignored)."""
        context = {"message": "Test message"}
        
        result = sanitizer.sanitize_context(context, sensitivity_level="high")
        
        assert result["message"] == "Test message"

    def test_sanitize_error_message(self, sanitizer):
        """Test sanitizing error message."""
        message = "Error with token=abc123"
        
        result = sanitizer.sanitize_error_message(message)
        
        assert result == "Error with token=[REDACTED]"

    def test_sanitize_stack_trace(self, sanitizer):
        """Test sanitizing stack trace."""
        stack_trace = "File '/app/config.py' contains key abc123def456ghi789jkl012mno345pqrs found"
        
        result = sanitizer.sanitize_stack_trace(stack_trace)
        
        assert "[REDACTED_KEY]" in result
        assert "abc123def456ghi789jkl012mno345pqrs" not in result

    def test_validate_context(self, sanitizer):
        """Test validating error context."""
        valid_context = {
            "error_type": "TestError",
            "component": "test_component"
        }
        
        invalid_context = {
            "error_type": "TestError"
            # Missing component
        }
        
        assert sanitizer.validate_context(valid_context) is True
        assert sanitizer.validate_context(invalid_context) is False


class TestGetSecuritySanitizer:
    """Test security sanitizer getter."""

    def test_get_security_sanitizer_singleton(self):
        """Test that get_security_sanitizer returns singleton instance."""
        sanitizer1 = get_security_sanitizer()
        sanitizer2 = get_security_sanitizer()
        
        assert isinstance(sanitizer1, SecuritySanitizer)
        assert sanitizer1 is sanitizer2

    def test_get_security_sanitizer_reset_singleton(self):
        """Test singleton behavior after reset."""
        # Reset the global instance
        import src.error_handling.security_validator
        src.error_handling.security_validator._sanitizer_instance = None
        
        sanitizer1 = get_security_sanitizer()
        sanitizer2 = get_security_sanitizer()
        
        assert sanitizer1 is sanitizer2


class TestSecurityRateLimiter:
    """Test security rate limiter."""

    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter instance."""
        return SecurityRateLimiter()

    def test_rate_limiter_initialization(self, rate_limiter):
        """Test rate limiter initialization."""
        assert hasattr(rate_limiter, 'request_counts')
        assert isinstance(rate_limiter.request_counts, dict)

    def test_is_allowed_always_true(self, rate_limiter):
        """Test that is_allowed always returns True."""
        assert rate_limiter.is_allowed("test_key") is True
        assert rate_limiter.is_allowed("another_key") is True

    def test_increment_no_op(self, rate_limiter):
        """Test that increment is a no-op."""
        # Should not raise any exceptions
        rate_limiter.increment("test_key")
        rate_limiter.increment("another_key")

    @pytest.mark.asyncio
    async def test_check_rate_limit_always_allowed(self, rate_limiter):
        """Test that check_rate_limit always allows."""
        result = await rate_limiter.check_rate_limit("component", "operation")
        
        assert isinstance(result, RateLimitResult)
        assert result.allowed is True
        assert result.reason is None
        assert result.suggested_retry_after is None

    @pytest.mark.asyncio
    async def test_check_rate_limit_with_context(self, rate_limiter):
        """Test check_rate_limit with context."""
        context = {"user_id": "123", "ip": "192.168.1.1"}
        result = await rate_limiter.check_rate_limit("auth", "login", context)
        
        assert result.allowed is True


class TestGetSecurityRateLimiter:
    """Test security rate limiter getter."""

    def test_get_security_rate_limiter(self):
        """Test getting security rate limiter instance."""
        limiter = get_security_rate_limiter()
        
        assert isinstance(limiter, SecurityRateLimiter)

    def test_get_security_rate_limiter_multiple_calls(self):
        """Test multiple calls return new instances."""
        limiter1 = get_security_rate_limiter()
        limiter2 = get_security_rate_limiter()
        
        # Should return new instances (not singleton)
        assert isinstance(limiter1, SecurityRateLimiter)
        assert isinstance(limiter2, SecurityRateLimiter)


class TestSensitivityLevel:
    """Test sensitivity level enum."""

    def test_sensitivity_level_values(self):
        """Test sensitivity level enum values."""
        assert SensitivityLevel.LOW.value == "low"
        assert SensitivityLevel.MEDIUM.value == "medium"
        assert SensitivityLevel.HIGH.value == "high"
        assert SensitivityLevel.CRITICAL.value == "critical"

    def test_sensitivity_level_comparison(self):
        """Test sensitivity level comparison."""
        assert SensitivityLevel.LOW != SensitivityLevel.HIGH
        assert SensitivityLevel.MEDIUM == SensitivityLevel.MEDIUM


class TestErrorPattern:
    """Test error pattern class."""

    def test_error_pattern_initialization_defaults(self):
        """Test error pattern initialization with defaults."""
        pattern = ErrorPattern("test_pattern")
        
        assert pattern.pattern_id == "test_pattern"
        assert pattern.pattern_type == "frequency"
        assert pattern.component == "unknown"
        assert pattern.error_type == "unknown"
        assert pattern.frequency == 1
        assert pattern.severity == "medium"
        assert pattern.confidence == Decimal("0.8")
        assert pattern.is_active is True

    def test_error_pattern_initialization_custom(self):
        """Test error pattern initialization with custom values."""
        pattern = ErrorPattern(
            "custom_pattern",
            pattern_type="correlation",
            component="data_service",
            error_type="ValidationError",
            frequency=5,
            severity="high",
            confidence=0.95,
            is_active=False
        )
        
        assert pattern.pattern_id == "custom_pattern"
        assert pattern.pattern_type == "correlation"
        assert pattern.component == "data_service"
        assert pattern.error_type == "ValidationError"
        assert pattern.frequency == 5
        assert pattern.severity == "high"
        assert pattern.confidence == Decimal("0.95")
        assert pattern.is_active is False

    def test_error_pattern_to_dict(self):
        """Test converting error pattern to dictionary."""
        pattern = ErrorPattern(
            "dict_test",
            component="test_component",
            error_type="TestError",
            frequency=3,
            severity="low",
            confidence=0.75,
            is_active=True
        )
        
        result = pattern.to_dict()
        
        expected = {
            "pattern_id": "dict_test",
            "pattern_type": "frequency",
            "component": "test_component",
            "error_type": "TestError",
            "frequency": 3,
            "severity": "low",
            "confidence": Decimal("0.75"),
            "is_active": True,
        }
        
        assert result == expected

    def test_error_pattern_confidence_decimal_conversion(self):
        """Test that confidence is properly converted to Decimal."""
        pattern = ErrorPattern("test", confidence=0.123456789)
        
        assert isinstance(pattern.confidence, Decimal)
        assert pattern.confidence == Decimal("0.123456789")