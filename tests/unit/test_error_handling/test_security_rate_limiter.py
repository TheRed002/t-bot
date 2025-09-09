"""
Tests for security rate limiter module.

Testing rate limiting, security threat classification, and configuration.
"""

import pytest

from src.error_handling.security_rate_limiter import (
    RateLimitConfig,
    SecurityRateLimiter,
    SecurityThreat,
    get_security_rate_limiter,
    record_recovery_failure,
)


class TestRateLimitConfig:
    """Test rate limit configuration."""

    def test_rate_limit_config_defaults(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        
        assert config.requests_per_second == 10
        assert config.requests_per_minute == 600
        assert config.requests_per_hour == 3600
        assert config.burst_allowance == 5

    def test_rate_limit_config_custom(self):
        """Test custom configuration values."""
        config = RateLimitConfig(
            requests_per_second=20,
            requests_per_minute=1200,
            requests_per_hour=7200,
            burst_allowance=10,
        )
        
        assert config.requests_per_second == 20
        assert config.requests_per_minute == 1200
        assert config.requests_per_hour == 7200
        assert config.burst_allowance == 10


class TestSecurityThreat:
    """Test security threat classification."""

    def test_security_threat_levels(self):
        """Test security threat level constants."""
        assert SecurityThreat.LOW == "low"
        assert SecurityThreat.MEDIUM == "medium"
        assert SecurityThreat.HIGH == "high"
        assert SecurityThreat.CRITICAL == "critical"
        assert SecurityThreat.SUSPICIOUS_ACTIVITY == "suspicious_activity"
        assert SecurityThreat.RATE_LIMIT_EXCEEDED == "rate_limit_exceeded"


class TestRecordRecoveryFailure:
    """Test recovery failure recording."""

    def test_record_recovery_failure_no_op(self):
        """Test that record_recovery_failure is a no-op."""
        # Should not raise any exceptions
        record_recovery_failure("component", "operation", "error")
        record_recovery_failure(
            "test_component", 
            "test_operation", 
            "high", 
            extra_param="value"
        )


class TestGetSecurityRateLimiter:
    """Test security rate limiter getter."""

    def test_get_security_rate_limiter(self):
        """Test getting security rate limiter instance."""
        limiter = get_security_rate_limiter()
        
        assert isinstance(limiter, SecurityRateLimiter)
        
        # Should return new instances (not singleton)
        limiter2 = get_security_rate_limiter()
        assert isinstance(limiter2, SecurityRateLimiter)


class TestSecurityRateLimiterIntegration:
    """Test security rate limiter integration."""

    def test_all_exports_available(self):
        """Test that all expected exports are available."""
        from src.error_handling.security_rate_limiter import __all__
        
        expected_exports = [
            "RateLimitConfig",
            "SecurityRateLimiter", 
            "get_security_rate_limiter",
            "record_recovery_failure",
            "SecurityThreat",
        ]
        
        for export in expected_exports:
            assert export in __all__

    def test_imports_from_security_validator(self):
        """Test that imports work correctly from security_validator."""
        # This should not raise ImportError
        from src.error_handling.security_rate_limiter import SecurityRateLimiter
        
        limiter = SecurityRateLimiter()
        assert hasattr(limiter, 'is_allowed')
        assert hasattr(limiter, 'increment')