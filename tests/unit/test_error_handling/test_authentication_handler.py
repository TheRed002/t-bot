"""
Tests for the authentication error handler.

This module tests the secure authentication error handling capabilities
including rate limiting, progressive delays, and security monitoring.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src.core.exceptions import AuthenticationError, ValidationError
from src.error_handling.handlers.authentication import AuthenticationErrorHandler
from src.error_handling.security_rate_limiter import SecurityThreat


class MockRateCheckResult:
    """Mock rate check result."""

    def __init__(self, allowed=True, suggested_retry_after=0, threat_level=SecurityThreat.LOW):
        self.allowed = allowed
        self.suggested_retry_after = suggested_retry_after
        self.threat_level = threat_level


class MockSecuritySanitizer:
    """Mock security sanitizer."""

    def sanitize_context(self, context, level):
        # Return all keys, but sensitive ones would be handled by _sanitize_auth_context
        return context.copy()

    def sanitize_error_message(self, message, level):
        return message.replace("secret", "[REDACTED]")


class MockRateLimiter:
    """Mock rate limiter."""

    def __init__(self):
        self.check_result = MockRateCheckResult()

    async def check_rate_limit(self, **kwargs):
        return self.check_result

    def record_failure(self, **kwargs):
        pass


class TestAuthenticationErrorHandler:
    """Test authentication error handler."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        with (
            patch(
                "src.error_handling.handlers.authentication.get_security_sanitizer"
            ) as mock_sanitizer,
            patch(
                "src.error_handling.handlers.authentication.get_security_rate_limiter"
            ) as mock_rate_limiter,
        ):
            mock_sanitizer.return_value = MockSecuritySanitizer()
            mock_rate_limiter.return_value = MockRateLimiter()

            return AuthenticationErrorHandler()

    def test_initialization(self, handler):
        """Test handler initialization."""
        assert handler is not None
        assert handler.safe_messages is not None
        assert handler.max_failed_attempts == 5
        assert handler.block_duration_minutes == 30

    def test_can_handle_authentication_errors(self, handler):
        """Test detection of authentication errors."""
        # Test by error type
        auth_error = AuthenticationError("Invalid credentials")
        assert handler.can_handle(auth_error)

        # Test by error message keywords
        generic_error = Exception("Authentication failed")
        assert handler.can_handle(generic_error)

        # Test HTTP codes
        http_error = Exception("HTTP 401 Unauthorized")
        assert handler.can_handle(http_error)

        # Test non-auth errors
        validation_error = ValidationError("Invalid input")
        assert not handler.can_handle(validation_error)

    def test_can_handle_various_auth_keywords(self, handler):
        """Test detection of various authentication keywords."""
        auth_keywords = [
            "permission denied",
            "access denied",
            "forbidden",
            "unauthorized",
            "invalid token",
            "session expired",
            "login failed",
        ]

        for keyword in auth_keywords:
            error = Exception(f"Error: {keyword}")
            assert handler.can_handle(error), f"Should handle error with keyword: {keyword}"

    @pytest.mark.asyncio
    async def test_handle_basic_auth_error(self, handler):
        """Test basic authentication error handling."""
        error = AuthenticationError("Invalid credentials")
        context = {"client_ip": "192.168.1.1", "user_id": "test_user"}

        result = await handler.handle(error, context)

        assert result["action"] == "reject"
        assert "authentication_failed" in result["reason"]
        assert "delay" in result
        assert result["recoverable"] is True

    @pytest.mark.asyncio
    async def test_handle_with_rate_limiting(self, handler):
        """Test handling when rate limited."""
        # Configure rate limiter to deny
        handler.rate_limiter.check_result = MockRateCheckResult(
            allowed=False, suggested_retry_after=60, threat_level=SecurityThreat.MEDIUM
        )

        error = AuthenticationError("Invalid credentials")
        context = {"client_ip": "192.168.1.1"}

        result = await handler.handle(error, context)

        assert result["action"] == "wait"
        assert result["reason"] == "authentication_rate_limited"
        assert result["retry_after"] == 60

    def test_extract_client_ip(self, handler):
        """Test client IP extraction."""
        # Test various IP keys
        contexts = [
            {"client_ip": "192.168.1.1"},
            {"remote_addr": "10.0.0.1"},
            {"x_forwarded_for": "172.16.0.1"},
        ]

        for context in contexts:
            ip = handler._extract_client_ip(context)
            assert ip is not None

        # Test invalid IP
        invalid_context = {"client_ip": "invalid_ip"}
        ip = handler._extract_client_ip(invalid_context)
        assert ip is None

    def test_extract_user_id_hashed(self, handler):
        """Test user ID extraction and hashing."""
        context = {"user_id": "test_user"}
        user_id = handler._extract_user_id(context)

        assert user_id is not None
        assert user_id.startswith("HASH_")
        assert "test_user" not in user_id  # Should be hashed

    def test_categorize_auth_errors(self, handler):
        """Test authentication error categorization."""
        test_cases = [
            ("Invalid credentials", "invalid_credentials"),
            ("Wrong password", "invalid_credentials"),
            ("Account locked", "account_locked"),
            ("Session expired", "session_expired"),
            ("Permission denied", "insufficient_permissions"),
            ("MFA required", "mfa_required"),
            ("Rate limited", "rate_limited"),
            ("Invalid token", "session_expired"),  # This maps to session_expired in the actual code
        ]

        for message, expected_category in test_cases:
            error = Exception(message)
            category = handler._categorize_auth_error(error)
            assert category == expected_category

    def test_progressive_delay_calculation(self, handler):
        """Test progressive delay calculation."""
        client_ip = "192.168.1.1"

        # No previous failures - when there are no failed attempts, max_failures = 0,
        # so delay = base_delay * (multiplier^0) = 1.0 * 1 = 1.0
        delay = handler._calculate_progressive_delay(client_ip, None)
        assert delay == handler.base_delay_seconds  # Should be 1.0, not 0

        # Simulate failures
        handler._failed_attempts[client_ip] = [datetime.now(timezone.utc) for _ in range(3)]

        delay = handler._calculate_progressive_delay(client_ip, None)
        assert delay > handler.base_delay_seconds  # Should be greater than base delay
        assert delay <= handler.max_delay_seconds

    def test_entity_blocking(self, handler):
        """Test entity blocking mechanism."""
        client_ip = "192.168.1.1"

        # Initially not blocked
        assert not handler._is_entity_blocked(client_ip)

        # Add to blocked entities
        future_time = datetime.now(timezone.utc) + timedelta(minutes=30)
        handler._blocked_entities[client_ip] = future_time

        # Should be blocked
        assert handler._is_entity_blocked(client_ip)

        # Add expired block
        past_time = datetime.now(timezone.utc) - timedelta(minutes=1)
        handler._blocked_entities[client_ip] = past_time

        # Should not be blocked (expired)
        assert not handler._is_entity_blocked(client_ip)
        # Should be removed from blocked list
        assert client_ip not in handler._blocked_entities

    def test_record_auth_failure(self, handler):
        """Test recording authentication failures."""
        client_ip = "192.168.1.1"
        user_id = "HASH_user123"

        # Record failures
        handler._record_auth_failure(client_ip, user_id, "invalid_credentials", {})

        # Check IP failures recorded
        assert client_ip in handler._failed_attempts
        assert len(handler._failed_attempts[client_ip]) == 1

        # Check user failures recorded
        user_key = f"user:{user_id}"
        assert user_key in handler._failed_attempts
        assert len(handler._failed_attempts[user_key]) == 1

    def test_threat_analysis(self, handler):
        """Test security threat analysis."""
        client_ip = "192.168.1.1"

        # Low threat (few failures)
        handler._failed_attempts[client_ip] = [datetime.now(timezone.utc) for _ in range(2)]

        threat = handler._analyze_threat_patterns(client_ip, None)
        assert threat == SecurityThreat.LOW

        # High threat (many failures - credential stuffing)
        handler._failed_attempts[client_ip] = [datetime.now(timezone.utc) for _ in range(60)]

        threat = handler._analyze_threat_patterns(client_ip, None)
        assert threat == SecurityThreat.HIGH

    def test_check_and_apply_blocks(self, handler):
        """Test blocking application."""
        client_ip = "192.168.1.1"

        # Add enough failures to trigger block
        handler._failed_attempts[client_ip] = [
            datetime.now(timezone.utc) for _ in range(handler.max_failed_attempts)
        ]

        handler._check_and_apply_blocks(client_ip, None)

        # Should be blocked now
        assert client_ip in handler._blocked_entities
        assert handler._is_entity_blocked(client_ip)

    @pytest.mark.asyncio
    async def test_handle_blocked_entity(self, handler):
        """Test handling of blocked entities."""
        client_ip = "192.168.1.1"

        # Block the IP
        future_time = datetime.now(timezone.utc) + timedelta(minutes=30)
        handler._blocked_entities[client_ip] = future_time

        error = AuthenticationError("Invalid credentials")
        context = {"client_ip": client_ip}

        result = await handler.handle(error, context)

        assert result["action"] == "block"
        assert result["reason"] == "authentication_blocked"
        assert result["recoverable"] is False

    def test_safe_user_messages(self, handler):
        """Test safe user messages."""
        # Verify no sensitive information in messages
        for category, message in handler.safe_messages.items():
            assert "database" not in message.lower()
            assert "server" not in message.lower()
            assert "internal" not in message.lower()
            assert "error" not in message.lower()

    def test_get_retry_recommendations(self, handler):
        """Test retry recommendations."""
        recommendations = handler._get_retry_recommendations("invalid_credentials")
        assert recommendations["should_retry"] is True
        assert "max_retries" in recommendations

        recommendations = handler._get_retry_recommendations("account_locked")
        assert recommendations["should_retry"] is False

    def test_get_security_headers(self, handler):
        """Test security headers."""
        headers = handler._get_security_headers()

        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Cache-Control",
        ]

        for header in required_headers:
            assert header in headers

    def test_cleanup_old_attempts(self, handler):
        """Test cleanup of old authentication attempts."""
        client_ip = "192.168.1.1"

        # Add old and new attempts
        old_time = datetime.now(timezone.utc) - timedelta(hours=25)
        new_time = datetime.now(timezone.utc) - timedelta(hours=1)

        handler._failed_attempts[client_ip] = [old_time, new_time]

        # Cleanup with 24 hour cutoff
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        handler._cleanup_old_attempts(cutoff)

        # Should only have new attempt
        assert len(handler._failed_attempts[client_ip]) == 1

    def test_is_valid_ip(self, handler):
        """Test IP validation."""
        assert handler._is_valid_ip("192.168.1.1")
        assert handler._is_valid_ip("::1")  # IPv6
        assert not handler._is_valid_ip("invalid_ip")
        assert not handler._is_valid_ip("999.999.999.999")

    def test_hash_identifier(self, handler):
        """Test identifier hashing."""
        identifier = "sensitive_user_id"
        hashed = handler._hash_identifier(identifier)

        assert hashed.startswith("HASH_")
        assert identifier not in hashed
        assert len(hashed) > len("HASH_")

    def test_sanitize_user_agent(self, handler):
        """Test user agent sanitization."""
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124"
        )
        sanitized = handler._sanitize_user_agent(user_agent)

        assert "Chrome" in sanitized
        assert "Windows NT 10.0" not in sanitized  # Should be replaced
        assert len(sanitized) <= 100

    def test_get_security_stats(self, handler):
        """Test security statistics."""
        # Add some test data
        client_ip = "192.168.1.1"
        handler._failed_attempts[client_ip] = [datetime.now(timezone.utc) for _ in range(3)]
        handler._blocked_entities[client_ip] = datetime.now(timezone.utc) + timedelta(minutes=30)

        stats = handler.get_security_stats()

        assert "total_blocked_entities" in stats
        assert "total_failed_attempts_tracked" in stats
        assert "recent_failures_last_hour" in stats
        assert "authentication_rules" in stats

        assert stats["total_blocked_entities"] == 1
        assert stats["total_failed_attempts_tracked"] == 3

    def test_reset_entity_failures(self, handler):
        """Test resetting entity failures."""
        client_ip = "192.168.1.1"

        # Add failures and block
        handler._failed_attempts[client_ip] = [datetime.now(timezone.utc)]
        handler._blocked_entities[client_ip] = datetime.now(timezone.utc) + timedelta(minutes=30)

        # Reset
        result = handler.reset_entity_failures(client_ip)

        assert result is True
        assert client_ip not in handler._failed_attempts
        assert client_ip not in handler._blocked_entities

        # Reset non-existent entity
        result = handler.reset_entity_failures("non_existent")
        assert result is False

    def test_sanitize_auth_context(self, handler):
        """Test authentication context sanitization."""
        context = {
            "user_id": "test_user",
            "password": "secret123",
            "token": "bearer_token",
            "safe_data": "visible",
        }

        sanitized = handler._sanitize_auth_context(context)

        assert sanitized["safe_data"] == "visible"
        assert sanitized["password"] == "[REDACTED_AUTH_DATA]"  # Sensitive key should be redacted
        assert sanitized["token"] == "[REDACTED_AUTH_DATA]"  # Sensitive key should be redacted
        assert "sanitized_at" in sanitized
        assert sanitized["security_level"] == "critical"

    @pytest.mark.asyncio
    async def test_log_security_event(self, handler):
        """Test security event logging."""
        with patch.object(handler._logger, "info") as mock_info:
            await handler._log_security_event(
                Exception("test"),
                "invalid_credentials",
                "192.168.1.1",
                "user123",
                SecurityThreat.LOW,
                {},
            )

            mock_info.assert_called_once()

    def test_create_secure_auth_response(self, handler):
        """Test secure authentication response creation."""
        error = AuthenticationError("Invalid credentials")
        response = handler._create_secure_auth_response(
            error, "invalid_credentials", 2.0, SecurityThreat.MEDIUM, {}
        )

        assert response["action"] == "reject"
        assert response["delay"] == "2.0"
        assert response["threat_level"] == SecurityThreat.MEDIUM
        assert "retry_recommendations" in response
        assert "security_headers" in response

    def test_create_blocked_response(self, handler):
        """Test blocked entity response creation."""
        response = handler._create_blocked_response("invalid_credentials")

        assert response["action"] == "block"
        assert response["recoverable"] is False
        assert response["retry_after"] == handler.block_duration_minutes * 60

    def test_create_rate_limited_response(self, handler):
        """Test rate limited response creation."""
        rate_check = MockRateCheckResult(
            allowed=False, suggested_retry_after=120, threat_level=SecurityThreat.MEDIUM
        )

        response = handler._create_rate_limited_response(rate_check)

        assert response["action"] == "wait"
        assert response["retry_after"] == 120
        assert response["recoverable"] is True
