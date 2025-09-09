"""
Secure authentication error handler.

This module provides specialized handling for authentication and authorization errors
with comprehensive security measures to prevent credential exposure and security breaches.

CRITICAL: Prevents exposure of authentication credentials, tokens, session data,
and other sensitive security information while maintaining proper error handling.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.core.logging import get_logger
from src.error_handling.base import ErrorHandlerBase
from src.error_handling.security_rate_limiter import (
    SecurityThreat,
    get_security_rate_limiter,
)
from src.error_handling.security_sanitizer import (
    SensitivityLevel,
    get_security_sanitizer,
)
from src.utils.error_categorization import (
    contains_keywords,
    detect_auth_token_error,
    is_sensitive_key,
)


class AuthenticationErrorHandler(ErrorHandlerBase):
    """
    Secure handler for authentication and authorization errors.

    Security features:
    - Prevents credential leakage in error messages
    - Implements progressive delays for brute force protection
    - Tracks and blocks suspicious authentication patterns
    - Sanitizes all authentication-related error information
    - Provides secure audit trail for security events
    - Rate limits authentication attempts per IP/user
    - Detects and responds to credential stuffing attacks
    """

    def __init__(self, next_handler: ErrorHandlerBase | None = None):
        super().__init__(next_handler)
        self._logger = get_logger(self.__class__.__module__)
        self.sanitizer = get_security_sanitizer()
        self.rate_limiter = get_security_rate_limiter()

        # Authentication attempt tracking
        self._failed_attempts: dict[str, list] = {}  # IP/User -> timestamps
        self._blocked_entities: dict[str, datetime] = {}  # IP/User -> block_until
        self._suspicious_patterns: dict[str, dict] = {}  # Pattern tracking

        # Progressive delay settings
        self.base_delay_seconds = Decimal("1.0")
        self.max_delay_seconds = Decimal("300.0")
        self.delay_multiplier = Decimal("2.0")

        # Blocking settings
        self.max_failed_attempts = 5
        self.block_duration_minutes = 30
        self.suspicious_threshold = 10  # Attempts to trigger pattern detection

        # Security monitoring
        self.credential_stuffing_threshold = 50  # Different passwords per IP
        self.account_enumeration_threshold = 20  # Different usernames per IP

        # Safe error messages
        self.safe_messages = {
            "invalid_credentials": "Invalid username or password.",
            "account_locked": "Account is temporarily locked. Please try again later.",
            "session_expired": "Your session has expired. Please log in again.",
            "insufficient_permissions": "You don't have permission to access this resource.",
            "token_invalid": "Authentication token is invalid or expired.",
            "mfa_required": "Multi-factor authentication is required.",
            "rate_limited": "Too many authentication attempts. Please wait before trying again.",
            "account_disabled": "Your account has been disabled. Please contact support.",
            "password_expired": "Your password has expired. Please update your password.",
            "ip_blocked": "Access from your location is temporarily restricted.",
        }

    def can_handle(self, error: Exception) -> bool:
        """Check if this is an authentication/authorization error."""
        error_type_name = type(error).__name__.lower()
        error_message = str(error).lower()

        # Authentication error types
        auth_error_types = [
            "authenticationerror",
            "authorizationerror",
            "permissionerror",
            "accessdeniederror",
            "tokenexpirederror",
            "invalidtokenerror",
            "sessionexpirederror",
            "credentialserror",
            "loginexception",
            "unauthorizederror",
            "forbiddenerror",
        ]

        # Authentication keywords in error messages
        auth_keywords = [
            "authentication",
            "authorization",
            "permission",
            "forbidden",
            "unauthorized",
            "access denied",
            "invalid credentials",
            "login failed",
            "token",
            "session",
            "password",
            "username",
            "api key",
            "secret",
            "credential",
            "jwt",
            "bearer",
            "oauth",
            "saml",
            "openid",
            "mfa",
            "2fa",
        ]

        # Check error type
        if any(auth_type in error_type_name for auth_type in auth_error_types):
            return True

        # Check error message
        if contains_keywords(error_message, auth_keywords):
            return True

        # Check HTTP status codes in error messages
        http_auth_codes = ["401", "403", "407"]
        if any(code in error_message for code in http_auth_codes):
            return True

        return False

    async def handle(
        self, error: Exception, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Handle authentication error with comprehensive security measures.

        Args:
            error: Authentication/authorization error
            context: Error context (will be sanitized)

        Returns:
            Secure recovery action dictionary
        """
        context = context or {}

        # Extract client information (sanitize first)
        client_ip = self._extract_client_ip(context)
        user_id = self._extract_user_id(context)
        self._extract_session_id(context)
        self._extract_user_agent(context)

        # Create sanitized context
        sanitized_context = self._sanitize_auth_context(context)

        # Determine error category
        error_category = self._categorize_auth_error(error)

        # Check if entity is blocked
        if self._is_entity_blocked(client_ip) or self._is_entity_blocked(user_id):
            return self._create_blocked_response(error_category)

        # Check rate limits
        rate_check = await self.rate_limiter.check_rate_limit(
            component="authentication", operation="auth_attempt", client_ip=client_ip
        )

        if not rate_check.allowed:
            return self._create_rate_limited_response(rate_check)

        # Record failed authentication attempt
        self._record_auth_failure(client_ip, user_id, error_category, sanitized_context)

        # Check for suspicious patterns
        threat_level = self._analyze_threat_patterns(client_ip, user_id)

        # Apply progressive delay
        delay = self._calculate_progressive_delay(client_ip, user_id)

        # Check if entity should be blocked
        self._check_and_apply_blocks(client_ip, user_id)

        # Create secure response
        response = self._create_secure_auth_response(
            error, error_category, delay, threat_level, sanitized_context
        )

        # Log security event
        await self._log_security_event(
            error, error_category, client_ip, user_id, threat_level, sanitized_context
        )

        # Record failure with rate limiter
        self.rate_limiter.record_failure(
            component="authentication",
            operation="auth_attempt",
            client_ip=client_ip,
            error_severity=(
                "high"
                if threat_level in [SecurityThreat.HIGH, SecurityThreat.CRITICAL]
                else "medium"
            ),
        )

        return response

    def _extract_client_ip(self, context: dict[str, Any]) -> str | None:
        """Extract and sanitize client IP address."""
        ip_keys = ["client_ip", "remote_addr", "x_forwarded_for", "x_real_ip", "user_ip"]

        for key in ip_keys:
            if key in context:
                ip = str(context[key])
                # Basic IP validation and sanitization
                if self._is_valid_ip(ip):
                    return ip

        return None

    def _extract_user_id(self, context: dict[str, Any]) -> str | None:
        """Extract user ID (sanitized)."""
        user_keys = ["user_id", "username", "email", "login", "account_id"]

        for key in user_keys:
            if context.get(key):
                # Return hash of user ID for privacy
                user_id = str(context[key])
                return self._hash_identifier(user_id)

        return None

    def _extract_session_id(self, context: dict[str, Any]) -> str | None:
        """Extract session ID (sanitized)."""
        session_keys = ["session_id", "session", "sid", "session_token"]

        for key in session_keys:
            if context.get(key):
                session_id = str(context[key])
                return self._hash_identifier(session_id)

        return None

    def _extract_user_agent(self, context: dict[str, Any]) -> str | None:
        """Extract and sanitize user agent."""
        ua_keys = ["user_agent", "http_user_agent", "user-agent"]

        for key in ua_keys:
            if context.get(key):
                user_agent = str(context[key])
                # Sanitize but keep basic browser info for pattern detection
                return self._sanitize_user_agent(user_agent)

        return None

    def _sanitize_auth_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """Comprehensively sanitize authentication context."""
        # Start with base sanitization
        sanitized = self.sanitizer.sanitize_context(context, SensitivityLevel.CRITICAL)

        # Remove or mask sensitive authentication data
        for key in list(sanitized.keys()):
            if is_sensitive_key(key):
                sanitized[key] = "[REDACTED_AUTH_DATA]"

        # Add security metadata
        sanitized["sanitized_at"] = datetime.now(timezone.utc).isoformat()
        sanitized["security_level"] = "critical"

        return sanitized

    def _categorize_auth_error(self, error: Exception) -> str:
        """Categorize authentication error for appropriate response."""
        error_message = str(error).lower()
        type(error).__name__.lower()

        # Invalid credentials
        if any(
            keyword in error_message
            for keyword in [
                "invalid credential",
                "wrong password",
                "incorrect password",
                "authentication failed",
                "login failed",
                "bad credentials",
            ]
        ):
            return "invalid_credentials"

        # Account locked/disabled
        if any(
            keyword in error_message
            for keyword in [
                "account locked",
                "user locked",
                "account disabled",
                "user disabled",
                "account suspended",
            ]
        ):
            return "account_locked"

        # Session/token issues
        if any(
            keyword in error_message
            for keyword in [
                "session expired",
                "token expired",
                "invalid token",
                "token invalid",
                "session invalid",
            ]
        ):
            return "session_expired"

        # Permission/authorization issues
        if any(
            keyword in error_message
            for keyword in [
                "permission denied",
                "access denied",
                "forbidden",
                "not authorized",
                "insufficient permission",
            ]
        ):
            return "insufficient_permissions"

        # MFA requirements
        if any(
            keyword in error_message
            for keyword in ["mfa", "2fa", "two factor", "multi factor", "verification code", "otp"]
        ):
            return "mfa_required"

        # Rate limiting
        if any(
            keyword in error_message
            for keyword in ["rate limit", "too many", "throttle", "quota exceeded"]
        ):
            return "rate_limited"

        # Generic token issues
        if detect_auth_token_error(error_message):
            return "token_invalid"

        return "invalid_credentials"  # Default safe response

    def _is_entity_blocked(self, entity: str | None) -> bool:
        """Check if entity (IP/User) is currently blocked."""
        if not entity or entity not in self._blocked_entities:
            return False

        block_until = self._blocked_entities[entity]
        if datetime.now(timezone.utc) < block_until:
            return True
        else:
            # Block expired, remove it
            del self._blocked_entities[entity]
            return False

    def _record_auth_failure(
        self,
        client_ip: str | None,
        user_id: str | None,
        error_category: str,
        context: dict[str, Any],
    ) -> None:
        """Record authentication failure for analysis."""
        current_time = datetime.now(timezone.utc)

        # Record IP-based failures
        if client_ip:
            if client_ip not in self._failed_attempts:
                self._failed_attempts[client_ip] = []
            self._failed_attempts[client_ip].append(current_time)

        # Record user-based failures
        if user_id:
            user_key = f"user:{user_id}"
            if user_key not in self._failed_attempts:
                self._failed_attempts[user_key] = []
            self._failed_attempts[user_key].append(current_time)

        # Clean up old attempts (keep last 24 hours)
        cutoff = current_time - timedelta(hours=24)
        self._cleanup_old_attempts(cutoff)

    def _analyze_threat_patterns(
        self, client_ip: str | None, user_id: str | None
    ) -> SecurityThreat:
        """Analyze authentication patterns for threats."""
        threat_level = SecurityThreat.LOW

        if client_ip:
            # Check for credential stuffing (many different passwords)
            ip_failures = len(self._failed_attempts.get(client_ip, []))
            if ip_failures > self.credential_stuffing_threshold:
                threat_level = SecurityThreat.HIGH
                self._logger.warning(
                    "Potential credential stuffing attack detected",
                    client_ip=client_ip,
                    failure_count=ip_failures,
                )

            # Check for account enumeration (many different usernames)
            elif ip_failures > self.account_enumeration_threshold:
                threat_level = SecurityThreat.MEDIUM
                self._logger.warning(
                    "Potential account enumeration detected",
                    client_ip=client_ip,
                    failure_count=ip_failures,
                )

        if user_id:
            # Check for brute force on specific account
            user_key = f"user:{user_id}"
            user_failures = len(self._failed_attempts.get(user_key, []))
            if user_failures > self.max_failed_attempts * 2:
                # Upgrade threat level if needed
                if threat_level == SecurityThreat.LOW or threat_level == SecurityThreat.MEDIUM:
                    threat_level = SecurityThreat.HIGH
                self._logger.warning(
                    "Brute force attack on user account detected",
                    user_id=user_id,
                    failure_count=user_failures,
                )

        return threat_level

    def _calculate_progressive_delay(self, client_ip: str | None, user_id: str | None) -> float:
        """Calculate progressive delay based on failure history."""
        max_failures = 0

        # Count IP failures
        if client_ip:
            ip_failures = len(self._failed_attempts.get(client_ip, []))
            max_failures = max(max_failures, ip_failures)

        # Count user failures
        if user_id:
            user_key = f"user:{user_id}"
            user_failures = len(self._failed_attempts.get(user_key, []))
            max_failures = max(max_failures, user_failures)

        # Calculate exponential delay
        delay = min(
            self.base_delay_seconds * (self.delay_multiplier**max_failures), self.max_delay_seconds
        )

        return delay

    def _check_and_apply_blocks(self, client_ip: str | None, user_id: str | None) -> None:
        """Check if entities should be blocked and apply blocks."""
        block_until = datetime.now(timezone.utc) + timedelta(minutes=self.block_duration_minutes)

        # Check IP blocking
        if client_ip:
            ip_failures = len(self._failed_attempts.get(client_ip, []))
            if ip_failures >= self.max_failed_attempts:
                self._blocked_entities[client_ip] = block_until
                self._logger.warning(
                    "IP address blocked due to excessive failures",
                    client_ip=client_ip,
                    failure_count=ip_failures,
                    block_duration_minutes=self.block_duration_minutes,
                )

        # Check user blocking
        if user_id:
            user_key = f"user:{user_id}"
            user_failures = len(self._failed_attempts.get(user_key, []))
            if user_failures >= self.max_failed_attempts:
                self._blocked_entities[user_key] = block_until
                self._logger.warning(
                    "User account blocked due to excessive failures",
                    user_id=user_id,
                    failure_count=user_failures,
                    block_duration_minutes=self.block_duration_minutes,
                )

    def _create_blocked_response(self, error_category: str) -> dict[str, Any]:
        """Create response for blocked entities."""
        return {
            "action": "block",
            "reason": "authentication_blocked",
            "user_message": self.safe_messages["ip_blocked"],
            "retry_after": self.block_duration_minutes * 60,
            "security_action": "blocked_entity",
            "recoverable": False,
            "error_category": error_category,
        }

    def _create_rate_limited_response(self, rate_check) -> dict[str, Any]:
        """Create response for rate limited requests."""
        return {
            "action": "wait",
            "reason": "authentication_rate_limited",
            "user_message": self.safe_messages["rate_limited"],
            "retry_after": rate_check.suggested_retry_after,
            "security_action": "rate_limited",
            "recoverable": True,
            "threat_level": rate_check.threat_level,
        }

    def _create_secure_auth_response(
        self,
        error: Exception,
        error_category: str,
        delay: float,
        threat_level: SecurityThreat,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Create secure authentication error response."""
        # Get safe user message
        user_message = self.safe_messages.get(
            error_category, self.safe_messages["invalid_credentials"]
        )

        # Create sanitized technical message
        sanitized_error = self.sanitizer.sanitize_error_message(
            str(error), SensitivityLevel.CRITICAL
        )

        response = {
            "action": "reject",
            "reason": f"authentication_failed_{error_category}",
            "user_message": user_message,
            "technical_message": f"Authentication error: {sanitized_error}",
            "delay": str(delay),
            "security_action": "progressive_delay",
            "recoverable": error_category not in ["account_locked", "account_disabled"],
            "error_category": error_category,
            "threat_level": threat_level,
            "retry_recommendations": self._get_retry_recommendations(error_category),
            "security_headers": self._get_security_headers(),
        }

        # Add context if not too sensitive
        if error_category not in ["invalid_credentials"]:
            response["context"] = {
                key: value
                for key, value in context.items()
                if key not in ["password", "token", "credential", "secret"]
            }

        return response

    def _get_retry_recommendations(self, error_category: str) -> dict[str, Any]:
        """Get retry recommendations based on error category."""
        recommendations = {
            "invalid_credentials": {
                "should_retry": True,
                "max_retries": 3,
                "suggested_actions": ["Check username and password", "Reset password if forgotten"],
            },
            "account_locked": {
                "should_retry": False,
                "suggested_actions": ["Wait for unlock", "Contact support"],
            },
            "session_expired": {
                "should_retry": True,
                "max_retries": 1,
                "suggested_actions": ["Re-authenticate", "Refresh session"],
            },
            "insufficient_permissions": {
                "should_retry": False,
                "suggested_actions": ["Contact administrator", "Request access"],
            },
            "token_invalid": {
                "should_retry": True,
                "max_retries": 1,
                "suggested_actions": ["Refresh token", "Re-authenticate"],
            },
            "mfa_required": {
                "should_retry": True,
                "max_retries": 3,
                "suggested_actions": ["Provide MFA code", "Check authenticator app"],
            },
        }

        return recommendations.get(
            error_category,
            {"should_retry": True, "max_retries": 1, "suggested_actions": ["Try again later"]},
        )

    def _get_security_headers(self) -> dict[str, str]:
        """Get security headers for response."""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        }

    async def _log_security_event(
        self,
        error: Exception,
        error_category: str,
        client_ip: str | None,
        user_id: str | None,
        threat_level: SecurityThreat,
        context: dict[str, Any],
    ) -> None:
        """Log security event for audit trail."""
        security_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "authentication_failure",
            "error_category": error_category,
            "threat_level": threat_level,
            "client_ip": client_ip,
            "user_id": user_id,
            "error_type": type(error).__name__,
            "sanitized_context": context,
        }

        # Log with appropriate level based on threat
        if threat_level in [SecurityThreat.HIGH, SecurityThreat.CRITICAL]:
            self._logger.critical("High-threat authentication failure", **security_event)
        elif threat_level == SecurityThreat.MEDIUM:
            self._logger.warning("Medium-threat authentication failure", **security_event)
        else:
            self._logger.info("Authentication failure", **security_event)

    def _cleanup_old_attempts(self, cutoff: datetime) -> None:
        """Clean up old authentication attempts."""
        for entity in list(self._failed_attempts.keys()):
            # Filter out attempts older than cutoff
            self._failed_attempts[entity] = [
                timestamp for timestamp in self._failed_attempts[entity] if timestamp > cutoff
            ]

            # Remove empty lists
            if not self._failed_attempts[entity]:
                del self._failed_attempts[entity]

    def _is_valid_ip(self, ip: str) -> bool:
        """Basic IP address validation."""
        try:
            import ipaddress

            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    def _hash_identifier(self, identifier: str) -> str:
        """Create hash of identifier for privacy."""
        hash_obj = hashlib.sha256(identifier.encode("utf-8"))
        return f"HASH_{hash_obj.hexdigest()[:16]}"

    def _sanitize_user_agent(self, user_agent: str) -> str:
        """Sanitize user agent while preserving browser info."""
        # Keep basic browser/OS info but remove detailed version/system info
        import re

        # Extract basic browser info
        browser_patterns = [
            (r"Chrome/[\d.]+", "Chrome"),
            (r"Firefox/[\d.]+", "Firefox"),
            (r"Safari/[\d.]+", "Safari"),
            (r"Edge/[\d.]+", "Edge"),
            (r"Opera/[\d.]+", "Opera"),
        ]

        for pattern, replacement in browser_patterns:
            user_agent = re.sub(pattern, replacement, user_agent, flags=re.IGNORECASE)

        # Remove detailed system information
        user_agent = re.sub(r"\([^)]*\)", "(System)", user_agent)

        # Limit length
        return user_agent[:100] if len(user_agent) > 100 else user_agent

    def get_security_stats(self) -> dict[str, Any]:
        """Get authentication security statistics."""
        current_time = datetime.now(timezone.utc)

        # Count recent failures (last hour)
        recent_cutoff = current_time - timedelta(hours=1)
        recent_failures = 0

        for attempts in self._failed_attempts.values():
            recent_failures += len([t for t in attempts if t > recent_cutoff])

        return {
            "total_blocked_entities": len(self._blocked_entities),
            "total_failed_attempts_tracked": sum(
                len(attempts) for attempts in self._failed_attempts.values()
            ),
            "recent_failures_last_hour": recent_failures,
            "unique_entities_with_failures": len(self._failed_attempts),
            "active_blocks": len(
                [
                    entity
                    for entity, block_until in self._blocked_entities.items()
                    if block_until > current_time
                ]
            ),
            "authentication_rules": {
                "max_failed_attempts": self.max_failed_attempts,
                "block_duration_minutes": self.block_duration_minutes,
                "max_delay_seconds": self.max_delay_seconds,
            },
        }

    def reset_entity_failures(self, entity: str) -> bool:
        """Reset failures for entity (admin function)."""
        reset_count = 0

        # Reset attempt history
        if entity in self._failed_attempts:
            reset_count += len(self._failed_attempts[entity])
            del self._failed_attempts[entity]

        # Remove blocks
        if entity in self._blocked_entities:
            del self._blocked_entities[entity]
            reset_count += 1

        if reset_count > 0:
            self._logger.info(f"Reset authentication failures for entity: {entity}")
            return True

        return False
