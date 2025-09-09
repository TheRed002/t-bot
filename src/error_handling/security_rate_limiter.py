"""
Security rate limiter for error handling.

Redirects to simplified security validator.
"""

from dataclasses import dataclass

from .constants import (
    DEFAULT_BURST_ALLOWANCE,
    DEFAULT_REQUESTS_PER_HOUR,
    DEFAULT_REQUESTS_PER_MINUTE,
    DEFAULT_REQUESTS_PER_SECOND,
)
from .security_validator import (
    SecurityRateLimiter,
    get_security_rate_limiter,
)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests_per_second: int = DEFAULT_REQUESTS_PER_SECOND
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE
    requests_per_hour: int = DEFAULT_REQUESTS_PER_HOUR
    burst_allowance: int = DEFAULT_BURST_ALLOWANCE


def record_recovery_failure(component: str, operation: str, error_severity: str, **kwargs) -> None:
    """Record recovery failure - no-op in simplified version."""
    # Simplified implementation does not track recovery failures
    return


class SecurityThreat:
    """Security threat stub for backward compatibility."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


__all__ = [
    "RateLimitConfig",
    "SecurityRateLimiter",
    "SecurityThreat",
    "get_security_rate_limiter",
    "record_recovery_failure",
]
