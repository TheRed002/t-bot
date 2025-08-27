"""
Security-focused rate limiting for error recovery operations.

This module provides global rate limiting for error recovery attempts to prevent
resource exhaustion attacks and cascading failures in trading systems.

CRITICAL: Prevents denial of service through excessive error recovery attempts
and protects against resource exhaustion attacks targeting the error handling system.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.core.logging import get_logger


class RateLimitType(Enum):
    """Types of rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"  # Fixed time window
    SLIDING_WINDOW = "sliding_window"  # Sliding time window
    TOKEN_BUCKET = "token_bucket"  # Token bucket algorithm
    ADAPTIVE = "adaptive"  # Adaptive rate limiting


class SecurityThreat(Enum):
    """Security threat levels for rate limiting."""

    LOW = "low"  # Normal operations
    MEDIUM = "medium"  # Suspicious activity
    HIGH = "high"  # Likely attack
    CRITICAL = "critical"  # Active attack


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    # Basic rate limiting
    max_attempts_per_minute: int = 60
    max_attempts_per_hour: int = 300
    max_attempts_per_day: int = 1000

    # Window settings
    window_size_seconds: int = 60
    sliding_window_segments: int = 6  # For sliding window

    # Token bucket settings
    bucket_capacity: int = 100
    refill_rate_per_second: float = 1.0

    # Adaptive settings
    enable_adaptive_limiting: bool = True
    adaptive_threshold_multiplier: float = 1.5
    adaptive_cooldown_seconds: int = 300

    # Security settings
    enable_threat_detection: bool = True
    max_failures_before_block: int = 10
    block_duration_seconds: int = 3600  # 1 hour

    # Component-specific limits
    component_limits: dict[str, dict[str, int]] = field(default_factory=dict)

    # IP-based limiting (for API endpoints)
    enable_ip_limiting: bool = True
    max_attempts_per_ip_per_minute: int = 30

    # Emergency limits
    emergency_max_global_rate: int = 10  # Per second globally
    emergency_trigger_threshold: int = 1000  # Errors per minute


@dataclass
class RateLimitResult:
    """Result of rate limit check."""

    allowed: bool
    remaining_attempts: int
    reset_time: datetime
    threat_level: SecurityThreat
    reason: str = ""
    suggested_retry_after: float = 0.0


@dataclass
class SecurityMetrics:
    """Security metrics for rate limiting."""

    total_attempts: int = 0
    blocked_attempts: int = 0
    threat_detections: int = 0
    emergency_activations: int = 0
    adaptive_adjustments: int = 0
    component_blocks: dict[str, int] = field(default_factory=dict)
    ip_blocks: dict[str, int] = field(default_factory=dict)


class SecurityRateLimiter:
    """
    Global rate limiter for error recovery operations with security focus.

    Provides multiple layers of protection:
    - Global rate limiting across all components
    - Component-specific rate limiting
    - IP-based rate limiting
    - Adaptive rate limiting based on threat level
    - Emergency throttling during attacks
    - Token bucket algorithm for burst handling
    - Sliding window for smooth rate limiting
    """

    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self.logger = get_logger(self.__class__.__module__)

        # Rate limiting state
        self._global_attempts: deque[float] = deque()  # Global attempt timestamps
        self._component_attempts: dict[str, deque[float]] = defaultdict(
            deque
        )  # Per-component attempts
        self._ip_attempts: dict[str, deque[float]] = defaultdict(deque)  # Per-IP attempts
        self._blocked_components: dict[str, float] = {}  # Component -> block_until_time
        self._blocked_ips: dict[str, float] = {}  # IP -> block_until_time

        # Token bucket state
        self._token_bucket = self.config.bucket_capacity
        self._last_refill_time = time.time()

        # Adaptive limiting state
        self._threat_level = SecurityThreat.LOW
        self._adaptive_multiplier = 1.0
        self._last_threat_assessment = time.time()

        # Emergency state
        self._emergency_mode = False
        self._emergency_start_time: float | None = None

        # Security metrics
        self._metrics = SecurityMetrics()

        # Cleanup task
        self._cleanup_task: asyncio.Task[None] | None = None
        # Don't start cleanup task immediately - start it when first rate limit check happens

    def _start_cleanup_task(self) -> None:
        """Start periodic cleanup of old data."""
        try:
            if self._cleanup_task is None or self._cleanup_task.done():
                # Create and store the task for proper lifecycle management
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
                # Add done callback to handle any exceptions
                self._cleanup_task.add_done_callback(self._cleanup_task_done_callback)
        except RuntimeError:
            # No event loop running, cleanup task will be started on first use
            pass

    def _cleanup_task_done_callback(self, task: asyncio.Task) -> None:
        """Handle cleanup task completion."""
        if task.exception():
            self.logger.error(f"Cleanup task failed: {task.exception()}")
        # Reset task reference so it can be restarted if needed
        if self._cleanup_task is task:
            self._cleanup_task = None

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of expired data."""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                current_time = time.time()

                # Clean up old attempt records
                self._cleanup_old_attempts(current_time)

                # Clean up expired blocks
                self._cleanup_expired_blocks(current_time)

                # Update threat assessment
                self._update_threat_level()

                # Check for emergency conditions
                self._check_emergency_conditions()

            except Exception as e:
                self.logger.error(f"Error in rate limiter cleanup: {e}")
                await asyncio.sleep(60)

    async def check_rate_limit(
        self,
        component: str,
        operation: str = "recovery",
        client_ip: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> RateLimitResult:
        """
        Check if operation should be rate limited.

        Args:
            component: Component requesting the operation
            operation: Type of operation (recovery, retry, etc.)
            client_ip: Client IP address if applicable
            context: Additional context for rate limiting decisions

        Returns:
            RateLimitResult indicating if operation is allowed
        """
        current_time = time.time()

        # Start cleanup task if not already running
        self._start_cleanup_task()

        # Update token bucket
        self._refill_token_bucket(current_time)

        # Check if component is blocked
        if component in self._blocked_components:
            block_until = self._blocked_components[component]
            if current_time < block_until:
                remaining_time = block_until - current_time
                return RateLimitResult(
                    allowed=False,
                    remaining_attempts=0,
                    reset_time=datetime.fromtimestamp(block_until, tz=timezone.utc),
                    threat_level=self._threat_level,
                    reason=f"Component {component} is blocked",
                    suggested_retry_after=remaining_time,
                )

        # Check if IP is blocked
        if client_ip and client_ip in self._blocked_ips:
            block_until = self._blocked_ips[client_ip]
            if current_time < block_until:
                remaining_time = block_until - current_time
                return RateLimitResult(
                    allowed=False,
                    remaining_attempts=0,
                    reset_time=datetime.fromtimestamp(block_until, tz=timezone.utc),
                    threat_level=self._threat_level,
                    reason=f"IP {client_ip} is blocked",
                    suggested_retry_after=remaining_time,
                )

        # Check emergency mode
        if self._emergency_mode:
            emergency_limit = self.config.emergency_max_global_rate
            recent_global = len(
                [t for t in self._global_attempts if current_time - t < 1.0]  # Last second
            )

            if recent_global >= emergency_limit:
                return RateLimitResult(
                    allowed=False,
                    remaining_attempts=0,
                    reset_time=datetime.fromtimestamp(current_time + 60, tz=timezone.utc),
                    threat_level=SecurityThreat.CRITICAL,
                    reason="Emergency rate limiting active",
                    suggested_retry_after=60.0,
                )

        # Check token bucket (for burst handling)
        if self._token_bucket < 1:
            time_to_token = (1.0 - self._token_bucket) / self.config.refill_rate_per_second
            return RateLimitResult(
                allowed=False,
                remaining_attempts=0,
                reset_time=datetime.fromtimestamp(current_time + time_to_token, tz=timezone.utc),
                threat_level=self._threat_level,
                reason="Token bucket exhausted",
                suggested_retry_after=time_to_token,
            )

        # Check global rate limits
        global_result = self._check_global_limits(current_time)
        if not global_result.allowed:
            return global_result

        # Check component-specific limits
        component_result = self._check_component_limits(component, current_time)
        if not component_result.allowed:
            return component_result

        # Check IP-specific limits
        if client_ip:
            ip_result = self._check_ip_limits(client_ip, current_time)
            if not ip_result.allowed:
                return ip_result

        # Check adaptive limits based on threat level
        adaptive_result = self._check_adaptive_limits(component, current_time)
        if not adaptive_result.allowed:
            return adaptive_result

        # All checks passed - allow the operation
        self._record_attempt(component, client_ip, current_time)
        self._consume_token()

        return RateLimitResult(
            allowed=True,
            remaining_attempts=self._calculate_remaining_attempts(component),
            reset_time=self._calculate_reset_time(current_time),
            threat_level=self._threat_level,
            reason="Rate limit check passed",
        )

    def record_failure(
        self,
        component: str,
        operation: str = "recovery",
        client_ip: str | None = None,
        error_severity: str = "medium",
    ) -> None:
        """
        Record a failed operation for threat assessment.

        Args:
            component: Component that failed
            operation: Type of operation that failed
            client_ip: Client IP if applicable
            error_severity: Severity of the error
        """
        current_time = time.time()

        # Update failure counts
        if component not in self._metrics.component_blocks:
            self._metrics.component_blocks[component] = 0
        self._metrics.component_blocks[component] += 1

        if client_ip:
            if client_ip not in self._metrics.ip_blocks:
                self._metrics.ip_blocks[client_ip] = 0
            self._metrics.ip_blocks[client_ip] += 1

        # Check if component should be blocked
        component_failures = self._metrics.component_blocks[component]
        if component_failures >= self.config.max_failures_before_block:
            block_until = current_time + self.config.block_duration_seconds
            self._blocked_components[component] = block_until

            self.logger.warning(
                f"Component {component} blocked due to excessive failures",
                failures=component_failures,
                block_duration=self.config.block_duration_seconds,
            )

        # Check if IP should be blocked
        if client_ip:
            ip_failures = self._metrics.ip_blocks.get(client_ip, 0)
            if ip_failures >= self.config.max_failures_before_block:
                block_until = current_time + self.config.block_duration_seconds
                self._blocked_ips[client_ip] = block_until

                self.logger.warning(
                    f"IP {client_ip} blocked due to excessive failures",
                    failures=ip_failures,
                    block_duration=self.config.block_duration_seconds,
                )

        # Update threat level based on severity
        if error_severity in ["critical", "high"]:
            self._escalate_threat_level()

    def _check_global_limits(self, current_time: float) -> RateLimitResult:
        """Check global rate limits."""
        minute_window = current_time - 60
        hour_window = current_time - 3600
        day_window = current_time - 86400

        # Count attempts in each window
        minute_attempts = len([t for t in self._global_attempts if t > minute_window])
        hour_attempts = len([t for t in self._global_attempts if t > hour_window])
        day_attempts = len([t for t in self._global_attempts if t > day_window])

        # Apply adaptive multiplier
        max_per_minute = int(self.config.max_attempts_per_minute / self._adaptive_multiplier)
        max_per_hour = int(self.config.max_attempts_per_hour / self._adaptive_multiplier)
        max_per_day = int(self.config.max_attempts_per_day / self._adaptive_multiplier)

        if minute_attempts >= max_per_minute:
            return RateLimitResult(
                allowed=False,
                remaining_attempts=max_per_minute - minute_attempts,
                reset_time=datetime.fromtimestamp(current_time + 60, tz=timezone.utc),
                threat_level=self._threat_level,
                reason="Global per-minute limit exceeded",
                suggested_retry_after=60.0,
            )

        if hour_attempts >= max_per_hour:
            return RateLimitResult(
                allowed=False,
                remaining_attempts=max_per_hour - hour_attempts,
                reset_time=datetime.fromtimestamp(current_time + 3600, tz=timezone.utc),
                threat_level=self._threat_level,
                reason="Global per-hour limit exceeded",
                suggested_retry_after=3600.0,
            )

        if day_attempts >= max_per_day:
            return RateLimitResult(
                allowed=False,
                remaining_attempts=max_per_day - day_attempts,
                reset_time=datetime.fromtimestamp(current_time + 86400, tz=timezone.utc),
                threat_level=self._threat_level,
                reason="Global per-day limit exceeded",
                suggested_retry_after=86400.0,
            )

        return RateLimitResult(
            allowed=True,
            remaining_attempts=min(
                max_per_minute - minute_attempts,
                max_per_hour - hour_attempts,
                max_per_day - day_attempts,
            ),
            reset_time=datetime.fromtimestamp(current_time + 60, tz=timezone.utc),
            threat_level=self._threat_level,
        )

    def _check_component_limits(self, component: str, current_time: float) -> RateLimitResult:
        """Check component-specific rate limits."""
        # Get component-specific limits or use defaults
        component_config = self.config.component_limits.get(component, {})
        max_per_minute = component_config.get(
            "max_per_minute", self.config.max_attempts_per_minute // 2
        )
        max_per_hour = component_config.get("max_per_hour", self.config.max_attempts_per_hour // 2)

        # Apply adaptive multiplier
        max_per_minute = int(max_per_minute / self._adaptive_multiplier)
        max_per_hour = int(max_per_hour / self._adaptive_multiplier)

        minute_window = current_time - 60
        hour_window = current_time - 3600

        component_attempts = self._component_attempts[component]
        minute_attempts = len([t for t in component_attempts if t > minute_window])
        hour_attempts = len([t for t in component_attempts if t > hour_window])

        if minute_attempts >= max_per_minute:
            return RateLimitResult(
                allowed=False,
                remaining_attempts=max_per_minute - minute_attempts,
                reset_time=datetime.fromtimestamp(current_time + 60, tz=timezone.utc),
                threat_level=self._threat_level,
                reason=f"Component {component} per-minute limit exceeded",
                suggested_retry_after=60.0,
            )

        if hour_attempts >= max_per_hour:
            return RateLimitResult(
                allowed=False,
                remaining_attempts=max_per_hour - hour_attempts,
                reset_time=datetime.fromtimestamp(current_time + 3600, tz=timezone.utc),
                threat_level=self._threat_level,
                reason=f"Component {component} per-hour limit exceeded",
                suggested_retry_after=3600.0,
            )

        return RateLimitResult(
            allowed=True,
            remaining_attempts=min(max_per_minute - minute_attempts, max_per_hour - hour_attempts),
            reset_time=datetime.fromtimestamp(current_time + 60, tz=timezone.utc),
            threat_level=self._threat_level,
        )

    def _check_ip_limits(self, client_ip: str, current_time: float) -> RateLimitResult:
        """Check IP-specific rate limits."""
        if not self.config.enable_ip_limiting:
            return RateLimitResult(
                allowed=True,
                remaining_attempts=1000,
                reset_time=datetime.fromtimestamp(current_time + 60, tz=timezone.utc),
                threat_level=self._threat_level,
            )

        minute_window = current_time - 60
        ip_attempts = self._ip_attempts[client_ip]
        minute_attempts = len([t for t in ip_attempts if t > minute_window])

        max_per_minute = int(self.config.max_attempts_per_ip_per_minute / self._adaptive_multiplier)

        if minute_attempts >= max_per_minute:
            return RateLimitResult(
                allowed=False,
                remaining_attempts=max_per_minute - minute_attempts,
                reset_time=datetime.fromtimestamp(current_time + 60, tz=timezone.utc),
                threat_level=self._threat_level,
                reason=f"IP {client_ip} per-minute limit exceeded",
                suggested_retry_after=60.0,
            )

        return RateLimitResult(
            allowed=True,
            remaining_attempts=max_per_minute - minute_attempts,
            reset_time=datetime.fromtimestamp(current_time + 60, tz=timezone.utc),
            threat_level=self._threat_level,
        )

    def _check_adaptive_limits(self, component: str, current_time: float) -> RateLimitResult:
        """Check adaptive rate limits based on threat level."""
        if not self.config.enable_adaptive_limiting:
            return RateLimitResult(
                allowed=True,
                remaining_attempts=1000,
                reset_time=datetime.fromtimestamp(current_time + 60, tz=timezone.utc),
                threat_level=self._threat_level,
            )

        # Adaptive limiting based on threat level
        if self._threat_level == SecurityThreat.CRITICAL:
            # Very restrictive limits during critical threats
            if self._adaptive_multiplier < 5.0:
                self._adaptive_multiplier = min(5.0, self._adaptive_multiplier * 1.5)
                self._metrics.adaptive_adjustments += 1

        elif self._threat_level == SecurityThreat.HIGH:
            # Moderate restrictions during high threats
            if self._adaptive_multiplier < 3.0:
                self._adaptive_multiplier = min(3.0, self._adaptive_multiplier * 1.2)
                self._metrics.adaptive_adjustments += 1

        elif self._threat_level == SecurityThreat.MEDIUM:
            # Light restrictions during medium threats
            if self._adaptive_multiplier < 2.0:
                self._adaptive_multiplier = min(2.0, self._adaptive_multiplier * 1.1)
                self._metrics.adaptive_adjustments += 1
        else:
            # Gradually reduce restrictions during low threats
            if self._adaptive_multiplier > 1.0:
                self._adaptive_multiplier = max(1.0, self._adaptive_multiplier * 0.95)

        return RateLimitResult(
            allowed=True,
            remaining_attempts=1000,
            reset_time=datetime.fromtimestamp(current_time + 60, tz=timezone.utc),
            threat_level=self._threat_level,
        )

    def _refill_token_bucket(self, current_time: float) -> None:
        """Refill the token bucket based on elapsed time."""
        time_elapsed = current_time - self._last_refill_time
        tokens_to_add = time_elapsed * self.config.refill_rate_per_second

        self._token_bucket = min(
            self.config.bucket_capacity, int(self._token_bucket + tokens_to_add)
        )
        self._last_refill_time = current_time

    def _consume_token(self) -> None:
        """Consume a token from the bucket."""
        self._token_bucket = max(0, self._token_bucket - 1)

    def _record_attempt(self, component: str, client_ip: str | None, current_time: float) -> None:
        """Record a rate limit attempt."""
        self._global_attempts.append(current_time)
        self._component_attempts[component].append(current_time)

        if client_ip:
            self._ip_attempts[client_ip].append(current_time)

        self._metrics.total_attempts += 1

    def _cleanup_old_attempts(self, current_time: float) -> None:
        """Clean up old attempt records."""
        day_window = current_time - 86400

        # Clean global attempts
        while self._global_attempts and self._global_attempts[0] < day_window:
            self._global_attempts.popleft()

        # Clean component attempts
        for component_attempts in self._component_attempts.values():
            while component_attempts and component_attempts[0] < day_window:
                component_attempts.popleft()

        # Clean IP attempts
        for ip_attempts in self._ip_attempts.values():
            while ip_attempts and ip_attempts[0] < day_window:
                ip_attempts.popleft()

    def _cleanup_expired_blocks(self, current_time: float) -> None:
        """Clean up expired component and IP blocks."""
        expired_components = [
            comp for comp, until in self._blocked_components.items() if current_time >= until
        ]
        for comp in expired_components:
            del self._blocked_components[comp]
            self.logger.info(f"Unblocked component: {comp}")

        expired_ips = [ip for ip, until in self._blocked_ips.items() if current_time >= until]
        for ip in expired_ips:
            del self._blocked_ips[ip]
            self.logger.info(f"Unblocked IP: {ip}")

    def _update_threat_level(self) -> None:
        """Update threat level based on current activity."""
        current_time = time.time()

        # Analyze recent activity patterns
        recent_window = current_time - 300  # Last 5 minutes
        recent_attempts = len([t for t in self._global_attempts if t > recent_window])

        # Calculate threat indicators
        blocked_components = len(self._blocked_components)
        blocked_ips = len(self._blocked_ips)
        failure_rate = sum(self._metrics.component_blocks.values()) / max(
            self._metrics.total_attempts, 1
        )

        # Determine threat level
        old_level = self._threat_level

        if (
            recent_attempts > 200
            or blocked_components > 5
            or blocked_ips > 10
            or failure_rate > 0.5
        ):
            self._threat_level = SecurityThreat.CRITICAL
        elif (
            recent_attempts > 100 or blocked_components > 2 or blocked_ips > 5 or failure_rate > 0.3
        ):
            self._threat_level = SecurityThreat.HIGH
        elif (
            recent_attempts > 50 or blocked_components > 1 or blocked_ips > 2 or failure_rate > 0.1
        ):
            self._threat_level = SecurityThreat.MEDIUM
        else:
            self._threat_level = SecurityThreat.LOW

        if self._threat_level != old_level:
            self._metrics.threat_detections += 1
            self.logger.warning(
                f"Threat level changed from {old_level.value} to {self._threat_level.value}",
                recent_attempts=recent_attempts,
                blocked_components=blocked_components,
                blocked_ips=blocked_ips,
                failure_rate=failure_rate,
            )

    def _check_emergency_conditions(self) -> None:
        """Check if emergency rate limiting should be activated."""
        current_time = time.time()
        minute_window = current_time - 60

        recent_attempts = len([t for t in self._global_attempts if t > minute_window])

        should_activate = recent_attempts > self.config.emergency_trigger_threshold

        if should_activate and not self._emergency_mode:
            self._emergency_mode = True
            self._emergency_start_time = current_time
            self._metrics.emergency_activations += 1

            self.logger.critical(
                "Emergency rate limiting activated",
                recent_attempts=recent_attempts,
                threshold=self.config.emergency_trigger_threshold,
            )

        elif not should_activate and self._emergency_mode:
            # Deactivate emergency mode after cooldown
            if self._emergency_start_time and (current_time - self._emergency_start_time) > 300:
                self._emergency_mode = False
                self._emergency_start_time = None

                self.logger.info("Emergency rate limiting deactivated")

    def _escalate_threat_level(self) -> None:
        """Escalate threat level due to critical/high severity errors."""
        if self._threat_level == SecurityThreat.LOW:
            self._threat_level = SecurityThreat.MEDIUM
        elif self._threat_level == SecurityThreat.MEDIUM:
            self._threat_level = SecurityThreat.HIGH
        elif self._threat_level == SecurityThreat.HIGH:
            self._threat_level = SecurityThreat.CRITICAL

        self._metrics.threat_detections += 1

        # Update adaptive multiplier based on new threat level
        if self._threat_level == SecurityThreat.CRITICAL:
            self._adaptive_multiplier = min(5.0, max(self._adaptive_multiplier, 3.0))
        elif self._threat_level == SecurityThreat.HIGH:
            self._adaptive_multiplier = min(3.0, max(self._adaptive_multiplier, 2.0))
        elif self._threat_level == SecurityThreat.MEDIUM:
            self._adaptive_multiplier = min(2.0, max(self._adaptive_multiplier, 1.5))
        self._metrics.adaptive_adjustments += 1

    def _calculate_remaining_attempts(self, component: str) -> int:
        """Calculate remaining attempts for component."""
        current_time = time.time()
        minute_window = current_time - 60

        component_config = self.config.component_limits.get(component, {})
        max_per_minute = component_config.get(
            "max_per_minute", self.config.max_attempts_per_minute // 2
        )
        max_per_minute = int(max_per_minute / self._adaptive_multiplier)

        component_attempts = self._component_attempts[component]
        minute_attempts = len([t for t in component_attempts if t > minute_window])

        return max(0, max_per_minute - minute_attempts)

    def _calculate_reset_time(self, current_time: float) -> datetime:
        """Calculate when rate limits reset."""
        # Return next minute boundary
        next_minute = int(current_time) + (60 - (int(current_time) % 60))
        return datetime.fromtimestamp(next_minute, tz=timezone.utc)

    def get_status(self) -> dict[str, Any]:
        """Get current rate limiter status."""
        current_time = time.time()

        return {
            "threat_level": self._threat_level.value,
            "emergency_mode": self._emergency_mode,
            "adaptive_multiplier": self._adaptive_multiplier,
            "token_bucket_level": self._token_bucket,
            "blocked_components": len(self._blocked_components),
            "blocked_ips": len(self._blocked_ips),
            "recent_attempts": len([t for t in self._global_attempts if current_time - t < 60]),
            "metrics": {
                "total_attempts": self._metrics.total_attempts,
                "blocked_attempts": self._metrics.blocked_attempts,
                "threat_detections": self._metrics.threat_detections,
                "emergency_activations": self._metrics.emergency_activations,
                "adaptive_adjustments": self._metrics.adaptive_adjustments,
            },
        }

    def reset_component_blocks(self, component: str) -> bool:
        """Reset blocks for a specific component (admin function)."""
        if component in self._blocked_components:
            del self._blocked_components[component]
            if component in self._metrics.component_blocks:
                self._metrics.component_blocks[component] = 0
            self.logger.info(f"Manually reset blocks for component: {component}")
            return True
        return False

    def reset_ip_blocks(self, ip: str) -> bool:
        """Reset blocks for a specific IP (admin function)."""
        if ip in self._blocked_ips:
            del self._blocked_ips[ip]
            if ip in self._metrics.ip_blocks:
                self._metrics.ip_blocks[ip] = 0
            self.logger.info(f"Manually reset blocks for IP: {ip}")
            return True
        return False


# Global rate limiter instance
_global_rate_limiter = None


def get_security_rate_limiter(config: RateLimitConfig | None = None) -> SecurityRateLimiter:
    """Get global security rate limiter instance."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = SecurityRateLimiter(config)
    return _global_rate_limiter


async def check_recovery_rate_limit(
    component: str, operation: str = "recovery", client_ip: str | None = None
) -> RateLimitResult:
    """Convenience function for checking recovery operation rate limits."""
    limiter = get_security_rate_limiter()
    return await limiter.check_rate_limit(component, operation, client_ip)


def record_recovery_failure(
    component: str,
    operation: str = "recovery",
    client_ip: str | None = None,
    error_severity: str = "medium",
) -> None:
    """Convenience function for recording recovery operation failures."""
    limiter = get_security_rate_limiter()
    limiter.record_failure(component, operation, client_ip, error_severity)
