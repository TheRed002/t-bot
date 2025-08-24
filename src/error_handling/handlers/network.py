"""Network-specific error handlers with secure data sanitization."""

import asyncio
from typing import Any

from src.error_handling.base import ErrorHandlerBase
from src.error_handling.security_sanitizer import (
    SensitivityLevel,
    get_security_sanitizer,
)


class NetworkErrorHandler(ErrorHandlerBase):
    """Handler for network-related errors."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        next_handler: ErrorHandlerBase | None = None,
    ):
        """
        Initialize network error handler with secure sanitization.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries (exponential backoff)
            next_handler: Next handler in chain
        """
        super().__init__(next_handler)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.sanitizer = get_security_sanitizer()

    def can_handle(self, error: Exception) -> bool:
        """Check if this is a network error."""
        network_errors = (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            OSError,  # Can indicate network issues
        )

        # Check for specific error messages
        error_msg = str(error).lower()
        network_keywords = [
            "connection",
            "timeout",
            "network",
            "socket",
            "refused",
            "reset",
            "broken pipe",
            "unreachable",
        ]

        return isinstance(error, network_errors) or any(
            keyword in error_msg for keyword in network_keywords
        )

    async def handle(
        self, error: Exception, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Handle network error with retry strategy.

        Args:
            error: The network error
            context: Optional context

        Returns:
            Recovery action dictionary
        """
        context = context or {}
        retry_count = context.get("retry_count", 0)

        if retry_count >= self.max_retries:
            sanitized_msg = self.sanitizer.sanitize_error_message(
                str(error), SensitivityLevel.MEDIUM
            )
            self._logger.error(
                f"Max retries ({self.max_retries}) exceeded for network error: {sanitized_msg}"
            )
            return {
                "action": "fail",
                "reason": "max_retries_exceeded",
                "sanitized_error": sanitized_msg,
            }

        # Calculate delay with exponential backoff
        delay = self.base_delay * (2**retry_count)

        sanitized_msg = self.sanitizer.sanitize_error_message(str(error), SensitivityLevel.MEDIUM)
        self._logger.warning(
            f"Network error occurred: {sanitized_msg}. "
            f"Retrying in {delay}s (attempt {retry_count + 1}/{self.max_retries})"
        )

        return {
            "action": "retry",
            "delay": delay,
            "retry_count": retry_count + 1,
            "max_retries": self.max_retries,
            "sanitized_error": sanitized_msg,
        }


class RateLimitErrorHandler(ErrorHandlerBase):
    """Handler for rate limit errors with secure sanitization."""

    def __init__(self, next_handler=None):
        super().__init__(next_handler)
        self.sanitizer = get_security_sanitizer()

    def can_handle(self, error: Exception) -> bool:
        """Check if this is a rate limit error."""
        error_msg = str(error).lower()
        rate_limit_keywords = [
            "rate limit",
            "too many requests",
            "429",
            "throttle",
            "quota exceeded",
        ]

        return any(keyword in error_msg for keyword in rate_limit_keywords)

    async def handle(
        self, error: Exception, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Handle rate limit error with backoff strategy.

        Args:
            error: The rate limit error
            context: Optional context

        Returns:
            Recovery action dictionary
        """
        # Try to extract retry-after from error or context
        retry_after = self._extract_retry_after(error, context)

        if retry_after is None:
            # Default to 60 seconds if not specified
            retry_after = 60

        sanitized_msg = self.sanitizer.sanitize_error_message(str(error), SensitivityLevel.MEDIUM)
        self._logger.warning(
            f"Rate limit exceeded: {sanitized_msg}. Waiting {retry_after}s before retry"
        )

        return {
            "action": "wait",
            "delay": retry_after,
            "reason": "rate_limit",
            "circuit_break": True,  # Suggest circuit breaker activation
            "sanitized_error": sanitized_msg,
        }

    def _extract_retry_after(
        self, error: Exception, context: dict[str, Any] | None
    ) -> float | None:
        """Extract retry-after value from error or context."""
        # Check context first
        if context and "retry_after" in context:
            return float(context["retry_after"])

        # Try to parse from error message
        import re

        error_str = str(error)

        # Look for patterns like "retry after 30 seconds"
        match = re.search(r"retry[\s\-_]?after[\s:]*(\d+)", error_str, re.IGNORECASE)
        if match:
            return float(match.group(1))

        # Look for "429" status with time
        match = re.search(r"429.*?(\d+)\s*(?:seconds?|s)", error_str, re.IGNORECASE)
        if match:
            return float(match.group(1))

        return None
