"""Network-specific error handlers with secure data sanitization."""

import asyncio
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from src.error_handling.base import ErrorHandlerBase
from src.error_handling.security_validator import SensitivityLevel
from src.utils.error_categorization import detect_rate_limiting
from src.utils.error_handling_utils import (
    create_recovery_response,
    extract_retry_after_from_error,
    get_or_create_sanitizer,
    sanitize_error_with_level,
)


class NetworkErrorHandler(ErrorHandlerBase):
    """Handler for network-related errors."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: Decimal = Decimal("1.0"),
        next_handler: ErrorHandlerBase | None = None,
        sanitizer=None,
    ) -> None:
        """
        Initialize network error handler with secure sanitization.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries (exponential backoff)
            next_handler: Next handler in chain
            sanitizer: Security sanitizer (injected via DI)
        """
        super().__init__(next_handler)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.sanitizer = get_or_create_sanitizer(sanitizer)

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
        Handle network error by delegating to service layer.

        Args:
            error: The network error
            context: Optional context

        Returns:
            Recovery action dictionary
        """
        context = context or {}
        retry_count = context.get("retry_count", 0)

        if retry_count >= self.max_retries:
            sanitized_msg = sanitize_error_with_level(
                error, SensitivityLevel.MEDIUM, self.sanitizer
            )
            self._logger.error(
                f"Max retries ({self.max_retries}) exceeded for network error: {sanitized_msg}"
            )
            return create_recovery_response(
                action="fail",
                reason="max_retries_exceeded",
                error=error,
                level=SensitivityLevel.MEDIUM,
                sanitizer=self.sanitizer,
            )

        # Delegate delay calculation to service layer
        delay = self._calculate_retry_delay(retry_count)

        sanitized_msg = sanitize_error_with_level(error, SensitivityLevel.MEDIUM, self.sanitizer)
        self._logger.warning(
            f"Network error occurred: {sanitized_msg}. "
            f"Retrying in {delay}s (attempt {retry_count + 1}/{self.max_retries})"
        )

        return create_recovery_response(
            action="retry",
            reason="network_error",
            error=error,
            level=SensitivityLevel.MEDIUM,
            sanitizer=self.sanitizer,
            delay=str(delay),
            retry_count=retry_count + 1,
            max_retries=self.max_retries,
        )

    def _calculate_retry_delay(self, retry_count: int) -> Decimal:
        """Calculate retry delay - moved business logic to separate method."""
        from decimal import localcontext

        with localcontext() as ctx:
            ctx.prec = 8
            ctx.rounding = ROUND_HALF_UP
            return self.base_delay * (Decimal("2") ** retry_count)


class RateLimitErrorHandler(ErrorHandlerBase):
    """Handler for rate limit errors with secure sanitization."""

    def __init__(self, next_handler: ErrorHandlerBase | None = None, sanitizer=None) -> None:
        super().__init__(next_handler)
        self.sanitizer = get_or_create_sanitizer(sanitizer)

    def can_handle(self, error: Exception) -> bool:
        """Check if this is a rate limit error."""
        error_msg = str(error).lower()
        return detect_rate_limiting(error_msg)

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
        retry_after = extract_retry_after_from_error(error, context)

        if retry_after is None:
            # Default to 60 seconds if not specified
            from decimal import Decimal

            retry_after = Decimal("60")

        sanitized_msg = sanitize_error_with_level(error, SensitivityLevel.MEDIUM, self.sanitizer)
        self._logger.warning(
            f"Rate limit exceeded: {sanitized_msg}. Waiting {retry_after}s before retry"
        )

        return create_recovery_response(
            action="wait",
            reason="rate_limit",
            error=error,
            level=SensitivityLevel.MEDIUM,
            sanitizer=self.sanitizer,
            delay=str(retry_after),
            circuit_break=True,
        )
