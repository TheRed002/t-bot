"""
Web Interface Utilities for T-Bot Trading System.

This module provides common utilities for web interface operations,
including error handling, response formatting, and service access patterns.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from fastapi import HTTPException, status

from src.core.exceptions import (
    EntityNotFoundError,
    ExecutionError,
    ServiceError,
    ValidationError,
)
from src.core.logging import get_logger

logger = get_logger(__name__)


def handle_api_error(
    error: Exception, operation: str, user: str | None = None, context: dict[str, Any] | None = None
) -> HTTPException:
    """
    Convert application exceptions to appropriate HTTP exceptions with consistent error propagation.

    Args:
        error: The original exception
        operation: The operation that failed (for logging)
        user: Username for logging context
        context: Additional context for logging

    Returns:
        HTTPException with appropriate status code and message
    """
    # Apply consistent error propagation patterns
    # Lazy import to avoid circular dependency
    from datetime import timezone

    from src.utils.messaging_patterns import ErrorPropagationMixin

    # Create log context with consistent metadata
    log_context = {
        "processing_mode": "async",  # API operations are async
        "message_pattern": "req_reply",  # HTTP requests use req/reply
        "data_format": "http_error_v1",
        "boundary_crossed": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "component": "web_interface_utils",
        "operation": operation,
    }

    if user:
        log_context["user"] = user
    if context:
        log_context.update(context)

    # Use ErrorPropagationMixin for consistent error handling
    error_propagator = ErrorPropagationMixin()

    # Handle specific exception types with consistent propagation
    try:
        if isinstance(error, EntityNotFoundError):
            logger.warning(f"{operation} failed - entity not found: {error}", extra=log_context)
            return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))

        elif isinstance(error, ValidationError):
            # Use consistent validation error propagation
            error_propagator.propagate_validation_error(error, f"web_api_{operation}")
            return HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
            )

        elif isinstance(error, ExecutionError):
            logger.error(f"{operation} failed - execution error: {error}", extra=log_context)
            return HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
            )

        elif isinstance(error, ServiceError):
            # Use consistent service error propagation
            error_propagator.propagate_service_error(error, f"web_api_{operation}")
            return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error))

        else:
            # Generic exception handling with consistent propagation
            logger.error(f"{operation} failed: {error}", extra=log_context)
            error_propagator.propagate_service_error(error, f"web_api_{operation}")
            return HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{operation} failed: {error!s}",
            )

    except Exception as propagation_error:
        # If error propagation fails, continue with original error
        logger.error(f"Error propagation failed: {propagation_error}", extra=log_context)
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{operation} failed: {error!s}",
        )


def safe_format_currency(amount: Decimal, currency: str = "USD") -> str:
    """
    Safely format currency amount with error handling.

    Args:
        amount: The amount to format
        currency: Currency code (default: USD)

    Returns:
        Formatted currency string
    """
    try:
        # Lazy import to avoid circular dependency
        from src.utils.formatters import format_currency

        return format_currency(amount, currency)
    except Exception as e:
        logger.warning(f"Currency formatting error: {e}")
        return f"${float(amount):,.2f}"


def safe_format_percentage(percentage: Decimal) -> str:
    """
    Safely format percentage with error handling.

    Args:
        percentage: The percentage to format (as decimal, e.g., 0.05 for 5%)

    Returns:
        Formatted percentage string
    """
    try:
        # Lazy import to avoid circular dependency
        from src.utils.formatters import format_percentage

        return format_percentage(percentage)
    except Exception as e:
        logger.warning(f"Percentage formatting error: {e}")
        return f"{percentage * 100:.2f}%"


def safe_get_api_facade():
    """
    Safely get API facade with proper error handling.

    Returns:
        API facade instance

    Raises:
        ServiceError: If facade cannot be accessed
    """
    try:
        from src.web_interface.facade import get_api_facade

        return get_api_facade()
    except Exception as e:
        logger.error(f"Error getting API facade: {e}")
        raise ServiceError(f"Service not available: {e}")


def create_error_response(message: str, status_code: int = 500) -> HTTPException:
    """
    Create a standardized error response.

    Args:
        message: Error message
        status_code: HTTP status code

    Returns:
        HTTPException with standardized format
    """
    return HTTPException(status_code=status_code, detail=message)


def handle_not_found(item_type: str, item_id: str) -> HTTPException:
    """
    Create a standardized not found response.

    Args:
        item_type: Type of item (e.g., "Bot", "Order")
        item_id: ID of the item

    Returns:
        404 HTTPException
    """
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=f"{item_type} not found: {item_id}"
    )


def extract_error_details(error: Exception) -> dict[str, Any]:
    """
    Extract standardized error details from exception.

    Args:
        error: The exception to process

    Returns:
        Dictionary with error details
    """
    error_details = {
        "type": type(error).__name__,
        "message": str(error),
    }

    # Add specific handling for common error patterns
    error_str = str(error).lower()
    if "not found" in error_str:
        error_details["category"] = "not_found"
    elif "already exists" in error_str:
        error_details["category"] = "already_exists"
    elif "validation" in error_str or "invalid" in error_str:
        error_details["category"] = "validation"
    elif "permission" in error_str or "unauthorized" in error_str:
        error_details["category"] = "permission"
    else:
        error_details["category"] = "general"

    return error_details
