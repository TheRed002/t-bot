"""Common utilities for service implementations to eliminate code duplication."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable
from decimal import Decimal

from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.core.logging import Logger

logger = get_logger(__name__)


async def safe_service_shutdown(
    service_name: str,
    cleanup_func: Callable[[], Any] | None = None,
    service_logger: Logger | None = None
) -> None:
    """
    Common pattern for safe service shutdown with error handling.

    Args:
        service_name: Name of the service being shut down
        cleanup_func: Optional cleanup function to call during shutdown
        service_logger: Optional service-specific logger
    """
    _logger = service_logger or logger

    try:
        if cleanup_func:
            if hasattr(cleanup_func, '__call__'):
                if hasattr(cleanup_func, '__await__'):
                    await cleanup_func()
                else:
                    cleanup_func()

        _logger.info(f"{service_name} service stopped and resources cleaned up")
    except Exception as e:
        _logger.error(f"Error during {service_name} shutdown: {e}")
        raise ServiceError(f"{service_name} shutdown failed: {e}") from e


def validate_positive_amount(
    amount: Decimal,
    field_name: str = "amount",
    operation: str = "operation"
) -> None:
    """
    Common validation for positive decimal amounts.

    Args:
        amount: The amount to validate
        field_name: Name of the field being validated
        operation: Name of the operation for error context
    """
    if amount <= 0:
        raise ValidationError(
            f"{operation.capitalize()} amount must be positive",
            field_name=field_name,
            field_value=amount,
            expected_type="positive decimal",
        )


def validate_non_empty_string(
    value: str,
    field_name: str,
    operation: str = "operation"
) -> None:
    """
    Common validation for non-empty string fields.

    Args:
        value: The string value to validate
        field_name: Name of the field being validated
        operation: Name of the operation for error context
    """
    if not value or not value.strip():
        raise ValidationError(
            f"{operation.capitalize()} {field_name} cannot be empty",
            field_name=field_name,
            field_value=value,
            expected_type="non-empty string",
        )


def get_resource_cleanup_manager():
    """
    Get the resource manager for cleanup operations.

    Returns:
        Resource manager instance or None if not available
    """
    try:
        from src.utils.resource_manager import get_resource_manager
        return get_resource_manager()
    except ImportError:
        logger.warning("Resource manager not available, cleanup will use basic patterns")
        return None


def safe_cleanup_with_logging(
    cleanup_operations: list[tuple[str, Callable]],
    service_logger: Logger | None = None
) -> None:
    """
    Safely execute a list of cleanup operations with logging.

    Args:
        cleanup_operations: List of (operation_name, operation_function) tuples
        service_logger: Optional service-specific logger
    """
    _logger = service_logger or logger

    for operation_name, operation_func in cleanup_operations:
        try:
            operation_func()
            _logger.debug(f"Cleanup operation '{operation_name}' completed successfully")
        except Exception as e:
            _logger.warning(f"Cleanup operation '{operation_name}' failed: {e}")
            # Continue with other cleanup operations