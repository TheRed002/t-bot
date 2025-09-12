"""
Capital Management Error Handling Utilities

This module provides shared error handling patterns for capital management operations
to ensure consistent error reporting and logging across all services.
"""

from decimal import Decimal
from typing import Any

from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.utils.formatters import format_currency

logger = get_logger(__name__)


def handle_service_error(
    error: Exception, operation: str, component: str, context: dict[str, Any], reraise: bool = True
) -> ServiceError | None:
    """
    Handle service errors with consistent logging and error propagation.

    Args:
        error: Original exception
        operation: Name of the operation that failed
        component: Component name for context
        context: Additional context for logging
        reraise: Whether to reraise the exception

    Returns:
        ServiceError if not reraising, None otherwise

    Raises:
        ServiceError: If reraise is True
    """
    error_details = {
        "operation": operation,
        "component": component,
        "error_type": type(error).__name__,
        "error": str(error),
        **context,
    }

    # Log based on error type
    if isinstance(error, (ValidationError, ServiceError)):
        # Service layer errors - log as warning
        logger.warning(f"{operation} failed - {type(error).__name__}", **error_details)
        if reraise:
            raise
        return error if isinstance(error, ServiceError) else ServiceError(str(error))
    else:
        # Unexpected errors - log as error with stack trace
        logger.error(f"Unexpected error in {operation}", exc_info=True, **error_details)

        service_error = ServiceError(
            f"{operation} failed: {error}", error_code="SERV_000", details=error_details
        )

        if reraise:
            raise service_error from error
        return service_error


def handle_repository_error(
    error: Exception,
    operation: str,
    component: str,
    context: dict[str, Any],
    fallback_result: Any = None,
) -> Any:
    """
    Handle repository errors with fallback behavior.

    Args:
        error: Repository error
        operation: Name of the repository operation
        component: Component name
        context: Additional context
        fallback_result: Result to return on error

    Returns:
        Fallback result or raises ServiceError
    """
    error_details = {
        "operation": operation,
        "component": component,
        "error_type": type(error).__name__,
        "error": str(error),
        **context,
    }

    if "repository" in str(error).lower() or "database" in str(error).lower():
        logger.error(f"Repository error during {operation}", **error_details)

        if fallback_result is not None:
            logger.info(f"Using fallback result for {operation}")
            return fallback_result

        raise ServiceError(f"Repository error during {operation}: {error}") from error
    else:
        # Re-raise non-repository errors
        raise


def log_allocation_operation(
    operation_type: str,
    strategy_id: str,
    exchange: str,
    amount: Decimal,
    component: str,
    success: bool = True,
    operation_id: str | None = None,
    **kwargs,
) -> None:
    """
    Log capital allocation operations with consistent format.

    Args:
        operation_type: Type of operation (allocate, release, update)
        strategy_id: Strategy identifier
        exchange: Exchange name
        amount: Amount involved
        component: Component performing the operation
        success: Whether operation was successful
        operation_id: Optional operation ID
        **kwargs: Additional logging context
    """
    log_data = {
        "operation_type": operation_type,
        "strategy_id": strategy_id,
        "exchange": exchange,
        "amount": format_currency(amount),
        "component": component,
        **kwargs,
    }

    if operation_id:
        log_data["operation_id"] = operation_id

    if success:
        logger.info(f"Capital {operation_type} successful", **log_data)
    else:
        logger.error(f"Capital {operation_type} failed", **log_data)


def log_fund_flow_operation(
    operation_type: str,
    amount: Decimal,
    currency: str,
    exchange: str,
    component: str,
    success: bool = True,
    reason: str | None = None,
    **kwargs,
) -> None:
    """
    Log fund flow operations with consistent format.

    Args:
        operation_type: Type of operation (deposit, withdrawal, reallocation)
        amount: Amount involved
        currency: Currency code
        exchange: Exchange name
        component: Component performing the operation
        success: Whether operation was successful
        reason: Optional reason for the operation
        **kwargs: Additional logging context
    """
    log_data = {
        "operation_type": operation_type,
        "amount": format_currency(amount, currency),
        "currency": currency,
        "exchange": exchange,
        "component": component,
        **kwargs,
    }

    if reason:
        log_data["reason"] = reason

    if success:
        logger.info(f"Fund flow {operation_type} successful", **log_data)
    else:
        logger.error(f"Fund flow {operation_type} failed", **log_data)


def create_operation_context(
    strategy_id: str | None = None,
    exchange: str | None = None,
    amount: Decimal | None = None,
    bot_id: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Create consistent operation context for error handling and logging.

    Args:
        strategy_id: Strategy identifier
        exchange: Exchange name
        amount: Amount involved
        bot_id: Bot instance ID
        **kwargs: Additional context

    Returns:
        Dict with operation context
    """
    context = {}

    if strategy_id:
        context["strategy_id"] = strategy_id
    if exchange:
        context["exchange"] = exchange
    if amount is not None:
        context["amount"] = format_currency(amount)
    if bot_id:
        context["bot_id"] = bot_id

    context.update(kwargs)
    return context


def wrap_service_operation(operation_name: str, component: str):
    """
    Decorator to wrap service operations with consistent error handling.

    Args:
        operation_name: Name of the operation for logging
        component: Component name for context
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"{operation_name} completed successfully", component=component)
                return result
            except (ValidationError, ServiceError):
                # Re-raise service layer exceptions
                raise
            except Exception as e:
                # Wrap unexpected errors
                context = create_operation_context(**kwargs)
                handle_service_error(e, operation_name, component, context)

        return wrapper

    return decorator


class CapitalErrorContext:
    """
    Context manager for capital management operations with error handling.
    """

    def __init__(self, operation: str, component: str, **context):
        self.operation = operation
        self.component = component
        self.context = context
        self.success = False

    async def __aenter__(self):
        logger.debug(f"Starting {self.operation}", component=self.component, **self.context)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.success = True
            logger.debug(f"{self.operation} completed successfully", component=self.component)
        else:
            logger.error(
                f"{self.operation} failed",
                component=self.component,
                error_type=exc_type.__name__,
                error=str(exc_val),
                **self.context,
            )
        return False  # Don't suppress exceptions
