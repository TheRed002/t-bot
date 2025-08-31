"""
Common Exchange Error Handling Utilities

This module contains shared error handling utilities to eliminate duplication
across exchange implementations. It provides:
- Common error handling patterns
- Exchange error mapping utilities
- Retry logic for exchange operations
- Error context creation and logging
"""

import asyncio
from typing import Any

from src.core.exceptions import (
    ExchangeConnectionError,
    ExchangeError,
    ExchangeInsufficientFundsError,
    ExchangeRateLimitError,
    ExecutionError,
    OrderRejectionError,
    ValidationError,
)
from src.core.logging import get_logger
from src.core.types import OrderRequest
from src.error_handling.error_handler import ErrorHandler
from src.utils.decorators import retry


class ExchangeErrorHandler:
    """Common error handling utilities for exchanges."""

    def __init__(self, exchange_name: str, config, error_handler: ErrorHandler | None = None):
        """
        Initialize exchange error handler.

        Args:
            exchange_name: Name of the exchange
            config: Application configuration
            error_handler: Optional error handler instance
        """
        self.exchange_name = exchange_name
        self.config = config

        # Use dependency injection if error_handler not provided
        if error_handler is None:
            from src.core.dependency_injection import injector

            try:
                self.error_handler = injector.resolve("ErrorHandler")
            except (KeyError, ImportError, AttributeError) as e:
                # Fallback to direct creation when dependency injection fails
                self.logger.warning(f"Failed to resolve ErrorHandler via DI: {e}. Using fallback creation.")
                self.error_handler = ErrorHandler(config)
        else:
            self.error_handler = error_handler

        self.logger = get_logger(f"{exchange_name}.error_handler")

    async def handle_exchange_error(
        self, error: Exception, operation: str, context: dict[str, Any] | None = None, order: OrderRequest | None = None
    ) -> None:
        """
        Handle exchange-specific errors using unified error mapping.

        Args:
            error: The exception that occurred
            operation: Operation being performed
            context: Additional context information
            order: Optional order involved in the operation
        """
        try:
            # Create error context
            error_context = self.error_handler.create_error_context(
                error=error,
                component=f"{self.exchange_name}_exchange",
                operation=operation,
                symbol=order.symbol if order else context.get("symbol") if context else None,
                order_id=order.client_order_id if order else context.get("order_id") if context else None,
                details={
                    "exchange_name": self.exchange_name,
                    "operation": operation,
                    "order_type": order.order_type.value if order else None,
                    **(context or {}),
                },
            )

            # Log the error with context
            self.logger.error(f"{self.exchange_name} {operation} failed: {error}", extra=error_context.details)

            # Handle using error handler
            await self.error_handler.handle_error(error, error_context)

        except Exception as handler_error:
            # Fallback to basic logging if error handling fails
            self.logger.error(f"Error handling failed for {operation}: {handler_error}")

    @retry(max_attempts=3, delay=1)
    async def handle_api_error(self, error: Exception, operation: str, context: dict[str, Any] | None = None) -> None:
        """
        Handle API errors with automatic retry logic and timeout protection.

        Args:
            error: The API exception
            operation: Operation being performed
            context: Additional context
        """
        try:
            await asyncio.wait_for(
                self.handle_exchange_error(error, operation, context), timeout=10.0  # Error handling timeout
            )
        except asyncio.TimeoutError:
            self.logger.error(f"Error handling timed out for {operation}")
        except Exception as handler_error:
            self.logger.error(f"Error handler failed for {operation}: {handler_error}")


class ErrorMappingUtils:
    """Utilities for mapping exchange-specific errors to unified exceptions."""

    # Common error patterns across exchanges
    INSUFFICIENT_FUNDS_PATTERNS = [
        "insufficient",
        "balance",
        "not enough",
        "insufficient funds",
        "insufficient balance",
        "account has insufficient",
    ]

    RATE_LIMIT_PATTERNS = ["rate limit", "too many requests", "rate exceeded", "throttle", "frequency", "rate limiting"]

    VALIDATION_PATTERNS = [
        "invalid",
        "parameter",
        "validation",
        "format",
        "malformed",
        "bad request",
        "invalid parameter",
    ]

    CONNECTION_PATTERNS = [
        "connection",
        "network",
        "timeout",
        "unreachable",
        "disconnected",
        "connection error",
        "network error",
    ]

    @staticmethod
    def map_exchange_error(
        error_data: dict[str, Any], exchange: str, default_error_class: type[Exception] = ExchangeError
    ) -> Exception:
        """
        Map exchange-specific error to unified exception.

        Args:
            error_data: Error data from exchange
            exchange: Exchange name
            default_error_class: Default exception class to use

        Returns:
            Exception: Mapped unified exception
        """
        message = str(error_data.get("message", error_data.get("msg", "Unknown error")))
        code = str(error_data.get("code", error_data.get("error_code", "")))

        message_lower = message.lower()

        # Check for insufficient funds
        if any(pattern in message_lower for pattern in ErrorMappingUtils.INSUFFICIENT_FUNDS_PATTERNS):
            return ExchangeInsufficientFundsError(f"Insufficient funds: {message}")

        # Check for rate limiting
        if any(pattern in message_lower for pattern in ErrorMappingUtils.RATE_LIMIT_PATTERNS):
            return ExchangeRateLimitError(f"Rate limit exceeded: {message}")

        # Check for validation errors
        if any(pattern in message_lower for pattern in ErrorMappingUtils.VALIDATION_PATTERNS):
            return ValidationError(f"Validation failed: {message}")

        # Check for connection errors
        if any(pattern in message_lower for pattern in ErrorMappingUtils.CONNECTION_PATTERNS):
            return ExchangeConnectionError(f"Connection error: {message}")

        # Exchange-specific mappings
        if exchange == "binance":
            return ErrorMappingUtils._map_binance_error(error_data, message, code)
        elif exchange == "coinbase":
            return ErrorMappingUtils._map_coinbase_error(error_data, message, code)
        elif exchange == "okx":
            return ErrorMappingUtils._map_okx_error(error_data, message, code)

        # Default mapping
        if "reject" in message_lower or "denied" in message_lower:
            return OrderRejectionError(f"Order rejected: {message}")

        return default_error_class(f"Exchange error: {message}")

    @staticmethod
    def _map_binance_error(error_data: dict[str, Any], message: str, code: str) -> Exception:
        """Map Binance-specific errors."""
        # Binance error code mappings
        binance_error_codes = {
            "-1021": ExchangeConnectionError,  # Timestamp outside recv window
            "-1022": ExchangeConnectionError,  # Invalid signature
            "-2010": OrderRejectionError,  # New order rejected
            "-2011": OrderRejectionError,  # Order canceled
            "-1013": ValidationError,  # Invalid quantity
            "-1111": ValidationError,  # Precision over maximum
            "-1112": ValidationError,  # No orders on book for symbol
        }

        exception_class = binance_error_codes.get(code, ExchangeError)
        return exception_class(f"Binance error [{code}]: {message}")

    @staticmethod
    def _map_coinbase_error(error_data: dict[str, Any], message: str, code: str) -> Exception:
        """Map Coinbase-specific errors."""
        # Coinbase error mappings
        if "authentication" in message.lower():
            return ExchangeConnectionError(f"Coinbase authentication error: {message}")

        if "invalid_request" in message.lower():
            return ValidationError(f"Coinbase invalid request: {message}")

        return ExchangeError(f"Coinbase error: {message}")

    @staticmethod
    def _map_okx_error(error_data: dict[str, Any], message: str, code: str) -> Exception:
        """Map OKX-specific errors."""
        # OKX error code mappings
        okx_error_codes = {
            "51008": ExchangeInsufficientFundsError,  # Insufficient balance
            "51009": ExchangeInsufficientFundsError,  # Order would immediately match
            "51010": ValidationError,  # Account level too low
            "51014": ValidationError,  # Invalid parameter
            "50001": ValidationError,  # Parameter error
            "50004": ExchangeConnectionError,  # Endpoint request timeout
        }

        exception_class = okx_error_codes.get(code, ExchangeError)
        return exception_class(f"OKX error [{code}]: {message}")


class RetryableOperationHandler:
    """Handler for retryable exchange operations."""

    def __init__(self, exchange_name: str, logger=None):
        """
        Initialize retryable operation handler.

        Args:
            exchange_name: Name of the exchange
            logger: Optional logger instance
        """
        self.exchange_name = exchange_name
        self.logger = logger or get_logger(f"{exchange_name}.retry")

    @retry(max_attempts=3, delay=1)
    async def execute_with_retry(self, operation_func, operation_name: str, *args, **kwargs) -> Any:
        """
        Execute an operation with retry logic and timeout protection.

        Args:
            operation_func: Function to execute
            operation_name: Name of the operation for logging
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Any: Result of the operation
        """
        try:
            self.logger.debug(f"Executing {operation_name}")
            # Add timeout to operation execution
            result = await asyncio.wait_for(operation_func(*args, **kwargs), timeout=30.0)  # Operation timeout
            self.logger.debug(f"Successfully executed {operation_name}")
            return result
        except asyncio.TimeoutError:
            self.logger.error(f"Operation {operation_name} timed out after 30 seconds")
            raise
        except Exception as e:
            self.logger.error(f"Failed to execute {operation_name}: {e}")
            raise

    @retry(max_attempts=5, delay=2)
    async def execute_with_aggressive_retry(self, operation_func, operation_name: str, *args, **kwargs) -> Any:
        """
        Execute an operation with aggressive retry logic and timeout protection.

        Args:
            operation_func: Function to execute
            operation_name: Name of the operation for logging
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Any: Result of the operation
        """
        try:
            # Add timeout to aggressive retry operations
            result = await asyncio.wait_for(
                operation_func(*args, **kwargs), timeout=60.0  # Longer timeout for aggressive retry
            )
            return result
        except asyncio.TimeoutError:
            self.logger.error(f"Aggressive retry operation {operation_name} timed out after 60 seconds")
            raise
        except (ExchangeConnectionError, ExchangeRateLimitError) as e:
            # These errors are worth aggressive retry
            self.logger.warning(f"Retryable error in {operation_name}: {e}")
            raise
        except Exception as e:
            # Other errors shouldn't be retried aggressively
            self.logger.error(f"Non-retryable error in {operation_name}: {e}")
            raise


class OperationTimeoutHandler:
    """Handler for operation timeouts."""

    @staticmethod
    async def execute_with_timeout(
        operation_func, timeout_seconds: int, operation_name: str = "operation", *args, **kwargs
    ) -> Any:
        """
        Execute an operation with timeout.

        Args:
            operation_func: Function to execute
            timeout_seconds: Timeout in seconds
            operation_name: Name of the operation for logging
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Any: Result of the operation

        Raises:
            ExecutionError: If operation times out
        """
        try:
            return await asyncio.wait_for(operation_func(*args, **kwargs), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise ExecutionError(f"{operation_name} timed out after {timeout_seconds} seconds")


class ExchangeCircuitBreaker:
    """Simple circuit breaker for exchange operations."""

    def __init__(
        self, failure_threshold: int = 5, recovery_timeout: int = 300, half_open_max_calls: int = 3  # 5 minutes
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before trying to close circuit (seconds)
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.half_open_calls = 0
        self.logger = get_logger(f"{self.__class__.__name__}")

    async def call(self, func, *args, **kwargs):
        """
        Execute function through circuit breaker with timeout protection.

        Args:
            func: Function to execute
            *args: Arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Any: Result of function call

        Raises:
            ExecutionError: If circuit is open or times out
        """
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                self.half_open_calls = 0
            else:
                raise ExecutionError("Circuit breaker is open - operation not allowed")

        if self.state == "half_open":
            if self.half_open_calls >= self.half_open_max_calls:
                raise ExecutionError("Circuit breaker half-open - max calls exceeded")
            self.half_open_calls += 1

        try:
            # Add timeout to circuit breaker calls
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=45.0)  # Circuit breaker timeout
            self._record_success()
            return result
        except asyncio.TimeoutError:
            self.logger.error("Circuit breaker function call timed out")
            self._record_failure()
            raise ExecutionError("Circuit breaker function call timed out")
        except Exception as e:
            self.logger.error(f"Circuit breaker function call failed: {type(e).__name__}: {e}")
            self._record_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        if not self.last_failure_time:
            return True

        import time

        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _record_success(self) -> None:
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"
        self.half_open_calls = 0

    def _record_failure(self) -> None:
        """Record failed operation."""
        import time

        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def get_state(self) -> dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "half_open_calls": self.half_open_calls,
        }
