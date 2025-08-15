"""
Error handling middleware for T-Bot web interface.

This middleware provides comprehensive error handling and logging
for all API requests with detailed error responses.
"""

import traceback
from collections.abc import Callable

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ExecutionError,
    NetworkError,
    TradingBotError,
    ValidationError,
)
from src.core.logging import get_logger

logger = get_logger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive error handling middleware.

    This middleware:
    - Catches and handles all exceptions
    - Provides structured error responses
    - Logs errors with detailed context
    - Sanitizes error messages for security
    - Maps internal exceptions to HTTP status codes
    """

    def __init__(self, app, debug: bool = False):
        """
        Initialize error handling middleware.

        Args:
            app: FastAPI application
            debug: Enable debug mode with detailed error info
        """
        super().__init__(app)
        self.debug = debug
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # Error code mappings
        self.exception_mapping = {
            ValidationError: 400,
            AuthenticationError: 401,
            PermissionError: 403,
            FileNotFoundError: 404,
            ExecutionError: 422,
            ConfigurationError: 500,
            NetworkError: 503,
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through error handling middleware.

        Args:
            request: HTTP request
            call_next: Next middleware/endpoint

        Returns:
            Response: HTTP response or error response
        """
        try:
            # Process request
            response = await call_next(request)
            return response

        except HTTPException as e:
            # FastAPI HTTPExceptions are handled by FastAPI itself
            # But we log them for monitoring
            self._log_http_exception(request, e)
            raise

        except TradingBotError as e:
            # Handle custom T-Bot exceptions
            return await self._handle_tbot_exception(request, e)

        except Exception as e:
            # Handle unexpected exceptions
            return await self._handle_unexpected_exception(request, e)

    async def _handle_tbot_exception(
        self, request: Request, exception: TradingBotError
    ) -> JSONResponse:
        """
        Handle T-Bot specific exceptions.

        Args:
            request: HTTP request
            exception: T-Bot exception

        Returns:
            JSONResponse: Error response
        """
        # Determine status code
        status_code = self.exception_mapping.get(type(exception), 500)

        # Create error response
        error_response = {
            "error": {
                "type": exception.__class__.__name__,
                "message": str(exception),
                "code": getattr(exception, "error_code", None),
                "timestamp": self._get_current_timestamp(),
                "request_id": self._get_request_id(request),
            }
        }

        # Add debug information if enabled
        if self.debug:
            error_response["error"]["details"] = {
                "path": str(request.url.path),
                "method": request.method,
                "traceback": traceback.format_exc(),
            }

        # Log the exception
        self._log_tbot_exception(request, exception, status_code)

        return JSONResponse(status_code=status_code, content=error_response)

    async def _handle_unexpected_exception(
        self, request: Request, exception: Exception
    ) -> JSONResponse:
        """
        Handle unexpected exceptions.

        Args:
            request: HTTP request
            exception: Unexpected exception

        Returns:
            JSONResponse: Error response
        """
        # Log the full exception details
        self._log_unexpected_exception(request, exception)

        # Create sanitized error response
        error_response = {
            "error": {
                "type": "InternalServerError",
                "message": "An internal server error occurred",
                "timestamp": self._get_current_timestamp(),
                "request_id": self._get_request_id(request),
            }
        }

        # Add debug information if enabled
        if self.debug:
            error_response["error"]["details"] = {
                "original_type": exception.__class__.__name__,
                "original_message": str(exception),
                "path": str(request.url.path),
                "method": request.method,
                "traceback": traceback.format_exc(),
            }

        return JSONResponse(status_code=500, content=error_response)

    def _log_http_exception(self, request: Request, exception: HTTPException) -> None:
        """
        Log HTTP exceptions.

        Args:
            request: HTTP request
            exception: HTTP exception
        """
        log_data = {
            "request_id": self._get_request_id(request),
            "path": str(request.url.path),
            "method": request.method,
            "status_code": exception.status_code,
            "detail": exception.detail,
            "client_ip": request.client.host if request.client else "unknown",
        }

        # Add user context if available
        if hasattr(request.state, "user") and request.state.user:
            log_data["username"] = request.state.user.get("username")
            log_data["user_id"] = request.state.user.get("user_id")

        if exception.status_code >= 500:
            self.logger.error("HTTP server error", **log_data)
        elif exception.status_code >= 400:
            self.logger.warning("HTTP client error", **log_data)

    def _log_tbot_exception(
        self, request: Request, exception: TradingBotError, status_code: int
    ) -> None:
        """
        Log T-Bot exceptions.

        Args:
            request: HTTP request
            exception: T-Bot exception
            status_code: HTTP status code
        """
        log_data = {
            "request_id": self._get_request_id(request),
            "path": str(request.url.path),
            "method": request.method,
            "exception_type": exception.__class__.__name__,
            "exception_message": str(exception),
            "status_code": status_code,
            "client_ip": request.client.host if request.client else "unknown",
        }

        # Add error code if available
        if hasattr(exception, "error_code"):
            log_data["error_code"] = exception.error_code

        # Add user context if available
        if hasattr(request.state, "user") and request.state.user:
            log_data["username"] = request.state.user.get("username")
            log_data["user_id"] = request.state.user.get("user_id")

        # Log level based on exception type
        if isinstance(exception, (AuthenticationError, ValidationError)):
            self.logger.warning("T-Bot client error", **log_data)
        else:
            self.logger.error("T-Bot server error", **log_data)

    def _log_unexpected_exception(self, request: Request, exception: Exception) -> None:
        """
        Log unexpected exceptions with full details.

        Args:
            request: HTTP request
            exception: Unexpected exception
        """
        log_data = {
            "request_id": self._get_request_id(request),
            "path": str(request.url.path),
            "method": request.method,
            "exception_type": exception.__class__.__name__,
            "exception_message": str(exception),
            "traceback": traceback.format_exc(),
            "client_ip": request.client.host if request.client else "unknown",
        }

        # Add user context if available
        if hasattr(request.state, "user") and request.state.user:
            log_data["username"] = request.state.user.get("username")
            log_data["user_id"] = request.state.user.get("user_id")

        # Add request body if small enough
        try:
            if hasattr(request, "_body"):
                body = request._body
                if body and len(body) < 1000:  # Only log small bodies
                    log_data["request_body"] = body.decode("utf-8", errors="ignore")
        except Exception:
            pass  # Ignore errors when trying to log request body

        self.logger.error("Unexpected server error", **log_data)

    def _get_request_id(self, request: Request) -> str:
        """
        Get or generate request ID.

        Args:
            request: HTTP request

        Returns:
            str: Request ID
        """
        # Try to get request ID from headers
        request_id = request.headers.get("X-Request-ID")
        if request_id:
            return request_id

        # Generate request ID from request properties
        import hashlib
        import time

        content = f"{request.method}:{request.url.path}:{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def _get_current_timestamp(self) -> str:
        """
        Get current timestamp in ISO format.

        Returns:
            str: ISO formatted timestamp
        """
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()

    def get_error_stats(self) -> dict:
        """
        Get error handling statistics.

        Returns:
            dict: Error handling statistics
        """
        return {
            "debug_mode": self.debug,
            "exception_mappings": {
                exc.__name__: status for exc, status in self.exception_mapping.items()
            },
            "handled_exception_types": [
                "HTTPException",
                "TradingBotError",
                "ValidationError",
                "AuthenticationError",
                "ExecutionError",
                "ConfigurationError",
                "NetworkError",
                "UnexpectedException",
            ],
            "features": [
                "structured_error_responses",
                "detailed_logging",
                "debug_information",
                "request_context",
                "user_context",
                "sanitized_messages",
            ],
        }
