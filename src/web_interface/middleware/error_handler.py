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
from src.error_handling import (
    ErrorContext,
    ErrorSeverity,
    get_global_error_handler,
)
from src.error_handling.pattern_analytics import ErrorPatternAnalytics

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

        # Integrate with global error handler with safe initialization
        try:
            self.global_error_handler = get_global_error_handler()
        except Exception as e:
            self.logger.warning(f"Global error handler not available: {e}")
            self.global_error_handler = None

        # Initialize pattern analytics with Config
        try:
            from src.core.config import Config
            config = Config()
            self.pattern_analytics = ErrorPatternAnalytics(config=config)
        except Exception as e:
            self.logger.warning(f"Error pattern analytics not available: {e}")
            self.pattern_analytics = None

        # Error code mappings - consistent with core.exceptions
        self.exception_mapping = {
            ValidationError: 400,
            AuthenticationError: 401,
            PermissionError: 403,
            FileNotFoundError: 404,
            ExecutionError: 422,
            ConfigurationError: 500,
            NetworkError: 503,
            # Add missing standard error mappings
            TradingBotError: 500,  # Generic catch-all for TradingBotError subclasses
        }

    def _validate_request_boundary(self, request: Request) -> None:
        """Validate incoming request at web_interface boundary."""
        try:
            from src.utils.messaging_patterns import BoundaryValidator, ProcessingParadigmAligner

            # Validate request boundary data - align with utils module patterns
            boundary_data = {
                "component": "web_interface",
                "operation": f"{request.method}_{request.url.path.replace('/', '_')}",
                "timestamp": request.headers.get("timestamp"),
                "processing_mode": "request_reply",  # Web requests are request/reply
                "data_format": "api_request_v1",
                "message_pattern": "req_reply",  # Align with utils patterns for API calls
                "boundary_crossed": True,
                "source": "external_client",
                "target": "web_interface"
            }

            # Apply boundary validation
            BoundaryValidator.validate_error_to_monitoring_boundary(boundary_data)

        except Exception as e:
            # Log but don't fail the request for boundary validation issues
            logger.debug(f"Request boundary validation warning: {e}")

    def _validate_response_boundary(self, request: Request, response: Response) -> None:
        """Validate outgoing response at web_interface boundary."""
        try:
            from src.utils.messaging_patterns import BoundaryValidator, ProcessingParadigmAligner

            # Validate response boundary data - align with utils module patterns
            boundary_data = {
                "component": "web_interface",
                "operation": f"response_{request.method}_{request.url.path.replace('/', '_')}",
                "timestamp": response.headers.get("timestamp"),
                "processing_mode": "request_reply",  # Web responses are request/reply
                "data_format": "api_response_v1",
                "message_pattern": "req_reply",  # Consistent with utils patterns for API responses
                "boundary_crossed": True,
                "source": "web_interface",
                "target": "external_client",
                "status_code": response.status_code
            }

            # Apply boundary validation
            BoundaryValidator.validate_risk_to_state_boundary(boundary_data)

        except Exception as e:
            # Log but don't fail the response for boundary validation issues
            logger.debug(f"Response boundary validation warning: {e}")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through error handling middleware with boundary validation.

        Args:
            request: HTTP request
            call_next: Next middleware/endpoint

        Returns:
            Response: HTTP response or error response
        """
        try:
            # Apply boundary validation for incoming requests
            self._validate_request_boundary(request)

            # Process request
            response = await call_next(request)

            # Apply boundary validation for outgoing responses
            self._validate_response_boundary(request, response)
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
            # Handle unexpected exceptions through global error handler
            error_context = ErrorContext.from_exception(
                error=e,
                component="web_interface",
                operation=f"{request.method} {request.url.path}",
                severity=ErrorSeverity.HIGH.value,
                method=request.method,
                path=str(request.url.path),
                client_ip=request.client.host if request.client else "unknown",
            )

            # Handle through global error handler if available
            handled = False
            recovery_details = None
            if self.global_error_handler:
                try:
                    # Match expected parameter names from global handler
                    result = await self.global_error_handler.handle_error(
                        error=e,
                        context={
                            "method": request.method,
                            "path": str(request.url.path),
                            "client_ip": request.client.host if request.client else "unknown",
                            "component": "web_interface",
                            "operation": f"{request.method} {request.url.path}",
                        },
                        severity="error",
                    )
                    handled = result.get("recovery_attempted", False)
                    # Extract recovery details from result if successful
                    if handled and result.get("recovery_result"):
                        recovery_details = {
                            "method": result.get("recovery_result", {}).get("method"),
                            "attempts": error_context.recovery_attempts,
                            "max_attempts": error_context.max_recovery_attempts,
                        }
                except Exception as handler_error:
                    self.logger.error(f"Error handler failed: {handler_error}")
                    handled = False

            if handled:
                # If error was handled, return appropriate response with recovery details
                if recovery_details:
                    error_context.details["recovery_details"] = recovery_details
                return await self._handle_recovered_error(request, e, error_context)
            else:
                # Otherwise, handle as unexpected exception
                return await self._handle_unexpected_exception(request, e)

    async def _handle_tbot_exception(
        self, request: Request, exception: TradingBotError
    ) -> JSONResponse:
        """
        Handle T-Bot specific exceptions with consistent data transformation.

        Args:
            request: HTTP request
            exception: T-Bot exception

        Returns:
            JSONResponse: Error response
        """
        # Apply consistent data transformation patterns matching error_handling module
        from datetime import datetime, timezone

        from src.utils.messaging_patterns import BoundaryValidator, ProcessingParadigmAligner

        # Determine status code
        status_code = self.exception_mapping.get(type(exception), 500)

        # Create error data using same transformer patterns as error_handling module
        from src.error_handling.data_transformer import ErrorDataTransformer

        # Transform error using same patterns as error_handling module
        error_context = {
            "component": "web_interface",
            "operation": f"{request.method} {request.url.path}",
            "error_type": exception.__class__.__name__,
            "error_message": str(exception),
            "severity": "medium" if status_code < 500 else "high",
        }

        # Use error_handling transformer for consistency
        aligned_error_data = ErrorDataTransformer.transform_for_pub_sub(
            event_type="web_interface_error",
            data=exception,
            metadata=error_context
        )

        # Validate at web_interface -> error_handling boundary for consistency
        try:
            BoundaryValidator.validate_monitoring_to_error_boundary(aligned_error_data)
        except ValidationError:
            # If boundary validation fails, continue with basic error response
            pass

        # Create error response with consistent structure
        error_response = {
            "error": {
                "type": exception.__class__.__name__,
                "message": str(exception),
                "code": getattr(exception, "error_code", None),
                "timestamp": self._get_current_timestamp(),
                "request_id": self._get_request_id(request),
                "processing_mode": "stream",
                "data_format": "error_response_v1",
                "boundary_crossed": True,
            }
        }

        # Add debug information if enabled
        if self.debug:
            error_response["error"]["details"] = {
                "path": str(request.url.path),
                "method": request.method,
                "traceback": traceback.format_exc(),
                "aligned_error_data": aligned_error_data,
            }

        # Log the exception
        self._log_tbot_exception(request, exception, status_code)

        return JSONResponse(status_code=status_code, content=error_response)

    async def _handle_unexpected_exception(
        self, request: Request, exception: Exception
    ) -> JSONResponse:
        """
        Handle unexpected exceptions with consistent error propagation patterns.

        Args:
            request: HTTP request
            exception: Unexpected exception

        Returns:
            JSONResponse: Error response
        """
        # Apply consistent data transformation patterns matching error_handling module
        from datetime import datetime, timezone

        from src.utils.messaging_patterns import BoundaryValidator, ProcessingParadigmAligner

        # Log the full exception details
        self._log_unexpected_exception(request, exception)

        # Create error data using same transformer patterns as error_handling module
        from src.error_handling.data_transformer import ErrorDataTransformer

        # Transform error using same patterns as error_handling module
        error_context = {
            "component": "web_interface",
            "operation": f"{request.method} {request.url.path}",
            "error_type": exception.__class__.__name__,
            "error_message": str(exception),
            "severity": "high",  # Unexpected errors are high severity
            "validation_status": "failed",
        }

        # Use error_handling transformer for consistency
        aligned_error_data = ErrorDataTransformer.transform_for_pub_sub(
            event_type="web_interface_unexpected_error",
            data=exception,
            metadata=error_context
        )

        # Validate at web_interface -> error_handling boundary for consistency
        try:
            BoundaryValidator.validate_monitoring_to_error_boundary(aligned_error_data)
        except ValidationError:
            # If boundary validation fails, continue with basic error response
            pass

        # Create sanitized error response with consistent structure
        error_response = {
            "error": {
                "type": "InternalServerError",
                "message": "An internal server error occurred",
                "timestamp": self._get_current_timestamp(),
                "request_id": self._get_request_id(request),
                "processing_mode": "stream",
                "data_format": "error_response_v1",
                "boundary_crossed": True,
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
                "aligned_error_data": aligned_error_data,
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
        if isinstance(exception, AuthenticationError | ValidationError):
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
        except Exception as e:
            # Ignore errors when trying to log request body
            self.logger.debug(f"Failed to log request body: {e}")

        self.logger.error("Unexpected server error", **log_data)

    async def _handle_recovered_error(
        self, request: Request, exception: Exception, error_context: ErrorContext
    ) -> JSONResponse:
        """
        Handle errors that were recovered by the global error handler with consistent patterns.

        Args:
            request: HTTP request
            exception: Original exception
            error_context: Error context with recovery information

        Returns:
            JSONResponse: Error response with recovery info
        """
        # Apply consistent data transformation patterns matching error_handling module
        from datetime import datetime, timezone

        from src.utils.messaging_patterns import BoundaryValidator, ProcessingParadigmAligner

        # Create recovery data using same transformer patterns as error_handling module
        from src.error_handling.data_transformer import ErrorDataTransformer

        # Transform recovery using same patterns as error_handling module
        recovery_context = {
            "component": "web_interface",
            "operation": f"{request.method} {request.url.path}",
            "error_type": exception.__class__.__name__,
            "severity": "medium",  # Recovery implies medium severity
            "validation_status": "recovered",
            "recovery_details": error_context.details.get("recovery_details", {}),
        }

        # Use error_handling transformer for consistency
        aligned_recovery_data = ErrorDataTransformer.transform_for_pub_sub(
            event_type="web_interface_recovery",
            data=exception,
            metadata=recovery_context
        )

        # Validate at web_interface -> error_handling boundary for consistency
        try:
            BoundaryValidator.validate_monitoring_to_error_boundary(aligned_recovery_data)
        except ValidationError:
            # If boundary validation fails, continue with basic recovery response
            pass

        # Log the recovery
        self.logger.info(
            "Error recovered by global handler",
            request_id=self._get_request_id(request),
            operation=error_context.operation,
            recovery_method=error_context.details.get("recovery_method", "unknown"),
        )

        # Create response with recovery information and consistent structure
        recovery_details = error_context.details.get("recovery_details", {})
        error_response = {
            "error": {
                "type": exception.__class__.__name__,
                "message": "Operation completed with recovery",
                "code": getattr(exception, "error_code", None),
                "timestamp": self._get_current_timestamp(),
                "request_id": self._get_request_id(request),
                "processing_mode": "stream",
                "data_format": "recovery_response_v1",
                "boundary_crossed": True,
                "recovery": {
                    "attempted": True,
                    "success": True,
                    "method": recovery_details.get(
                        "method", error_context.details.get("recovery_method", "unknown")
                    ),
                    "attempts": recovery_details.get("attempts", error_context.recovery_attempts),
                    "max_attempts": recovery_details.get(
                        "max_attempts", error_context.max_recovery_attempts
                    ),
                    "details": error_context.details.get("recovery_message", "Recovery successful"),
                    "aligned_data": aligned_recovery_data,
                },
            }
        }

        # Return 206 Partial Content to indicate recovery
        return JSONResponse(status_code=206, content=error_response)

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
        stats = {
            "debug_mode": self.debug,
            "exception_mappings": {
                exc.__name__: status for exc, status in self.exception_mapping.items()
            },
            "handled_exception_types": [exc.__name__ for exc in self.exception_mapping.keys()]
            + ["HTTPException", "TradingBotError", "UnexpectedException"],
            "features": [
                "structured_error_responses",
                "detailed_logging",
                "debug_information",
                "request_context",
                "user_context",
                "sanitized_messages",
                "global_error_handler_integration",
                "pattern_analytics",
                "recovery_handling",
            ],
        }

        # Add global error handler stats if available
        if self.global_error_handler:
            try:
                handler_stats = self.global_error_handler.get_statistics()
                stats["global_handler_stats"] = handler_stats
            except Exception as e:
                stats["global_handler_stats"] = {"status": "unavailable", "error": str(e)}

        # Add pattern analytics stats if available
        if self.pattern_analytics:
            try:
                # Use safe method calls that match the actual interface
                pattern_stats = {"status": "available"}
                # Check if methods exist before calling
                if hasattr(self.pattern_analytics, "get_pattern_summary"):
                    try:
                        pattern_summary = self.pattern_analytics.get_pattern_summary()
                        pattern_stats["pattern_count"] = pattern_summary.get("total_errors_tracked", 0)
                    except Exception as e:
                        logger.debug(f"Error getting pattern count: {e}")
                        pattern_stats["pattern_count"] = 0
                if hasattr(self.pattern_analytics, "get_recent_errors"):
                    try:
                        pattern_stats["recent_events"] = len(
                            self.pattern_analytics.get_recent_errors(hours=24)
                        )
                    except Exception as e:
                        logger.debug(f"Error getting recent errors: {e}")
                        pattern_stats["recent_events"] = 0
                if hasattr(self.pattern_analytics, "error_history"):
                    try:
                        pattern_stats["total_events"] = len(self.pattern_analytics.error_history)
                    except Exception as e:
                        logger.debug(f"Error getting total events: {e}")
                        pattern_stats["total_events"] = 0
                stats["pattern_analytics_stats"] = pattern_stats
            except Exception as e:
                stats["pattern_analytics_stats"] = {"status": "unavailable", "error": str(e)}

        return stats
