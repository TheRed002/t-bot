"""
Authentication middleware for T-Bot web interface.

This middleware handles authentication for all requests and provides
request context with user information.
"""

import time
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.logging import get_logger
from src.web_interface.security.jwt_handler import JWTHandler

logger = get_logger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for request processing.

    This middleware:
    - Extracts and validates JWT tokens from requests
    - Adds user context to request state
    - Logs authentication events
    - Handles authentication errors gracefully
    """

    def __init__(self, app, jwt_handler: JWTHandler):
        """
        Initialize authentication middleware.

        Args:
            app: FastAPI application
            jwt_handler: JWT token handler
        """
        super().__init__(app)
        self.jwt_handler = jwt_handler
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # Paths that don't require authentication
        self.exempt_paths = {
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/auth/login",
            "/auth/refresh",
            "/favicon.ico",
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through authentication middleware.

        Args:
            request: HTTP request
            call_next: Next middleware/endpoint

        Returns:
            Response: HTTP response
        """
        start_time = time.time()

        # Skip authentication for exempt paths
        if request.url.path in self.exempt_paths:
            response = await call_next(request)
            self._add_timing_header(response, start_time)
            return response

        # Skip authentication for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            response = await call_next(request)
            self._add_timing_header(response, start_time)
            return response

        # Extract token from Authorization header
        auth_header = request.headers.get("authorization")
        token = self._extract_token(auth_header)

        # Add authentication context to request
        request.state.authenticated = False
        request.state.user = None
        request.state.token_data = None

        if token:
            try:
                # Validate token
                token_data = self.jwt_handler.validate_token(token)

                # Add user context to request
                request.state.authenticated = True
                request.state.token_data = token_data
                request.state.user = {
                    "user_id": token_data.user_id,
                    "username": token_data.username,
                    "scopes": token_data.scopes,
                }

                self.logger.debug(
                    "Request authenticated",
                    path=request.url.path,
                    method=request.method,
                    username=token_data.username,
                    user_id=token_data.user_id,
                )

            except Exception as e:
                self.logger.warning(
                    "Authentication failed",
                    path=request.url.path,
                    method=request.method,
                    error=str(e),
                )
                # Continue without authentication (endpoint can handle unauthorized access)

        # Process request
        response = await call_next(request)

        # Add timing and security headers
        self._add_timing_header(response, start_time)
        self._add_security_headers(response)

        # Log request completion
        self._log_request_completion(request, response, start_time)

        return response

    def _extract_token(self, auth_header: str) -> str | None:
        """
        Extract JWT token from Authorization header.

        Args:
            auth_header: Authorization header value

        Returns:
            str: JWT token or None if not found
        """
        if not auth_header:
            return None

        # Handle Bearer token format
        if auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix

        # Handle direct token
        return auth_header if len(auth_header) > 20 else None

    def _add_timing_header(self, response: Response, start_time: float) -> None:
        """
        Add request timing header to response.

        Args:
            response: HTTP response
            start_time: Request start time
        """
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

    def _add_security_headers(self, response: Response) -> None:
        """
        Add security headers to response.

        Args:
            response: HTTP response
        """
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # API-specific headers
        response.headers["X-API-Version"] = "1.0"
        response.headers["X-Powered-By"] = "T-Bot Trading System"

    def _log_request_completion(
        self, request: Request, response: Response, start_time: float
    ) -> None:
        """
        Log request completion details.

        Args:
            request: HTTP request
            response: HTTP response
            start_time: Request start time
        """
        process_time = time.time() - start_time

        log_data = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time": round(process_time, 4),
            "authenticated": getattr(request.state, "authenticated", False),
        }

        # Add user info if authenticated
        if hasattr(request.state, "user") and request.state.user:
            log_data["username"] = request.state.user.get("username")
            log_data["user_id"] = request.state.user.get("user_id")

        # Add client IP
        client_ip = request.client.host if request.client else "unknown"
        log_data["client_ip"] = client_ip

        # Log level based on status code
        if response.status_code >= 500:
            self.logger.error("Request completed with server error", **log_data)
        elif response.status_code >= 400:
            self.logger.warning("Request completed with client error", **log_data)
        else:
            self.logger.info("Request completed successfully", **log_data)
