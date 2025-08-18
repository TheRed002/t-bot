"""
Correlation ID middleware for request tracking.

This middleware generates and manages correlation IDs for all requests
to enable tracing through the entire system.
"""

import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.logging import correlation_context, get_logger

logger = get_logger(__name__)


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Middleware to handle correlation IDs for request tracking."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request with correlation ID.

        Args:
            request: The incoming request
            call_next: The next middleware or endpoint

        Returns:
            The response with correlation ID header
        """
        # Get correlation ID from headers or generate new one
        correlation_id = (
            request.headers.get("X-Correlation-ID")
            or request.headers.get("X-Request-ID")
            or str(uuid.uuid4())
        )

        # Use correlation context for the entire request lifecycle
        with correlation_context.correlation_context(correlation_id):
            # Log request start
            logger.info(
                "Request started",
                method=request.method,
                path=request.url.path,
                correlation_id=correlation_id,
            )

            # Process request
            response = await call_next(request)

            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id

            # Log request completion
            logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                correlation_id=correlation_id,
            )

            return response
