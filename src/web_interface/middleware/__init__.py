"""
Middleware components for the T-Bot web interface.

This module provides various middleware for cross-cutting concerns including
authentication, rate limiting, error handling, CORS, and security headers.
"""

from .auth import AuthMiddleware
from .error_handler import ErrorHandlerMiddleware
from .rate_limit import RateLimitMiddleware

__all__ = ["AuthMiddleware", "ErrorHandlerMiddleware", "RateLimitMiddleware"]
