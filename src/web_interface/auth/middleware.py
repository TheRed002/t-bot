"""
Authentication middleware for T-Bot Trading System.

This module provides middleware for handling authentication
in the unified auth system.
"""

from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.base import BaseComponent

from .auth_manager import get_auth_manager


class AuthMiddleware(BaseHTTPMiddleware, BaseComponent):
    """Middleware to handle authentication for all requests."""

    def __init__(self, app):
        super(BaseHTTPMiddleware, self).__init__(app)
        BaseComponent.__init__(self)
        self.auth_manager = get_auth_manager()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process authentication for incoming requests."""

        # Skip authentication for certain paths
        skip_paths = {"/health", "/docs", "/redoc", "/openapi.json", "/api/versions"}
        if request.url.path in skip_paths:
            return await call_next(request)

        # Try to authenticate the user
        user = None

        # Try Authorization header first
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
            user = await self.auth_manager.validate_token(token)

        # Try cookies if no bearer token
        if not user:
            token = request.cookies.get("access_token")
            if token:
                user = await self.auth_manager.validate_token(token)

        # Store user in request state
        request.state.current_user = user
        request.state.authenticated = user is not None

        # Process the request
        response = await call_next(request)

        # Add authentication headers
        if user:
            response.headers["X-User-ID"] = user.user_id
            response.headers["X-User-Roles"] = ",".join(role.name for role in user.roles)

        return response
