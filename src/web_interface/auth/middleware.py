"""
Authentication middleware for T-Bot Trading System.

This module provides middleware for handling authentication
in the unified auth system.
"""

from collections.abc import Callable

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.base import BaseComponent
from src.database.models.user import User

from .auth_manager import get_auth_manager

# Security scheme for FastAPI
security = HTTPBearer()


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
            token_from_cookie = request.cookies.get("access_token")
            if token_from_cookie:
                user = await self.auth_manager.validate_token(token_from_cookie)

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


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """
    Dependency to get the current authenticated user.

    Args:
        credentials: The HTTP bearer token credentials

    Returns:
        The authenticated user

    Raises:
        HTTPException: If authentication fails
    """
    auth_manager = get_auth_manager()

    # Validate the token
    user = await auth_manager.validate_token(credentials.credentials)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user
