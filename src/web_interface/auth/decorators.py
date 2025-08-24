"""
Authentication decorators for T-Bot Trading System.

This module provides decorators for authentication and authorization
in FastAPI endpoints using the unified auth system.
"""

from functools import wraps

from fastapi import Depends, HTTPException, Request, status

from .auth_manager import get_auth_manager
from .models import PermissionType, User


async def get_current_user(request: Request) -> User | None:
    """Get the current authenticated user from the request."""
    auth_manager = get_auth_manager()

    # Try to get token from Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]
        user = await auth_manager.validate_token(token)
        return user

    # Try to get token from cookies
    token = request.cookies.get("access_token")
    if token:
        user = await auth_manager.validate_token(token)
        return user

    return None


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get the current active user, raise exception if not found."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not current_user.is_active():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User account is not active"
        )

    return current_user


async def get_trading_user(current_user: User = Depends(get_current_active_user)) -> User:
    """Get user with trading permissions."""
    if not current_user.has_role("trader") and not current_user.has_role("admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Trading permissions required"
        )
    return current_user


async def get_admin_user(current_user: User = Depends(get_current_active_user)) -> User:
    """Get user with admin permissions."""
    if not current_user.has_role("admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin permissions required"
        )
    return current_user


def require_auth(func):
    """Decorator to require authentication for an endpoint."""

    @wraps(func)
    async def wrapper(*args, current_user: User = Depends(get_current_active_user), **kwargs):
        return await func(*args, current_user=current_user, **kwargs)

    return wrapper


def require_permission(permission: PermissionType, resource: str | None = None):
    """Decorator to require a specific permission."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, current_user: User = Depends(get_current_active_user), **kwargs):
            if not current_user.has_permission(permission, resource):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission.value}' required",
                )
            return await func(*args, current_user=current_user, **kwargs)

        return wrapper

    return decorator


def require_role(role_name: str):
    """Decorator to require a specific role."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, current_user: User = Depends(get_current_active_user), **kwargs):
            if not current_user.has_role(role_name):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail=f"Role '{role_name}' required"
                )
            return await func(*args, current_user=current_user, **kwargs)

        return wrapper

    return decorator
