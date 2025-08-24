"""
Comprehensive Authentication Layer for T-Bot Trading System.

This package provides a unified authentication and authorization system
with role-based access control, session management, and security features.
"""

from .auth_manager import AuthManager, get_auth_manager, initialize_auth_manager
from .decorators import require_auth, require_permission, require_role
from .middleware import AuthMiddleware
from .models import AuthToken, Permission, Role, User
from .providers import JWTAuthProvider, SessionAuthProvider

__all__ = [
    "AuthManager",
    "AuthMiddleware",
    "AuthToken",
    "JWTAuthProvider",
    "Permission",
    "Role",
    "SessionAuthProvider",
    "User",
    "get_auth_manager",
    "initialize_auth_manager",
    "require_auth",
    "require_permission",
    "require_role",
]
