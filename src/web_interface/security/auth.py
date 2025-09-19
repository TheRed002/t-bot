"""
Authentication logic for T-Bot web interface.

This module provides core authentication functions including user verification,
token creation, and dependency injection for protected endpoints.
"""

from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.core.config import Config
from src.core.exceptions import AuthenticationError, ServiceError
from src.core.logging import get_logger
from src.database.models.user import User as DBUser
# Removed direct database service import - using service layer instead

from .jwt_handler import JWTHandler

logger = get_logger(__name__)


def get_auth_service():
    """Get auth service through dependency injection."""
    try:
        from src.core.dependency_injection import DependencyInjector

        injector = DependencyInjector.get_instance()
        if injector and injector.has_service("WebAuthService"):
            return injector.resolve("WebAuthService")
    except Exception as e:
        logger.warning(f"Could not get auth service: {e}")

    # Fallback to manual creation for development
    from src.web_interface.services.auth_service import WebAuthService
    return WebAuthService()


class UserInDB(BaseModel):
    """User model for database storage."""

    id: str  # Match database model field name
    username: str
    email: str
    password_hash: str  # Match database model field name
    is_active: bool = True
    is_verified: bool = False  # Add missing field from database model
    is_admin: bool = False  # Add missing field from database model
    scopes: list[str] = ["read"]
    created_at: str | None = None
    last_login_at: str | None = None  # Match database model field name


class User(BaseModel):
    """User model for API responses."""

    id: str  # Match database model field name
    username: str
    email: str
    is_active: bool
    is_verified: bool  # Add missing field from database model
    scopes: list[str]


class Token(BaseModel):
    """Token response model."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class LoginRequest(BaseModel):
    """Login request model."""

    username: str
    password: str


# Global instances (initialized by app startup)
jwt_handler: JWTHandler | None = None
security = HTTPBearer()

# Async session maker for database operations
_session_factory: async_sessionmaker | None = None


def init_auth(config: Config) -> None:
    """
    Initialize authentication system.

    Args:
        config: Application configuration
    """
    global jwt_handler, _session_factory
    jwt_handler = JWTHandler(config)

    # Initialize async session factory for database operations
    try:
        # The session factory will be provided by the database connection manager
        # We just mark that auth is initialized - actual sessions come from get_async_session()
        logger.info("Authentication system initialized (using global async session manager)")
    except Exception as e:
        logger.error(f"Failed to initialize authentication: {e}")

    logger.info("Authentication system initialized")


def _convert_db_user_to_user_in_db(db_user: DBUser) -> UserInDB:
    """Convert database User model to UserInDB."""
    return UserInDB(
        id=str(db_user.id),
        username=db_user.username,
        email=db_user.email,
        password_hash=db_user.password_hash,
        is_active=db_user.is_active,
        is_verified=db_user.is_verified,
        is_admin=db_user.is_admin,
        scopes=db_user.scopes if db_user.scopes else ["read"],
        created_at=db_user.created_at.isoformat() if db_user.created_at else None,
        last_login_at=db_user.last_login_at.isoformat() if db_user.last_login_at else None,
    )


async def get_user(username: str, database_service=None) -> UserInDB | None:
    """
    Get user by username through service layer.

    Args:
        username: Username to lookup
        database_service: Deprecated parameter (ignored)

    Returns:
        UserInDB: User data or None if not found
    """
    try:
        auth_service = get_auth_service()
        return await auth_service.get_user_by_username(username)

    except Exception as e:
        logger.error(f"Error getting user '{username}': {e}")
        return None


async def authenticate_user(
    username: str, password: str, database_service=None
) -> User | None:
    """
    Authenticate user credentials through service layer.

    Args:
        username: Username
        password: Plain text password
        database_service: Deprecated parameter (ignored)

    Returns:
        User: Authenticated user or None if authentication fails
    """
    try:
        auth_service = get_auth_service()
        return await auth_service.authenticate_user(username, password)

    except Exception as e:
        logger.error(f"Authentication error: {e}", username=username)
        return None


def create_access_token(user: UserInDB, expires_delta: timedelta | None = None) -> Token:
    """
    Create access token for authenticated user.

    Args:
        user: Authenticated user
        expires_delta: Custom expiration time

    Returns:
        Token: Token response with access and refresh tokens

    Raises:
        AuthenticationError: If token creation fails
    """
    try:
        if not jwt_handler:
            raise AuthenticationError("Authentication system not initialized")

        # Create access token
        access_token = jwt_handler.create_access_token(
            user_id=user.id,
            username=user.username,
            scopes=user.scopes,
            expires_delta=expires_delta,
        )

        # Create refresh token
        refresh_token = jwt_handler.create_refresh_token(
            user_id=user.id, username=user.username
        )

        # Calculate expires_in
        expires_in = int((expires_delta or timedelta(minutes=30)).total_seconds())

        return Token(access_token=access_token, refresh_token=refresh_token, expires_in=expires_in)

    except Exception as e:
        logger.error(f"Token creation failed: {e}", username=user.username)
        raise AuthenticationError(f"Token creation failed: {e}") from e


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """
    Get current authenticated user from JWT token.

    Args:
        credentials: HTTP bearer credentials

    Returns:
        User: Current authenticated user

    Raises:
        HTTPException: If authentication fails
    """
    try:
        if not jwt_handler:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication system not initialized",
            )

        # Validate token
        token_data = jwt_handler.validate_token(credentials.credentials)

        # Get user from database
        user = await get_user(token_data.username)
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

        if not user.is_active:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Inactive user")

        # Return public user model
        return User(
            id=user.id,
            username=user.username,
            email=user.email,
            is_active=user.is_active,
            is_verified=user.is_verified,
            scopes=user.scopes,
        )

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except Exception as e:
        logger.error(f"User authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


async def get_current_user_with_scopes(required_scopes: list[str]):
    """
    Create dependency for checking user has required scopes.

    Args:
        required_scopes: List of required scopes

    Returns:
        Dependency function
    """

    async def check_scopes(
        current_user: User = Depends(get_current_user),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> User:
        if not jwt_handler:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication system not initialized",
            )

        # Validate token and check scopes
        token_data = jwt_handler.validate_token(credentials.credentials)

        if not jwt_handler.validate_scopes(token_data, required_scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required scopes: {required_scopes}",
            )

        return current_user

    return check_scopes


async def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency for admin-only endpoints.

    Args:
        current_user: Current authenticated user

    Returns:
        User: Admin user

    Raises:
        HTTPException: If user is not admin
    """
    if "admin" not in current_user.scopes:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user


async def get_trading_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency for trading endpoints.

    Args:
        current_user: Current authenticated user

    Returns:
        User: User with trading permissions

    Raises:
        HTTPException: If user lacks trading permissions
    """
    if "trade" not in current_user.scopes and "admin" not in current_user.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Trading permissions required"
        )
    return current_user


async def create_user(
    username: str,
    email: str,
    password: str,
    scopes: list[str] | None = None,
    database_service=None,
) -> User:
    """
    Create a new user through service layer.

    Args:
        username: Username
        email: Email address
        password: Plain text password
        scopes: User permission scopes
        database_service: Deprecated parameter (ignored)

    Returns:
        User: Created user

    Raises:
        ValueError: If user creation fails
    """
    try:
        auth_service = get_auth_service()
        return await auth_service.create_user(username, email, password, scopes)

    except Exception as e:
        logger.error(f"User creation failed: {e}", username=username)
        raise ServiceError(f"User creation failed: {e}") from e


async def get_auth_summary(database_service=None) -> dict:
    """
    Get authentication system summary through service layer.

    Args:
        database_service: Deprecated parameter (ignored)

    Returns:
        dict: Authentication system status
    """
    try:
        auth_service = get_auth_service()
        summary = await auth_service.get_auth_summary()

        return {
            "initialized": jwt_handler is not None,
            **summary,
            "jwt_handler": jwt_handler.get_security_summary() if jwt_handler else None,
            "available_scopes": ["read", "write", "trade", "manage", "admin"],
            "security_features": [
                "jwt_authentication",
                "bcrypt_password_hashing",
                "scope_based_authorization",
                "token_refresh",
                "token_revocation",
                "account_lockout",
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get auth summary: {e}")
        return {
            "initialized": False,
            "total_users": 0,
            "active_users": 0,
            "admin_users": 0,
            "verified_users": 0,
        }


def require_permissions(required_permissions: list[str]):
    """
    Create a dependency that requires specific permissions.

    Args:
        required_permissions: List of required permission strings

    Returns:
        Dependency function that checks permissions
    """

    def check_permissions(current_user: User = Depends(get_current_user)) -> User:
        """
        Check if user has required permissions.

        Args:
            current_user: Current authenticated user

        Returns:
            Current user (if authorized)

        Raises:
            HTTPException: If user lacks required permissions
        """
        user_permissions = set(current_user.scopes)
        required_perms = set(required_permissions)

        # Admin users have all permissions
        if "admin" in user_permissions:
            return current_user

        # Check if user has at least one of the required permissions
        if not required_perms.intersection(user_permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required permissions: {', '.join(required_permissions)}",
            )

        return current_user

    return check_permissions
