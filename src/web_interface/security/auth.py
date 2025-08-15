"""
Authentication logic for T-Bot web interface.

This module provides core authentication functions including user verification,
token creation, and dependency injection for protected endpoints.
"""

from datetime import timedelta

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.core.config import Config
from src.core.exceptions import AuthenticationError
from src.core.logging import get_logger

from .jwt_handler import JWTHandler

logger = get_logger(__name__)

# Global instances (initialized by app startup)
jwt_handler: JWTHandler | None = None
security = HTTPBearer()


class UserInDB(BaseModel):
    """User model for database storage."""

    user_id: str
    username: str
    email: str
    hashed_password: str
    is_active: bool = True
    scopes: list[str] = ["read"]
    created_at: str | None = None
    last_login: str | None = None


class User(BaseModel):
    """User model for API responses."""

    user_id: str
    username: str
    email: str
    is_active: bool
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


# Mock user database (in production, use actual database)
fake_users_db = {
    "admin": UserInDB(
        user_id="admin-001",
        username="admin",
        email="admin@tbot.com",
        hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        scopes=["admin", "read", "write", "trade", "manage"],
    ),
    "trader": UserInDB(
        user_id="trader-001",
        username="trader",
        email="trader@tbot.com",
        hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        scopes=["read", "write", "trade"],
    ),
    "viewer": UserInDB(
        user_id="viewer-001",
        username="viewer",
        email="viewer@tbot.com",
        hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        scopes=["read"],
    ),
}


def init_auth(config: Config) -> None:
    """
    Initialize authentication system.

    Args:
        config: Application configuration
    """
    global jwt_handler
    jwt_handler = JWTHandler(config)
    logger.info("Authentication system initialized")


def get_user(username: str) -> UserInDB | None:
    """
    Get user by username.

    Args:
        username: Username to lookup

    Returns:
        UserInDB: User data or None if not found
    """
    return fake_users_db.get(username)


def authenticate_user(username: str, password: str) -> UserInDB | None:
    """
    Authenticate user credentials.

    Args:
        username: Username
        password: Plain text password

    Returns:
        UserInDB: Authenticated user or None if authentication fails
    """
    try:
        user = get_user(username)
        if not user:
            logger.warning("Authentication failed: user not found", username=username)
            return None

        if not user.is_active:
            logger.warning("Authentication failed: user inactive", username=username)
            return None

        if not jwt_handler.verify_password(password, user.hashed_password):
            logger.warning("Authentication failed: invalid password", username=username)
            return None

        logger.info("User authenticated successfully", username=username)
        return user

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
            user_id=user.user_id,
            username=user.username,
            scopes=user.scopes,
            expires_delta=expires_delta,
        )

        # Create refresh token
        refresh_token = jwt_handler.create_refresh_token(
            user_id=user.user_id, username=user.username
        )

        # Calculate expires_in
        expires_in = int((expires_delta or timedelta(minutes=30)).total_seconds())

        return Token(access_token=access_token, refresh_token=refresh_token, expires_in=expires_in)

    except Exception as e:
        logger.error(f"Token creation failed: {e}", username=user.username)
        raise AuthenticationError(f"Token creation failed: {e}")


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
        user = get_user(token_data.username)
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

        if not user.is_active:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Inactive user")

        # Return public user model
        return User(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            is_active=user.is_active,
            scopes=user.scopes,
        )

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"User authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


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


def create_user(username: str, email: str, password: str, scopes: list[str] = None) -> UserInDB:
    """
    Create a new user (for admin functionality).

    Args:
        username: Username
        email: Email address
        password: Plain text password
        scopes: User permission scopes

    Returns:
        UserInDB: Created user

    Raises:
        ValueError: If user creation fails
    """
    try:
        if username in fake_users_db:
            raise ValueError(f"User {username} already exists")

        if not jwt_handler:
            raise ValueError("Authentication system not initialized")

        # Hash password
        hashed_password = jwt_handler.hash_password(password)

        # Create user
        user = UserInDB(
            user_id=f"{username}-{len(fake_users_db) + 1:03d}",
            username=username,
            email=email,
            hashed_password=hashed_password,
            scopes=scopes or ["read"],
        )

        # Store in database
        fake_users_db[username] = user

        logger.info("User created successfully", username=username, scopes=user.scopes)
        return user

    except Exception as e:
        logger.error(f"User creation failed: {e}", username=username)
        raise ValueError(f"User creation failed: {e}")


def get_auth_summary() -> dict:
    """
    Get authentication system summary.

    Returns:
        dict: Authentication system status
    """
    return {
        "initialized": jwt_handler is not None,
        "total_users": len(fake_users_db),
        "active_users": sum(1 for user in fake_users_db.values() if user.is_active),
        "jwt_handler": jwt_handler.get_security_summary() if jwt_handler else None,
        "available_scopes": ["read", "write", "trade", "manage", "admin"],
        "security_features": [
            "jwt_authentication",
            "bcrypt_password_hashing",
            "scope_based_authorization",
            "token_refresh",
            "token_revocation",
        ],
    }
