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
from src.core.exceptions import AuthenticationError
from src.core.logging import get_logger
from src.database.models.user import User as DBUser
from src.database.service import DatabaseService

from .jwt_handler import JWTHandler

logger = get_logger(__name__)


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
        user_id=str(db_user.id),
        username=db_user.username,
        email=db_user.email,
        hashed_password=db_user.password_hash,
        is_active=db_user.is_active,
        scopes=db_user.scopes if db_user.scopes else ["read"],
        created_at=db_user.created_at.isoformat() if db_user.created_at else None,
        last_login=db_user.last_login_at.isoformat() if db_user.last_login_at else None,
    )


async def get_user(username: str, database_service: DatabaseService = None) -> UserInDB | None:
    """
    Get user by username from database.

    Args:
        username: Username to lookup
        database_service: Optional database service (will use DI if not provided)

    Returns:
        UserInDB: User data or None if not found
    """
    try:
        # Get database service from DI if not provided
        if database_service is None:
            from src.core.dependency_injection import DependencyInjector

            injector = DependencyInjector.get_instance()
            database_service = injector.resolve("DatabaseService")
            should_stop_service = False  # Service managed by DI container
        else:
            should_stop_service = True  # Service was explicitly passed

        try:
            # Query users by username filter
            users = await database_service.list_entities(
                model_class=DBUser, filters={"username": username}, limit=1
            )

            if not users:
                return None

            db_user = users[0]
            if not db_user.is_active:
                return None

            return _convert_db_user_to_user_in_db(db_user)
        finally:
            if should_stop_service:
                await database_service.stop()

    except Exception as e:
        logger.error(f"Database error getting user {username}: {e}")
        return None


async def authenticate_user(
    username: str, password: str, database_service: DatabaseService = None
) -> UserInDB | None:
    """
    Authenticate user credentials with security measures.

    Args:
        username: Username
        password: Plain text password
        database_service: Optional database service (will use DI if not provided)

    Returns:
        UserInDB: Authenticated user or None if authentication fails
    """
    try:
        # Get database service from DI if not provided
        if database_service is None:
            from src.core.dependency_injection import DependencyInjector

            injector = DependencyInjector.get_instance()
            database_service = injector.resolve("DatabaseService")
            should_stop_service = False
        else:
            should_stop_service = True

        try:
            # Query users by username filter
            users = await database_service.list_entities(
                model_class=DBUser, filters={"username": username}, limit=1
            )

            if not users:
                logger.warning("Authentication failed: user not found", username=username)
                return None

            db_user = users[0]

            # Check if account is locked
            if db_user.locked_until and datetime.now(timezone.utc) < db_user.locked_until:
                logger.warning("Authentication failed: account locked", username=username)
                return None

            # Check if user is active
            if not db_user.is_active:
                logger.warning("Authentication failed: user inactive", username=username)
                return None

            # Verify password
            if not jwt_handler.verify_password(password, db_user.password_hash):
                logger.warning("Authentication failed: invalid password", username=username)

                # Increment failed login attempts
                db_user.failed_login_attempts = (db_user.failed_login_attempts or 0) + 1

                # Lock account after 5 failed attempts for 15 minutes
                if db_user.failed_login_attempts >= 5:
                    db_user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=15)
                    logger.warning(
                        f"Account locked due to {db_user.failed_login_attempts} failed attempts",
                        username=username,
                    )

                # Update user using service
                await database_service.update_entity(db_user)
                return None

            # Reset failed attempts on successful login
            db_user.failed_login_attempts = 0
            db_user.locked_until = None
            db_user.last_login_at = datetime.now(timezone.utc)

            # Update user using service
            await database_service.update_entity(db_user)

            user_in_db = _convert_db_user_to_user_in_db(db_user)
            logger.info("User authenticated successfully", username=username)
            return user_in_db
        finally:
            if should_stop_service:
                await database_service.stop()

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
    database_service: DatabaseService = None,
) -> UserInDB:
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
        if not jwt_handler:
            raise ValueError("Authentication system not initialized")

        # Get database service from DI if not provided
        if database_service is None:
            from src.core.dependency_injection import DependencyInjector

            injector = DependencyInjector.get_instance()
            database_service = injector.resolve("DatabaseService")
            should_stop_service = False
        else:
            should_stop_service = True

        try:
            # Check if user exists
            existing_users = await database_service.list_entities(
                model_class=DBUser, filters={"username": username}, limit=1
            )
            if existing_users:
                raise ValueError(f"User {username} already exists")

            # Hash password
            hashed_password = jwt_handler.hash_password(password)

            # Create user in database
            new_user = DBUser(
                username=username,
                email=email,
                password_hash=hashed_password,
                scopes=scopes or ["read"],
                is_active=True,
                is_verified=False,
            )

            created_user = await database_service.create_entity(new_user)

            user_in_db = _convert_db_user_to_user_in_db(created_user)
            logger.info("User created successfully", username=username, scopes=user_in_db.scopes)
            return user_in_db
        finally:
            if should_stop_service:
                await database_service.stop()

    except Exception as e:
        logger.error(f"User creation failed: {e}", username=username)
        raise ValueError(f"User creation failed: {e}") from e


async def get_auth_summary(database_service: DatabaseService = None) -> dict:
    """
    Get authentication system summary.

    Returns:
        dict: Authentication system status
    """
    try:
        total_users = 0
        active_users = 0

        # Get database service from DI if not provided
        if database_service is None:
            from src.core.dependency_injection import DependencyInjector

            injector = DependencyInjector.get_instance()
            database_service = injector.resolve("DatabaseService")
            should_stop_service = False
        else:
            should_stop_service = True

        try:
            all_users = await database_service.list_entities(model_class=DBUser)
            total_users = len(all_users)
            active_users = sum(1 for user in all_users if user.is_active)
        finally:
            if should_stop_service:
                await database_service.stop()
    except Exception as e:
        logger.error(f"Failed to get user stats: {e}")

    return {
        "initialized": jwt_handler is not None,
        "total_users": total_users,
        "active_users": active_users,
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
