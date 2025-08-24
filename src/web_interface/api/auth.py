"""
Authentication API endpoints for T-Bot web interface.

This module provides authentication endpoints including login, logout,
token refresh, and user management functionality.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel, EmailStr

from src.core.logging import get_logger
from src.web_interface.security.auth import (
    LoginRequest,
    Token,
    User,
    authenticate_user,
    create_access_token,
    create_user,
    get_admin_user,
    get_auth_summary,
    get_current_user,
    jwt_handler,
)

logger = get_logger(__name__)
router = APIRouter()
security = HTTPBearer()


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""

    refresh_token: str


class CreateUserRequest(BaseModel):
    """Create user request model."""

    username: str
    email: EmailStr
    password: str
    scopes: list[str] | None = None


class ChangePasswordRequest(BaseModel):
    """Change password request model."""

    current_password: str
    new_password: str


class AuthResponse(BaseModel):
    """Authentication response model."""

    success: bool
    message: str
    user: User | None = None
    token: Token | None = None


@router.post("/login", response_model=AuthResponse)
async def login(login_request: LoginRequest):
    """
    Authenticate user and return access token.

    Args:
        login_request: Login credentials

    Returns:
        AuthResponse: Authentication result with token

    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Authenticate user
        user = await authenticate_user(login_request.username, login_request.password)
        if not user:
            logger.warning(f"Login failed for username: {login_request.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password"
            )

        # Create access token
        token = create_access_token(user)

        # Convert to public user model
        public_user = User(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            is_active=user.is_active,
            scopes=user.scopes,
        )

        logger.info("User logged in successfully", username=user.username, user_id=user.user_id)

        return AuthResponse(success=True, message="Login successful", user=public_user, token=token)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed due to server error",
        )


@router.post("/refresh", response_model=AuthResponse)
async def refresh_token(refresh_request: RefreshTokenRequest):
    """
    Refresh access token using refresh token.

    Args:
        refresh_request: Refresh token request

    Returns:
        AuthResponse: New access and refresh tokens

    Raises:
        HTTPException: If refresh fails
    """
    try:
        if not jwt_handler:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication system not initialized",
            )

        # Refresh tokens
        new_access_token, new_refresh_token = jwt_handler.refresh_access_token(
            refresh_request.refresh_token
        )

        # Create token response
        token = Token(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            expires_in=1800,  # 30 minutes
        )

        logger.info("Token refreshed successfully")

        return AuthResponse(success=True, message="Token refreshed successfully", token=token)

    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """
    Logout user and revoke token.

    Args:
        current_user: Current authenticated user

    Returns:
        dict: Logout confirmation
    """
    try:
        # Note: In a full implementation, we would revoke the token
        # by adding it to a blacklist or removing it from a whitelist

        logger.info("User logged out", username=current_user.username, user_id=current_user.user_id)

        return {"success": True, "message": "Logout successful"}

    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Logout failed"
        )


@router.get("/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current user information.

    Args:
        current_user: Current authenticated user

    Returns:
        User: Current user information
    """
    return current_user


@router.put("/me/password")
async def change_password(
    change_request: ChangePasswordRequest, current_user: User = Depends(get_current_user)
):
    """
    Change user password.

    Args:
        change_request: Password change request
        current_user: Current authenticated user

    Returns:
        dict: Password change confirmation

    Raises:
        HTTPException: If password change fails
    """
    try:
        if not jwt_handler:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication system not initialized",
            )

        # Verify current password
        from src.web_interface.security.auth import get_user

        user_db = await get_user(current_user.username)
        if not user_db or not jwt_handler.verify_password(
            change_request.current_password, user_db.hashed_password
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect"
            )

        # Hash new password
        new_hashed_password = jwt_handler.hash_password(change_request.new_password)

        # Update password in database (simplified - in production use proper database)
        from src.web_interface.security.auth import fake_users_db

        fake_users_db[current_user.username].hashed_password = new_hashed_password

        logger.info("Password changed successfully", username=current_user.username)

        return {"success": True, "message": "Password changed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Password change failed"
        )


@router.post("/users", response_model=User)
async def create_new_user(
    user_request: CreateUserRequest, admin_user: User = Depends(get_admin_user)
):
    """
    Create a new user (admin only).

    Args:
        user_request: User creation request
        admin_user: Current admin user

    Returns:
        User: Created user information

    Raises:
        HTTPException: If user creation fails
    """
    try:
        # Create user
        user_db = await create_user(
            username=user_request.username,
            email=user_request.email,
            password=user_request.password,
            scopes=user_request.scopes or ["read"],
        )

        # Return public user model
        public_user = User(
            user_id=user_db.user_id,
            username=user_db.username,
            email=user_db.email,
            is_active=user_db.is_active,
            scopes=user_db.scopes,
        )

        logger.info(
            "User created by admin",
            created_username=user_db.username,
            admin_username=admin_user.username,
        )

        return public_user

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"User creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User creation failed"
        )


@router.get("/users")
async def list_users(admin_user: User = Depends(get_admin_user)):
    """
    List all users (admin only).

    Args:
        admin_user: Current admin user

    Returns:
        dict: List of users
    """
    try:
        from src.web_interface.security.auth import fake_users_db

        users = []
        for user_db in fake_users_db.values():
            users.append(
                {
                    "user_id": user_db.user_id,
                    "username": user_db.username,
                    "email": user_db.email,
                    "is_active": user_db.is_active,
                    "scopes": user_db.scopes,
                }
            )

        return {"users": users, "total": len(users)}

    except Exception as e:
        logger.error(f"User listing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list users"
        )


@router.get("/demo-credentials")
async def get_demo_credentials():
    """
    Get demo credentials for development environment.

    Returns:
        dict: Demo credentials (only in development mode)
    """
    import os

    # Only return demo credentials in development mode
    environment = os.getenv("ENVIRONMENT", "development")
    if environment != "development":
        return {
            "available": False,
            "message": "Demo credentials only available in development mode",
        }

    return {
        "available": True,
        "credentials": [
            {
                "username": "admin",
                "password": "admin123",
                "description": "Admin user with full access",
            },
            {
                "username": "trader1",
                "password": "trader123",
                "description": "Trader with trading permissions",
            },
            {"username": "viewer", "password": "viewer123", "description": "Read-only viewer"},
            {"username": "demo", "password": "demo123", "description": "Demo user for testing"},
        ],
        "message": "Use these credentials to login in development mode",
    }


@router.get("/status")
async def get_auth_status():
    """
    Get authentication system status.

    Returns:
        dict: Authentication system status
    """
    try:
        return await get_auth_summary()

    except Exception as e:
        logger.error(f"Auth status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get authentication status",
        )
