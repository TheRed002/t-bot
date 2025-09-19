"""
Authentication service for web interface business logic.

This service handles all authentication-related business logic that was previously
embedded in the auth module, ensuring proper separation of concerns.
"""

from datetime import datetime, timezone
from typing import Any

from src.core.base import BaseService
from src.core.exceptions import AuthenticationError, ServiceError
from src.database.models.user import User as DBUser
from src.web_interface.interfaces import WebAuthServiceInterface
from src.web_interface.security.auth import User, UserInDB


class WebAuthService(BaseService):
    """Service handling authentication business logic for web interface."""

    def __init__(self, user_repository=None):
        super().__init__("WebAuthService")
        self.user_repository = user_repository

    async def _do_start(self) -> None:
        """Start the auth service."""
        self.logger.info("Starting web auth service")

    async def _do_stop(self) -> None:
        """Stop the auth service."""
        self.logger.info("Stopping web auth service")

    async def get_user_by_username(self, username: str) -> UserInDB | None:
        """Get user by username through service layer."""
        try:
            if self.user_repository:
                user_data = await self.user_repository.get_by_username(username)
                if user_data:
                    return self._convert_db_user_to_user_in_db(user_data)
                return None
            else:
                # Mock implementation for development
                if username in ["admin", "trader", "viewer"]:
                    return UserInDB(
                        id=f"user_{username}",
                        username=username,
                        email=f"{username}@example.com",
                        password_hash="$2b$12$dummy_hash",
                        is_active=True,
                        is_verified=True,
                        is_admin=username == "admin",
                        scopes=self._get_user_scopes(username),
                        created_at=datetime.now(timezone.utc).isoformat(),
                        last_login_at=datetime.now(timezone.utc).isoformat(),
                    )
                return None

        except Exception as e:
            self.logger.error(f"Error getting user by username {username}: {e}")
            raise ServiceError(f"Failed to get user: {e}")

    async def authenticate_user(self, username: str, password: str) -> User | None:
        """Authenticate user through service layer."""
        try:
            user_in_db = await self.get_user_by_username(username)
            if not user_in_db:
                return None

            # Verify password (in real implementation, use proper hashing)
            if self._verify_password(password, user_in_db.password_hash):
                # Update last login
                await self._update_last_login(username)

                # Convert to public user model
                return User(
                    id=user_in_db.id,
                    username=user_in_db.username,
                    email=user_in_db.email,
                    is_active=user_in_db.is_active,
                    is_verified=user_in_db.is_verified,
                    scopes=user_in_db.scopes,
                )
            return None

        except Exception as e:
            self.logger.error(f"Error authenticating user {username}: {e}")
            raise ServiceError(f"Failed to authenticate user: {e}")

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        scopes: list[str] | None = None,
        is_admin: bool = False,
    ) -> User:
        """Create new user through service layer."""
        try:
            # Check if user already exists
            existing_user = await self.get_user_by_username(username)
            if existing_user:
                raise ServiceError(f"User with username '{username}' already exists")

            # Validate email uniqueness (mock validation)
            if "@" not in email:
                raise ServiceError("Invalid email format")

            scopes = scopes or ["read"]
            password_hash = self._hash_password(password)

            if self.user_repository:
                user_data = {
                    "username": username,
                    "email": email,
                    "password_hash": password_hash,
                    "is_active": True,
                    "is_verified": False,
                    "is_admin": is_admin,
                    "scopes": scopes,
                    "created_at": datetime.now(timezone.utc),
                }
                created_user = await self.user_repository.create(user_data)
                return self._convert_db_user_to_user(created_user)
            else:
                # Mock implementation for development
                return User(
                    id=f"user_{username}",
                    username=username,
                    email=email,
                    is_active=True,
                    is_verified=False,
                    scopes=scopes,
                )

        except Exception as e:
            self.logger.error(f"Error creating user {username}: {e}")
            raise ServiceError(f"Failed to create user: {e}")

    async def get_auth_summary(self) -> dict[str, Any]:
        """Get authentication system summary through service layer."""
        try:
            if self.user_repository:
                users = await self.user_repository.list_all()
                return {
                    "total_users": len(users),
                    "active_users": len([u for u in users if u.get("is_active", False)]),
                    "admin_users": len([u for u in users if u.get("is_admin", False)]),
                    "verified_users": len([u for u in users if u.get("is_verified", False)]),
                }
            else:
                # Mock data for development
                return {
                    "total_users": 3,
                    "active_users": 3,
                    "admin_users": 1,
                    "verified_users": 2,
                }

        except Exception as e:
            self.logger.error(f"Error getting auth summary: {e}")
            raise ServiceError(f"Failed to get auth summary: {e}")

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash (mock implementation)."""
        # In real implementation, use bcrypt or similar
        return password == "password" or password_hash == "$2b$12$dummy_hash"

    def _hash_password(self, password: str) -> str:
        """Hash password (mock implementation)."""
        # In real implementation, use bcrypt or similar
        return "$2b$12$dummy_hash"

    def _get_user_scopes(self, username: str) -> list[str]:
        """Get user scopes based on username (business logic)."""
        if username == "admin":
            return ["read", "write", "admin", "trading"]
        elif username == "trader":
            return ["read", "write", "trading"]
        elif username == "viewer":
            return ["read"]
        else:
            return ["read"]

    async def _update_last_login(self, username: str) -> None:
        """Update user's last login timestamp."""
        try:
            if self.user_repository:
                await self.user_repository.update_last_login(username, datetime.now(timezone.utc))
            # For mock implementation, just log
            self.logger.debug(f"Updated last login for user {username}")

        except Exception as e:
            self.logger.warning(f"Failed to update last login for {username}: {e}")

    def _convert_db_user_to_user_in_db(self, db_user: Any) -> UserInDB:
        """Convert database user model to UserInDB."""
        return UserInDB(
            id=str(db_user.id),
            username=db_user.username,
            email=db_user.email,
            password_hash=db_user.password_hash,
            is_active=db_user.is_active,
            is_verified=getattr(db_user, "is_verified", False),
            is_admin=getattr(db_user, "is_admin", False),
            scopes=getattr(db_user, "scopes", ["read"]),
            created_at=db_user.created_at.isoformat() if hasattr(db_user, "created_at") else None,
            last_login_at=db_user.last_login_at.isoformat() if hasattr(db_user, "last_login_at") else None,
        )

    def _convert_db_user_to_user(self, db_user: Any) -> User:
        """Convert database user model to User."""
        return User(
            id=str(db_user.id),
            username=db_user.username,
            email=db_user.email,
            is_active=db_user.is_active,
            is_verified=getattr(db_user, "is_verified", False),
            scopes=getattr(db_user, "scopes", ["read"]),
        )

    def health_check(self) -> dict[str, Any]:
        """Perform health check and return status."""
        return {
            "service": "WebAuthService",
            "status": "healthy",
            "user_repository_available": self.user_repository is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_user_roles(self, current_user: Any) -> list[str]:
        """Extract user roles from current_user object (business logic)."""
        if isinstance(current_user, dict):
            roles = current_user.get("roles", [])
            role = current_user.get("role", "")
            if not roles and role:
                roles = [role]
            return roles
        elif hasattr(current_user, "roles"):
            return getattr(current_user, "roles", [])
        elif hasattr(current_user, "scopes"):
            return getattr(current_user, "scopes", [])
        else:
            return []

    def check_permission(self, current_user: Any, required_roles: list[str]) -> bool:
        """Check if user has required permissions (business logic)."""
        user_roles = self.get_user_roles(current_user)
        return any(role in user_roles for role in required_roles)

    def require_permission(self, current_user: Any, required_roles: list[str]) -> None:
        """Require user to have specific permissions or raise ServiceError."""
        if not self.check_permission(current_user, required_roles):
            user_roles = self.get_user_roles(current_user)
            raise ServiceError(
                f"Insufficient permissions. User roles: {user_roles}, Required: {required_roles}"
            )

    def require_admin(self, current_user: Any) -> None:
        """Require user to be admin or raise ServiceError."""
        self.require_permission(current_user, ["admin"])

    def require_trading_permission(self, current_user: Any) -> None:
        """Require user to have trading permissions or raise ServiceError."""
        self.require_permission(current_user, ["admin", "trader", "trading"])

    def require_risk_manager_permission(self, current_user: Any) -> None:
        """Require user to have risk management permissions or raise ServiceError."""
        self.require_permission(current_user, ["admin", "risk_manager"])

    def require_admin_or_developer_permission(self, current_user: Any) -> None:
        """Require user to have admin or developer permissions or raise ServiceError."""
        self.require_permission(current_user, ["admin", "developer"])

    def require_management_permission(self, current_user: Any) -> None:
        """Require user to have management permissions or raise ServiceError."""
        self.require_permission(current_user, ["admin", "trader", "manager"])

    def require_treasurer_permission(self, current_user: Any) -> None:
        """Require user to have treasurer permissions or raise ServiceError."""
        self.require_permission(current_user, ["admin", "treasurer"])

    def require_operator_permission(self, current_user: Any) -> None:
        """Require user to have operator permissions or raise ServiceError."""
        self.require_permission(current_user, ["admin", "operator"])

    def get_service_info(self) -> dict[str, Any]:
        """Get service information and capabilities."""
        return {
            "service": "WebAuthService",
            "description": "Web authentication service handling user management business logic",
            "capabilities": [
                "user_authentication",
                "user_creation",
                "user_lookup",
                "auth_summary",
                "password_verification",
                "role_based_authorization",
                "permission_checking",
            ],
            "version": "1.0.0",
        }