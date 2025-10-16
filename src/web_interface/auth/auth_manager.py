"""
Authentication Manager for T-Bot Trading System.

This module provides a unified authentication manager that coordinates
different authentication providers and handles user sessions.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.base import BaseComponent
from src.core.logging import get_logger

from .models import AuthToken, User, UserStatus, create_system_roles
from .providers import APIKeyAuthProvider, AuthProvider, JWTAuthProvider, SessionAuthProvider


class AuthManager(BaseComponent):
    """Unified authentication manager."""

    def __init__(self, jwt_handler=None, config: dict[str, Any] | None = None):
        super().__init__()
        self.jwt_handler = jwt_handler
        self.config = config or {}
        self.providers: dict[str, AuthProvider] = {}
        self.default_provider = "jwt"

        # User store (in production, this would be a database)
        self.users: dict[str, User] = {}

        # Initialize providers
        self._initialize_providers()

        # Create default users
        self._create_default_users()

        # Start cleanup tasks
        self._start_cleanup_tasks()

    def configure_dependencies(self, injector):
        """Configure dependencies through DI if not provided in constructor."""
        if self.jwt_handler is None:
            try:
                self.jwt_handler = injector.resolve("JWTHandler")
            except Exception as e:
                # JWT handler is optional for fallback
                self.logger.debug(f"JWTHandler not available from DI: {e}")

    def _initialize_providers(self) -> None:
        """Initialize authentication providers."""
        # JWT Provider
        import os

        jwt_config = self.config.get("jwt", {})
        self.providers["jwt"] = JWTAuthProvider(
            secret_key=jwt_config.get(
                "secret_key", os.getenv("JWT_SECRET_KEY", "default-dev-secret-change-in-production")
            ),
            algorithm=jwt_config.get("algorithm", os.getenv("JWT_ALGORITHM", "HS256")),
            access_token_expire_minutes=jwt_config.get(
                "access_token_expire_minutes", int(os.getenv("JWT_EXPIRE_MINUTES", "30"))
            ),
            refresh_token_expire_days=jwt_config.get(
                "refresh_token_expire_days", int(os.getenv("JWT_REFRESH_EXPIRE_DAYS", "7"))
            ),
        )

        # Session Provider
        session_config = self.config.get("session", {})
        self.providers["session"] = SessionAuthProvider(
            session_timeout_minutes=session_config.get(
                "timeout_minutes", int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))
            )
        )

        # API Key Provider
        self.providers["api_key"] = APIKeyAuthProvider()

        self.logger.info("Authentication providers initialized")

    def _create_default_users(self) -> None:
        """Create default system users."""
        system_roles = create_system_roles()

        # Create admin user
        admin_user = User(
            username="admin",
            email="admin@tbot.com",
            full_name="System Administrator",
            status=UserStatus.ACTIVE,
            allocated_capital=Decimal("100000.0"),
            risk_level="high",
        )
        admin_user.add_role(system_roles["admin"])
        self.users[admin_user.user_id] = admin_user

        # Create demo trader
        trader_user = User(
            username="trader",
            email="trader@tbot.com",
            full_name="Demo Trader",
            status=UserStatus.ACTIVE,
            allocated_capital=Decimal("10000.0"),
            risk_level="medium",
        )
        trader_user.add_role(system_roles["trader"])
        self.users[trader_user.user_id] = trader_user

        # Create demo user
        demo_user = User(
            username="demo",
            email="demo@tbot.com",
            full_name="Demo User",
            status=UserStatus.ACTIVE,
            allocated_capital=Decimal("1000.0"),
            risk_level="low",
        )
        demo_user.add_role(system_roles["user"])
        self.users[demo_user.user_id] = demo_user

        self.logger.info("Default users created")

    def _start_cleanup_tasks(self) -> None:
        """Start background cleanup tasks."""
        # Only start cleanup tasks if there's a running event loop
        try:
            loop = asyncio.get_running_loop()
            # Start session cleanup task
            asyncio.create_task(self._cleanup_expired_sessions())
        except RuntimeError:
            # No running event loop - cleanup tasks will not run
            # This is normal in test environments without async context
            pass

    async def authenticate(
        self, credentials: dict[str, Any], provider_type: str | None = None
    ) -> tuple[User, AuthToken] | None:
        """
        Authenticate user with credentials.

        Args:
            credentials: Authentication credentials
            provider_type: Specific provider to use (optional)

        Returns:
            Tuple of (User, AuthToken) if successful, None otherwise
        """
        if provider_type is None:
            provider_type = self.default_provider

        provider = self.providers.get(provider_type)
        if not provider:
            self.logger.error(f"Unknown authentication provider: {provider_type}")
            return None

        try:
            # Authenticate with provider
            user = await provider.authenticate(credentials)
            if not user:
                return None

            # Check if user account is active
            if not user.is_active():
                self.logger.warning(
                    f"Authentication failed - user account not active: {user.username}"
                )
                return None

            # Create token
            token = await provider.create_token(user)

            # Update user last login
            user.reset_login_attempts()

            # Store/update user
            self.users[user.user_id] = user

            self.logger.info(f"User authenticated successfully: {user.username}")
            return user, token

        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return None

    async def validate_token(
        self, token_value: str, provider_type: str | None = None
    ) -> User | None:
        """
        Validate a token and return associated user.

        Args:
            token_value: Token to validate
            provider_type: Specific provider to use (optional)

        Returns:
            User if token is valid, None otherwise
        """
        # If no provider specified, try all providers
        providers_to_try = [provider_type] if provider_type else self.providers.keys()

        for provider_name in providers_to_try:
            provider = self.providers.get(provider_name)
            if not provider:
                continue

            try:
                user = await provider.validate_token(token_value)
                if user and user.is_active():
                    return user
            except Exception as e:
                self.logger.debug(f"Token validation failed with provider {provider_name}: {e}")
                continue

        return None

    async def revoke_token(self, token_value: str, provider_type: str | None = None) -> bool:
        """
        Revoke a token.

        Args:
            token_value: Token to revoke
            provider_type: Specific provider to use (optional)

        Returns:
            True if token was revoked, False otherwise
        """
        if provider_type is None:
            provider_type = self.default_provider

        provider = self.providers.get(provider_type)
        if not provider:
            return False

        try:
            return await provider.revoke_token(token_value)
        except Exception as e:
            self.logger.error(f"Token revocation error: {e}")
            return False

    async def refresh_token(self, refresh_token_value: str) -> AuthToken | None:
        """
        Refresh an access token using a refresh token.

        Args:
            refresh_token_value: Refresh token

        Returns:
            New access token if successful, None otherwise
        """
        jwt_provider = self.providers.get("jwt")
        if not isinstance(jwt_provider, JWTAuthProvider):
            return None

        try:
            return await jwt_provider.refresh_access_token(refresh_token_value)
        except Exception as e:
            self.logger.error(f"Token refresh error: {e}")
            return None

    async def create_api_key(self, user: User) -> AuthToken | None:
        """
        Create an API key for a user.

        Args:
            user: User to create API key for

        Returns:
            API key token if successful, None otherwise
        """
        api_key_provider = self.providers.get("api_key")
        if not api_key_provider:
            return None

        try:
            return await api_key_provider.create_token(user)
        except Exception as e:
            self.logger.error(f"API key creation error: {e}")
            return None

    def get_user(self, user_id: str) -> User | None:
        """Get a user by ID."""
        return self.users.get(user_id)

    def get_user_by_username(self, username: str) -> User | None:
        """Get a user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None

    async def create_user(self, user_data: dict[str, Any]) -> User | None:
        """
        Create a new user.

        Args:
            user_data: User data including username, email, etc.

        Returns:
            Created user if successful, None otherwise
        """
        try:
            # Check if username already exists
            existing_user = self.get_user_by_username(user_data.get("username", ""))
            if existing_user:
                self.logger.error(f"Username already exists: {user_data.get('username')}")
                return None

            # Create new user
            user = User(
                username=user_data.get("username", ""),
                email=user_data.get("email", ""),
                full_name=user_data.get("full_name", ""),
                status=UserStatus.PENDING_VERIFICATION,
                allocated_capital=Decimal(str(user_data.get("allocated_capital", "0.0"))),
                risk_level=user_data.get("risk_level", "medium"),
            )

            # Add default role
            system_roles = create_system_roles()
            default_role = user_data.get("default_role", "user")
            if default_role in system_roles:
                user.add_role(system_roles[default_role])

            # Store user
            self.users[user.user_id] = user

            self.logger.info(f"User created: {user.username}")
            return user

        except Exception as e:
            self.logger.error(f"User creation error: {e}")
            return None

    async def update_user(self, user_id: str, updates: dict[str, Any]) -> bool:
        """
        Update user information.

        Args:
            user_id: ID of user to update
            updates: Dictionary of fields to update

        Returns:
            True if successful, False otherwise
        """
        user = self.get_user(user_id)
        if not user:
            return False

        try:
            # Update allowed fields
            updatable_fields = [
                "email",
                "full_name",
                "allocated_capital",
                "max_daily_loss",
                "risk_level",
            ]

            for field, value in updates.items():
                if field in updatable_fields and hasattr(user, field):
                    setattr(user, field, value)

            self.logger.info(f"User updated: {user.username}")
            return True

        except Exception as e:
            self.logger.error(f"User update error: {e}")
            return False

    async def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """
        Change user password.

        Args:
            user_id: ID of user
            old_password: Current password
            new_password: New password

        Returns:
            True if successful, False otherwise
        """
        # In production, implement proper password hashing and validation
        # This is a placeholder implementation
        user = self.get_user(user_id)
        if not user:
            return False

        # Validate old password (placeholder logic)
        # In production, hash and compare passwords properly

        # Update password hash (placeholder)
        user.metadata["password_updated"] = datetime.now(timezone.utc)

        self.logger.info(f"Password changed for user: {user.username}")
        return True

    def get_user_stats(self) -> dict[str, Any]:
        """Get user statistics."""
        total_users = len(self.users)
        active_users = sum(1 for user in self.users.values() if user.status == UserStatus.ACTIVE)
        locked_users = sum(1 for user in self.users.values() if user.is_locked())

        return {
            "total_users": total_users,
            "active_users": active_users,
            "locked_users": locked_users,
            "pending_verification": total_users - active_users - locked_users,
        }

    async def _cleanup_expired_sessions(self) -> None:
        """Background task to cleanup expired sessions."""
        while True:
            try:
                session_provider = self.providers.get("session")
                if isinstance(session_provider, SessionAuthProvider):
                    expired_count = await session_provider.cleanup_expired_sessions()
                    if expired_count > 0:
                        self.logger.info(f"Cleaned up {expired_count} expired sessions")

                # Sleep for 5 minutes before next cleanup
                await asyncio.sleep(300)

            except asyncio.CancelledError:
                self.logger.info("Session cleanup task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Session cleanup error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute on error


# Global auth manager instance
_global_auth_manager: AuthManager | None = None


def get_auth_manager(injector=None, config: dict[str, Any] | None = None) -> AuthManager:
    """Get or create the global authentication manager using dependency injection."""
    global _global_auth_manager
    if _global_auth_manager is None:
        # Default configuration if none provided
        import os

        default_config = {
            "jwt": {
                "secret_key": os.getenv(
                    "JWT_SECRET_KEY", "default-dev-secret-change-in-production"
                ),
                "algorithm": os.getenv("JWT_ALGORITHM", "HS256"),
                "access_token_expire_minutes": int(os.getenv("JWT_EXPIRE_MINUTES", "30")),
                "refresh_token_expire_days": int(os.getenv("JWT_REFRESH_EXPIRE_DAYS", "7")),
            },
            "session": {"timeout_minutes": int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))},
        }

        auth_config = config or default_config
        jwt_handler = None

        if injector:
            try:
                # Try to get JWT handler from injector
                jwt_handler = injector.resolve("JWTHandler")
            except Exception as e:
                # JWT handler is optional, will be configured later
                get_logger(__name__).debug(f"JWTHandler not available from injector: {e}")

        _global_auth_manager = AuthManager(jwt_handler=jwt_handler, config=auth_config)

        # Configure dependencies if injector is available
        if injector and hasattr(_global_auth_manager, "configure_dependencies"):
            _global_auth_manager.configure_dependencies(injector)

    return _global_auth_manager


def initialize_auth_manager(config: dict[str, Any]) -> AuthManager:
    """Initialize the global authentication manager with custom config."""
    global _global_auth_manager
    _global_auth_manager = AuthManager(config)
    return _global_auth_manager
