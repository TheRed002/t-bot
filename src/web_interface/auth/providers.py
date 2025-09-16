"""
Authentication providers for T-Bot Trading System.

This module implements various authentication providers including
JWT tokens, session-based auth, and API key authentication.
"""

import secrets
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from src.core.base import BaseComponent

from .models import AuthToken, TokenType, User, UserStatus


class AuthProvider(ABC):
    """Abstract base class for authentication providers."""

    @abstractmethod
    async def authenticate(self, credentials: dict[str, Any]) -> User | None:
        """Authenticate user with provided credentials."""
        pass

    @abstractmethod
    async def create_token(self, user: User) -> AuthToken:
        """Create an authentication token for the user."""
        pass

    @abstractmethod
    async def validate_token(self, token_value: str) -> User | None:
        """Validate a token and return the associated user."""
        pass

    @abstractmethod
    async def revoke_token(self, token_value: str) -> bool:
        """Revoke a token."""
        pass


class JWTAuthProvider(AuthProvider, BaseComponent):
    """JWT-based authentication provider."""

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
    ):
        super().__init__()
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days

        # In-memory token storage (in production, use Redis or database)
        self.active_tokens: dict[str, AuthToken] = {}
        self.revoked_tokens: set = set()

    async def authenticate(self, credentials: dict[str, Any]) -> User | None:
        """Authenticate user with username/password credentials."""
        username = credentials.get("username")
        password = credentials.get("password")

        if not username or not password:
            return None

        # SECURITY: All hardcoded credentials removed
        # Use proper database authentication through auth.py
        from ..security.auth import authenticate_user

        try:
            authenticated_user = await authenticate_user(username, password)
            if authenticated_user:
                # Convert to provider User model
                user = User(
                    username=authenticated_user.username,
                    email=authenticated_user.email,
                    full_name=authenticated_user.username,  # Use username as fallback
                    status=(
                        UserStatus.ACTIVE if authenticated_user.is_active else UserStatus.INACTIVE
                    ),
                )

                # Add roles based on scopes
                from .models import create_system_roles

                system_roles = create_system_roles()

                for scope in authenticated_user.scopes:
                    if scope in system_roles:
                        user.add_role(system_roles[scope])

                return user
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")

        return None

    async def create_token(self, user: User) -> AuthToken:
        """Create a JWT access token for the user."""
        # Create access token
        access_expires = datetime.now(timezone.utc) + timedelta(
            minutes=self.access_token_expire_minutes
        )
        access_payload = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": [role.name for role in user.roles],
            "exp": access_expires,
            "iat": datetime.now(timezone.utc),
            "type": "access",
        }

        access_token_value = jwt.encode(access_payload, self.secret_key, algorithm=self.algorithm)

        access_token = AuthToken(
            user_id=user.user_id,
            token_type=TokenType.ACCESS,
            token_value=access_token_value,
            expires_at=access_expires,
            metadata={"username": user.username, "roles": [role.name for role in user.roles]},
        )

        # Store token
        self.active_tokens[access_token_value] = access_token

        return access_token

    async def create_refresh_token(self, user: User) -> AuthToken:
        """Create a JWT refresh token for the user."""
        refresh_expires = datetime.now(timezone.utc) + timedelta(
            days=self.refresh_token_expire_days
        )
        refresh_payload = {
            "user_id": user.user_id,
            "username": user.username,
            "exp": refresh_expires,
            "iat": datetime.now(timezone.utc),
            "type": "refresh",
        }

        refresh_token_value = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)

        refresh_token = AuthToken(
            user_id=user.user_id,
            token_type=TokenType.REFRESH,
            token_value=refresh_token_value,
            expires_at=refresh_expires,
            metadata={"username": user.username},
        )

        # Store token
        self.active_tokens[refresh_token_value] = refresh_token

        return refresh_token

    async def validate_token(self, token_value: str) -> User | None:
        """Validate a JWT token and return the associated user."""
        try:
            # Check if token is revoked
            if token_value in self.revoked_tokens:
                return None

            # Decode and validate JWT
            payload = jwt.decode(token_value, self.secret_key, algorithms=[self.algorithm])

            # Check token type
            if payload.get("type") != "access":
                return None

            # Get user information from payload
            user_id = payload.get("user_id")
            username = payload.get("username")
            role_names = payload.get("roles", [])

            if not user_id or not username:
                return None

            # Create user object from token data
            # In production, fetch full user data from database
            from .models import create_system_roles

            system_roles = create_system_roles()

            user = User(user_id=user_id, username=username, status=UserStatus.ACTIVE)

            # Add roles from token
            for role_name in role_names:
                if role_name in system_roles:
                    user.add_role(system_roles[role_name])

            # Update token last used time
            if token_value in self.active_tokens:
                self.active_tokens[token_value].touch()

            return user

        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token expired")
            # Remove expired token from active tokens
            self.active_tokens.pop(token_value, None)
            return None

        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid JWT token: {e}")
            return None

        except Exception as e:
            self.logger.error(f"Error validating JWT token: {e}")
            return None

    async def revoke_token(self, token_value: str) -> bool:
        """Revoke a JWT token."""
        try:
            # Add to revoked tokens set
            self.revoked_tokens.add(token_value)

            # Remove from active tokens
            if token_value in self.active_tokens:
                token = self.active_tokens.pop(token_value)
                token.revoke()

            return True

        except Exception as e:
            self.logger.error(f"Error revoking token: {e}")
            return False

    async def refresh_access_token(self, refresh_token_value: str) -> AuthToken | None:
        """Create a new access token using a refresh token."""
        try:
            # Validate refresh token
            payload = jwt.decode(refresh_token_value, self.secret_key, algorithms=[self.algorithm])

            if payload.get("type") != "refresh":
                return None

            user_id = payload.get("user_id")
            username = payload.get("username")

            if not user_id or not username:
                return None

            # Check if refresh token is still active
            if refresh_token_value in self.revoked_tokens:
                return None

            # Create new access token
            # In production, fetch full user data from database
            from .models import create_system_roles

            system_roles = create_system_roles()

            user = User(user_id=user_id, username=username, status=UserStatus.ACTIVE)

            # Add appropriate role (this would come from database in production)
            if username == "admin":
                user.add_role(system_roles["admin"])
            elif username == "trader":
                user.add_role(system_roles["trader"])
            else:
                user.add_role(system_roles["user"])

            return await self.create_token(user)

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception as e:
            self.logger.error(f"Error refreshing token: {e}")
            return None


class SessionAuthProvider(AuthProvider, BaseComponent):
    """Session-based authentication provider."""

    def __init__(self, session_timeout_minutes: int = 60):
        super().__init__()
        self.session_timeout_minutes = session_timeout_minutes
        self.active_sessions: dict[str, dict[str, Any]] = {}

    async def authenticate(self, credentials: dict[str, Any]) -> User | None:
        """Authenticate user with credentials (delegates to password checking)."""
        # This would typically check username/password against database
        # For demo, using simple hardcoded users
        return await self._check_credentials(credentials)

    async def _check_credentials(self, credentials: dict[str, Any]) -> User | None:
        """Check user credentials."""
        username = credentials.get("username")
        password = credentials.get("password")

        # SECURITY: All hardcoded credentials removed
        # Use proper database authentication
        from ..security.auth import authenticate_user

        try:
            authenticated_user = await authenticate_user(username, password)
            if authenticated_user:
                # Convert to provider User model
                user = User(
                    username=authenticated_user.username,
                    email=authenticated_user.email,
                    full_name=authenticated_user.username,
                    status=(
                        UserStatus.ACTIVE if authenticated_user.is_active else UserStatus.INACTIVE
                    ),
                )

                # Add roles based on scopes
                from .models import create_system_roles

                system_roles = create_system_roles()

                for scope in authenticated_user.scopes:
                    if scope in system_roles:
                        user.add_role(system_roles[scope])

                return user
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")

        return None

    async def create_token(self, user: User) -> AuthToken:
        """Create a session token."""
        # Generate session ID
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=self.session_timeout_minutes)

        # Store session data
        session_data = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": [role.name for role in user.roles],
            "created_at": datetime.now(timezone.utc),
            "expires_at": expires_at,
            "last_activity": datetime.now(timezone.utc),
        }

        self.active_sessions[session_id] = session_data

        return AuthToken(
            user_id=user.user_id,
            token_type=TokenType.ACCESS,
            token_value=session_id,
            expires_at=expires_at,
            metadata={"session_data": session_data},
        )

    async def validate_token(self, token_value: str) -> User | None:
        """Validate a session token."""
        session_data = self.active_sessions.get(token_value)

        if not session_data:
            return None

        # Check expiration
        if datetime.now(timezone.utc) > session_data["expires_at"]:
            # Remove expired session
            self.active_sessions.pop(token_value, None)
            return None

        # Update last activity
        session_data["last_activity"] = datetime.now(timezone.utc)

        # Create user object from session data
        from .models import create_system_roles

        system_roles = create_system_roles()

        user = User(
            user_id=session_data["user_id"],
            username=session_data["username"],
            status=UserStatus.ACTIVE,
        )

        # Add roles from session
        for role_name in session_data["roles"]:
            if role_name in system_roles:
                user.add_role(system_roles[role_name])

        return user

    async def revoke_token(self, token_value: str) -> bool:
        """Revoke a session token."""
        if token_value in self.active_sessions:
            del self.active_sessions[token_value]
            return True
        return False

    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions and return count."""
        now = datetime.now(timezone.utc)
        expired_sessions = [
            session_id
            for session_id, data in self.active_sessions.items()
            if now > data["expires_at"]
        ]

        for session_id in expired_sessions:
            del self.active_sessions[session_id]

        return len(expired_sessions)


class APIKeyAuthProvider(AuthProvider, BaseComponent):
    """API key-based authentication provider."""

    def __init__(self):
        super().__init__()
        # In production, store in database
        self.api_keys: dict[str, dict[str, Any]] = {}

    async def authenticate(self, credentials: dict[str, Any]) -> User | None:
        """Authenticate using API key."""
        api_key = credentials.get("api_key")
        if not api_key:
            return None

        return await self.validate_api_key(api_key)

    async def create_token(self, user: User) -> AuthToken:
        """Create an API key token."""
        api_key = self._generate_api_key()

        # Store API key data
        self.api_keys[api_key] = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": [role.name for role in user.roles],
            "created_at": datetime.now(timezone.utc),
            "last_used": None,
            "is_active": True,
        }

        return AuthToken(
            user_id=user.user_id,
            token_type=TokenType.API_KEY,
            token_value=api_key,
            metadata={"api_key_data": self.api_keys[api_key]},
        )

    async def validate_token(self, token_value: str) -> User | None:
        """Validate an API key token."""
        return await self.validate_api_key(token_value)

    async def validate_api_key(self, api_key: str) -> User | None:
        """Validate an API key."""
        key_data = self.api_keys.get(api_key)

        if not key_data or not key_data["is_active"]:
            return None

        # Update last used time
        key_data["last_used"] = datetime.now(timezone.utc)

        # Create user object from API key data
        from .models import create_system_roles

        system_roles = create_system_roles()

        user = User(
            user_id=key_data["user_id"], username=key_data["username"], status=UserStatus.ACTIVE
        )

        # Add roles from API key
        for role_name in key_data["roles"]:
            if role_name in system_roles:
                user.add_role(system_roles[role_name])

        return user

    async def revoke_token(self, token_value: str) -> bool:
        """Revoke an API key."""
        if token_value in self.api_keys:
            self.api_keys[token_value]["is_active"] = False
            return True
        return False

    def _generate_api_key(self) -> str:
        """Generate a new API key."""
        # Create a secure random API key
        prefix = "tbot_"
        key_part = secrets.token_urlsafe(32)
        return f"{prefix}{key_part}"
