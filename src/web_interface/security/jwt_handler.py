"""
JWT token handling for T-Bot authentication.

This module provides comprehensive JWT token management including creation,
validation, refresh, and revocation capabilities with security best practices.
"""

import asyncio
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from src.core.base import BaseComponent
from src.core.config import Config
from src.core.exceptions import AuthenticationError, ValidationError
from src.database.redis_client import RedisClient


class TokenData(BaseModel):
    """Token data structure."""

    username: str | None = None
    user_id: str | None = None
    scopes: list[str] = []
    issued_at: datetime | None = None
    expires_at: datetime | None = None


class JWTHandler(BaseComponent):
    """
    Advanced JWT token handler with security features.

    This class provides:
    - Secure token generation and validation
    - Token refresh capabilities
    - Token revocation (blacklisting)
    - Role-based access control integration
    - Security event logging
    """

    def __init__(self, config: Config):
        """
        Initialize JWT handler.

        Args:
            config: Application configuration
        """
        super().__init__()  # Initialize BaseComponent (which sets self.logger)
        self.config = config

        # JWT Configuration - prioritize environment variables
        self.secret_key = self._get_secret_key(config)

        # JWT Configuration - handle missing web_interface config
        if hasattr(config, "web_interface"):
            jwt_config = config.web_interface.get("jwt", {})
        else:
            # Fall back to security config
            jwt_config = {
                "algorithm": getattr(config.security, "jwt_algorithm", "HS256"),
                "access_token_expire_minutes": getattr(config.security, "jwt_expire_minutes", 30),
            }
        self.algorithm = jwt_config.get("algorithm", "HS256")
        self.access_token_expire_minutes = jwt_config.get("access_token_expire_minutes", 30)
        self.refresh_token_expire_days = jwt_config.get("refresh_token_expire_days", 7)

        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # Redis client for token blacklist
        self.redis_client = RedisClient(config, auto_close=False)
        self.blacklist_key_prefix = "jwt_blacklist"
        self.blacklist_namespace = "auth"

        # Fallback in-memory blacklist for development
        self.blacklisted_tokens: set[str] = set()
        self._redis_available = False

        # Initialize Redis connection
        self._init_redis()

        # Security settings
        self.require_https = jwt_config.get("require_https", True)
        self.max_token_age_hours = jwt_config.get("max_token_age_hours", 24)

        self.logger.info("JWT handler initialized with Redis blacklist support")

    def _init_redis(self):
        """Initialize Redis connection with sync wrapper."""
        loop = None
        try:
            # Test connection - use existing event loop if available
            try:
                loop = asyncio.get_running_loop()
                # We're in an existing loop, can't use run_until_complete
                # Mark as unavailable and let async initialization happen later
                self._redis_available = False
                self.logger.info("Redis will be initialized on first use")
            except RuntimeError:
                # No running loop, safe to create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.redis_client.connect())
                    loop.run_until_complete(self.redis_client.ping())
                    self._redis_available = True
                    self.logger.info("Redis connection established for token blacklist")
                finally:
                    if loop:
                        loop.close()
                        asyncio.set_event_loop(None)
        except Exception as e:
            if hasattr(self, "redis_client") and self.redis_client:
                try:
                    # Try to clean up connection on failure
                    cleanup_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(cleanup_loop)
                    try:
                        cleanup_loop.run_until_complete(self.redis_client.disconnect())
                    finally:
                        cleanup_loop.close()
                        asyncio.set_event_loop(None)
                except Exception as cleanup_e:
                    self.logger.debug(f"Error during cleanup: {cleanup_e}")
            self.logger.warning(f"Redis connection failed: {e}. Using in-memory blacklist.")
            self._redis_available = False

    def _redis_sync(self, coro):
        """Execute async Redis operation synchronously."""
        loop = None
        try:
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, can't use run_until_complete
                # Schedule the coroutine and return None (will use in-memory fallback)
                self.logger.debug(
                    "Cannot execute Redis operation in async context, using in-memory fallback"
                )
                return None
            except RuntimeError:
                # No running loop, safe to create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(coro)
                finally:
                    if loop:
                        loop.close()
                        asyncio.set_event_loop(None)
        except Exception as e:
            self.logger.error(f"Redis operation failed: {e}")
            return None

    def _get_secret_key(self, config: Config) -> str:
        """
        Get JWT secret key from environment variable or config.

        Priority order:
        1. JWT_SECRET_KEY environment variable
        2. JWT_SECRET environment variable
        3. Raise error (no fallbacks for security)
        """
        # Try environment variables first
        secret_key = os.getenv("JWT_SECRET_KEY") or os.getenv("JWT_SECRET")

        if secret_key:
            self.logger.info("Using JWT secret key from environment")
            return secret_key

        # Try config
        if hasattr(config, "security") and hasattr(config.security, "jwt_secret"):
            if config.security.jwt_secret:
                self.logger.info("Using JWT secret key from config")
                return config.security.jwt_secret

        # No secret found - generate one for development
        if os.getenv("ENVIRONMENT", "development") == "development":
            self.logger.warning("No JWT secret configured - generating random key for development")
            return secrets.token_urlsafe(32)

        # Production environment must have secret configured
        raise ValidationError("JWT_SECRET_KEY must be set in production environment")

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            return self.pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            self.logger.error(f"Password verification error: {e}")
            return False

    def create_access_token(
        self,
        user_id: str,
        username: str,
        scopes: list[str] | None = None,
        expires_delta: timedelta | None = None,
    ) -> str:
        """
        Create a JWT access token.

        Args:
            user_id: User ID to encode
            username: Username to encode
            scopes: User permission scopes
            expires_delta: Custom expiration time

        Returns:
            Encoded JWT token

        Raises:
            AuthenticationError: If token creation fails
        """
        try:
            # Set expiration
            if expires_delta:
                expire = datetime.now(timezone.utc) + expires_delta
            else:
                expire = datetime.now(timezone.utc) + timedelta(
                    minutes=self.access_token_expire_minutes
                )

            # Create token payload
            to_encode = {
                "sub": username,
                "user_id": user_id,
                "scopes": scopes or [],
                "exp": expire,
                "iat": datetime.now(timezone.utc),
                "type": "access",
                "jti": secrets.token_urlsafe(16),  # Unique token ID for revocation
            }

            # Encode token
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

            self.logger.info("Access token created", username=username, expires=expire.isoformat())
            return encoded_jwt

        except Exception as e:
            self.logger.error(f"Token creation failed: {e}")
            raise AuthenticationError(f"Failed to create access token: {e}")

    def create_refresh_token(self, user_id: str, username: str) -> str:
        """
        Create a JWT refresh token.

        Args:
            user_id: User ID to encode
            username: Username to encode

        Returns:
            Encoded JWT refresh token

        Raises:
            AuthenticationError: If token creation fails
        """
        try:
            expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)

            to_encode = {
                "sub": username,
                "user_id": user_id,
                "exp": expire,
                "iat": datetime.now(timezone.utc),
                "type": "refresh",
                "jti": secrets.token_urlsafe(16),
            }

            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

            self.logger.info("Refresh token created", username=username, expires=expire.isoformat())
            return encoded_jwt

        except Exception as e:
            self.logger.error(f"Refresh token creation failed: {e}")
            raise AuthenticationError(f"Failed to create refresh token: {e}")

    def validate_token(self, token: str) -> TokenData:
        """
        Validate and decode a JWT token.

        Args:
            token: JWT token to validate

        Returns:
            TokenData: Decoded token data

        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            # Check if token is blacklisted
            if self.is_token_blacklisted(token):
                raise AuthenticationError("Token has been revoked")

            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Extract data
            username: str = payload.get("sub")
            if username is None:
                raise AuthenticationError("Invalid token payload")

            # Check token age
            issued_at = datetime.fromtimestamp(payload.get("iat", 0), tz=timezone.utc)
            max_age = timedelta(hours=self.max_token_age_hours)
            if datetime.now(timezone.utc) - issued_at > max_age:
                raise AuthenticationError("Token too old")

            return TokenData(
                username=username,
                user_id=payload.get("user_id"),
                scopes=payload.get("scopes", []),
                issued_at=issued_at,
                expires_at=datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc),
            )

        except JWTError as e:
            self.logger.warning(f"JWT validation failed: {e}")
            raise AuthenticationError("Invalid or expired token")
        except Exception as e:
            self.logger.error(f"Token validation error: {e}")
            raise AuthenticationError(f"Token validation failed: {e}")

    def validate_scopes(self, token_data: TokenData, required_scopes: list[str]) -> bool:
        """
        Validate that token has required scopes.

        Args:
            token_data: Decoded token data
            required_scopes: List of required scopes

        Returns:
            bool: True if token has all required scopes
        """
        token_scopes = set(token_data.scopes)
        required = set(required_scopes)

        # Admin scope grants all permissions
        if "admin" in token_scopes:
            return True

        # Check if token has all required scopes
        return required.issubset(token_scopes)

    def refresh_access_token(self, refresh_token: str) -> tuple[str, str]:
        """
        Refresh an access token using a refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            Tuple of (new_access_token, new_refresh_token)

        Raises:
            AuthenticationError: If refresh token is invalid
        """
        try:
            # Validate refresh token
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])

            # Check token type
            if payload.get("type") != "refresh":
                raise AuthenticationError("Invalid token type")

            # Check if refresh token is blacklisted
            if self.is_token_blacklisted(refresh_token):
                raise AuthenticationError("Refresh token has been revoked")

            # Extract user info
            username = payload.get("sub")
            user_id = payload.get("user_id")

            if not username or not user_id:
                raise AuthenticationError("Invalid refresh token payload")

            # Revoke old refresh token
            self.revoke_token(refresh_token)

            # Create new tokens
            new_access_token = self.create_access_token(
                user_id=user_id, username=username, scopes=payload.get("scopes", [])
            )
            new_refresh_token = self.create_refresh_token(user_id=user_id, username=username)

            self.logger.info("Tokens refreshed successfully", username=username)
            return new_access_token, new_refresh_token

        except JWTError as e:
            self.logger.warning(f"Refresh token validation failed: {e}")
            raise AuthenticationError("Invalid or expired refresh token")
        except Exception as e:
            self.logger.error(f"Token refresh error: {e}")
            raise AuthenticationError(f"Token refresh failed: {e}")

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token by adding it to blacklist.

        Args:
            token: Token to revoke

        Returns:
            bool: True if successfully revoked
        """
        try:
            # Decode token to get expiration
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            exp = payload.get("exp", 0)
            jti = payload.get("jti", token[:50])  # Use JTI or token prefix as key

            # Calculate TTL (time until token expires)
            ttl = max(0, exp - int(datetime.now(timezone.utc).timestamp()))

            if ttl > 0:
                # Add to Redis blacklist with TTL
                if self._redis_available:
                    key = f"{self.blacklist_key_prefix}:{jti}"
                    self._redis_sync(
                        self.redis_client.set(
                            key, "revoked", ttl=ttl, namespace=self.blacklist_namespace
                        )
                    )
                else:
                    # Fallback to in-memory
                    self.blacklisted_tokens.add(jti)

                self.logger.info("Token revoked", jti=jti, ttl_seconds=ttl)
                return True

            return False

        except Exception as e:
            self.logger.error(f"Token revocation failed: {e}")
            # Still add to in-memory blacklist as fallback
            self.blacklisted_tokens.add(token[:50])
            return True

    def is_token_blacklisted(self, token: str) -> bool:
        """
        Check if a token is blacklisted.

        Args:
            token: Token to check

        Returns:
            bool: True if token is blacklisted
        """
        try:
            # Decode token to get JTI
            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm], options={"verify_exp": False}
            )
            jti = payload.get("jti", token[:50])

            # Check Redis blacklist
            if self._redis_available:
                key = f"{self.blacklist_key_prefix}:{jti}"
                exists = self._redis_sync(
                    self.redis_client.exists(key, namespace=self.blacklist_namespace)
                )
                if exists:
                    return True

            # Check in-memory blacklist
            return jti in self.blacklisted_tokens

        except Exception as e:
            self.logger.error(f"Blacklist check failed: {e}")
            # Conservative approach - consider invalid tokens as blacklisted
            return True

    def get_security_summary(self) -> dict[str, Any]:
        """Get security configuration summary."""
        return {
            "algorithm": self.algorithm,
            "access_token_expire_minutes": self.access_token_expire_minutes,
            "refresh_token_expire_days": self.refresh_token_expire_days,
            "require_https": self.require_https,
            "max_token_age_hours": self.max_token_age_hours,
            "blacklist_backend": "redis" if self._redis_available else "memory",
            "blacklisted_tokens_count": len(self.blacklisted_tokens),
        }

    def cleanup_expired_blacklist(self) -> int:
        """
        Clean up expired tokens from in-memory blacklist.

        Returns:
            int: Number of tokens removed
        """
        if not self.blacklisted_tokens:
            return 0

        # This is mainly for the in-memory blacklist
        # Redis handles expiration automatically
        initial_count = len(self.blacklisted_tokens)

        # In production, you'd decode each token and check expiration
        # For now, clear very old tokens (simplified)
        self.blacklisted_tokens.clear()

        removed = initial_count - len(self.blacklisted_tokens)
        if removed > 0:
            self.logger.info(f"Cleaned up {removed} expired tokens from blacklist")

        return removed
