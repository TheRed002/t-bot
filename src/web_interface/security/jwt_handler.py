"""
JWT token handling for T-Bot authentication.

This module provides comprehensive JWT token management including creation,
validation, refresh, and revocation capabilities with security best practices.
"""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from src.core.config import Config
from src.core.exceptions import AuthenticationError, ValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)


class TokenData(BaseModel):
    """Token data structure."""

    username: str | None = None
    user_id: str | None = None
    scopes: list[str] = []
    issued_at: datetime | None = None
    expires_at: datetime | None = None


class JWTHandler:
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
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # JWT Configuration - handle missing web_interface config
        if hasattr(config, "web_interface"):
            jwt_config = config.web_interface.get("jwt", {})
        else:
            # Fall back to security config
            jwt_config = {
                "secret_key": config.security.secret_key,
                "algorithm": config.security.jwt_algorithm,
                "access_token_expire_minutes": config.security.jwt_expire_minutes,
            }

        self.secret_key = jwt_config.get("secret_key", self._generate_secret_key())
        self.algorithm = jwt_config.get("algorithm", "HS256")
        self.access_token_expire_minutes = jwt_config.get("access_token_expire_minutes", 30)
        self.refresh_token_expire_days = jwt_config.get("refresh_token_expire_days", 7)

        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # Token blacklist (in production, use Redis or database)
        self.blacklisted_tokens: set[str] = set()

        # Security settings
        self.require_https = jwt_config.get("require_https", True)
        self.max_token_age_hours = jwt_config.get("max_token_age_hours", 24)

        self.logger.info("JWT handler initialized")

    def _generate_secret_key(self) -> str:
        """Generate a secure secret key if none provided."""
        secret_key = secrets.token_urlsafe(32)
        self.logger.warning(
            "Generated temporary JWT secret key - use environment variable in production"
        )
        return secret_key

    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            str: Hashed password
        """
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password from database

        Returns:
            bool: True if password matches
        """
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_access_token(
        self,
        user_id: str,
        username: str,
        scopes: list[str] | None = None,
        expires_delta: timedelta | None = None,
    ) -> str:
        """
        Create JWT access token.

        Args:
            user_id: User identifier
            username: Username
            scopes: Permission scopes
            expires_delta: Custom expiration time

        Returns:
            str: JWT token

        Raises:
            ValidationError: If token creation fails
        """
        try:
            if expires_delta:
                expire = datetime.now(timezone.utc) + expires_delta
            else:
                expire = datetime.now(timezone.utc) + timedelta(
                    minutes=self.access_token_expire_minutes
                )

            issued_at = datetime.now(timezone.utc)
            scopes = scopes or ["read"]

            # Create JWT payload
            to_encode = {
                "sub": username,  # Subject (username)
                "user_id": user_id,
                "scopes": scopes,
                "iat": int(issued_at.timestamp()),  # Issued at
                "exp": int(expire.timestamp()),  # Expiration
                "type": "access",
                "jti": secrets.token_urlsafe(16),  # JWT ID for revocation
            }

            # Encode token
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

            self.logger.info(
                "Access token created",
                username=username,
                user_id=user_id,
                scopes=scopes,
                expires_at=expire.isoformat(),
            )

            return encoded_jwt

        except Exception as e:
            self.logger.error(f"Token creation failed: {e}", username=username)
            raise ValidationError(f"Token creation failed: {e}")

    def create_refresh_token(self, user_id: str, username: str) -> str:
        """
        Create JWT refresh token.

        Args:
            user_id: User identifier
            username: Username

        Returns:
            str: JWT refresh token
        """
        try:
            expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
            issued_at = datetime.now(timezone.utc)

            to_encode = {
                "sub": username,
                "user_id": user_id,
                "iat": int(issued_at.timestamp()),
                "exp": int(expire.timestamp()),
                "type": "refresh",
                "jti": secrets.token_urlsafe(16),
            }

            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

            self.logger.info(
                "Refresh token created",
                username=username,
                user_id=user_id,
                expires_at=expire.isoformat(),
            )

            return encoded_jwt

        except Exception as e:
            self.logger.error(f"Refresh token creation failed: {e}", username=username)
            raise ValidationError(f"Refresh token creation failed: {e}")

    def validate_token(self, token: str) -> TokenData:
        """
        Validate and decode JWT token.

        Args:
            token: JWT token to validate

        Returns:
            TokenData: Decoded token data

        Raises:
            AuthenticationError: If token is invalid or expired
        """
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                raise AuthenticationError("Token has been revoked")

            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Extract token data
            username = payload.get("sub")
            user_id = payload.get("user_id")
            scopes = payload.get("scopes", [])
            issued_at = datetime.fromtimestamp(payload.get("iat", 0), tz=timezone.utc)
            expires_at = datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc)

            if username is None or user_id is None:
                raise AuthenticationError("Invalid token payload")

            # Check token age
            token_age = datetime.now(timezone.utc) - issued_at
            if token_age > timedelta(hours=self.max_token_age_hours):
                raise AuthenticationError("Token is too old")

            # Check if token is expired
            if datetime.now(timezone.utc) > expires_at:
                raise AuthenticationError("Token has expired")

            return TokenData(
                username=username,
                user_id=user_id,
                scopes=scopes,
                issued_at=issued_at,
                expires_at=expires_at,
            )

        except JWTError as e:
            self.logger.warning(f"Token validation failed: {e}")
            raise AuthenticationError("Invalid token")
        except Exception as e:
            self.logger.error(f"Token validation error: {e}")
            raise AuthenticationError("Token validation failed")

    def refresh_access_token(self, refresh_token: str) -> tuple[str, str]:
        """
        Create new access token using refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            tuple: (new_access_token, new_refresh_token)

        Raises:
            AuthenticationError: If refresh token is invalid
        """
        try:
            # Validate refresh token
            token_data = self.validate_token(refresh_token)

            # Verify it's a refresh token
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])
            if payload.get("type") != "refresh":
                raise AuthenticationError("Invalid token type for refresh")

            # Create new tokens
            new_access_token = self.create_access_token(
                user_id=token_data.user_id, username=token_data.username, scopes=token_data.scopes
            )

            new_refresh_token = self.create_refresh_token(
                user_id=token_data.user_id, username=token_data.username
            )

            # Blacklist old refresh token
            self.revoke_token(refresh_token)

            self.logger.info(
                "Tokens refreshed successfully",
                username=token_data.username,
                user_id=token_data.user_id,
            )

            return new_access_token, new_refresh_token

        except Exception as e:
            self.logger.error(f"Token refresh failed: {e}")
            raise AuthenticationError("Token refresh failed")

    def revoke_token(self, token: str) -> bool:
        """
        Revoke (blacklist) a token.

        Args:
            token: Token to revoke

        Returns:
            bool: True if successfully revoked
        """
        try:
            # Add to blacklist
            self.blacklisted_tokens.add(token)

            # In production, store in persistent storage (Redis/Database)
            # self.redis_client.sadd("blacklisted_tokens", token)

            self.logger.info("Token revoked successfully")
            return True

        except Exception as e:
            self.logger.error(f"Token revocation failed: {e}")
            return False

    def validate_scopes(self, token_data: TokenData, required_scopes: list[str]) -> bool:
        """
        Validate token has required scopes.

        Args:
            token_data: Decoded token data
            required_scopes: List of required scopes

        Returns:
            bool: True if token has all required scopes
        """
        if not required_scopes:
            return True

        # Check if user has admin scope (grants all permissions)
        if "admin" in token_data.scopes:
            return True

        # Check if user has all required scopes
        return all(scope in token_data.scopes for scope in required_scopes)

    def get_token_info(self, token: str) -> dict[str, Any]:
        """
        Get information about a token without validation.

        Args:
            token: JWT token

        Returns:
            dict: Token information
        """
        try:
            # Decode without verification to get payload
            payload = jwt.decode(token, options={"verify_signature": False})

            return {
                "username": payload.get("sub"),
                "user_id": payload.get("user_id"),
                "scopes": payload.get("scopes", []),
                "issued_at": datetime.fromtimestamp(payload.get("iat", 0), tz=timezone.utc),
                "expires_at": datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc),
                "token_type": payload.get("type", "unknown"),
                "jti": payload.get("jti"),
            }

        except Exception as e:
            self.logger.error(f"Token info extraction failed: {e}")
            return {"error": str(e)}

    def cleanup_expired_blacklist(self) -> int:
        """
        Clean up expired tokens from blacklist.

        Returns:
            int: Number of tokens removed
        """
        try:
            removed_count = 0
            current_time = datetime.now(timezone.utc)

            # Check each blacklisted token
            tokens_to_remove = []
            for token in self.blacklisted_tokens:
                try:
                    token_info = self.get_token_info(token)
                    expires_at = token_info.get("expires_at")

                    if expires_at and current_time > expires_at:
                        tokens_to_remove.append(token)

                except Exception:
                    # If we can't parse the token, remove it
                    tokens_to_remove.append(token)

            # Remove expired tokens
            for token in tokens_to_remove:
                self.blacklisted_tokens.discard(token)
                removed_count += 1

            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} expired tokens from blacklist")

            return removed_count

        except Exception as e:
            self.logger.error(f"Blacklist cleanup failed: {e}")
            return 0

    def get_security_summary(self) -> dict[str, Any]:
        """
        Get security handler summary.

        Returns:
            dict: Security status and statistics
        """
        return {
            "jwt_algorithm": self.algorithm,
            "access_token_expire_minutes": self.access_token_expire_minutes,
            "refresh_token_expire_days": self.refresh_token_expire_days,
            "blacklisted_tokens_count": len(self.blacklisted_tokens),
            "require_https": self.require_https,
            "max_token_age_hours": self.max_token_age_hours,
            "security_features": [
                "password_hashing",
                "token_blacklisting",
                "scope_validation",
                "token_refresh",
                "security_logging",
            ],
        }
