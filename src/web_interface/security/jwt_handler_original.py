"""
JWT token handling for T-Bot authentication.

This module provides comprehensive JWT token management including creation,
validation, refresh, and revocation capabilities with security best practices.
"""

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import redis
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from src.base import BaseComponent
from src.core.config import Config
from src.core.exceptions import AuthenticationError, ValidationError


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
        self.initialize()  # Mark as initialized
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
        self.redis_client = self._setup_redis_client(config)
        self.blacklist_key_prefix = "jwt_blacklist:"

        # Fallback in-memory blacklist for development
        self.blacklisted_tokens: set[str] = set()

        # Security settings
        self.require_https = jwt_config.get("require_https", True)
        self.max_token_age_hours = jwt_config.get("max_token_age_hours", 24)

        self.logger.info("JWT handler initialized with Redis blacklist support")

    def _setup_redis_client(self, config: Config):
        """Setup Redis client for token blacklist."""
        try:
            # Try to get Redis configuration
            redis_config = getattr(config, "redis", None) or {}
            redis_url = os.getenv("REDIS_URL")

            if redis_url:
                client = redis.from_url(redis_url, decode_responses=True)
            else:
                # Use individual config parameters
                host = redis_config.get("host", "localhost")
                port = redis_config.get("port", 6379)
                password = redis_config.get("password")
                db = redis_config.get("database", 0)

                client = redis.Redis(
                    host=host,
                    port=port,
                    password=password,
                    db=db,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                )

            # Test connection
            client.ping()
            self.logger.info("Redis connection established for token blacklist")
            return client

        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}. Using in-memory blacklist.")
            return None

    def _get_secret_key(self, config: Config) -> str:
        """
        Get JWT secret key from database, environment variable, or config.

        Priority order:
        1. Database system_config table
        2. JWT_SECRET_KEY environment variable
        3. JWT_SECRET environment variable
        4. Raise error (no fallbacks for security)
        """
        # First try to get from database
        try:
            if hasattr(self, "redis_client") and self.redis_client:
                db_secret = self.redis_client.get("system:jwt_secret")
                if db_secret:
                    self.logger.info("Using JWT secret key from database")
                    return db_secret
        except Exception as e:
            self.logger.debug(f"Could not retrieve JWT secret from database: {e}")

        # Check environment variables
        secret_key = os.getenv("JWT_SECRET_KEY") or os.getenv("JWT_SECRET")

        if secret_key and len(secret_key) >= 32:
            self.logger.info("Using JWT secret key from environment variable")
            return secret_key

        # Check if we're in development and generate/store a key
        if getattr(config, "environment", "production") == "development":
            secret_key = secrets.token_urlsafe(64)  # 64 bytes = 512 bits

            # Try to store in Redis for consistency
            try:
                if hasattr(self, "redis_client") and self.redis_client:
                    self.redis_client.setex("system:jwt_secret", 86400 * 30, secret_key)  # 30 days
                    self.logger.info("Generated and stored JWT secret in Redis for development")
            except Exception as e:
                self.logger.debug(f"Could not store JWT secret in Redis: {e}")

            self.logger.warning(
                "Generated temporary JWT secret key for development. "
                "Set JWT_SECRET_KEY environment variable for production!"
            )
            return secret_key
        else:
            # Production environment must have a secret key
            raise ValueError(
                "JWT secret key is required in production. Set JWT_SECRET_KEY environment variable "
                "with at least 32 characters or store in system_config database table."
            )

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

    def _is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted using Redis or fallback storage."""
        try:
            # First check Redis if available
            if self.redis_client:
                blacklist_key = f"{self.blacklist_key_prefix}{token}"
                is_blacklisted = self.redis_client.exists(blacklist_key)
                if is_blacklisted:
                    return True

            # Fallback to in-memory blacklist
            return token in self.blacklisted_tokens

        except Exception as e:
            self.logger.error(f"Error checking token blacklist: {e}")
            # Fallback to in-memory check
            return token in self.blacklisted_tokens

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
            if self._is_token_blacklisted(token):
                self.logger.warning("Attempted use of blacklisted token")
                raise AuthenticationError("Token has been revoked")

            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Extract token data
            username = payload.get("sub")
            user_id = payload.get("user_id")
            scopes = payload.get("scopes", [])
            issued_at = datetime.fromtimestamp(payload.get("iat", 0), tz=timezone.utc)
            expires_at = datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc)
            jti = payload.get("jti")  # JWT ID for unique identification

            if username is None or user_id is None or jti is None:
                raise AuthenticationError("Invalid token payload")

            # Check token age (prevent very old tokens even if not expired)
            token_age = datetime.now(timezone.utc) - issued_at
            if token_age > timedelta(hours=self.max_token_age_hours):
                raise AuthenticationError("Token is too old")

            # Check if token is expired
            if datetime.now(timezone.utc) > expires_at:
                raise AuthenticationError("Token has expired")

            # Additional security: verify token was issued by this system
            if not self._verify_token_signature(token):
                raise AuthenticationError("Token signature verification failed")

            return TokenData(
                username=username,
                user_id=user_id,
                scopes=scopes,
                issued_at=issued_at,
                expires_at=expires_at,
            )

        except JWTError as e:
            self.logger.warning(f"JWT validation failed: {e}")
            raise AuthenticationError("Invalid token")
        except AuthenticationError:
            # Re-raise auth errors as-is
            raise
        except Exception as e:
            self.logger.error(f"Token validation error: {e}")
            raise AuthenticationError("Token validation failed")

    def _verify_token_signature(self, token: str) -> bool:
        """Verify token signature without full decoding."""
        try:
            # This will raise an exception if signature is invalid
            jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm], options={"verify_exp": False}
            )
            return True
        except JWTError:
            return False

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
            # Get token info to determine expiration
            token_info = self.get_token_info(token)
            expires_at = token_info.get("expires_at")

            if expires_at and isinstance(expires_at, datetime):
                # Calculate TTL in seconds until token expires
                ttl_seconds = int((expires_at - datetime.now(timezone.utc)).total_seconds())

                if ttl_seconds > 0:
                    # Store in Redis with TTL if available
                    if self.redis_client:
                        try:
                            blacklist_key = f"{self.blacklist_key_prefix}{token}"
                            self.redis_client.setex(blacklist_key, ttl_seconds, "revoked")
                            self.logger.info(f"Token blacklisted in Redis with TTL {ttl_seconds}s")
                        except Exception as e:
                            self.logger.error(f"Failed to blacklist in Redis: {e}")
                            # Fallback to in-memory storage
                            self.blacklisted_tokens.add(token)
                    else:
                        # Use in-memory storage
                        self.blacklisted_tokens.add(token)
                        self.logger.info("Token blacklisted in memory")
                else:
                    self.logger.info("Token already expired, not adding to blacklist")
            else:
                # If we can't determine expiration, blacklist it anyway
                if self.redis_client:
                    try:
                        blacklist_key = f"{self.blacklist_key_prefix}{token}"
                        # Use default TTL of 24 hours for unknown tokens
                        self.redis_client.setex(blacklist_key, 86400, "revoked")
                    except Exception:
                        self.blacklisted_tokens.add(token)
                else:
                    self.blacklisted_tokens.add(token)

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
