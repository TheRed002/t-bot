"""Security configuration for the T-Bot trading system."""

import os

from pydantic import Field

from .base import BaseConfig


class SecurityConfig(BaseConfig):
    """Security configuration for JWT, authentication, and other security settings."""

    # JWT Configuration
    secret_key: str = Field(
        default_factory=lambda: os.getenv(
            "JWT_SECRET_KEY", "your-super-secure-secret-key-change-this-in-production"
        ),
        description="JWT secret key for token signing",
        min_length=32,
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm",
    )
    jwt_expire_minutes: int = Field(
        default=30,
        description="JWT access token expiration time in minutes",
        ge=1,
        le=60 * 24,  # Max 24 hours
    )
    jwt_refresh_expire_days: int = Field(
        default=7,
        description="JWT refresh token expiration time in days",
        ge=1,
        le=30,  # Max 30 days
    )

    # Session Configuration
    session_timeout_minutes: int = Field(
        default=60,
        description="Session timeout in minutes",
        ge=1,
        le=60 * 24,  # Max 24 hours
    )

    # Security Headers
    enable_hsts: bool = Field(
        default=True,
        description="Enable HTTP Strict Transport Security",
    )
    enable_csp: bool = Field(
        default=True,
        description="Enable Content Security Policy",
    )
    enable_frame_protection: bool = Field(
        default=True,
        description="Enable X-Frame-Options protection",
    )

    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(
        default=60,
        description="Rate limit requests per minute per IP",
        ge=1,
    )
    rate_limit_burst: int = Field(
        default=10,
        description="Rate limit burst allowance",
        ge=1,
    )

    # Password Security
    password_min_length: int = Field(
        default=8,
        description="Minimum password length",
        ge=6,
    )
    password_require_uppercase: bool = Field(
        default=True,
        description="Require uppercase letters in passwords",
    )
    password_require_lowercase: bool = Field(
        default=True,
        description="Require lowercase letters in passwords",
    )
    password_require_numbers: bool = Field(
        default=True,
        description="Require numbers in passwords",
    )
    password_require_special: bool = Field(
        default=True,
        description="Require special characters in passwords",
    )

    # Account Security
    max_login_attempts: int = Field(
        default=5,
        description="Maximum login attempts before account lockout",
        ge=1,
    )
    account_lockout_duration_minutes: int = Field(
        default=15,
        description="Account lockout duration in minutes",
        ge=1,
    )

    # API Security
    require_https: bool = Field(
        default=True,
        description="Require HTTPS for API endpoints",
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests",
    )
    cors_allowed_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins",
    )

    def get_jwt_config(self) -> dict:
        """Get JWT configuration as dictionary."""
        return {
            "secret_key": self.secret_key,
            "algorithm": self.jwt_algorithm,
            "access_token_expire_minutes": self.jwt_expire_minutes,
            "refresh_token_expire_days": self.jwt_refresh_expire_days,
        }

    def get_session_config(self) -> dict:
        """Get session configuration as dictionary."""
        return {
            "timeout_minutes": self.session_timeout_minutes,
        }

    def get_cors_config(self) -> dict:
        """Get CORS configuration as dictionary."""
        return {
            "allow_origins": self.cors_allowed_origins,
            "allow_credentials": self.cors_allow_credentials,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }

    def get_rate_limit_config(self) -> dict:
        """Get rate limiting configuration as dictionary."""
        return {
            "requests_per_minute": self.rate_limit_requests_per_minute,
            "burst": self.rate_limit_burst,
        }
