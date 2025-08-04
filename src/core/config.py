"""
Configuration management for the trading bot framework.

This module provides comprehensive Pydantic-based configuration system that will be
extended by ALL subsequent prompts. Use exact patterns from @COMMON_PATTERNS.md.

CRITICAL: This file will be extended by ALL subsequent prompts. Use exact patterns from @COMMON_PATTERNS.md.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import os


class BaseConfig(BaseSettings):
    """Base configuration class with common patterns."""
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "validate_assignment": True,
        "extra": "ignore"
    }


class DatabaseConfig(BaseConfig):
    """Database configuration for PostgreSQL, Redis, and InfluxDB."""
    
    # PostgreSQL
    postgresql_host: str = Field(default="localhost", description="PostgreSQL host")
    postgresql_port: int = Field(default=5432, description="PostgreSQL port")
    postgresql_database: str = Field(default="trading_bot", description="PostgreSQL database name")
    postgresql_username: str = Field(default="trading_bot", description="PostgreSQL username")
    postgresql_password: str = Field(default="trading_bot_password", description="PostgreSQL password")
    postgresql_pool_size: int = Field(default=10, description="PostgreSQL connection pool size")
    
    # Redis
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_db: int = Field(default=0, description="Redis database number")
    
    # InfluxDB
    influxdb_host: str = Field(default="localhost", description="InfluxDB host")
    influxdb_port: int = Field(default=8086, description="InfluxDB port")
    influxdb_token: str = Field(default="test-token", description="InfluxDB token")
    influxdb_org: str = Field(default="test-org", description="InfluxDB organization")
    influxdb_bucket: str = Field(default="trading-data", description="InfluxDB bucket")

    @field_validator('postgresql_port', 'redis_port', 'influxdb_port')
    @classmethod
    def validate_ports(cls, v):
        """Validate port numbers are within valid range."""
        if not 1 <= v <= 65535:
            raise ValueError(f'Port must be between 1 and 65535, got {v}')
        return v

    @field_validator('postgresql_pool_size')
    @classmethod
    def validate_pool_size(cls, v):
        """Validate database pool size."""
        if not 1 <= v <= 100:
            raise ValueError(f'Pool size must be between 1 and 100, got {v}')
        return v


class SecurityConfig(BaseConfig):
    """Security configuration for authentication and encryption."""
    
    secret_key: str = Field(default="test-secret-key-32-chars-long-for-testing", description="Secret key for JWT")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(default=30, description="JWT expiration time in minutes")
    encryption_key: str = Field(default="test-encryption-key-32-chars-long-for-testing", description="Encryption key")

    @field_validator('jwt_expire_minutes')
    @classmethod
    def validate_jwt_expire(cls, v):
        """Validate JWT expiration time."""
        if not 1 <= v <= 1440:  # 1 minute to 24 hours
            raise ValueError(f'JWT expire minutes must be between 1 and 1440, got {v}')
        return v

    @field_validator('secret_key', 'encryption_key')
    @classmethod
    def validate_key_length(cls, v):
        """Validate key lengths for security."""
        if len(v) < 32:
            raise ValueError(f'Key must be at least 32 characters long, got {len(v)}')
        return v


class ErrorHandlingConfig(BaseConfig):
    """Error handling configuration for P-002A framework."""
    
    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = Field(default=5, description="Circuit breaker failure threshold")
    circuit_breaker_recovery_timeout: int = Field(default=30, description="Circuit breaker recovery timeout in seconds")
    
    # Retry policies
    max_retry_attempts: int = Field(default=3, description="Maximum retry attempts")
    retry_backoff_factor: float = Field(default=2.0, description="Retry backoff factor")
    retry_max_delay: int = Field(default=60, description="Maximum retry delay in seconds")
    
    # Error pattern analytics
    pattern_detection_enabled: bool = Field(default=True, description="Enable error pattern detection")
    correlation_analysis_enabled: bool = Field(default=True, description="Enable correlation analysis")
    predictive_alerts_enabled: bool = Field(default=True, description="Enable predictive alerts")
    
    # State monitoring
    state_validation_frequency: int = Field(default=60, description="State validation frequency in seconds")
    auto_reconciliation_enabled: bool = Field(default=True, description="Enable automatic reconciliation")
    max_discrepancy_threshold: float = Field(default=0.01, description="Maximum discrepancy threshold")
    
    # Recovery scenarios
    partial_fill_min_percentage: float = Field(default=0.5, description="Minimum fill percentage for partial orders")
    network_max_offline_duration: int = Field(default=300, description="Maximum offline duration in seconds")
    data_feed_max_staleness: int = Field(default=30, description="Maximum data staleness in seconds")
    order_rejection_max_retries: int = Field(default=2, description="Maximum retry attempts for rejected orders")
    
    @field_validator('circuit_breaker_failure_threshold', 'max_retry_attempts', 'order_rejection_max_retries')
    @classmethod
    def validate_positive_integers(cls, v):
        """Validate positive integer values."""
        if v <= 0:
            raise ValueError(f'Value must be positive, got {v}')
        return v
    
    @field_validator('retry_backoff_factor', 'partial_fill_min_percentage', 'max_discrepancy_threshold')
    @classmethod
    def validate_positive_floats(cls, v):
        """Validate positive float values."""
        if v <= 0:
            raise ValueError(f'Value must be positive, got {v}')
        return v


class Config(BaseConfig):
    """Master configuration class for the entire application."""
    
    # Environment
    environment: str = Field(default="development", description="Application environment")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Application
    app_name: str = Field(default="trading-bot-suite", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    
    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    security: SecurityConfig = SecurityConfig()
    error_handling: ErrorHandlingConfig = ErrorHandlingConfig()
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        """Validate environment setting."""
        allowed = ['development', 'staging', 'production']
        if v not in allowed:
            raise ValueError(f'Environment must be one of: {allowed}')
        return v
    
    def generate_schema(self) -> None:
        """Generate JSON schema for configuration validation."""
        schema = self.model_json_schema()
        schema_path = Path("config/config.schema.json")
        schema_path.parent.mkdir(exist_ok=True)
        with open(schema_path, 'w') as f:
            json.dump(schema, f, indent=2)

    def get_database_url(self) -> str:
        """Generate PostgreSQL database URL."""
        return (
            f"postgresql://{self.database.postgresql_username}:"
            f"{self.database.postgresql_password}@"
            f"{self.database.postgresql_host}:"
            f"{self.database.postgresql_port}/"
            f"{self.database.postgresql_database}"
        )
    
    def get_async_database_url(self) -> str:
        """Generate async PostgreSQL database URL."""
        return (
            f"postgresql+asyncpg://{self.database.postgresql_username}:"
            f"{self.database.postgresql_password}@"
            f"{self.database.postgresql_host}:"
            f"{self.database.postgresql_port}/"
            f"{self.database.postgresql_database}"
        )

    def get_redis_url(self) -> str:
        """Generate Redis URL."""
        if self.database.redis_password:
            return (
                f"redis://:{self.database.redis_password}@"
                f"{self.database.redis_host}:"
                f"{self.database.redis_port}/{self.database.redis_db}"
            )
        return (
            f"redis://{self.database.redis_host}:"
            f"{self.database.redis_port}/{self.database.redis_db}"
        )

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


# REVERSE INTEGRATION POINTS: Future prompts will add these config classes:
# - P-002: Extended DatabaseConfig with connection pooling settings
# - P-003: ExchangeConfig for API credentials and rate limits
# - P-008: RiskConfig for risk management parameters
# - P-010A: CapitalConfig for fund management settings
# - P-011: StrategyConfig for strategy parameters
# - P-017: MLConfig for model registry and training
# - P-026: WebConfig for FastAPI and security settings 