"""Database configuration for the T-Bot trading system."""

from pydantic import AliasChoices, Field, field_validator

from .base import BaseConfig


class DatabaseConfig(BaseConfig):
    """Database configuration for PostgreSQL, Redis, and InfluxDB."""

    # PostgreSQL - Using environment variable names from .env
    postgresql_host: str = Field(
        default="localhost",
        description="PostgreSQL host",
        validation_alias=AliasChoices("DB_HOST", "POSTGRESQL_HOST"),
    )
    postgresql_port: int = Field(default=5432, description="PostgreSQL port", alias="DB_PORT")
    postgresql_database: str = Field(
        default="tbot_dev",
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="PostgreSQL database name",
        alias="DB_NAME",
    )
    postgresql_username: str = Field(
        default="tbot",
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="PostgreSQL username",
        alias="DB_USER",
    )
    postgresql_password: str = Field(
        default="tbot_password",
        min_length=8,
        description="PostgreSQL password",
        alias="DB_PASSWORD",
    )
    postgresql_pool_size: int = Field(default=20, description="PostgreSQL connection pool size")
    postgresql_max_overflow: int = Field(
        default=40, description="PostgreSQL max overflow connections"
    )
    postgresql_pool_timeout: int = Field(
        default=30, description="PostgreSQL pool timeout in seconds"
    )

    # Redis - Using environment variable names from .env
    redis_host: str = Field(default="localhost", description="Redis host", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, description="Redis port", alias="REDIS_PORT")
    redis_password: str | None = Field(
        default=None, description="Redis password", alias="REDIS_PASSWORD"
    )
    redis_db: int = Field(default=0, description="Redis database number", alias="REDIS_DB")
    redis_max_connections: int = Field(default=50, description="Redis max connections in pool")
    redis_socket_timeout: int = Field(default=5, description="Redis socket timeout in seconds")

    # InfluxDB - Using environment variable names from .env
    influxdb_host: str = Field(
        default="localhost", description="InfluxDB host", alias="INFLUXDB_HOST"
    )
    influxdb_port: int = Field(default=8086, description="InfluxDB port", alias="INFLUXDB_PORT")
    influxdb_token: str = Field(default="", description="InfluxDB token", alias="INFLUXDB_TOKEN")
    influxdb_org: str = Field(
        default="tbot", description="InfluxDB organization", alias="INFLUXDB_ORG"
    )
    influxdb_bucket: str = Field(
        default="trading_data", description="InfluxDB bucket", alias="INFLUXDB_BUCKET"
    )
    influxdb_timeout: int = Field(default=10000, description="InfluxDB timeout in milliseconds")

    @field_validator("postgresql_port", "redis_port", "influxdb_port")
    @classmethod
    def validate_ports(cls, v: int) -> int:
        """Validate port numbers are within valid range."""
        if not 1 <= v <= 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {v}")
        return v

    @field_validator("postgresql_pool_size", "postgresql_max_overflow")
    @classmethod
    def validate_pool_size(cls, v: int) -> int:
        """Validate database pool size."""
        if not 1 <= v <= 100:
            raise ValueError(f"Pool size must be between 1 and 100, got {v}")
        return v

    @field_validator("redis_db")
    @classmethod
    def validate_redis_db(cls, v: int) -> int:
        """Validate Redis database number."""
        if not 0 <= v <= 15:
            raise ValueError(f"Redis DB must be between 0 and 15, got {v}")
        return v

    @property
    def postgresql_url(self) -> str:
        """Build PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgresql_username}:{self.postgresql_password}"
            f"@{self.postgresql_host}:{self.postgresql_port}/{self.postgresql_database}"
        )

    @property
    def redis_url(self) -> str:
        """Build Redis connection URL."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
