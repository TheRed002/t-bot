"""
Configuration management for the trading bot framework.

This module provides comprehensive Pydantic-based configuration system with
environment variable support, validation, and YAML file loading capabilities.
"""

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


class BaseConfig(BaseSettings):
    """Base configuration class with common patterns.

    Provides common Pydantic settings configuration with environment
    variable support, case insensitive matching, and validation.
    """

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        validate_assignment=True,
        extra="ignore",
        populate_by_name=True,  # Allow both field names and aliases
    )


class DatabaseConfig(BaseConfig):
    """Database configuration for PostgreSQL, Redis, and InfluxDB."""

    # PostgreSQL - Using environment variable names from .env
    postgresql_host: str = Field(
        default="localhost", description="PostgreSQL host", alias="DB_HOST"
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

    # Redis - Using environment variable names from .env
    redis_host: str = Field(default="localhost", description="Redis host", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, description="Redis port", alias="REDIS_PORT")
    redis_password: str | None = Field(
        default=None, description="Redis password", alias="REDIS_PASSWORD"
    )
    redis_db: int = Field(default=0, description="Redis database number", alias="REDIS_DB")

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

    @field_validator("postgresql_port", "redis_port", "influxdb_port")
    @classmethod
    def validate_ports(cls, v: int) -> int:
        """Validate port numbers are within valid range."""
        if not 1 <= v <= 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {v}")
        return v

    @field_validator("postgresql_pool_size")
    @classmethod
    def validate_pool_size(cls, v):
        """Validate database pool size."""
        if not 1 <= v <= 100:
            raise ValueError(f"Pool size must be between 1 and 100, got {v}")
        return v

    @property
    def postgresql_url(self) -> str:
        """Generate PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgresql_username}:{self.postgresql_password}"
            f"@{self.postgresql_host}:{self.postgresql_port}/{self.postgresql_database}"
        )

    @property
    def redis_url(self) -> str:
        """Generate Redis connection URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


class SecurityConfig(BaseConfig):
    """Security configuration for authentication and encryption."""

    secret_key: str = Field(
        default="",
        description="Secret key for JWT (set via SECRET_KEY env var)",
        alias="SECRET_KEY",
    )
    jwt_algorithm: str = Field(
        default="HS256",
        pattern=r"^(HS256|HS384|HS512|RS256|RS384|RS512)$",
        description="JWT algorithm",
    )
    jwt_expire_minutes: int = Field(default=30, description="JWT expiration time in minutes")
    encryption_key: str = Field(
        default="",
        description="Encryption key (set via ENCRYPTION_KEY env var)",
        alias="ENCRYPTION_KEY",
    )

    @field_validator("jwt_expire_minutes")
    @classmethod
    def validate_jwt_expire(cls, v):
        """Validate JWT expiration time."""
        if not 1 <= v <= 1440:  # 1 minute to 24 hours
            raise ValueError(f"JWT expire minutes must be between 1 and 1440, got {v}")
        return v

    @classmethod
    def _validate_key_basic(cls, v: str) -> None:
        """Validate basic key requirements."""
        if not v:
            raise ValueError("Key must be set via environment variable")
        if len(v) < 32:
            raise ValueError(f"Key must be at least 32 characters long, got {len(v)}")

    @classmethod
    def _validate_key_character_types(cls, v: str) -> None:
        """Validate key contains sufficient character type diversity."""
        has_lower = any(c.islower() for c in v)
        has_upper = any(c.isupper() for c in v)
        has_digit = any(c.isdigit() for c in v)
        has_special = any(not c.isalnum() for c in v)

        char_types = sum([has_lower, has_upper, has_digit, has_special])
        if char_types < 3:
            raise ValueError(
                "Key must contain at least 3 different character types "
                "(lowercase, uppercase, digits, special characters)"
            )

    @classmethod
    def _validate_key_patterns(cls, v: str) -> None:
        """Validate key doesn't contain common weak patterns."""
        lower_v = v.lower()
        weak_patterns = ["password", "secret", "admin", "12345", "qwerty"]
        if any(pattern in lower_v for pattern in weak_patterns):
            raise ValueError("Key contains common weak patterns")

        if len(set(v)) < 10:
            raise ValueError("Key has insufficient entropy (too few unique characters)")

    @classmethod
    def _calculate_shannon_entropy(cls, s: str) -> float:
        """Calculate Shannon entropy of a string."""
        import math

        if not s:
            return 0.0

        # Count frequency of each character
        char_counts = {}
        for c in s:
            char_counts[c] = char_counts.get(c, 0) + 1

        # Calculate entropy
        entropy = 0.0
        length = len(s)
        for count in char_counts.values():
            if count > 0:
                p_c = count / length
                entropy -= p_c * math.log2(p_c)

        return entropy

    @classmethod
    def _validate_key_entropy(cls, v: str) -> None:
        """Validate key cryptographic properties."""
        import hashlib

        entropy = cls._calculate_shannon_entropy(v)
        min_entropy = 3.0  # Approximately 8 bits of entropy
        if entropy < min_entropy:
            raise ValueError(f"Key entropy too low: {entropy:.2f} < {min_entropy}")

        # Use cryptographically secure hash for pattern detection
        key_hash = hashlib.sha256(v.encode("utf-8")).hexdigest()
        if len(set(key_hash[:16])) < 8:  # Check hash diversity
            raise ValueError("Key shows poor cryptographic properties")

    @field_validator("secret_key", "encryption_key")
    @classmethod
    def validate_key_length(cls, v):
        """Validate key lengths and entropy for security."""
        cls._validate_key_basic(v)
        cls._validate_key_character_types(v)
        cls._validate_key_patterns(v)
        cls._validate_key_entropy(v)
        return v


class ErrorHandlingConfig(BaseConfig):
    """Error handling configuration for P-002A framework."""

    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = Field(
        default=5, description="Circuit breaker failure threshold"
    )
    circuit_breaker_recovery_timeout: int = Field(
        default=30, description="Circuit breaker recovery timeout in seconds"
    )

    # Retry policies
    max_retry_attempts: int = Field(default=3, description="Maximum retry attempts")
    retry_backoff_factor: float = Field(default=2.0, description="Retry backoff factor")
    retry_max_delay: int = Field(default=60, description="Maximum retry delay in seconds")

    # Error pattern analytics
    pattern_detection_enabled: bool = Field(
        default=True, description="Enable error pattern detection"
    )
    correlation_analysis_enabled: bool = Field(
        default=True, description="Enable correlation analysis"
    )
    predictive_alerts_enabled: bool = Field(default=True, description="Enable predictive alerts")

    # State monitoring
    state_validation_frequency: int = Field(
        default=60, description="State validation frequency in seconds"
    )
    auto_reconciliation_enabled: bool = Field(
        default=True, description="Enable automatic reconciliation"
    )
    max_discrepancy_threshold: float = Field(
        default=0.01, description="Maximum discrepancy threshold"
    )

    # Recovery scenarios
    partial_fill_min_percentage: float = Field(
        default=0.5, description="Minimum fill percentage for partial orders"
    )
    network_max_offline_duration: int = Field(
        default=300, description="Maximum offline duration in seconds"
    )
    data_feed_max_staleness: int = Field(
        default=30, description="Maximum data staleness in seconds"
    )
    order_rejection_max_retries: int = Field(
        default=2, description="Maximum retry attempts for rejected orders"
    )

    @field_validator(
        "circuit_breaker_failure_threshold", "max_retry_attempts", "order_rejection_max_retries"
    )
    @classmethod
    def validate_positive_integers(cls, v):
        """Validate positive integer values."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    @field_validator(
        "retry_backoff_factor", "partial_fill_min_percentage", "max_discrepancy_threshold"
    )
    @classmethod
    def validate_positive_floats(cls, v):
        """Validate positive float values."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v


class ExchangeConfig(BaseConfig):
    """Exchange configuration for API credentials and rate limits."""

    # Default settings
    default_timeout: int = Field(default=30, description="Default exchange API timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts for exchange API calls")

    # Binance
    binance_api_key: str = Field(default="", min_length=0, description="Binance API key")
    binance_api_secret: str = Field(default="", min_length=0, description="Binance API secret")
    binance_testnet: bool = Field(default=True, description="Use Binance testnet")

    # OKX
    okx_api_key: str = Field(default="", min_length=0, description="OKX API key")
    okx_api_secret: str = Field(default="", min_length=0, description="OKX API secret")
    okx_passphrase: str = Field(default="", min_length=0, description="OKX API passphrase")
    okx_sandbox: bool = Field(default=True, description="Use OKX sandbox")

    # Coinbase
    coinbase_api_key: str = Field(default="", min_length=0, description="Coinbase API key")
    coinbase_api_secret: str = Field(default="", min_length=0, description="Coinbase API secret")
    coinbase_sandbox: bool = Field(default=True, description="Use Coinbase sandbox")

    # Rate limiting configuration
    rate_limits: dict[str, dict[str, int]] = Field(
        default={
            "binance": {
                "requests_per_minute": 1200,
                "orders_per_second": 10,
                "websocket_connections": 5,
            },
            "okx": {
                "requests_per_minute": 600,
                "orders_per_second": 20,
                "websocket_connections": 3,
            },
            "coinbase": {
                "requests_per_minute": 600,
                "orders_per_second": 15,
                "websocket_connections": 4,
            },
        },
        description="Exchange-specific rate limits",
    )

    # Supported exchanges
    supported_exchanges: list[str] = Field(
        default=["binance", "okx", "coinbase"], description="List of supported exchanges"
    )

    @field_validator(
        "binance_api_key",
        "binance_api_secret",
        "okx_api_key",
        "okx_api_secret",
        "okx_passphrase",
        "coinbase_api_key",
        "coinbase_api_secret",
    )
    @classmethod
    def validate_api_credentials(cls, v: str) -> str:
        """Validate API credentials.

        Allows any length in non-production environments to enable testing with
        dummy values (e.g., "test_api_key"). Enforces a minimum length in
        production-like environments.
        """
        # Strip whitespace first
        v = v.strip()

        # Empty strings are allowed (no API key configured) but not whitespace
        if not v:
            return ""

        env = os.getenv("ENVIRONMENT", "development").lower()

        # In production/staging, enforce minimum length
        if env in ("production", "staging"):
            if len(v) < 16:
                raise ValueError(
                    f"API credential must be at least 16 characters long in {env}, got {len(v)}"
                )
            # Basic format validation - should contain alphanumeric characters
            if not any(c.isalnum() for c in v):
                raise ValueError("API credential must contain alphanumeric characters")

        return v

    @property
    def default_exchange(self) -> str:
        """Get default exchange."""
        return self.supported_exchanges[0] if self.supported_exchanges else "binance"

    @property
    def testnet_mode(self) -> bool:
        """Get testnet mode for default exchange."""
        default = self.default_exchange
        if default == "binance":
            return self.binance_testnet
        elif default == "okx":
            return self.okx_sandbox
        elif default == "coinbase":
            return self.coinbase_sandbox
        return True  # Default to testnet for safety

    @property
    def rate_limit_per_second(self) -> int:
        """Get rate limit per second for default exchange."""
        default = self.default_exchange
        return self.rate_limits.get(default, {}).get("orders_per_second", 10)

    def get_exchange_credentials(self, exchange: str) -> dict[str, Any]:
        """Get credentials for specified exchange."""
        if exchange == "binance":
            return {
                "api_key": self.binance_api_key,
                "api_secret": self.binance_api_secret,
                "testnet": self.binance_testnet,
            }
        elif exchange == "okx":
            return {
                "api_key": self.okx_api_key,
                "api_secret": self.okx_api_secret,
                "passphrase": self.okx_passphrase,
                "sandbox": self.okx_sandbox,
            }
        elif exchange == "coinbase":
            return {
                "api_key": self.coinbase_api_key,
                "api_secret": self.coinbase_api_secret,
                "sandbox": self.coinbase_sandbox,
            }
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")

    def get_websocket_config(self, exchange: str) -> dict[str, Any]:
        """Get WebSocket configuration for specified exchange."""
        base_config = {
            "reconnect_attempts": 5,
            "ping_interval": 30,
            "timeout": 30,
        }

        if exchange == "binance":
            base_config["url"] = "wss://stream.binance.com:9443/ws/"
            if self.binance_testnet:
                base_config["url"] = "wss://testnet.binance.vision/ws/"
        elif exchange == "okx":
            base_config["url"] = "wss://ws.okx.com:8443/ws/v5/public"
            if self.okx_sandbox:
                base_config["url"] = "wss://wspap.okx.com:8443/ws/v5/public"
        elif exchange == "coinbase":
            base_config["url"] = "wss://ws-feed.pro.coinbase.com"
            if self.coinbase_sandbox:
                base_config["url"] = "wss://ws-feed-public.sandbox.pro.coinbase.com"

        return base_config


class RiskConfig(BaseConfig):
    """Risk management configuration for P-008 framework."""

    # Position sizing
    default_position_size_method: str = Field(
        default="fixed_percentage", description="Default position sizing method"
    )
    default_position_size_pct: Decimal = Field(
        default=Decimal("0.02"), description="Default position size percentage (2%)"
    )
    max_position_size_pct: Decimal = Field(
        default=Decimal("0.1"), description="Maximum position size percentage (10%)"
    )

    # Portfolio limits
    max_total_positions: int = Field(default=10, description="Maximum total positions")
    max_positions_per_symbol: int = Field(default=1, description="Maximum positions per symbol")
    max_portfolio_exposure: Decimal = Field(
        default=Decimal("0.95"), description="Maximum portfolio exposure (95%)"
    )
    max_sector_exposure: Decimal = Field(default=Decimal("0.25"), description="Maximum sector exposure (25%)")
    max_correlation_exposure: Decimal = Field(
        default=Decimal("0.5"), description="Maximum correlation exposure (50%)"
    )
    max_leverage: Decimal = Field(
        default=Decimal("1.0"), description="Maximum leverage (no leverage by default)"
    )

    # Risk thresholds
    max_daily_loss_pct: Decimal = Field(
        default=Decimal("0.05"), description="Maximum daily loss percentage (5%)"
    )
    max_drawdown_pct: Decimal = Field(default=Decimal("0.15"), description="Maximum drawdown percentage (15%)")
    var_confidence_level: Decimal = Field(
        default=Decimal("0.95"), description="VaR confidence level (95%)"
    )

    # Kelly Criterion settings
    kelly_lookback_days: int = Field(
        default=30, description="Kelly Criterion lookback period in days"
    )
    kelly_max_fraction: Decimal = Field(default=Decimal("0.25"), description="Maximum Kelly fraction (25%)")

    # Volatility adjustment
    volatility_window: int = Field(default=20, description="Volatility calculation window")
    volatility_target: Decimal = Field(default=Decimal("0.02"), description="Volatility target (2% daily)")

    # Risk calculation settings
    var_calculation_window: int = Field(
        default=252, description="VaR calculation window (trading days)"
    )
    drawdown_calculation_window: int = Field(
        default=252, description="Drawdown calculation window (trading days)"
    )
    correlation_calculation_window: int = Field(
        default=60, description="Correlation calculation window (days)"
    )

    # Emergency controls (P-009)
    emergency_close_positions: bool = Field(
        default=True, description="Close all positions during emergency stop"
    )
    emergency_recovery_timeout_hours: int = Field(
        default=1, description="Recovery timeout in hours after emergency stop"
    )
    emergency_manual_override_enabled: bool = Field(
        default=True, description="Enable manual override for emergency controls"
    )

    @field_validator(
        "default_position_size_pct",
        "max_position_size_pct",
        "max_portfolio_exposure",
        "max_sector_exposure",
        "max_correlation_exposure",
        "max_daily_loss_pct",
        "max_drawdown_pct",
        "var_confidence_level",
        "kelly_max_fraction",
        "volatility_target",
    )
    @classmethod
    def validate_percentage_fields(cls, v):
        """Validate percentage fields are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Percentage must be between 0 and 1, got {v}")
        return v

    @field_validator(
        "max_total_positions",
        "kelly_lookback_days",
        "volatility_window",
        "var_calculation_window",
        "drawdown_calculation_window",
        "correlation_calculation_window",
    )
    @classmethod
    def validate_positive_integers(cls, v):
        """Validate positive integer fields."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v


class CapitalManagementConfig(BaseConfig):
    """Capital management configuration for P-010A framework."""

    # Base settings
    base_currency: str = Field(
        default="USDT", pattern=r"^[A-Z]{3,10}$", description="Base currency for all calculations"
    )
    total_capital: Decimal = Field(
        default=Decimal("10000.0"), gt=0, description="Total available capital"
    )
    emergency_reserve_pct: Decimal = Field(
        default=Decimal("0.1"), description="Emergency reserve percentage (10%)"
    )

    # Allocation strategy
    allocation_strategy: str = Field(
        default="risk_parity", description="Capital allocation strategy"
    )
    rebalance_frequency_hours: int = Field(default=24, description="Rebalancing frequency in hours")
    min_allocation_pct: Decimal = Field(
        default=Decimal("0.01"), description="Minimum allocation percentage (1%)"
    )
    max_allocation_pct: Decimal = Field(
        default=Decimal("0.25"), description="Maximum allocation percentage (25%)"
    )

    # Exchange distribution
    max_exchange_allocation_pct: Decimal = Field(
        default=Decimal("0.6"), description="Maximum exchange allocation (60%)"
    )
    min_exchange_balance: Decimal = Field(
        default=Decimal("100.0"), description="Minimum exchange balance"
    )

    # Fund flow controls
    max_daily_reallocation_pct: Decimal = Field(
        default=Decimal("0.2"), description="Maximum daily reallocation (20%)"
    )
    fund_flow_cooldown_minutes: int = Field(default=60, description="Fund flow cooldown in minutes")
    min_deposit_amount: Decimal = Field(
        default=Decimal("1000.0"), description="Minimum deposit amount"
    )
    min_withdrawal_amount: Decimal = Field(
        default=Decimal("100.0"), description="Minimum withdrawal amount"
    )
    max_withdrawal_pct: Decimal = Field(
        default=Decimal("0.2"), description="Maximum withdrawal percentage (20%)"
    )

    # Currency management
    supported_currencies: list[str] = Field(
        default=["USDT", "BUSD", "USDC", "BTC", "ETH"],
        description="Supported currencies for trading",
    )
    hedging_enabled: bool = Field(default=True, description="Enable currency hedging")
    hedging_threshold: Decimal = Field(default=Decimal("0.1"), description="Hedging threshold (10% exposure)")
    hedge_ratio: Decimal = Field(default=Decimal("0.8"), description="Hedge coverage ratio (80%)")

    # Withdrawal rules
    withdrawal_rules: dict[str, dict[str, Any]] = Field(
        default={
            "profit_only": {"enabled": True, "description": "Only withdraw realized profits"},
            "maintain_minimum": {
                "enabled": True,
                "description": "Keep minimum capital for each strategy",
            },
            "performance_based": {
                "enabled": False,
                "threshold": 0.05,
                "description": "Allow withdrawals only if performance > threshold",
            },
        },
        description="Withdrawal rule configurations",
    )

    # Auto-compounding
    auto_compound_enabled: bool = Field(default=True, description="Enable auto-compounding")
    auto_compound_frequency: str = Field(default="weekly", description="Auto-compound frequency")
    profit_threshold: Decimal = Field(
        default=Decimal("100.0"), description="Minimum profit for compounding"
    )

    # Capital protection
    max_daily_loss_pct: Decimal = Field(default=Decimal("0.05"), description="Maximum daily loss (5%)")
    max_weekly_loss_pct: Decimal = Field(default=Decimal("0.10"), description="Maximum weekly loss (10%)")
    max_monthly_loss_pct: Decimal = Field(default=Decimal("0.15"), description="Maximum monthly loss (15%)")
    profit_lock_pct: Decimal = Field(default=Decimal("0.5"), description="Profit lock percentage (50%)")

    # Per-strategy minimum allocations
    per_strategy_minimum: dict[str, Decimal] = Field(
        default={
            "mean_reversion": Decimal("5000.0"),
            "trend_following": Decimal("10000.0"),
            "ml_strategy": Decimal("15000.0"),
            "arbitrage": Decimal("20000.0"),
            "market_making": Decimal("25000.0"),
        },
        description="Minimum capital requirements per strategy",
    )

    # Exchange allocation weights
    exchange_allocation_weights: dict[str, Decimal] = Field(
        default={"binance": Decimal("0.5"), "okx": Decimal("0.3"), "coinbase": Decimal("0.2")},
        description="Default exchange allocation weights",
    )

    @field_validator(
        "emergency_reserve_pct",
        "min_allocation_pct",
        "max_allocation_pct",
        "max_exchange_allocation_pct",
        "max_daily_reallocation_pct",
        "hedging_threshold",
        "hedge_ratio",
        "max_daily_loss_pct",
        "max_weekly_loss_pct",
        "max_monthly_loss_pct",
        "profit_lock_pct",
    )
    @classmethod
    def validate_percentage_fields(cls, v):
        """Validate percentage fields are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Percentage must be between 0 and 1, got {v}")
        return v

    @field_validator("rebalance_frequency_hours", "fund_flow_cooldown_minutes")
    @classmethod
    def validate_positive_integers(cls, v):
        """Validate positive integer fields."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    @field_validator(
        "total_capital",
        "min_exchange_balance",
        "min_deposit_amount",
        "min_withdrawal_amount",
        "profit_threshold",
    )
    @classmethod
    def validate_positive_decimals(cls, v: Decimal) -> Decimal:
        """Validate positive Decimal fields."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v


class StrategyManagementConfig(BaseConfig):
    """Strategy management configuration for P-011 framework."""

    # Global strategy settings
    max_concurrent_strategies: int = Field(default=10, description="Maximum concurrent strategies")
    strategy_restart_delay: int = Field(default=60, description="Strategy restart delay in seconds")
    performance_window_days: int = Field(
        default=30, description="Performance evaluation window in days"
    )

    # Default strategy parameters
    default_min_confidence: Decimal = Field(
        default=Decimal("0.6"), description="Default minimum signal confidence"
    )
    default_position_size: Decimal = Field(
        default=Decimal("0.02"), description="Default position size percentage"
    )
    default_stop_loss: Decimal = Field(default=Decimal("0.02"), description="Default stop loss percentage")
    default_take_profit: Decimal = Field(default=Decimal("0.04"), description="Default take profit percentage")

    # Hot reloading
    enable_hot_reload: bool = Field(default=True, description="Enable hot reloading of strategies")
    config_check_interval: int = Field(
        default=30, description="Configuration check interval in seconds"
    )

    # Strategy performance thresholds
    min_win_rate: Decimal = Field(default=Decimal("0.4"), description="Minimum acceptable win rate")
    min_sharpe_ratio: Decimal = Field(default=Decimal("0.5"), description="Minimum acceptable Sharpe ratio")
    max_drawdown_threshold: Decimal = Field(default=Decimal("0.15"), description="Maximum acceptable drawdown")

    # Strategy monitoring
    performance_evaluation_frequency: int = Field(
        default=24, description="Performance evaluation frequency in hours"
    )
    auto_disable_poor_performers: bool = Field(
        default=True, description="Auto-disable poorly performing strategies"
    )
    performance_alert_threshold: Decimal = Field(
        default=Decimal("0.3"), description="Performance alert threshold"
    )

    @field_validator(
        "max_concurrent_strategies",
        "strategy_restart_delay",
        "performance_window_days",
        "config_check_interval",
        "performance_evaluation_frequency",
    )
    @classmethod
    def validate_positive_integers(cls, v):
        """Validate positive integer fields."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    @field_validator(
        "default_min_confidence",
        "default_position_size",
        "default_stop_loss",
        "default_take_profit",
        "min_win_rate",
        "min_sharpe_ratio",
        "max_drawdown_threshold",
        "performance_alert_threshold",
    )
    @classmethod
    def validate_percentage_fields(cls, v):
        """Validate percentage fields are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Percentage must be between 0 and 1, got {v}")
        return v


class MLConfig(BaseConfig):
    """Machine Learning configuration for P-014 framework."""

    # Model registry settings
    model_registry_path: str = Field(
        default="models/registry", description="Path to model registry"
    )
    artifact_store_path: str = Field(
        default="models/artifacts", description="Path to model artifacts"
    )
    model_cache_size: int = Field(default=10, description="Number of models to cache in memory")

    # Training settings
    default_train_test_split: Decimal = Field(
        default=Decimal("0.8"), description="Default train/test split ratio"
    )
    default_validation_split: Decimal = Field(
        default=Decimal("0.2"), description="Default validation split ratio"
    )
    max_training_time_hours: int = Field(default=24, description="Maximum training time in hours")

    # Feature engineering
    feature_selection_threshold: Decimal = Field(
        default=Decimal("0.01"), description="Feature importance threshold for selection"
    )
    max_features: int = Field(default=1000, description="Maximum number of features")
    feature_cache_ttl_hours: int = Field(default=24, description="Feature cache TTL in hours")

    # Hyperparameter optimization
    optuna_n_trials: int = Field(default=100, description="Number of Optuna trials")
    optuna_timeout_hours: int = Field(default=12, description="Optuna timeout in hours")
    optuna_pruning_enabled: bool = Field(default=True, description="Enable Optuna pruning")

    # Model validation
    cross_validation_folds: int = Field(default=5, description="Number of CV folds")
    validation_frequency_days: int = Field(default=7, description="Validation frequency in days")
    performance_degradation_threshold: Decimal = Field(
        default=Decimal("0.1"), description="Performance degradation threshold"
    )

    # Drift detection
    drift_detection_enabled: bool = Field(default=True, description="Enable drift detection")
    drift_check_frequency_hours: int = Field(
        default=6, description="Drift check frequency in hours"
    )
    drift_threshold: Decimal = Field(default=Decimal("0.05"), description="Drift detection threshold")

    # Model serving
    inference_batch_size: int = Field(default=1000, description="Inference batch size")
    prediction_cache_ttl_minutes: int = Field(
        default=5, description="Prediction cache TTL in minutes"
    )
    model_warmup_enabled: bool = Field(default=True, description="Enable model warmup on startup")

    # Supported model types
    supported_model_types: list[str] = Field(
        default=[
            "price_predictor",
            "direction_classifier",
            "volatility_forecaster",
            "regime_detector",
        ],
        description="Supported ML model types",
    )

    # Model performance thresholds
    min_accuracy_threshold: Decimal = Field(
        default=Decimal("0.55"), description="Minimum model accuracy threshold"
    )
    min_precision_threshold: Decimal = Field(
        default=Decimal("0.50"), description="Minimum model precision threshold"
    )
    min_recall_threshold: Decimal = Field(default=Decimal("0.50"), description="Minimum model recall threshold")
    min_f1_threshold: Decimal = Field(default=Decimal("0.50"), description="Minimum model F1 score threshold")

    # Resource limits
    max_memory_gb: float = Field(default=8.0, description="Maximum memory usage in GB")
    max_cpu_cores: int = Field(default=4, description="Maximum CPU cores for training")
    gpu_enabled: bool = Field(default=False, description="Enable GPU acceleration")

    # Additional validation parameters
    validation_threshold: Decimal = Field(
        default=Decimal("0.6"), description="Overall model validation threshold"
    )
    significance_level: Decimal = Field(default=Decimal("0.05"), description="Statistical significance level")
    stability_window: int = Field(
        default=10, description="Number of periods for stability analysis"
    )
    min_validation_samples: int = Field(default=100, description="Minimum samples for validation")

    # Additional drift detection parameters
    min_drift_samples: int = Field(default=50, description="Minimum samples for drift detection")
    drift_reference_window: int = Field(
        default=1000, description="Reference window size for drift detection"
    )
    drift_detection_window: int = Field(
        default=500, description="Detection window size for drift detection"
    )

    # Additional model parameters
    batch_size: int = Field(default=1000, description="Batch size for processing")
    max_workers: int = Field(default=4, description="Maximum worker threads")
    use_multiprocessing: bool = Field(default=False, description="Enable multiprocessing")
    chunk_size: int = Field(default=100, description="Chunk size for batch processing")
    max_memory_mb: int = Field(default=2048, description="Maximum memory in MB")

    # Model-specific parameters
    use_log_returns: bool = Field(default=True, description="Use log returns for calculations")
    scaling_method: str = Field(default="standard", description="Feature scaling method")
    class_weights: str | None = Field(
        default="balanced", description="Class weights for imbalanced data"
    )
    random_state: int = Field(default=42, description="Random seed for reproducibility")

    @field_validator(
        "default_train_test_split",
        "default_validation_split",
        "feature_selection_threshold",
        "performance_degradation_threshold",
        "drift_threshold",
        "min_accuracy_threshold",
        "min_precision_threshold",
        "min_recall_threshold",
        "min_f1_threshold",
    )
    @classmethod
    def validate_percentage_fields(cls, v):
        """Validate percentage fields are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Percentage must be between 0 and 1, got {v}")
        return v

    @field_validator(
        "model_cache_size",
        "max_training_time_hours",
        "max_features",
        "feature_cache_ttl_hours",
        "optuna_n_trials",
        "optuna_timeout_hours",
        "cross_validation_folds",
        "validation_frequency_days",
        "drift_check_frequency_hours",
        "inference_batch_size",
        "prediction_cache_ttl_minutes",
        "max_cpu_cores",
    )
    @classmethod
    def validate_positive_integers(cls, v):
        """Validate positive integer fields."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    @field_validator("max_memory_gb")
    @classmethod
    def validate_positive_float(cls, v):
        """Validate positive float fields."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v


class ExecutionConfig(BaseConfig):
    """Execution engine configuration for order processing and algorithms."""

    # Order manager timing settings
    cleanup_interval_hours: int = Field(default=1, description="Cleanup interval in hours")
    idempotency_cleanup_interval_minutes: int = Field(default=60, description="Idempotency cleanup interval in minutes")
    connection_retry_delay_seconds: float = Field(default=5.0, description="Connection retry delay in seconds")
    connection_retry_max_delay_seconds: float = Field(default=30.0, description="Maximum connection retry delay")
    connection_retry_backoff_factor: float = Field(default=2.0, description="Connection retry backoff multiplier")

    # Order processing timing
    order_processing_delay_seconds: float = Field(default=0.1, description="Small delay between order processing")
    order_sync_delay_seconds: float = Field(default=1.0, description="Order synchronization delay")
    rate_limit_delay_seconds: float = Field(default=1.0, description="Rate limiting delay")

    # Algorithm-specific timing
    twap_slice_interval_buffer_seconds: float = Field(default=1.0, description="TWAP slice interval buffer")
    twap_max_wait_seconds: int = Field(default=300, description="TWAP maximum wait time")
    twap_error_recovery_delay_seconds: int = Field(default=30, description="TWAP error recovery delay")

    vwap_max_wait_seconds: int = Field(default=300, description="VWAP maximum wait time")
    vwap_error_delay_seconds: int = Field(default=10, description="VWAP error delay")
    vwap_slice_retry_delay_seconds: int = Field(default=5, description="VWAP slice retry delay")
    vwap_min_slice_interval_seconds: int = Field(default=30, description="VWAP minimum slice interval")
    vwap_monitoring_delay_seconds: int = Field(default=60, description="VWAP monitoring delay")

    iceberg_fill_monitoring_interval_seconds: int = Field(default=1, description="Iceberg fill monitoring interval")
    iceberg_error_delay_seconds: int = Field(default=10, description="Iceberg error delay")
    iceberg_retry_delay_seconds: int = Field(default=5, description="Iceberg retry delay")
    iceberg_refresh_delay_seconds: int = Field(default=2, description="Iceberg refresh delay base")
    iceberg_random_delay_range: int = Field(default=3, description="Iceberg random delay range")

    # Error handling delays
    rate_limit_retry_delay_seconds: int = Field(default=60, description="Rate limit retry delay")
    general_error_delay_seconds: int = Field(default=5, description="General error delay")
    busy_wait_prevention_delay_seconds: int = Field(default=1, description="Busy wait prevention delay")
    connection_error_delay_seconds: int = Field(default=10, description="Connection error delay")


class Config(BaseConfig):
    """Master configuration class for the entire application.

    Aggregates all configuration classes and provides utility methods
    for database URLs, environment checks, and schema generation.

    Attributes:
        environment: Application environment (development/staging/production)
        debug: Debug mode flag
        app_name: Application name
        app_version: Application version
        database: Database configuration settings
        security: Security configuration settings
        error_handling: Error handling configuration
        exchanges: Exchange API configuration
        risk: Risk management configuration
        capital_management: Capital management configuration
        strategies: Strategy management configuration
        execution: Execution engine configuration
        ml: Machine learning configuration
    """

    # Environment
    environment: str = Field(default="development", description="Application environment")
    debug: bool = Field(default=False, description="Debug mode")

    # Application
    app_name: str = Field(default="trading-bot-suite", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")

    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    security: SecurityConfig = SecurityConfig()
    error_handling: ErrorHandlingConfig = ErrorHandlingConfig()
    exchanges: ExchangeConfig = ExchangeConfig()
    risk: RiskConfig = RiskConfig()
    capital_management: CapitalManagementConfig = CapitalManagementConfig()
    strategies: StrategyManagementConfig = StrategyManagementConfig()
    execution: ExecutionConfig = ExecutionConfig()
    ml: MLConfig = MLConfig()

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v

    def generate_schema(self) -> None:
        """Generate JSON schema for configuration validation."""
        schema = self.model_json_schema()
        schema_path = Path("config/config.schema.json")
        schema_path.parent.mkdir(exist_ok=True)
        with open(schema_path, "w") as file_handle:
            json.dump(schema, file_handle, indent=2)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Config":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config instance loaded from YAML

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValidationError: If configuration validation fails
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        try:
            with open(yaml_path, encoding="utf-8") as file_handle:
                yaml_data = yaml.safe_load(file_handle)

            # Merge with environment variables (env vars take precedence)
            return cls(**yaml_data)

        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML configuration: {e}") from e

    @classmethod
    def from_yaml_with_env_override(cls, yaml_path: str | Path) -> "Config":
        """Load configuration from YAML file with environment variable overrides.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config instance with YAML base and env overrides
        """
        # First try to load from YAML if it exists
        yaml_path = Path(yaml_path)
        yaml_data: dict[str, Any] = {}

        if yaml_path.exists():
            try:
                with open(yaml_path, encoding="utf-8") as file_handle:
                    yaml_data = yaml.safe_load(file_handle) or {}
            except yaml.YAMLError as e:
                # If YAML parsing fails, log the error and fall back to env-only
                import logging

                logging.getLogger(__name__).warning(
                    "Failed to parse YAML configuration, falling back to environment variables",
                    extra={"yaml_path": str(yaml_path), "error_type": type(e).__name__},
                )
                yaml_data = {}

        # Create config with YAML data as base, env vars will override
        # due to pydantic-settings precedence
        os.environ.update(
            {k: str(v) for k, v in yaml_data.items() if k not in os.environ and v is not None}
        )

        return cls()

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save current configuration to YAML file.

        Args:
            yaml_path: Path where to save YAML configuration
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert config to dict and handle nested models
        config_dict = self.model_dump()

        with open(yaml_path, "w", encoding="utf-8") as file_handle:
            yaml.dump(config_dict, file_handle, default_flow_style=False, indent=2)

    def get_database_url(self) -> str:
        """Generate PostgreSQL database URL.

        Returns:
            PostgreSQL connection URL string
        """
        return (
            f"postgresql://{self.database.postgresql_username}:"
            f"{self.database.postgresql_password}@"
            f"{self.database.postgresql_host}:"
            f"{self.database.postgresql_port}/"
            f"{self.database.postgresql_database}"
        )

    def get_async_database_url(self) -> str:
        """Generate async PostgreSQL database URL.

        Returns:
            Async PostgreSQL connection URL string
        """
        return (
            f"postgresql+asyncpg://{self.database.postgresql_username}:"
            f"{self.database.postgresql_password}@"
            f"{self.database.postgresql_host}:"
            f"{self.database.postgresql_port}/"
            f"{self.database.postgresql_database}"
        )

    def get_redis_url(self) -> str:
        """Generate Redis connection URL.

        Returns:
            Redis connection URL string with optional password
        """
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

    def validate_yaml_config(self, yaml_path: str | Path) -> bool:
        """Validate YAML configuration file without loading.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            True if valid, False otherwise
        """
        try:
            self.from_yaml(yaml_path)
            return True
        except (FileNotFoundError, ValueError, yaml.YAMLError):
            return False
