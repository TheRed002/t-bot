"""
Configuration management for the trading bot framework.

This module provides comprehensive Pydantic-based configuration system that will be
extended by ALL subsequent prompts. Use exact patterns from @COMMON_PATTERNS.md.

CRITICAL: This file will be extended by ALL subsequent prompts. Use exact patterns from @COMMON_PATTERNS.md.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Dict, List, Optional, Any, Union
import json
import yaml
from pathlib import Path
import os


class BaseConfig(BaseSettings):
    """Base configuration class with common patterns.
    
    Provides common Pydantic settings configuration with environment
    variable support, case insensitive matching, and validation.
    """
    
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
    postgresql_database: str = Field(default="trading_bot", pattern=r"^[a-zA-Z0-9_-]+$", description="PostgreSQL database name")
    postgresql_username: str = Field(default="trading_bot", pattern=r"^[a-zA-Z0-9_-]+$", description="PostgreSQL username")
    postgresql_password: str = Field(default="trading_bot_password", min_length=8, description="PostgreSQL password")
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
    def validate_ports(cls, v: int) -> int:
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
    jwt_algorithm: str = Field(default="HS256", pattern=r"^(HS256|HS384|HS512|RS256|RS384|RS512)$", description="JWT algorithm")
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
    rate_limits: Dict[str, Dict[str, int]] = Field(
        default={
            "binance": {
                "requests_per_minute": 1200,
                "orders_per_second": 10,
                "websocket_connections": 5
            },
            "okx": {
                "requests_per_minute": 600,
                "orders_per_second": 20,
                "websocket_connections": 3
            },
            "coinbase": {
                "requests_per_minute": 600,
                "orders_per_second": 15,
                "websocket_connections": 4
            }
        },
        description="Exchange-specific rate limits"
    )
    
    # Supported exchanges
    supported_exchanges: List[str] = Field(
        default=["binance", "okx", "coinbase"],
        description="List of supported exchanges"
    )
    
    @field_validator('binance_api_key', 'binance_api_secret', 'okx_api_key', 'okx_api_secret', 'okx_passphrase', 'coinbase_api_key', 'coinbase_api_secret')
    @classmethod
    def validate_api_credentials(cls, v: str) -> str:
        """Validate API credentials - either empty (for testing) or minimum length."""
        if v and len(v) < 16:  # If provided, must be at least 16 chars
            raise ValueError(f'API credential must be at least 16 characters long, got {len(v)}')
        return v


class RiskConfig(BaseConfig):
    """Risk management configuration for P-008 framework."""
    
    # Position sizing
    default_position_size_method: str = Field(
        default="fixed_percentage", 
        description="Default position sizing method"
    )
    default_position_size_pct: float = Field(
        default=0.02, 
        description="Default position size percentage (2%)"
    )
    max_position_size_pct: float = Field(
        default=0.1, 
        description="Maximum position size percentage (10%)"
    )
    
    # Portfolio limits
    max_total_positions: int = Field(
        default=10, 
        description="Maximum total positions"
    )
    max_positions_per_symbol: int = Field(
        default=1, 
        description="Maximum positions per symbol"
    )
    max_portfolio_exposure: float = Field(
        default=0.95, 
        description="Maximum portfolio exposure (95%)"
    )
    max_sector_exposure: float = Field(
        default=0.25, 
        description="Maximum sector exposure (25%)"
    )
    max_correlation_exposure: float = Field(
        default=0.5, 
        description="Maximum correlation exposure (50%)"
    )
    max_leverage: float = Field(
        default=1.0, 
        description="Maximum leverage (no leverage by default)"
    )
    
    # Risk thresholds
    max_daily_loss_pct: float = Field(
        default=0.05, 
        description="Maximum daily loss percentage (5%)"
    )
    max_drawdown_pct: float = Field(
        default=0.15, 
        description="Maximum drawdown percentage (15%)"
    )
    var_confidence_level: float = Field(
        default=0.95, 
        ge=0.5, 
        le=0.999,
        description="VaR confidence level (95%)"
    )
    
    # Kelly Criterion settings
    kelly_lookback_days: int = Field(
        default=30, 
        description="Kelly Criterion lookback period in days"
    )
    kelly_max_fraction: float = Field(
        default=0.25, 
        description="Maximum Kelly fraction (25%)"
    )
    
    # Volatility adjustment
    volatility_window: int = Field(
        default=20, 
        description="Volatility calculation window"
    )
    volatility_target: float = Field(
        default=0.02, 
        description="Volatility target (2% daily)"
    )
    
    # Risk calculation settings
    var_calculation_window: int = Field(
        default=252, 
        description="VaR calculation window (trading days)"
    )
    drawdown_calculation_window: int = Field(
        default=252, 
        description="Drawdown calculation window (trading days)"
    )
    correlation_calculation_window: int = Field(
        default=60, 
        description="Correlation calculation window (days)"
    )
    
    # Emergency controls (P-009)
    emergency_close_positions: bool = Field(
        default=True, 
        description="Close all positions during emergency stop"
    )
    emergency_recovery_timeout_hours: int = Field(
        default=1, 
        description="Recovery timeout in hours after emergency stop"
    )
    emergency_manual_override_enabled: bool = Field(
        default=True, 
        description="Enable manual override for emergency controls"
    )
    
    @field_validator('default_position_size_pct', 'max_position_size_pct', 'max_portfolio_exposure', 
                    'max_sector_exposure', 'max_correlation_exposure', 'max_daily_loss_pct', 
                    'max_drawdown_pct', 'var_confidence_level', 'kelly_max_fraction', 'volatility_target')
    @classmethod
    def validate_percentage_fields(cls, v):
        """Validate percentage fields are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f'Percentage must be between 0 and 1, got {v}')
        return v
    
    @field_validator('max_total_positions', 'kelly_lookback_days', 'volatility_window', 
                    'var_calculation_window', 'drawdown_calculation_window', 'correlation_calculation_window')
    @classmethod
    def validate_positive_integers(cls, v):
        """Validate positive integer fields."""
        if v <= 0:
            raise ValueError(f'Value must be positive, got {v}')
        return v


class CapitalManagementConfig(BaseConfig):
    """Capital management configuration for P-010A framework."""
    
    # Base settings
    base_currency: str = Field(default="USDT", pattern=r"^[A-Z]{3,10}$", description="Base currency for all calculations")
    total_capital: float = Field(default=10000.0, gt=0, description="Total available capital")
    emergency_reserve_pct: float = Field(default=0.1, description="Emergency reserve percentage (10%)")
    
    # Allocation strategy
    allocation_strategy: str = Field(default="risk_parity", description="Capital allocation strategy")
    rebalance_frequency_hours: int = Field(default=24, description="Rebalancing frequency in hours")
    min_allocation_pct: float = Field(default=0.01, description="Minimum allocation percentage (1%)")
    max_allocation_pct: float = Field(default=0.25, description="Maximum allocation percentage (25%)")
    
    # Exchange distribution
    max_exchange_allocation_pct: float = Field(default=0.6, description="Maximum exchange allocation (60%)")
    min_exchange_balance: float = Field(default=100.0, description="Minimum exchange balance")
    
    # Fund flow controls
    max_daily_reallocation_pct: float = Field(default=0.2, description="Maximum daily reallocation (20%)")
    fund_flow_cooldown_minutes: int = Field(default=60, description="Fund flow cooldown in minutes")
    min_deposit_amount: float = Field(default=1000.0, description="Minimum deposit amount")
    min_withdrawal_amount: float = Field(default=100.0, description="Minimum withdrawal amount")
    max_withdrawal_pct: float = Field(default=0.2, description="Maximum withdrawal percentage (20%)")
    
    # Currency management
    supported_currencies: List[str] = Field(
        default=["USDT", "BUSD", "USDC", "BTC", "ETH"],
        description="Supported currencies for trading"
    )
    hedging_enabled: bool = Field(default=True, description="Enable currency hedging")
    hedging_threshold: float = Field(default=0.1, description="Hedging threshold (10% exposure)")
    hedge_ratio: float = Field(default=0.8, description="Hedge coverage ratio (80%)")
    
    # Withdrawal rules
    withdrawal_rules: Dict[str, Dict[str, Any]] = Field(
        default={
            "profit_only": {
                "enabled": True,
                "description": "Only withdraw realized profits"
            },
            "maintain_minimum": {
                "enabled": True,
                "description": "Keep minimum capital for each strategy"
            },
            "performance_based": {
                "enabled": False,
                "threshold": 0.05,
                "description": "Allow withdrawals only if performance > threshold"
            }
        },
        description="Withdrawal rule configurations"
    )
    
    # Auto-compounding
    auto_compound_enabled: bool = Field(default=True, description="Enable auto-compounding")
    auto_compound_frequency: str = Field(default="weekly", description="Auto-compound frequency")
    profit_threshold: float = Field(default=100.0, description="Minimum profit for compounding")
    
    # Capital protection
    max_daily_loss_pct: float = Field(default=0.05, description="Maximum daily loss (5%)")
    max_weekly_loss_pct: float = Field(default=0.10, description="Maximum weekly loss (10%)")
    max_monthly_loss_pct: float = Field(default=0.15, description="Maximum monthly loss (15%)")
    profit_lock_pct: float = Field(default=0.5, description="Profit lock percentage (50%)")
    
    # Per-strategy minimum allocations
    per_strategy_minimum: Dict[str, float] = Field(
        default={
            "mean_reversion": 5000.0,
            "trend_following": 10000.0,
            "ml_strategy": 15000.0,
            "arbitrage": 20000.0,
            "market_making": 25000.0
        },
        description="Minimum capital requirements per strategy"
    )
    
    # Exchange allocation weights
    exchange_allocation_weights: Dict[str, float] = Field(
        default={
            "binance": 0.5,
            "okx": 0.3,
            "coinbase": 0.2
        },
        description="Default exchange allocation weights"
    )
    
    @field_validator('emergency_reserve_pct', 'min_allocation_pct', 'max_allocation_pct', 
                    'max_exchange_allocation_pct', 'max_daily_reallocation_pct', 
                    'hedging_threshold', 'hedge_ratio', 'max_daily_loss_pct', 
                    'max_weekly_loss_pct', 'max_monthly_loss_pct', 'profit_lock_pct')
    @classmethod
    def validate_percentage_fields(cls, v):
        """Validate percentage fields are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f'Percentage must be between 0 and 1, got {v}')
        return v
    
    @field_validator('rebalance_frequency_hours', 'fund_flow_cooldown_minutes')
    @classmethod
    def validate_positive_integers(cls, v):
        """Validate positive integer fields."""
        if v <= 0:
            raise ValueError(f'Value must be positive, got {v}')
        return v
    
    @field_validator('total_capital', 'min_exchange_balance', 'profit_threshold')
    @classmethod
    def validate_positive_floats(cls, v):
        """Validate positive float fields."""
        if v <= 0:
            raise ValueError(f'Value must be positive, got {v}')
        return v


class StrategyManagementConfig(BaseConfig):
    """Strategy management configuration for P-011 framework."""
    
    # Global strategy settings
    max_concurrent_strategies: int = Field(default=10, description="Maximum concurrent strategies")
    strategy_restart_delay: int = Field(default=60, description="Strategy restart delay in seconds")
    performance_window_days: int = Field(default=30, description="Performance evaluation window in days")
    
    # Default strategy parameters
    default_min_confidence: float = Field(default=0.6, description="Default minimum signal confidence")
    default_position_size: float = Field(default=0.02, description="Default position size percentage")
    default_stop_loss: float = Field(default=0.02, description="Default stop loss percentage")
    default_take_profit: float = Field(default=0.04, description="Default take profit percentage")
    
    # Hot reloading
    enable_hot_reload: bool = Field(default=True, description="Enable hot reloading of strategies")
    config_check_interval: int = Field(default=30, description="Configuration check interval in seconds")
    
    # Strategy performance thresholds
    min_win_rate: float = Field(default=0.4, description="Minimum acceptable win rate")
    min_sharpe_ratio: float = Field(default=0.5, description="Minimum acceptable Sharpe ratio")
    max_drawdown_threshold: float = Field(default=0.15, description="Maximum acceptable drawdown")
    
    # Strategy monitoring
    performance_evaluation_frequency: int = Field(default=24, description="Performance evaluation frequency in hours")
    auto_disable_poor_performers: bool = Field(default=True, description="Auto-disable poorly performing strategies")
    performance_alert_threshold: float = Field(default=0.3, description="Performance alert threshold")
    
    @field_validator('max_concurrent_strategies', 'strategy_restart_delay', 'performance_window_days', 
                    'config_check_interval', 'performance_evaluation_frequency')
    @classmethod
    def validate_positive_integers(cls, v):
        """Validate positive integer fields."""
        if v <= 0:
            raise ValueError(f'Value must be positive, got {v}')
        return v
    
    @field_validator('default_min_confidence', 'default_position_size', 'default_stop_loss', 
                    'default_take_profit', 'min_win_rate', 'min_sharpe_ratio', 'max_drawdown_threshold',
                    'performance_alert_threshold')
    @classmethod
    def validate_percentage_fields(cls, v):
        """Validate percentage fields are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f'Percentage must be between 0 and 1, got {v}')
        return v


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
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
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
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config':
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
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            # Merge with environment variables (env vars take precedence)
            return cls(**yaml_data)
            
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML configuration: {e}")
    
    @classmethod
    def from_yaml_with_env_override(cls, yaml_path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML file with environment variable overrides.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Config instance with YAML base and env overrides
        """
        # First try to load from YAML if it exists
        yaml_path = Path(yaml_path)
        yaml_data = {}
        
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                # If YAML parsing fails, fall back to env-only
                yaml_data = {}
        
        # Create config with YAML data as base, env vars will override
        # due to pydantic-settings precedence
        os.environ.update({k: str(v) for k, v in yaml_data.items() 
                          if k not in os.environ and v is not None})
        
        return cls()
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save current configuration to YAML file.
        
        Args:
            yaml_path: Path where to save YAML configuration
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dict and handle nested models
        config_dict = self.model_dump()
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

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
    
    def validate_yaml_config(self, yaml_path: Union[str, Path]) -> bool:
        """Validate YAML configuration file without loading.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            self.from_yaml(yaml_path)
            return True
        except Exception:
            return False


# REVERSE INTEGRATION POINTS: Future prompts will add these config classes:
# - P-002: Extended DatabaseConfig with connection pooling settings
# - P-003: ExchangeConfig for API credentials and rate limits
# - P-008: RiskConfig for risk management parameters
# - P-010A: CapitalConfig for fund management settings
# - P-011: StrategyConfig for strategy parameters
# - P-017: MLConfig for model registry and training
# - P-026: WebConfig for FastAPI and security settings 