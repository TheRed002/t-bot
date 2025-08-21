"""Main configuration aggregator for the T-Bot trading system."""

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from .capital import CapitalManagementConfig
from .database import DatabaseConfig
from .exchange import ExchangeConfig
from .risk import RiskConfig
from .security import SecurityConfig
from .strategy import StrategyConfig


class Config:
    """
    Main configuration aggregator that maintains backward compatibility.

    This class aggregates all domain-specific configurations and provides
    a unified interface for accessing configuration values.
    """

    def __init__(self, config_file: str | None = None, env_file: str | None = ".env"):
        """
        Initialize configuration from environment and optional config file.

        Args:
            config_file: Optional path to YAML/JSON config file
            env_file: Path to .env file (default: ".env")
        """
        # Set env file for all configs
        if env_file and os.path.exists(env_file):
            os.environ["ENV_FILE"] = env_file

        # Initialize domain configs
        self.database = DatabaseConfig()
        self.exchange = ExchangeConfig()
        self.strategy = StrategyConfig()
        self.risk = RiskConfig()
        self.security = SecurityConfig()
        self.capital_management = CapitalManagementConfig()

        # Load from config file if provided
        if config_file:
            self.load_from_file(config_file)

        # App-level configuration (not domain-specific)
        self.app_name = os.getenv("APP_NAME", "T-Bot Trading System")
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.data_dir = Path(os.getenv("DATA_DIR", "./data"))
        self.logs_dir = Path(os.getenv("LOGS_DIR", "./logs"))

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def load_from_file(self, config_file: str) -> None:
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_path) as f:
            if config_path.suffix in [".yaml", ".yml"]:
                config_data = yaml.safe_load(f)
            elif config_path.suffix == ".json":
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        # Update domain configs with file data
        if "database" in config_data:
            for key, value in config_data["database"].items():
                if hasattr(self.database, key):
                    setattr(self.database, key, value)

        if "exchange" in config_data:
            for key, value in config_data["exchange"].items():
                if hasattr(self.exchange, key):
                    setattr(self.exchange, key, value)

        if "strategy" in config_data:
            for key, value in config_data["strategy"].items():
                if hasattr(self.strategy, key):
                    setattr(self.strategy, key, value)

        if "risk" in config_data:
            for key, value in config_data["risk"].items():
                if hasattr(self.risk, key):
                    setattr(self.risk, key, value)

        if "security" in config_data:
            for key, value in config_data["security"].items():
                if hasattr(self.security, key):
                    setattr(self.security, key, value)

        if "capital_management" in config_data:
            for key, value in config_data["capital_management"].items():
                if hasattr(self.capital_management, key):
                    setattr(self.capital_management, key, value)

    def save_to_file(self, config_file: str) -> None:
        """Save current configuration to YAML or JSON file."""
        config_path = Path(config_file)

        config_data = {
            "app": {
                "name": self.app_name,
                "environment": self.environment,
                "debug": self.debug,
                "log_level": self.log_level,
                "data_dir": str(self.data_dir),
                "logs_dir": str(self.logs_dir),
            },
            "database": self.database.model_dump(),
            "exchange": self.exchange.model_dump(),
            "strategy": self.strategy.model_dump(),
            "risk": self.risk.model_dump(),
            "security": self.security.model_dump(),
            "capital_management": self.capital_management.model_dump(),
        }

        with open(config_path, "w") as f:
            if config_path.suffix in [".yaml", ".yml"]:
                yaml.dump(config_data, f, default_flow_style=False)
            elif config_path.suffix == ".json":
                json.dump(config_data, f, indent=2, default=self._json_serializer)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    def validate(self) -> None:
        """Validate all configurations.

        Pydantic models validate automatically on instantiation,
        so this method just performs any additional custom validation.
        """
        # Pydantic validates on instantiation, but we can add custom checks here
        if not self.database.postgresql_database:
            raise ValueError("PostgreSQL database name is required")
        if not self.exchange.enabled_exchanges:
            raise ValueError("At least one exchange must be enabled")

    # Backward compatibility properties
    @property
    def db_url(self) -> str:
        """Backward compatibility for database URL."""
        return self.database.postgresql_url

    @property
    def redis_url(self) -> str:
        """Backward compatibility for Redis URL."""
        return self.database.redis_url

    @property
    def postgresql_host(self) -> str:
        """Backward compatibility."""
        return self.database.postgresql_host

    @property
    def postgresql_port(self) -> int:
        """Backward compatibility."""
        return self.database.postgresql_port

    @property
    def postgresql_database(self) -> str:
        """Backward compatibility."""
        return self.database.postgresql_database

    @property
    def postgresql_username(self) -> str:
        """Backward compatibility."""
        return self.database.postgresql_username

    @property
    def postgresql_password(self) -> str:
        """Backward compatibility."""
        return self.database.postgresql_password

    @property
    def redis_host(self) -> str:
        """Backward compatibility."""
        return self.database.redis_host

    @property
    def redis_port(self) -> int:
        """Backward compatibility."""
        return self.database.redis_port

    @property
    def binance_api_key(self) -> str:
        """Backward compatibility."""
        return self.exchange.binance_api_key

    @property
    def binance_api_secret(self) -> str:
        """Backward compatibility."""
        return self.exchange.binance_api_secret

    @property
    def max_position_size(self) -> Any:
        """Backward compatibility."""
        return self.risk.max_position_size

    @property
    def risk_per_trade(self) -> float:
        """Backward compatibility."""
        return self.risk.risk_per_trade

    def get_exchange_config(self, exchange: str) -> dict[str, Any]:
        """Get configuration for a specific exchange."""
        return self.exchange.get_exchange_credentials(exchange)

    def get_strategy_config(self, strategy_type: str) -> dict[str, Any]:
        """Get configuration for a specific strategy."""
        return self.strategy.get_strategy_params(strategy_type)

    def get_risk_config(self) -> dict[str, Any]:
        """Get risk management configuration."""
        return self.risk.get_position_size_params()

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "app": {
                "name": self.app_name,
                "environment": self.environment,
                "debug": self.debug,
                "log_level": self.log_level,
            },
            "database": self.database.model_dump(),
            "exchange": self.exchange.model_dump(),
            "strategy": self.strategy.model_dump(),
            "risk": self.risk.model_dump(),
            "security": self.security.model_dump(),
            "capital_management": self.capital_management.model_dump(),
        }

    def _json_serializer(self, obj):
        """Custom JSON serializer for Decimal objects."""
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# Global config instance (singleton pattern)
_config_instance: Config | None = None


def get_config(config_file: str | None = None, reload: bool = False) -> Config:
    """
    Get or create the global configuration instance.

    Args:
        config_file: Optional path to config file
        reload: Force reload of configuration

    Returns:
        Global Config instance
    """
    global _config_instance

    if _config_instance is None or reload:
        _config_instance = Config(config_file=config_file)

    return _config_instance
