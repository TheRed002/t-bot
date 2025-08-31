"""Main configuration aggregator for the T-Bot trading system."""

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from .bot_management import BotManagementConfig
from .capital import CapitalManagementConfig
from .database import DatabaseConfig
from .exchange import ExchangeConfig
from .risk import RiskConfig
from .security import SecurityConfig
from .state_management import StateManagementConfig
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
        self.bot_management = BotManagementConfig()
        self.state_management = StateManagementConfig()

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
        self._validate_config_file(config_path)

        config_data = self._parse_config_file(config_path)
        self._apply_config_data(config_data)

    def _validate_config_file(self, config_path: Path) -> None:
        """Validate config file exists and has supported format."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if config_path.suffix not in [".yaml", ".yml", ".json"]:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    def _parse_config_file(self, config_path: Path) -> dict[str, Any]:
        """Parse config file based on format."""
        file_handle = None
        try:
            file_handle = open(config_path)
            if config_path.suffix in [".yaml", ".yml"]:
                return yaml.safe_load(file_handle)
            elif config_path.suffix == ".json":
                return json.load(file_handle)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        finally:
            if file_handle:
                file_handle.close()

    def _apply_config_data(self, config_data: dict[str, Any]) -> None:
        """Apply configuration data to domain configs."""
        # Define config section mappings
        config_mappings = {
            "database": self.database,
            "exchange": self.exchange,
            "strategy": self.strategy,
            "risk": self.risk,
            "security": self.security,
            "capital_management": self.capital_management,
            "bot_management": self.bot_management,
            "state_management": self.state_management,
        }

        for section_name, config_obj in config_mappings.items():
            self._update_config_section(config_data, section_name, config_obj)

    def _update_config_section(
        self, config_data: dict[str, Any], section_name: str, config_obj: Any
    ) -> None:
        """Update a single config section if it exists in the data."""
        if section_name not in config_data:
            return

        section_data = config_data[section_name]
        for key, value in section_data.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)

    def save_to_file(self, config_file: str) -> None:
        """Save current configuration to YAML or JSON file."""
        config_path = Path(config_file)
        config_data = self._build_config_data()
        self._write_config_file(config_path, config_data)

    def _build_config_data(self) -> dict[str, Any]:
        """Build complete configuration data for serialization."""
        return {
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
            "bot_management": self.bot_management.model_dump(),
            "state_management": self.state_management.model_dump(),
        }

    def _write_config_file(self, config_path: Path, config_data: dict[str, Any]) -> None:
        """Write configuration data to file in appropriate format."""
        with open(config_path, "w") as file_handle:
            if config_path.suffix in [".yaml", ".yml"]:
                yaml.dump(config_data, file_handle, default_flow_style=False)
            elif config_path.suffix == ".json":
                json.dump(config_data, file_handle, indent=2, default=self._json_serializer)
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
    def postgresql_password(self) -> str | None:
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
        # Reuse the config data building logic but with minimal app data
        config_data = self._build_config_data()
        # Simplify app section for to_dict
        config_data["app"] = {
            "name": self.app_name,
            "environment": self.environment,
            "debug": self.debug,
            "log_level": self.log_level,
        }
        return config_data

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
