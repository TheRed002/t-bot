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
from .environment import EnvironmentConfig
from .exchange import ExchangeConfig
from .execution import ExecutionConfig
from .risk import RiskConfig
from .sandbox import SandboxExchangeConfig
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
        self.environment_config = EnvironmentConfig()
        self.exchange = ExchangeConfig()
        self.sandbox = SandboxExchangeConfig()
        self.execution = ExecutionConfig()
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
        try:
            with open(config_path) as file_handle:
                if config_path.suffix in [".yaml", ".yml"]:
                    return yaml.safe_load(file_handle)
                elif config_path.suffix == ".json":
                    return json.load(file_handle)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        except Exception as e:
            raise ValueError(f"Failed to parse config file {config_path}: {e!s}") from e

    def _apply_config_data(self, config_data: dict[str, Any]) -> None:
        """Apply configuration data to domain configs."""
        # Define config section mappings
        config_mappings = {
            "database": self.database,
            "environment": self.environment_config,
            "exchange": self.exchange,
            "sandbox": self.sandbox,
            "execution": self.execution,
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
            "environment": self.environment_config.model_dump(),
            "exchange": self.exchange.model_dump(),
            "execution": self.execution.model_dump(),
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
        """Get configuration for a specific exchange with environment awareness."""
        # Get base configuration from exchange config
        base_config = self.exchange.get_exchange_credentials(exchange)

        # Merge with environment-specific settings
        env_config = self.get_environment_exchange_config(exchange)

        # Merge configurations (environment takes precedence)
        merged_config = {**base_config, **env_config}

        return merged_config

    def get_environment_exchange_config(self, exchange: str) -> dict[str, Any]:
        """Get environment-aware configuration for a specific exchange."""
        # Get environment-specific endpoints
        endpoints = self.environment_config.get_exchange_endpoints(exchange)

        # Get environment-specific credentials
        env_credentials = self.environment_config.get_exchange_credentials(exchange)

        # Get sandbox credentials if in sandbox mode
        sandbox_credentials = {}
        if not self.environment_config.is_production_environment(exchange):
            try:
                sandbox_credentials = self.sandbox.get_sandbox_credentials(exchange)
            except (AttributeError, KeyError, ValueError) as e:
                # Fallback if sandbox config not available
                self.logger.debug(f"Sandbox credentials not available for {exchange}: {e}")

        # Merge credentials (sandbox overrides environment for non-production)
        if sandbox_credentials:
            credentials = {**env_credentials, **sandbox_credentials}
        else:
            credentials = env_credentials

        # Combine endpoints and credentials
        return {
            **endpoints,
            **credentials,
            "environment_mode": self.environment_config.get_exchange_environment(exchange).value,
            "is_production": self.environment_config.is_production_environment(exchange),
        }

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
        """Custom JSON serializer for Decimal and Enum objects."""
        from datetime import datetime
        from enum import Enum
        from pathlib import Path

        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    def switch_environment(self, environment: str, exchange: str = None) -> bool:
        """
        Switch trading environment globally or for a specific exchange.
        
        Args:
            environment: Target environment ('sandbox', 'live', 'mock', 'hybrid')
            exchange: Optional specific exchange name, None for global switch
            
        Returns:
            bool: True if switch was successful
            
        Raises:
            ValueError: If environment is invalid or switching failed
        """
        from .environment import ExchangeEnvironment, TradingEnvironment

        if exchange:
            # Switch specific exchange environment
            exchange = exchange.lower()
            try:
                env_value = ExchangeEnvironment(environment.lower())
                if exchange == "binance":
                    self.environment_config.binance_environment = env_value
                elif exchange == "coinbase":
                    self.environment_config.coinbase_environment = env_value
                elif exchange == "okx":
                    self.environment_config.okx_environment = env_value
                else:
                    raise ValueError(f"Unknown exchange: {exchange}")

                return True
            except ValueError:
                raise ValueError(f"Invalid environment for exchange {exchange}: {environment}")
        else:
            # Switch global environment
            try:
                self.environment_config.global_environment = TradingEnvironment(environment.lower())
                return True
            except ValueError:
                raise ValueError(f"Invalid global environment: {environment}")

    def validate_environment_switch(self, environment: str, exchange: str = None) -> dict[str, Any]:
        """
        Validate if environment switch is safe and possible.
        
        Args:
            environment: Target environment
            exchange: Optional specific exchange name
            
        Returns:
            dict: Validation results with 'valid' bool and details
        """
        from .environment import ExchangeEnvironment, TradingEnvironment

        results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "required_actions": []
        }

        try:
            if exchange:
                # Validate exchange-specific switch
                env_value = ExchangeEnvironment(environment.lower())
                target_exchange = exchange.lower()

                # Check if production credentials are available for live trading
                if env_value in (ExchangeEnvironment.LIVE, ExchangeEnvironment.PRODUCTION):
                    if not self.environment_config.validate_production_credentials(target_exchange):
                        results["valid"] = False
                        results["errors"].append(
                            f"Production credentials not configured for {target_exchange}"
                        )
                        results["required_actions"].append(
                            f"Configure production API credentials for {target_exchange}"
                        )

                # Check production safeguards
                if (env_value in (ExchangeEnvironment.LIVE, ExchangeEnvironment.PRODUCTION) and
                    self.environment_config.enable_production_safeguards):
                    results["warnings"].append(
                        "Switching to production environment - ensure proper risk controls are in place"
                    )
                    if self.environment_config.production_confirmation:
                        results["required_actions"].append(
                            "Production confirmation required before switching"
                        )
            else:
                # Validate global switch
                env_value = TradingEnvironment(environment.lower())

                if env_value == TradingEnvironment.LIVE:
                    # Check all exchanges for production readiness
                    for exch in ["binance", "coinbase", "okx"]:
                        if not self.environment_config.validate_production_credentials(exch):
                            results["warnings"].append(
                                f"Production credentials not configured for {exch}"
                            )

        except ValueError:
            results["valid"] = False
            results["errors"].append(f"Invalid environment: {environment}")

        return results

    def get_current_environment_status(self) -> dict[str, Any]:
        """Get detailed status of current environment configuration."""
        return {
            "global_environment": self.environment_config.global_environment.value,
            "environment_summary": self.environment_config.get_environment_summary(),
            "exchange_configurations": {
                exchange: self.get_environment_exchange_config(exchange)
                for exchange in ["binance", "coinbase", "okx"]
            },
            "production_safeguards": {
                "enabled": self.environment_config.enable_production_safeguards,
                "confirmation_required": self.environment_config.production_confirmation,
                "credentials_validation": self.environment_config.require_credentials_validation,
            }
        }

    def is_production_mode(self, exchange: str = None) -> bool:
        """
        Check if system is in production mode.
        
        Args:
            exchange: Optional specific exchange, None for any exchange
            
        Returns:
            bool: True if in production mode
        """
        if exchange:
            return self.environment_config.is_production_environment(exchange)
        else:
            # Check if any exchange is in production mode
            return any(
                self.environment_config.is_production_environment(exch)
                for exch in ["binance", "coinbase", "okx"]
            )


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
