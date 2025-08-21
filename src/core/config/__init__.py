"""
Configuration management for the T-Bot trading system.

This module provides both traditional configuration access and modern
service-based configuration management with dependency injection.

Modern Usage (Recommended):
    ```python
    # In service constructors
    def __init__(self, config_service: ConfigService):
        self.config_service = config_service


    # Accessing configuration
    db_config = self.config_service.get_database_config()
    ```

Legacy Usage (Backward Compatibility):
    ```python
    from src.core.config import get_config

    config = get_config()
    ```
"""

from .base import BaseConfig
from .capital import CapitalManagementConfig
from .database import DatabaseConfig
from .exchange import ExchangeConfig
from .main import Config, get_config
from .migration import migrate_legacy_config_usage, setup_config_service, validate_migration_status
from .risk import RiskConfig
from .security import SecurityConfig
from .service import (
    ConfigCache,
    ConfigChangeEvent,
    ConfigProvider,
    ConfigService,
    ConfigValidator,
    EnvironmentConfigProvider,
    FileConfigProvider,
    get_config_service,
    register_config_service_in_container,
    shutdown_config_service,
)
from .strategy import StrategyConfig


def load_config(config_file: str | None = None) -> Config:
    """
    Backward compatibility function for loading configuration.

    Args:
        config_file: Optional path to config file

    Returns:
        Config: Configuration instance
    """
    return get_config(config_file=config_file)


# For backward compatibility, export Config as the default
__all__ = [
    # Legacy configuration classes
    "BaseConfig",
    "CapitalManagementConfig",
    "Config",
    "ConfigCache",
    "ConfigChangeEvent",
    "ConfigProvider",
    # Modern service-based configuration
    "ConfigService",
    "ConfigValidator",
    "DatabaseConfig",
    "EnvironmentConfigProvider",
    "ExchangeConfig",
    "FileConfigProvider",
    "RiskConfig",
    "SecurityConfig",
    "StrategyConfig",
    "get_config",
    "get_config_service",
    "load_config",
    "migrate_legacy_config_usage",
    "register_config_service_in_container",
    # Migration utilities
    "setup_config_service",
    "shutdown_config_service",
    "validate_migration_status",
]
