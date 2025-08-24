"""Configuration Service for the T-Bot trading system.

This module provides a service-based approach to configuration management with
dependency injection support, eliminating direct config imports throughout the codebase.

Key Features:
- Dependency injection ready
- Thread-safe configuration access
- Configuration validation and caching
- Hot-reload capability for development
- Environment-specific configuration support
- Comprehensive logging and monitoring
- Type-safe configuration access
- Configuration change notifications

Architecture:
- ConfigService: Main service interface for configuration access
- ConfigProvider: Abstract interface for configuration providers
- FileConfigProvider: Implementation for file-based configuration
- EnvironmentConfigProvider: Implementation for environment-based configuration
- ConfigCache: Thread-safe caching layer for performance
- ConfigValidator: Validation service for configuration integrity

Usage Example:
    ```python
    # In service constructors - dependency injection
    def __init__(self, config_service: ConfigService):
        self.config_service = config_service


    # Accessing configuration
    db_config = self.config_service.get_database_config()
    exchange_config = self.config_service.get_exchange_config("binance")

    # Type-safe access
    max_position_size: Decimal = self.config_service.get_risk_config().max_position_size
    ```

TODO: Update the following modules to use ConfigService:
- src/main.py (replace direct Config import)
- src/risk_management/*.py (15+ files with direct imports)
- src/bot_management/*.py (20+ files with direct imports)
- src/execution/*.py (10+ files with direct imports)
- src/strategies/*.py (25+ files with direct imports)
- All other modules with direct config imports (50+ files)
"""

import asyncio
import json
import logging
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, Field, ValidationError

from ..exceptions import (
    ConfigurationError,
    ErrorCategory,
    ValidationError as TradingValidationError,
)
from ..types.base import BaseValidatedModel
from .database import DatabaseConfig
from .exchange import ExchangeConfig
from .main import Config
from .risk import RiskConfig
from .strategy import StrategyConfig

# Type aliases for better readability
ConfigDict = dict[str, Any]
ConfigKey = str
ConfigValue = Any
ConfigCallback = Callable[[ConfigKey, ConfigValue, ConfigValue], None]

T = TypeVar("T", bound=BaseModel)


class ConfigProvider(Protocol):
    """Protocol for configuration providers."""

    async def load_config(self) -> ConfigDict:
        """Load configuration from the provider."""
        ...

    async def save_config(self, config: ConfigDict) -> None:
        """Save configuration to the provider."""
        ...

    async def watch_changes(self, callback: ConfigCallback) -> None:
        """Watch for configuration changes."""
        ...


class ConfigChangeEvent(BaseValidatedModel):
    """Configuration change event."""

    key: str = Field(..., description="Configuration key that changed")
    old_value: Any = Field(None, description="Previous value")
    new_value: Any = Field(..., description="New value")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = Field("unknown", description="Source of the change")


class ConfigCache:
    """Thread-safe configuration cache with TTL support."""

    def __init__(self, default_ttl: int = 300):  # 5 minutes default TTL
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl
        self._access_count: dict[str, int] = {}
        self.logger = logging.getLogger(f"{__name__}.ConfigCache")

    def get(self, key: str, default: Any = None) -> Any:
        """Get cached value with TTL check."""
        with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if datetime.now(timezone.utc) < expiry:
                    self._access_count[key] = self._access_count.get(key, 0) + 1
                    return value
                else:
                    # Expired - remove from cache
                    del self._cache[key]
                    if key in self._access_count:
                        del self._access_count[key]
            return default

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set cached value with TTL."""
        ttl = ttl or self._default_ttl
        expiry = datetime.now(timezone.utc).timestamp() + ttl
        expiry_dt = datetime.fromtimestamp(expiry, tz=timezone.utc)

        with self._lock:
            self._cache[key] = (value, expiry_dt)
            self.logger.debug(f"Cached config key '{key}' with TTL {ttl}s")

    def invalidate(self, key: str) -> None:
        """Invalidate specific cache key."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self.logger.debug(f"Invalidated cache key '{key}'")
            if key in self._access_count:
                del self._access_count[key]

    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
            self._access_count.clear()
            self.logger.info("Cleared all cache entries")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "total_keys": len(self._cache),
                "access_counts": self._access_count.copy(),
                "cache_keys": list(self._cache.keys()),
                "total_accesses": sum(self._access_count.values()),
            }


class FileConfigProvider:
    """File-based configuration provider."""

    def __init__(self, config_file: Path, watch_changes: bool = False):
        self.config_file = config_file
        self.watch_changes_enabled = watch_changes
        self.logger = logging.getLogger(f"{__name__}.FileConfigProvider")
        self._watchers: list[ConfigCallback] = []
        self._last_modified: datetime | None = None

    async def load_config(self) -> ConfigDict:
        """Load configuration from file."""
        try:
            if not self.config_file.exists():
                raise ConfigurationError(
                    f"Configuration file not found: {self.config_file}",
                    error_code="CONFIG_FILE_NOT_FOUND",
                    category=ErrorCategory.CONFIGURATION,
                    details={"file_path": str(self.config_file)},
                )

            # Check if file was modified
            current_modified = datetime.fromtimestamp(
                self.config_file.stat().st_mtime, tz=timezone.utc
            )

            with open(self.config_file) as f:
                if self.config_file.suffix.lower() in [".yaml", ".yml"]:
                    import yaml

                    config_data = yaml.safe_load(f)
                elif self.config_file.suffix.lower() == ".json":
                    config_data = json.load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported config file format: {self.config_file.suffix}",
                        error_code="CONFIG_UNSUPPORTED_FORMAT",
                        category=ErrorCategory.CONFIGURATION,
                        details={
                            "file_path": str(self.config_file),
                            "suffix": self.config_file.suffix,
                        },
                    )

            self._last_modified = current_modified
            self.logger.info(f"Loaded configuration from {self.config_file}")
            return config_data or {}

        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(
                f"Failed to load configuration from {self.config_file}: {e!s}",
                error_code="CONFIG_LOAD_FAILED",
                category=ErrorCategory.CONFIGURATION,
                details={"file_path": str(self.config_file), "error": str(e)},
            )

    async def save_config(self, config: ConfigDict) -> None:
        """Save configuration to file."""
        try:
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_file, "w") as f:
                if self.config_file.suffix.lower() in [".yaml", ".yml"]:
                    import yaml

                    yaml.dump(config, f, default_flow_style=False, sort_keys=True)
                elif self.config_file.suffix.lower() == ".json":
                    json.dump(config, f, indent=2, sort_keys=True, default=str)
                else:
                    raise ConfigurationError(
                        f"Unsupported config file format for saving: {self.config_file.suffix}",
                        error_code="CONFIG_SAVE_UNSUPPORTED_FORMAT",
                        category=ErrorCategory.CONFIGURATION,
                    )

            self.logger.info(f"Saved configuration to {self.config_file}")

        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(
                f"Failed to save configuration to {self.config_file}: {e!s}",
                error_code="CONFIG_SAVE_FAILED",
                category=ErrorCategory.CONFIGURATION,
                details={"file_path": str(self.config_file), "error": str(e)},
            )

    async def watch_changes(self, callback: ConfigCallback) -> None:
        """Watch for file changes (simplified implementation)."""
        if not self.watch_changes_enabled:
            return

        self._watchers.append(callback)
        # In a real implementation, this would use file system watchers
        # For now, we'll implement periodic checking in the ConfigService


class EnvironmentConfigProvider:
    """Environment-based configuration provider."""

    def __init__(self, prefix: str = "TBOT_"):
        self.prefix = prefix
        self.logger = logging.getLogger(f"{__name__}.EnvironmentConfigProvider")

    async def load_config(self) -> ConfigDict:
        """Load configuration from environment variables."""
        import os

        config: dict[str, Any] = {}
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # Convert TBOT_DATABASE_HOST to database.host
                config_key = key[len(self.prefix) :].lower().replace("_", ".")
                config = self._set_nested_value(config, config_key, value)

        self.logger.debug(f"Loaded {len(config)} environment configuration keys")
        return config

    def _set_nested_value(self, config: dict[str, Any], key: str, value: str) -> dict[str, Any]:
        """Set nested dictionary value from dot notation key."""
        keys = key.split(".")
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
        return config

    async def save_config(self, config: ConfigDict) -> None:
        """Environment provider doesn't support saving."""
        raise ConfigurationError(
            "Environment configuration provider does not support saving",
            error_code="CONFIG_ENV_SAVE_NOT_SUPPORTED",
            category=ErrorCategory.CONFIGURATION,
        )

    async def watch_changes(self, callback: ConfigCallback) -> None:
        """Environment provider doesn't support watching."""
        # Environment variables don't typically change during runtime
        pass


class ConfigValidator:
    """Configuration validation service."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ConfigValidator")
        self._custom_validators: dict[str, Callable[[dict], Any]] = {}

    async def validate_database_config(self, config: dict) -> DatabaseConfig:
        """Validate database configuration."""
        try:
            return DatabaseConfig.model_validate(config)
        except ValidationError as e:
            raise TradingValidationError(
                f"Invalid database configuration: {e!s}",
                error_code="CONFIG_DATABASE_INVALID",
                category=ErrorCategory.VALIDATION,
                details={"validation_errors": e.errors(), "config": config},
            )

    async def validate_exchange_config(self, config: dict) -> ExchangeConfig:
        """Validate exchange configuration."""
        try:
            return ExchangeConfig.model_validate(config)
        except ValidationError as e:
            raise TradingValidationError(
                f"Invalid exchange configuration: {e!s}",
                error_code="CONFIG_EXCHANGE_INVALID",
                category=ErrorCategory.VALIDATION,
                details={"validation_errors": e.errors(), "config": config},
            )

    async def validate_risk_config(self, config: dict) -> RiskConfig:
        """Validate risk configuration."""
        try:
            return RiskConfig.model_validate(config)
        except ValidationError as e:
            raise TradingValidationError(
                f"Invalid risk configuration: {e!s}",
                error_code="CONFIG_RISK_INVALID",
                category=ErrorCategory.VALIDATION,
                details={"validation_errors": e.errors(), "config": config},
            )

    async def validate_strategy_config(self, config: dict) -> StrategyConfig:
        """Validate strategy configuration."""
        try:
            return StrategyConfig.model_validate(config)
        except ValidationError as e:
            raise TradingValidationError(
                f"Invalid strategy configuration: {e!s}",
                error_code="CONFIG_STRATEGY_INVALID",
                category=ErrorCategory.VALIDATION,
                details={"validation_errors": e.errors(), "config": config},
            )

    def register_validator(self, config_section: str, validator: Callable[[dict], Any]) -> None:
        """Register a custom validator for a configuration section."""
        self._custom_validators[config_section] = validator
        self.logger.info(f"Registered custom validator for config section: {config_section}")

    async def validate_custom_config(self, section: str, config: dict) -> Any:
        """Validate configuration using a custom validator."""
        if section not in self._custom_validators:
            raise TradingValidationError(
                f"No custom validator registered for section: {section}",
                error_code="CONFIG_CUSTOM_VALIDATOR_NOT_FOUND",
                category=ErrorCategory.VALIDATION,
            )

        try:
            return self._custom_validators[section](config)
        except Exception as e:
            raise TradingValidationError(
                f"Custom validation failed for section {section}: {e!s}",
                error_code="CONFIG_CUSTOM_VALIDATION_FAILED",
                category=ErrorCategory.VALIDATION,
                details={"section": section, "config": config, "error": str(e)},
            )


class ConfigService:
    """Main configuration service with dependency injection support.

    This service provides a clean interface for accessing configuration
    throughout the application without direct imports or global state.

    Features:
    - Thread-safe configuration access
    - Caching with TTL support
    - Configuration validation
    - Hot-reload capability
    - Change notifications
    - Multiple configuration providers
    - Type-safe configuration access

    Example Usage:
        ```python
        # Initialize service (usually done in main application)
        config_service = ConfigService()
        await config_service.initialize()

        # Inject into services
        risk_service = RiskService(config_service=config_service)

        # Access configuration
        db_config = config_service.get_database_config()
        exchange_config = config_service.get_exchange_config("binance")
        ```
    """

    def __init__(
        self,
        providers: list[ConfigProvider] | None = None,
        cache_ttl: int = 300,
        enable_hot_reload: bool = False,
        hot_reload_interval: int = 30,
    ):
        """Initialize ConfigService.

        Args:
            providers: List of configuration providers
            cache_ttl: Cache time-to-live in seconds
            enable_hot_reload: Enable automatic configuration reloading
            hot_reload_interval: Hot reload check interval in seconds
        """
        self.providers = providers or []
        self.cache = ConfigCache(default_ttl=cache_ttl)
        self.validator = ConfigValidator()
        self.enable_hot_reload = enable_hot_reload
        self.hot_reload_interval = hot_reload_interval

        self._config: Config | None = None
        self._lock = asyncio.Lock()
        self._initialized = False
        self._change_listeners: list[ConfigCallback] = []
        self._hot_reload_task: asyncio.Task | None = None

        self.logger = logging.getLogger(f"{__name__}.ConfigService")

    async def initialize(
        self, config_file: str | Path | None = None, watch_changes: bool = False
    ) -> None:
        """Initialize the configuration service.

        Args:
            config_file: Optional configuration file path
            watch_changes: Enable file watching for changes
        """
        async with self._lock:
            if self._initialized:
                self.logger.warning("ConfigService already initialized")
                return

            # Add default providers if none provided
            if not self.providers:
                # Environment provider (highest priority)
                self.providers.append(EnvironmentConfigProvider())

                # File provider (if file specified)
                if config_file:
                    file_path = Path(config_file) if isinstance(config_file, str) else config_file
                    self.providers.append(FileConfigProvider(file_path, watch_changes))

            # Load initial configuration
            await self._load_configuration()

            # Start hot reload if enabled
            if self.enable_hot_reload:
                self._hot_reload_task = asyncio.create_task(self._hot_reload_loop())

            self._initialized = True
            self.logger.info("ConfigService initialized successfully")

    async def shutdown(self) -> None:
        """Shutdown the configuration service."""
        async with self._lock:
            if self._hot_reload_task and not self._hot_reload_task.done():
                self._hot_reload_task.cancel()
                try:
                    await self._hot_reload_task
                except asyncio.CancelledError:
                    pass

            self.cache.clear()
            self._change_listeners.clear()
            self._initialized = False
            self.logger.info("ConfigService shutdown completed")

    async def _load_configuration(self) -> None:
        """Load configuration from all providers."""
        merged_config: dict[str, Any] = {}

        # Load from all providers (later providers override earlier ones)
        for provider in self.providers:
            try:
                provider_config = await provider.load_config()
                merged_config = self._deep_merge(merged_config, provider_config)
                self.logger.debug(f"Loaded config from {provider.__class__.__name__}")
            except Exception as e:
                self.logger.error(f"Failed to load from {provider.__class__.__name__}: {e}")
                # Continue with other providers
                continue

        # Create Config instance
        try:
            # Convert merged config to Config instance
            # Since Config expects individual domain configs, we need to structure the data
            self._config = Config()

            # Update domain configs if present in merged config
            if "database" in merged_config:
                self._config.database = await self.validator.validate_database_config(
                    merged_config["database"]
                )

            if "exchange" in merged_config:
                self._config.exchange = await self.validator.validate_exchange_config(
                    merged_config["exchange"]
                )

            if "risk" in merged_config:
                self._config.risk = await self.validator.validate_risk_config(merged_config["risk"])

            if "strategy" in merged_config:
                self._config.strategy = await self.validator.validate_strategy_config(
                    merged_config["strategy"]
                )

            # Update app-level configs
            app_config = merged_config.get("app", {})
            if "environment" in app_config:
                self._config.environment = app_config["environment"]
            if "debug" in app_config:
                self._config.debug = app_config["debug"]
            if "log_level" in app_config:
                self._config.log_level = app_config["log_level"]

            # Clear cache after config reload
            self.cache.clear()

            self.logger.info("Configuration loaded and validated successfully")

        except Exception as e:
            raise ConfigurationError(
                f"Failed to create configuration instance: {e!s}",
                error_code="CONFIG_CREATION_FAILED",
                category=ErrorCategory.CONFIGURATION,
                details={"merged_config": merged_config, "error": str(e)},
            )

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    async def _hot_reload_loop(self) -> None:
        """Hot reload loop for configuration changes."""
        while True:
            try:
                await asyncio.sleep(self.hot_reload_interval)

                # Check if any provider has changes
                # This is a simplified implementation
                # Real implementation would use file system watchers

                old_config = self._config
                await self._load_configuration()

                # Notify listeners of changes
                if old_config and self._config:
                    await self._notify_changes(old_config, self._config)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in hot reload loop: {e}")
                # Continue the loop

    async def _notify_changes(self, old_config: Config, new_config: Config) -> None:
        """Notify listeners of configuration changes."""
        # Compare configurations and notify of changes
        # This is a simplified implementation
        changes = []

        # Compare each domain config
        if old_config.database != new_config.database:
            changes.append(("database", old_config.database, new_config.database))

        if old_config.exchange != new_config.exchange:
            changes.append(("exchange", old_config.exchange, new_config.exchange))

        # Notify listeners
        for key, old_value, new_value in changes:
            for listener in self._change_listeners:
                try:
                    listener(key, old_value, new_value)
                except Exception as e:
                    self.logger.error(f"Error notifying config change listener: {e}")

    def add_change_listener(self, callback: ConfigCallback) -> None:
        """Add a configuration change listener."""
        self._change_listeners.append(callback)

    def remove_change_listener(self, callback: ConfigCallback) -> None:
        """Remove a configuration change listener."""
        if callback in self._change_listeners:
            self._change_listeners.remove(callback)

    # Configuration Access Methods

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        self._ensure_initialized()

        cached = self.cache.get("database_config")
        if cached is not None:
            return cached

        assert self._config is not None  # _ensure_initialized guarantees this
        config = self._config.database
        self.cache.set("database_config", config)
        return config

    def get_exchange_config(self, exchange: str | None = None) -> ExchangeConfig | dict[str, Any]:
        """Get exchange configuration."""
        self._ensure_initialized()

        assert self._config is not None  # _ensure_initialized guarantees this
        if exchange:
            cache_key = f"exchange_config_{exchange}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

            config = self._config.get_exchange_config(exchange)
            self.cache.set(cache_key, config)
            return config
        else:
            cached = self.cache.get("exchange_config")
            if cached is not None:
                return cached

            config = self._config.exchange
            self.cache.set("exchange_config", config)
            return config

    def get_risk_config(self) -> RiskConfig:
        """Get risk management configuration."""
        self._ensure_initialized()

        cached = self.cache.get("risk_config")
        if cached is not None:
            return cached

        assert self._config is not None  # _ensure_initialized guarantees this
        config = self._config.risk
        self.cache.set("risk_config", config)
        return config

    def get_strategy_config(
        self, strategy_type: str | None = None
    ) -> StrategyConfig | dict[str, Any]:
        """Get strategy configuration."""
        self._ensure_initialized()

        assert self._config is not None  # _ensure_initialized guarantees this
        if strategy_type:
            cache_key = f"strategy_config_{strategy_type}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

            config = self._config.get_strategy_config(strategy_type)
            self.cache.set(cache_key, config)
            return config
        else:
            cached = self.cache.get("strategy_config")
            if cached is not None:
                return cached

            config = self._config.strategy
            self.cache.set("strategy_config", config)
            return config

    def get_app_config(self) -> dict[str, Any]:
        """Get application-level configuration."""
        self._ensure_initialized()

        cached = self.cache.get("app_config")
        if cached is not None:
            return cached

        assert self._config is not None  # _ensure_initialized guarantees this
        config = {
            "name": self._config.app_name,
            "environment": self._config.environment,
            "debug": self._config.debug,
            "log_level": self._config.log_level,
            "data_dir": str(self._config.data_dir),
            "logs_dir": str(self._config.logs_dir),
        }

        self.cache.set("app_config", config)
        return config

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key."""
        self._ensure_initialized()

        # Cache the lookup
        cache_key = f"config_value_{key}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Navigate through the configuration
        assert self._config is not None  # _ensure_initialized guarantees this
        value = self._get_nested_value(self._config.to_dict(), key, default)
        self.cache.set(cache_key, value)
        return value

    def _get_nested_value(self, config: dict, key: str, default: Any = None) -> Any:
        """Get nested dictionary value from dot notation key."""
        keys = key.split(".")
        current = config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def _ensure_initialized(self) -> None:
        """Ensure the service is initialized."""
        if not self._initialized or self._config is None:
            raise ConfigurationError(
                "ConfigService not initialized. Call initialize() first.",
                error_code="CONFIG_SERVICE_NOT_INITIALIZED",
                category=ErrorCategory.CONFIGURATION,
            )

    # Context manager support
    async def __aenter__(self) -> "ConfigService":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()

    # Statistics and debugging
    def get_cache_stats(self) -> dict[str, Any]:
        """Get configuration cache statistics."""
        return self.cache.get_stats()

    def invalidate_cache(self, key: str | None = None) -> None:
        """Invalidate configuration cache."""
        if key:
            self.cache.invalidate(key)
        else:
            self.cache.clear()

    def get_loaded_config(self) -> dict[str, Any] | None:
        """Get the currently loaded configuration as dictionary."""
        if self._config is not None:
            return self._config.to_dict()
        return None
    
    def get_config_dict(self) -> dict[str, Any]:
        """Get the currently loaded configuration as dictionary.
        
        This is an alias for get_loaded_config() for compatibility.
        Returns empty dict if config is not loaded.
        """
        loaded = self.get_loaded_config()
        return loaded if loaded is not None else {}
    
    def get_config(self) -> dict[str, Any]:
        """Get the currently loaded configuration as dictionary.
        
        This is an alias for get_config_dict() for compatibility.
        Returns empty dict if config is not loaded.
        """
        return self.get_config_dict()


# Singleton instance for backward compatibility
_config_service_instance: ConfigService | None = None


async def get_config_service(
    config_file: str | Path | None = None, reload: bool = False
) -> ConfigService:
    """Get or create the global ConfigService instance.

    Args:
        config_file: Optional configuration file path
        reload: Force reload of configuration service

    Returns:
        Global ConfigService instance
    """
    global _config_service_instance

    if _config_service_instance is None or reload:
        _config_service_instance = ConfigService()
        await _config_service_instance.initialize(config_file=config_file)

    return _config_service_instance


async def shutdown_config_service() -> None:
    """Shutdown the global configuration service."""
    global _config_service_instance

    if _config_service_instance:
        await _config_service_instance.shutdown()
        _config_service_instance = None


def register_config_service_in_container(
    config_file: str | None = None, enable_hot_reload: bool = False
) -> None:
    """Register ConfigService in the dependency injection container.

    This function should be called during application startup to make
    ConfigService available for dependency injection throughout the system.

    Args:
        config_file: Optional path to configuration file
        enable_hot_reload: Enable hot reloading of configuration
    """
    from ..dependency_injection import get_container

    # Create ConfigService factory
    async def config_service_factory() -> ConfigService:
        service = ConfigService(enable_hot_reload=enable_hot_reload)
        await service.initialize(config_file=config_file)
        return service

    # Register as singleton in the container
    container = get_container()
    container.register("ConfigService", config_service_factory, singleton=True)
    container.register("config_service", config_service_factory, singleton=True)

    # Log successful registration
    import logging

    logger = logging.getLogger(__name__)
    logger.info("ConfigService registered in dependency injection container")
