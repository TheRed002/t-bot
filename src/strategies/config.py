"""
Strategy configuration management for unified parameter handling.

This module provides configuration validation, default parameter management,
and environment-specific configurations for all strategies.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.core.exceptions import ConfigurationError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import StrategyConfig, StrategyType

# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution
from src.utils.validators import validate_strategy_config

logger = get_logger(__name__)


class StrategyConfigurationManager:
    """Manager for strategy configuration handling."""

    def __init__(self, config_dir: str = "config/strategies"):
        """Initialize configuration manager.

        Args:
            config_dir: Directory containing strategy configurations
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Default configurations for different strategy types
        self._default_configs = self._initialize_default_configs()

        logger.info("Strategy configuration manager initialized", config_dir=str(self.config_dir))

    def _initialize_default_configs(self) -> dict[str, dict[str, Any]]:
        """Initialize default configurations for different strategy types.

        Returns:
            Dictionary of default configurations
        """
        return {
            "mean_reversion": {
                "name": "mean_reversion",
                "strategy_type": StrategyType.STATIC,
                "enabled": True,
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "timeframe": "5m",
                "min_confidence": 0.6,
                "max_positions": 5,
                "position_size_pct": 0.02,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "parameters": {
                    "lookback_period": 20,
                    "entry_threshold": 2.0,
                    "exit_threshold": 0.5,
                    "volatility_window": 20,
                },
            },
            "trend_following": {
                "name": "trend_following",
                "strategy_type": StrategyType.STATIC,
                "enabled": True,
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "timeframe": "1h",
                "min_confidence": 0.6,
                "max_positions": 5,
                "position_size_pct": 0.02,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "parameters": {
                    "fast_ma": 20,
                    "slow_ma": 50,
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30,
                },
            },
            "breakout": {
                "name": "breakout",
                "strategy_type": StrategyType.STATIC,
                "enabled": True,
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "timeframe": "1h",
                "min_confidence": 0.6,
                "max_positions": 5,
                "position_size_pct": 0.02,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "parameters": {
                    "support_resistance_period": 20,
                    "breakout_confirmation_periods": 3,
                    "volume_threshold": 1.5,
                    "consolidation_period": 10,
                },
            },
        }

    @time_execution
    def load_strategy_config(self, strategy_name: str) -> StrategyConfig:
        """Load strategy configuration from file or defaults.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Strategy configuration

        Raises:
            ConfigurationError: If configuration loading fails
        """
        try:
            # Try to load from file first
            config_file = self.config_dir / f"{strategy_name}.yaml"
            if config_file.exists():
                config_data = self._load_config_file(config_file)
                logger.info(
                    "Loaded strategy config from file",
                    strategy_name=strategy_name,
                    config_file=str(config_file),
                )
            else:
                # Use default configuration
                config_data = self._get_default_config(strategy_name)
                logger.info("Using default strategy config", strategy_name=strategy_name)

            # Validate configuration
            validate_strategy_config(config_data)

            # Create config object
            config = StrategyConfig(**config_data)

            return config

        except Exception as e:
            logger.error(
                "Failed to load strategy config", strategy_name=strategy_name, error=str(e)
            )
            raise ConfigurationError(f"Failed to load configuration for {strategy_name}: {e!s}")

    def _load_config_file(self, config_file: Path) -> dict[str, Any]:
        """Load configuration from file.

        Args:
            config_file: Path to configuration file

        Returns:
            Configuration dictionary
        """
        with open(config_file) as f:
            if config_file.suffix == ".yaml" or config_file.suffix == ".yml":
                return yaml.safe_load(f)
            elif config_file.suffix == ".json":
                return json.load(f)
            else:
                raise ConfigurationError(f"Unsupported config file format: {config_file.suffix}")

    def _get_default_config(self, strategy_name: str) -> dict[str, Any]:
        """Get default configuration for strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Default configuration dictionary
        """
        if strategy_name not in self._default_configs:
            raise ConfigurationError(f"No default configuration for strategy: {strategy_name}")

        return self._default_configs[strategy_name].copy()

    @time_execution
    def save_strategy_config(self, strategy_name: str, config: StrategyConfig) -> None:
        """Save strategy configuration to file.

        Args:
            strategy_name: Name of the strategy
            config: Strategy configuration to save
        """
        try:
            config_file = self.config_dir / f"{strategy_name}.yaml"

            # Convert to dictionary
            config_dict = config.model_dump()

            # Ensure strategy_type is a string for YAML serialization
            if "strategy_type" in config_dict and hasattr(config_dict["strategy_type"], "value"):
                config_dict["strategy_type"] = config_dict["strategy_type"].value

            # Add metadata
            config_dict["_metadata"] = {
                "last_updated": datetime.now().isoformat(),
                "version": "1.0.0",
            }

            # Save to file
            with open(config_file, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            logger.info(
                "Strategy config saved", strategy_name=strategy_name, config_file=str(config_file)
            )

        except Exception as e:
            logger.error(
                "Failed to save strategy config", strategy_name=strategy_name, error=str(e)
            )
            raise ConfigurationError(f"Failed to save configuration for {strategy_name}: {e!s}")

    @time_execution
    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate strategy configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            validate_strategy_config(config)
            return True
        except Exception as e:
            logger.warning("Configuration validation failed", error=str(e))
            return False

    def get_available_strategies(self) -> list[str]:
        """Get list of available strategies.

        Returns:
            List of strategy names
        """
        strategies = []

        # Add strategies with config files
        for config_file in self.config_dir.glob("*.yaml"):
            strategy_name = config_file.stem
            strategies.append(strategy_name)

        # Add strategies with default configs
        for strategy_name in self._default_configs.keys():
            if strategy_name not in strategies:
                strategies.append(strategy_name)

        return sorted(strategies)

    def get_config_schema(self) -> dict[str, Any]:
        """Get configuration schema for validation.

        Returns:
            Configuration schema
        """
        return StrategyConfig.model_json_schema()

    @time_execution
    def update_config_parameter(self, strategy_name: str, parameter: str, value: Any) -> bool:
        """Update a single configuration parameter.

        Args:
            strategy_name: Name of the strategy
            parameter: Parameter name to update
            value: New parameter value

        Returns:
            True if update successful, False otherwise
        """
        try:
            # Load current config
            config = self.load_strategy_config(strategy_name)

            # Update parameter
            config_dict = config.model_dump()

            # Ensure strategy_type is a string for validation
            if "strategy_type" in config_dict and hasattr(config_dict["strategy_type"], "value"):
                config_dict["strategy_type"] = config_dict["strategy_type"].value

            if parameter in config_dict:
                config_dict[parameter] = value

                # Validate updated config
                if self.validate_config(config_dict):
                    # Save updated config
                    updated_config = StrategyConfig(**config_dict)
                    self.save_strategy_config(strategy_name, updated_config)
                    logger.info(
                        "Config parameter updated",
                        strategy_name=strategy_name,
                        parameter=parameter,
                        value=value,
                    )
                    return True
                else:
                    logger.error(
                        "Invalid configuration after parameter update",
                        strategy_name=strategy_name,
                        parameter=parameter,
                    )
                    return False
            else:
                logger.error(
                    "Parameter not found in config",
                    strategy_name=strategy_name,
                    parameter=parameter,
                )
                return False

        except Exception as e:
            logger.error(
                "Failed to update config parameter",
                strategy_name=strategy_name,
                parameter=parameter,
                error=str(e),
            )
            return False

    def create_strategy_config(
        self, strategy_name: str, strategy_type: StrategyType, symbols: list[str], **kwargs
    ) -> StrategyConfig:
        """Create a new strategy configuration.

        Args:
            strategy_name: Name of the strategy
            strategy_type: Type of strategy
            symbols: Trading symbols
            **kwargs: Additional configuration parameters

        Returns:
            New strategy configuration
        """
        try:
            # Start with default config for strategy type
            config_data = self._get_default_config(strategy_name)

            # Update with provided parameters
            config_data.update(
                {
                    "name": strategy_name,
                    "strategy_type": strategy_type.value
                    if hasattr(strategy_type, "value")
                    else strategy_type,
                    "symbols": symbols,
                    **kwargs,
                }
            )

            # Validate configuration
            validate_strategy_config(config_data)

            # Create and save configuration
            config = StrategyConfig(**config_data)
            self.save_strategy_config(strategy_name, config)

            logger.info(
                "Strategy config created",
                strategy_name=strategy_name,
                strategy_type=strategy_type.value
                if hasattr(strategy_type, "value")
                else strategy_type,
            )

            return config

        except Exception as e:
            logger.error(
                "Failed to create strategy config", strategy_name=strategy_name, error=str(e)
            )
            raise ConfigurationError(f"Failed to create configuration for {strategy_name}: {e!s}")

    def delete_strategy_config(self, strategy_name: str) -> bool:
        """Delete strategy configuration file.

        Args:
            strategy_name: Name of the strategy

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            config_file = self.config_dir / f"{strategy_name}.yaml"
            if config_file.exists():
                config_file.unlink()
                logger.info(
                    "Strategy config deleted",
                    strategy_name=strategy_name,
                    config_file=str(config_file),
                )
                return True
            else:
                logger.warning(
                    "Strategy config file not found for deletion", strategy_name=strategy_name
                )
                return False

        except Exception as e:
            logger.error(
                "Failed to delete strategy config", strategy_name=strategy_name, error=str(e)
            )
            return False

    def get_config_summary(self) -> dict[str, Any]:
        """Get summary of all configurations.

        Returns:
            Configuration summary
        """
        summary = {
            "config_directory": str(self.config_dir),
            "total_strategies": len(self.get_available_strategies()),
            "config_files": [],
            "default_configs": list(self._default_configs.keys()),
        }

        # List config files
        for config_file in self.config_dir.glob("*.yaml"):
            summary["config_files"].append(
                {
                    "name": config_file.stem,
                    "size": config_file.stat().st_size,
                    "modified": datetime.fromtimestamp(config_file.stat().st_mtime).isoformat(),
                }
            )

        return summary
