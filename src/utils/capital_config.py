"""
Capital Management Configuration Utilities

This module provides shared configuration loading and management functions
for capital management services to eliminate code duplication.
"""

from decimal import Decimal
from typing import Any

from src.core.logging import get_logger

logger = get_logger(__name__)


def get_capital_config_defaults() -> dict[str, Any]:
    """
    Get default capital management configuration values.

    Returns:
        Dict with default configuration values
    """
    return {
        # Capital Management
        "total_capital": 100000,
        "emergency_reserve_pct": 0.1,
        "max_allocation_pct": 0.2,
        "max_daily_reallocation_pct": 0.1,
        # Currency Management
        "base_currency": "USDT",
        "hedging_threshold": 0.1,
        "hedge_ratio": 0.5,
        "supported_currencies": ["USDT", "BTC", "ETH"],
        "hedging_enabled": False,
        # Exchange Distribution
        "exchange_allocation_weights": "dynamic",
        "min_exchange_deposit_amount": 1000,
        "rebalance_frequency_hours": 24,
        # Fund Flow Management
        "auto_compound_enabled": True,
        "auto_compound_frequency": "weekly",
        "profit_threshold": 1000,
        "min_deposit_amount": 100,
        "min_withdrawal_amount": 100,
        "max_withdrawal_pct": 0.2,
        "fund_flow_cooldown_minutes": 60,
        "profit_lock_pct": 0.5,
        # Risk Management
        "max_daily_loss_pct": 0.05,
        "max_weekly_loss_pct": 0.10,
        "max_monthly_loss_pct": 0.20,
        # Rules and Limits
        "withdrawal_rules": {},
        "per_strategy_minimum": {},
    }


def load_capital_config(
    config_service: Any = None, config_key: str = "capital_management"
) -> dict[str, Any]:
    """
    Load capital management configuration with fallback to defaults.

    Args:
        config_service: ConfigService instance (optional)
        config_key: Configuration key to load

    Returns:
        Dict with configuration values
    """
    defaults = get_capital_config_defaults()

    if config_service is None:
        logger.warning("No ConfigService available, using default capital config")
        return defaults

    try:
        if isinstance(config_service, dict):
            # Direct config dict (for testing)
            config = config_service.copy()
        else:
            # Standard config service - load configuration
            config = config_service.get_config_value(config_key, {})

        # Merge with defaults to ensure all required values are present
        merged_config = defaults.copy()
        merged_config.update(config)

        return merged_config

    except Exception as e:
        logger.warning(f"Failed to load configuration from ConfigService: {e}, using defaults")
        return defaults


def resolve_config_service(dependency_resolver: Any = None) -> Any | None:
    """
    Resolve ConfigService from dependency injection with error handling.

    Args:
        dependency_resolver: Object with resolve_dependency method

    Returns:
        ConfigService instance or None if not available
    """
    if dependency_resolver is None:
        return None

    try:
        if hasattr(dependency_resolver, "resolve_dependency"):
            return dependency_resolver.resolve_dependency("ConfigService")
        else:
            logger.warning("Dependency resolver does not have resolve_dependency method")
            return None
    except Exception as e:
        logger.warning(f"Failed to resolve ConfigService: {e}")
        return None


def extract_decimal_config(config: dict[str, Any], key: str, default: Decimal) -> Decimal:
    """
    Safely extract Decimal value from configuration.

    Args:
        config: Configuration dictionary
        key: Configuration key
        default: Default Decimal value

    Returns:
        Decimal value from config or default
    """
    try:
        value = config.get(key)
        if value is None:
            return default
        return Decimal(str(value))
    except (ValueError, TypeError):
        logger.warning(
            f"Invalid decimal config value for {key}: {config.get(key)}, using default: {default}"
        )
        return default


def extract_percentage_config(config: dict[str, Any], key: str, default: float) -> Decimal:
    """
    Safely extract percentage value from configuration as Decimal.

    Args:
        config: Configuration dictionary
        key: Configuration key
        default: Default percentage value (0.0 to 1.0)

    Returns:
        Decimal percentage value from config or default
    """
    try:
        value = config.get(key, default)
        decimal_value = Decimal(str(value))
        # Ensure percentage is within reasonable bounds
        if decimal_value < Decimal("0") or decimal_value > Decimal("1"):
            logger.warning(
                f"Percentage config value {key}={decimal_value} out of range, using default: {default}"
            )
            return Decimal(str(default))
        return decimal_value
    except (ValueError, TypeError):
        logger.warning(
            f"Invalid percentage config value for {key}: {config.get(key)}, using default: {default}"
        )
        return Decimal(str(default))


def get_supported_currencies(config: dict[str, Any]) -> list[str]:
    """
    Get supported currencies list from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of supported currency codes
    """
    try:
        currencies = config.get("supported_currencies", ["USDT", "BTC", "ETH"])
        if isinstance(currencies, list) and all(isinstance(c, str) for c in currencies):
            return currencies
        else:
            logger.warning("Invalid supported_currencies config, using defaults")
            return ["USDT", "BTC", "ETH"]
    except Exception as e:
        logger.warning(f"Error getting supported currencies: {e}, using defaults")
        return ["USDT", "BTC", "ETH"]


def validate_config_values(config: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and sanitize configuration values.

    Args:
        config: Configuration dictionary

    Returns:
        Validated configuration dictionary
    """
    validated = config.copy()

    # Ensure all percentage values are reasonable
    percentage_keys = [
        "emergency_reserve_pct",
        "max_allocation_pct",
        "max_daily_reallocation_pct",
        "hedging_threshold",
        "hedge_ratio",
        "max_withdrawal_pct",
        "profit_lock_pct",
        "max_daily_loss_pct",
        "max_weekly_loss_pct",
        "max_monthly_loss_pct",
    ]

    for key in percentage_keys:
        if key in validated:
            try:
                value = Decimal(str(validated[key]))
                if value < Decimal("0") or value > Decimal("1"):
                    logger.warning(f"Invalid percentage value for {key}: {value}, using default")
                    defaults = get_capital_config_defaults()
                    validated[key] = defaults.get(key, 0.1)
            except (ValueError, TypeError):
                logger.warning(f"Invalid percentage config for {key}, using default")
                defaults = get_capital_config_defaults()
                validated[key] = defaults.get(key, 0.1)

    # Ensure positive amounts
    amount_keys = [
        "total_capital",
        "profit_threshold",
        "min_deposit_amount",
        "min_withdrawal_amount",
    ]

    for key in amount_keys:
        if key in validated:
            try:
                value = Decimal(str(validated[key]))
                if value <= Decimal("0"):
                    logger.warning(f"Invalid amount value for {key}: {value}, using default")
                    defaults = get_capital_config_defaults()
                    validated[key] = defaults.get(key, 1000)
            except (ValueError, TypeError):
                logger.warning(f"Invalid amount config for {key}, using default")
                defaults = get_capital_config_defaults()
                validated[key] = defaults.get(key, 1000)

    return validated
