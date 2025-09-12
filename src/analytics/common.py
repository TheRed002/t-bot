"""
Analytics Common Utilities - Shared functionality across analytics module.

This module consolidates common patterns and utilities used throughout the analytics module.
"""

from decimal import Decimal
from typing import Any

from src.analytics.types import AnalyticsConfiguration
from src.core.exceptions import ComponentError
from src.utils.datetime_utils import get_current_utc_timestamp


class AnalyticsErrorHandler:
    """Centralized error handling for analytics operations."""

    @staticmethod
    def create_operation_error(
        component_name: str,
        operation: str,
        target_entity: str | None = None,
        original_error: Exception | None = None,
    ) -> ComponentError:
        """Create standardized ComponentError for analytics operations."""
        message = f"Failed to {operation}"
        if target_entity:
            message += f" {target_entity}"

        return ComponentError(
            message,
            component=component_name,
            operation=operation,
        )


class AnalyticsCalculations:
    """Common calculation utilities for analytics."""

    @staticmethod
    def calculate_percentage_change(old_value: Decimal, new_value: Decimal) -> Decimal:
        """Calculate percentage change between two values."""
        if old_value == 0:
            return Decimal("0")
        return (new_value - old_value) / old_value * Decimal("100")

    @staticmethod
    def calculate_simple_var(total_exposure: Decimal, confidence_level: Decimal) -> Decimal:
        """Calculate simple Value at Risk."""
        return total_exposure * confidence_level

    @staticmethod
    def calculate_position_weight(
        position_value: Decimal, total_portfolio_value: Decimal
    ) -> Decimal:
        """Calculate position weight in portfolio."""
        if total_portfolio_value == 0:
            return Decimal("0")
        return position_value / total_portfolio_value


class ConfigurationDefaults:
    """Default configuration values for analytics services."""

    @staticmethod
    def get_default_config() -> AnalyticsConfiguration:
        """Get default analytics configuration."""
        return AnalyticsConfiguration()

    @staticmethod
    def merge_config(config: AnalyticsConfiguration | None) -> AnalyticsConfiguration:
        """Merge provided config with defaults."""
        if config is None:
            return ConfigurationDefaults.get_default_config()
        return config


class ServiceInitializationHelper:
    """Helper for consistent service initialization patterns."""

    @staticmethod
    def prepare_service_config(config: AnalyticsConfiguration | None) -> dict[str, Any]:
        """Prepare config dict for service initialization."""
        merged_config = ConfigurationDefaults.merge_config(config)
        return merged_config.model_dump()

    @staticmethod
    def initialize_common_state() -> dict[str, Any]:
        """Initialize common state variables used across services."""
        return {
            "last_calculation_time": get_current_utc_timestamp(),
            "calculation_count": 0,
            "error_count": 0,
        }


class MetricsDefaults:
    """Default return values for metrics when data is unavailable."""

    @staticmethod
    def empty_portfolio_metrics() -> dict[str, Any]:
        """Return empty portfolio metrics structure."""
        return {
            "total_value": Decimal("0"),
            "total_pnl": Decimal("0"),
            "positions_count": 0,
            "timestamp": get_current_utc_timestamp(),
        }

    @staticmethod
    def empty_risk_metrics() -> dict[str, Any]:
        """Return empty risk metrics structure."""
        return {
            "var_95": Decimal("0"),
            "total_exposure": Decimal("0"),
            "positions_count": 0,
            "timestamp": get_current_utc_timestamp(),
        }

    @staticmethod
    def empty_operational_metrics() -> dict[str, Any]:
        """Return empty operational metrics structure."""
        return {
            "timestamp": get_current_utc_timestamp(),
            "services_active": 0,
            "calculations_completed": 0,
        }
