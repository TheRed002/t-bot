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
    """Centralized error handling for analytics operations aligned with core patterns."""

    @staticmethod
    def create_operation_error(
        component_name: str,
        operation: str,
        target_entity: str | None = None,
        original_error: Exception | None = None,
    ) -> ComponentError:
        """Create standardized ComponentError for analytics operations aligned with core patterns."""

        message = f"Failed to {operation}"
        if target_entity:
            message += f" {target_entity}"

        # Apply consistent error metadata aligned with core patterns
        error_details = {
            "component": component_name,
            "operation": operation,
            "target_entity": target_entity,
            "original_error": str(original_error) if original_error else None,
            "error_type": type(original_error).__name__ if original_error else "AnalyticsError",
            "timestamp": get_current_utc_timestamp().isoformat(),
            "processing_mode": "stream",  # Align with core default
            "message_pattern": "pub_sub",  # Consistent messaging pattern
            "data_format": "analytics_error_v1",  # Versioned error format
            "boundary_crossed": True,  # Error crossing module boundaries
            "source_module": "analytics",  # Source identification
        }

        return ComponentError(
            message,
            component=component_name,
            operation=operation,
            details=error_details,
        )

    @staticmethod
    def propagate_analytics_error(
        error: Exception,
        context: str,
        target_module: str = "core"
    ) -> None:
        """Propagate analytics errors with consistent patterns aligned with core."""
        from datetime import datetime, timezone

        from src.core.logging import get_logger

        logger = get_logger(__name__)

        # Apply consistent error propagation metadata aligned with core patterns
        error_metadata = {
            "error_type": type(error).__name__,
            "context": context,
            "propagation_pattern": "analytics_to_core",
            "data_format": "analytics_error_propagation_v1",
            "processing_mode": "stream",  # Align with core events
            "message_pattern": "pub_sub",  # Consistent messaging
            "boundary_crossed": True,
            "validation_status": "failed",
            "source_module": "analytics",
            "target_module": target_module,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.error(f"Analytics error in {context}: {error}", extra=error_metadata)

        # Add propagation metadata to error if supported (aligned with core patterns)
        if hasattr(error, "__dict__"):
            try:
                error.__dict__.update({
                    "propagation_metadata": error_metadata,
                    "boundary_validation_applied": True,
                    "core_alignment": True,
                })
            except (AttributeError, TypeError):
                # Some exception types don't allow attribute modification
                pass

        # Apply boundary validation for consistency with core patterns
        try:
            from src.utils.messaging_patterns import BoundaryValidator
            boundary_data = {
                "component": "analytics",
                "severity": "medium",
                "timestamp": error_metadata["timestamp"],
                "processing_mode": "stream",
                "data_format": "analytics_error_propagation_v1",
                "message_pattern": "pub_sub",
                "boundary_crossed": True,
            }
            BoundaryValidator.validate_error_to_monitoring_boundary(boundary_data)
        except Exception as validation_error:
            logger.warning(f"Analytics error boundary validation failed: {validation_error}")

        raise error


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
    def prepare_service_config(config: AnalyticsConfiguration | dict | None) -> dict[str, Any]:
        """Prepare config dict for service initialization."""
        if isinstance(config, dict):
            # Convert dict to AnalyticsConfiguration first, then merge with defaults
            try:
                config = AnalyticsConfiguration(**config)
            except Exception:
                # If dict doesn't fit AnalyticsConfiguration, just return it
                return config
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
