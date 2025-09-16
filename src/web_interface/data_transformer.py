"""
Simple data transformation utilities for web interface.

Provides essential data transformation functions without over-engineering.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any


def format_decimal(value: Any) -> str:
    """Convert any numeric value to string representation."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (int, float)):
        return str(Decimal(str(value)))
    return str(value)


def add_timestamp(data: dict[str, Any]) -> dict[str, Any]:
    """Add current timestamp to data."""
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    return data


def ensure_decimal_strings(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure all numeric values are string representations."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (Decimal, int, float)):
            result[key] = format_decimal(value)
        elif isinstance(value, dict):
            result[key] = ensure_decimal_strings(value)
        elif isinstance(value, list):
            processed_list = []
            for item in value:
                if isinstance(item, dict):
                    processed_list.append(ensure_decimal_strings(item))
                elif isinstance(item, (Decimal, int, float)):
                    processed_list.append(format_decimal(item))
                else:
                    processed_list.append(item)
            result[key] = processed_list
        else:
            result[key] = value
    return result


# Legacy compatibility class - simplified
class WebInterfaceDataTransformer:
    """Simplified data transformer for backward compatibility."""

    @staticmethod
    def format_portfolio_composition(data: Any) -> Any:
        """Simple portfolio composition formatting."""
        if isinstance(data, dict):
            return ensure_decimal_strings(add_timestamp(data))
        return data

    @staticmethod
    def format_stress_test_results(data: Any) -> Any:
        """Simple stress test formatting."""
        if isinstance(data, dict):
            return ensure_decimal_strings(add_timestamp(data))
        return data

    @staticmethod
    def format_operational_metrics(data: Any) -> Any:
        """Simple operational metrics formatting."""
        if isinstance(data, dict):
            return ensure_decimal_strings(add_timestamp(data))
        return data

    @staticmethod
    def transform_risk_data_to_event_data(data: Any, **kwargs) -> Any:
        """Simple risk data transformation."""
        if isinstance(data, dict):
            return ensure_decimal_strings(add_timestamp(data))
        return data

    @staticmethod
    def validate_financial_precision(data: Any) -> Any:
        """Simple financial precision validation."""
        if isinstance(data, dict):
            return ensure_decimal_strings(data)
        return data
