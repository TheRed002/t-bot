"""
Financial Utilities for Data Validation and Transformation.

This module provides centralized financial data validation and transformation
utilities to prevent code duplication across modules.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from src.utils.decimal_utils import to_decimal


def get_logger_safe(name: str):
    """Get logger with robust fallback for test environments."""
    try:
        from src.core.logging import get_logger
        return get_logger(name)
    except ImportError:
        import logging
        return logging.getLogger(name)


logger = get_logger_safe(__name__)


def validate_financial_precision(data: dict[str, Any], financial_fields: list[str] | None = None) -> dict[str, Any]:
    """
    Centralized function to validate financial precision for decimal values.

    This function replaces duplicate implementations across data transformers
    to ensure consistent financial precision validation.

    Args:
        data: Data dictionary to validate
        financial_fields: List of field names to validate. If None, uses default fields.

    Returns:
        Validated data with proper Decimal conversions
    """
    validated_data = data.copy()

    # Default financial fields if none specified
    if financial_fields is None:
        financial_fields = [
            "price", "quantity", "volume", "fees", "pnl", "size",
            "entry_price", "exit_price", "target_quantity", "filled_quantity",
            "total_return_pct", "annual_return_pct", "max_drawdown_pct"
        ]

    # Also check for fields ending with common financial suffixes
    financial_suffixes = [
        "_amount", "_pnl", "_return", "_ratio", "_drawdown",
        "_percentage", "_price", "_quantity", "_volume", "_fees"
    ]

    for key, value in validated_data.items():
        should_validate = False

        # Check if field is in explicit list
        if key in financial_fields:
            should_validate = True

        # Check if field ends with financial suffix
        if not should_validate:
            for suffix in financial_suffixes:
                if key.endswith(suffix):
                    should_validate = True
                    break

        if should_validate and value is not None:
            try:
                if isinstance(value, str):
                    # Use core decimal utils for proper conversion
                    decimal_val = to_decimal(value)
                    validated_data[key] = str(decimal_val)
                elif isinstance(value, (int, float)):
                    # Convert to Decimal and back to string for consistency
                    decimal_val = to_decimal(value)
                    validated_data[key] = str(decimal_val)
            except Exception as e:
                logger.warning(f"Failed to convert {key}={value} to Decimal: {e}")
                # Keep original value when conversion fails
                validated_data[key] = value

    return validated_data


def ensure_boundary_fields(
    data: dict[str, Any],
    source: str,
    processing_mode: str | None = None,
    data_format: str | None = None
) -> dict[str, Any]:
    """
    Centralized function to ensure boundary fields for cross-service communication.

    This function replaces duplicate implementations across data transformers
    to ensure consistent boundary field handling.

    Args:
        data: Data dictionary to enhance
        source: Source module/service name
        processing_mode: Processing mode (stream, batch, etc.). If None, uses default based on source.
        data_format: Data format string. If None, uses default based on source.

    Returns:
        Data with required boundary fields
    """
    enhanced_data = data.copy()

    # Set source
    if "source" not in enhanced_data:
        enhanced_data["source"] = source

    # Set correlation_id if not present
    if "correlation_id" not in enhanced_data:
        enhanced_data["correlation_id"] = str(uuid.uuid4())

    # Set timestamp if not present
    if "timestamp" not in enhanced_data:
        enhanced_data["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Set processed_at if not present
    if "processed_at" not in enhanced_data:
        enhanced_data["processed_at"] = datetime.now(timezone.utc).isoformat()

    # Set processing_mode based on source defaults or parameter
    if "processing_mode" not in enhanced_data:
        if processing_mode:
            enhanced_data["processing_mode"] = processing_mode
        elif source in ["backtesting", "optimization"]:
            enhanced_data["processing_mode"] = "batch"
        else:
            enhanced_data["processing_mode"] = "stream"

    # Set data_format based on source defaults or parameter
    if "data_format" not in enhanced_data:
        if data_format:
            enhanced_data["data_format"] = data_format
        elif source == "backtesting":
            enhanced_data["data_format"] = "batch_event_data_v1"
        elif source == "optimization":
            enhanced_data["data_format"] = "optimization_event_v1"
        else:
            enhanced_data["data_format"] = "bot_event_v1"

    return enhanced_data