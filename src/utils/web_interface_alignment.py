"""
Web Interface Alignment Utilities

This module provides utilities to ensure data flow consistency between
utils and web_interface modules, aligning messaging patterns and data formats.
"""

from datetime import datetime, timezone
from typing import Any

from src.core.exceptions import ValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)


def align_utils_to_web_patterns(data: dict[str, Any], operation_type: str) -> dict[str, Any]:
    """
    Align utils module data with web_interface patterns for consistency.

    Args:
        data: Data to align
        operation_type: Type of operation (websocket, api, stream, batch)

    Returns:
        Data aligned with web_interface patterns
    """
    aligned_data = data.copy()

    # Apply web_interface alignment patterns
    if operation_type in ["websocket", "stream", "real_time"]:
        # Keep stream processing but ensure web_interface compatibility
        aligned_data["processing_mode"] = "stream"
        aligned_data["message_pattern"] = "pub_sub"
        aligned_data["web_compatible"] = True
    elif operation_type in ["api", "request", "response"]:
        # Align with web_interface API patterns
        aligned_data["processing_mode"] = "request_reply"
        aligned_data["message_pattern"] = "req_reply"
        aligned_data["web_compatible"] = True
    elif operation_type == "batch":
        # Keep batch but add web compatibility
        aligned_data["processing_mode"] = "batch"
        aligned_data["message_pattern"] = "batch"
        aligned_data["web_compatible"] = True

    # Ensure web_interface compatible data format
    if "data_format" not in aligned_data:
        aligned_data["data_format"] = "api_event_v1"  # Web-compatible format

    # Add boundary crossing metadata
    aligned_data["boundary_crossed"] = True
    aligned_data["module_alignment"] = "utils_to_web_interface"
    aligned_data["alignment_timestamp"] = datetime.now(timezone.utc).isoformat()

    return aligned_data


def validate_utils_to_web_boundary(data: dict[str, Any]) -> None:
    """
    Validate data flowing from utils to web_interface modules.

    Args:
        data: Data to validate

    Raises:
        ValidationError: If validation fails
    """
    from src.utils.messaging_patterns import BoundaryValidator

    # Ensure web_interface compatible fields
    if "component" not in data:
        data["component"] = "utils_module"

    if "timestamp" not in data:
        data["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Apply web_interface boundary validation
    BoundaryValidator.validate_error_to_monitoring_boundary(data)


def validate_web_to_utils_boundary(data: dict[str, Any]) -> None:
    """
    Validate data flowing from web_interface to utils modules.

    Args:
        data: Data to validate

    Raises:
        ValidationError: If validation fails
    """
    from src.utils.messaging_patterns import BoundaryValidator

    # Ensure utils compatible fields
    required_fields = ["component", "timestamp"]
    for field in required_fields:
        if field not in data:
            raise ValidationError(
                f"Required field '{field}' missing in web_interface to utils boundary data",
                field_name=field,
                field_value=None,
                expected_type="string",
            )

    # Apply utils boundary validation
    BoundaryValidator.validate_monitoring_to_error_boundary(data)


class UtilsWebInterfaceDataBridge:
    """Bridge for consistent data flow between utils and web_interface modules."""

    @staticmethod
    def transform_utils_to_web(data: dict[str, Any], operation_type: str = "stream") -> dict[str, Any]:
        """Transform utils data for web_interface consumption."""
        # Apply alignment
        aligned_data = align_utils_to_web_patterns(data, operation_type)

        # Validate boundary
        validate_utils_to_web_boundary(aligned_data)

        return aligned_data

    @staticmethod
    def transform_web_to_utils(data: dict[str, Any], operation_type: str = "stream") -> dict[str, Any]:
        """Transform web_interface data for utils consumption."""
        aligned_data = data.copy()

        # Apply utils defaults
        if "processing_mode" not in aligned_data:
            aligned_data["processing_mode"] = "stream"  # Utils default

        if "message_pattern" not in aligned_data:
            aligned_data["message_pattern"] = "pub_sub"  # Utils default

        if "data_format" not in aligned_data:
            aligned_data["data_format"] = "bot_event_v1"  # Utils standard

        # Add metadata
        aligned_data["boundary_crossed"] = True
        aligned_data["module_alignment"] = "web_interface_to_utils"

        # Validate boundary
        validate_web_to_utils_boundary(aligned_data)

        return aligned_data

    @staticmethod
    def ensure_financial_precision(data: dict[str, Any]) -> dict[str, Any]:
        """Ensure financial data maintains precision across module boundaries."""
        from src.utils.decimal_utils import to_decimal

        financial_fields = ["price", "quantity", "volume", "value", "amount"]
        for field in financial_fields:
            if field in data and data[field] is not None:
                try:
                    data[field] = to_decimal(data[field])
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to convert {field} to Decimal: {e}")

        return data


def get_aligned_error_context(source_module: str, target_module: str, operation: str) -> dict[str, Any]:
    """
    Get aligned error context for consistent error propagation.

    Args:
        source_module: Source module name
        target_module: Target module name
        operation: Operation being performed

    Returns:
        Aligned error context
    """
    return {
        "source_module": source_module,
        "target_module": target_module,
        "operation": operation,
        "boundary_type": f"{source_module}_to_{target_module}",
        "alignment_applied": True,
        "error_timestamp": datetime.now(timezone.utc).isoformat(),
        "processing_mode": "stream" if source_module == "utils" else "request_reply",
        "message_pattern": "pub_sub" if source_module == "utils" else "req_reply",
    }