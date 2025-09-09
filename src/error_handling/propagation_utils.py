"""
Error propagation utilities for consistent data flow between error_handling and core modules.

This module provides utilities to ensure error propagation follows consistent patterns
across module boundaries, aligning with both pub/sub messaging and stream processing paradigms.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.core.exceptions import ValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)


class PropagationMethod(Enum):
    """Error propagation methods aligned with core patterns."""

    STRUCTURED_LOGGING = "structured_logging"
    EVENT_EMISSION = "event_emission"
    MESSAGE_QUEUE = "message_queue"
    DIRECT_CALL = "direct_call"
    STREAM_PROCESSING = "stream_processing"


class ProcessingStage(Enum):
    """Processing stages for error flow tracking."""

    ERROR_DETECTION = "error_detection"
    ERROR_VALIDATION = "error_validation"
    ERROR_HANDLING = "error_handling"
    ERROR_LOGGING = "error_logging"
    ERROR_RECOVERY = "error_recovery"
    ERROR_ESCALATION = "error_escalation"


def create_propagation_metadata(
    source_module: str,
    target_modules: list[str],
    method: PropagationMethod,
    stage: ProcessingStage,
    correlation_id: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Create consistent error propagation metadata.

    Args:
        source_module: Module where error originated
        target_modules: Modules that will receive the error
        method: Propagation method being used
        stage: Current processing stage
        correlation_id: Optional correlation ID for tracking
        **kwargs: Additional metadata

    Returns:
        Standardized propagation metadata dictionary
    """
    metadata = {
        "source_module": source_module,
        "target_modules": target_modules,
        "propagation_method": method.value,
        "processing_stage": stage.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "boundary_crossed": source_module != target_modules[0] if target_modules else False,
        "data_format": "error_propagation_v1",
        **kwargs,
    }

    if correlation_id:
        metadata["correlation_id"] = correlation_id

    return metadata


def validate_propagation_data(
    data: dict[str, Any], source_module: str, target_module: str
) -> dict[str, Any]:
    """
    Validate error data at propagation boundaries.

    Args:
        data: Error data to validate
        source_module: Source module name
        target_module: Target module name

    Returns:
        Validated error data

    Raises:
        ValidationError: If validation fails
    """
    # Basic validation
    if not isinstance(data, dict):
        raise ValidationError(
            "Error propagation data must be dictionary",
            error_code="PROP_001",
            field_name="propagation_data",
        )

    # Required fields validation
    required_fields = ["error_type", "timestamp"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValidationError(
            f"Missing required fields for error propagation: {missing_fields}",
            error_code="PROP_002",
            field_name="required_fields",
        )

    # Module-specific validations
    validated_data = data.copy()

    if target_module == "core":
        # Core module expects specific format
        if "data_format" not in validated_data:
            validated_data["data_format"] = "error_context_v1"
        if "processing_stage" not in validated_data:
            validated_data["processing_stage"] = "error_propagation"

    elif target_module == "database":
        # Database requires entity tracking
        if "entity_id" in validated_data:
            if not isinstance(validated_data["entity_id"], str | int):
                raise ValidationError(
                    "Database entity_id must be string or int",
                    error_code="PROP_003",
                    field_name="entity_id",
                )

    elif target_module == "monitoring":
        # Monitoring needs metrics metadata
        if "severity" not in validated_data:
            validated_data["severity"] = "medium"
        if "metric_type" not in validated_data:
            validated_data["metric_type"] = "error_count"

    # Add propagation metadata
    validated_data["propagation_metadata"] = create_propagation_metadata(
        source_module=source_module,
        target_modules=[target_module],
        method=PropagationMethod.DIRECT_CALL,
        stage=ProcessingStage.ERROR_VALIDATION,
        correlation_id=validated_data.get("correlation_id"),
    )

    return validated_data


def transform_error_for_module(
    error_data: dict[str, Any], target_module: str, processing_mode: str = "stream"
) -> dict[str, Any]:
    """
    Transform error data for specific target module compatibility.

    Args:
        error_data: Raw error data
        target_module: Target module name
        processing_mode: Processing mode (stream/batch/sync)

    Returns:
        Transformed error data for target module
    """
    transformed = error_data.copy()

    # Add processing mode metadata
    transformed.update(
        {
            "processing_mode": processing_mode,
            "target_module": target_module,
            "transformation_timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    # Module-specific transformations
    if target_module == "core":
        # Core expects event-style format
        transformed = _transform_for_core(transformed)
    elif target_module == "database":
        # Database expects entity-relationship format
        transformed = _transform_for_database(transformed)
    elif target_module == "monitoring":
        # Monitoring expects metrics format
        transformed = _transform_for_monitoring(transformed)
    elif target_module == "exchanges":
        # Exchanges expect financial data format
        transformed = _transform_for_exchanges(transformed)

    return transformed


def _transform_for_core(data: dict[str, Any]) -> dict[str, Any]:
    """Transform error data for core module compatibility with consistent patterns."""
    core_format = {
        "event_type": "error_occurred",
        "event_data": data,
        "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "data_format": "event_data_v1",  # Align with state module format
        "processing_stage": "core_event_processing",
        "processing_mode": "stream",  # Consistent processing mode
        "message_pattern": "pub_sub",  # Consistent messaging pattern
        "boundary_crossed": True,  # Cross-module event
    }

    # Preserve financial data precision
    financial_fields = ["price", "quantity", "amount", "balance"]
    for field in financial_fields:
        if field in data and data[field] is not None:
            try:
                from src.utils.decimal_utils import to_decimal

                core_format["event_data"][field] = to_decimal(data[field])
            except Exception as e:
                logger.warning(f"Failed to convert {field} to Decimal: {e}")

    return core_format


def _transform_for_database(data: dict[str, Any]) -> dict[str, Any]:
    """Transform error data for database module compatibility."""
    db_format: dict[str, Any] = {
        "table_context": "error_logs",
        "entity_data": data,
        "operation_type": "insert",
        "data_format": "database_entity_v1",
    }

    # Add database-specific fields
    if "error_id" not in db_format["entity_data"]:
        import uuid

        db_format["entity_data"]["error_id"] = str(uuid.uuid4())

    return db_format


def _transform_for_monitoring(data: dict[str, Any]) -> dict[str, Any]:
    """Transform error data for monitoring module compatibility with consistent patterns."""
    monitoring_format = {
        "metric_name": "error_occurred",
        "metric_value": 1,
        "metric_type": "counter",
        "tags": {
            "error_type": data.get("error_type", "unknown"),
            "component": data.get("component", "unknown"),
            "severity": data.get("severity", "medium"),
        },
        "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "data_format": "monitoring_metric_v1",
        "processing_mode": "stream",  # Consistent processing mode
        "message_pattern": "pub_sub",  # Consistent messaging pattern
        "boundary_crossed": True,  # Cross-module event
    }

    return monitoring_format


def _transform_for_exchanges(data: dict[str, Any]) -> dict[str, Any]:
    """Transform error data for exchanges module compatibility."""
    exchange_format: dict[str, Any] = {
        "exchange_error": True,
        "error_context": data,
        "data_format": "exchange_error_v1",
    }

    # Ensure financial precision for exchange errors
    financial_fields = ["price", "quantity", "amount", "balance", "fee"]
    for field in financial_fields:
        if field in data and data[field] is not None:
            try:
                from src.utils.decimal_utils import to_decimal

                exchange_format["error_context"][field] = to_decimal(data[field])
            except Exception as e:
                logger.warning(f"Failed to convert {field} to Decimal for exchange: {e}")

    return exchange_format


def get_propagation_chain(error_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract propagation chain from error data.

    Args:
        error_data: Error data containing propagation history

    Returns:
        List of propagation steps
    """
    return error_data.get("propagation_chain", [])


def add_propagation_step(
    error_data: dict[str, Any],
    source_module: str,
    target_module: str,
    method: PropagationMethod,
    stage: ProcessingStage,
) -> dict[str, Any]:
    """
    Add propagation step to error data chain.

    Args:
        error_data: Error data to update
        source_module: Source module
        target_module: Target module
        method: Propagation method
        stage: Processing stage

    Returns:
        Updated error data with propagation step
    """
    if "propagation_chain" not in error_data:
        error_data["propagation_chain"] = []

    step = {
        "step_id": len(error_data["propagation_chain"]) + 1,
        "source_module": source_module,
        "target_module": target_module,
        "method": method.value,
        "stage": stage.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    error_data["propagation_chain"].append(step)
    return error_data


__all__ = [
    "ProcessingStage",
    "PropagationMethod",
    "add_propagation_step",
    "create_propagation_metadata",
    "get_propagation_chain",
    "transform_error_for_module",
    "validate_propagation_data",
]
