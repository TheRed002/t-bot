"""
Data transformation utilities for web interface aligned with core module patterns.

Provides consistent data transformation that aligns with core data transformer standards.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.data_transformer import CoreDataTransformer
from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.utils.decimal_utils import to_decimal

logger = get_logger(__name__)


def format_decimal(value: Any) -> str:
    """Convert any numeric value to string representation using core standards."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (int, float)):
        return str(to_decimal(value))
    return str(value)


def add_timestamp(data: dict[str, Any]) -> dict[str, Any]:
    """Add current timestamp to data using core standards."""
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    return data


def ensure_decimal_strings(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure all numeric values are string representations using core financial precision."""
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


def transform_for_web_response(
    data: Any,
    event_type: str = "web_response",
    processing_mode: str = "stream"  # Default to stream to align with strategies module
) -> dict[str, Any]:
    """
    Transform data for web response using core data transformer patterns aligned with strategies module.

    Args:
        data: Data to transform
        event_type: Type of web event
        processing_mode: Processing mode (defaults to stream for consistency with strategies module)

    Returns:
        Dict formatted according to core standards
    """
    # Use core data transformer for consistency
    transformed = CoreDataTransformer.transform_event_to_standard_format(
        event_type=event_type,
        data=data,
        source="web_interface"
    )

    # Align processing paradigm consistently - default to stream for real-time web interfaces
    # Override only for specific non-real-time operations
    if event_type.lower().startswith(("api_", "batch_")):
        processing_mode = "request_reply"  # Use request_reply only for API calls
    else:
        processing_mode = "stream"  # Default to stream for consistency with strategies module

    transformed = CoreDataTransformer.align_processing_paradigm(
        transformed, processing_mode
    )

    # Apply cross-module consistency with strategies module alignment
    transformed = CoreDataTransformer.apply_cross_module_consistency(
        transformed,
        target_module="strategies",  # Target strategies for alignment
        source_module="web_interface"
    )

    # Import MessagePattern enum for consistency
    from src.utils.messaging_patterns import MessagePattern

    # Ensure message pattern consistency with risk_management module
    if processing_mode == "stream":
        transformed["message_pattern"] = MessagePattern.PUB_SUB.value  # Align with risk_management patterns
    elif processing_mode == "request_reply":
        transformed["message_pattern"] = MessagePattern.REQ_REPLY.value  # Align with risk_management module pattern

    # Use consistent data format version with risk_management module
    transformed["data_format"] = "bot_event_v1"  # Align with risk_management format

    return transformed


def transform_for_api_response(data: Any, endpoint: str) -> dict[str, Any]:
    """
    Transform data for API response with consistent formatting aligned with strategies module.

    Args:
        data: Response data
        endpoint: API endpoint name

    Returns:
        Dict formatted for API response
    """
    # Transform using request/reply pattern for API responses aligned with strategies module
    response_data = transform_for_web_response(
        data=data,
        event_type=f"api_{endpoint}_response",
        processing_mode="request_reply"  # Keep request_reply for API endpoints
    )

    # Ensure financial precision for API responses using strategies module standards
    if isinstance(data, dict):
        response_data["data"] = ensure_decimal_strings(data)

    # Add boundary crossing metadata for consistency with risk_management module
    response_data["boundary_crossed"] = True
    response_data["module_alignment"] = "web_interface_to_risk_management"
    response_data["data_format"] = "bot_event_v1"  # Consistent with risk_management module

    return response_data


def align_with_risk_management_patterns(data: dict[str, Any], operation_type: str) -> dict[str, Any]:
    """
    Align web_interface data with risk_management module patterns for consistency.

    Args:
        data: Data to align
        operation_type: Type of operation (websocket, api, stream, batch)

    Returns:
        Data aligned with risk_management module patterns
    """
    aligned_data = data.copy()

    # Import MessagePattern enum for consistency
    from src.utils.messaging_patterns import MessagePattern

    # Apply risk_management module default patterns
    if operation_type in ["websocket", "stream", "real_time", "monitoring"]:
        aligned_data["processing_mode"] = "stream"  # Align with risk_management defaults
        aligned_data["message_pattern"] = MessagePattern.PUB_SUB.value  # Align with risk_management patterns
    elif operation_type in ["api", "request", "response"]:
        aligned_data["processing_mode"] = "request_reply"  # Keep for API calls
        aligned_data["message_pattern"] = MessagePattern.REQ_REPLY.value  # Align with risk_management module pattern
    elif operation_type == "batch":
        aligned_data["processing_mode"] = "batch"  # Keep batch operations
        aligned_data["message_pattern"] = MessagePattern.BATCH.value  # Maintain batch patterns

    # Ensure consistent data format versioning with risk_management module
    if "data_format" not in aligned_data:
        aligned_data["data_format"] = "bot_event_v1"  # Align with risk_management standards

    # Add boundary crossing metadata for consistency with risk_management module
    aligned_data["boundary_crossed"] = True
    aligned_data["module_alignment"] = "web_interface_to_risk_management"
    aligned_data["cross_module_validation"] = True  # Match risk_management patterns

    return aligned_data


def validate_web_to_risk_management_boundary(data: dict[str, Any]) -> None:
    """
    Validate data flowing from web_interface to risk_management modules.

    Args:
        data: Data to validate

    Raises:
        ValidationError: If validation fails
    """
    from src.utils.messaging_patterns import BoundaryValidator

    # Ensure required boundary fields aligned with risk_management module expectations
    if "processing_mode" not in data:
        data["processing_mode"] = "stream"  # Default to risk_management standard

    # Import MessagePattern enum for consistency
    from src.utils.messaging_patterns import MessagePattern

    if "message_pattern" not in data:
        if data["processing_mode"] == "stream":
            data["message_pattern"] = MessagePattern.PUB_SUB.value  # Align with risk_management patterns
        else:
            data["message_pattern"] = MessagePattern.REQ_REPLY.value  # Align with risk_management module pattern

    if "data_format" not in data:
        data["data_format"] = "bot_event_v1"  # Align with risk_management standards

    # Add required fields for risk_management boundary validation
    if "source" not in data:
        data["source"] = "web_interface"
    if "target_module" not in data:
        data["target_module"] = "risk_management"
    if "component" not in data:
        data["component"] = "web_interface"

    # Apply risk_management compatible boundary validation
    BoundaryValidator.validate_web_interface_to_risk_management_boundary(data)


def validate_bot_management_to_web_boundary(data: dict[str, Any]) -> None:
    """
    FIXED: Validate data flowing from bot_management to web_interface modules.

    Args:
        data: Data to validate

    Raises:
        ValidationError: If validation fails
    """
    from src.utils.messaging_patterns import BoundaryValidator

    # FIXED: Ensure web interface compatible fields from bot_management
    if "component" not in data:
        data["component"] = "bot_management_to_web_interface"

    if "timestamp" not in data:
        data["timestamp"] = datetime.now(timezone.utc).isoformat()

    # FIXED: Ensure consistent data format from bot_management
    if "data_format" not in data:
        data["data_format"] = "bot_event_v1"

    if "source" not in data:
        data["source"] = "bot_management"

    # FIXED: Apply boundary validation for bot_management to web interface
    BoundaryValidator.validate_error_to_monitoring_boundary(data)


def align_error_propagation_patterns() -> None:
    """
    Ensure consistent error propagation patterns between web_interface and risk_management modules.

    This function establishes shared error handling patterns for cross-module communication.
    """
    # This function serves as documentation for error propagation alignment
    # Both modules should use:
    # 1. WebInterfaceErrorPropagator.propagate_to_risk_management() for web->risk_management errors
    # 2. RiskDataTransformer.transform_error_to_event_data() for risk_management->web errors
    # 3. Consistent data_format: "bot_event_v1"
    # 4. Consistent processing_mode: "stream" (default)
    # 5. Consistent message_pattern: "pub_sub" (for streaming errors)
    pass


class WebInterfaceErrorPropagator:
    """Error propagation patterns aligned with risk_management module."""

    @staticmethod
    def propagate_to_risk_management(error: Exception, context: str) -> None:
        """Propagate errors from web_interface to risk_management modules consistently."""
        from src.utils.messaging_patterns import ErrorPropagationMixin

        error_handler = ErrorPropagationMixin()

        # Apply consistent error propagation pattern aligned with risk_management
        error_handler.propagate_service_error(error, f"web_interface_to_risk_management:{context}")

    @staticmethod
    def propagate_validation_error(error: Exception, context: str) -> None:
        """Propagate validation errors with web_interface context aligned with execution."""
        from src.utils.messaging_patterns import ErrorPropagationMixin

        error_handler = ErrorPropagationMixin()
        error_handler.propagate_validation_error(error, f"web_interface:{context}")

    @staticmethod
    def propagate_to_monitoring(error: Exception, context: str) -> None:
        """Propagate errors to monitoring with consistent patterns aligned with strategies."""
        from src.utils.messaging_patterns import ErrorPropagationMixin

        error_handler = ErrorPropagationMixin()
        error_handler.propagate_monitoring_error(error, f"web_interface:{context}")


# Legacy compatibility class - aligned with core patterns
class WebInterfaceDataTransformer:
    """Data transformer for backward compatibility, aligned with core standards."""

    @staticmethod
    def format_portfolio_composition(data: Any) -> Any:
        """Format portfolio composition using core standards."""
        if isinstance(data, dict):
            return transform_for_web_response(
                data=data,
                event_type="portfolio_composition",
                processing_mode="request_reply"
            )
        return data

    @staticmethod
    def format_stress_test_results(data: Any) -> Any:
        """Format stress test results using core standards."""
        if isinstance(data, dict):
            return transform_for_web_response(
                data=data,
                event_type="stress_test_results",
                processing_mode="batch"  # Stress tests are batch operations
            )
        return data

    @staticmethod
    def format_operational_metrics(data: Any) -> Any:
        """Format operational metrics using core standards aligned with execution module."""
        if isinstance(data, dict):
            return transform_for_web_response(
                data=data,
                event_type="operational_metrics",
                processing_mode="stream"  # Align with execution module for real-time metrics
            )
        return data

    @staticmethod
    def transform_risk_data_to_event_data(data: Any, **kwargs) -> Any:
        """Transform risk data using core patterns aligned with risk_management module."""
        if isinstance(data, dict):
            # Align with risk_management patterns for consistent data flow
            aligned_data = align_with_risk_management_patterns(data, "stream")

            return CoreDataTransformer.apply_cross_module_consistency(
                data=transform_for_web_response(
                    data=aligned_data,
                    event_type="risk_data_event",
                    processing_mode="stream"
                ),
                target_module="risk_management",  # Align with risk_management module
                source_module="web_interface"
            )
        return data

    @staticmethod
    def validate_financial_precision(data: Any) -> Any:
        """Validate financial precision using core standards."""
        if isinstance(data, dict):
            # Use core financial precision validation
            validated = CoreDataTransformer._apply_financial_precision({
                "data": data,
                "source": "web_interface"
            })
            return validated.get("data", data)
        return data


def align_processing_paradigms_with_risk_management(
    data: dict[str, Any],
    operation_type: str,
    target_mode: str | None = None
) -> dict[str, Any]:
    """
    Align web_interface processing paradigms with risk_management module for consistent data flow.

    Args:
        data: Data to align
        operation_type: Type of operation (api, websocket, batch, stream)
        target_mode: Override target processing mode

    Returns:
        Data aligned with risk_management module processing paradigms
    """
    from src.utils.messaging_patterns import ProcessingParadigmAligner

    aligned_data = data.copy()

    # Determine source and target modes based on operation type and risk_management expectations
    if operation_type in ["api", "request", "response"]:
        source_mode = "request_reply"
        target_mode = target_mode or "stream"  # Risk management typically consume in stream mode
    elif operation_type in ["websocket", "real_time", "monitoring"]:
        source_mode = "stream"
        target_mode = target_mode or "stream"  # Perfect alignment
    elif operation_type == "batch":
        source_mode = "batch"
        target_mode = target_mode or "stream"  # Convert batch to stream for risk_management
    else:
        source_mode = "stream"  # Default
        target_mode = target_mode or "stream"

    # Apply paradigm alignment if modes differ
    if source_mode != target_mode:
        aligned_data = ProcessingParadigmAligner.align_processing_modes(
            source_mode, target_mode, aligned_data
        )

    # Add web_interface to risk_management specific metadata
    aligned_data.update({
        "web_interface_to_risk_management": True,
        "source_operation_type": operation_type,
        "paradigm_alignment_applied": source_mode != target_mode,
        "original_source_mode": source_mode,
        "final_target_mode": target_mode,
        "alignment_timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return aligned_data


def handle_batch_to_stream_conversion_for_risk_management(
    batch_data: list[dict[str, Any]],
    operation_context: str = "web_interface_batch"
) -> list[dict[str, Any]]:
    """
    Convert batch data to stream format optimized for risk_management module consumption.

    Args:
        batch_data: List of data items in batch format
        operation_context: Context for the batch operation

    Returns:
        List of stream-formatted data items for risk_management consumption
    """
    from src.utils.messaging_patterns import ProcessingParadigmAligner

    # Create batch structure for processing
    batch_structure = ProcessingParadigmAligner.create_batch_from_stream(batch_data)

    # Convert back to stream format optimized for strategies
    stream_items = ProcessingParadigmAligner.create_stream_from_batch(batch_structure)

    # Import MessagePattern for consistency
    from src.utils.messaging_patterns import MessagePattern

    # Apply risk_management-specific enhancements to each stream item
    for item in stream_items:
        item.update({
            "risk_management_optimized": True,
            "web_interface_source": True,
            "batch_to_stream_converted": True,
            "operation_context": operation_context,
            "processing_mode": "stream",
            "message_pattern": MessagePattern.PUB_SUB.value,
            "data_format": "bot_event_v1",
            "target_module": "risk_management",
        })

    return stream_items


def validate_paradigm_alignment_for_risk_management(data: dict[str, Any]) -> None:
    """
    Validate that data paradigm alignment is correct for risk_management module consumption.

    Args:
        data: Data to validate

    Raises:
        ValidationError: If paradigm alignment is incorrect
    """
    # Check processing mode compatibility
    if "processing_mode" in data:
        valid_modes = ["stream", "request_reply", "batch"]
        if data["processing_mode"] not in valid_modes:
            raise ValidationError(
                f"Invalid processing mode for risk_management: {data['processing_mode']}",
                field_name="processing_mode",
                field_value=data["processing_mode"],
                expected_type=f"one of {valid_modes}"
            )

    # Check message pattern alignment
    if "message_pattern" in data:
        # Import MessagePattern enum for validation
        from src.utils.messaging_patterns import MessagePattern

        valid_patterns = [pattern.value for pattern in MessagePattern]
        if data["message_pattern"] not in valid_patterns:
            raise ValidationError(
                f"Invalid message pattern for risk_management: {data['message_pattern']}",
                field_name="message_pattern",
                field_value=data["message_pattern"],
                expected_type=f"one of {valid_patterns}"
            )

    # Ensure data format is risk_management-compatible
    if "data_format" in data:
        valid_formats = ["bot_event_v1", "signal_v1", "api_response_v1", "risk_event_v1"]
        if not any(data["data_format"].startswith(fmt.split("_")[0]) for fmt in valid_formats):
            raise ValidationError(
                f"Invalid data format for risk_management: {data['data_format']}",
                field_name="data_format",
                field_value=data["data_format"],
                expected_type="risk_management-compatible format"
            )

    # Validate financial precision for risk_management
    if "data" in data and isinstance(data["data"], dict):
        financial_fields = ["price", "quantity", "volume", "amount", "position_size", "var_1d", "expected_shortfall"]
        for field in financial_fields:
            if field in data["data"] and data["data"][field] is not None:
                try:
                    value = float(data["data"][field])
                    if value < 0:
                        raise ValidationError(
                            f"Financial field {field} cannot be negative for risk_management",
                            field_name=f"data.{field}",
                            field_value=value,
                            validation_rule="must be non-negative"
                        )
                except (ValueError, TypeError):
                    raise ValidationError(
                        f"Financial field {field} must be numeric for risk_management",
                        field_name=f"data.{field}",
                        field_value=data["data"][field],
                        expected_type="numeric"
                    )
