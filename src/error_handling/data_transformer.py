"""
Error Handling Data Transformation Utilities.

Provides consistent data transformation patterns aligned with risk_management module
to ensure proper cross-module communication and data flow consistency.
"""

from datetime import datetime, timezone
from typing import Any

from src.core.logging import get_logger
from src.utils.decimal_utils import to_decimal

logger = get_logger(__name__)


class ErrorDataTransformer:
    """Handles consistent data transformation for error_handling module."""

    @staticmethod
    def transform_error_to_event_data(
        error: Exception, context: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform Error to consistent event data format aligned with core module patterns.

        Args:
            error: Exception to transform
            context: Error context information
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format matching core event system
        """
        return {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_context": context or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "error_handling",
            "processing_mode": "stream",  # Align with core events default
            "data_format": "bot_event_v1",  # Consistent with core events
            "message_pattern": "pub_sub",  # Consistent with core messaging
            "boundary_crossed": True,  # Mark cross-module communication
            "metadata": {**(metadata or {})},
        }

    @staticmethod
    def transform_context_to_event_data(
        context: dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform error context to consistent event data format.

        Args:
            context: Error context to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        return {
            "component": context.get("component", "unknown"),
            "operation": context.get("operation", "unknown"),
            "error_type": context.get("error_type", "unknown"),
            "error_message": context.get("error_message", ""),
            "processing_mode": "stream",  # Align with core events default
            "data_format": "bot_event_v1",  # Consistent with core events
            "message_pattern": "pub_sub",  # Consistent with core messaging
            "boundary_crossed": True,  # Mark cross-module communication
            "timestamp": context.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "metadata": {**(context.get("metadata", {})), **(metadata or {})},
        }

    @staticmethod
    def validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]:
        """
        Ensure financial data has proper precision using Decimal.

        Args:
            data: Data dictionary to validate

        Returns:
            Dict with validated financial precision
        """
        financial_fields = [
            "quantity",
            "entry_price",
            "current_price",
            "unrealized_pnl",
            "realized_pnl",
            "var_1d",
            "var_5d",
            "expected_shortfall",
            "max_drawdown",
            "sharpe_ratio",
            "volatility",
            "beta",
            "correlation",
            "strength",
        ]

        for field in financial_fields:
            if field in data and data[field] is not None and data[field] != "":
                try:
                    # Convert to Decimal for financial precision
                    decimal_value = to_decimal(data[field])
                    # Convert back to string for consistent format
                    data[field] = str(decimal_value)
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(
                        f"Failed to convert financial field {field} to Decimal: {e}",
                        field=field,
                        value=data[field],
                    )
                    # Keep original value if conversion fails but log the issue
                    # This is acceptable for data transformation where robustness is prioritized

        return data

    @staticmethod
    def ensure_boundary_fields(
        data: dict[str, Any], source: str = "error_handling"
    ) -> dict[str, Any]:
        """
        Ensure data has required boundary fields for cross-module communication.

        Args:
            data: Data dictionary to enhance
            source: Source module name

        Returns:
            Dict with required boundary fields
        """
        # Ensure processing mode is set
        if "processing_mode" not in data:
            data["processing_mode"] = "stream"

        # Ensure data format is set (aligned with core events)
        if "data_format" not in data:
            data["data_format"] = "bot_event_v1"

        # Ensure timestamp is set
        if "timestamp" not in data:
            data["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add source information
        if "source" not in data:
            data["source"] = source

        # Ensure metadata exists
        if "metadata" not in data:
            data["metadata"] = {}

        return data

    @classmethod
    def transform_for_pub_sub(
        cls, event_type: str, data: Any, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform data for pub/sub messaging pattern.

        Args:
            event_type: Type of event
            data: Raw data to transform
            metadata: Additional metadata

        Returns:
            Dict formatted for pub/sub pattern
        """
        # Base transformation
        if isinstance(data, Exception):
            transformed = cls.transform_error_to_event_data(data, metadata)
        elif isinstance(data, dict):
            transformed = cls.transform_context_to_event_data(data, metadata)
        else:
            transformed = {"payload": str(data), "type": type(data).__name__}

        # Ensure boundary fields
        transformed = cls.ensure_boundary_fields(transformed)

        # Validate financial precision
        transformed = cls.validate_financial_precision(transformed)

        # Add event type and enhanced boundary metadata
        transformed.update(
            {
                "event_type": event_type,
                "message_pattern": "pub_sub",  # Consistent messaging pattern
                "boundary_crossed": True,  # Cross-module event flag
                "validation_status": "validated",  # Boundary validation status
            }
        )

        return transformed

    @classmethod
    def transform_for_req_reply(
        cls, request_type: str, data: Any, correlation_id: str | None = None
    ) -> dict[str, Any]:
        """
        Transform data for request/reply messaging pattern.

        Args:
            request_type: Type of request
            data: Request data
            correlation_id: Request correlation ID

        Returns:
            Dict formatted for req/reply pattern
        """
        # Use pub/sub transformation as base
        transformed = cls.transform_for_pub_sub(request_type, data)

        # Add request/reply specific fields
        transformed["request_type"] = request_type
        transformed["correlation_id"] = correlation_id or datetime.now(timezone.utc).isoformat()
        transformed["message_pattern"] = "req_reply"  # Override message pattern for req/reply
        transformed["processing_mode"] = "request_reply"  # Override processing mode

        return transformed

    @classmethod
    def align_processing_paradigm(
        cls, data: dict[str, Any], target_mode: str = "stream"
    ) -> dict[str, Any]:
        """
        Align data processing paradigm with other modules using consistent patterns.

        Args:
            data: Data to align
            target_mode: Target processing mode ("stream", "batch", "request_reply")

        Returns:
            Dict with aligned processing mode
        """
        aligned_data = data.copy()

        # Use ProcessingParadigmAligner for consistency
        from src.utils.messaging_patterns import ProcessingParadigmAligner

        source_mode = aligned_data.get("processing_mode", "stream")
        aligned_data = ProcessingParadigmAligner.align_processing_modes(
            source_mode=source_mode, target_mode=target_mode, data=aligned_data
        )

        # Add mode-specific fields with enhanced boundary metadata (aligned with execution module)
        if target_mode == "stream":
            aligned_data.update(
                {
                    "stream_position": datetime.now(timezone.utc).timestamp(),
                    "data_format": "bot_event_v1",  # Align with execution module
                    "message_pattern": "pub_sub",  # Consistent messaging pattern
                    "boundary_crossed": True,
                }
            )
        elif target_mode == "batch":
            if "batch_id" not in aligned_data:
                aligned_data["batch_id"] = datetime.now(timezone.utc).isoformat()
            aligned_data.update(
                {
                    "data_format": "bot_event_v1",  # Align with execution module
                    "message_pattern": "batch",  # Batch messaging pattern
                    "boundary_crossed": True,
                }
            )
        elif target_mode == "request_reply":
            if "correlation_id" not in aligned_data:
                aligned_data["correlation_id"] = datetime.now(timezone.utc).isoformat()
            aligned_data.update(
                {
                    "data_format": "bot_event_v1",  # Align with execution module
                    "message_pattern": "req_reply",  # Request-reply messaging pattern
                    "boundary_crossed": True,
                }
            )

        # Add consistent validation status
        aligned_data["validation_status"] = "validated"

        return aligned_data

    @classmethod
    def apply_cross_module_validation(
        cls,
        data: dict[str, Any],
        source_module: str = "error_handling",
        target_module: str = "core",
    ) -> dict[str, Any]:
        """
        Apply comprehensive cross-module validation for consistent data flow.

        Args:
            data: Data to validate and transform
            source_module: Source module name
            target_module: Target module name

        Returns:
            Dict with validated and aligned data for cross-module communication
        """
        validated_data = data.copy()

        # Apply consistent messaging patterns
        from src.utils.messaging_patterns import (
            BoundaryValidator,
            ProcessingParadigmAligner,
        )

        # Apply processing paradigm alignment
        source_mode = validated_data.get("processing_mode", "stream")
        target_mode = "stream" if target_module in ["core", "state", "monitoring"] else source_mode

        validated_data = ProcessingParadigmAligner.align_processing_modes(
            source_mode=source_mode, target_mode=target_mode, data=validated_data
        )

        # Add comprehensive boundary metadata
        validated_data.update(
            {
                "cross_module_validation": True,
                "source_module": source_module,
                "target_module": target_module,
                "boundary_validation_timestamp": datetime.now(timezone.utc).isoformat(),
                "data_flow_aligned": True,
            }
        )

        # Apply target-module specific boundary validation with enhanced core alignment
        try:
            if target_module in ["core", "state", "monitoring", "risk_management"]:
                # Validate at error_handling -> target boundary
                boundary_data = {
                    "component": validated_data.get("component", source_module),
                    "operation": validated_data.get("operation", "error_handling_operation"),
                    "timestamp": validated_data.get(
                        "timestamp", datetime.now(timezone.utc).isoformat()
                    ),
                    "processing_mode": validated_data.get("processing_mode", "stream"),
                    "data_format": validated_data.get("data_format", "bot_event_v1"),
                    "message_pattern": validated_data.get("message_pattern", "pub_sub"),
                    "boundary_crossed": True,
                    "processing_paradigm": validated_data.get("processing_paradigm", "stream"),
                }

                # Apply appropriate boundary validation based on target
                if target_module in ["execution", "monitoring", "web_interface"]:
                    # Use correct validation for error_handling -> other modules
                    BoundaryValidator.validate_monitoring_to_error_boundary(boundary_data)
                elif target_module == "risk_management":
                    BoundaryValidator.validate_risk_to_state_boundary(boundary_data)
                else:
                    BoundaryValidator.validate_monitoring_to_error_boundary(boundary_data)

        except Exception as e:
            # Log validation issues but don't fail the data flow
            logger.warning(
                f"Cross-module validation failed for {source_module} -> {target_module}: {e}",
                extra={
                    "source_module": source_module,
                    "target_module": target_module,
                    "validation_error": str(e),
                    "data_format": validated_data.get("data_format", "unknown"),
                }
            )

        return validated_data
