"""
State Data Transformation Utilities.

Provides consistent data transformation patterns aligned with risk_management module
to ensure proper cross-module communication and data flow consistency.
"""

from datetime import datetime, timezone
from typing import Any

from src.core.logging import get_logger
from src.core.types import StateType
from src.utils.decimal_utils import to_decimal
from src.utils.messaging_patterns import MessagePattern

logger = get_logger(__name__)


class StateDataTransformer:
    """Handles consistent data transformation for state module."""

    @staticmethod
    def transform_state_change_to_event_data(
        state_type: StateType,
        state_id: str,
        state_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform state change to consistent event data format aligned with risk_management patterns.

        Args:
            state_type: Type of state being changed
            state_id: ID of the state
            state_data: State data to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format matching core event system
        """
        return {
            "state_type": state_type.value,
            "state_id": state_id,
            "state_data": state_data or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "state_management",
            "processing_mode": "stream",  # Align with core events default
            "data_format": "bot_event_v1",  # Consistent with core events
            "message_pattern": MessagePattern.PUB_SUB.value,  # Use enum for consistency
            "boundary_crossed": True,  # Mark cross-module communication
            "metadata": metadata or {},
        }

    @staticmethod
    def transform_state_snapshot_to_event_data(
        snapshot_data: dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform state snapshot to consistent event data format.

        Args:
            snapshot_data: Snapshot data to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        return {
            "snapshot_id": snapshot_data.get("snapshot_id", ""),
            "state_type": snapshot_data.get("state_type", "unknown"),
            "state_count": snapshot_data.get("state_count", 0),
            "created_at": snapshot_data.get("created_at", datetime.now(timezone.utc).isoformat()),
            "processing_mode": "stream",  # Align with core events default
            "data_format": "bot_event_v1",  # Consistent with core events
            "message_pattern": MessagePattern.PUB_SUB.value,  # Use enum for consistency
            "boundary_crossed": True,  # Mark cross-module communication
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

    @staticmethod
    def transform_error_to_event_data(
        error: Exception,
        context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Transform error to consistent event data format aligned with error_handling module.

        Args:
            error: Exception to transform
            context: Error context information
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        # Use error_handling module's transformer for consistency
        from src.error_handling.data_transformer import ErrorDataTransformer

        return ErrorDataTransformer.transform_error_to_event_data(error, context, metadata)

    @staticmethod
    def validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]:
        """
        Ensure financial data has proper precision using centralized financial utils.

        Args:
            data: Data dictionary to validate

        Returns:
            Dict with validated financial precision
        """
        from src.utils.financial_utils import validate_financial_precision

        # Define state-specific fields in addition to defaults
        state_fields = [
            "price", "quantity", "volume", "entry_price", "current_price",
            "stop_loss", "take_profit", "value", "amount",
            "unrealized_pnl", "realized_pnl"
        ]

        return validate_financial_precision(data, state_fields)

    @staticmethod
    def ensure_boundary_fields(
        data: dict[str, Any], source: str = "state_management"
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
        if isinstance(data, dict) and "state_type" in data:
            transformed = cls.transform_state_change_to_event_data(
                StateType(data.get("state_type", "unknown")),
                data.get("state_id", ""),
                data,
                metadata
            )
        elif isinstance(data, dict) and "snapshot_id" in data:
            transformed = cls.transform_state_snapshot_to_event_data(data, metadata)
        elif isinstance(data, Exception):
            transformed = cls.transform_error_to_event_data(data, metadata)
        elif isinstance(data, dict):
            transformed = data.copy()
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
                "message_pattern": MessagePattern.PUB_SUB.value,  # Use enum for consistency
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
        transformed["message_pattern"] = MessagePattern.REQ_REPLY.value  # Override processing mode

        return transformed

    @classmethod
    def align_processing_paradigm(
        cls, data: dict[str, Any], target_mode: str = "stream"
    ) -> dict[str, Any]:
        """
        Align data processing paradigm with risk_management module expectations using consistent patterns.

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

        # Add mode-specific fields with enhanced boundary metadata (aligned with core events)
        if target_mode == "stream":
            aligned_data.update(
                {
                    "stream_processing": True,  # Align with core events stream processing
                    "stream_position": datetime.now(timezone.utc).timestamp(),
                    "processing_paradigm": "stream",  # Match core events paradigm field
                    "data_format": "bot_event_v1",  # Align with core events
                    "message_pattern": MessagePattern.PUB_SUB.value,  # Use enum for consistency
                    "boundary_crossed": True,
                }
            )
        elif target_mode == "batch":
            if "batch_id" not in aligned_data:
                aligned_data["batch_id"] = datetime.now(timezone.utc).isoformat()
            aligned_data.update(
                {
                    "batch_processing": True,  # Align with core events batch processing
                    "processing_paradigm": "batch",  # Match core events paradigm field
                    "data_format": "bot_event_v1",  # Align with core events
                    "message_pattern": MessagePattern.BATCH.value,  # Batch messaging pattern
                    "boundary_crossed": True,
                }
            )
        elif target_mode == "request_reply":
            if "correlation_id" not in aligned_data:
                aligned_data["correlation_id"] = datetime.now(timezone.utc).isoformat()
            aligned_data.update(
                {
                    "data_format": "bot_event_v1",  # Align with core events
                    "message_pattern": MessagePattern.REQ_REPLY.value,  # Request-reply messaging pattern
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
        source_module: str = "state_management",
        target_module: str = "risk_management",
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
        target_mode = "stream" if target_module == "risk_management" else source_mode

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
            if target_module == "core" or target_module == "risk_management" or target_module == "execution":
                # Validate at state -> core/risk_management/execution boundary
                boundary_data = {
                    "component": validated_data.get("component", source_module),
                    "operation": validated_data.get("operation", "state_operation"),
                    "timestamp": validated_data.get(
                        "timestamp", datetime.now(timezone.utc).isoformat()
                    ),
                    "processing_mode": validated_data.get("processing_mode", "stream"),
                    "data_format": validated_data.get(
                        "data_format", "bot_event_v1"
                    ),  # Align with core
                    "message_pattern": validated_data.get(
                        "message_pattern", MessagePattern.PUB_SUB.value
                    ),  # Core consistency
                    "boundary_crossed": True,
                    "processing_paradigm": validated_data.get(
                        "processing_paradigm", "stream"
                    ),  # Core alignment
                }

                # Apply appropriate boundary validation based on target
                if target_module == "core":
                    # Core module expects event-like data with specific patterns
                    BoundaryValidator.validate_monitoring_to_error_boundary(boundary_data)
                elif target_module == "risk_management":
                    BoundaryValidator.validate_state_to_risk_boundary(boundary_data)
                else:
                    BoundaryValidator.validate_database_entity(boundary_data, "validate")

        except Exception as e:
            # Log validation issues but don't fail the data flow
            logger.warning(
                f"Cross-module validation failed for {source_module} -> {target_module}: {e}",
                extra={
                    "source_module": source_module,
                    "target_module": target_module,
                    "validation_error": str(e),
                    "data_format": validated_data.get("data_format", "unknown"),
                },
            )

        return validated_data
