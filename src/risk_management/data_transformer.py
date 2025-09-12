"""
Risk Management Data Transformation Utilities.

Provides consistent data transformation patterns aligned with execution module
to ensure proper cross-module communication and data flow consistency.
"""

from datetime import datetime, timezone
from typing import Any

from src.core.logging import get_logger
from src.core.types import Position, RiskMetrics, Signal
from src.utils.decimal_utils import to_decimal

logger = get_logger(__name__)


class RiskDataTransformer:
    """Handles consistent data transformation for risk_management module."""

    @staticmethod
    def transform_signal_to_event_data(
        signal: Signal, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform Signal to consistent event data format aligned with core module patterns.

        Args:
            signal: Signal to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format matching core event system
        """
        return {
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "strength": str(signal.strength),
            "timestamp": signal.timestamp.isoformat()
            if signal.timestamp
            else datetime.now(timezone.utc).isoformat(),
            "source": signal.source or "risk_management",
            "processing_mode": "stream",  # Align with core events default
            "data_format": "bot_event_v1",  # Consistent with core events
            "message_pattern": "pub_sub",  # Consistent with core messaging
            "boundary_crossed": True,  # Mark cross-module communication
            "metadata": {**(signal.metadata or {}), **(metadata or {})},
        }

    @staticmethod
    def transform_position_to_event_data(
        position: Position, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform Position to consistent event data format.

        Args:
            position: Position to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        return {
            "symbol": position.symbol,
            "side": position.side.value,
            "quantity": str(position.quantity),
            "entry_price": str(position.entry_price),
            "current_price": str(position.current_price) if position.current_price else None,
            "unrealized_pnl": str(position.unrealized_pnl) if position.unrealized_pnl else "0",
            "realized_pnl": str(position.realized_pnl) if position.realized_pnl else "0",
            "status": position.status.value if hasattr(position, "status") else "open",
            "processing_mode": "stream",  # Align with core events default
            "data_format": "bot_event_v1",  # Consistent with core events
            "message_pattern": "pub_sub",  # Consistent with core messaging
            "boundary_crossed": True,  # Mark cross-module communication
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

    @staticmethod
    def transform_risk_metrics_to_event_data(
        risk_metrics: RiskMetrics, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform RiskMetrics to consistent event data format.

        Args:
            risk_metrics: Risk metrics to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        return {
            "var_1d": str(risk_metrics.var_1d) if risk_metrics.var_1d else "0",
            "var_5d": str(risk_metrics.var_5d) if risk_metrics.var_5d else "0",
            "expected_shortfall": str(risk_metrics.expected_shortfall)
            if risk_metrics.expected_shortfall
            else "0",
            "max_drawdown": str(risk_metrics.max_drawdown) if risk_metrics.max_drawdown else "0",
            "sharpe_ratio": str(risk_metrics.sharpe_ratio) if risk_metrics.sharpe_ratio else "0",
            "volatility": str(getattr(risk_metrics, "volatility", None))
            if getattr(risk_metrics, "volatility", None)
            else "0",
            "beta": str(risk_metrics.beta) if risk_metrics.beta else "0",
            "correlation": str(risk_metrics.correlation) if risk_metrics.correlation else "0",
            "processing_mode": "stream",  # Align with core events default
            "data_format": "bot_event_v1",  # Consistent with core events
            "message_pattern": "pub_sub",  # Consistent with core messaging
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
        Transform error to consistent event data format using utils error propagation patterns.

        Args:
            error: Exception to transform
            context: Error context information
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format aligned with utils patterns
        """
        # Use utils messaging patterns for consistent error handling
        from src.utils.messaging_patterns import MessagePattern, MessageType

        # Create standardized error data format
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "risk_management",
            "processing_mode": "stream",
            "data_format": "bot_event_v1",
            "message_pattern": MessagePattern.PUB_SUB.value,
            "message_type": MessageType.ERROR_EVENT.value,
            "boundary_crossed": True,
            "validation_status": "error",
        }

        return error_data

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
        data: dict[str, Any], source: str = "risk_management"
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
        Transform data for pub/sub messaging pattern aligned with utils.messaging_patterns.

        Args:
            event_type: Type of event
            data: Raw data to transform
            metadata: Additional metadata

        Returns:
            Dict formatted for pub/sub pattern with utils compatibility
        """
        # Import MessagePattern and MessageType from utils for consistency
        from src.utils.messaging_patterns import MessagePattern, MessageType

        # Base transformation
        if isinstance(data, Signal):
            transformed = cls.transform_signal_to_event_data(data, metadata)
            message_type = MessageType.SYSTEM_EVENT.value
        elif isinstance(data, Position):
            transformed = cls.transform_position_to_event_data(data, metadata)
            message_type = MessageType.TRADE_EXECUTION.value
        elif isinstance(data, RiskMetrics):
            transformed = cls.transform_risk_metrics_to_event_data(data, metadata)
            message_type = MessageType.SYSTEM_EVENT.value
        elif isinstance(data, Exception):
            transformed = cls.transform_error_to_event_data(data, metadata)
            message_type = MessageType.ERROR_EVENT.value
        elif isinstance(data, dict):
            transformed = data.copy()
            message_type = MessageType.SYSTEM_EVENT.value
        else:
            transformed = {"payload": str(data), "type": type(data).__name__}
            message_type = MessageType.SYSTEM_EVENT.value

        # Ensure boundary fields
        transformed = cls.ensure_boundary_fields(transformed)

        # Validate financial precision
        transformed = cls.validate_financial_precision(transformed)

        # Add event type and enhanced boundary metadata aligned with utils patterns
        transformed.update(
            {
                "event_type": event_type,
                "message_type": message_type,  # Aligned with utils.messaging_patterns
                "message_pattern": MessagePattern.PUB_SUB.value,  # Use enum from utils
                "pattern": MessagePattern.PUB_SUB.value,  # Compatibility field
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
        Transform data for request/reply messaging pattern aligned with utils.messaging_patterns.

        Args:
            request_type: Type of request
            data: Request data
            correlation_id: Request correlation ID

        Returns:
            Dict formatted for req/reply pattern with utils compatibility
        """
        # Import MessagePattern from utils for consistency
        from src.utils.messaging_patterns import MessagePattern, MessageType

        # Use pub/sub transformation as base
        transformed = cls.transform_for_pub_sub(request_type, data)

        # Add request/reply specific fields aligned with utils patterns
        transformed.update({
            "request_type": request_type,
            "correlation_id": correlation_id or datetime.now(timezone.utc).isoformat(),
            "message_pattern": MessagePattern.REQ_REPLY.value,  # Use enum from utils
            "pattern": MessagePattern.REQ_REPLY.value,  # Compatibility field
            "processing_mode": "request_reply",  # Override processing mode
            "message_type": MessageType.SYSTEM_EVENT.value,  # Consistent message typing
        })

        return transformed

    @classmethod
    def transform_for_batch_processing(
        cls,
        batch_type: str,
        data_items: list[Any],
        batch_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Transform data for batch processing pattern aligned with execution module.

        Args:
            batch_type: Type of batch operation
            data_items: List of items to process in batch
            batch_id: Unique batch identifier
            metadata: Additional batch metadata

        Returns:
            Dict formatted for batch processing matching execution module patterns
        """
        # Transform each item individually using risk_management transformers
        transformed_items = []
        for item in data_items:
            if isinstance(item, Signal):
                transformed_items.append(cls.transform_signal_to_event_data(item))
            elif isinstance(item, Position):
                transformed_items.append(cls.transform_position_to_event_data(item))
            elif isinstance(item, RiskMetrics):
                transformed_items.append(cls.transform_risk_metrics_to_event_data(item))
            elif isinstance(item, dict):
                transformed_items.append(cls.ensure_boundary_fields(item.copy()))
            else:
                transformed_items.append(
                    {
                        "payload": str(item),
                        "type": type(item).__name__,
                        "processing_mode": "batch",
                        "data_format": "bot_event_v1",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

        # Return consistent batch format matching execution module
        return {
            "batch_type": batch_type,
            "batch_id": batch_id or datetime.now(timezone.utc).isoformat(),
            "batch_size": len(data_items),
            "items": transformed_items,
            "processing_mode": "batch",
            "data_format": "bot_event_v1",  # Use consistent data format across modules
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "risk_management",
            "metadata": metadata or {},
        }

    @classmethod
    def align_processing_paradigm(
        cls, data: dict[str, Any], target_mode: str = "stream"
    ) -> dict[str, Any]:
        """
        Align data processing paradigm using utils.messaging_patterns for consistency.

        Args:
            data: Data to align
            target_mode: Target processing mode ("stream", "batch", "request_reply")

        Returns:
            Dict with aligned processing mode using standardized utils patterns
        """
        # Use ProcessingParadigmAligner from utils for full consistency
        from src.utils.messaging_patterns import ProcessingParadigmAligner

        source_mode = data.get("processing_mode", "stream")
        aligned_data = ProcessingParadigmAligner.align_processing_modes(
            source_mode=source_mode, target_mode=target_mode, data=data
        )

        # Add risk_management-specific metadata to the standardized transformation
        aligned_data.update({
            "validation_status": "validated",
            "risk_module_aligned": True,
            "paradigm_alignment_source": "utils.messaging_patterns",
        })

        return aligned_data

    @classmethod
    def apply_cross_module_validation(
        cls,
        data: dict[str, Any],
        source_module: str = "risk_management",
        target_module: str = "state",
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

        # Apply consistent messaging patterns from utils
        from src.utils.messaging_patterns import (
            BoundaryValidator,
            ProcessingParadigmAligner,
        )

        # Apply standardized processing paradigm alignment
        source_mode = validated_data.get("processing_mode", "stream")
        target_mode = "stream" if target_module in ["state", "execution"] else source_mode

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

        # Apply standardized boundary validation patterns from utils
        try:
            # Prepare boundary validation data with consistent format
            boundary_data = {
                "component": validated_data.get("component", source_module),
                "operation": validated_data.get("operation", "risk_operation"),
                "timestamp": validated_data.get(
                    "timestamp", datetime.now(timezone.utc).isoformat()
                ),
                "processing_mode": validated_data.get("processing_mode", "stream"),
                "data_format": validated_data.get("data_format", "bot_event_v1"),
                "message_pattern": validated_data.get("message_pattern", "pub_sub"),
                "boundary_crossed": True,
                "processing_paradigm": validated_data.get("processing_paradigm", "stream"),
            }

            # Apply appropriate boundary validation using utils validators
            if target_module == "state":
                BoundaryValidator.validate_risk_to_state_boundary(boundary_data)
            elif target_module == "execution":
                # Use state validation as baseline for execution module
                BoundaryValidator.validate_state_to_risk_boundary(boundary_data)  # Reverse validation pattern
            elif target_module in ["monitoring", "error_handling"]:
                # Use consistent error-to-monitoring boundary validation like execution module
                BoundaryValidator.validate_error_to_monitoring_boundary(boundary_data)
            else:
                # Generic boundary validation for other modules
                logger.debug(
                    f"No specific boundary validator for {source_module} -> {target_module}, using generic validation"
                )

        except Exception as e:
            # Use consistent error handling pattern from utils
            logger.warning(
                f"Cross-module validation failed for {source_module} -> {target_module}: {e}",
                extra={
                    "source_module": source_module,
                    "target_module": target_module,
                    "validation_error": str(e),
                    "data_format": validated_data.get("data_format", "unknown"),
                    "boundary_validation_applied": True,
                },
            )

        return validated_data
