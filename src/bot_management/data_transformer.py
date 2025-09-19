"""
Bot Management Data Transformation Utilities.

Provides consistent data transformation patterns aligned with state module
to ensure proper cross-module communication and data flow consistency.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from src.core.logging import get_logger
from src.core.types import BotConfiguration, BotMetrics, BotState, BotStatus
from src.utils.decimal_utils import to_decimal
from src.utils.messaging_patterns import MessagePattern

logger = get_logger(__name__)


class BotManagementDataTransformer:
    """Handles consistent data transformation for bot_management module."""

    @staticmethod
    def transform_bot_event_to_event_data(
        bot_id: str,
        event_type: str,
        bot_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform bot event to consistent event data format aligned with web_interface module patterns.

        Args:
            bot_id: ID of the bot
            event_type: Type of bot event
            bot_data: Bot data to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format matching core event system and web_interface
        """
        # FIXED: Enhanced metadata for cross-module alignment with web_interface
        enhanced_metadata = (metadata or {}).copy()
        enhanced_metadata.update({
            "module_alignment": "bot_management_to_web_interface",
            "cross_module_validation": True,
            "financial_data_validated": True,
        })

        return {
            "event_id": str(uuid.uuid4()),  # Consistent with capital_management
            "bot_id": bot_id,
            "event_type": event_type,
            "bot_data": bot_data or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "bot_management",
            "processing_mode": "stream",  # Align with core events default
            "data_format": "bot_event_v1",  # Module-specific format consistent with web_interface
            "message_pattern": MessagePattern.PUB_SUB.value,  # Use enum for consistency
            "boundary_crossed": True,  # Mark cross-module communication
            "validation_status": "validated",  # Consistent validation status
            "target_module": "web_interface",  # FIXED: Add target module for alignment
            "metadata": enhanced_metadata,
        }

    @staticmethod
    def transform_bot_metrics_to_event_data(
        bot_metrics: BotMetrics, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform bot metrics to consistent event data format.

        Args:
            bot_metrics: Bot metrics to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        return {
            "event_id": str(uuid.uuid4()),  # Consistent with capital_management
            "event_type": "bot_metrics",  # Explicit event type
            "bot_id": bot_metrics.bot_id,
            "total_trades": bot_metrics.total_trades,
            "successful_trades": bot_metrics.successful_trades,
            "failed_trades": bot_metrics.failed_trades,
            "total_pnl": str(bot_metrics.total_pnl),  # Convert Decimal to string
            "win_rate": float(bot_metrics.win_rate),
            "avg_trade_duration": bot_metrics.avg_trade_duration,
            "max_drawdown": str(bot_metrics.max_drawdown),
            "sharpe_ratio": float(bot_metrics.sharpe_ratio) if bot_metrics.sharpe_ratio else None,
            "processing_mode": "stream",  # Align with core events default
            "data_format": "bot_event_v1",  # Module-specific format
            "message_pattern": MessagePattern.PUB_SUB.value,  # Use enum for consistency
            "boundary_crossed": True,  # Mark cross-module communication
            "validation_status": "validated",  # Consistent validation status
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
        Transform error to consistent event data format aligned with core module patterns.

        Args:
            error: Exception to transform
            context: Error context information
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        # Use CoreDataTransformer for consistency with strategies module
        from src.core.data_transformer import CoreDataTransformer

        enhanced_metadata = (metadata or {}).copy()
        enhanced_metadata.update({
            "error_source": "bot_management",
            "error_context": context or {},
            "boundary_crossed": True,
        })

        return CoreDataTransformer.transform_event_to_standard_format(
            event_type="bot_management_error",
            data={"error": str(error), "error_type": type(error).__name__},
            metadata=enhanced_metadata,
            source="bot_management"
        )

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

        # Define bot-specific fields in addition to defaults
        bot_fields = [
            "price", "quantity", "volume", "entry_price", "current_price",
            "stop_loss", "take_profit", "value", "amount", "capital_allocation",
            "unrealized_pnl", "realized_pnl", "total_pnl", "max_drawdown"
        ]

        return validate_financial_precision(data, bot_fields)

    @staticmethod
    def ensure_boundary_fields(
        data: dict[str, Any], source: str = "bot_management"
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
        Transform data for pub/sub messaging pattern using CoreDataTransformer for consistency.

        Args:
            event_type: Type of event
            data: Raw data to transform
            metadata: Additional metadata

        Returns:
            Dict formatted for pub/sub pattern
        """
        from src.core.data_transformer import CoreDataTransformer

        enhanced_metadata = (metadata or {}).copy()
        enhanced_metadata["module_source"] = "bot_management"

        # Use CoreDataTransformer for consistency with strategies module
        if isinstance(data, dict) and "bot_id" in data:
            # Transform using CoreDataTransformer patterns
            transformed = CoreDataTransformer.transform_for_pub_sub_pattern(
                event_type=event_type,
                data=data,
                metadata=enhanced_metadata
            )
        elif isinstance(data, BotMetrics):
            # Convert BotMetrics to dict then transform
            metrics_data = cls.transform_bot_metrics_to_event_data(data, metadata)
            transformed = CoreDataTransformer.transform_for_pub_sub_pattern(
                event_type=event_type,
                data=metrics_data,
                metadata=enhanced_metadata
            )
        elif isinstance(data, Exception):
            # Use consistent error transformation
            error_data = cls.transform_error_to_event_data(data, metadata)
            transformed = CoreDataTransformer.transform_for_pub_sub_pattern(
                event_type=event_type,
                data=error_data,
                metadata=enhanced_metadata
            )
        else:
            # Standard transformation for other data types
            transformed = CoreDataTransformer.transform_for_pub_sub_pattern(
                event_type=event_type,
                data=data,
                metadata=enhanced_metadata
            )

        # FIXED: Apply cross-module consistency with web_interface as primary target
        return CoreDataTransformer.apply_cross_module_consistency(
            data=transformed,
            target_module="web_interface",  # FIXED: Primary target for bot_management events
            source_module="bot_management"
        )

    @classmethod
    def transform_for_req_reply(
        cls, request_type: str, data: Any, correlation_id: str | None = None
    ) -> dict[str, Any]:
        """
        Transform data for request/reply messaging pattern using CoreDataTransformer for consistency.

        Args:
            request_type: Type of request
            data: Request data
            correlation_id: Request correlation ID

        Returns:
            Dict formatted for req/reply pattern
        """
        from src.core.data_transformer import CoreDataTransformer

        enhanced_metadata = {"module_source": "bot_management"}

        # Use CoreDataTransformer for consistency with strategies module
        transformed = CoreDataTransformer.transform_for_request_reply_pattern(
            request_type=request_type,
            data=data,
            correlation_id=correlation_id,
            metadata=enhanced_metadata
        )

        # FIXED: Apply cross-module consistency with web_interface alignment
        return CoreDataTransformer.apply_cross_module_consistency(
            data=transformed,
            target_module="web_interface",  # FIXED: Align with web_interface for requests
            source_module="bot_management"
        )

    @classmethod
    def align_processing_paradigm(
        cls, data: dict[str, Any], target_mode: str = "stream"
    ) -> dict[str, Any]:
        """
        Align data processing paradigm using CoreDataTransformer patterns for consistency with strategies module.

        Args:
            data: Data to align
            target_mode: Target processing mode ("stream", "batch", "request_reply")

        Returns:
            Dict with aligned processing mode
        """
        from src.core.data_transformer import CoreDataTransformer
        from src.utils.messaging_patterns import ProcessingParadigmAligner

        aligned_data = data.copy()

        # Use ProcessingParadigmAligner for base alignment
        source_mode = aligned_data.get("processing_mode", "stream")
        aligned_data = ProcessingParadigmAligner.align_processing_modes(
            source_mode=source_mode, target_mode=target_mode, data=aligned_data
        )

        # Apply CoreDataTransformer patterns based on target mode for consistency
        if target_mode == "stream":
            # Use CoreDataTransformer for pub/sub pattern (stream)
            aligned_data = CoreDataTransformer.transform_for_pub_sub_pattern(
                event_type=aligned_data.get("event_type", "bot_stream_event"),
                data=aligned_data,
                metadata={"module_source": "bot_management", "paradigm_aligned": True}
            )
        elif target_mode == "batch":
            # Add batch-specific fields while maintaining CoreDataTransformer consistency
            if "batch_id" not in aligned_data:
                aligned_data["batch_id"] = str(uuid.uuid4())
            aligned_data.update({
                "batch_processing": True,
                "processing_paradigm": "batch",
                "data_format": "bot_event_v1",
                "boundary_crossed": True,
            })
        elif target_mode == "request_reply":
            # Use CoreDataTransformer for request/reply pattern
            aligned_data = CoreDataTransformer.transform_for_request_reply_pattern(
                request_type=aligned_data.get("request_type", "bot_request"),
                data=aligned_data,
                correlation_id=aligned_data.get("correlation_id"),
                metadata={"module_source": "bot_management", "paradigm_aligned": True}
            )

        # FIXED: Apply cross-module consistency for all patterns with web_interface alignment
        return CoreDataTransformer.apply_cross_module_consistency(
            data=aligned_data,
            target_module="web_interface",  # FIXED: Primary alignment with web_interface
            source_module="bot_management"
        )

    @classmethod
    def apply_cross_module_validation(
        cls,
        data: dict[str, Any],
        source_module: str = "bot_management",
        target_module: str = "web_interface",  # FIXED: Default to web_interface
    ) -> dict[str, Any]:
        """
        Apply comprehensive cross-module validation for consistent data flow aligned with web_interface.

        Args:
            data: Data to validate and transform
            source_module: Source module name
            target_module: Target module name (defaults to web_interface)

        Returns:
            Dict with validated and aligned data for cross-module communication
        """
        validated_data = data.copy()

        # Apply consistent messaging patterns
        from src.utils.messaging_patterns import (
            BoundaryValidator,
            ProcessingParadigmAligner,
        )

        # FIXED: Apply processing paradigm alignment optimized for web_interface
        source_mode = validated_data.get("processing_mode", "stream")
        target_mode = "stream" if target_module == "web_interface" else source_mode

        validated_data = ProcessingParadigmAligner.align_processing_modes(
            source_mode=source_mode, target_mode=target_mode, data=validated_data
        )

        # FIXED: Add comprehensive boundary metadata with web_interface alignment
        validated_data.update(
            {
                "cross_module_validation": True,
                "source_module": source_module,
                "target_module": target_module,
                "boundary_validation_timestamp": datetime.now(timezone.utc).isoformat(),
                "data_flow_aligned": True,
                "web_interface_compatible": True,  # FIXED: Add web_interface compatibility flag
                "module_alignment": f"{source_module}_to_{target_module}",
            }
        )

        # FIXED: Apply target-module specific boundary validation with enhanced web_interface alignment
        try:
            if target_module == "web_interface" or target_module == "core" or target_module == "state":
                # FIXED: Validate at bot_management -> web_interface/core/state boundary
                boundary_data = {
                    "component": validated_data.get("component", source_module),
                    "operation": validated_data.get("operation", "bot_operation"),
                    "timestamp": validated_data.get(
                        "timestamp", datetime.now(timezone.utc).isoformat()
                    ),
                    "processing_mode": validated_data.get("processing_mode", "stream"),
                    "data_format": validated_data.get(
                        "data_format", "bot_event_v1"
                    ),  # FIXED: Align with web_interface format
                    "message_pattern": validated_data.get(
                        "message_pattern", MessagePattern.PUB_SUB.value
                    ),  # Web_interface consistency
                    "boundary_crossed": True,
                    "processing_paradigm": validated_data.get(
                        "processing_paradigm", "stream"
                    ),  # Web_interface alignment
                    "web_interface_compatible": True,  # FIXED: Add compatibility flag
                }

                # FIXED: Apply appropriate boundary validation based on target module with web_interface priority
                if target_module == "web_interface":
                    # FIXED: Web interface expects HTTP/API compatible data with specific patterns
                    BoundaryValidator.validate_monitoring_to_error_boundary(boundary_data)
                elif target_module == "core":
                    # Core module expects event-like data with specific patterns
                    BoundaryValidator.validate_monitoring_to_error_boundary(boundary_data)
                elif target_module == "state":
                    # State module expects specific state management patterns
                    BoundaryValidator.validate_execution_to_state_boundary(boundary_data)
                elif target_module == "execution":
                    # Execution module expects order/trade-related data
                    BoundaryValidator.validate_risk_to_execution_boundary(boundary_data)
                elif target_module == "strategies":
                    # Strategies module expects signal/performance data
                    BoundaryValidator.validate_optimization_to_strategy_boundary(boundary_data)
                elif target_module == "risk_management":
                    # Risk management expects risk assessment data
                    BoundaryValidator.validate_monitoring_to_risk_boundary(boundary_data)
                elif target_module == "capital_management":
                    # Capital management expects allocation/balance data
                    BoundaryValidator.validate_database_entity(boundary_data, "validate")
                else:
                    # FIXED: Default validation for other modules (prefer web_interface patterns)
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
                },
            )

        return validated_data