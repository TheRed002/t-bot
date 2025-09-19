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
        from src.core.data_transformer import CoreDataTransformer

        # Use core data transformer for consistency with web_interface
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_context": context or {},
        }

        # Transform using core patterns like web_interface
        transformed = CoreDataTransformer.transform_event_to_standard_format(
            event_type="error_event",
            data=error_data,
            metadata=metadata,
            source="error_handling"
        )

        return transformed

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
        from src.core.data_transformer import CoreDataTransformer

        # Use core data transformer for consistency with web_interface
        context_data = {
            "component": context.get("component", "unknown"),
            "operation": context.get("operation", "unknown"),
            "error_type": context.get("error_type", "unknown"),
            "error_message": context.get("error_message", ""),
        }

        # Transform using core patterns like web_interface
        transformed = CoreDataTransformer.transform_event_to_standard_format(
            event_type="context_event",
            data=context_data,
            metadata={**(context.get("metadata", {})), **(metadata or {})},
            source="error_handling"
        )

        return transformed

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

        # Define error handling-specific fields (same as risk management)
        error_fields = [
            "quantity", "entry_price", "current_price", "unrealized_pnl",
            "realized_pnl", "var_1d", "var_5d", "expected_shortfall",
            "max_drawdown", "sharpe_ratio", "volatility", "beta",
            "correlation", "strength"
        ]

        return validate_financial_precision(data, error_fields)

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
        from src.core.data_transformer import CoreDataTransformer

        # Base transformation using core patterns like web_interface
        if isinstance(data, Exception):
            transformed = cls.transform_error_to_event_data(data, metadata=metadata)
        elif isinstance(data, dict):
            transformed = cls.transform_context_to_event_data(data, metadata)
        else:
            # Use core transformer for consistency
            transformed = CoreDataTransformer.transform_for_pub_sub_pattern(
                event_type=event_type,
                data=data,
                metadata=metadata
            )

        # Apply pub/sub specific formatting using core patterns
        if not isinstance(data, (Exception, dict)):
            return transformed

        # Apply consistent processing paradigm alignment
        transformed = CoreDataTransformer.align_processing_paradigm(transformed, "stream")

        # Apply cross-module consistency
        transformed = CoreDataTransformer.apply_cross_module_consistency(
            transformed,
            target_module="error_handling",
            source_module="error_handling"
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
        from src.core.data_transformer import CoreDataTransformer

        # Base transformation using core patterns like web_interface
        if isinstance(data, Exception):
            transformed = cls.transform_error_to_event_data(data)
        elif isinstance(data, dict):
            transformed = cls.transform_context_to_event_data(data)
        else:
            # Use core transformer for req/reply pattern consistency
            transformed = CoreDataTransformer.transform_for_request_reply_pattern(
                request_type=request_type,
                data=data,
                correlation_id=correlation_id
            )

        # Apply req/reply specific formatting using core patterns
        if not isinstance(data, (Exception, dict)):
            return transformed

        # Apply consistent processing paradigm alignment for request/reply
        transformed = CoreDataTransformer.align_processing_paradigm(transformed, "request_reply")

        # Apply cross-module consistency
        transformed = CoreDataTransformer.apply_cross_module_consistency(
            transformed,
            target_module="error_handling",
            source_module="error_handling"
        )

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
        from src.core.data_transformer import CoreDataTransformer

        # Use CoreDataTransformer for consistency with web_interface
        return CoreDataTransformer.align_processing_paradigm(data, target_mode)

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
        from src.core.data_transformer import CoreDataTransformer

        # Use CoreDataTransformer for consistency with web_interface
        return CoreDataTransformer.apply_cross_module_consistency(
            data=data,
            target_module=target_module,
            source_module=source_module
        )
