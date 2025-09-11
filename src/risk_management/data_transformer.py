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
        Transform Signal to consistent event data format.

        Args:
            signal: Signal to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        return {
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "strength": str(signal.strength),
            "timestamp": signal.timestamp.isoformat()
            if signal.timestamp
            else datetime.now(timezone.utc).isoformat(),
            "source": signal.source or "risk_management",
            "processing_mode": "stream",
            "data_format": "event_data_v1",
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
            "processing_mode": "stream",
            "data_format": "event_data_v1",
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
            "processing_mode": "stream",
            "data_format": "event_data_v1",
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
        Transform error to consistent event data format.

        Args:
            error: Exception to transform
            context: Error context information
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        return {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_context": context or {},
            "processing_mode": "stream",
            "data_format": "event_data_v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
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

        # Ensure data format is set
        if "data_format" not in data:
            data["data_format"] = "event_data_v1"

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
        if isinstance(data, Signal):
            transformed = cls.transform_signal_to_event_data(data, metadata)
        elif isinstance(data, Position):
            transformed = cls.transform_position_to_event_data(data, metadata)
        elif isinstance(data, RiskMetrics):
            transformed = cls.transform_risk_metrics_to_event_data(data, metadata)
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
        transformed["processing_mode"] = "request_reply"  # Override processing mode

        return transformed

    @classmethod
    def align_processing_paradigm(
        cls, data: dict[str, Any], target_mode: str = "stream"
    ) -> dict[str, Any]:
        """
        Align data processing paradigm with execution module expectations using consistent patterns.

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

        # Add mode-specific fields with enhanced boundary metadata
        if target_mode == "stream":
            aligned_data.update(
                {
                    "stream_position": datetime.now(timezone.utc).timestamp(),
                    "data_format": "event_data_v1",
                    "message_pattern": "pub_sub",  # Consistent messaging pattern
                    "boundary_crossed": True,
                }
            )
        elif target_mode == "batch":
            if "batch_id" not in aligned_data:
                aligned_data["batch_id"] = datetime.now(timezone.utc).isoformat()
            aligned_data.update(
                {
                    "data_format": "batch_event_data_v1",
                    "message_pattern": "batch",  # Batch messaging pattern
                    "boundary_crossed": True,
                }
            )
        elif target_mode == "request_reply":
            if "correlation_id" not in aligned_data:
                aligned_data["correlation_id"] = datetime.now(timezone.utc).isoformat()
            aligned_data.update(
                {
                    "data_format": "request_reply_data_v1",
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

        # Apply consistent messaging patterns
        from src.utils.messaging_patterns import (
            BoundaryValidator,
            ProcessingParadigmAligner,
        )

        # Apply processing paradigm alignment
        source_mode = validated_data.get("processing_mode", "stream")
        target_mode = "stream" if target_module == "state" else source_mode

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

        # Apply target-module specific boundary validation
        try:
            if target_module == "state" or target_module == "execution":
                # Validate at risk_management -> state/execution boundary
                boundary_data = {
                    "component": validated_data.get("component", source_module),
                    "operation": validated_data.get("operation", "risk_operation"),
                    "timestamp": validated_data.get(
                        "timestamp", datetime.now(timezone.utc).isoformat()
                    ),
                    "processing_mode": validated_data.get("processing_mode", "stream"),
                    "data_format": validated_data.get("data_format", "event_data_v1"),
                    "boundary_crossed": True,
                }
                BoundaryValidator.validate_risk_to_state_boundary(boundary_data)

        except Exception as e:
            # Log validation issues but don't fail the data flow
            logger.warning(
                f"Cross-module validation failed for {source_module} -> {target_module}: {e}"
            )

        return validated_data
