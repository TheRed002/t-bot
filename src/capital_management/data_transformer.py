"""
Capital Management Data Transformation Utilities.

Handles consistent data transformation for capital_management module
with proper event data formatting and messaging patterns.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.core.types import CapitalAllocation, CapitalMetrics, StateType
from src.utils.decimal_utils import to_decimal
from src.utils.messaging_patterns import MessagePattern, MessageType

logger = get_logger(__name__)


class CapitalDataTransformer:
    """Handles consistent data transformation for capital_management module."""

    @staticmethod
    def transform_allocation_to_event_data(
        allocation: CapitalAllocation, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform CapitalAllocation to event data format consistent with core patterns.

        Args:
            allocation: CapitalAllocation to transform
            metadata: Additional metadata

        Returns:
            Dict with event data aligned with core module patterns
        """
        event_data = {
            "event_id": str(uuid.uuid4()),
            "event_type": "capital_allocation",
            "processing_mode": "stream",  # Align with core default
            "data_format": "capital_event_v1",  # Module-specific format
            "message_pattern": MessagePattern.PUB_SUB.value,  # Use enum for consistency
            "allocation_id": allocation.allocation_id,
            "strategy_id": allocation.strategy_id,
            "symbol": allocation.symbol,
            "allocated_amount": str(allocation.allocated_amount),
            "utilized_amount": str(allocation.utilized_amount),
            "available_amount": str(allocation.available_amount),
            "allocation_percentage": str(allocation.allocation_percentage),
            "target_allocation_pct": str(allocation.target_allocation_pct),
            "min_allocation": str(allocation.min_allocation),
            "max_allocation": str(allocation.max_allocation),
            "last_rebalance": allocation.last_rebalance.isoformat()
            if allocation.last_rebalance
            else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "capital_management",
            "boundary_crossed": True,
            "validation_status": "validated",
            "metadata": metadata or {},
        }

        # Apply financial precision validation
        return CapitalDataTransformer.validate_financial_precision(event_data)

    @staticmethod
    def transform_metrics_to_event_data(
        metrics: CapitalMetrics, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform CapitalMetrics to event data format consistent with core patterns.

        Args:
            metrics: Capital metrics to transform
            metadata: Additional metadata

        Returns:
            Dict with event data aligned with core module patterns
        """
        event_data = {
            "event_id": str(uuid.uuid4()),
            "event_type": "capital_metrics",
            "processing_mode": "stream",  # Align with core default
            "data_format": "capital_event_v1",  # Module-specific format
            "message_pattern": MessagePattern.PUB_SUB.value,  # Use enum for consistency
            "total_capital": str(metrics.total_capital),
            "allocated_amount": str(metrics.allocated_amount),
            "available_amount": str(metrics.available_amount),
            "total_pnl": str(metrics.total_pnl),
            "realized_pnl": str(metrics.realized_pnl),
            "unrealized_pnl": str(metrics.unrealized_pnl),
            "daily_return": str(metrics.daily_return),
            "weekly_return": str(metrics.weekly_return),
            "monthly_return": str(metrics.monthly_return),
            "yearly_return": str(metrics.yearly_return),
            "total_return": str(metrics.total_return),
            "sharpe_ratio": str(metrics.sharpe_ratio),
            "sortino_ratio": str(metrics.sortino_ratio),
            "calmar_ratio": str(metrics.calmar_ratio),
            "current_drawdown": str(metrics.current_drawdown),
            "max_drawdown": str(metrics.max_drawdown),
            "var_95": str(metrics.var_95),
            "expected_shortfall": str(metrics.expected_shortfall),
            "strategies_active": metrics.strategies_active,
            "positions_open": metrics.positions_open,
            "leverage_used": str(metrics.leverage_used),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "capital_management",
            "boundary_crossed": True,
            "validation_status": "validated",
            "metadata": metadata or {},
        }

        # Apply financial precision validation
        return CapitalDataTransformer.validate_financial_precision(event_data)

    @staticmethod
    def transform_error_to_event_data(
        error: Exception,
        operation: str,
        allocation_id: str | None = None,
        strategy_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Transform error to event data format consistent with core patterns.

        Args:
            error: Exception that occurred
            operation: Operation that failed
            allocation_id: Optional allocation ID
            strategy_id: Optional strategy ID
            metadata: Additional metadata

        Returns:
            Dict with error event data aligned with core module patterns
        """
        return {
            "event_id": str(uuid.uuid4()),
            "event_type": "capital_error",
            "processing_mode": "stream",  # Align with core default for errors
            "data_format": "capital_event_v1",  # Module-specific format
            "message_pattern": MessagePattern.PUB_SUB.value,  # Use enum for consistency
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "error_context": getattr(error, "__dict__", {}),
            "operation": operation,
            "allocation_id": allocation_id,
            "strategy_id": strategy_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "capital_management",
            "boundary_crossed": True,
            "validation_status": "error",
            "severity": "high" if isinstance(error, (ValueError, TypeError)) else "medium",
            "metadata": metadata or {},
        }

    @staticmethod
    def validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]:
        """
        Validate financial precision using centralized financial utils.

        Args:
            data: Data to validate

        Returns:
            Validated data with proper Decimal conversions
        """
        from src.utils.financial_utils import validate_financial_precision
        return validate_financial_precision(data)

    @staticmethod
    def ensure_boundary_fields(data: dict[str, Any], source: str = "capital_management") -> dict[str, Any]:
        """
        Ensure boundary fields using centralized financial utils.

        Args:
            data: Data to process
            source: Source service identifier

        Returns:
            Data with boundary fields
        """
        from src.utils.financial_utils import ensure_boundary_fields
        return ensure_boundary_fields(data, source)

    @classmethod
    def transform_for_pub_sub(
        cls, event_type: str, data: Any, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform data for pub/sub messaging pattern consistent with core module.

        Args:
            event_type: Type of event
            data: Event data
            metadata: Additional metadata

        Returns:
            Formatted data for pub/sub
        """
        return {
            "message_id": str(uuid.uuid4()),
            "event_type": event_type,
            "processing_mode": "stream",  # Align with core default
            "data_format": "capital_event_v1",  # Module-specific format
            "message_pattern": MessagePattern.PUB_SUB.value,  # Use enum for consistency  # Align with core pattern
            "data": data,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "capital_management",
            "boundary_crossed": True,
            "validation_status": "validated",
            "distribution_mode": "broadcast",
            "acknowledgment_required": False,
        }

    @classmethod
    def transform_for_req_reply(
        cls, request_type: str, data: Any, correlation_id: str | None = None
    ) -> dict[str, Any]:
        """
        Transform data for request/reply messaging pattern consistent with core module.

        Args:
            request_type: Type of request
            data: Request data
            correlation_id: Optional correlation ID

        Returns:
            Formatted data for req/reply
        """
        return {
            "request_id": str(uuid.uuid4()),
            "correlation_id": correlation_id or str(uuid.uuid4()),
            "event_type": request_type,  # Align with core naming
            "processing_mode": "request_reply",  # Align with core pattern
            "data_format": "capital_event_v1",  # Module-specific format
            "message_pattern": MessagePattern.REQ_REPLY.value,  # Use enum for consistency
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "capital_management",
            "boundary_crossed": True,
            "validation_status": "validated",
            "response_required": True,
            "acknowledgment_required": True,
            "synchronous_processing": True,
        }

    @classmethod
    def align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = "stream") -> dict[str, Any]:
        """
        Align data processing paradigm using utils.messaging_patterns for consistency with risk_management.

        Args:
            data: Data to align
            target_mode: Target processing mode ('stream', 'batch', 'request_reply')

        Returns:
            Aligned data format using standardized utils patterns
        """
        # Use ProcessingParadigmAligner from utils for full consistency with risk_management
        from src.utils.messaging_patterns import ProcessingParadigmAligner

        source_mode = data.get("processing_mode", "stream")
        aligned_data = ProcessingParadigmAligner.align_processing_modes(
            source_mode=source_mode, target_mode=target_mode, data=data
        )

        # Add mode-specific fields matching bot_management patterns
        if target_mode == "stream":
            aligned_data.update({
                "stream_processing": True,
                "stream_position": datetime.now(timezone.utc).timestamp(),
                "processing_paradigm": "stream",
                "data_format": "capital_event_v1",
                "message_pattern": MessagePattern.PUB_SUB.value,
                "boundary_crossed": True,
            })
        elif target_mode == "batch":
            if "batch_id" not in aligned_data:
                aligned_data["batch_id"] = str(uuid.uuid4())
            aligned_data.update({
                "batch_processing": True,
                "processing_paradigm": "batch",
                "data_format": "capital_event_v1",
                "message_pattern": MessagePattern.BATCH.value,
                "boundary_crossed": True,
            })
        elif target_mode == "request_reply":
            if "correlation_id" not in aligned_data:
                aligned_data["correlation_id"] = str(uuid.uuid4())
            aligned_data.update({
                "data_format": "capital_event_v1",
                "message_pattern": MessagePattern.REQ_REPLY.value,
                "boundary_crossed": True,
            })

        # Add capital_management-specific metadata
        aligned_data.update({
            "validation_status": "validated",
            "capital_module_aligned": True,
            "paradigm_alignment_source": "utils.messaging_patterns",
        })

        return aligned_data

    @classmethod
    def transform_allocation_data_for_state(
        cls, allocation_data: dict[str, Any], operation: str = "unknown"
    ) -> dict[str, Any]:
        """
        Transform allocation data for state management integration.

        Args:
            allocation_data: Raw allocation data
            operation: Operation being performed

        Returns:
            State-compatible data format with StateType
        """
        return {
            "state_type": StateType.CAPITAL_STATE,
            "state_id": f"capital_allocation:{allocation_data.get('allocation_id', 'unknown')}",
            "state_data": allocation_data,
            "operation": operation,
            "version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "source": "capital_management",
                "operation_type": operation,
                "state_category": "capital_allocation",
            },
        }

    @staticmethod
    def validate_boundary_fields(data: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and ensure required boundary fields for cross-module communication.

        Args:
            data: Data dictionary to validate

        Returns:
            Dict with validated boundary fields

        Raises:
            ValidationError: If required fields are missing or invalid
        """
        required_fields = [
            "processing_mode",
            "data_format",
            "message_pattern",
            "timestamp",
            "source"
        ]

        for field in required_fields:
            if field not in data:
                raise ValidationError(
                    f"Missing required boundary field: {field}",
                    field_name=field,
                    field_value=None,
                    expected_type="string"
                )

        # Validate processing mode values
        valid_modes = ["stream", "batch", "request_reply"]
        if data["processing_mode"] not in valid_modes:
            raise ValidationError(
                f"Invalid processing_mode: {data['processing_mode']}. Must be one of {valid_modes}",
                field_name="processing_mode",
                field_value=data["processing_mode"],
                expected_type="string"
            )

        # Validate message pattern values using enum
        valid_patterns = [pattern.value for pattern in MessagePattern]
        if data["message_pattern"] not in valid_patterns:
            raise ValidationError(
                f"Invalid message_pattern: {data['message_pattern']}. Must be one of {valid_patterns}",
                field_name="message_pattern",
                field_value=data["message_pattern"],
                expected_type="string"
            )

        # Add validation metadata
        data["boundary_validation"] = "applied"
        data["validation_timestamp"] = datetime.now(timezone.utc).isoformat()

        return data

    @classmethod
    def apply_cross_module_consistency(
        cls,
        data: dict[str, Any],
        target_module: str,
        source_module: str = "capital_management"
    ) -> dict[str, Any]:
        """
        Apply comprehensive cross-module consistency for data flow alignment.

        Args:
            data: Data to make consistent
            target_module: Target module name
            source_module: Source module name

        Returns:
            Dict with cross-module consistency applied
        """
        consistent_data = data.copy()

        # Apply consistent messaging patterns from utils like risk_management does
        from src.utils.messaging_patterns import (
            BoundaryValidator,
            ProcessingParadigmAligner,
        )

        # Apply standardized processing paradigm alignment
        source_mode = consistent_data.get("processing_mode", "stream")
        target_mode = "stream" if target_module in ["state", "execution", "risk_management"] else source_mode

        consistent_data = ProcessingParadigmAligner.align_processing_modes(
            source_mode=source_mode, target_mode=target_mode, data=consistent_data
        )

        # Apply boundary validation and financial precision
        consistent_data = cls.validate_boundary_fields(consistent_data)
        consistent_data = cls.validate_financial_precision(consistent_data)

        # Add comprehensive boundary metadata consistent with risk_management
        consistent_data.update(
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
                "component": consistent_data.get("component", source_module),
                "operation": consistent_data.get("operation", "capital_operation"),
                "timestamp": consistent_data.get(
                    "timestamp", datetime.now(timezone.utc).isoformat()
                ),
                "processing_mode": consistent_data.get("processing_mode", "stream"),
                "data_format": consistent_data.get("data_format", "bot_event_v1"),
                "message_pattern": consistent_data.get("message_pattern", MessagePattern.PUB_SUB.value),
                "boundary_crossed": True,
                "processing_paradigm": consistent_data.get("processing_paradigm", "stream"),
            }

            # Apply appropriate boundary validation based on target (align with state module patterns)
            if target_module == "core":
                # Core module expects event-like data with specific patterns
                BoundaryValidator.validate_monitoring_to_error_boundary(boundary_data)
            elif target_module == "risk_management":
                BoundaryValidator.validate_state_to_risk_boundary(boundary_data)
            elif target_module == "state":
                BoundaryValidator.validate_risk_to_state_boundary(boundary_data)
            elif target_module in ["monitoring", "error_handling"]:
                BoundaryValidator.validate_error_to_monitoring_boundary(boundary_data)
            else:
                BoundaryValidator.validate_database_entity(boundary_data, "validate")

        except Exception as e:
            # Use consistent error handling pattern from utils like risk_management
            logger.warning(
                f"Cross-module validation failed for {source_module} -> {target_module}: {e}",
                extra={
                    "source_module": source_module,
                    "target_module": target_module,
                    "validation_error": str(e),
                    "data_format": consistent_data.get("data_format", "unknown"),
                    "boundary_validation_applied": True,
                },
            )

        return consistent_data

    @classmethod
    def propagate_capital_error(
        cls,
        error: Exception,
        operation: str,
        allocation_id: str | None = None,
        strategy_id: str | None = None,
        target_module: str = "core"
    ) -> dict[str, Any]:
        """
        Propagate capital management errors consistently with bot_management patterns.

        Args:
            error: Exception to propagate
            operation: Operation that failed
            allocation_id: Optional allocation ID
            strategy_id: Optional strategy ID
            target_module: Target module for error propagation

        Returns:
            Properly formatted error event data
        """
        error_data = cls.transform_error_to_event_data(
            error, operation, allocation_id, strategy_id
        )

        # Apply cross-module consistency for error propagation
        return cls.apply_cross_module_consistency(
            error_data, target_module, "capital_management"
        )

    @staticmethod
    def _apply_boundary_validation(
        data: dict[str, Any],
        source_module: str,
        target_module: str
    ) -> None:
        """
        Apply boundary validation using the same patterns as core module.

        Args:
            data: Data to validate
            source_module: Source module name
            target_module: Target module name
        """
        try:
            # Use the same boundary validation utilities as core module
            from src.utils.messaging_patterns import BoundaryValidator

            if target_module == "core":
                # Validate capital_management -> core boundary
                boundary_data = {
                    "component": source_module,
                    "error_type": data.get("event_type", "CapitalEvent"),
                    "severity": data.get("severity", "medium"),
                    "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    "processing_mode": data.get("processing_mode", "stream"),
                    "data_format": data.get("data_format", "capital_event_v1"),  # Align with risk_management
                    "message_pattern": data.get("message_pattern", MessagePattern.PUB_SUB.value),
                    "boundary_crossed": True,
                }
                BoundaryValidator.validate_monitoring_to_error_boundary(boundary_data)

            elif target_module == "risk_management":
                # Validate capital_management -> risk_management boundary
                boundary_data = {
                    "component": source_module,
                    "operation": data.get("event_type", "capital_operation"),
                    "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    "processing_mode": data.get("processing_mode", "stream"),
                    "data_format": data.get("data_format", "capital_event_v1"),  # Align with risk_management
                    "boundary_crossed": True,
                }
                BoundaryValidator.validate_monitoring_to_error_boundary(boundary_data)

        except Exception as e:
            # Log validation issues but don't fail the data transformation - matching bot_management pattern
            logger.warning(
                f"Cross-module validation failed for {source_module} -> {target_module}: {e}",
                extra={
                    "source_module": source_module,
                    "target_module": target_module,
                    "validation_error": str(e),
                    "data_format": data.get("data_format", "unknown"),
                    "boundary_validation_applied": True,
                },
            )
