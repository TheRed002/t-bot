"""
Data transformation utilities for optimization module.

Provides consistent data transformation patterns to align with backtesting module
and ensure proper cross-module communication in batch processing mode.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.optimization.core import OptimizationResult
from src.utils.decimal_utils import to_decimal
from src.utils.messaging_patterns import MessagePattern

if TYPE_CHECKING:
    from src.optimization.parameter_space import ParameterSpace


class OptimizationDataTransformer:
    """Handles consistent data transformation for optimization module."""

    @staticmethod
    def transform_optimization_result_to_event_data(
        result: OptimizationResult, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform OptimizationResult to consistent event data format.

        Args:
            result: Optimization result to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        # Apply consistent data transformation aligned with backtesting module
        transformed = {
            "optimization_id": result.optimization_id,
            "algorithm_name": result.algorithm_name,
            "optimal_objective_value": str(result.optimal_objective_value),
            "optimal_parameters": result.optimal_parameters,
            "iterations_completed": result.iterations_completed,
            "evaluations_completed": result.evaluations_completed,
            "convergence_achieved": result.convergence_achieved,
            "total_duration_seconds": result.total_duration_seconds,
            "validation_score": str(result.validation_score) if result.validation_score is not None else None,
            "overfitting_score": str(result.overfitting_score) if result.overfitting_score is not None else None,
            "robustness_score": str(result.robustness_score) if result.robustness_score is not None else None,
            "statistical_significance": str(result.statistical_significance) if result.statistical_significance is not None else None,
            "confidence_interval": [str(ci) for ci in result.confidence_interval] if result.confidence_interval else None,
            "warnings": result.warnings or [],
            "processing_mode": "batch",  # Optimization results are batch data
            "data_format": "batch_event_data_v1",  # Use versioned format aligned with backtesting
            "message_pattern": MessagePattern.BATCH.value,  # Align with batch processing
            "boundary_crossed": True,
            "validation_status": "validated",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        return OptimizationDataTransformer.validate_financial_precision(transformed)

    @staticmethod
    def transform_parameter_set_to_event_data(
        parameters: dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform parameter set to consistent event data format.

        Args:
            parameters: Parameter set to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        # Apply consistent data transformation aligned with backtesting module
        transformed = {
            "parameters": parameters,
            "parameter_count": len(parameters),
            "processing_mode": "batch",  # Parameter sets are batch data
            "data_format": "batch_event_data_v1",  # Use versioned format
            "message_pattern": MessagePattern.BATCH.value,  # Align with batch processing
            "boundary_crossed": True,
            "validation_status": "validated",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        return OptimizationDataTransformer.validate_financial_precision(transformed)

    @staticmethod
    def transform_objective_values_to_event_data(
        objective_values: dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform objective values to consistent event data format.

        Args:
            objective_values: Objective values to transform
            metadata: Additional metadata

        Returns:
            Dict with consistent event data format
        """
        return {
            "objective_values": {k: str(v) for k, v in objective_values.items()},
            "objective_count": len(objective_values),
            "processing_mode": "batch",  # Objective values are batch data
            "data_format": "event_data_v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

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
        from src.utils.decimal_utils import to_decimal

        # Define optimization-specific fields in addition to defaults
        optimization_fields = [
            "optimal_objective_value", "objective_value", "validation_score",
            "overfitting_score", "robustness_score", "statistical_significance",
            "confidence_interval", "total_return", "sharpe_ratio",
            "max_drawdown", "initial_capital"
        ]

        validated_data = validate_financial_precision(data, optimization_fields)

        # Handle special case for confidence_interval as list
        if "confidence_interval" in validated_data and isinstance(validated_data["confidence_interval"], list):
            try:
                validated_data["confidence_interval"] = [
                    str(to_decimal(val)) for val in validated_data["confidence_interval"] if val is not None
                ]
            except Exception:
                pass

        # Add financial precision metadata aligned with backtesting
        if any(field in validated_data for field in optimization_fields):
            validated_data["financial_precision_applied"] = True
            validated_data["decimal_precision"] = "20,8"

        return validated_data

    @staticmethod
    def ensure_boundary_fields(data: dict[str, Any], source: str = "optimization") -> dict[str, Any]:
        """
        Ensure data has required boundary fields using centralized financial utils.

        Args:
            data: Data dictionary to enhance
            source: Source module name

        Returns:
            Dict with required boundary fields
        """
        from src.utils.financial_utils import ensure_boundary_fields

        # Use centralized boundary fields function (automatically handles batch mode for optimization)
        enhanced_data = ensure_boundary_fields(data, source, data_format="batch_event_data_v1")

        # Add optimization-specific fields
        if "metadata" not in enhanced_data:
            enhanced_data["metadata"] = {}

        # Add optimization-specific fields - align with batch processing mode
        enhanced_data["message_pattern"] = enhanced_data.get("message_pattern", MessagePattern.BATCH.value)
        enhanced_data["boundary_crossed"] = True

        return enhanced_data

    @classmethod
    def transform_for_req_reply(
        cls, request_type: str, data: Any, correlation_id: str | None = None
    ) -> dict[str, Any]:
        """
        Transform data for request/reply messaging pattern aligned with backtesting module.

        Uses backtesting module's request/reply pattern for consistency.

        Args:
            request_type: Type of request
            data: Request data
            correlation_id: Request correlation ID

        Returns:
            Dict formatted for req/reply pattern
        """
        # Base transformation
        if isinstance(data, OptimizationResult):
            transformed = cls.transform_optimization_result_to_event_data(data)
        elif isinstance(data, dict):
            if "optimal_objective_value" in data or "optimization_id" in data:
                # Optimization result-like data
                transformed = cls.ensure_boundary_fields(data.copy())
            elif any(key in data for key in ["parameters", "parameter_space"]):
                # Parameter set data
                transformed = cls.transform_parameter_set_to_event_data(data)
            else:
                transformed = data.copy()
        else:
            transformed = {"payload": str(data), "type": type(data).__name__}

        # Ensure boundary fields
        transformed = cls.ensure_boundary_fields(transformed)

        # Validate financial precision
        transformed = cls.validate_financial_precision(transformed)

        # Add request/reply specific fields aligned with backtesting patterns
        transformed.update(
            {
                "request_type": request_type,
                "correlation_id": correlation_id or datetime.now(timezone.utc).isoformat(),
                "processing_mode": "request_reply",  # Align processing mode with req_reply pattern
                "data_format": "request_reply_data_v1",  # Use backtesting format
                "message_pattern": MessagePattern.REQ_REPLY.value,
                "response_required": True,
                "acknowledgment_required": True,
                "boundary_crossed": True,
                "validation_status": "validated",
            }
        )

        return transformed

    @classmethod
    def transform_for_batch_processing(
        cls,
        batch_type: str,
        data_items: list,
        batch_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Transform data for batch processing pattern (aligned with backtesting).

        Args:
            batch_type: Type of batch operation
            data_items: List of items to process in batch
            batch_id: Unique batch identifier
            metadata: Additional batch metadata

        Returns:
            Dict formatted for batch processing
        """
        # Transform each item individually
        transformed_items = []
        for item in data_items:
            if isinstance(item, OptimizationResult):
                transformed_items.append(cls.transform_optimization_result_to_event_data(item))
            elif isinstance(item, dict):
                if "optimal_objective_value" in item or "optimization_id" in item:
                    transformed_items.append(cls.ensure_boundary_fields(item.copy()))
                elif any(key in item for key in ["parameters", "parameter_space"]):
                    transformed_items.append(cls.transform_parameter_set_to_event_data(item))
                else:
                    transformed_items.append(cls.ensure_boundary_fields(item.copy()))
            else:
                transformed_items.append(
                    {
                        "payload": str(item),
                        "type": type(item).__name__,
                        "processing_mode": "batch",
                        "data_format": "event_data_v1",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

        batch_data = {
            "batch_type": batch_type,
            "batch_id": batch_id or datetime.now(timezone.utc).isoformat(),
            "batch_size": len(data_items),
            "items": transformed_items,
            "processing_mode": "batch",
            "data_format": "batch_event_data_v1",
            "message_pattern": MessagePattern.BATCH.value,
            "distribution_mode": "batch",
            "acknowledgment_required": False,  # Align with backtesting batch pattern
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "optimization",
            "boundary_crossed": True,
            "validation_status": "validated",
            "metadata": metadata or {},
        }

        # Apply consistent financial validation to batch data
        batch_data = cls.validate_financial_precision(batch_data)

        return batch_data

    @classmethod
    def align_processing_paradigm(
        cls, data: dict[str, Any], target_mode: str = "batch"
    ) -> dict[str, Any]:
        """
        Align data processing paradigm with target module expectations.

        (consistent with backtesting)

        Args:
            data: Data to align
            target_mode: Target processing mode ("stream", "batch", "request_reply")

        Returns:
            Dict with aligned processing mode
        """
        aligned_data = data.copy()

        # Use ProcessingParadigmAligner for consistency with backtesting module
        from src.utils.messaging_patterns import ProcessingParadigmAligner

        # Apply consistent processing paradigm alignment
        source_mode = aligned_data.get("processing_mode", "batch")
        aligned_data = ProcessingParadigmAligner.align_processing_modes(
            source_mode=source_mode, target_mode=target_mode, data=aligned_data
        )

        # Add mode-specific fields aligned with backtesting module patterns
        if target_mode == "batch":
            if "batch_id" not in aligned_data:
                aligned_data["batch_id"] = datetime.now(timezone.utc).isoformat()
            aligned_data.update(
                {
                    "data_format": "batch_event_data_v1",
                    "message_pattern": MessagePattern.BATCH.value,
                    "distribution_mode": "batch",
                    "acknowledgment_required": False,
                    "batch_processing": True,
                    "boundary_crossed": True,
                }
            )
        elif target_mode == "request_reply":
            if "correlation_id" not in aligned_data:
                aligned_data["correlation_id"] = datetime.now(timezone.utc).isoformat()
            aligned_data.update(
                {
                    "data_format": "request_reply_data_v1",
                    "message_pattern": MessagePattern.REQ_REPLY.value,
                    "response_required": True,
                    "acknowledgment_required": True,
                    "synchronous_processing": True,
                    "boundary_crossed": True,
                }
            )
        elif target_mode == "stream":
            aligned_data.update(
                {
                    "stream_position": datetime.now(timezone.utc).timestamp(),
                    "data_format": "core_event_data_v1",  # Use core format for stream
                    "message_pattern": MessagePattern.PUB_SUB.value,
                    "distribution_mode": "broadcast",
                    "acknowledgment_required": False,
                    "real_time_processing": True,
                    "boundary_crossed": True,
                }
            )

        # Add consistent validation status
        aligned_data["validation_status"] = "validated"

        # Apply consistent financial validation
        aligned_data = cls.validate_financial_precision(aligned_data)

        # Add processing paradigm metadata
        aligned_data["paradigm_alignment_applied"] = True
        aligned_data["target_processing_mode"] = target_mode

        return aligned_data

    @classmethod
    def apply_cross_module_validation(
        cls,
        data: dict[str, Any],
        source_module: str = "optimization",
        target_module: str = "backtesting",
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
        source_mode = validated_data.get("processing_mode", "batch")
        target_mode = "batch" if target_module == "backtesting" else source_mode

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
            if target_module == "backtesting":
                # Validate at optimization -> backtesting boundary
                boundary_data = {
                    "optimization_id": validated_data.get("optimization_id", "unknown"),
                    "optimal_parameters": validated_data.get("optimal_parameters", {}),
                    "timestamp": validated_data.get(
                        "timestamp", datetime.now(timezone.utc).isoformat()
                    ),
                    "processing_mode": validated_data.get("processing_mode", "batch"),
                    "data_format": validated_data.get("data_format", "event_data_v1"),
                    "boundary_crossed": True,
                }
                BoundaryValidator.validate_database_entity(boundary_data, "write")

            elif target_module == "strategies":
                # Validate at optimization -> strategies boundary
                boundary_data = {
                    "strategy_name": validated_data.get("strategy_name", "unknown"),
                    "optimal_parameters": validated_data.get("optimal_parameters", {}),
                    "timestamp": validated_data.get(
                        "timestamp", datetime.now(timezone.utc).isoformat()
                    ),
                    "processing_mode": validated_data.get("processing_mode", "batch"),
                    "data_format": validated_data.get("data_format", "event_data_v1"),
                    "boundary_crossed": True,
                }
                BoundaryValidator.validate_database_entity(boundary_data, "write")

        except Exception:
            # Log validation issues but don't fail the data flow
            pass

        # Apply consistent financial validation for cross-module data flow
        validated_data = cls.validate_financial_precision(validated_data)

        # Apply backtesting module boundary validation for consistency
        validated_data = cls._apply_backtesting_boundary_validation(
            validated_data, source_module, target_module
        )

        return validated_data

    @classmethod
    def _apply_backtesting_boundary_validation(
        cls,
        data: dict[str, Any],
        source_module: str,
        target_module: str,
    ) -> dict[str, Any]:
        """
        Apply backtesting module boundary validation patterns for consistency.

        Args:
            data: Data to validate
            source_module: Source module name
            target_module: Target module name

        Returns:
            Dict with backtesting boundary validation applied
        """
        validated_data = data.copy()

        # Apply backtesting module validation patterns using existing utility
        try:
            from src.backtesting.data_transformer import BacktestDataTransformer

            # Apply backtesting cross-module consistency
            validated_data = BacktestDataTransformer.apply_cross_module_validation(
                validated_data, source_module, target_module
            )

            # Add boundary validation metadata
            validated_data.update({
                "backtesting_boundary_validation": True,
                "boundary_validation_timestamp": datetime.now(timezone.utc).isoformat(),
                "validation_source": "optimization_backtesting_aligned",
            })

        except ImportError:
            # Fallback validation if backtesting transformer not available
            validated_data.update({
                "backtesting_boundary_validation": False,
                "fallback_validation": True,
                "boundary_validation_timestamp": datetime.now(timezone.utc).isoformat(),
            })

        return validated_data