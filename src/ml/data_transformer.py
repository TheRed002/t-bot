"""
ML data transformation utilities aligned with core module patterns.

This module provides standardized data transformation patterns that align
with core module patterns to ensure consistency across the trading system.
"""

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from pydantic import BaseModel

from src.core.data_transformer import CoreDataTransformer
from src.core.exceptions import ModelError, ValidationError
from src.core.logging import get_logger
from src.utils.decimal_utils import to_decimal

logger = get_logger(__name__)


class MLDataTransformer:
    """Handles consistent data transformation for ML module operations aligned with core patterns."""

    @staticmethod
    def transform_ml_request_to_standard_format(
        request_type: str, 
        data: Any, 
        metadata: dict[str, Any] | None = None,
        processing_mode: str = "stream"  # Default aligned with utils messaging patterns
    ) -> dict[str, Any]:
        """
        Transform ML request data to standard format aligned with core module patterns.
        
        Args:
            request_type: Type of ML request (prediction, training, etc.)
            data: Request data to transform
            metadata: Additional metadata
            processing_mode: Processing mode ("batch", "stream", "request_reply")
            
        Returns:
            Dict with standardized ML request format
        """
        # Use core transformer as base to ensure alignment
        transformed = CoreDataTransformer.transform_event_to_standard_format(
            request_type, data, metadata, source="ml"
        )
        
        # Override with ML-specific defaults aligned with utils messaging patterns
        transformed.update({
            "processing_mode": processing_mode,
            "data_format": "bot_event_v1",  # Align with utils messaging format
            "message_pattern": "pub_sub",  # Default to pub_sub pattern for consistency
            "ml_operation_type": request_type,
            "boundary_crossed": True,  # Align with utils boundary pattern
        })
        
        # Apply ML-specific data handling
        if isinstance(data, BaseModel):
            ml_data = data.model_dump()
        elif isinstance(data, pd.DataFrame):
            ml_data = {
                "dataframe_info": {
                    "shape": list(data.shape),
                    "columns": list(data.columns),
                    "dtypes": data.dtypes.to_dict(),
                    "has_na": data.isna().any().any(),
                },
                "data_type": "pandas_dataframe",
                "serialized": True,
            }
        elif isinstance(data, dict):
            ml_data = data.copy()
        else:
            ml_data = {"payload": str(data), "type": type(data).__name__}
            
        transformed["data"] = ml_data
        
        return MLDataTransformer._apply_ml_financial_precision(transformed)

    @staticmethod
    def transform_for_inference_pattern(
        model_id: str,
        input_data: Any,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform data for inference operations using aligned messaging patterns.
        
        Args:
            model_id: Model identifier for inference
            input_data: Input data for inference
            metadata: Additional metadata
            
        Returns:
            Dict formatted for inference operations
        """
        transformed = MLDataTransformer.transform_ml_request_to_standard_format(
            "inference_request", input_data, metadata, processing_mode="request_reply"
        )
        
        # Add inference-specific fields aligned with utils messaging patterns
        transformed.update({
            "message_pattern": "req_reply",  # Use req_reply for inference responses
            "model_id": model_id,
            "inference_mode": "real_time",
            "response_required": True,  # Keep response requirement for inference
        })
        
        return transformed

    @staticmethod
    def transform_for_training_pattern(
        training_data: Any,
        target_data: Any,
        model_type: str,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform data for training operations using aligned messaging patterns.
        
        Args:
            training_data: Training dataset
            target_data: Target values
            model_type: Type of model to train
            metadata: Additional metadata
            
        Returns:
            Dict formatted for training operations
        """
        combined_data = {
            "training_data": training_data,
            "target_data": target_data,
            "model_type": model_type,
        }
        
        transformed = MLDataTransformer.transform_ml_request_to_standard_format(
            "training_request", combined_data, metadata, processing_mode="batch"
        )
        
        # Add training-specific fields aligned with utils messaging patterns
        transformed.update({
            "message_pattern": "batch",  # Use batch pattern for training operations
            "training_mode": "supervised", 
            "batch_processing": True,
        })
        
        return transformed

    @staticmethod
    def align_with_core_processing_paradigm(
        data: dict[str, Any], 
        target_mode: str | None = None
    ) -> dict[str, Any]:
        """
        Align ML data processing paradigm with core module patterns.
        
        Args:
            data: ML data to align
            target_mode: Target processing mode (if None, infer from operation type)
            
        Returns:
            Dict with aligned processing mode
        """
        if target_mode is None:
            # Infer appropriate mode based on ML operation type aligned with data module logic
            operation_type = data.get("ml_operation_type", "")
            data_size = data.get("data", {}).get("batch_size", 1) if isinstance(data.get("data"), dict) else 1
            
            # Apply utils messaging alignment logic: single items use stream, large batches use batch
            if "training" in operation_type or "batch" in operation_type:
                target_mode = "batch" if data_size > 1 else "stream"
            elif "inference" in operation_type or "prediction" in operation_type:
                target_mode = "request_reply" if data_size <= 10 else "batch" 
            else:
                target_mode = "stream"  # Default fallback aligned with utils messaging
        
        # Use core transformer to ensure consistent paradigm alignment
        return CoreDataTransformer.align_processing_paradigm(data, target_mode)

    @staticmethod
    def validate_ml_boundary_fields(data: dict[str, Any]) -> dict[str, Any]:
        """
        Validate ML-specific boundary fields using core validation patterns.
        
        This method only handles data structure validation, not business logic validation.
        Business logic validation should be performed in the appropriate service layer.
        
        Args:
            data: ML data dictionary to validate
            
        Returns:
            Dict with validated ML boundary fields
            
        Raises:
            ValidationError: If required ML fields are missing or invalid
        """
        # First apply core boundary validation
        validated_data = CoreDataTransformer.validate_boundary_fields(data)
        
        # Only validate data structure, not business rules
        # Business rule validation should be in services
        if "ml_operation_type" in validated_data:
            # Add ML boundary validation metadata aligned with utils boundary pattern
            validated_data["ml_boundary_validation"] = "applied"
            validated_data["ml_validation_timestamp"] = datetime.now(timezone.utc).isoformat()
            validated_data["boundary_crossed"] = True  # Align with utils boundary pattern
            validated_data["validation_status"] = "passed"  # Add validation status for consistency
        
        return validated_data

    @staticmethod
    def _apply_ml_financial_precision(data: dict[str, Any]) -> dict[str, Any]:
        """
        Apply financial precision to ML-specific fields using core patterns.
        
        Args:
            data: ML data dictionary to process
            
        Returns:
            Dict with validated ML financial precision
        """
        # First apply core financial precision
        processed_data = CoreDataTransformer._apply_financial_precision(data)
        
        # Add ML-specific financial fields
        ml_financial_fields = [
            "prediction_value", "confidence_threshold", "risk_score",
            "feature_importance", "model_score", "validation_metric"
        ]
        
        data_dict = processed_data.get("data", {})
        if not isinstance(data_dict, dict):
            return processed_data
            
        for field in ml_financial_fields:
            if field in data_dict and data_dict[field] is not None:
                try:
                    # Apply same decimal precision as core module
                    if isinstance(data_dict[field], (int, float, str)):
                        decimal_value = to_decimal(data_dict[field])
                        data_dict[field] = str(decimal_value)
                    elif isinstance(data_dict[field], dict):
                        # Handle nested dictionaries (e.g., feature importance)
                        for k, v in data_dict[field].items():
                            if v is not None:
                                try:
                                    decimal_value = to_decimal(v)
                                    data_dict[field][k] = str(decimal_value)
                                except (ValueError, TypeError):
                                    continue
                except (ValueError, TypeError):
                    logger.warning(f"Failed to convert ML field {field} to Decimal: {data_dict[field]}")
        
        # Add ML financial precision metadata
        if any(field in data_dict for field in ml_financial_fields):
            processed_data["ml_financial_precision_applied"] = True
            
        return processed_data

    @classmethod
    def apply_cross_module_consistency_from_ml(
        cls,
        data: dict[str, Any],
        target_module: str
    ) -> dict[str, Any]:
        """
        Apply cross-module consistency when sending data from ML to other modules.
        
        This method only handles data transformation, not business logic decisions.
        Business logic about target module requirements should be in services.
        
        Args:
            data: ML data to make consistent
            target_module: Target module name
            
        Returns:
            Dict with cross-module consistency applied
        """
        # Use core cross-module consistency as base
        consistent_data = CoreDataTransformer.apply_cross_module_consistency(
            data, target_module, source_module="ml"
        )
        
        # Add ML-specific cross-module metadata (data transformation only)
        consistent_data.update({
            "ml_to_module_consistency": True,
            "ml_operation_aligned": True,
        })
        
        # Business logic decisions about target module processing modes
        # should be handled by the service layer, not the data transformer
        
        return consistent_data

    @staticmethod
    def handle_ml_error_propagation(
        error: Exception,
        context: dict[str, Any],
        target_module: str = "core"
    ) -> dict[str, Any]:
        """
        Handle ML error propagation using consistent patterns aligned with data module.
        
        Args:
            error: Exception to propagate
            context: Error context information
            target_module: Target module for error propagation
            
        Returns:
            Dict with standardized error format aligned with data module patterns
        """
        # Map ML errors to consistent error types aligned with data module
        if isinstance(error, (ValueError, TypeError)):
            # Convert to ValidationError for consistency
            standardized_error = ValidationError(
                f"ML validation failed: {str(error)}",
                field_name=context.get("field_name", "ml_data"),
                field_value=context.get("field_value"),
                expected_type=context.get("expected_type", "valid_ml_data")
            )
        elif isinstance(error, (KeyError, AttributeError)):
            # Convert to ModelError for ML-specific issues
            standardized_error = ModelError(f"ML model error: {str(error)}")
        else:
            # Keep original error type if already standardized
            standardized_error = error
            
        # Use core transformer for consistent error formatting
        error_data = CoreDataTransformer.transform_event_to_standard_format(
            "ml_error_event",
            standardized_error,
            metadata=context,
            source="ml"
        )
        
        # Add ML-specific error metadata aligned with utils messaging patterns
        error_data.update({
            "error_category": "ml_operation",
            "ml_context": context,
            "target_module": target_module,
            "component": context.get("component", "MLDataTransformer"),  # Align with utils pattern
            "operation": context.get("operation", "ml_processing"),  # Align with utils pattern
            "boundary_crossed": True,  # Align with utils boundary pattern
            "severity": context.get("severity", "medium"),  # Add severity for utils consistency
            "data_format": "error_propagation_v1",  # Align with utils error format
            "message_pattern": "pub_sub",  # Default error propagation pattern
        })
        
        return error_data

    @staticmethod
    def batch_transform_ml_data(
        batch_data: list[dict[str, Any]],
        operation_type: str = "batch_processing",
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform batch ML data using consistent patterns aligned with utils messaging.
        
        Args:
            batch_data: List of ML data items to process as batch
            operation_type: Type of batch operation
            metadata: Additional metadata for batch processing
            
        Returns:
            Dict with standardized batch format aligned with utils patterns
        """
        from uuid import uuid4
        
        # Create batch wrapper aligned with utils ProcessingParadigmAligner
        batch_wrapper = {
            "items": batch_data,
            "batch_id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "size": len(batch_data),
            "processing_mode": "batch",
            "message_pattern": "batch",  # Use batch pattern for consistency
            "data_format": "bot_event_v1",  # Align with utils format
            "operation_type": operation_type,
            "boundary_crossed": True,
            "ml_batch_processing": True,
        }
        
        # Add metadata if provided
        if metadata:
            batch_wrapper["metadata"] = metadata
            
        # Apply financial precision to all items in batch
        for item in batch_wrapper["items"]:
            if isinstance(item, dict):
                MLDataTransformer._apply_ml_financial_precision_to_item(item)
        
        return batch_wrapper

    @staticmethod
    def stream_to_batch_ml_data(
        stream_items: list[dict[str, Any]],
        batch_size: int = 100
    ) -> list[dict[str, Any]]:
        """
        Convert ML stream data to batch format aligned with utils patterns.
        
        Args:
            stream_items: List of stream items to batch
            batch_size: Maximum size per batch
            
        Returns:
            List of batch dictionaries
        """
        batches = []
        
        # Split stream items into batches
        for i in range(0, len(stream_items), batch_size):
            batch_items = stream_items[i:i + batch_size]
            batch_data = MLDataTransformer.batch_transform_ml_data(
                batch_items,
                operation_type="stream_to_batch_conversion"
            )
            batches.append(batch_data)
            
        return batches

    @staticmethod
    def _apply_ml_financial_precision_to_item(item: dict[str, Any]) -> None:
        """
        Apply financial precision to a single ML data item.
        
        Args:
            item: ML data item to process (modified in place)
        """
        ml_financial_fields = [
            "prediction_value", "confidence_threshold", "risk_score",
            "feature_importance", "model_score", "validation_metric",
            "price", "quantity", "volume", "amount"
        ]
        
        for field in ml_financial_fields:
            if field in item and item[field] is not None:
                try:
                    if isinstance(item[field], (int, float, str)):
                        decimal_value = to_decimal(item[field])
                        item[field] = str(decimal_value)
                    elif isinstance(item[field], dict):
                        # Handle nested dictionaries (e.g., feature importance)
                        for k, v in item[field].items():
                            if v is not None:
                                try:
                                    decimal_value = to_decimal(v)
                                    item[field][k] = str(decimal_value)
                                except (ValueError, TypeError):
                                    continue
                except (ValueError, TypeError):
                    logger.warning(f"Failed to convert ML field {field} to Decimal: {item[field]}")
                    continue

    @staticmethod
    def validate_ml_to_utils_boundary(data: dict[str, Any]) -> dict[str, Any]:
        """
        Validate data flowing from ML to utils modules for boundary consistency.
        
        Args:
            data: ML data to validate at boundary
            
        Returns:
            Dict with validated data and boundary metadata
            
        Raises:
            ValidationError: If boundary validation fails
        """
        if not isinstance(data, dict):
            raise ValidationError(
                "ML to utils boundary data must be a dictionary",
                field_name="boundary_data",
                field_value=type(data).__name__,
                expected_type="dict",
            )

        # Required fields for ML to utils boundary
        required_fields = ["processing_mode", "message_pattern", "data_format"]
        for field in required_fields:
            if field not in data:
                # Set defaults for missing fields to maintain compatibility
                if field == "processing_mode":
                    data[field] = "stream"
                elif field == "message_pattern":
                    data[field] = "pub_sub"
                elif field == "data_format":
                    data[field] = "bot_event_v1"

        # Validate processing mode consistency
        valid_modes = ["stream", "batch", "request_reply"]
        if data["processing_mode"] not in valid_modes:
            raise ValidationError(
                f"Invalid processing_mode for ML to utils boundary: {data['processing_mode']}",
                field_name="processing_mode",
                field_value=data["processing_mode"],
                expected_type=f"one of {valid_modes}",
            )

        # Validate message pattern consistency
        valid_patterns = ["pub_sub", "req_reply", "stream", "batch"]
        if data["message_pattern"] not in valid_patterns:
            raise ValidationError(
                f"Invalid message_pattern for ML to utils boundary: {data['message_pattern']}",
                field_name="message_pattern",
                field_value=data["message_pattern"],
                expected_type=f"one of {valid_patterns}",
            )

        # Validate data format consistency
        if not data["data_format"].endswith("_v1"):
            raise ValidationError(
                f"Invalid data_format version for ML to utils boundary: {data['data_format']}",
                field_name="data_format",
                field_value=data["data_format"],
                expected_type="format ending with _v1",
            )

        # Apply ML financial validation
        ml_financial_fields = ["prediction_value", "confidence_threshold", "risk_score", "model_score"]
        for field in ml_financial_fields:
            if field in data and data[field] is not None:
                try:
                    value = float(data[field])
                    if field in ["confidence_threshold", "risk_score"] and not (0 <= value <= 1):
                        raise ValidationError(
                            f"ML field {field} must be between 0 and 1 at boundary",
                            field_name=field,
                            field_value=value,
                            expected_type="float between 0 and 1",
                        )
                except (ValueError, TypeError):
                    raise ValidationError(
                        f"ML field {field} must be numeric at boundary",
                        field_name=field,
                        field_value=data[field],
                        expected_type="numeric",
                    )

        # Add boundary validation metadata
        data.update({
            "ml_boundary_validation": "passed",
            "boundary_crossed": True,
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "ml_to_utils_boundary": True,
        })

        return data

    @staticmethod
    def validate_utils_to_ml_boundary(data: dict[str, Any]) -> dict[str, Any]:
        """
        Validate data flowing from utils to ML modules for boundary consistency.
        
        Args:
            data: Utils data to validate at ML boundary
            
        Returns:
            Dict with validated data and boundary metadata
            
        Raises:
            ValidationError: If boundary validation fails
        """
        if not isinstance(data, dict):
            raise ValidationError(
                "Utils to ML boundary data must be a dictionary",
                field_name="boundary_data",
                field_value=type(data).__name__,
                expected_type="dict",
            )

        # Check required fields for utils to ML flow
        required_fields = ["processing_mode", "data_format"]
        for field in required_fields:
            if field not in data:
                # Set defaults for missing fields
                if field == "processing_mode":
                    data[field] = "stream"
                elif field == "data_format":
                    data[field] = "bot_event_v1"

        # Validate ML-specific data structure requirements
        if "market_data" in data and isinstance(data["market_data"], dict):
            market_data = data["market_data"]
            ml_required_fields = ["symbol"]
            for ml_field in ml_required_fields:
                if ml_field not in market_data:
                    raise ValidationError(
                        f"Required ML field '{ml_field}' missing in market_data at boundary",
                        field_name=f"market_data.{ml_field}",
                        field_value=None,
                        expected_type="string",
                    )

        # Validate financial precision for ML processing
        if "price" in data and data["price"] is not None:
            try:
                # Convert to decimal for ML processing precision
                decimal_price = to_decimal(data["price"])
                data["price"] = str(decimal_price)
            except (ValueError, TypeError):
                raise ValidationError(
                    "Price must be convertible to Decimal for ML processing",
                    field_name="price",
                    field_value=data["price"],
                    expected_type="decimal-convertible",
                )

        # Add boundary validation metadata
        data.update({
            "utils_to_ml_boundary": True,
            "boundary_crossed": True,
            "ml_processing_ready": True,
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return data

    @classmethod
    def create_ml_error_propagation_mixin(cls) -> "MLErrorPropagationMixin":
        """Create ML-specific error propagation mixin for consistency with utils patterns."""
        return MLErrorPropagationMixin()


class MLErrorPropagationMixin:
    """ML-specific error propagation mixin aligned with utils messaging patterns."""

    def propagate_ml_validation_error(self, error: Exception, context: str) -> None:
        """Propagate ML validation errors consistently with utils patterns."""
        # Apply consistent error propagation metadata aligned with utils
        error_metadata = {
            "error_type": type(error).__name__,
            "context": context,
            "propagation_pattern": "ml_validation_direct",
            "data_format": "error_propagation_v1",
            "processing_mode": "stream",  # Align with utils default processing mode
            "message_pattern": "pub_sub",
            "boundary_crossed": True,
            "validation_status": "failed",
            "ml_component": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.error(f"ML validation error in {context}: {error}", extra=error_metadata)

        # Add propagation metadata to error if supported
        if hasattr(error, "__dict__"):
            try:
                error.__dict__.update({
                    "propagation_metadata": error_metadata,
                    "ml_boundary_validation_applied": True,
                })
            except (AttributeError, TypeError):
                pass

        raise error

    def propagate_ml_model_error(self, error: Exception, context: str) -> None:
        """Propagate ML model errors consistently with utils patterns."""
        error_metadata = {
            "error_type": type(error).__name__,
            "context": context,
            "propagation_pattern": "ml_model_to_service",
            "data_format": "error_propagation_v1",
            "processing_mode": "stream",  # Align with utils patterns
            "message_pattern": "pub_sub",
            "boundary_crossed": True,
            "ml_component": True,
            "component": "ml_model",
            "severity": "high",  # ML model errors are high severity
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.error(f"ML model error in {context}: {error}", extra=error_metadata)

        raise ModelError(
            f"ML model operation failed in {context}: {error}",
            details={
                "original_error": str(error),
                "error_context": context,
                "propagation_source": "ml_model",
                "data_format": "error_propagation_v1",
                "processing_mode": "stream",
                "message_pattern": "pub_sub",
                "ml_model_error": True,
                "utils_compatible": True,
                "error_metadata": error_metadata,
            },
        ) from error

    def propagate_ml_training_error(self, error: Exception, context: str) -> None:
        """Propagate ML training errors consistently with utils patterns."""
        error_metadata = {
            "error_type": type(error).__name__,
            "context": context,
            "propagation_pattern": "ml_training_to_service",
            "data_format": "error_propagation_v1",
            "processing_mode": "batch",  # Training uses batch processing
            "message_pattern": "batch",
            "boundary_crossed": True,
            "ml_component": True,
            "component": "ml_training",
            "severity": "medium",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.error(f"ML training error in {context}: {error}", extra=error_metadata)

        from src.core.exceptions import ServiceError

        raise ServiceError(
            f"ML training operation failed in {context}: {error}",
            details={
                "original_error": str(error),
                "error_context": context,
                "propagation_source": "ml_training",
                "data_format": "error_propagation_v1",
                "processing_mode": "batch",
                "message_pattern": "batch",
                "ml_training_error": True,
                "utils_compatible": True,
                "error_metadata": error_metadata,
            },
        ) from error

    def propagate_ml_inference_error(self, error: Exception, context: str) -> None:
        """Propagate ML inference errors consistently with utils patterns."""
        error_metadata = {
            "error_type": type(error).__name__,
            "context": context,
            "propagation_pattern": "ml_inference_to_service",
            "data_format": "error_propagation_v1",
            "processing_mode": "request_reply",  # Inference uses request_reply
            "message_pattern": "req_reply",
            "boundary_crossed": True,
            "ml_component": True,
            "component": "ml_inference",
            "severity": "high",  # Inference errors are high priority for trading
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.error(f"ML inference error in {context}: {error}", extra=error_metadata)

        from src.core.exceptions import ServiceError

        raise ServiceError(
            f"ML inference operation failed in {context}: {error}",
            details={
                "original_error": str(error),
                "error_context": context,
                "propagation_source": "ml_inference",
                "data_format": "error_propagation_v1",
                "processing_mode": "request_reply",
                "message_pattern": "req_reply",
                "ml_inference_error": True,
                "response_required": True,  # Inference requires response
                "utils_compatible": True,
                "error_metadata": error_metadata,
            },
        ) from error