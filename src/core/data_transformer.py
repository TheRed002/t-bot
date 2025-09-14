"""
Core data transformation utilities for consistent data flow patterns.

This module provides standardized data transformation patterns that align
with execution module patterns to ensure consistency across the trading system.
"""

from datetime import datetime, timezone
from typing import Any

from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.utils.decimal_utils import to_decimal

# Import BaseModel from pydantic via base types
try:
    from pydantic import BaseModel
except ImportError:
    # Fallback for testing/development
    BaseModel = object

logger = get_logger(__name__)


class CoreDataTransformer:
    """Handles consistent data transformation for core module events and messaging."""

    @staticmethod
    def transform_event_to_standard_format(
        event_type: str, 
        data: Any, 
        metadata: dict[str, Any] | None = None,
        source: str = "core"
    ) -> dict[str, Any]:
        """
        Transform event data to standard format aligned with execution module patterns.
        
        Args:
            event_type: Type of event
            data: Event data to transform
            metadata: Additional metadata
            source: Source module name
            
        Returns:
            Dict with standardized event format
        """
        transformed = {
            "event_type": event_type,
            "processing_mode": "stream",  # Default to stream processing
            "data_format": "core_event_data_v1",
            "message_pattern": "pub_sub",  # Align with execution module default
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "boundary_crossed": True,
            "validation_status": "validated",
            "metadata": metadata or {},
        }
        
        # Handle different data types
        if isinstance(data, BaseModel):
            transformed["data"] = data.model_dump()
        elif isinstance(data, dict):
            transformed["data"] = data.copy()
        elif isinstance(data, Exception):
            transformed["data"] = {
                "error_type": type(data).__name__,
                "error_message": str(data),
                "error_context": getattr(data, "__dict__", {}),
            }
        else:
            transformed["data"] = {
                "payload": str(data),
                "type": type(data).__name__,
            }
            
        return CoreDataTransformer._apply_financial_precision(transformed)

    @staticmethod
    def transform_for_pub_sub_pattern(
        event_type: str,
        data: Any,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform data for pub/sub messaging pattern consistent with execution module.
        
        Args:
            event_type: Type of event
            data: Raw data to transform
            metadata: Additional metadata
            
        Returns:
            Dict formatted for pub/sub pattern
        """
        transformed = CoreDataTransformer.transform_event_to_standard_format(
            event_type, data, metadata
        )
        
        # Add pub/sub specific fields
        transformed.update({
            "message_pattern": "pub_sub",
            "distribution_mode": "broadcast",
            "acknowledgment_required": False,
        })
        
        return transformed

    @staticmethod
    def transform_for_request_reply_pattern(
        request_type: str,
        data: Any,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Transform data for request/reply messaging pattern.
        
        Args:
            request_type: Type of request
            data: Request data
            correlation_id: Request correlation ID
            metadata: Additional metadata
            
        Returns:
            Dict formatted for request/reply pattern
        """
        transformed = CoreDataTransformer.transform_event_to_standard_format(
            request_type, data, metadata
        )
        
        # Override with request/reply specific fields
        transformed.update({
            "message_pattern": "req_reply",
            "processing_mode": "request_reply",
            "correlation_id": correlation_id or datetime.now(timezone.utc).isoformat(),
            "response_required": True,
            "acknowledgment_required": True,
        })
        
        return transformed

    @staticmethod
    def align_processing_paradigm(
        data: dict[str, Any], 
        target_mode: str = "stream"
    ) -> dict[str, Any]:
        """
        Align data processing paradigm for consistency across modules.
        
        Args:
            data: Data to align
            target_mode: Target processing mode ("stream", "batch", "request_reply")
            
        Returns:
            Dict with aligned processing mode
        """
        aligned_data = data.copy()
        aligned_data["processing_mode"] = target_mode
        
        # Add mode-specific fields
        if target_mode == "stream":
            aligned_data.update({
                "stream_position": datetime.now(timezone.utc).timestamp(),
                "data_format": aligned_data.get("data_format", "core_event_data_v1"),
                "message_pattern": "pub_sub",
                "real_time_processing": True,
            })
        elif target_mode == "batch":
            if "batch_id" not in aligned_data:
                aligned_data["batch_id"] = datetime.now(timezone.utc).isoformat()
            aligned_data.update({
                "data_format": "batch_event_data_v1",
                "message_pattern": "batch",
                "batch_processing": True,
            })
        elif target_mode == "request_reply":
            if "correlation_id" not in aligned_data:
                aligned_data["correlation_id"] = datetime.now(timezone.utc).isoformat()
            aligned_data.update({
                "data_format": "request_reply_data_v1",
                "message_pattern": "req_reply",
                "synchronous_processing": True,
            })
            
        return aligned_data

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
            
        # Validate message pattern values
        valid_patterns = ["pub_sub", "req_reply", "batch", "stream"]
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

    @staticmethod
    def _apply_financial_precision(data: dict[str, Any]) -> dict[str, Any]:
        """
        Ensure financial data has proper precision using Decimal for consistency.
        
        Args:
            data: Data dictionary to process
            
        Returns:
            Dict with validated financial precision
        """
        if "data" not in data:
            return data
            
        financial_fields = [
            "price", "quantity", "volume", "fees", "amount", 
            "target_quantity", "filled_quantity", "value",
            "balance", "available_balance", "total"
        ]
        
        data_dict = data["data"]
        if not isinstance(data_dict, dict):
            return data
            
        for field in financial_fields:
            if field in data_dict and data_dict[field] is not None:
                try:
                    # Convert to Decimal for financial precision
                    decimal_value = to_decimal(data_dict[field])
                    # Convert back to string for consistent serialization
                    data_dict[field] = str(decimal_value)
                except (ValueError, TypeError):
                    # Keep original value if conversion fails
                    logger.warning(f"Failed to convert {field} to Decimal: {data_dict[field]}")
                    
        # Add financial precision metadata
        if any(field in data_dict for field in financial_fields):
            data["financial_precision_applied"] = True
            data["decimal_precision"] = "20,8"
            
        return data

    @classmethod
    def apply_cross_module_consistency(
        cls,
        data: dict[str, Any],
        target_module: str,
        source_module: str = "core"
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
        
        # Apply boundary validation
        consistent_data = cls.validate_boundary_fields(consistent_data)
        
        # Apply financial precision
        consistent_data = cls._apply_financial_precision(consistent_data)
        
        # Add cross-module metadata
        consistent_data.update({
            "cross_module_consistency": True,
            "source_module": source_module,
            "target_module": target_module,
            "boundary_alignment_timestamp": datetime.now(timezone.utc).isoformat(),
            "data_flow_aligned": True,
        })
        
        # Apply target-specific processing paradigm and boundary validation
        if target_module == "execution":
            # Execution module expects stream processing by default
            consistent_data = cls.align_processing_paradigm(consistent_data, "stream")
            cls._apply_boundary_validation(consistent_data, source_module, target_module)
        elif target_module in ["analytics", "backtesting"]:
            # Analytics modules often use batch processing
            consistent_data = cls.align_processing_paradigm(consistent_data, "batch")
        elif target_module in ["risk_management", "state"]:
            # Risk and state modules use stream processing for real-time
            consistent_data = cls.align_processing_paradigm(consistent_data, "stream")
            cls._apply_boundary_validation(consistent_data, source_module, target_module)
            
        return consistent_data

    @staticmethod
    def _apply_boundary_validation(
        data: dict[str, Any], 
        source_module: str, 
        target_module: str
    ) -> None:
        """
        Apply boundary validation using the same patterns as execution module.
        
        Args:
            data: Data to validate
            source_module: Source module name
            target_module: Target module name
        """
        try:
            # Use the same boundary validation utilities as execution module
            from src.utils.messaging_patterns import BoundaryValidator
            
            if target_module == "execution":
                # Validate core -> execution boundary
                boundary_data = {
                    "component": source_module,
                    "error_type": data.get("event_type", "CoreEvent"),
                    "severity": data.get("severity", "medium"),
                    "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    "processing_mode": data.get("processing_mode", "stream"),
                    "data_format": data.get("data_format", "core_event_data_v1"),
                    "message_pattern": data.get("message_pattern", "pub_sub"),
                    "boundary_crossed": True,
                }
                BoundaryValidator.validate_error_to_monitoring_boundary(boundary_data)
                
            elif target_module == "risk_management":
                # Validate core -> risk_management boundary
                boundary_data = {
                    "component": source_module,
                    "operation": data.get("event_type", "core_operation"),
                    "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    "processing_mode": data.get("processing_mode", "stream"),
                    "data_format": data.get("data_format", "core_event_data_v1"),
                    "boundary_crossed": True,
                }
                BoundaryValidator.validate_risk_to_state_boundary(boundary_data)
                
        except Exception as e:
            # Log validation issues but don't fail the data transformation
            logger.debug(f"Boundary validation failed for {source_module} -> {target_module}: {e}")