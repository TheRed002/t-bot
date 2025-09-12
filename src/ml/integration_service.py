"""
ML Integration Service - Handle ML module integration with other modules.

This service handles business logic for ML module integration with other modules,
ensuring proper processing modes and data flow patterns based on target module requirements.
"""

from typing import Any

from src.core.base.service import BaseService
from src.core.types.base import ConfigDict
from src.ml.data_transformer import MLDataTransformer


class MLIntegrationService(BaseService):
    """Service for handling ML module integration with other modules."""

    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None):
        super().__init__(
            name="MLIntegrationService",
            config=config,
            correlation_id=correlation_id,
        )

    def determine_target_processing_mode(self, target_module: str, operation_type: str) -> str:
        """
        Determine appropriate processing mode for target module based on business rules.
        
        Args:
            target_module: Target module name
            operation_type: Type of ML operation
            
        Returns:
            Processing mode for target module
        """
        # Business logic for determining processing modes based on target module requirements
        if target_module == "analytics":
            # Analytics typically expects batch processing for ML results
            if operation_type in ["batch_prediction", "training_result"]:
                return "batch"
            else:
                return "stream"
        elif target_module == "execution":
            # Execution expects stream processing for real-time ML signals
            return "stream"
        elif target_module == "risk_management":
            # Risk management expects stream processing for real-time ML risk assessments
            return "stream"
        elif target_module == "strategies":
            # Strategies expect stream processing for real-time ML enhancements
            return "stream"
        else:
            # Default to stream for unknown modules
            return "stream"

    def prepare_data_for_target_module(
        self, 
        data: dict[str, Any], 
        target_module: str,
        operation_type: str | None = None
    ) -> dict[str, Any]:
        """
        Prepare ML data for sending to target module with appropriate processing mode.
        
        Args:
            data: ML data to prepare
            target_module: Target module name
            operation_type: Type of ML operation
            
        Returns:
            Dict prepared for target module
        """
        # Apply cross-module consistency using data transformer (structure only)
        consistent_data = MLDataTransformer.apply_cross_module_consistency_from_ml(
            data, target_module
        )
        
        # Apply business logic decisions for processing mode
        if operation_type:
            target_processing_mode = self.determine_target_processing_mode(target_module, operation_type)
            
            # Apply processing mode alignment based on business rules
            aligned_data = MLDataTransformer.align_with_core_processing_paradigm(
                consistent_data, target_processing_mode
            )
        else:
            aligned_data = consistent_data
        
        # Add integration metadata based on business requirements
        aligned_data.update({
            "ml_integration_prepared": True,
            "target_module": target_module,
            "integration_mode": self.determine_integration_mode(target_module),
        })
        
        return aligned_data

    def determine_integration_mode(self, target_module: str) -> str:
        """
        Determine integration mode based on target module requirements.
        
        Args:
            target_module: Target module name
            
        Returns:
            Integration mode string
        """
        # Business logic for integration modes
        integration_modes = {
            "analytics": "batch_analysis",
            "execution": "real_time_signal",
            "risk_management": "real_time_assessment",
            "strategies": "signal_enhancement",
        }
        
        return integration_modes.get(target_module, "default_integration")

    def validate_cross_module_compatibility(
        self, 
        data: dict[str, Any], 
        target_module: str
    ) -> bool:
        """
        Validate that ML data is compatible with target module requirements.
        
        Args:
            data: ML data to validate
            target_module: Target module name
            
        Returns:
            True if compatible, False otherwise
        """
        # Business rules for cross-module compatibility
        required_fields = {
            "analytics": ["ml_operation_type", "data"],
            "execution": ["ml_operation_type", "data", "timestamp"],
            "risk_management": ["ml_operation_type", "data", "risk_assessment"],
            "strategies": ["ml_operation_type", "data"],
        }
        
        target_required_fields = required_fields.get(target_module, ["ml_operation_type", "data"])
        
        for field in target_required_fields:
            if field not in data:
                self._logger.warning(
                    f"Missing required field for {target_module}: {field}",
                    target_module=target_module,
                    missing_field=field,
                    available_fields=list(data.keys())
                )
                return False
        
        return True