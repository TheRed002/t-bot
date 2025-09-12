"""
ML Validation Service - Business logic validation for ML operations.

This service handles ML-specific business rule validation that was moved from
the data transformer to maintain proper separation of concerns.
"""

from typing import Any

from src.core.base.service import BaseService
from src.core.exceptions import ValidationError
from src.core.types.base import ConfigDict


class MLValidationService(BaseService):
    """Service for ML-specific business logic validation."""

    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None):
        super().__init__(
            name="MLValidationService",
            config=config,
            correlation_id=correlation_id,
        )

    def validate_ml_operation_type(self, ml_operation_type: str) -> bool:
        """
        Validate ML operation type according to business rules.
        
        Args:
            ml_operation_type: The ML operation type to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If ML operation type is invalid
        """
        valid_ml_operations = [
            "inference_request", "training_request", "feature_engineering",
            "model_registration", "batch_prediction", "pipeline_processing"
        ]
        
        if ml_operation_type not in valid_ml_operations:
            raise ValidationError(
                f"Invalid ml_operation_type: {ml_operation_type}. Must be one of {valid_ml_operations}",
                field_name="ml_operation_type",
                field_value=ml_operation_type,
                expected_type="string"
            )
        
        return True

    def validate_ml_request_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Validate ML request data according to business rules.
        
        Args:
            data: ML request data to validate
            
        Returns:
            Dict with validated ML request data
            
        Raises:
            ValidationError: If required ML fields are missing or invalid
        """
        # Check required fields based on business rules
        ml_required_fields = ["ml_operation_type"]
        
        for field in ml_required_fields:
            if field not in data:
                raise ValidationError(
                    f"Missing required ML business field: {field}",
                    field_name=field,
                    field_value=None,
                    expected_type="string"
                )
        
        # Validate operation type using business rules
        self.validate_ml_operation_type(data["ml_operation_type"])
        
        # Add validation metadata
        validated_data = data.copy()
        validated_data["ml_business_validation"] = "applied"
        
        return validated_data

    def validate_model_parameters(self, model_type: str, parameters: dict[str, Any]) -> bool:
        """
        Validate model parameters according to business rules.
        
        Args:
            model_type: Type of the model
            parameters: Model parameters to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If model parameters are invalid
        """
        # Basic model type validation
        valid_model_types = [
            "random_forest", "logistic_regression", "linear_regression",
            "gradient_boosting", "neural_network", "svm"
        ]
        
        if model_type not in valid_model_types:
            raise ValidationError(
                f"Invalid model_type: {model_type}. Must be one of {valid_model_types}",
                field_name="model_type",
                field_value=model_type,
                expected_type="string"
            )
        
        # Validate parameter types based on model type
        if model_type in ["random_forest", "gradient_boosting"]:
            if "n_estimators" in parameters and not isinstance(parameters["n_estimators"], int):
                raise ValidationError(
                    "n_estimators must be an integer",
                    field_name="n_estimators",
                    field_value=parameters["n_estimators"],
                    expected_type="integer"
                )
        
        return True

    def validate_feature_data(self, feature_data: Any) -> bool:
        """
        Validate feature data according to ML business rules.
        
        Args:
            feature_data: Feature data to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If feature data is invalid
        """
        if feature_data is None:
            raise ValidationError(
                "Feature data cannot be None",
                field_name="feature_data",
                field_value=None,
                expected_type="pandas.DataFrame or dict"
            )
        
        # Additional feature-specific business rules can be added here
        return True