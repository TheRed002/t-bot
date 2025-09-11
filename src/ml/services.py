"""
Mock ML Service Implementations.

This module provides mock implementations of ML services that are referenced
but not yet implemented, to maintain service layer integrity.
"""

from typing import Any

import pandas as pd

from src.core.base.service import BaseService
from src.core.types.base import ConfigDict
from src.utils.constants import ML_MODEL_CONSTANTS
from src.ml.interfaces import (
    IBatchPredictionService,
    IDriftDetectionService,
    IModelValidationService,
    ITrainingService,
)


class ModelValidationService(BaseService, IModelValidationService):
    """Mock model validation service."""

    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None):
        super().__init__(
            name="ModelValidationService",
            config=config,
            correlation_id=correlation_id,
        )

    async def validate_model_performance(
        self, model: Any, test_data: tuple[pd.DataFrame, pd.Series]
    ) -> dict[str, Any]:
        """Mock validation of model performance."""
        X_test, y_test = test_data

        # Basic mock validation
        try:
            predictions = model.predict(X_test)
            return {
                "validation_passed": True,
                "sample_predictions": len(predictions),
                "test_samples": len(y_test),
                "prediction_shape": getattr(predictions, "shape", "unknown"),
            }
        except Exception as e:
            return {
                "validation_passed": False,
                "error": str(e),
            }

    async def validate_production_readiness(
        self, model: Any, validation_data: tuple[pd.DataFrame, pd.Series]
    ) -> dict[str, bool]:
        """Mock production readiness validation."""
        return {
            "has_predict_method": hasattr(model, "predict"),
            "has_model_attributes": hasattr(model, "model_name") or hasattr(model, "__class__"),
            "validation_data_available": validation_data is not None,
            "overall_pass": hasattr(model, "predict"),
        }


class DriftDetectionService(BaseService, IDriftDetectionService):
    """Mock drift detection service."""

    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None):
        super().__init__(
            name="DriftDetectionService",
            config=config,
            correlation_id=correlation_id,
        )
        self.reference_data: dict[str, Any] = {}

    async def get_reference_data(self, data_type: str) -> Any:
        """Get reference data for drift detection."""
        return self.reference_data.get(data_type)

    async def set_reference_data(self, data: pd.DataFrame, data_type: str) -> None:
        """Set reference data for drift detection."""
        self.reference_data[data_type] = data

    async def detect_feature_drift(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> dict[str, Any]:
        """Mock feature drift detection."""
        return {
            "drift_detected": False,
            "drift_score": ML_MODEL_CONSTANTS["default_drift_score"],
            "reference_samples": len(reference_data),
            "current_samples": len(current_data),
            "method": "mock_detection",
        }

    async def detect_prediction_drift(
        self, reference_predictions: pd.Series, current_predictions: pd.Series, model_name: str
    ) -> dict[str, Any]:
        """Mock prediction drift detection."""
        return {
            "drift_detected": False,
            "drift_score": ML_MODEL_CONSTANTS["default_prediction_drift_score"],
            "model_name": model_name,
            "reference_samples": len(reference_predictions),
            "current_samples": len(current_predictions),
        }

    async def detect_performance_drift(
        self,
        reference_metrics: dict[str, float],
        current_metrics: dict[str, float],
        model_name: str,
    ) -> dict[str, Any]:
        """Mock performance drift detection."""
        return {
            "drift_detected": False,
            "performance_decline": ML_MODEL_CONSTANTS["default_performance_decline"],
            "model_name": model_name,
            "reference_metrics": reference_metrics,
            "current_metrics": current_metrics,
        }


class TrainingService(BaseService, ITrainingService):
    """Mock training service."""

    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None):
        super().__init__(
            name="TrainingService",
            config=config,
            correlation_id=correlation_id,
        )

    async def train_model(
        self,
        model: Any,
        training_data: pd.DataFrame,
        symbol: str,
        training_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Mock model training."""
        return {
            "training_time": ML_MODEL_CONSTANTS["default_training_time_seconds"],
            "training_samples": len(training_data),
            "model_name": getattr(model, "model_name", "unknown"),
            "symbol": symbol,
            "status": "completed",
        }

    async def save_artifacts(
        self,
        model: Any,
        training_result: dict[str, Any],
        validation_result: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Mock artifact saving."""
        return {
            "artifacts_saved": True,
            "model_file": f"model_{id(model)}.pkl",
            "training_data_file": "training_data.csv",
            "metadata_file": "metadata.json",
        }


class BatchPredictionService(BaseService, IBatchPredictionService):
    """Mock batch prediction service."""

    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None):
        super().__init__(
            name="BatchPredictionService",
            config=config,
            correlation_id=correlation_id,
        )

    async def process_batch_predictions(
        self, requests: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Mock batch prediction processing."""
        return [
            {
                "request_id": req.get("request_id", f"batch_{i}"),
                "predictions": ML_MODEL_CONSTANTS["mock_batch_predictions"],
                "status": "completed",
            }
            for i, req in enumerate(requests)
        ]
