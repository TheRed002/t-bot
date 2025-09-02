"""
ML Service Layer Interfaces.

This module defines the interfaces for all ML services to ensure proper
service layer separation and dependency management.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from src.ml.feature_engineering import FeatureRequest, FeatureResponse
    from src.ml.inference.inference_engine import (
        InferencePredictionRequest,
        InferencePredictionResponse,
    )
    from src.ml.registry.model_registry import (
        ModelLoadRequest,
        ModelRegistrationRequest,
    )


class IFeatureEngineeringService(ABC):
    """Interface for feature engineering service."""

    @abstractmethod
    async def compute_features(self, request: "FeatureRequest") -> "FeatureResponse":
        """Compute features from market data."""
        pass

    @abstractmethod
    async def select_features(
        self,
        features_df: pd.DataFrame,
        target_series: pd.Series,
        method: str = "mutual_info",
        max_features: int | None = None,
        percentile: float | None = None,
    ) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
        """Select the most important features."""
        pass

    @abstractmethod
    async def clear_cache(self) -> dict[str, int]:
        """Clear feature engineering cache."""
        pass


class IModelRegistryService(ABC):
    """Interface for model registry service."""

    @abstractmethod
    async def register_model(self, request: "ModelRegistrationRequest") -> str:
        """Register a new model version in the registry."""
        pass

    @abstractmethod
    async def load_model(self, request: "ModelLoadRequest") -> dict[str, Any]:
        """Load a model from the registry."""
        pass

    @abstractmethod
    async def list_models(
        self,
        model_type: str | None = None,
        stage: str | None = None,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """List all models in the registry."""
        pass

    @abstractmethod
    async def promote_model(self, model_id: str, stage: str, description: str = "") -> bool:
        """Promote a model to a different stage."""
        pass

    @abstractmethod
    async def deactivate_model(self, model_id: str, reason: str = "") -> bool:
        """Deactivate a model in the registry."""
        pass

    @abstractmethod
    async def delete_model(self, model_id: str, remove_files: bool = True) -> bool:
        """Delete a model from the registry."""
        pass

    @abstractmethod
    async def get_model_metrics(self, model_id: str) -> dict[str, Any]:
        """Get detailed metrics for a model."""
        pass


class IInferenceService(ABC):
    """Interface for inference service."""

    @abstractmethod
    async def predict(
        self,
        model_id: str,
        features: pd.DataFrame,
        return_probabilities: bool = False,
        use_cache: bool = True,
        request_id: str | None = None,
    ) -> "InferencePredictionResponse":
        """Make a single prediction."""
        pass

    @abstractmethod
    async def predict_batch(
        self, requests: list["InferencePredictionRequest"]
    ) -> list["InferencePredictionResponse"]:
        """Process a batch of prediction requests."""
        pass

    @abstractmethod
    async def predict_with_features(
        self,
        model_id: str,
        market_data: pd.DataFrame,
        symbol: str,
        return_probabilities: bool = False,
    ) -> "InferencePredictionResponse":
        """Make prediction with automatic feature engineering."""
        pass

    @abstractmethod
    async def warm_up_models(self, model_ids: list[str]) -> dict[str, bool]:
        """Warm up models by loading them into cache."""
        pass

    @abstractmethod
    async def clear_cache(self) -> dict[str, int]:
        """Clear inference caches."""
        pass


class IModelValidationService(ABC):
    """Interface for model validation service."""

    @abstractmethod
    async def validate_model_performance(
        self, model: Any, test_data: tuple[pd.DataFrame, pd.Series]
    ) -> dict[str, Any]:
        """Validate model performance against test data."""
        pass

    @abstractmethod
    async def validate_production_readiness(
        self, model: Any, validation_data: tuple[pd.DataFrame, pd.Series]
    ) -> dict[str, bool]:
        """Validate model is ready for production deployment."""
        pass


class IDriftDetectionService(ABC):
    """Interface for drift detection service."""

    @abstractmethod
    async def detect_feature_drift(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> dict[str, Any]:
        """Detect feature drift between reference and current data."""
        pass

    @abstractmethod
    async def detect_prediction_drift(
        self, reference_predictions: pd.Series, current_predictions: pd.Series, model_name: str
    ) -> dict[str, Any]:
        """Detect prediction drift."""
        pass

    @abstractmethod
    async def detect_performance_drift(
        self,
        reference_metrics: dict[str, float],
        current_metrics: dict[str, float],
        model_name: str,
    ) -> dict[str, Any]:
        """Detect performance drift."""
        pass


class ITrainingService(ABC):
    """Interface for training service."""

    @abstractmethod
    async def train_model(
        self,
        model: Any,
        training_data: pd.DataFrame,
        symbol: str,
        training_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Train a model with the given data."""
        pass

    @abstractmethod
    async def save_artifacts(
        self,
        model: Any,
        training_result: dict[str, Any],
        validation_result: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Save training artifacts."""
        pass


class IBatchPredictionService(ABC):
    """Interface for batch prediction service."""

    @abstractmethod
    async def process_batch_predictions(
        self, requests: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process batch prediction requests."""
        pass


class IModelFactory(ABC):
    """Interface for model factory service."""

    @abstractmethod
    def create_model(
        self,
        model_type: str,
        model_name: str | None = None,
        version: str = "1.0.0",
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a model instance of the specified type."""
        pass

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """Get list of available model types."""
        pass

    @abstractmethod
    def register_custom_model(
        self,
        name: str,
        model_class: type,
        config: dict[str, Any] | None = None,
        singleton: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a custom model type."""
        pass
