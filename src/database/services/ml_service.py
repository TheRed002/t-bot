"""ML service layer implementing business logic for machine learning operations."""

from typing import Any

from src.core.base.service import BaseService
from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.database.interfaces import MLServiceInterface
from src.database.repository.ml import (
    MLModelMetadataRepository,
    MLPredictionRepository,
    MLTrainingJobRepository,
)

logger = get_logger(__name__)


class MLService(BaseService, MLServiceInterface):
    """Service layer for ML operations with business logic."""

    def __init__(
        self,
        prediction_repo: MLPredictionRepository,
        model_repo: MLModelMetadataRepository,
        training_repo: MLTrainingJobRepository,
    ):
        """Initialize with injected repositories."""
        super().__init__(name="MLService")
        self.prediction_repo = prediction_repo
        self.model_repo = model_repo
        self.training_repo = training_repo

    async def get_model_performance_summary(self, model_name: str, days: int = 30) -> dict[str, Any]:
        """
        Get comprehensive performance summary for a model.

        This is business logic that aggregates data from multiple repositories.

        Args:
            model_name: Name of the model
            days: Number of days to analyze

        Returns:
            Dictionary with performance summary

        Raises:
            ValidationError: If model_name is invalid
            ServiceError: If operation fails
        """
        try:
            if not model_name or not model_name.strip():
                raise ValidationError("Model name is required")

            if days <= 0:
                raise ValidationError("Days must be positive")

            # Business logic: Aggregate data from multiple repositories
            accuracy = await self.prediction_repo.get_prediction_accuracy(model_name, days=days)
            latest_model = await self.model_repo.get_latest_model(model_name, "prediction")
            recent_jobs = await self.training_repo.get_job_by_model(model_name, status="completed")

            # Business logic: Format and aggregate the response
            return {
                "model_name": model_name,
                "accuracy_metrics": accuracy,
                "latest_version": latest_model.version if latest_model else None,
                "model_parameters": latest_model.parameters if latest_model else {},
                "recent_training_jobs": len(recent_jobs),
                "last_training": recent_jobs[0].completed_at if recent_jobs else None,
            }

        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Failed to get model performance summary for {model_name}: {e}")
            raise ServiceError(f"Model performance summary failed: {e}")

    async def validate_model_deployment(self, model_name: str, version: int) -> bool:
        """
        Validate if a model version is ready for deployment.

        Business logic for model deployment validation.

        Args:
            model_name: Name of the model
            version: Model version

        Returns:
            True if model is ready for deployment
        """
        try:
            # Business logic: Check model exists and is trained
            model = await self.model_repo.get_by_version(model_name, version)
            if not model:
                raise ValidationError(f"Model {model_name} version {version} not found")

            # Business logic: Check if model has completed training
            training_jobs = await self.training_repo.get_job_by_model(model_name, status="completed")
            if not training_jobs:
                logger.warning(f"No completed training jobs for model {model_name}")
                return False

            # Business logic: Check prediction accuracy meets threshold
            accuracy = await self.prediction_repo.get_prediction_accuracy(model_name, days=7)
            if accuracy and accuracy.get("overall_accuracy", 0) < 0.7:
                logger.warning(f"Model {model_name} accuracy below threshold")
                return False

            return True

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to validate model deployment for {model_name}: {e}")
            raise ServiceError(f"Model validation failed: {e}")

    async def get_model_recommendations(self, symbol: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Get ML model recommendations for a trading symbol.

        Business logic for model recommendations.

        Args:
            symbol: Trading symbol
            limit: Maximum recommendations to return

        Returns:
            List of model recommendations
        """
        try:
            if not symbol or not symbol.strip():
                raise ValidationError("Symbol is required")

            # Business logic: Get recent predictions for symbol
            active_models = await self.model_repo.get_active_models()
            recommendations = []

            for model in active_models:
                predictions = await self.prediction_repo.get_by_model_and_symbol(model.model_name, symbol, limit=10)

                if predictions:
                    # Business logic: Calculate confidence and trend
                    avg_confidence = sum(p.confidence_score for p in predictions) / len(predictions)
                    latest_prediction = predictions[0]

                    recommendations.append(
                        {
                            "model_name": model.model_name,
                            "model_type": model.model_type,
                            "latest_prediction": latest_prediction.predicted_value,
                            "confidence": avg_confidence,
                            "timestamp": latest_prediction.timestamp,
                            "symbol": symbol,
                        }
                    )

            # Business logic: Sort by confidence and limit results
            recommendations.sort(key=lambda x: x["confidence"], reverse=True)
            return recommendations[:limit]

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to get model recommendations for {symbol}: {e}")
            raise ServiceError(f"Model recommendations failed: {e}")
