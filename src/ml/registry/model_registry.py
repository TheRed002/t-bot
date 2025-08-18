"""
Model Registry for ML Model Versioning and Storage Management.

This module provides comprehensive model versioning, storage, and lifecycle management
for ML models with database integration and audit trails.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import desc

from src.core.config import Config
from src.core.exceptions import ModelError, ValidationError
from src.core.logging import get_logger
from src.database.connection import get_sync_session
from src.database.models import MLModel
from src.ml.models.base_model import BaseModel
from src.utils.decorators import log_calls, time_execution

logger = get_logger(__name__)


class ModelRegistry:
    """
    Model registry for managing ML model versions and storage.

    This class provides centralized management of ML models including versioning,
    storage, retrieval, and lifecycle management with database integration.

    Attributes:
        config: Application configuration
        registry_path: Path to the model registry directory
        artifact_path: Path to model artifacts directory
    """

    def __init__(self, config: Config):
        """
        Initialize the model registry.

        Args:
            config: Application configuration
        """
        self.config = config
        self.registry_path = Path(config.ml.model_registry_path)
        self.artifact_path = Path(config.ml.artifact_store_path)

        # Create directories if they don't exist
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.artifact_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Model registry initialized",
            registry_path=str(self.registry_path),
            artifact_path=str(self.artifact_path),
        )

    @time_execution
    @log_calls
    def register_model(
        self,
        model: BaseModel,
        description: str = "",
        tags: dict[str, str] | None = None,
        stage: str = "development",
    ) -> str:
        """
        Register a new model version in the registry.

        Args:
            model: The trained model to register
            description: Description of the model
            tags: Optional tags for the model
            stage: Model stage (development, staging, production)

        Returns:
            Model ID in the registry

        Raises:
            ModelError: If registration fails
            ValidationError: If model validation fails
        """
        if not model.is_trained:
            raise ValidationError("Cannot register untrained model")

        try:
            with get_sync_session() as session:
                # Check if model with same name exists
                existing_model = (
                    session.query(MLModel)
                    .filter(
                        MLModel.name == model.model_name, MLModel.model_type == model.model_type
                    )
                    .order_by(desc(MLModel.created_at))
                    .first()
                )

                # Generate new version
                if existing_model:
                    # Parse existing version and increment
                    version_parts = existing_model.version.split(".")
                    major, minor, patch = map(int, version_parts)
                    new_version = f"{major}.{minor}.{patch + 1}"
                else:
                    new_version = "1.0.0"

                # Create model file path
                model_filename = f"{model.model_name}_{model.model_type}_{new_version}.joblib"
                model_file_path = self.artifact_path / model_filename

                # Save model to file
                model.version = new_version
                saved_path = model.save(model_file_path)

                # Create database record
                db_model = MLModel(
                    name=model.model_name,
                    model_type=model.model_type,
                    version=new_version,
                    file_path=str(saved_path),
                    metrics=model.metrics,
                    parameters=model.metadata.get("hyperparameters", {}),
                    training_data_range=self._format_training_range(model),
                    is_active=True,
                )

                session.add(db_model)
                session.commit()

                # Create registry entry
                registry_entry = {
                    "model_id": db_model.id,
                    "name": model.model_name,
                    "type": model.model_type,
                    "version": new_version,
                    "description": description,
                    "tags": tags or {},
                    "stage": stage,
                    "file_path": str(saved_path),
                    "metrics": model.metrics,
                    "feature_count": len(model.feature_names),
                    "feature_names": model.feature_names,
                    "registered_at": datetime.utcnow().isoformat(),
                    "metadata": model.metadata,
                }

                # Save registry entry
                registry_file = self.registry_path / f"{db_model.id}.json"
                with open(registry_file, "w") as f:
                    json.dump(registry_entry, f, indent=2)

                logger.info(
                    "Model registered successfully",
                    model_id=db_model.id,
                    model_name=model.model_name,
                    version=new_version,
                    stage=stage,
                )

                return db_model.id

        except Exception as e:
            logger.error("Model registration failed", model_name=model.model_name, error=str(e))
            raise ModelError(f"Failed to register model {model.model_name}: {e}") from e

    @time_execution
    @log_calls
    def get_model(
        self,
        model_id: str | None = None,
        model_name: str | None = None,
        model_type: str | None = None,
        version: str | None = None,
        stage: str | None = None,
    ) -> BaseModel:
        """
        Retrieve a model from the registry.

        Args:
            model_id: Specific model ID to retrieve
            model_name: Model name to search for
            model_type: Model type to filter by
            version: Specific version to retrieve (latest if not specified)
            stage: Model stage to filter by

        Returns:
            The requested model instance

        Raises:
            ModelError: If model not found or loading fails
        """
        try:
            with get_sync_session() as session:
                query = session.query(MLModel)

                if model_id:
                    db_model = query.filter(MLModel.id == model_id).first()
                else:
                    # Build query based on provided parameters
                    if model_name:
                        query = query.filter(MLModel.name == model_name)
                    if model_type:
                        query = query.filter(MLModel.model_type == model_type)
                    if version:
                        query = query.filter(MLModel.version == version)

                    # Get the latest version if no specific version requested
                    query = query.filter(MLModel.is_active)
                    db_model = query.order_by(desc(MLModel.created_at)).first()

                if not db_model:
                    raise ModelError(
                        f"Model not found with criteria: id={model_id}, name={model_name}, "
                        f"type={model_type}, version={version}, stage={stage}"
                    )

                # Load model from file
                model_path = Path(db_model.file_path)
                if not model_path.exists():
                    raise ModelError(f"Model file not found: {model_path}")

                # Dynamically import the appropriate model class
                model_class = self._get_model_class(db_model.model_type)
                model = model_class.load(model_path, self.config)

                logger.info(
                    "Model retrieved successfully",
                    model_id=db_model.id,
                    model_name=db_model.name,
                    version=db_model.version,
                )

                return model

        except Exception as e:
            logger.error(
                "Model retrieval failed", model_id=model_id, model_name=model_name, error=str(e)
            )
            raise ModelError(f"Failed to retrieve model: {e}") from e

    @time_execution
    @log_calls
    def list_models(
        self, model_type: str | None = None, stage: str | None = None, active_only: bool = True
    ) -> pd.DataFrame:
        """
        List all models in the registry.

        Args:
            model_type: Filter by model type
            stage: Filter by stage (not yet implemented in DB)
            active_only: Only return active models

        Returns:
            DataFrame with model information
        """
        try:
            with get_sync_session() as session:
                query = session.query(MLModel)

                if model_type:
                    query = query.filter(MLModel.model_type == model_type)
                if active_only:
                    query = query.filter(MLModel.is_active)

                models = query.order_by(desc(MLModel.created_at)).all()

                # Convert to DataFrame
                model_data = []
                for model in models:
                    model_info = {
                        "model_id": model.id,
                        "name": model.name,
                        "type": model.model_type,
                        "version": model.version,
                        "is_active": model.is_active,
                        "created_at": model.created_at,
                        "updated_at": model.updated_at,
                        "file_path": model.file_path,
                        "training_data_range": model.training_data_range,
                    }

                    # Add metrics as separate columns
                    if model.metrics:
                        for key, value in model.metrics.items():
                            model_info[f"metric_{key}"] = value

                    model_data.append(model_info)

                df = pd.DataFrame(model_data)

                logger.info(
                    "Models listed successfully",
                    total_models=len(df),
                    model_type=model_type,
                    active_only=active_only,
                )

                return df

        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            raise ModelError(f"Failed to list models: {e}") from e

    @time_execution
    @log_calls
    def promote_model(self, model_id: str, stage: str, description: str = "") -> bool:
        """
        Promote a model to a different stage.

        Args:
            model_id: Model ID to promote
            stage: Target stage (staging, production)
            description: Promotion description

        Returns:
            True if promotion successful

        Raises:
            ModelError: If promotion fails
        """
        try:
            # Update registry entry
            registry_file = self.registry_path / f"{model_id}.json"
            if registry_file.exists():
                with open(registry_file) as f:
                    entry = json.load(f)

                entry["stage"] = stage
                entry["promotion_history"] = entry.get("promotion_history", [])
                entry["promotion_history"].append(
                    {
                        "stage": stage,
                        "description": description,
                        "promoted_at": datetime.utcnow().isoformat(),
                    }
                )

                with open(registry_file, "w") as f:
                    json.dump(entry, f, indent=2)

                logger.info("Model promoted successfully", model_id=model_id, stage=stage)

                return True
            else:
                raise ModelError(f"Model registry entry not found: {model_id}")

        except Exception as e:
            logger.error("Model promotion failed", model_id=model_id, stage=stage, error=str(e))
            raise ModelError(f"Failed to promote model {model_id}: {e}") from e

    @time_execution
    @log_calls
    def deactivate_model(self, model_id: str, reason: str = "") -> bool:
        """
        Deactivate a model in the registry.

        Args:
            model_id: Model ID to deactivate
            reason: Reason for deactivation

        Returns:
            True if deactivation successful

        Raises:
            ModelError: If deactivation fails
        """
        try:
            with get_sync_session() as session:
                db_model = session.query(MLModel).filter(MLModel.id == model_id).first()
                if not db_model:
                    raise ModelError(f"Model not found: {model_id}")

                db_model.is_active = False
                session.commit()

                # Update registry entry
                registry_file = self.registry_path / f"{model_id}.json"
                if registry_file.exists():
                    with open(registry_file) as f:
                        entry = json.load(f)

                    entry["is_active"] = False
                    entry["deactivation_history"] = entry.get("deactivation_history", [])
                    entry["deactivation_history"].append(
                        {"reason": reason, "deactivated_at": datetime.utcnow().isoformat()}
                    )

                    with open(registry_file, "w") as f:
                        json.dump(entry, f, indent=2)

                logger.info("Model deactivated successfully", model_id=model_id, reason=reason)

                return True

        except Exception as e:
            logger.error("Model deactivation failed", model_id=model_id, error=str(e))
            raise ModelError(f"Failed to deactivate model {model_id}: {e}") from e

    @time_execution
    @log_calls
    def delete_model(self, model_id: str, remove_files: bool = True) -> bool:
        """
        Delete a model from the registry.

        Args:
            model_id: Model ID to delete
            remove_files: Whether to remove model files

        Returns:
            True if deletion successful

        Raises:
            ModelError: If deletion fails
        """
        try:
            with get_sync_session() as session:
                db_model = session.query(MLModel).filter(MLModel.id == model_id).first()
                if not db_model:
                    raise ModelError(f"Model not found: {model_id}")

                # Remove files if requested
                if remove_files:
                    model_path = Path(db_model.file_path)
                    if model_path.exists():
                        model_path.unlink()

                    registry_file = self.registry_path / f"{model_id}.json"
                    if registry_file.exists():
                        registry_file.unlink()

                # Remove from database
                session.delete(db_model)
                session.commit()

                logger.info(
                    "Model deleted successfully", model_id=model_id, remove_files=remove_files
                )

                return True

        except Exception as e:
            logger.error("Model deletion failed", model_id=model_id, error=str(e))
            raise ModelError(f"Failed to delete model {model_id}: {e}") from e

    def get_model_metrics(self, model_id: str) -> dict[str, Any]:
        """
        Get detailed metrics for a model.

        Args:
            model_id: Model ID

        Returns:
            Dictionary with model metrics and information
        """
        try:
            with get_sync_session() as session:
                db_model = session.query(MLModel).filter(MLModel.id == model_id).first()
                if not db_model:
                    raise ModelError(f"Model not found: {model_id}")

                # Get registry entry for additional info
                registry_file = self.registry_path / f"{model_id}.json"
                registry_info = {}
                if registry_file.exists():
                    with open(registry_file) as f:
                        registry_info = json.load(f)

                metrics = {
                    "model_id": db_model.id,
                    "name": db_model.name,
                    "type": db_model.model_type,
                    "version": db_model.version,
                    "is_active": db_model.is_active,
                    "created_at": db_model.created_at.isoformat(),
                    "updated_at": db_model.updated_at.isoformat(),
                    "metrics": db_model.metrics,
                    "parameters": db_model.parameters,
                    "training_data_range": db_model.training_data_range,
                    "registry_info": registry_info,
                }

                return metrics

        except Exception as e:
            logger.error("Failed to get model metrics", model_id=model_id, error=str(e))
            raise ModelError(f"Failed to get metrics for model {model_id}: {e}") from e

    def _format_training_range(self, model: BaseModel) -> str:
        """Format training data range for storage."""
        # This is a placeholder - in practice, you'd extract this from model metadata
        return f"Training data from {datetime.utcnow().strftime('%Y-%m-%d')}"

    def _get_model_class(self, model_type: str):
        """Get the appropriate model class for the given type."""
        # Import here to avoid circular imports
        from src.ml.models.direction_classifier import DirectionClassifier
        from src.ml.models.price_predictor import PricePredictor
        from src.ml.models.regime_detector import RegimeDetector
        from src.ml.models.volatility_forecaster import VolatilityForecaster

        model_classes = {
            "price_predictor": PricePredictor,
            "direction_classifier": DirectionClassifier,
            "volatility_forecaster": VolatilityForecaster,
            "regime_detector": RegimeDetector,
        }

        if model_type not in model_classes:
            raise ModelError(f"Unknown model type: {model_type}")

        return model_classes[model_type]

    def cleanup_old_versions(self, keep_versions: int = 5) -> int:
        """
        Clean up old model versions, keeping only the most recent.

        Args:
            keep_versions: Number of versions to keep per model

        Returns:
            Number of models cleaned up
        """
        cleaned_count = 0

        try:
            with get_sync_session() as session:
                # Group by model name and type
                models = session.query(MLModel).filter(MLModel.is_active).all()
                model_groups = {}

                for model in models:
                    key = (model.name, model.model_type)
                    if key not in model_groups:
                        model_groups[key] = []
                    model_groups[key].append(model)

                # For each group, keep only the most recent versions
                for (_name, _model_type), model_list in model_groups.items():
                    model_list.sort(key=lambda x: x.created_at, reverse=True)

                    if len(model_list) > keep_versions:
                        models_to_remove = model_list[keep_versions:]

                        for model in models_to_remove:
                            try:
                                self.delete_model(model.id, remove_files=True)
                                cleaned_count += 1
                            except Exception as e:
                                logger.warning(
                                    "Failed to clean up model", model_id=model.id, error=str(e)
                                )

                logger.info(
                    "Model cleanup completed",
                    cleaned_count=cleaned_count,
                    keep_versions=keep_versions,
                )

                return cleaned_count

        except Exception as e:
            logger.error("Model cleanup failed", error=str(e))
            raise ModelError(f"Failed to cleanup old versions: {e}") from e
