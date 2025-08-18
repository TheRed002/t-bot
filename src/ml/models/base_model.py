"""
Abstract base class for all ML models in the trading system.

This module provides the foundational interface that all ML models must implement,
ensuring consistency, proper lifecycle management, and integration with the ML infrastructure.
"""

import abc
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.core.config import Config
from src.core.exceptions import ModelError, ValidationError
from src.core.logging import get_logger
from src.utils.decorators import log_calls, memory_usage, time_execution

logger = get_logger(__name__)


class BaseModel(abc.ABC):
    """
    Abstract base class for all ML models in the trading system.

    This class provides the foundation for model lifecycle management including
    training, prediction, serialization, validation, and performance tracking.
    All concrete model implementations must inherit from this class.

    Attributes:
        config: Application configuration
        model_type: Type identifier for the model
        model_name: Human-readable name for the model
        version: Model version string
        model: The underlying ML model object
        is_trained: Whether the model has been trained
        feature_names: List of feature names used by the model
        target_name: Name of the target variable
        metrics: Performance metrics dictionary
        metadata: Additional model metadata
    """

    def __init__(self, config: Config, model_name: str, version: str = "1.0.0"):
        """
        Initialize the base model.

        Args:
            config: Application configuration
            model_name: Human-readable name for the model
            version: Model version string
        """
        self.config = config
        self.model_name = model_name
        self.version = version
        self.model_type = self._get_model_type()

        # Model state
        self.model: BaseEstimator | None = None
        self.is_trained: bool = False
        self.feature_names: list[str] = []
        self.target_name: str | None = None

        # Performance tracking
        self.metrics: dict[str, float] = {}
        self.training_history: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "training_data_hash": None,
            "hyperparameters": {},
        }

        logger.info(
            "Base model initialized",
            model_name=self.model_name,
            model_type=self.model_type,
            version=self.version,
        )

    @abc.abstractmethod
    def _get_model_type(self) -> str:
        """Return the model type identifier."""
        pass

    @abc.abstractmethod
    def _create_model(self, **kwargs) -> BaseEstimator:
        """Create and return the underlying ML model."""
        pass

    @abc.abstractmethod
    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and preprocess features for the model."""
        pass

    @abc.abstractmethod
    def _validate_targets(self, y: pd.Series) -> pd.Series:
        """Validate and preprocess targets for the model."""
        pass

    @abc.abstractmethod
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Calculate model-specific performance metrics."""
        pass

    @time_execution
    @memory_usage
    @log_calls
    def prepare_data(
        self, X: pd.DataFrame, y: pd.Series | None = None, feature_selection: bool = True
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """
        Prepare training or prediction data.

        Args:
            X: Feature data
            y: Target data (optional for prediction)
            feature_selection: Whether to apply feature selection

        Returns:
            Tuple of processed features and targets

        Raises:
            ValidationError: If data validation fails
        """
        try:
            # Validate features
            X_processed = self._validate_features(X)

            # Apply feature selection if enabled and model is trained
            if feature_selection and self.feature_names:
                missing_features = set(self.feature_names) - set(X_processed.columns)
                if missing_features:
                    raise ValidationError(f"Missing required features: {missing_features}")
                X_processed = X_processed[self.feature_names]

            # Validate targets if provided
            y_processed = None
            if y is not None:
                y_processed = self._validate_targets(y)

                # Ensure X and y have same length
                if len(X_processed) != len(y_processed):
                    raise ValidationError(
                        f"Feature and target lengths don't match: {len(X_processed)} vs {len(y_processed)}"
                    )

            logger.info(
                "Data preparation completed",
                features_shape=X_processed.shape,
                targets_shape=y_processed.shape if y_processed is not None else None,
                model_name=self.model_name,
            )

            return X_processed, y_processed

        except Exception as e:
            logger.error("Data preparation failed", model_name=self.model_name, error=str(e))
            raise ValidationError(f"Data preparation failed: {e}") from e

    @time_execution
    @memory_usage
    @log_calls
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: tuple[pd.DataFrame, pd.Series] | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """
        Train the model on provided data.

        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation data tuple
            **kwargs: Additional training parameters

        Returns:
            Training metrics dictionary

        Raises:
            ModelError: If training fails
        """
        try:
            # Prepare training data
            X_train, y_train = self.prepare_data(X, y, feature_selection=False)

            # Store feature names and target name
            self.feature_names = list(X_train.columns)
            self.target_name = y_train.name if hasattr(y_train, "name") else "target"

            # Create data hash for tracking
            data_hash = self._calculate_data_hash(X_train, y_train)
            self.metadata["training_data_hash"] = data_hash

            # Create model if not exists
            if self.model is None:
                self.model = self._create_model(**kwargs)
                self.metadata["hyperparameters"] = self.model.get_params()

            # Train the model
            logger.info(
                "Starting model training",
                model_name=self.model_name,
                features_shape=X_train.shape,
                targets_shape=y_train.shape,
            )

            self.model.fit(X_train, y_train)

            # Calculate training metrics
            y_train_pred = self.model.predict(X_train)
            train_metrics = self._calculate_metrics(y_train.values, y_train_pred)
            train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}

            # Calculate validation metrics if validation data provided
            val_metrics = {}
            if validation_data is not None:
                X_val, y_val = validation_data
                X_val_processed, y_val_processed = self.prepare_data(X_val, y_val)
                y_val_pred = self.model.predict(X_val_processed)
                val_metrics = self._calculate_metrics(y_val_processed.values, y_val_pred)
                val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            self.metrics.update(all_metrics)

            # Update training history
            training_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": all_metrics,
                "hyperparameters": self.model.get_params(),
                "data_hash": data_hash,
            }
            self.training_history.append(training_record)

            # Mark as trained
            self.is_trained = True
            self.metadata["updated_at"] = datetime.utcnow().isoformat()

            logger.info("Model training completed", model_name=self.model_name, metrics=all_metrics)

            return all_metrics

        except Exception as e:
            logger.error("Model training failed", model_name=self.model_name, error=str(e))
            raise ModelError(f"Training failed for {self.model_name}: {e}") from e

    @time_execution
    @log_calls
    def predict(
        self, X: pd.DataFrame, return_probabilities: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the trained model.

        Args:
            X: Feature data for prediction
            return_probabilities: Whether to return prediction probabilities

        Returns:
            Predictions array or tuple of (predictions, probabilities)

        Raises:
            ModelError: If prediction fails or model not trained
        """
        if not self.is_trained or self.model is None:
            raise ModelError(f"Model {self.model_name} is not trained")

        try:
            # Prepare prediction data
            X_processed, _ = self.prepare_data(X, feature_selection=True)

            # Make predictions
            predictions = self.model.predict(X_processed)

            logger.info(
                "Predictions generated",
                model_name=self.model_name,
                input_shape=X_processed.shape,
                output_shape=predictions.shape,
            )

            # Return probabilities if requested and supported
            if return_probabilities and hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(X_processed)
                return predictions, probabilities

            return predictions

        except Exception as e:
            logger.error("Prediction failed", model_name=self.model_name, error=str(e))
            raise ModelError(f"Prediction failed for {self.model_name}: {e}") from e

    @time_execution
    @log_calls
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            X: Test features
            y: Test targets

        Returns:
            Evaluation metrics dictionary

        Raises:
            ModelError: If evaluation fails
        """
        if not self.is_trained:
            raise ModelError(f"Model {self.model_name} is not trained")

        try:
            # Prepare test data
            X_test, y_test = self.prepare_data(X, y)

            # Make predictions
            y_pred = self.model.predict(X_test)

            # Calculate metrics
            eval_metrics = self._calculate_metrics(y_test.values, y_pred)
            eval_metrics = {f"test_{k}": v for k, v in eval_metrics.items()}

            logger.info(
                "Model evaluation completed", model_name=self.model_name, metrics=eval_metrics
            )

            return eval_metrics

        except Exception as e:
            logger.error("Model evaluation failed", model_name=self.model_name, error=str(e))
            raise ModelError(f"Evaluation failed for {self.model_name}: {e}") from e

    @time_execution
    @log_calls
    def save(self, filepath: str | Path) -> Path:
        """
        Save the model to disk.

        Args:
            filepath: Path to save the model

        Returns:
            Actual filepath where model was saved

        Raises:
            ModelError: If saving fails
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Prepare model data for saving
            model_data = {
                "model": self.model,
                "model_name": self.model_name,
                "model_type": self.model_type,
                "version": self.version,
                "is_trained": self.is_trained,
                "feature_names": self.feature_names,
                "target_name": self.target_name,
                "metrics": self.metrics,
                "training_history": self.training_history,
                "metadata": self.metadata,
            }

            # Save using joblib for better sklearn compatibility
            joblib.dump(model_data, filepath)

            logger.info(
                "Model saved successfully", model_name=self.model_name, filepath=str(filepath)
            )

            return filepath

        except Exception as e:
            logger.error(
                "Model saving failed",
                model_name=self.model_name,
                filepath=str(filepath),
                error=str(e),
            )
            raise ModelError(f"Failed to save model {self.model_name}: {e}") from e

    @classmethod
    @time_execution
    @log_calls
    def load(cls, filepath: str | Path, config: Config) -> "BaseModel":
        """
        Load a model from disk.

        Args:
            filepath: Path to the saved model
            config: Application configuration

        Returns:
            Loaded model instance

        Raises:
            ModelError: If loading fails
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")

            # Load model data
            model_data = joblib.load(filepath)

            # Create new instance
            instance = cls(
                config=config, model_name=model_data["model_name"], version=model_data["version"]
            )

            # Restore model state
            instance.model = model_data["model"]
            instance.model_type = model_data["model_type"]
            instance.is_trained = model_data["is_trained"]
            instance.feature_names = model_data["feature_names"]
            instance.target_name = model_data["target_name"]
            instance.metrics = model_data["metrics"]
            instance.training_history = model_data["training_history"]
            instance.metadata = model_data["metadata"]

            logger.info(
                "Model loaded successfully", model_name=instance.model_name, filepath=str(filepath)
            )

            return instance

        except Exception as e:
            logger.error("Model loading failed", filepath=str(filepath), error=str(e))
            raise ModelError(f"Failed to load model from {filepath}: {e}") from e

    def get_feature_importance(self) -> pd.Series | None:
        """
        Get feature importance if available.

        Returns:
            Series with feature importance or None if not available
        """
        if not self.is_trained or self.model is None:
            return None

        # Check if model has feature importance
        if hasattr(self.model, "feature_importances_"):
            return pd.Series(
                self.model.feature_importances_, index=self.feature_names, name="feature_importance"
            ).sort_values(ascending=False)

        # Check for coefficients (linear models)
        elif hasattr(self.model, "coef_"):
            coef = self.model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            return pd.Series(
                np.abs(coef), index=self.feature_names, name="feature_importance"
            ).sort_values(ascending=False)

        return None

    def get_model_info(self) -> dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with model information
        """
        info = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "is_trained": self.is_trained,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "training_history_length": len(self.training_history),
        }

        # Add model-specific info if available
        if self.model is not None:
            info["model_params"] = self.model.get_params()
            info["model_class"] = self.model.__class__.__name__

        return info

    def _calculate_data_hash(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Calculate hash of training data for tracking."""
        # Combine features and targets for hashing
        data_str = str(X.values.tobytes()) + str(y.values.tobytes())
        return hashlib.md5(data_str.encode()).hexdigest()

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.model_name}', "
            f"type='{self.model_type}', "
            f"version='{self.version}', "
            f"trained={self.is_trained})"
        )
