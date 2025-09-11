"""
Direction Classification Models for Trading.

This module provides direction classification models that predict whether
prices will move up, down, or remain stable in the next period.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from src.core.exceptions import ValidationError
from src.ml.models.base_model import BaseMLModel
from src.utils.decorators import UnifiedDecorator

# Initialize decorator instance
dec = UnifiedDecorator()


class DirectionClassifier(BaseMLModel):
    """
    Direction classification model for predicting price movement direction.

    This model classifies price movements into discrete categories:
    - UP: Price increase above threshold
    - DOWN: Price decrease below threshold
    - NEUTRAL: Price change within threshold range

    Supports multiple algorithms: logistic regression, random forest, SVM, XGBoost.

    Attributes:
        algorithm: Classification algorithm to use
        direction_threshold: Threshold for determining direction classes
        prediction_horizon: Number of periods ahead to predict
        class_names: Names of the prediction classes
        class_weights: Weights for handling class imbalance
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        model_name: str = "direction_classifier",
        version: str = "1.0.0",
        algorithm: str = "random_forest",
        direction_threshold: float = 0.01,
        prediction_horizon: int = 1,
        correlation_id: str | None = None,
    ):
        """
        Initialize the direction classifier.

        Args:
            config: Application configuration
            algorithm: Algorithm to use ('logistic', 'random_forest', 'svm', 'xgboost')
            direction_threshold: Threshold for direction classification (e.g., 0.01 = 1%)
            prediction_horizon: Number of periods ahead to predict
        """
        super().__init__(model_name, version, config, correlation_id)

        # Decimal context is already setup globally in decimal_utils

        self.algorithm = algorithm
        self.direction_threshold = direction_threshold
        self.prediction_horizon = prediction_horizon

        # Class definitions
        self.class_names = ["DOWN", "NEUTRAL", "UP"]
        self.num_classes = len(self.class_names)

        # Model-specific parameters
        ml_config = config.get("ml", {}) if config else {}
        self.class_weights = ml_config.get("class_weights", "balanced")
        self.random_state = ml_config.get("random_state", 42)

        # Initialize the underlying model
        self.model = self._create_model()

        # Training state
        self.feature_importance_ = None
        self.class_distribution_ = None

        self._logger.info(
            "Direction classifier initialized",
            algorithm=self.algorithm,
            direction_threshold=self.direction_threshold,
            prediction_horizon=self.prediction_horizon,
            model_name=self.model_name,
        )

    def _get_model_type(self) -> str:
        """Return the model type identifier."""
        return "direction_classifier"

    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and preprocess features using utilities."""
        from src.utils.ml_validation import validate_features
        return validate_features(X, self.model_name)

    def _validate_targets(self, y: pd.Series) -> pd.Series:
        """Validate and preprocess targets using utilities."""
        from src.utils.ml_validation import validate_targets
        return validate_targets(y, self.model_name)

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Calculate model-specific performance metrics using utilities."""
        from src.utils.ml_metrics import calculate_classification_metrics
        return calculate_classification_metrics(y_true, y_pred)

    def _create_model(self) -> Any:
        """Create the underlying classification model."""
        try:
            if self.algorithm == "logistic":
                return LogisticRegression(
                    class_weight=self.class_weights,
                    random_state=self.random_state,
                    max_iter=1000,
                    solver="liblinear",
                )

            elif self.algorithm == "random_forest":
                return RandomForestClassifier(
                    n_estimators=100,
                    class_weight=self.class_weights,
                    random_state=self.random_state,
                    n_jobs=-1,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                )

            elif self.algorithm == "svm":
                return SVC(
                    class_weight=self.class_weights,
                    random_state=self.random_state,
                    probability=True,  # Enable probability estimates
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                )

            elif self.algorithm == "xgboost":
                if XGBClassifier is None:
                    raise ValidationError("XGBoost not installed. Please install xgboost package.")
                return XGBClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multi:softprob",
                    num_class=self.num_classes,
                )

            else:
                raise ValidationError(f"Unknown algorithm: {self.algorithm}")

        except Exception as e:
            self._logger.error(f"Failed to create model: {e}")
            raise ValidationError(f"Model creation failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
    def fit(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """
        Train the direction classifier.

        Args:
            X: Feature data
            y: Target price data (will be converted to direction classes)

        Returns:
            Training metrics and statistics

        Raises:
            ValidationError: If training fails
        """
        try:
            if X.empty or y.empty:
                raise ValidationError("Training data cannot be empty")

            if len(X) != len(y):
                raise ValidationError("Feature and target data must have same length")

            # Convert price targets to direction classes
            y_direction = self._convert_to_direction_classes(y)

            # Check class distribution
            class_counts = pd.Series(y_direction).value_counts()
            self.class_distribution_ = class_counts.to_dict()

            self._logger.info(
                "Class distribution",
                class_counts=self.class_distribution_,
                model_name=self.model_name,
            )

            # Train the model
            self.model.fit(X, y_direction)

            # Calculate training metrics
            y_pred = self.model.predict(X)
            training_accuracy = accuracy_score(y_direction, y_pred)

            # Get feature importance if available
            if hasattr(self.model, "feature_importances_"):
                self.feature_importance_ = pd.Series(
                    self.model.feature_importances_, index=X.columns
                ).sort_values(ascending=False)

            # Calculate class-specific metrics
            classification_rep = classification_report(
                y_direction, y_pred, target_names=self.class_names, output_dict=True
            )

            # Training metrics
            metrics = {
                "training_accuracy": training_accuracy,
                "algorithm": self.algorithm,
                "direction_threshold": self.direction_threshold,
                "prediction_horizon": self.prediction_horizon,
                "class_distribution": self.class_distribution_,
                "classification_report": classification_rep,
                "feature_count": len(X.columns),
                "training_samples": len(X),
            }

            # Add algorithm-specific metrics
            if hasattr(self.model, "feature_importances_"):
                metrics["top_features"] = self.feature_importance_.head(10).to_dict()

            self.is_trained = True

            self._logger.info(
                "Direction classifier training completed",
                training_accuracy=training_accuracy,
                algorithm=self.algorithm,
                model_name=self.model_name,
            )

            return metrics

        except Exception as e:
            self._logger.error(
                "Direction classifier training failed", algorithm=self.algorithm, error=str(e)
            )
            raise ValidationError(f"Training failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make direction predictions.

        Args:
            X: Feature data

        Returns:
            Array of predicted direction classes (0=DOWN, 1=NEUTRAL, 2=UP)

        Raises:
            ValidationError: If prediction fails
        """
        try:
            if not self.is_trained:
                raise ValidationError("Model must be trained before prediction")

            if X.empty:
                raise ValidationError("Feature data cannot be empty")

            predictions = self.model.predict(X)

            self._logger.debug(
                "Direction predictions made",
                prediction_count=len(predictions),
                model_name=self.model_name,
            )

            return predictions

        except Exception as e:
            self._logger.error(
                "Direction prediction failed", error=str(e), model_name=self.model_name
            )
            raise ValidationError(f"Prediction failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature data

        Returns:
            Array of class probabilities [P(DOWN), P(NEUTRAL), P(UP)]

        Raises:
            ValidationError: If prediction fails
        """
        try:
            if not self.is_trained:
                raise ValidationError("Model must be trained before prediction")

            if X.empty:
                raise ValidationError("Feature data cannot be empty")

            probabilities = self.model.predict_proba(X)

            self._logger.debug(
                "Direction probabilities predicted",
                prediction_count=len(probabilities),
                model_name=self.model_name,
            )

            return probabilities

        except Exception as e:
            self._logger.error(
                "Direction probability prediction failed", error=str(e), model_name=self.model_name
            )
            raise ValidationError(f"Probability prediction failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            X: Feature data
            y: Target price data

        Returns:
            Evaluation metrics

        Raises:
            ValidationError: If evaluation fails
        """
        try:
            if not self.is_trained:
                raise ValidationError("Model must be trained before evaluation")

            if X.empty or y.empty:
                raise ValidationError("Evaluation data cannot be empty")

            # Convert targets to direction classes
            y_true = self._convert_to_direction_classes(y)

            # Make predictions
            y_pred = self.predict(X)
            y_proba = self.predict_proba(X)

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)

            # Classification report
            classification_rep = classification_report(
                y_true, y_pred, target_names=self.class_names, output_dict=True
            )

            # Confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)

            # Class-specific accuracies
            class_accuracies = {}
            for i, class_name in enumerate(self.class_names):
                class_mask = y_true == i
                if class_mask.sum() > 0:
                    class_accuracy = accuracy_score(y_true[class_mask], y_pred[class_mask])
                    class_accuracies[f"{class_name}_accuracy"] = class_accuracy

            # Directional accuracy (for financial relevance)
            directional_accuracy = self._calculate_directional_accuracy(y_true, y_pred)

            metrics = {
                "test_accuracy": accuracy,
                "directional_accuracy": directional_accuracy,
                "classification_report": classification_rep,
                "confusion_matrix": conf_matrix.tolist(),
                "class_accuracies": class_accuracies,
                "algorithm": self.algorithm,
                "direction_threshold": self.direction_threshold,
                "test_samples": len(X),
            }

            self._logger.info(
                "Direction classifier evaluation completed",
                test_accuracy=accuracy,
                directional_accuracy=directional_accuracy,
                model_name=self.model_name,
            )

            return metrics

        except Exception as e:
            self._logger.error(
                "Direction classifier evaluation failed", error=str(e), model_name=self.model_name
            )
            raise ValidationError(f"Evaluation failed: {e}")

    def _convert_to_direction_classes(self, price_data: pd.Series) -> np.ndarray:
        """Convert price data to direction classes using utilities."""
        try:
            from src.utils.ml_data_transforms import create_returns_series

            # Calculate returns using utility function
            returns = create_returns_series(price_data, self.prediction_horizon, "simple")

            # Convert to direction classes
            direction_classes = np.zeros(len(returns))

            # UP: return > threshold
            direction_classes[returns > self.direction_threshold] = 2

            # DOWN: return < -threshold
            direction_classes[returns < -self.direction_threshold] = 0

            # NEUTRAL: -threshold <= return <= threshold
            neutral_mask = (returns >= -self.direction_threshold) & (
                returns <= self.direction_threshold
            )
            direction_classes[neutral_mask] = 1

            # Remove NaN values
            valid_mask = ~np.isnan(returns)

            return direction_classes[valid_mask].astype(int)

        except Exception as e:
            self._logger.error(f"Failed to convert to direction classes: {e}")
            raise ValidationError(f"Direction class conversion failed: {e}")

    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (ignoring neutral class)."""
        try:
            # Focus only on UP vs DOWN predictions (ignore NEUTRAL)
            non_neutral_mask = (y_true != 1) & (y_pred != 1)

            if non_neutral_mask.sum() == 0:
                return 0.0  # No directional predictions

            directional_correct = (y_true[non_neutral_mask] == y_pred[non_neutral_mask]).sum()
            directional_total = non_neutral_mask.sum()

            return directional_correct / directional_total

        except Exception as e:
            self._logger.error(f"Failed to calculate directional accuracy: {e}")
            return 0.0

    def get_feature_importance(self) -> pd.Series | None:
        """
        Get feature importance scores.

        Returns:
            Series with feature importance scores or None if not available
        """
        if not self.is_trained:
            self._logger.warning("Model not trained, no feature importance available")
            return None

        return self.feature_importance_

    def get_class_distribution(self) -> dict[str, int] | None:
        """
        Get training class distribution.

        Returns:
            Dictionary with class distribution or None if not trained
        """
        if not self.is_trained:
            self._logger.warning("Model not trained, no class distribution available")
            return None

        return self.class_distribution_

    def predict_direction_labels(self, X: pd.DataFrame) -> list[str]:
        """
        Predict direction as string labels.

        Args:
            X: Feature data

        Returns:
            List of direction labels ('UP', 'DOWN', 'NEUTRAL')
        """
        try:
            predictions = self.predict(X)
            return [self.class_names[pred] for pred in predictions]

        except Exception as e:
            self._logger.error(f"Direction label prediction failed: {e}")
            raise ValidationError(f"Direction label prediction failed: {e}")

    def get_prediction_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction confidence scores.

        Args:
            X: Feature data

        Returns:
            Array of confidence scores (maximum probability)
        """
        try:
            probabilities = self.predict_proba(X)
            return np.max(probabilities, axis=1)

        except Exception as e:
            self._logger.error(f"Confidence calculation failed: {e}")
            raise ValidationError(f"Confidence calculation failed: {e}")
