"""
Market Regime Detection Models for Trading.

This module provides market regime detection models that identify different
market conditions such as trending, ranging, volatile, or calm periods.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.core.config import Config
from src.core.exceptions import ValidationError
from src.ml.models.base_model import BaseMLModel
from src.utils.decorators import UnifiedDecorator

# Initialize decorator instance
dec = UnifiedDecorator()


class RegimeDetector(BaseMLModel):
    """
    Market regime detection model for identifying market conditions.

    This model detects various market regimes:
    - Trending vs Ranging markets
    - High vs Low volatility periods
    - Bull vs Bear vs Sideways markets
    - Risk-on vs Risk-off periods

    Supports both supervised and unsupervised approaches.

    Attributes:
        detection_method: Method for regime detection
        n_regimes: Number of regimes to detect
        regime_features: Features to use for regime detection
        use_supervised: Whether to use supervised learning
        lookback_window: Window for calculating regime features
    """

    def __init__(
        self,
        config: Config,
        detection_method: str = "kmeans",
        n_regimes: int = 3,
        use_supervised: bool = False,
        lookback_window: int = 20,
    ):
        """
        Initialize the regime detector.

        Args:
            config: Application configuration
            detection_method: Detection method ('kmeans', 'gmm', 'random_forest')
            n_regimes: Number of regimes to detect
            use_supervised: Whether to use supervised learning
            lookback_window: Window for calculating regime features
        """
        # Set attributes before calling super() since _get_model_type() needs them
        self.detection_method = detection_method
        self.n_regimes = n_regimes
        self.use_supervised = use_supervised
        self.lookback_window = lookback_window

        super().__init__(
            model_name="regime_detector",
            version="1.0.0",
            config=config,
        )

        # Model parameters - use config dictionary with fallbacks
        ml_config = config.get("ml", {})
        self.random_state = ml_config.get("random_state", 42)
        self.scale_features = True

        # Initialize models
        self.model = self._create_model()
        self.scaler = StandardScaler() if self.scale_features else None

        # Regime definitions
        self.regime_names = self._define_regime_names()
        self.regime_types = self.regime_names  # Alias for compatibility

        # Training state
        self.regime_stats_ = None
        self.feature_importance_ = None

        self.logger.info(
            "Regime detector initialized",
            detection_method=self.detection_method,
            n_regimes=self.n_regimes,
            use_supervised=self.use_supervised,
            lookback_window=self.lookback_window,
            model_name=self.model_name,
        )

    def _create_model(self) -> Any:
        """Create the underlying regime detection model."""
        try:
            if self.detection_method == "kmeans":
                return KMeans(
                    n_clusters=self.n_regimes,
                    random_state=self.random_state,
                    n_init=10,
                    max_iter=300,
                )

            elif self.detection_method == "gmm":
                return GaussianMixture(
                    n_components=self.n_regimes,
                    random_state=self.random_state,
                    covariance_type="full",
                    max_iter=100,
                )

            elif self.detection_method == "random_forest":
                return RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1,
                    max_depth=10,
                    min_samples_split=5,
                    class_weight="balanced",
                )

            else:
                raise ValidationError(f"Unknown detection method: {self.detection_method}")

        except Exception as e:
            self.logger.error(f"Failed to create model: {e}")
            raise ValidationError(f"Model creation failed: {e}")

    def _define_regime_names(self) -> list[str]:
        """Define regime names based on number of regimes."""
        if self.n_regimes == 2:
            return ["Low_Volatility", "High_Volatility"]
        elif self.n_regimes == 3:
            return ["Bear_Market", "Sideways_Market", "Bull_Market"]
        elif self.n_regimes == 4:
            return ["Bear_Low_Vol", "Bear_High_Vol", "Bull_Low_Vol", "Bull_High_Vol"]
        else:
            return [f"Regime_{i}" for i in range(self.n_regimes)]

    @dec.enhance(log=True, monitor=True, log_level="info")
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, Any]:
        """
        Train the regime detector.

        Args:
            X: Market data (OHLCV)
            y: Optional regime labels for supervised learning

        Returns:
            Training metrics and statistics

        Raises:
            ValidationError: If training fails
        """
        try:
            if X.empty:
                raise ValidationError("Training data cannot be empty")

            # Create regime features
            regime_features = self._create_regime_features(X)

            # Remove NaN values
            regime_features_clean = regime_features.dropna()

            if regime_features_clean.empty:
                raise ValidationError("No valid features after cleaning")

            # Scale features if requested
            if self.scaler:
                features_scaled = self.scaler.fit_transform(regime_features_clean)
                features_df = pd.DataFrame(
                    features_scaled,
                    index=regime_features_clean.index,
                    columns=regime_features_clean.columns,
                )
            else:
                features_df = regime_features_clean

            # Train model
            if self.use_supervised and y is not None:
                # Supervised learning
                y_aligned = y.loc[features_df.index]
                self.model.fit(features_df, y_aligned)

                # Calculate training metrics
                y_pred = self.model.predict(features_df)
                training_accuracy = accuracy_score(y_aligned, y_pred)

                metrics = {
                    "training_accuracy": training_accuracy,
                    "detection_method": self.detection_method,
                    "supervised": True,
                }

                # Get feature importance if available
                if hasattr(self.model, "feature_importances_"):
                    self.feature_importance_ = pd.Series(
                        self.model.feature_importances_, index=features_df.columns
                    ).sort_values(ascending=False)
                    metrics["top_features"] = self.feature_importance_.head(10).to_dict()

            else:
                # Unsupervised learning
                regime_labels = self.model.fit_predict(features_df)

                # Calculate clustering metrics
                silhouette = silhouette_score(features_df, regime_labels)

                # Calculate regime statistics
                self.regime_stats_ = self._calculate_regime_statistics(features_df, regime_labels)

                metrics = {
                    "silhouette_score": silhouette,
                    "detection_method": self.detection_method,
                    "supervised": False,
                    "regime_distribution": pd.Series(regime_labels).value_counts().to_dict(),
                    "regime_stats": self.regime_stats_,
                }

            # Common metrics
            metrics.update(
                {
                    "n_regimes": self.n_regimes,
                    "lookback_window": self.lookback_window,
                    "feature_count": len(features_df.columns),
                    "training_samples": len(features_df),
                }
            )

            self.is_trained = True

            self.logger.info(
                "Regime detector training completed",
                detection_method=self.detection_method,
                n_regimes=self.n_regimes,
                supervised=self.use_supervised,
                model_name=self.model_name,
            )

            return metrics

        except Exception as e:
            self.logger.error(
                "Regime detector training failed",
                detection_method=self.detection_method,
                error=str(e),
            )
            raise ValidationError(f"Training failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Detect market regimes.

        Args:
            X: Market data

        Returns:
            Array of regime labels

        Raises:
            ValidationError: If prediction fails
        """
        try:
            if not self.is_trained:
                raise ValidationError("Model must be trained before prediction")

            if X.empty:
                raise ValidationError("Market data cannot be empty")

            # Create regime features
            regime_features = self._create_regime_features(X)

            # Handle NaN values for prediction
            regime_features_clean = regime_features.ffill().fillna(0)

            # Scale features if scaler was used during training
            if self.scaler:
                features_scaled = self.scaler.transform(regime_features_clean)
                features_df = pd.DataFrame(
                    features_scaled,
                    index=regime_features_clean.index,
                    columns=regime_features_clean.columns,
                )
            else:
                features_df = regime_features_clean

            # Make predictions
            regime_labels = self.model.predict(features_df)

            self.logger.debug(
                "Regime detection completed",
                prediction_count=len(regime_labels),
                unique_regimes=len(np.unique(regime_labels)),
                model_name=self.model_name,
            )

            return regime_labels

        except Exception as e:
            self.logger.error("Regime detection failed", error=str(e), model_name=self.model_name)
            raise ValidationError(f"Regime detection failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
    def evaluate(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, Any]:
        """
        Evaluate the regime detector.

        Args:
            X: Market data
            y: Optional true regime labels

        Returns:
            Evaluation metrics

        Raises:
            ValidationError: If evaluation fails
        """
        try:
            if not self.is_trained:
                raise ValidationError("Model must be trained before evaluation")

            if X.empty:
                raise ValidationError("Evaluation data cannot be empty")

            # Make predictions
            y_pred = self.predict(X)

            metrics = {
                "detection_method": self.detection_method,
                "n_regimes": self.n_regimes,
                "test_samples": len(X),
                "regime_distribution": pd.Series(y_pred).value_counts().to_dict(),
            }

            if y is not None and self.use_supervised:
                # Supervised evaluation
                y_aligned = y.iloc[: len(y_pred)]  # Align lengths

                test_accuracy = accuracy_score(y_aligned, y_pred)
                metrics["test_accuracy"] = test_accuracy

                self.logger.info(
                    "Supervised regime detector evaluation completed",
                    test_accuracy=test_accuracy,
                    model_name=self.model_name,
                )

            else:
                # Unsupervised evaluation
                regime_features = self._create_regime_features(X)
                regime_features_clean = regime_features.ffill().fillna(0)

                if self.scaler:
                    features_scaled = self.scaler.transform(regime_features_clean)
                    features_df = pd.DataFrame(
                        features_scaled,
                        index=regime_features_clean.index,
                        columns=regime_features_clean.columns,
                    )
                else:
                    features_df = regime_features_clean

                # Calculate clustering metrics
                silhouette = silhouette_score(features_df, y_pred)
                metrics["silhouette_score"] = silhouette

                self.logger.info(
                    "Unsupervised regime detector evaluation completed",
                    silhouette_score=silhouette,
                    model_name=self.model_name,
                )

            return metrics

        except Exception as e:
            self.logger.error(
                "Regime detector evaluation failed", error=str(e), model_name=self.model_name
            )
            raise ValidationError(f"Evaluation failed: {e}")

    def _create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for regime detection."""
        try:
            features = pd.DataFrame(index=data.index)

            # Price-based features
            if "close" in data.columns:
                # Returns and volatility
                returns = data["close"].pct_change()
                features["returns_mean"] = returns.rolling(self.lookback_window).mean()
                features["returns_std"] = returns.rolling(self.lookback_window).std()
                features["returns_skew"] = returns.rolling(self.lookback_window).skew()
                features["returns_kurt"] = returns.rolling(self.lookback_window).kurt()

                # Trend indicators
                features["sma_ratio"] = (
                    data["close"] / data["close"].rolling(self.lookback_window).mean()
                )
                features["price_momentum"] = data["close"] / data["close"].shift(
                    self.lookback_window
                )

                # Range indicators
                if "high" in data.columns and "low" in data.columns:
                    features["range_ratio"] = (data["high"] - data["low"]) / data["close"]
                    features["range_expansion"] = (
                        features["range_ratio"]
                        / features["range_ratio"].rolling(self.lookback_window).mean()
                    )

            # Volume-based features
            if "volume" in data.columns:
                features["volume_ratio"] = (
                    data["volume"] / data["volume"].rolling(self.lookback_window).mean()
                )
                features["volume_trend"] = data["volume"] / data["volume"].shift(
                    self.lookback_window
                )

                if "close" in data.columns:
                    # Price-volume relationship
                    features["pv_corr"] = returns.rolling(self.lookback_window).corr(
                        data["volume"].pct_change()
                    )

            # Volatility regime features
            if "close" in data.columns:
                # Different volatility measures
                features["realized_vol"] = returns.rolling(self.lookback_window).std() * np.sqrt(
                    252
                )

                if "high" in data.columns and "low" in data.columns:
                    # Parkinson volatility estimator
                    log_hl = np.log(data["high"] / data["low"])
                    parkinson_base = log_hl.rolling(self.lookback_window).mean() / (4 * np.log(2))
                    # Ensure non-negative values before sqrt to avoid warnings
                    parkinson_base = parkinson_base.clip(lower=0)
                    features["parkinson_vol"] = np.sqrt(parkinson_base) * np.sqrt(252)

                # Volatility clustering
                squared_returns = returns**2
                features["vol_clustering"] = squared_returns.rolling(self.lookback_window).mean()

            # Market microstructure features
            if all(col in data.columns for col in ["open", "high", "low", "close"]):
                # Gaps and overnight returns
                features["gap"] = (data["open"] - data["close"].shift(1)) / data["close"].shift(1)
                features["overnight_return"] = (data["open"] - data["close"].shift(1)) / data[
                    "close"
                ].shift(1)
                features["intraday_return"] = (data["close"] - data["open"]) / data["open"]

                # OHLC ratios
                features["body_ratio"] = np.abs(data["close"] - data["open"]) / (
                    data["high"] - data["low"]
                )
                features["upper_shadow"] = (
                    data["high"] - np.maximum(data["open"], data["close"])
                ) / (data["high"] - data["low"])
                features["lower_shadow"] = (
                    np.minimum(data["open"], data["close"]) - data["low"]
                ) / (data["high"] - data["low"])

            # Temporal features
            if data.index.dtype.kind == "M":  # datetime index
                features["day_of_week"] = data.index.dayofweek
                features["month"] = data.index.month
                features["hour"] = data.index.hour if hasattr(data.index, "hour") else 0

            self.logger.debug(
                "Regime features created",
                feature_count=len(features.columns),
                sample_count=len(features),
            )

            return features

        except Exception as e:
            self.logger.error(f"Failed to create regime features: {e}")
            raise ValidationError(f"Regime feature creation failed: {e}")

    def _calculate_regime_statistics(
        self, features: pd.DataFrame, regime_labels: np.ndarray
    ) -> dict[str, Any]:
        """Calculate statistics for each detected regime."""
        try:
            regime_stats = {}

            for regime_id in np.unique(regime_labels):
                regime_mask = regime_labels == regime_id
                regime_features = features[regime_mask]

                regime_stats[f"regime_{regime_id}"] = {
                    "count": regime_mask.sum(),
                    "percentage": regime_mask.mean() * 100,
                    "feature_means": regime_features.mean().to_dict(),
                    "feature_stds": regime_features.std().to_dict(),
                }

            return regime_stats

        except Exception as e:
            self.logger.error(f"Failed to calculate regime statistics: {e}")
            return {}

    def predict_regime_labels(self, X: pd.DataFrame) -> list[str]:
        """
        Predict regime as string labels.

        Args:
            X: Market data

        Returns:
            List of regime labels
        """
        try:
            predictions = self.predict(X)
            return [
                self.regime_names[pred] if pred < len(self.regime_names) else f"Regime_{pred}"
                for pred in predictions
            ]

        except Exception as e:
            self.logger.error(f"Regime label prediction failed: {e}")
            raise ValidationError(f"Regime label prediction failed: {e}")

    def get_regime_probabilities(self, X: pd.DataFrame) -> np.ndarray | None:
        """
        Get regime probabilities (if supported by the model).

        Args:
            X: Market data

        Returns:
            Array of regime probabilities or None if not supported
        """
        try:
            if not hasattr(self.model, "predict_proba"):
                self.logger.warning("Model does not support probability prediction")
                return None

            # Create regime features
            regime_features = self._create_regime_features(X)
            regime_features_clean = regime_features.ffill().fillna(0)

            if self.scaler:
                features_scaled = self.scaler.transform(regime_features_clean)
                features_df = pd.DataFrame(
                    features_scaled,
                    index=regime_features_clean.index,
                    columns=regime_features_clean.columns,
                )
            else:
                features_df = regime_features_clean

            return self.model.predict_proba(features_df)

        except Exception as e:
            self.logger.error(f"Regime probability prediction failed: {e}")
            return None

    def get_regime_statistics(self) -> dict[str, Any] | None:
        """
        Get regime statistics from training.

        Returns:
            Dictionary with regime statistics or None if not available
        """
        if not self.is_trained:
            self.logger.warning("Model not trained, no regime statistics available")
            return None

        return self.regime_stats_

    def get_feature_importance(self) -> pd.Series | None:
        """
        Get feature importance scores (supervised models only).

        Returns:
            Series with feature importance scores or None if not available
        """
        if not self.is_trained or not self.use_supervised:
            self.logger.warning("Feature importance only available for supervised models")
            return None

        return self.feature_importance_

    def get_regime_names(self) -> list[str]:
        """Get list of regime names."""
        return self.regime_names

    def interpret_regime(self, regime_id: int) -> str:
        """Interpret regime ID as human-readable name."""
        if regime_id < len(self.regime_names):
            return self.regime_names[regime_id]
        return f"Regime_{regime_id}"

    def _get_model_type(self) -> str:
        """Return the model type identifier."""
        return "clustering" if not self.use_supervised else "classification"

    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and preprocess features for the model."""
        if X.empty:
            raise ValidationError("Feature data cannot be empty")
        return X

    def _validate_targets(self, y: pd.Series) -> pd.Series:
        """Validate and preprocess targets for the model."""
        if y is not None and y.empty:
            raise ValidationError("Target data cannot be empty")
        return y

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate metrics for model evaluation."""
        from sklearn.metrics import accuracy_score

        if self.use_supervised:
            return {"accuracy": accuracy_score(y_true, y_pred)}
        else:
            # For unsupervised, we can't calculate accuracy without true labels
            return {"silhouette_score": 0.5}  # Default placeholder
