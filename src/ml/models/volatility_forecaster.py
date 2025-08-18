"""
Volatility Forecasting Models for Trading.

This module provides volatility forecasting models that predict future
volatility levels for risk management and position sizing.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.ml.models.base_model import BaseModel
from src.utils.decorators import log_calls, time_execution

logger = get_logger(__name__)


class VolatilityForecaster(BaseModel):
    """
    Volatility forecasting model for predicting future volatility.

    This model forecasts various volatility measures:
    - Realized volatility (historical)
    - GARCH-style volatility
    - Intraday volatility
    - Volume-weighted volatility

    Supports multiple algorithms and volatility estimation methods.

    Attributes:
        algorithm: Forecasting algorithm to use
        volatility_method: Method for calculating volatility
        forecast_horizon: Number of periods ahead to forecast
        volatility_window: Window size for volatility calculation
        use_log_returns: Whether to use log returns for volatility
        scaling_method: Feature scaling method
    """

    def __init__(
        self,
        config: Config,
        algorithm: str = "random_forest",
        volatility_method: str = "realized",
        forecast_horizon: int = 1,
        volatility_window: int = 20,
    ):
        """
        Initialize the volatility forecaster.

        Args:
            config: Application configuration
            algorithm: Algorithm to use ('linear', 'random_forest', 'xgboost')
            volatility_method: Volatility calculation method ('realized', 'garch', 'intraday')
            forecast_horizon: Number of periods ahead to forecast
            volatility_window: Window size for volatility calculation
        """
        super().__init__(
            config=config,
            model_name="volatility_forecaster",
            model_type="regression",
            version="1.0.0",
        )

        self.algorithm = algorithm
        self.volatility_method = volatility_method
        self.forecast_horizon = forecast_horizon
        self.volatility_window = volatility_window

        # Model parameters
        self.use_log_returns = config.ml.use_log_returns
        self.scaling_method = config.ml.scaling_method
        self.random_state = config.ml.random_state

        # Initialize the underlying model and scaler
        self.model = self._create_model()
        self.scaler = StandardScaler() if self.scaling_method == "standard" else None

        # Training state
        self.feature_importance_ = None
        self.volatility_stats_ = None

        logger.info(
            "Volatility forecaster initialized",
            algorithm=self.algorithm,
            volatility_method=self.volatility_method,
            forecast_horizon=self.forecast_horizon,
            volatility_window=self.volatility_window,
            model_name=self.model_name,
        )

    def _create_model(self) -> Any:
        """Create the underlying forecasting model."""
        try:
            if self.algorithm == "linear":
                return LinearRegression(fit_intercept=True, n_jobs=-1)

            elif self.algorithm == "random_forest":
                return RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features="sqrt",
                )

            elif self.algorithm == "xgboost":
                return XGBRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                )

            else:
                raise ValidationError(f"Unknown algorithm: {self.algorithm}")

        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise ValidationError(f"Model creation failed: {e}") from e

    @time_execution
    @log_calls
    def fit(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """
        Train the volatility forecaster.

        Args:
            X: Feature data
            y: Target price data (will be converted to volatility)

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

            # Convert price data to volatility targets
            volatility_targets = self._calculate_volatility_targets(y)

            # Align features with volatility targets
            min_length = min(len(X), len(volatility_targets))
            X_aligned = X.iloc[:min_length]
            y_volatility = volatility_targets[:min_length]

            # Remove NaN values
            valid_mask = ~(np.isnan(y_volatility) | np.isinf(y_volatility))
            X_clean = X_aligned[valid_mask]
            y_clean = y_volatility[valid_mask]

            if len(X_clean) == 0:
                raise ValidationError("No valid training data after cleaning")

            # Calculate volatility statistics
            self.volatility_stats_ = {
                "mean": np.mean(y_clean),
                "std": np.std(y_clean),
                "min": np.min(y_clean),
                "max": np.max(y_clean),
                "median": np.median(y_clean),
                "q25": np.percentile(y_clean, 25),
                "q75": np.percentile(y_clean, 75),
            }

            # Scale features if requested
            if self.scaler:
                X_scaled = self.scaler.fit_transform(X_clean)
                X_train = pd.DataFrame(X_scaled, columns=X_clean.columns, index=X_clean.index)
            else:
                X_train = X_clean

            # Train the model
            self.model.fit(X_train, y_clean)

            # Calculate training metrics
            y_pred = self.model.predict(X_train)

            training_mse = mean_squared_error(y_clean, y_pred)
            training_mae = mean_absolute_error(y_clean, y_pred)
            training_r2 = r2_score(y_clean, y_pred)
            training_rmse = np.sqrt(training_mse)

            # Get feature importance if available
            if hasattr(self.model, "feature_importances_"):
                self.feature_importance_ = pd.Series(
                    self.model.feature_importances_, index=X_train.columns
                ).sort_values(ascending=False)

            # Calculate volatility-specific metrics
            volatility_accuracy = self._calculate_volatility_accuracy(y_clean, y_pred)
            directional_volatility_accuracy = self._calculate_directional_volatility_accuracy(
                y_clean, y_pred
            )

            # Training metrics
            metrics = {
                "training_mse": training_mse,
                "training_mae": training_mae,
                "training_rmse": training_rmse,
                "training_r2": training_r2,
                "volatility_accuracy": volatility_accuracy,
                "directional_volatility_accuracy": directional_volatility_accuracy,
                "algorithm": self.algorithm,
                "volatility_method": self.volatility_method,
                "forecast_horizon": self.forecast_horizon,
                "volatility_window": self.volatility_window,
                "volatility_stats": self.volatility_stats_,
                "feature_count": len(X_train.columns),
                "training_samples": len(X_train),
            }

            # Add algorithm-specific metrics
            if hasattr(self.model, "feature_importances_"):
                metrics["top_features"] = self.feature_importance_.head(10).to_dict()

            self.is_trained = True

            logger.info(
                "Volatility forecaster training completed",
                training_rmse=training_rmse,
                training_r2=training_r2,
                volatility_accuracy=volatility_accuracy,
                algorithm=self.algorithm,
                model_name=self.model_name,
            )

            return metrics

        except Exception as e:
            logger.error(
                "Volatility forecaster training failed", algorithm=self.algorithm, error=str(e)
            )
            raise ValidationError(f"Training failed: {e}") from e

    @time_execution
    @log_calls
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make volatility predictions.

        Args:
            X: Feature data

        Returns:
            Array of predicted volatility values

        Raises:
            ValidationError: If prediction fails
        """
        try:
            if not self.is_trained:
                raise ValidationError("Model must be trained before prediction")

            if X.empty:
                raise ValidationError("Feature data cannot be empty")

            # Scale features if scaler was used during training
            if self.scaler:
                X_scaled = self.scaler.transform(X)
                X_pred = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            else:
                X_pred = X

            predictions = self.model.predict(X_pred)

            # Ensure predictions are non-negative (volatility must be positive)
            predictions = np.maximum(predictions, 0)

            logger.debug(
                "Volatility predictions made",
                prediction_count=len(predictions),
                mean_prediction=np.mean(predictions),
                model_name=self.model_name,
            )

            return predictions

        except Exception as e:
            logger.error("Volatility prediction failed", error=str(e), model_name=self.model_name)
            raise ValidationError(f"Prediction failed: {e}") from e

    @time_execution
    @log_calls
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

            # Convert price data to volatility targets
            volatility_targets = self._calculate_volatility_targets(y)

            # Align features with volatility targets
            min_length = min(len(X), len(volatility_targets))
            X_aligned = X.iloc[:min_length]
            y_volatility = volatility_targets[:min_length]

            # Remove NaN values
            valid_mask = ~(np.isnan(y_volatility) | np.isinf(y_volatility))
            X_test = X_aligned[valid_mask]
            y_true = y_volatility[valid_mask]

            if len(X_test) == 0:
                raise ValidationError("No valid test data after cleaning")

            # Make predictions
            y_pred = self.predict(X_test)

            # Align predictions with true values
            min_pred_length = min(len(y_true), len(y_pred))
            y_true_aligned = y_true[:min_pred_length]
            y_pred_aligned = y_pred[:min_pred_length]

            # Calculate standard metrics
            test_mse = mean_squared_error(y_true_aligned, y_pred_aligned)
            test_mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
            test_r2 = r2_score(y_true_aligned, y_pred_aligned)
            test_rmse = np.sqrt(test_mse)

            # Calculate volatility-specific metrics
            volatility_accuracy = self._calculate_volatility_accuracy(
                y_true_aligned, y_pred_aligned
            )
            directional_volatility_accuracy = self._calculate_directional_volatility_accuracy(
                y_true_aligned, y_pred_aligned
            )

            # Relative error metrics
            mape = (
                np.mean(np.abs((y_true_aligned - y_pred_aligned) / (y_true_aligned + 1e-8))) * 100
            )

            # Volatility regime accuracy (low, medium, high)
            regime_accuracy = self._calculate_volatility_regime_accuracy(
                y_true_aligned, y_pred_aligned
            )

            metrics = {
                "test_mse": test_mse,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "test_mape": mape,
                "volatility_accuracy": volatility_accuracy,
                "directional_volatility_accuracy": directional_volatility_accuracy,
                "volatility_regime_accuracy": regime_accuracy,
                "algorithm": self.algorithm,
                "volatility_method": self.volatility_method,
                "test_samples": len(X_test),
            }

            logger.info(
                "Volatility forecaster evaluation completed",
                test_rmse=test_rmse,
                test_r2=test_r2,
                volatility_accuracy=volatility_accuracy,
                model_name=self.model_name,
            )

            return metrics

        except Exception as e:
            logger.error(
                "Volatility forecaster evaluation failed", error=str(e), model_name=self.model_name
            )
            raise ValidationError(f"Evaluation failed: {e}") from e

    def _calculate_volatility_targets(self, price_data: pd.Series) -> np.ndarray:
        """Calculate volatility targets from price data."""
        try:
            if self.volatility_method == "realized":
                return self._calculate_realized_volatility(price_data)
            elif self.volatility_method == "garch":
                return self._calculate_garch_volatility(price_data)
            elif self.volatility_method == "intraday":
                return self._calculate_intraday_volatility(price_data)
            else:
                raise ValidationError(f"Unknown volatility method: {self.volatility_method}")

        except Exception as e:
            logger.error(f"Failed to calculate volatility targets: {e}")
            raise ValidationError(f"Volatility calculation failed: {e}") from e

    def _calculate_realized_volatility(self, price_data: pd.Series) -> np.ndarray:
        """Calculate realized volatility using rolling standard deviation."""
        try:
            # Calculate returns
            if self.use_log_returns:
                returns = np.log(price_data / price_data.shift(1))
            else:
                returns = price_data.pct_change()

            # Calculate rolling volatility
            volatility = returns.rolling(window=self.volatility_window).std()

            # Annualize volatility (assuming daily data)
            volatility = volatility * np.sqrt(252)

            # Shift to create forecast targets
            if self.forecast_horizon > 0:
                volatility = volatility.shift(-self.forecast_horizon)

            return volatility.values

        except Exception as e:
            logger.error(f"Realized volatility calculation failed: {e}")
            raise ValidationError(f"Realized volatility calculation failed: {e}") from e

    def _calculate_garch_volatility(self, price_data: pd.Series) -> np.ndarray:
        """Calculate GARCH-style volatility (simplified exponential smoothing)."""
        try:
            # Calculate returns
            if self.use_log_returns:
                returns = np.log(price_data / price_data.shift(1))
            else:
                returns = price_data.pct_change()

            # Remove NaN values
            returns = returns.dropna()

            # GARCH parameters (simplified)
            alpha = 0.1  # Weight on squared return
            beta = 0.8  # Weight on previous variance

            # Initialize volatility
            volatility = np.zeros(len(returns))
            volatility[0] = returns.std()  # Initial volatility

            # Calculate GARCH volatility
            for i in range(1, len(returns)):
                volatility[i] = np.sqrt(
                    alpha * returns.iloc[i - 1] ** 2
                    + beta * volatility[i - 1] ** 2
                    + (1 - alpha - beta) * returns.var()
                )

            # Annualize volatility
            volatility = volatility * np.sqrt(252)

            # Create full series with original index
            full_volatility = pd.Series(np.nan, index=price_data.index)
            full_volatility.iloc[1 : len(volatility) + 1] = volatility

            # Shift to create forecast targets
            if self.forecast_horizon > 0:
                full_volatility = full_volatility.shift(-self.forecast_horizon)

            return full_volatility.values

        except Exception as e:
            logger.error(f"GARCH volatility calculation failed: {e}")
            raise ValidationError(f"GARCH volatility calculation failed: {e}") from e

    def _calculate_intraday_volatility(self, price_data: pd.Series) -> np.ndarray:
        """Calculate intraday volatility (simplified high-low approach)."""
        try:
            # This is a simplified version - in practice, you'd need OHLC data
            # Using rolling range as a proxy for intraday volatility

            # Calculate rolling high-low range
            rolling_high = price_data.rolling(window=self.volatility_window).max()
            rolling_low = price_data.rolling(window=self.volatility_window).min()

            # Parkinson estimator approximation
            volatility = (rolling_high - rolling_low) / rolling_low

            # Annualize volatility
            volatility = volatility * np.sqrt(252)

            # Shift to create forecast targets
            if self.forecast_horizon > 0:
                volatility = volatility.shift(-self.forecast_horizon)

            return volatility.values

        except Exception as e:
            logger.error(f"Intraday volatility calculation failed: {e}")
            raise ValidationError(f"Intraday volatility calculation failed: {e}") from e

    def _calculate_volatility_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate volatility-specific accuracy metric."""
        try:
            # Use relative accuracy within a tolerance band
            tolerance = 0.2  # 20% tolerance

            relative_error = np.abs((y_true - y_pred) / (y_true + 1e-8))
            accurate_predictions = (relative_error <= tolerance).sum()

            return accurate_predictions / len(y_true)

        except Exception as e:
            logger.error(f"Volatility accuracy calculation failed: {e}")
            return 0.0

    def _calculate_directional_volatility_accuracy(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Calculate directional volatility accuracy (increasing vs decreasing)."""
        try:
            if len(y_true) < 2:
                return 0.0

            # Calculate volatility changes
            true_changes = np.diff(y_true)
            pred_changes = np.diff(y_pred)

            # Check directional agreement
            directional_agreement = (
                (true_changes > 0) & (pred_changes > 0) | (true_changes < 0) & (pred_changes < 0)
            ).sum()

            return directional_agreement / len(true_changes)

        except Exception as e:
            logger.error(f"Directional volatility accuracy calculation failed: {e}")
            return 0.0

    def _calculate_volatility_regime_accuracy(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Calculate accuracy for volatility regime classification."""
        try:
            # Define volatility regimes based on percentiles
            low_threshold = np.percentile(y_true, 33)
            high_threshold = np.percentile(y_true, 67)

            # Classify true volatility regimes
            true_regimes = np.zeros(len(y_true))
            true_regimes[y_true > high_threshold] = 2  # High
            true_regimes[y_true <= low_threshold] = 0  # Low
            true_regimes[(y_true > low_threshold) & (y_true <= high_threshold)] = 1  # Medium

            # Classify predicted volatility regimes
            pred_regimes = np.zeros(len(y_pred))
            pred_regimes[y_pred > high_threshold] = 2  # High
            pred_regimes[y_pred <= low_threshold] = 0  # Low
            pred_regimes[(y_pred > low_threshold) & (y_pred <= high_threshold)] = 1  # Medium

            # Calculate regime accuracy
            regime_accuracy = (true_regimes == pred_regimes).sum() / len(y_true)

            return regime_accuracy

        except Exception as e:
            logger.error(f"Volatility regime accuracy calculation failed: {e}")
            return 0.0

    def get_feature_importance(self) -> pd.Series | None:
        """
        Get feature importance scores.

        Returns:
            Series with feature importance scores or None if not available
        """
        if not self.is_trained:
            logger.warning("Model not trained, no feature importance available")
            return None

        return self.feature_importance_

    def get_volatility_stats(self) -> dict[str, float] | None:
        """
        Get volatility statistics from training.

        Returns:
            Dictionary with volatility statistics or None if not trained
        """
        if not self.is_trained:
            logger.warning("Model not trained, no volatility statistics available")
            return None

        return self.volatility_stats_

    def predict_volatility_regime(self, X: pd.DataFrame) -> list[str]:
        """
        Predict volatility regime (low, medium, high).

        Args:
            X: Feature data

        Returns:
            List of volatility regime labels
        """
        try:
            predictions = self.predict(X)

            if self.volatility_stats_ is None:
                logger.warning("No volatility statistics available for regime classification")
                return ["unknown"] * len(predictions)

            # Use training statistics to define regimes
            low_threshold = self.volatility_stats_["q25"]
            high_threshold = self.volatility_stats_["q75"]

            regimes = []
            for pred in predictions:
                if pred <= low_threshold:
                    regimes.append("low")
                elif pred >= high_threshold:
                    regimes.append("high")
                else:
                    regimes.append("medium")

            return regimes

        except Exception as e:
            logger.error(f"Volatility regime prediction failed: {e}")
            raise ValidationError(f"Volatility regime prediction failed: {e}") from e
