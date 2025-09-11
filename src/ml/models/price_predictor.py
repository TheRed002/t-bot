"""
Price Prediction Models for Financial Trading.

This module implements ML models specifically designed for predicting future prices
in financial markets using various algorithms and architectures.
GPU acceleration is used when available for improved performance.
"""

from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.core.exceptions import ValidationError
from src.ml.models.base_model import BaseMLModel

# Try to import GPU-accelerated libraries
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from cuml.ensemble import RandomForestRegressor as cuRFRegressor
    from cuml.linear_model import LinearRegression as cuLinearRegression

    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False


class PricePredictor(BaseMLModel):
    """
    Price prediction model for financial instruments.

    This model predicts future prices using various ML algorithms including
    linear regression, random forest, and gradient boosting methods.

    Attributes:
        algorithm: ML algorithm to use ('linear', 'random_forest', 'xgboost')
        prediction_horizon: Number of time steps to predict ahead
        scaler: Feature scaler for preprocessing
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        model_name: str = "price_predictor",
        version: str = "1.0.0",
        algorithm: str = "random_forest",
        prediction_horizon: int = 1,
        correlation_id: str | None = None,
        **model_params: Any,
    ):
        """
        Initialize the price predictor.

        Args:
            config: Application configuration
            model_name: Name of the model
            version: Model version
            algorithm: ML algorithm to use
            prediction_horizon: Steps ahead to predict
            **model_params: Additional model parameters
        """
        super().__init__(model_name, version, config, correlation_id)

        self.algorithm = algorithm
        self.prediction_horizon = prediction_horizon
        self.scaler = StandardScaler()
        self.model_params = model_params

        # Update metadata
        self.metadata.update(
            {
                "algorithm": algorithm,
                "prediction_horizon": prediction_horizon,
                "model_params": model_params,
            }
        )

        self._logger.info(
            "Price predictor initialized",
            model_name=model_name,
            algorithm=algorithm,
            prediction_horizon=prediction_horizon,
        )

    def _get_model_type(self) -> str:
        """Return the model type identifier."""
        return "price_predictor"

    def _create_model(self, **kwargs: Any) -> Any:
        """Create and return the underlying ML model with GPU acceleration if available."""
        # Merge kwargs with stored model_params
        params = {**self.model_params, **kwargs}

        # GPU availability check
        gpu_available = CUML_AVAILABLE or (XGB_AVAILABLE and LGB_AVAILABLE)

        if gpu_available:
            self._logger.info(f"GPU acceleration available for {self.algorithm} model")

        if self.algorithm == "linear":
            # Use GPU-accelerated version if available
            if CUML_AVAILABLE:
                self._logger.info("Using cuML LinearRegression with GPU acceleration")
                return cuLinearRegression(**params)
            return LinearRegression(**params)

        elif self.algorithm == "random_forest":
            # Default parameters for random forest
            default_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
            }
            default_params.update(params)

            # Use GPU-accelerated version if available
            if CUML_AVAILABLE:
                self._logger.info("Using cuML RandomForestRegressor with GPU acceleration")
                # Adjust parameters for cuML
                cuml_params = default_params.copy()
                cuml_params.pop("min_samples_split", None)
                cuml_params.pop("min_samples_leaf", None)
                return cuRFRegressor(**cuml_params)
            return RandomForestRegressor(**default_params)

        elif self.algorithm == "xgboost":
            if not XGB_AVAILABLE:
                raise ImportError("XGBoost not available. Please install xgboost.")

            default_params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }

            # Add GPU parameters if available
            if XGB_AVAILABLE:
                try:
                    # Try to enable GPU acceleration
                    default_params.update(
                        {"tree_method": "gpu_hist", "predictor": "gpu_predictor", "gpu_id": 0}
                    )
                    self._logger.info("Using XGBoost with GPU acceleration")
                except Exception as e:
                    # Fall back to CPU if GPU not available
                    self._logger.info(f"GPU not available, using CPU for XGBoost: {e}")

            default_params.update(params)
            return xgb.XGBRegressor(**dict(default_params))

        elif self.algorithm == "lightgbm":
            if not LGB_AVAILABLE:
                raise ImportError("LightGBM not available. Please install lightgbm.")

            default_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "objective": "regression",
                "metric": "rmse",
            }

            # Add GPU parameters if available
            if LGB_AVAILABLE:
                try:
                    # Try to enable GPU acceleration
                    default_params.update(
                        {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}
                    )
                    self._logger.info("Using LightGBM with GPU acceleration")
                except Exception as e:
                    # Fall back to CPU if GPU not available
                    self._logger.info(f"GPU not available, using CPU for LightGBM: {e}")

            default_params.update(params)
            return lgb.LGBMRegressor(**dict(default_params))

        else:
            raise ValidationError(f"Unknown algorithm: {self.algorithm}")

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
        from src.utils.ml_metrics import calculate_regression_metrics
        return calculate_regression_metrics(y_true, y_pred)

    def create_target_from_prices(
        self, prices: pd.Series, target_type: str = "return", horizon: int | None = None
    ) -> pd.Series:
        """
        Create prediction targets from price series using utilities.

        Args:
            prices: Price series
            target_type: Type of target ('return', 'price', 'log_return')
            horizon: Prediction horizon (uses instance default if None)

        Returns:
            Target series for prediction
        """
        from src.utils.ml_data_transforms import create_returns_series

        horizon = horizon or self.prediction_horizon

        if target_type == "price":
            # Predict future price directly
            return prices.shift(-horizon).iloc[:-horizon]
        elif target_type in ["return", "log_return"]:
            return_type = "simple" if target_type == "return" else "log"
            return create_returns_series(prices, horizon, return_type)
        else:
            raise ValidationError(f"Unknown target type: {target_type}")

    def predict_price_sequence(self, X: pd.DataFrame, sequence_length: int = 10) -> np.ndarray:
        """
        Predict a sequence of future prices using iterative prediction.

        Args:
            X: Feature data (last observation)
            sequence_length: Number of future steps to predict

        Returns:
            Array of predicted prices
        """
        if not self.is_trained:
            raise ValidationError("Model must be trained before prediction")

        # Start with the last observation
        current_features = X.iloc[-1:].copy()
        predictions = []

        for _step in range(sequence_length):
            # Make prediction for current step
            pred = self.predict(current_features)[0]
            predictions.append(pred)

            # Update features for next prediction
            # This is a simplified approach - in practice, you'd update
            # the features based on the predicted price

            # For now, we'll just use the same features
            # In a real implementation, you'd update price-based features

        return np.array(predictions)

    def calculate_trading_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, transaction_cost: Decimal = Decimal("0.001")
    ) -> dict[str, Any]:
        """
        Calculate trading-specific performance metrics using utilities.

        Args:
            y_true: True prices/returns
            y_pred: Predicted prices/returns
            transaction_cost: Transaction cost as percentage

        Returns:
            Trading performance metrics
        """
        from src.utils.ml_metrics import calculate_trading_metrics
        return calculate_trading_metrics(y_true, y_pred, float(transaction_cost))


    def get_feature_importance_analysis(self) -> dict[str, Any]:
        """
        Get detailed feature importance analysis.

        Returns:
            Dictionary with feature importance analysis
        """
        if not self.is_trained or not hasattr(self.model, "feature_importances_"):
            return {}

        importance = self.get_feature_importance()
        if importance is None:
            return {}

        analysis = {
            "top_features": importance.head(10).to_dict(),
            "total_features": len(importance),
            "importance_concentration": {
                "top_5_sum": importance.head(5).sum(),
                "top_10_sum": importance.head(10).sum(),
                "top_20_pct": importance.head(int(len(importance) * 0.2)).sum(),
            },
        }

        return analysis
