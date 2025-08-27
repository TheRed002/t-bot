"""
Price Prediction Models for Financial Trading.

This module implements ML models specifically designed for predicting future prices
in financial markets using various algorithms and architectures.
GPU acceleration is used when available for improved performance.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.core.config import Config
from src.core.exceptions import ValidationError
from src.ml.models.base_model import BaseModel
from src.utils.gpu_utils import gpu_manager

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


class PricePredictor(BaseModel):
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
        config: Config,
        model_name: str = "price_predictor",
        version: str = "1.0.0",
        algorithm: str = "random_forest",
        prediction_horizon: int = 1,
        **model_params,
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
        super().__init__(config, model_name, version)

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

        self.logger.info(
            "Price predictor initialized",
            model_name=model_name,
            algorithm=algorithm,
            prediction_horizon=prediction_horizon,
        )

    def _get_model_type(self) -> str:
        """Return the model type identifier."""
        return "price_predictor"

    def _create_model(self, **kwargs):
        """Create and return the underlying ML model with GPU acceleration if available."""
        # Merge kwargs with stored model_params
        params = {**self.model_params, **kwargs}

        # Log GPU availability
        if gpu_manager.gpu_available:
            self.logger.info(f"GPU acceleration available for {self.algorithm} model")

        if self.algorithm == "linear":
            # Use GPU-accelerated version if available
            if CUML_AVAILABLE and gpu_manager.gpu_available:
                self.logger.info("Using cuML LinearRegression with GPU acceleration")
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
            if CUML_AVAILABLE and gpu_manager.gpu_available:
                self.logger.info("Using cuML RandomForestRegressor with GPU acceleration")
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
            if gpu_manager.gpu_available:
                default_params.update(
                    {"tree_method": "gpu_hist", "predictor": "gpu_predictor", "gpu_id": 0}
                )
                self.logger.info("Using XGBoost with GPU acceleration")

            default_params.update(params)
            return xgb.XGBRegressor(**default_params)

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
            if gpu_manager.gpu_available:
                default_params.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
                self.logger.info("Using LightGBM with GPU acceleration")

            default_params.update(params)
            return lgb.LGBMRegressor(**default_params)

        else:
            raise ValidationError(f"Unknown algorithm: {self.algorithm}")

    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and preprocess features for the model."""
        if X.empty:
            raise ValidationError("Features cannot be empty")

        # Check for required columns (basic validation)
        if len(X.columns) == 0:
            raise ValidationError("No feature columns found")

        # Handle missing values
        if X.isnull().any().any():
            self.logger.warning("Missing values found in features, filling with forward fill")
            X = X.fillna(method="ffill").fillna(method="bfill").fillna(0)

        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)

        # Ensure all columns are numeric
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                try:
                    X[col] = pd.to_numeric(X[col], errors="coerce")
                except (ValueError, TypeError) as e:
                    self.logger.warning(
                        f"Could not convert column {col} to numeric: {e}, dropping column"
                    )
                    X = X.drop(columns=[col])
                except Exception as e:
                    self.logger.error(f"Unexpected error converting column {col}: {e}")
                    raise DataValidationError(
                        f"Failed to process column {col}",
                        validation_rule="numeric_conversion",
                        invalid_fields=[col],
                    ) from e

        # Fill any remaining NaN values from conversion
        X = X.fillna(0)

        return X

    def _validate_targets(self, y: pd.Series) -> pd.Series:
        """Validate and preprocess targets for the model."""
        if y.empty:
            raise ValidationError("Targets cannot be empty")

        # Handle missing values
        if y.isnull().any():
            self.logger.warning("Missing values found in targets, filling with forward fill")
            y = y.fillna(method="ffill").fillna(method="bfill").fillna(0)

        # Handle infinite values
        y = y.replace([np.inf, -np.inf], 0)

        # Ensure targets are numeric
        if not pd.api.types.is_numeric_dtype(y):
            try:
                y = pd.to_numeric(y, errors="coerce")
                y = y.fillna(0)
            except (ValueError, TypeError) as e:
                raise DataValidationError(
                    "Could not convert targets to numeric",
                    validation_rule="target_numeric_conversion",
                    invalid_fields=["targets"],
                ) from e
            except Exception as e:
                self.logger.error(f"Unexpected error in target validation: {e}")
                raise ModelError(
                    "Critical error in target preprocessing",
                    model_name=self.__class__.__name__,
                ) from e

        return y

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Calculate model-specific performance metrics."""
        try:
            metrics = {
                "mae": mean_absolute_error(y_true, y_pred),
                "mse": mean_squared_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "r2": r2_score(y_true, y_pred),
            }

            # Calculate directional accuracy (important for trading)
            y_true_diff = np.diff(y_true)
            y_pred_diff = np.diff(y_pred)
            directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
            metrics["directional_accuracy"] = directional_accuracy

            # Calculate mean absolute percentage error
            non_zero_mask = y_true != 0
            if np.any(non_zero_mask):
                mape = (
                    np.mean(
                        np.abs(
                            (y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]
                        )
                    )
                    * 100
                )
                metrics["mape"] = mape
            else:
                metrics["mape"] = 0.0

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to calculate metrics: {e}")
            return {
                "mae": float("inf"),
                "mse": float("inf"),
                "rmse": float("inf"),
                "r2": -float("inf"),
                "directional_accuracy": 0.0,
                "mape": float("inf"),
            }

    def create_target_from_prices(
        self, prices: pd.Series, target_type: str = "return", horizon: int | None = None
    ) -> pd.Series:
        """
        Create prediction targets from price series.

        Args:
            prices: Price series
            target_type: Type of target ('return', 'price', 'log_return')
            horizon: Prediction horizon (uses instance default if None)

        Returns:
            Target series for prediction
        """
        horizon = horizon or self.prediction_horizon

        if target_type == "price":
            # Predict future price directly
            targets = prices.shift(-horizon)

        elif target_type == "return":
            # Predict future return
            future_prices = prices.shift(-horizon)
            targets = (future_prices / prices) - 1

        elif target_type == "log_return":
            # Predict future log return
            future_prices = prices.shift(-horizon)
            targets = np.log(future_prices / prices)

        else:
            raise ValidationError(f"Unknown target type: {target_type}")

        # Remove last N values that don't have future data
        targets = targets.iloc[:-horizon]

        return targets

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
        self, y_true: np.ndarray, y_pred: np.ndarray, transaction_cost: float = 0.001
    ) -> dict[str, float]:
        """
        Calculate trading-specific performance metrics.

        Args:
            y_true: True prices/returns
            y_pred: Predicted prices/returns
            transaction_cost: Transaction cost as percentage

        Returns:
            Trading performance metrics
        """
        # Calculate directional signals
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))

        # Directional accuracy
        directional_accuracy = np.mean(true_direction == pred_direction)

        # Calculate hypothetical returns from trading signals
        returns = np.diff(y_true) / y_true[:-1]  # Actual returns

        # Simple trading strategy: buy if prediction > current, sell if < current
        positions = pred_direction

        # Calculate strategy returns (simplified)
        strategy_returns = positions * returns[1:]  # Align arrays

        # Apply transaction costs
        position_changes = np.diff(np.concatenate([[0], positions]))
        transaction_costs = np.abs(position_changes) * transaction_cost
        strategy_returns_net = strategy_returns - transaction_costs[1:]

        # Calculate metrics
        metrics = {
            "directional_accuracy": directional_accuracy,
            "strategy_return": np.sum(strategy_returns_net),
            "strategy_sharpe": (
                np.mean(strategy_returns_net) / np.std(strategy_returns_net)
                if np.std(strategy_returns_net) > 0
                else 0
            ),
            "max_drawdown": self._calculate_max_drawdown(np.cumsum(strategy_returns_net)),
            "hit_rate": np.mean(strategy_returns_net > 0),
            "avg_win": (
                np.mean(strategy_returns_net[strategy_returns_net > 0])
                if np.any(strategy_returns_net > 0)
                else 0
            ),
            "avg_loss": (
                np.mean(strategy_returns_net[strategy_returns_net < 0])
                if np.any(strategy_returns_net < 0)
                else 0
            ),
        }

        return metrics

    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns."""
        if len(cumulative_returns) == 0:
            return 0.0

        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown)

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
