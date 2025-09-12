"""
ML Data Validation Utilities.

This module provides common validation functions for ML models to eliminate
duplicate validation code across the ML module.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.core.exceptions import DataValidationError, ValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)


def validate_features(X: pd.DataFrame, model_name: str = "Unknown") -> pd.DataFrame:
    """
    Validate and preprocess features for ML models.

    Args:
        X: Feature data to validate
        model_name: Name of the model for error context

    Returns:
        Cleaned feature DataFrame

    Raises:
        ValidationError: If features are invalid
    """
    if X.empty:
        raise ValidationError(f"{model_name}: Features cannot be empty")

    # Check for required columns (basic validation)
    if len(X.columns) == 0:
        raise ValidationError(f"{model_name}: No feature columns found")

    X_clean = X.copy()

    # Handle missing values
    if X_clean.isnull().any().any():
        logger.warning(f"{model_name}: Missing values found in features, filling with forward fill")
        X_clean = X_clean.ffill().bfill().fillna(0)

    # Handle infinite values
    X_clean = X_clean.replace([np.inf, -np.inf], 0)

    # Ensure all columns are numeric
    for col in X_clean.columns:
        if not pd.api.types.is_numeric_dtype(X_clean[col]):
            try:
                X_clean[col] = pd.to_numeric(X_clean[col], errors="coerce")
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"{model_name}: Could not convert column {col} to numeric: {e}, dropping column"
                )
                X_clean = X_clean.drop(columns=[col])
            except Exception as e:
                # Apply consistent error propagation patterns
                from src.utils.messaging_patterns import ErrorPropagationMixin

                logger.error(f"{model_name}: Unexpected error converting column {col}: {e}")

                # Use consistent error propagation
                error_propagator = ErrorPropagationMixin()
                validation_error = DataValidationError(
                    f"Failed to process column {col} in {model_name}",
                    validation_rule="numeric_conversion",
                    invalid_fields=[col],
                )

                try:
                    error_propagator.propagate_validation_error(
                        validation_error, "ml_validation.validate_features"
                    )
                except Exception as prop_error:
                    # Fallback if error propagation fails - log for debugging but don't break flow
                    logger.debug(f"Error propagation failed: {prop_error}")

                raise validation_error from e

    # Fill any remaining NaN values from conversion
    X_clean = X_clean.fillna(0)

    return X_clean


def validate_targets(y: pd.Series, model_name: str = "Unknown") -> pd.Series:
    """
    Validate and preprocess targets for ML models.

    Args:
        y: Target data to validate
        model_name: Name of the model for error context

    Returns:
        Cleaned target Series

    Raises:
        ValidationError: If targets are invalid
    """
    if y.empty:
        raise ValidationError(f"{model_name}: Targets cannot be empty")

    y_clean = y.copy()

    # Handle missing values
    if y_clean.isnull().any():
        logger.warning(f"{model_name}: Missing values found in targets, filling with forward fill")
        y_clean = y_clean.ffill().bfill().fillna(0)

    # Handle infinite values
    y_clean = y_clean.replace([np.inf, -np.inf], 0)

    # Ensure targets are numeric
    if not pd.api.types.is_numeric_dtype(y_clean):
        try:
            original_values = y_clean.copy()
            y_clean = pd.to_numeric(y_clean, errors="coerce")

            # Check if conversion resulted in too many NaNs (indicating conversion failure)
            if y_clean.isnull().sum() > len(original_values) * 0.5:  # More than 50% NaN
                raise ValueError("Conversion to numeric resulted in too many invalid values")

            y_clean = y_clean.fillna(0)
        except (ValueError, TypeError) as e:
            raise DataValidationError(
                f"{model_name}: Could not convert targets to numeric",
                validation_rule="target_numeric_conversion",
                invalid_fields=["targets"],
            ) from e
        except Exception as e:
            logger.error(f"{model_name}: Unexpected error in target validation: {e}")
            raise DataValidationError(
                f"{model_name}: Critical error in target preprocessing",
                validation_rule="target_preprocessing",
                invalid_fields=["targets"],
            ) from e

    return y_clean


def validate_training_data(
    X: pd.DataFrame, y: pd.Series, model_name: str = "Unknown"
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Validate both features and targets for training.

    Args:
        X: Feature data
        y: Target data
        model_name: Name of the model for error context

    Returns:
        Tuple of (validated_features, validated_targets)

    Raises:
        ValidationError: If data is invalid
    """
    if len(X) != len(y):
        raise ValidationError(f"{model_name}: Feature and target data must have same length")

    X_clean = validate_features(X, model_name)
    y_clean = validate_targets(y, model_name)

    # Final alignment check
    if len(X_clean) != len(y_clean):
        min_len = min(len(X_clean), len(y_clean))
        X_clean = X_clean.iloc[:min_len]
        y_clean = y_clean.iloc[:min_len]
        logger.warning(f"{model_name}: Aligned data to {min_len} samples after cleaning")

    return X_clean, y_clean


def validate_market_data(
    data: pd.DataFrame, required_columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Validate market data has required columns and proper format.

    Args:
        data: Market data DataFrame
        required_columns: List of required column names

    Returns:
        Validated market data

    Raises:
        ValidationError: If market data is invalid
    """
    if data.empty:
        raise ValidationError("Market data cannot be empty")

    if required_columns is None:
        required_columns = ["open", "high", "low", "close", "volume"]

    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValidationError(f"Missing required columns: {missing_columns}")

    # Check for non-positive prices
    price_columns = [col for col in ["open", "high", "low", "close"] if col in data.columns]
    for col in price_columns:
        if (data[col] <= 0).any():
            raise ValidationError(f"Non-positive values found in {col}")

    # Check for invalid OHLC relationships
    if all(col in data.columns for col in ["open", "high", "low", "close"]):
        invalid_high = data["high"] < data[["open", "close"]].max(axis=1)
        invalid_low = data["low"] > data[["open", "close"]].min(axis=1)

        if invalid_high.any():
            raise ValidationError("High price is lower than open/close in some records")
        if invalid_low.any():
            raise ValidationError("Low price is higher than open/close in some records")

    return data


def validate_prediction_data(X: pd.DataFrame, model_name: str = "Unknown") -> pd.DataFrame:
    """
    Validate data for prediction (no targets required).
    
    Args:
        X: Feature data
        model_name: Name of the model for error context
        
    Returns:
        Validated feature DataFrame
        
    Raises:
        ValidationError: If prediction data is invalid
    """
    if X.empty:
        raise ValidationError(f"{model_name}: Prediction data cannot be empty")

    return validate_features(X, model_name)


def validate_direction_threshold(threshold: float, model_name: str = "Unknown") -> float:
    """
    Validate direction classification threshold.
    
    Args:
        threshold: Direction threshold value
        model_name: Name of the model for error context
        
    Returns:
        Validated threshold
        
    Raises:
        ValidationError: If threshold is invalid
    """
    if not isinstance(threshold, (int, float)):
        raise ValidationError(f"{model_name}: Direction threshold must be numeric")

    if threshold <= 0 or threshold >= 1:
        raise ValidationError(f"{model_name}: Direction threshold must be between 0 and 1")

    return float(threshold)


def validate_prediction_horizon(horizon: int, model_name: str = "Unknown") -> int:
    """
    Validate prediction horizon parameter.
    
    Args:
        horizon: Prediction horizon value
        model_name: Name of the model for error context
        
    Returns:
        Validated horizon
        
    Raises:
        ValidationError: If horizon is invalid
    """
    if not isinstance(horizon, int):
        raise ValidationError(f"{model_name}: Prediction horizon must be integer")

    if horizon <= 0:
        raise ValidationError(f"{model_name}: Prediction horizon must be positive")

    if horizon > 100:  # Reasonable upper bound
        raise ValidationError(f"{model_name}: Prediction horizon too large (max 100)")

    return horizon


def validate_algorithm_choice(algorithm: str, allowed_algorithms: list[str], model_name: str = "Unknown") -> str:
    """
    Validate algorithm choice against allowed options.
    
    Args:
        algorithm: Algorithm name
        allowed_algorithms: List of allowed algorithm names
        model_name: Name of the model for error context
        
    Returns:
        Validated algorithm name
        
    Raises:
        ValidationError: If algorithm is not allowed
    """
    if algorithm not in allowed_algorithms:
        raise ValidationError(
            f"{model_name}: Unknown algorithm '{algorithm}'. Allowed: {allowed_algorithms}"
        )

    return algorithm


def validate_class_weights(class_weights: Any, model_name: str = "Unknown") -> Any:
    """
    Validate class weights parameter.
    
    Args:
        class_weights: Class weights specification
        model_name: Name of the model for error context
        
    Returns:
        Validated class weights
        
    Raises:
        ValidationError: If class weights are invalid
    """
    if class_weights is None:
        return None

    if isinstance(class_weights, str):
        if class_weights not in ["balanced", "balanced_subsample"]:
            raise ValidationError(
                f"{model_name}: Invalid class weights string. Use 'balanced' or 'balanced_subsample'"
            )
    elif isinstance(class_weights, dict):
        if not all(isinstance(k, (int, str)) and isinstance(v, (int, float)) for k, v in class_weights.items()):
            raise ValidationError(f"{model_name}: Class weights dict must have numeric values")
    else:
        raise ValidationError(f"{model_name}: Class weights must be None, string, or dict")

    return class_weights


def check_data_quality(
    X: pd.DataFrame,
    y: pd.Series = None,
    max_missing_pct: float = 0.1,
    max_constant_pct: float = 0.95,
) -> dict[str, Any]:
    """
    Check data quality and return quality metrics.

    Args:
        X: Feature data
        y: Optional target data
        max_missing_pct: Maximum allowed missing data percentage
        max_constant_pct: Maximum allowed constant values percentage

    Returns:
        Dictionary with quality metrics and warnings
    """
    quality_report = {
        "total_samples": len(X),
        "total_features": len(X.columns),
        "warnings": [],
        "passed": True,
    }

    # Check missing data
    missing_pct = X.isnull().sum() / len(X)
    high_missing_features = missing_pct[missing_pct > max_missing_pct]

    if not high_missing_features.empty:
        quality_report["warnings"].append(
            f"Features with high missing data: {high_missing_features.to_dict()}"
        )
        quality_report["passed"] = False

    # Check constant features
    constant_features = []
    for col in X.columns:
        if X[col].nunique() == 1:
            constant_features.append(col)
        elif (X[col].value_counts().iloc[0] / len(X)) > max_constant_pct:
            constant_features.append(col)

    if constant_features:
        quality_report["warnings"].append(f"Constant/near-constant features: {constant_features}")
        quality_report["passed"] = False

    # Check target distribution if provided
    if y is not None:
        if y.nunique() == 1:
            quality_report["warnings"].append("Target has only one unique value")
            quality_report["passed"] = False
        elif y.nunique() < 0.1 * len(y):
            quality_report["warnings"].append("Target has very few unique values")

    quality_report["missing_data"] = missing_pct.to_dict()
    quality_report["constant_features"] = constant_features

    return quality_report


def validate_model_hyperparameters(params: dict[str, Any], model_type: str, model_name: str = "Unknown") -> dict[str, Any]:
    """
    Validate model hyperparameters based on model type.
    
    Args:
        params: Hyperparameter dictionary
        model_type: Type of model
        model_name: Name of the model for error context
        
    Returns:
        Validated hyperparameters
        
    Raises:
        ValidationError: If hyperparameters are invalid
    """
    if not isinstance(params, dict):
        raise ValidationError(f"{model_name}: Hyperparameters must be a dictionary")

    validated_params = params.copy()

    # Common validations
    if "random_state" in validated_params:
        if not isinstance(validated_params["random_state"], (int, type(None))):
            raise ValidationError(f"{model_name}: random_state must be integer or None")

    if "n_estimators" in validated_params:
        if not isinstance(validated_params["n_estimators"], int) or validated_params["n_estimators"] <= 0:
            raise ValidationError(f"{model_name}: n_estimators must be positive integer")

    if "max_depth" in validated_params:
        if validated_params["max_depth"] is not None:
            if not isinstance(validated_params["max_depth"], int) or validated_params["max_depth"] <= 0:
                raise ValidationError(f"{model_name}: max_depth must be positive integer or None")

    if "learning_rate" in validated_params:
        if not isinstance(validated_params["learning_rate"], (int, float)) or validated_params["learning_rate"] <= 0:
            raise ValidationError(f"{model_name}: learning_rate must be positive number")

    return validated_params
