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
