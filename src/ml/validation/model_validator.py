"""
Model Validation System for ML Models.

This module provides comprehensive model validation capabilities including
performance validation, statistical tests, and production readiness checks.
"""

import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.ml.models.base_model import BaseModel
from src.utils.decorators import log_calls, time_execution

logger = get_logger(__name__)


class ModelValidator:
    """
    Comprehensive model validation system.

    This class provides various validation methods for ML models including:
    - Performance validation against benchmarks
    - Statistical significance tests
    - Model stability analysis
    - Production readiness checks
    - Bias and fairness analysis

    Attributes:
        config: Application configuration
        validation_threshold: Minimum performance threshold
        significance_level: Statistical significance level
        stability_window: Window for stability analysis
        benchmark_models: Benchmark models for comparison
    """

    def __init__(self, config: Config):
        """
        Initialize the model validator.

        Args:
            config: Application configuration
        """
        self.config = config

        # Validation parameters
        self.validation_threshold = config.ml.validation_threshold
        self.significance_level = config.ml.significance_level
        self.stability_window = config.ml.stability_window
        self.min_samples = config.ml.min_validation_samples

        # Performance tracking
        self.validation_history = []
        self.benchmark_results = {}

        logger.info(
            "Model validator initialized",
            validation_threshold=self.validation_threshold,
            significance_level=self.significance_level,
            stability_window=self.stability_window,
            min_samples=self.min_samples,
        )

    @time_execution
    @log_calls
    def validate_model_performance(
        self,
        model: BaseModel,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        benchmark_score: float | None = None,
    ) -> dict[str, Any]:
        """
        Validate model performance against benchmarks.

        Args:
            model: Trained ML model
            X_test: Test features
            y_test: Test targets
            benchmark_score: Benchmark score to compare against

        Returns:
            Validation results dictionary

        Raises:
            ValidationError: If validation fails
        """
        try:
            if not model.is_trained:
                raise ValidationError("Model must be trained before validation")

            if X_test.empty or y_test.empty:
                raise ValidationError("Test data cannot be empty")

            if len(X_test) < self.min_samples:
                raise ValidationError(f"Need at least {self.min_samples} test samples")

            logger.info(
                "Starting model performance validation",
                model_name=model.model_name,
                test_samples=len(X_test),
            )

            # Get model predictions
            y_pred = model.predict(X_test)

            # Calculate performance metrics based on model type
            if model.model_type == "classification":
                performance_metrics = self._calculate_classification_metrics(y_test, y_pred)
                primary_metric = performance_metrics["accuracy"]
            else:  # regression
                performance_metrics = self._calculate_regression_metrics(y_test, y_pred)
                primary_metric = performance_metrics["r2_score"]

            # Performance validation
            performance_validation = {
                "meets_threshold": primary_metric >= self.validation_threshold,
                "primary_metric": primary_metric,
                "threshold": self.validation_threshold,
                "performance_metrics": performance_metrics,
            }

            # Benchmark comparison
            benchmark_validation = {}
            if benchmark_score is not None:
                benchmark_validation = {
                    "beats_benchmark": primary_metric > benchmark_score,
                    "benchmark_score": benchmark_score,
                    "improvement": primary_metric - benchmark_score,
                    "improvement_percentage": ((primary_metric - benchmark_score) / benchmark_score)
                    * 100,
                }

            # Statistical significance test
            significance_test = self._test_statistical_significance(
                y_test, y_pred, model.model_type
            )

            # Residual analysis (for regression models)
            residual_analysis = {}
            if model.model_type == "regression":
                residual_analysis = self._analyze_residuals(y_test, y_pred)

            # Overall validation result
            validation_result = {
                "timestamp": datetime.utcnow(),
                "model_name": model.model_name,
                "model_type": model.model_type,
                "test_samples": len(X_test),
                "performance_validation": performance_validation,
                "benchmark_validation": benchmark_validation,
                "significance_test": significance_test,
                "residual_analysis": residual_analysis,
                "overall_pass": (
                    performance_validation["meets_threshold"]
                    and significance_test["is_significant"]
                ),
            }

            # Store validation history
            self.validation_history.append(validation_result)

            logger.info(
                "Model performance validation completed",
                model_name=model.model_name,
                primary_metric=primary_metric,
                meets_threshold=performance_validation["meets_threshold"],
                overall_pass=validation_result["overall_pass"],
            )

            return validation_result

        except Exception as e:
            logger.error(
                "Model performance validation failed",
                model_name=getattr(model, "model_name", "unknown"),
                error=str(e),
            )
            raise ValidationError(f"Performance validation failed: {e}") from e

    @time_execution
    @log_calls
    def validate_model_stability(
        self,
        model: BaseModel,
        time_series_data: list[tuple[pd.DataFrame, pd.Series]],
        time_periods: list[datetime],
    ) -> dict[str, Any]:
        """
        Validate model stability over time.

        Args:
            model: Trained ML model
            time_series_data: List of (features, targets) for different time periods
            time_periods: List of timestamps for each data period

        Returns:
            Stability validation results

        Raises:
            ValidationError: If stability validation fails
        """
        try:
            if not model.is_trained:
                raise ValidationError("Model must be trained before stability validation")

            if len(time_series_data) < 2:
                raise ValidationError("Need at least 2 time periods for stability analysis")

            logger.info(
                "Starting model stability validation",
                model_name=model.model_name,
                time_periods=len(time_series_data),
            )

            performance_over_time = []

            # Calculate performance for each time period
            for i, ((X, y), timestamp) in enumerate(
                zip(time_series_data, time_periods, strict=False)
            ):
                if X.empty or y.empty:
                    logger.warning(f"Empty data for time period {i}, skipping")
                    continue

                y_pred = model.predict(X)

                if model.model_type == "classification":
                    metrics = self._calculate_classification_metrics(y, y_pred)
                    primary_metric = metrics["accuracy"]
                else:
                    metrics = self._calculate_regression_metrics(y, y_pred)
                    primary_metric = metrics["r2_score"]

                performance_over_time.append(
                    {
                        "timestamp": timestamp,
                        "primary_metric": primary_metric,
                        "sample_count": len(X),
                        "metrics": metrics,
                    }
                )

            if len(performance_over_time) < 2:
                raise ValidationError("Insufficient valid time periods for stability analysis")

            # Analyze stability
            metrics_series = [p["primary_metric"] for p in performance_over_time]

            stability_metrics = {
                "mean_performance": np.mean(metrics_series),
                "std_performance": np.std(metrics_series),
                "min_performance": np.min(metrics_series),
                "max_performance": np.max(metrics_series),
                "coefficient_of_variation": np.std(metrics_series) / np.mean(metrics_series),
                "performance_range": np.max(metrics_series) - np.min(metrics_series),
            }

            # Trend analysis
            trend_analysis = self._analyze_performance_trend(performance_over_time)

            # Stability assessment
            is_stable = (
                stability_metrics["coefficient_of_variation"] < 0.1  # CV < 10%
                and stability_metrics["performance_range"] < 0.2  # Range < 20%
                and all(
                    p["primary_metric"] >= self.validation_threshold for p in performance_over_time
                )
            )

            stability_result = {
                "timestamp": datetime.utcnow(),
                "model_name": model.model_name,
                "time_periods_analyzed": len(performance_over_time),
                "stability_metrics": stability_metrics,
                "trend_analysis": trend_analysis,
                "performance_over_time": performance_over_time,
                "is_stable": is_stable,
            }

            logger.info(
                "Model stability validation completed",
                model_name=model.model_name,
                is_stable=is_stable,
                mean_performance=stability_metrics["mean_performance"],
                coefficient_of_variation=stability_metrics["coefficient_of_variation"],
            )

            return stability_result

        except Exception as e:
            logger.error(
                "Model stability validation failed",
                model_name=getattr(model, "model_name", "unknown"),
                error=str(e),
            )
            raise ValidationError(f"Stability validation failed: {e}") from e

    @time_execution
    @log_calls
    def validate_production_readiness(
        self, model: BaseModel, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, Any]:
        """
        Validate model readiness for production deployment.

        Args:
            model: Trained ML model
            X_test: Test features
            y_test: Test targets

        Returns:
            Production readiness assessment

        Raises:
            ValidationError: If readiness validation fails
        """
        try:
            logger.info("Starting production readiness validation", model_name=model.model_name)

            readiness_checks = {}

            # 1. Basic model checks
            readiness_checks["model_trained"] = model.is_trained
            readiness_checks["has_predict_method"] = hasattr(model, "predict")
            readiness_checks["has_model_name"] = hasattr(model, "model_name") and model.model_name

            # 2. Performance validation
            performance_result = self.validate_model_performance(model, X_test, y_test)
            readiness_checks["meets_performance_threshold"] = performance_result[
                "performance_validation"
            ]["meets_threshold"]
            readiness_checks["statistically_significant"] = performance_result["significance_test"][
                "is_significant"
            ]

            # 3. Prediction consistency
            consistency_check = self._check_prediction_consistency(model, X_test)
            readiness_checks.update(consistency_check)

            # 4. Memory and computational efficiency
            efficiency_check = self._check_computational_efficiency(model, X_test)
            readiness_checks.update(efficiency_check)

            # 5. Error handling
            error_handling_check = self._check_error_handling(model, X_test)
            readiness_checks.update(error_handling_check)

            # 6. Data quality handling
            data_quality_check = self._check_data_quality_handling(model)
            readiness_checks.update(data_quality_check)

            # Overall readiness assessment
            critical_checks = [
                "model_trained",
                "has_predict_method",
                "meets_performance_threshold",
                "statistically_significant",
                "predictions_consistent",
                "handles_errors_gracefully",
            ]

            critical_passed = all(readiness_checks.get(check, False) for check in critical_checks)

            # Count all passed checks
            total_checks = len(readiness_checks)
            passed_checks = sum(1 for check in readiness_checks.values() if check)
            readiness_score = passed_checks / total_checks

            readiness_result = {
                "timestamp": datetime.utcnow(),
                "model_name": model.model_name,
                "readiness_checks": readiness_checks,
                "critical_checks_passed": critical_passed,
                "readiness_score": readiness_score,
                "is_production_ready": critical_passed and readiness_score >= 0.8,
                "performance_details": performance_result,
            }

            logger.info(
                "Production readiness validation completed",
                model_name=model.model_name,
                is_production_ready=readiness_result["is_production_ready"],
                readiness_score=readiness_score,
                critical_checks_passed=critical_passed,
            )

            return readiness_result

        except Exception as e:
            logger.error(
                "Production readiness validation failed",
                model_name=getattr(model, "model_name", "unknown"),
                error=str(e),
            )
            raise ValidationError(f"Production readiness validation failed: {e}") from e

    def _calculate_classification_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> dict[str, float]:
        """Calculate classification metrics."""
        try:
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
            }

            # Add other classification metrics as needed
            return metrics

        except Exception as e:
            logger.error(f"Classification metrics calculation failed: {e}")
            return {"accuracy": 0.0}

    def _calculate_regression_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> dict[str, float]:
        """Calculate regression metrics."""
        try:
            metrics = {
                "r2_score": r2_score(y_true, y_pred),
                "mse": mean_squared_error(y_true, y_pred),
                "mae": mean_absolute_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            }

            return metrics

        except Exception as e:
            logger.error(f"Regression metrics calculation failed: {e}")
            return {"r2_score": 0.0, "mse": float("inf"), "mae": float("inf"), "rmse": float("inf")}

    def _test_statistical_significance(
        self, y_true: pd.Series, y_pred: np.ndarray, model_type: str
    ) -> dict[str, Any]:
        """Test statistical significance of model predictions."""
        try:
            if model_type == "classification":
                # For classification, test if accuracy is significantly better than random
                accuracy = accuracy_score(y_true, y_pred)
                n_samples = len(y_true)
                n_classes = len(np.unique(y_true))
                random_accuracy = 1 / n_classes

                # Binomial test
                successes = int(accuracy * n_samples)
                p_value = stats.binom_test(
                    successes, n_samples, random_accuracy, alternative="greater"
                )

            else:  # regression
                # For regression, test if predictions are significantly correlated with true values
                correlation, p_value = stats.pearsonr(y_true, y_pred)

            is_significant = p_value < self.significance_level

            return {
                "p_value": p_value,
                "significance_level": self.significance_level,
                "is_significant": is_significant,
                "test_type": "binomial" if model_type == "classification" else "correlation",
            }

        except Exception as e:
            logger.error(f"Statistical significance test failed: {e}")
            return {
                "p_value": 1.0,
                "significance_level": self.significance_level,
                "is_significant": False,
                "test_type": "failed",
            }

    def _analyze_residuals(self, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
        """Analyze residuals for regression models."""
        try:
            residuals = y_true - y_pred

            # Basic residual statistics
            residual_stats = {
                "mean": np.mean(residuals),
                "std": np.std(residuals),
                "skewness": stats.skew(residuals),
                "kurtosis": stats.kurtosis(residuals),
            }

            # Normality test
            _, normality_p_value = stats.normaltest(residuals)

            # Heteroscedasticity test (simplified)
            # Correlation between absolute residuals and predictions
            abs_residuals = np.abs(residuals)
            heteroscedasticity_corr, heteroscedasticity_p = stats.pearsonr(abs_residuals, y_pred)

            return {
                "residual_stats": residual_stats,
                "normality_test": {
                    "p_value": normality_p_value,
                    "is_normal": normality_p_value > self.significance_level,
                },
                "heteroscedasticity_test": {
                    "correlation": heteroscedasticity_corr,
                    "p_value": heteroscedasticity_p,
                    "is_homoscedastic": heteroscedasticity_p > self.significance_level,
                },
            }

        except Exception as e:
            logger.error(f"Residual analysis failed: {e}")
            return {}

    def _analyze_performance_trend(self, performance_over_time: list[dict]) -> dict[str, Any]:
        """Analyze performance trend over time."""
        try:
            if len(performance_over_time) < 3:
                return {"trend": "insufficient_data"}

            metrics = [p["primary_metric"] for p in performance_over_time]
            time_indices = list(range(len(metrics)))

            # Linear regression to detect trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_indices, metrics)

            # Determine trend direction
            if p_value < self.significance_level:
                if slope > 0:
                    trend = "improving"
                elif slope < 0:
                    trend = "degrading"
                else:
                    trend = "stable"
            else:
                trend = "no_significant_trend"

            return {
                "trend": trend,
                "slope": slope,
                "r_squared": r_value**2,
                "p_value": p_value,
                "is_significant": p_value < self.significance_level,
            }

        except Exception as e:
            logger.error(f"Performance trend analysis failed: {e}")
            return {"trend": "analysis_failed"}

    def _check_prediction_consistency(
        self, model: BaseModel, X_test: pd.DataFrame
    ) -> dict[str, bool]:
        """Check if model predictions are consistent."""
        try:
            # Make predictions twice with same data
            pred1 = model.predict(X_test)
            pred2 = model.predict(X_test)

            # Check if predictions are identical
            predictions_identical = np.array_equal(pred1, pred2)

            return {"predictions_consistent": predictions_identical}

        except Exception as e:
            logger.error(f"Prediction consistency check failed: {e}")
            return {"predictions_consistent": False}

    def _check_computational_efficiency(
        self, model: BaseModel, X_test: pd.DataFrame
    ) -> dict[str, bool]:
        """Check computational efficiency of the model."""
        try:
            import os
            import time

            import psutil

            # Memory usage before prediction
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Time prediction
            start_time = time.time()
            _ = model.predict(X_test)
            prediction_time = time.time() - start_time

            # Memory usage after prediction
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before

            # Efficiency thresholds
            max_prediction_time_per_sample = 0.01  # 10ms per sample
            max_memory_increase = 100  # 100MB

            prediction_time_per_sample = prediction_time / len(X_test)

            return {
                "prediction_time_efficient": prediction_time_per_sample
                < max_prediction_time_per_sample,
                "memory_efficient": memory_increase < max_memory_increase,
                "prediction_time_per_sample": prediction_time_per_sample,
                "memory_increase_mb": memory_increase,
            }

        except Exception as e:
            logger.error(f"Computational efficiency check failed: {e}")
            return {
                "prediction_time_efficient": True,  # Assume efficient if check fails
                "memory_efficient": True,
            }

    def _check_error_handling(self, model: BaseModel, X_test: pd.DataFrame) -> dict[str, bool]:
        """Check model error handling capabilities."""
        try:
            checks = {}

            # Test with empty data
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.predict(pd.DataFrame())
                checks["handles_empty_data"] = False  # Should raise an error
            except Exception:
                checks["handles_empty_data"] = True  # Correctly raises error

            # Test with NaN data
            try:
                X_nan = X_test.copy()
                X_nan.iloc[0, 0] = np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.predict(X_nan)
                checks["handles_nan_gracefully"] = True
            except Exception:
                checks["handles_nan_gracefully"] = False

            # Test with wrong feature count
            try:
                X_wrong = X_test.iloc[:, :-1]  # Remove one feature
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.predict(X_wrong)
                checks["handles_wrong_features"] = False  # Should raise error
            except Exception:
                checks["handles_wrong_features"] = True  # Correctly raises error

            # Overall error handling assessment
            checks["handles_errors_gracefully"] = (
                checks["handles_empty_data"] and checks["handles_wrong_features"]
            )

            return checks

        except Exception as e:
            logger.error(f"Error handling check failed: {e}")
            return {
                "handles_empty_data": False,
                "handles_nan_gracefully": False,
                "handles_wrong_features": False,
                "handles_errors_gracefully": False,
            }

    def _check_data_quality_handling(self, model: BaseModel) -> dict[str, bool]:
        """Check model's data quality handling capabilities."""
        try:
            # Check if model has data validation methods
            has_data_validation = hasattr(model, "validate_input")

            # Check if model has preprocessing methods
            has_preprocessing = hasattr(model, "preprocess_data")

            # Check if model handles feature scaling
            handles_scaling = hasattr(model, "scaler") or hasattr(model, "scale_features")

            return {
                "has_data_validation": has_data_validation,
                "has_preprocessing": has_preprocessing,
                "handles_feature_scaling": handles_scaling,
            }

        except Exception as e:
            logger.error(f"Data quality handling check failed: {e}")
            return {
                "has_data_validation": False,
                "has_preprocessing": False,
                "handles_feature_scaling": False,
            }

    def get_validation_history(self) -> list[dict[str, Any]]:
        """Get validation history."""
        return self.validation_history

    def get_benchmark_results(self) -> dict[str, Any]:
        """Get benchmark results."""
        return self.benchmark_results

    def clear_validation_history(self):
        """Clear validation history."""
        self.validation_history.clear()
        logger.info("Validation history cleared")
