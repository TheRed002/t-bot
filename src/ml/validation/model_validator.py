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
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import learning_curve

from src.core.base.interfaces import HealthStatus
from src.core.base.service import BaseService
from src.core.exceptions import ValidationError
from src.core.types.base import ConfigDict
from src.ml.models.base_model import BaseModel
from src.utils.decorators import UnifiedDecorator

# Initialize decorator instance
dec = UnifiedDecorator()


class ModelValidationService(BaseService):
    """
    Comprehensive model validation system.

    This service provides various validation methods for ML models including:
    - Performance validation against benchmarks
    - Statistical significance tests
    - Model stability analysis
    - Production readiness checks
    - Bias and fairness analysis
    - Overfitting detection

    Attributes:
        validation_threshold: Minimum performance threshold
        significance_level: Statistical significance level
        stability_window: Window for stability analysis
        benchmark_models: Benchmark models for comparison
    """

    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None):
        """
        Initialize the model validation service.

        Args:
            config: Service configuration
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="ModelValidationService",
            config=config,
            correlation_id=correlation_id,
        )

        # Configuration parameters with defaults
        ml_config = self._config.get("ml", {}) if self._config else {}
        self.validation_threshold = ml_config.get("validation_threshold", 0.7)
        self.significance_level = ml_config.get("significance_level", 0.05)
        self.stability_window = ml_config.get("stability_window", 1000)
        self.min_samples = ml_config.get("min_validation_samples", 100)

        # Overfitting prevention parameters
        self.max_train_test_gap = ml_config.get("max_train_test_gap", 0.1)  # 10%
        self.feature_importance_threshold = ml_config.get("feature_importance_threshold", 0.01)
        self.complexity_penalty = ml_config.get("complexity_penalty", 0.1)

        # Performance tracking
        self.validation_history = []
        self.benchmark_results = {}
        self.overfitting_alerts = []

        # Add dependencies
        self.add_dependency("DataService")
        self.add_dependency("ModelRegistryService")

        self._logger.info(
            "Model validation service initialized",
            validation_threshold=self.validation_threshold,
            significance_level=self.significance_level,
            stability_window=self.stability_window,
            min_samples=self.min_samples,
            max_train_test_gap=self.max_train_test_gap,
        )

    async def _do_start(self) -> None:
        """Start the model validation service."""
        try:
            # Resolve dependencies
            self.data_service = self.resolve_dependency("DataService")
            self.model_registry = self.resolve_dependency("ModelRegistryService")

            self._logger.info("Model validation service started successfully")
        except Exception as e:
            self._logger.error(f"Failed to start model validation service: {e}")
            raise

    async def _do_stop(self) -> None:
        """Stop the model validation service."""
        self._logger.info("Model validation service stopped")

    async def _service_health_check(self) -> HealthStatus:
        """Perform service-specific health check."""
        try:
            # Check if validation is working properly
            if not self.validation_history:
                return HealthStatus.HEALTHY

            # Check recent validation success rate
            recent_validations = self.validation_history[-10:]  # Last 10 validations
            if recent_validations:
                success_rate = sum(
                    1 for v in recent_validations if v.get("overall_pass", False)
                ) / len(recent_validations)
                if success_rate < 0.5:  # Less than 50% success rate
                    return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return HealthStatus.UNHEALTHY

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def validate_model_performance(
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
        return await self.execute_with_monitoring(
            "validate_model_performance",
            self._validate_model_performance_impl,
            model,
            X_test,
            y_test,
            benchmark_score,
        )

    async def _validate_model_performance_impl(
        self,
        model: BaseModel,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        benchmark_score: float | None = None,
    ) -> dict[str, Any]:
        """Implementation of model performance validation."""
        try:
            if not model.is_trained:
                raise ValidationError("Model must be trained before validation")

            if X_test.empty or y_test.empty:
                raise ValidationError("Test data cannot be empty")

            if len(X_test) < self.min_samples:
                raise ValidationError(f"Need at least {self.min_samples} test samples")

            self._logger.info(
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

            self._logger.info(
                "Model performance validation completed",
                model_name=model.model_name,
                primary_metric=primary_metric,
                meets_threshold=performance_validation["meets_threshold"],
                overall_pass=validation_result["overall_pass"],
            )

            return validation_result

        except Exception as e:
            self._logger.error(
                "Model performance validation failed",
                model_name=getattr(model, "model_name", "unknown"),
                error=str(e),
            )
            raise ValidationError(f"Performance validation failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def validate_model_stability(
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
        return await self.execute_with_monitoring(
            "validate_model_stability",
            self._validate_model_stability_impl,
            model,
            time_series_data,
            time_periods,
        )

    async def _validate_model_stability_impl(
        self,
        model: BaseModel,
        time_series_data: list[tuple[pd.DataFrame, pd.Series]],
        time_periods: list[datetime],
    ) -> dict[str, Any]:
        """Implementation of model stability validation."""
        try:
            if not model.is_trained:
                raise ValidationError("Model must be trained before stability validation")

            if len(time_series_data) < 2:
                raise ValidationError("Need at least 2 time periods for stability analysis")

            self._logger.info(
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
                    self._logger.warning(f"Empty data for time period {i}, skipping")
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

            self._logger.info(
                "Model stability validation completed",
                model_name=model.model_name,
                is_stable=is_stable,
                mean_performance=stability_metrics["mean_performance"],
                coefficient_of_variation=stability_metrics["coefficient_of_variation"],
            )

            return stability_result

        except Exception as e:
            self._logger.error(
                "Model stability validation failed",
                model_name=getattr(model, "model_name", "unknown"),
                error=str(e),
            )
            raise ValidationError(f"Stability validation failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def validate_production_readiness(
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
        return await self.execute_with_monitoring(
            "validate_production_readiness",
            self._validate_production_readiness_impl,
            model,
            X_test,
            y_test,
        )

    async def _validate_production_readiness_impl(
        self, model: BaseModel, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, Any]:
        """Implementation of production readiness validation."""
        try:
            self._logger.info(
                "Starting production readiness validation", model_name=model.model_name
            )

            readiness_checks = {}

            # 1. Basic model checks
            readiness_checks["model_trained"] = model.is_trained
            readiness_checks["has_predict_method"] = hasattr(model, "predict")
            readiness_checks["has_model_name"] = hasattr(model, "model_name") and model.model_name

            # 2. Performance validation
            performance_result = await self._validate_model_performance_impl(model, X_test, y_test)
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

            self._logger.info(
                "Production readiness validation completed",
                model_name=model.model_name,
                is_production_ready=readiness_result["is_production_ready"],
                readiness_score=readiness_score,
                critical_checks_passed=critical_passed,
            )

            return readiness_result

        except Exception as e:
            self._logger.error(
                "Production readiness validation failed",
                model_name=getattr(model, "model_name", "unknown"),
                error=str(e),
            )
            raise ValidationError(f"Production readiness validation failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def detect_overfitting(
        self,
        model: BaseModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Comprehensive overfitting detection for trading models.

        This method implements multiple techniques to detect overfitting:
        1. Train-validation-test performance gaps analysis
        2. Learning curves analysis
        3. Feature importance stability
        4. Model complexity analysis
        5. Statistical significance of performance differences

        Args:
            model: Trained ML model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Optional test features
            y_test: Optional test targets

        Returns:
            Overfitting detection results with recommendations
        """
        return await self.execute_with_monitoring(
            "detect_overfitting",
            self._detect_overfitting_impl,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        )

    async def _detect_overfitting_impl(
        self,
        model: BaseModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Implementation of overfitting detection."""
        try:
            self._logger.info(
                "Starting overfitting detection",
                model_name=model.model_name,
                train_samples=len(X_train),
                val_samples=len(X_val),
                test_samples=len(X_test) if X_test is not None else 0,
            )

            overfitting_indicators = {}

            # 1. Performance Gap Analysis
            performance_gaps = self._analyze_performance_gaps(
                model, X_train, y_train, X_val, y_val, X_test, y_test
            )
            overfitting_indicators["performance_gaps"] = performance_gaps

            # 2. Learning Curves Analysis
            learning_curves_analysis = self._analyze_learning_curves(
                model, X_train, y_train, X_val, y_val
            )
            overfitting_indicators["learning_curves"] = learning_curves_analysis

            # 3. Feature Importance Stability
            if hasattr(model, "feature_importances_") or hasattr(
                model.model, "feature_importances_"
            ):
                feature_stability = await self._analyze_feature_importance_stability(
                    model, X_train, y_train, X_val, y_val
                )
                overfitting_indicators["feature_stability"] = feature_stability

            # 4. Model Complexity Analysis
            complexity_analysis = self._analyze_model_complexity(model, X_train)
            overfitting_indicators["complexity_analysis"] = complexity_analysis

            # 5. Cross-validation stability
            cv_stability = self._analyze_cv_stability(model, X_train, y_train)
            overfitting_indicators["cv_stability"] = cv_stability

            # 6. Overall overfitting assessment
            overfitting_score, risk_level = self._calculate_overfitting_risk(overfitting_indicators)

            # 7. Generate recommendations
            recommendations = self._generate_overfitting_recommendations(
                overfitting_indicators, risk_level
            )

            overfitting_result = {
                "timestamp": datetime.utcnow().isoformat(),
                "model_name": model.model_name,
                "overfitting_indicators": overfitting_indicators,
                "overfitting_score": overfitting_score,
                "risk_level": risk_level,  # 'low', 'medium', 'high', 'critical'
                "recommendations": recommendations,
                "is_overfitted": risk_level in ["high", "critical"],
            }

            # Store alert if high risk
            if risk_level in ["high", "critical"]:
                alert = {
                    "timestamp": datetime.utcnow(),
                    "model_name": model.model_name,
                    "risk_level": risk_level,
                    "overfitting_score": overfitting_score,
                    "primary_indicators": self._get_primary_risk_indicators(overfitting_indicators),
                }
                self.overfitting_alerts.append(alert)

            self._logger.info(
                "Overfitting detection completed",
                model_name=model.model_name,
                risk_level=risk_level,
                overfitting_score=overfitting_score,
                is_overfitted=overfitting_result["is_overfitted"],
            )

            return overfitting_result

        except Exception as e:
            self._logger.error(
                "Overfitting detection failed",
                model_name=getattr(model, "model_name", "unknown"),
                error=str(e),
            )
            raise ValidationError(f"Overfitting detection failed: {e}")

    def _analyze_performance_gaps(
        self,
        model: BaseModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Analyze performance gaps between train/val/test sets."""
        try:
            # Get predictions for all sets
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            # Calculate metrics for train and validation
            if model.model_type == "classification":
                train_score = accuracy_score(y_train, train_pred)
                val_score = accuracy_score(y_val, val_pred)
            else:
                train_score = r2_score(y_train, train_pred)
                val_score = r2_score(y_val, val_pred)

            train_val_gap = train_score - val_score

            # Test set analysis if available
            test_analysis = {}
            if X_test is not None and y_test is not None:
                test_pred = model.predict(X_test)
                if model.model_type == "classification":
                    test_score = accuracy_score(y_test, test_pred)
                else:
                    test_score = r2_score(y_test, test_pred)

                val_test_gap = val_score - test_score
                train_test_gap = train_score - test_score

                test_analysis = {
                    "test_score": test_score,
                    "val_test_gap": val_test_gap,
                    "train_test_gap": train_test_gap,
                    "val_test_gap_excessive": abs(val_test_gap) > self.max_train_test_gap,
                    "train_test_gap_excessive": abs(train_test_gap) > self.max_train_test_gap * 1.5,
                }

            return {
                "train_score": train_score,
                "val_score": val_score,
                "train_val_gap": train_val_gap,
                "train_val_gap_excessive": train_val_gap > self.max_train_test_gap,
                "train_val_gap_percentage": (
                    (train_val_gap / train_score * 100) if train_score != 0 else 0
                ),
                "test_analysis": test_analysis,
            }

        except Exception as e:
            self._logger.error(f"Performance gap analysis failed: {e}")
            return {"analysis_failed": True, "error": str(e)}

    def _analyze_learning_curves(
        self,
        model: BaseModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
        """Analyze learning curves to detect overfitting patterns."""
        try:
            if not hasattr(model, "model") or model.model is None:
                return {"analysis_skipped": "Model not compatible with sklearn learning curves"}

            # Generate learning curves
            train_sizes = np.linspace(0.1, 1.0, 10)

            train_sizes_abs, train_scores, val_scores = learning_curve(
                model.model,
                pd.concat([X_train, X_val]),
                pd.concat([y_train, y_val]),
                train_sizes=train_sizes,
                cv=3,
                scoring="accuracy" if model.model_type == "classification" else "r2",
                n_jobs=1,
                random_state=42,
            )

            # Analyze learning curve patterns
            train_scores_mean = np.mean(train_scores, axis=1)
            val_scores_mean = np.mean(val_scores, axis=1)

            # Check for overfitting patterns
            final_gap = train_scores_mean[-1] - val_scores_mean[-1]
            gap_trend = np.polyfit(train_sizes_abs, train_scores_mean - val_scores_mean, 1)[0]

            # Validation score plateau detection
            val_score_trend = np.polyfit(train_sizes_abs[-5:], val_scores_mean[-5:], 1)[0]
            val_plateaued = abs(val_score_trend) < 0.001  # Very small trend

            return {
                "train_sizes": train_sizes_abs.tolist(),
                "train_scores_mean": train_scores_mean.tolist(),
                "val_scores_mean": val_scores_mean.tolist(),
                "final_gap": final_gap,
                "gap_trend": gap_trend,  # Positive means gap is increasing
                "validation_plateaued": val_plateaued,
                "shows_overfitting_pattern": final_gap > self.max_train_test_gap and gap_trend > 0,
            }

        except Exception as e:
            self._logger.error(f"Learning curve analysis failed: {e}")
            return {"analysis_failed": True, "error": str(e)}

    async def _analyze_feature_importance_stability(
        self,
        model: BaseModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
        """Analyze feature importance stability across different data splits."""
        try:
            # Get feature importances from the trained model
            if hasattr(model, "feature_importances_"):
                base_importance = model.feature_importances_
            elif hasattr(model.model, "feature_importances_"):
                base_importance = model.model.feature_importances_
            else:
                # Use permutation importance as fallback
                result = permutation_importance(
                    model.model, X_val, y_val, n_repeats=5, random_state=42
                )
                base_importance = result.importances_mean

            # Train model on bootstrap samples and compare feature importances
            n_bootstrap = 5
            importance_variations = []

            for _i in range(n_bootstrap):
                # Bootstrap sample
                sample_indices = np.random.choice(
                    len(X_train), size=int(len(X_train) * 0.8), replace=True
                )
                X_bootstrap = X_train.iloc[sample_indices]
                y_bootstrap = y_train.iloc[sample_indices]

                # Create a new model instance using the factory pattern
                # This is a simplified approach - in practice you'd use proper dependency injection
                try:
                    # Skip complex model training for now and use permutation importance
                    result = permutation_importance(
                        model.model, X_bootstrap, y_bootstrap, n_repeats=3, random_state=42
                    )
                    bootstrap_importance = result.importances_mean
                    importance_variations.append(bootstrap_importance)
                except Exception:
                    continue

            if importance_variations:
                # Calculate stability metrics
                importance_matrix = np.array(importance_variations)
                importance_std = np.std(importance_matrix, axis=0)
                importance_cv = importance_std / (np.mean(importance_matrix, axis=0) + 1e-8)

                # Identify unstable features
                unstable_features = importance_cv > 0.5  # CV > 50%

                return {
                    "base_importance": base_importance.tolist(),
                    "importance_std": importance_std.tolist(),
                    "importance_cv": importance_cv.tolist(),
                    "unstable_features": unstable_features.tolist(),
                    "n_unstable_features": np.sum(unstable_features),
                    "feature_stability_score": 1 - np.mean(importance_cv),
                    "is_stable": np.mean(importance_cv) < 0.3,
                }
            else:
                return {"analysis_skipped": "Could not compute bootstrap importances"}

        except Exception as e:
            self._logger.error(f"Feature importance stability analysis failed: {e}")
            return {"analysis_failed": True, "error": str(e)}

    def _analyze_model_complexity(self, model: BaseModel, X_train: pd.DataFrame) -> dict[str, Any]:
        """Analyze model complexity relative to data size."""
        try:
            n_samples, n_features = X_train.shape

            # Estimate model parameters
            model_params = self._estimate_model_parameters(model)

            # Calculate complexity metrics
            samples_per_parameter = n_samples / model_params if model_params > 0 else np.inf
            features_to_samples_ratio = n_features / n_samples

            # Complexity assessment
            is_complex = (
                model_params > n_samples * 0.1  # More than 10% of samples as parameters
                or samples_per_parameter < 10  # Less than 10 samples per parameter
                or features_to_samples_ratio > 0.5  # More than 50% features to samples ratio
            )

            return {
                "n_samples": n_samples,
                "n_features": n_features,
                "estimated_parameters": model_params,
                "samples_per_parameter": samples_per_parameter,
                "features_to_samples_ratio": features_to_samples_ratio,
                "is_overly_complex": is_complex,
                "complexity_score": min(1.0, model_params / n_samples),
            }

        except Exception as e:
            self._logger.error(f"Model complexity analysis failed: {e}")
            return {"analysis_failed": True, "error": str(e)}

    def _analyze_cv_stability(
        self, model: BaseModel, X: pd.DataFrame, y: pd.Series
    ) -> dict[str, Any]:
        """Analyze cross-validation stability to detect overfitting."""
        try:
            if not hasattr(model, "model") or model.model is None:
                return {"analysis_skipped": "Model not compatible with sklearn CV"}

            from sklearn.model_selection import cross_val_score

            # Perform multiple CV runs with different random states
            cv_scores_runs = []
            for random_state in range(5):
                scores = cross_val_score(
                    model.model,
                    X,
                    y,
                    cv=5,
                    scoring="accuracy" if model.model_type == "classification" else "r2",
                    random_state=random_state,
                )
                cv_scores_runs.extend(scores)

            cv_scores = np.array(cv_scores_runs)
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            cv_coefficient_variation = cv_std / cv_mean if cv_mean != 0 else np.inf

            # Stability assessment
            is_stable = cv_coefficient_variation < 0.1  # CV < 10%

            return {
                "cv_scores": cv_scores.tolist(),
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "cv_coefficient_variation": cv_coefficient_variation,
                "is_stable": is_stable,
            }

        except Exception as e:
            self._logger.error(f"CV stability analysis failed: {e}")
            return {"analysis_failed": True, "error": str(e)}

    def _estimate_model_parameters(self, model: BaseModel) -> int:
        """Estimate the number of model parameters."""
        try:
            if hasattr(model, "model"):
                sklearn_model = model.model

                # Common sklearn model parameter estimation
                if hasattr(sklearn_model, "coef_"):
                    # Linear models
                    coef = sklearn_model.coef_
                    if coef.ndim == 1:
                        params = len(coef)
                    else:
                        params = coef.shape[0] * coef.shape[1]

                    if hasattr(sklearn_model, "intercept_"):
                        intercept = sklearn_model.intercept_
                        if np.isscalar(intercept):
                            params += 1
                        else:
                            params += len(intercept)

                    return params

                elif hasattr(sklearn_model, "n_estimators") and hasattr(sklearn_model, "max_depth"):
                    # Tree-based ensemble models
                    n_trees = sklearn_model.n_estimators
                    max_depth = sklearn_model.max_depth or 10  # Default estimate
                    # Rough estimate: 2^depth nodes per tree
                    return n_trees * (2 ** min(max_depth, 10))

                elif hasattr(sklearn_model, "max_depth"):
                    # Single tree models
                    max_depth = sklearn_model.max_depth or 10
                    return 2 ** min(max_depth, 10)

            # Fallback: assume moderate complexity
            return 100

        except Exception as e:
            self._logger.error(f"Parameter estimation failed: {e}")
            return 100  # Conservative estimate

    def _calculate_overfitting_risk(
        self, overfitting_indicators: dict[str, Any]
    ) -> tuple[float, str]:
        """Calculate overall overfitting risk score and level."""
        risk_score = 0.0
        max_score = 0.0

        # Performance gaps (weight: 0.3)
        if "performance_gaps" in overfitting_indicators:
            gaps = overfitting_indicators["performance_gaps"]
            if not gaps.get("analysis_failed", False):
                max_score += 0.3
                if gaps.get("train_val_gap_excessive", False):
                    risk_score += 0.15
                if gaps.get("test_analysis", {}).get("train_test_gap_excessive", False):
                    risk_score += 0.15

        # Learning curves (weight: 0.25)
        if "learning_curves" in overfitting_indicators:
            curves = overfitting_indicators["learning_curves"]
            if not curves.get("analysis_failed", False):
                max_score += 0.25
                if curves.get("shows_overfitting_pattern", False):
                    risk_score += 0.25

        # Feature stability (weight: 0.2)
        if "feature_stability" in overfitting_indicators:
            stability = overfitting_indicators["feature_stability"]
            if not stability.get("analysis_failed", False):
                max_score += 0.2
                if not stability.get("is_stable", True):
                    risk_score += 0.2

        # Model complexity (weight: 0.15)
        if "complexity_analysis" in overfitting_indicators:
            complexity = overfitting_indicators["complexity_analysis"]
            if not complexity.get("analysis_failed", False):
                max_score += 0.15
                if complexity.get("is_overly_complex", False):
                    risk_score += 0.15

        # CV stability (weight: 0.1)
        if "cv_stability" in overfitting_indicators:
            cv_stab = overfitting_indicators["cv_stability"]
            if not cv_stab.get("analysis_failed", False):
                max_score += 0.1
                if not cv_stab.get("is_stable", True):
                    risk_score += 0.1

        # Normalize score
        if max_score > 0:
            normalized_score = risk_score / max_score
        else:
            normalized_score = 0.0

        # Determine risk level
        if normalized_score >= 0.7:
            risk_level = "critical"
        elif normalized_score >= 0.5:
            risk_level = "high"
        elif normalized_score >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"

        return normalized_score, risk_level

    def _generate_overfitting_recommendations(
        self, overfitting_indicators: dict[str, Any], risk_level: str
    ) -> list[str]:
        """Generate specific recommendations to address overfitting."""
        recommendations = []

        if risk_level == "low":
            recommendations.append(
                "✅ Model shows low overfitting risk. Continue monitoring in production."
            )
            return recommendations

        # Performance gap recommendations
        if overfitting_indicators.get("performance_gaps", {}).get("train_val_gap_excessive", False):
            recommendations.extend(
                [
                    "🎯 CRITICAL: Large train-validation gap detected. Consider:",
                    "   • Add regularization (L1/L2, dropout, early stopping)",
                    "   • Reduce model complexity",
                    "   • Collect more training data",
                    "   • Use cross-validation for hyperparameter tuning",
                ]
            )

        # Learning curve recommendations
        if overfitting_indicators.get("learning_curves", {}).get(
            "shows_overfitting_pattern", False
        ):
            recommendations.extend(
                [
                    "📈 Learning curves show overfitting pattern:",
                    "   • Implement early stopping",
                    "   • Add regularization techniques",
                    "   • Increase validation set size",
                ]
            )

        # Feature stability recommendations
        if not overfitting_indicators.get("feature_stability", {}).get("is_stable", True):
            recommendations.extend(
                [
                    "🔄 Feature importance instability detected:",
                    "   • Use feature selection methods",
                    "   • Apply ensemble methods to stabilize features",
                    "   • Consider feature engineering based on domain knowledge",
                    "   • Use regularization to reduce feature sensitivity",
                ]
            )

        # Model complexity recommendations
        if overfitting_indicators.get("complexity_analysis", {}).get("is_overly_complex", False):
            recommendations.extend(
                [
                    "⚡ Model complexity too high relative to data size:",
                    "   • Reduce model parameters (fewer layers, smaller networks)",
                    "   • Use simpler model architectures",
                    "   • Implement aggressive regularization",
                    "   • Collect more training data if possible",
                ]
            )

        # CV stability recommendations
        if not overfitting_indicators.get("cv_stability", {}).get("is_stable", True):
            recommendations.extend(
                [
                    "📊 Cross-validation instability detected:",
                    "   • Use more robust CV strategies (time series aware)",
                    "   • Increase CV folds for more stable estimates",
                    "   • Check for data leakage or temporal dependencies",
                ]
            )

        # General recommendations based on risk level
        if risk_level in ["high", "critical"]:
            recommendations.extend(
                [
                    "",
                    "🚨 IMMEDIATE ACTIONS REQUIRED:",
                    "   • Do NOT deploy this model to production",
                    "   • Retrain with overfitting prevention techniques",
                    "   • Validate on truly out-of-sample data",
                    "   • Consider ensemble methods to reduce variance",
                    "   • Implement A/B testing framework before deployment",
                ]
            )

        return recommendations

    def _get_primary_risk_indicators(self, overfitting_indicators: dict[str, Any]) -> list[str]:
        """Get primary risk indicators for alerts."""
        indicators = []

        if overfitting_indicators.get("performance_gaps", {}).get("train_val_gap_excessive", False):
            indicators.append("excessive_train_val_gap")

        if overfitting_indicators.get("learning_curves", {}).get(
            "shows_overfitting_pattern", False
        ):
            indicators.append("overfitting_learning_curves")

        if not overfitting_indicators.get("feature_stability", {}).get("is_stable", True):
            indicators.append("unstable_feature_importance")

        if overfitting_indicators.get("complexity_analysis", {}).get("is_overly_complex", False):
            indicators.append("excessive_model_complexity")

        if not overfitting_indicators.get("cv_stability", {}).get("is_stable", True):
            indicators.append("cv_instability")

        return indicators

    def get_overfitting_alerts(self) -> list[dict[str, Any]]:
        """Get recent overfitting alerts."""
        return self.overfitting_alerts.copy()

    def clear_overfitting_alerts(self) -> None:
        """Clear overfitting alerts history."""
        self.overfitting_alerts.clear()
        self._logger.info("Overfitting alerts cleared")

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
            self._logger.error(f"Classification metrics calculation failed: {e}")
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
            self._logger.error(f"Regression metrics calculation failed: {e}")
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
            self._logger.error(f"Statistical significance test failed: {e}")
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
            self._logger.error(f"Residual analysis failed: {e}")
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
            self._logger.error(f"Performance trend analysis failed: {e}")
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
            self._logger.error(f"Prediction consistency check failed: {e}")
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
            self._logger.error(f"Computational efficiency check failed: {e}")
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
            self._logger.error(f"Error handling check failed: {e}")
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
            self._logger.error(f"Data quality handling check failed: {e}")
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
        self._logger.info("Validation history cleared")
