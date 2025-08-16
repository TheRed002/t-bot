"""
Drift Detection for ML Models.

This module provides drift detection capabilities for monitoring changes
in data distribution and model performance over time.
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.database.connection import get_sync_session
from src.database.models import MLDriftDetection
from src.utils.decorators import log_calls, time_execution

logger = get_logger(__name__)


class DriftDetector:
    """
    Drift detection system for monitoring data and model drift.

    This class provides various drift detection methods:
    - Statistical drift detection (KS test, Chi-square test)
    - Distribution shift detection
    - Performance drift detection
    - Feature drift detection
    - Prediction drift detection

    Attributes:
        config: Application configuration
        drift_threshold: Threshold for drift detection
        min_samples: Minimum samples for drift detection
        reference_window: Size of reference window
        detection_window: Size of detection window
        drift_history: History of drift detections
    """

    def __init__(self, config: Config):
        """
        Initialize the drift detector.

        Args:
            config: Application configuration
        """
        self.config = config

        # Drift detection parameters
        self.drift_threshold = config.ml.drift_threshold
        self.min_samples = config.ml.min_drift_samples
        self.reference_window = config.ml.drift_reference_window
        self.detection_window = config.ml.drift_detection_window
        self.significance_level = config.ml.significance_level

        # Drift monitoring state
        self.reference_data = {}
        self.drift_history = []
        self.feature_statistics = {}

        logger.info(
            "Drift detector initialized",
            drift_threshold=self.drift_threshold,
            min_samples=self.min_samples,
            reference_window=self.reference_window,
            detection_window=self.detection_window,
        )

    @time_execution
    @log_calls
    def detect_feature_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Detect feature drift between reference and current data.

        Args:
            reference_data: Reference dataset
            current_data: Current dataset to compare
            feature_columns: Specific features to check (None for all)

        Returns:
            Drift detection results

        Raises:
            ValidationError: If drift detection fails
        """
        try:
            if reference_data.empty or current_data.empty:
                raise ValidationError("Reference and current data cannot be empty")

            if len(reference_data) < self.min_samples or len(current_data) < self.min_samples:
                raise ValidationError(
                    f"Need at least {self.min_samples} samples for drift detection"
                )

            # Determine features to check
            if feature_columns is None:
                feature_columns = list(set(reference_data.columns) & set(current_data.columns))

            if not feature_columns:
                raise ValidationError("No common features found between datasets")

            logger.info(
                "Starting feature drift detection",
                reference_samples=len(reference_data),
                current_samples=len(current_data),
                features_to_check=len(feature_columns),
            )

            drift_results = {}
            overall_drift_detected = False

            for feature in feature_columns:
                try:
                    feature_drift = self._detect_single_feature_drift(
                        reference_data[feature].dropna(), current_data[feature].dropna(), feature
                    )
                    drift_results[feature] = feature_drift

                    if feature_drift["drift_detected"]:
                        overall_drift_detected = True

                except Exception as e:
                    logger.warning(f"Failed to detect drift for feature {feature}: {e}")
                    drift_results[feature] = {"drift_detected": False, "error": str(e)}

            # Calculate overall drift statistics
            drift_scores = [
                r.get("drift_score", 0) for r in drift_results.values() if "drift_score" in r
            ]

            overall_result = {
                "timestamp": datetime.utcnow(),
                "drift_type": "feature_drift",
                "overall_drift_detected": overall_drift_detected,
                "features_checked": len(feature_columns),
                "features_with_drift": sum(
                    1 for r in drift_results.values() if r.get("drift_detected", False)
                ),
                "average_drift_score": np.mean(drift_scores) if drift_scores else 0,
                "max_drift_score": np.max(drift_scores) if drift_scores else 0,
                "feature_results": drift_results,
                "reference_samples": len(reference_data),
                "current_samples": len(current_data),
            }

            # Store drift detection result
            self._store_drift_result(overall_result)

            logger.info(
                "Feature drift detection completed",
                overall_drift_detected=overall_drift_detected,
                features_with_drift=overall_result["features_with_drift"],
                average_drift_score=overall_result["average_drift_score"],
            )

            return overall_result

        except Exception as e:
            logger.error(f"Feature drift detection failed: {e}")
            raise ValidationError(f"Feature drift detection failed: {e}") from e

    @time_execution
    @log_calls
    def detect_prediction_drift(
        self, reference_predictions: np.ndarray, current_predictions: np.ndarray, model_name: str
    ) -> dict[str, Any]:
        """
        Detect drift in model predictions.

        Args:
            reference_predictions: Reference predictions
            current_predictions: Current predictions to compare
            model_name: Name of the model

        Returns:
            Prediction drift detection results

        Raises:
            ValidationError: If prediction drift detection fails
        """
        try:
            if (
                len(reference_predictions) < self.min_samples
                or len(current_predictions) < self.min_samples
            ):
                raise ValidationError(
                    f"Need at least {self.min_samples} predictions for drift detection"
                )

            logger.info(
                "Starting prediction drift detection",
                model_name=model_name,
                reference_predictions=len(reference_predictions),
                current_predictions=len(current_predictions),
            )

            # Statistical tests for prediction drift
            drift_tests = {}

            # Kolmogorov-Smirnov test
            ks_statistic, ks_p_value = stats.ks_2samp(reference_predictions, current_predictions)
            drift_tests["ks_test"] = {
                "statistic": ks_statistic,
                "p_value": ks_p_value,
                "drift_detected": ks_p_value < self.significance_level,
            }

            # Mann-Whitney U test (non-parametric)
            mw_statistic, mw_p_value = stats.mannwhitneyu(
                reference_predictions, current_predictions, alternative="two-sided"
            )
            drift_tests["mann_whitney_test"] = {
                "statistic": mw_statistic,
                "p_value": mw_p_value,
                "drift_detected": mw_p_value < self.significance_level,
            }

            # Distribution statistics comparison
            ref_stats = self._calculate_distribution_stats(reference_predictions)
            curr_stats = self._calculate_distribution_stats(current_predictions)

            stats_comparison = {
                "mean_shift": abs(ref_stats["mean"] - curr_stats["mean"]),
                "std_ratio": curr_stats["std"] / ref_stats["std"] if ref_stats["std"] > 0 else 1,
                "skewness_shift": abs(ref_stats["skewness"] - curr_stats["skewness"]),
                "kurtosis_shift": abs(ref_stats["kurtosis"] - curr_stats["kurtosis"]),
            }

            # Overall drift detection
            drift_detected = any(test["drift_detected"] for test in drift_tests.values())

            # Calculate drift score
            drift_score = max(ks_statistic, 1 - min(drift_tests["mann_whitney_test"]["p_value"], 1))

            prediction_drift_result = {
                "timestamp": datetime.utcnow(),
                "drift_type": "prediction_drift",
                "model_name": model_name,
                "drift_detected": drift_detected,
                "drift_score": drift_score,
                "drift_tests": drift_tests,
                "reference_stats": ref_stats,
                "current_stats": curr_stats,
                "stats_comparison": stats_comparison,
                "reference_samples": len(reference_predictions),
                "current_samples": len(current_predictions),
            }

            # Store drift detection result
            self._store_drift_result(prediction_drift_result)

            logger.info(
                "Prediction drift detection completed",
                model_name=model_name,
                drift_detected=drift_detected,
                drift_score=drift_score,
            )

            return prediction_drift_result

        except Exception as e:
            logger.error(f"Prediction drift detection failed: {e}")
            raise ValidationError(f"Prediction drift detection failed: {e}") from e

    @time_execution
    @log_calls
    def detect_performance_drift(
        self,
        reference_performance: dict[str, float],
        current_performance: dict[str, float],
        model_name: str,
    ) -> dict[str, Any]:
        """
        Detect drift in model performance metrics.

        Args:
            reference_performance: Reference performance metrics
            current_performance: Current performance metrics
            model_name: Name of the model

        Returns:
            Performance drift detection results

        Raises:
            ValidationError: If performance drift detection fails
        """
        try:
            if not reference_performance or not current_performance:
                raise ValidationError("Performance metrics cannot be empty")

            # Find common metrics
            common_metrics = set(reference_performance.keys()) & set(current_performance.keys())

            if not common_metrics:
                raise ValidationError("No common performance metrics found")

            logger.info(
                "Starting performance drift detection",
                model_name=model_name,
                common_metrics=len(common_metrics),
            )

            performance_changes = {}
            drift_detected = False

            for metric in common_metrics:
                ref_value = reference_performance[metric]
                curr_value = current_performance[metric]

                # Calculate relative change
                if ref_value != 0:
                    relative_change = (curr_value - ref_value) / ref_value
                else:
                    relative_change = 0 if curr_value == 0 else float("inf")

                # Check if change exceeds threshold
                metric_drift = abs(relative_change) > self.drift_threshold

                performance_changes[metric] = {
                    "reference_value": ref_value,
                    "current_value": curr_value,
                    "absolute_change": curr_value - ref_value,
                    "relative_change": relative_change,
                    "drift_detected": metric_drift,
                }

                if metric_drift:
                    drift_detected = True

            # Calculate overall performance drift score
            relative_changes = [
                abs(pc["relative_change"])
                for pc in performance_changes.values()
                if not np.isinf(pc["relative_change"])
            ]
            drift_score = np.mean(relative_changes) if relative_changes else 0

            performance_drift_result = {
                "timestamp": datetime.utcnow(),
                "drift_type": "performance_drift",
                "model_name": model_name,
                "drift_detected": drift_detected,
                "drift_score": drift_score,
                "metrics_checked": len(common_metrics),
                "metrics_with_drift": sum(
                    1 for pc in performance_changes.values() if pc["drift_detected"]
                ),
                "performance_changes": performance_changes,
            }

            # Store drift detection result
            self._store_drift_result(performance_drift_result)

            logger.info(
                "Performance drift detection completed",
                model_name=model_name,
                drift_detected=drift_detected,
                drift_score=drift_score,
                metrics_with_drift=performance_drift_result["metrics_with_drift"],
            )

            return performance_drift_result

        except Exception as e:
            logger.error(f"Performance drift detection failed: {e}")
            raise ValidationError(f"Performance drift detection failed: {e}") from e

    def _detect_single_feature_drift(
        self, reference_feature: pd.Series, current_feature: pd.Series, feature_name: str
    ) -> dict[str, Any]:
        """Detect drift for a single feature."""
        try:
            # Determine feature type
            is_categorical = (
                reference_feature.dtype == "object" or reference_feature.dtype.name == "category"
            )

            if is_categorical:
                return self._detect_categorical_drift(
                    reference_feature, current_feature, feature_name
                )
            else:
                return self._detect_numerical_drift(
                    reference_feature, current_feature, feature_name
                )

        except Exception as e:
            logger.error(f"Single feature drift detection failed for {feature_name}: {e}")
            return {"drift_detected": False, "error": str(e)}

    def _detect_numerical_drift(
        self, reference_data: pd.Series, current_data: pd.Series, feature_name: str
    ) -> dict[str, Any]:
        """Detect drift for numerical features."""
        try:
            # Statistical tests
            ks_statistic, ks_p_value = stats.ks_2samp(reference_data, current_data)

            # Mann-Whitney U test
            mw_statistic, mw_p_value = stats.mannwhitneyu(
                reference_data, current_data, alternative="two-sided"
            )

            # Distribution statistics
            ref_stats = self._calculate_distribution_stats(reference_data)
            curr_stats = self._calculate_distribution_stats(current_data)

            # Jensen-Shannon distance
            js_distance = self._calculate_js_distance(reference_data, current_data)

            # Overall drift detection
            drift_detected = (
                ks_p_value < self.significance_level
                or mw_p_value < self.significance_level
                or js_distance > self.drift_threshold
            )

            # Drift score (combination of different metrics)
            drift_score = max(ks_statistic, js_distance, 1 - min(mw_p_value, 1))

            return {
                "drift_detected": drift_detected,
                "drift_score": drift_score,
                "ks_test": {"statistic": ks_statistic, "p_value": ks_p_value},
                "mann_whitney_test": {"statistic": mw_statistic, "p_value": mw_p_value},
                "js_distance": js_distance,
                "reference_stats": ref_stats,
                "current_stats": curr_stats,
                "feature_type": "numerical",
            }

        except Exception as e:
            logger.error(f"Numerical drift detection failed: {e}")
            return {"drift_detected": False, "error": str(e)}

    def _detect_categorical_drift(
        self, reference_data: pd.Series, current_data: pd.Series, feature_name: str
    ) -> dict[str, Any]:
        """Detect drift for categorical features."""
        try:
            # Get value counts
            ref_counts = reference_data.value_counts(normalize=True)
            curr_counts = current_data.value_counts(normalize=True)

            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            ref_aligned = ref_counts.reindex(all_categories, fill_value=0)
            curr_aligned = curr_counts.reindex(all_categories, fill_value=0)

            # Chi-square test
            # Add small constant to avoid zero counts
            ref_expected = (ref_aligned + 1e-10) * len(reference_data)
            curr_observed = curr_aligned * len(current_data)

            chi2_statistic, chi2_p_value = stats.chisquare(curr_observed, ref_expected)

            # Jensen-Shannon divergence for categorical data
            js_divergence = self._calculate_js_divergence_categorical(ref_aligned, curr_aligned)

            # Overall drift detection
            drift_detected = (
                chi2_p_value < self.significance_level or js_divergence > self.drift_threshold
            )

            # Drift score
            drift_score = max(js_divergence, 1 - min(chi2_p_value, 1))

            return {
                "drift_detected": drift_detected,
                "drift_score": drift_score,
                "chi2_test": {"statistic": chi2_statistic, "p_value": chi2_p_value},
                "js_divergence": js_divergence,
                "reference_distribution": ref_aligned.to_dict(),
                "current_distribution": curr_aligned.to_dict(),
                "feature_type": "categorical",
            }

        except Exception as e:
            logger.error(f"Categorical drift detection failed: {e}")
            return {"drift_detected": False, "error": str(e)}

    def _calculate_distribution_stats(self, data: pd.Series) -> dict[str, float]:
        """Calculate distribution statistics."""
        try:
            return {
                "mean": float(data.mean()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "median": float(data.median()),
                "skewness": float(stats.skew(data)),
                "kurtosis": float(stats.kurtosis(data)),
                "q25": float(data.quantile(0.25)),
                "q75": float(data.quantile(0.75)),
            }
        except Exception as e:
            logger.error(f"Distribution statistics calculation failed: {e}")
            return {}

    def _calculate_js_distance(self, ref_data: pd.Series, curr_data: pd.Series) -> float:
        """Calculate Jensen-Shannon distance for numerical data."""
        try:
            # Create histograms
            min_val = min(ref_data.min(), curr_data.min())
            max_val = max(ref_data.max(), curr_data.max())

            bins = np.linspace(min_val, max_val, 50)

            ref_hist, _ = np.histogram(ref_data, bins=bins, density=True)
            curr_hist, _ = np.histogram(curr_data, bins=bins, density=True)

            # Normalize to probabilities
            ref_hist = ref_hist / ref_hist.sum()
            curr_hist = curr_hist / curr_hist.sum()

            # Add small constant to avoid log(0)
            ref_hist = ref_hist + 1e-10
            curr_hist = curr_hist + 1e-10

            # Calculate Jensen-Shannon distance
            m = 0.5 * (ref_hist + curr_hist)
            js_distance = 0.5 * stats.entropy(ref_hist, m) + 0.5 * stats.entropy(curr_hist, m)

            return np.sqrt(js_distance)

        except Exception as e:
            logger.error(f"Jensen-Shannon distance calculation failed: {e}")
            return 0.0

    def _calculate_js_divergence_categorical(
        self, ref_dist: pd.Series, curr_dist: pd.Series
    ) -> float:
        """Calculate Jensen-Shannon divergence for categorical data."""
        try:
            # Add small constant to avoid log(0)
            ref_probs = ref_dist + 1e-10
            curr_probs = curr_dist + 1e-10

            # Normalize
            ref_probs = ref_probs / ref_probs.sum()
            curr_probs = curr_probs / curr_probs.sum()

            # Calculate JS divergence
            m = 0.5 * (ref_probs + curr_probs)
            js_divergence = 0.5 * stats.entropy(ref_probs, m) + 0.5 * stats.entropy(curr_probs, m)

            return js_divergence

        except Exception as e:
            logger.error(f"Jensen-Shannon divergence calculation failed: {e}")
            return 0.0

    def _store_drift_result(self, drift_result: dict[str, Any]):
        """Store drift detection result in database."""
        try:
            with get_sync_session() as session:
                drift_record = MLDriftDetection(
                    model_name=drift_result.get("model_name", "unknown"),
                    drift_type=drift_result["drift_type"],
                    detection_timestamp=drift_result["timestamp"],
                    drift_detected=drift_result["drift_detected"],
                    drift_score=drift_result.get("drift_score", 0.0),
                    drift_details=drift_result,
                    reference_samples=drift_result.get("reference_samples", 0),
                    current_samples=drift_result.get("current_samples", 0),
                )

                session.add(drift_record)
                session.commit()

                logger.debug(
                    "Drift detection result stored",
                    drift_type=drift_result["drift_type"],
                    drift_detected=drift_result["drift_detected"],
                )

        except Exception as e:
            logger.error(f"Failed to store drift result: {e}")

    def get_drift_history(
        self,
        model_name: str | None = None,
        drift_type: str | None = None,
        days_back: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Get drift detection history.

        Args:
            model_name: Filter by model name
            drift_type: Filter by drift type
            days_back: Number of days to look back

        Returns:
            List of drift detection results
        """
        try:
            with get_sync_session() as session:
                query = session.query(MLDriftDetection)

                # Apply filters
                if model_name:
                    query = query.filter(MLDriftDetection.model_name == model_name)

                if drift_type:
                    query = query.filter(MLDriftDetection.drift_type == drift_type)

                # Time filter
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(MLDriftDetection.detection_timestamp >= cutoff_date)

                # Execute query
                results = query.order_by(MLDriftDetection.detection_timestamp.desc()).all()

                # Convert to dictionaries
                drift_history = []
                for result in results:
                    drift_history.append(
                        {
                            "id": result.id,
                            "model_name": result.model_name,
                            "drift_type": result.drift_type,
                            "detection_timestamp": result.detection_timestamp,
                            "drift_detected": result.drift_detected,
                            "drift_score": result.drift_score,
                            "drift_details": result.drift_details,
                            "reference_samples": result.reference_samples,
                            "current_samples": result.current_samples,
                        }
                    )

                return drift_history

        except Exception as e:
            logger.error(f"Failed to get drift history: {e}")
            return []

    def set_reference_data(self, reference_data: pd.DataFrame, data_type: str = "features"):
        """
        Set reference data for drift detection.

        Args:
            reference_data: Reference dataset
            data_type: Type of data ('features', 'predictions', 'performance')
        """
        try:
            self.reference_data[data_type] = {
                "data": reference_data.copy(),
                "timestamp": datetime.utcnow(),
                "stats": {},
            }

            # Calculate and store reference statistics
            if data_type == "features":
                for column in reference_data.columns:
                    if reference_data[column].dtype in ["int64", "float64"]:
                        self.reference_data[data_type]["stats"][column] = (
                            self._calculate_distribution_stats(reference_data[column])
                        )

            logger.info(
                "Reference data set",
                data_type=data_type,
                samples=len(reference_data),
                features=len(reference_data.columns) if hasattr(reference_data, "columns") else 0,
            )

        except Exception as e:
            logger.error(f"Failed to set reference data: {e}")
            raise ValidationError(f"Failed to set reference data: {e}") from e

    def get_reference_data(self, data_type: str = "features") -> pd.DataFrame | None:
        """
        Get reference data.

        Args:
            data_type: Type of data to retrieve

        Returns:
            Reference data or None if not set
        """
        return self.reference_data.get(data_type, {}).get("data")

    def clear_reference_data(self, data_type: str | None = None):
        """
        Clear reference data.

        Args:
            data_type: Specific data type to clear (None for all)
        """
        if data_type:
            self.reference_data.pop(data_type, None)
            logger.info(f"Reference data cleared for {data_type}")
        else:
            self.reference_data.clear()
            logger.info("All reference data cleared")
