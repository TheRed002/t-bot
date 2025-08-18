"""
Cross-Validation for ML Models.

This module provides comprehensive cross-validation capabilities including
time series cross-validation, nested cross-validation, and custom validation strategies.
"""

from collections.abc import Generator
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
)

from src.core.config import Config
from src.core.exceptions import ModelError, ValidationError
from src.core.logging import get_logger
from src.ml.models.base_model import BaseModel
from src.utils.decorators import log_calls, time_execution

logger = get_logger(__name__)


class TimeSeriesValidator:
    """
    Time series specific cross-validation strategies.

    This class provides validation strategies that respect the temporal nature
    of financial time series data.
    """

    @staticmethod
    def walk_forward_split(
        data: pd.DataFrame, min_train_size: int, test_size: int, step_size: int = 1
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Walk-forward time series split.

        Args:
            data: Time series data
            min_train_size: Minimum training set size
            test_size: Test set size
            step_size: Step size for walking forward

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(data)

        for i in range(min_train_size, n_samples - test_size + 1, step_size):
            train_indices = np.arange(0, i)
            test_indices = np.arange(i, min(i + test_size, n_samples))

            if len(test_indices) == test_size:
                yield train_indices, test_indices

    @staticmethod
    def expanding_window_split(
        data: pd.DataFrame, min_train_size: int, test_size: int, step_size: int = 1
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Expanding window time series split.

        Args:
            data: Time series data
            min_train_size: Minimum training set size
            test_size: Test set size
            step_size: Step size for expanding

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(data)

        for i in range(min_train_size, n_samples - test_size + 1, step_size):
            train_indices = np.arange(0, i)
            test_indices = np.arange(i, min(i + test_size, n_samples))

            if len(test_indices) == test_size:
                yield train_indices, test_indices

    @staticmethod
    def sliding_window_split(
        data: pd.DataFrame, train_size: int, test_size: int, step_size: int = 1
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Sliding window time series split.

        Args:
            data: Time series data
            train_size: Training set size
            test_size: Test set size
            step_size: Step size for sliding

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(data)

        for i in range(0, n_samples - train_size - test_size + 1, step_size):
            train_indices = np.arange(i, i + train_size)
            test_indices = np.arange(i + train_size, i + train_size + test_size)

            yield train_indices, test_indices


class CrossValidator:
    """
    Cross-validation system for ML models.

    This class provides comprehensive cross-validation capabilities including
    standard CV, time series CV, nested CV, and custom validation strategies.

    Attributes:
        config: Application configuration
        validation_history: History of validation runs
    """

    def __init__(self, config: Config):
        """
        Initialize the cross-validator.

        Args:
            config: Application configuration
        """
        self.config = config
        self.validation_history: list[dict[str, Any]] = []
        self.ts_validator = TimeSeriesValidator()

        # Configuration
        self.cv_folds = config.ml.cross_validation_folds

        logger.info("Cross-validator initialized", cv_folds=self.cv_folds)

    @time_execution
    @log_calls
    def validate_model(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        cv_strategy: str = "kfold",
        scoring: str | list[str] = "accuracy",
        cv_folds: int | None = None,
        return_train_score: bool = False,
        **cv_kwargs,
    ) -> dict[str, Any]:
        """
        Perform cross-validation on a model.

        Args:
            model: Model to validate
            X: Feature data
            y: Target data
            cv_strategy: Cross-validation strategy
            scoring: Scoring metric(s)
            cv_folds: Number of folds (uses config default if None)
            return_train_score: Whether to return training scores
            **cv_kwargs: Additional CV parameters

        Returns:
            Cross-validation results

        Raises:
            ModelError: If validation fails
            ValidationError: If input validation fails
        """
        try:
            if X.empty or y.empty:
                raise ValidationError("Feature and target data cannot be empty")

            cv_folds = cv_folds or self.cv_folds

            logger.info(
                "Starting cross-validation",
                model_name=model.model_name,
                cv_strategy=cv_strategy,
                cv_folds=cv_folds,
                scoring=scoring,
            )

            # Create cross-validation splitter
            cv_splitter = self._create_cv_splitter(cv_strategy, cv_folds, y, **cv_kwargs)

            # Perform cross-validation
            if isinstance(scoring, str):
                # Single metric
                if hasattr(model, "model") and model.model is not None:
                    # Use sklearn's cross_validate for sklearn models
                    cv_results = cross_validate(
                        model.model,
                        X,
                        y,
                        cv=cv_splitter,
                        scoring=scoring,
                        return_train_score=return_train_score,
                        n_jobs=1,
                    )
                else:
                    # Manual cross-validation for custom models
                    cv_results = self._manual_cross_validation(
                        model, X, y, cv_splitter, scoring, return_train_score
                    )
            else:
                # Multiple metrics
                cv_results = {}
                for metric in scoring:
                    if hasattr(model, "model") and model.model is not None:
                        metric_results = cross_validate(
                            model.model,
                            X,
                            y,
                            cv=cv_splitter,
                            scoring=metric,
                            return_train_score=return_train_score,
                            n_jobs=1,
                        )
                        cv_results[metric] = metric_results
                    else:
                        metric_results = self._manual_cross_validation(
                            model, X, y, cv_splitter, metric, return_train_score
                        )
                        cv_results[metric] = metric_results

            # Process results
            validation_result = self._process_cv_results(
                cv_results, model, cv_strategy, scoring, cv_folds
            )

            # Store in history
            self.validation_history.append(validation_result)

            logger.info(
                "Cross-validation completed",
                model_name=model.model_name,
                mean_score=validation_result.get("mean_test_score"),
                std_score=validation_result.get("std_test_score"),
            )

            return validation_result

        except Exception as e:
            logger.error("Cross-validation failed", model_name=model.model_name, error=str(e))
            raise ModelError(f"Cross-validation failed: {e}") from e

    @time_execution
    @log_calls
    def time_series_validation(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        ts_strategy: str = "walk_forward",
        min_train_size: int | None = None,
        test_size: int | None = None,
        step_size: int = 1,
        scoring: str = "r2",
    ) -> dict[str, Any]:
        """
        Perform time series cross-validation.

        Args:
            model: Model to validate
            X: Feature data with datetime index
            y: Target data with datetime index
            ts_strategy: Time series strategy ('walk_forward', 'expanding', 'sliding')
            min_train_size: Minimum training size
            test_size: Test set size
            step_size: Step size for validation
            scoring: Scoring metric

        Returns:
            Time series validation results
        """
        try:
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValidationError("Time series validation requires datetime index")

            # Default sizes
            if min_train_size is None:
                min_train_size = max(100, len(X) // 4)
            if test_size is None:
                test_size = max(10, len(X) // 20)

            logger.info(
                "Starting time series validation",
                model_name=model.model_name,
                strategy=ts_strategy,
                min_train_size=min_train_size,
                test_size=test_size,
            )

            # Get time series splits
            if ts_strategy == "walk_forward":
                splits = list(
                    self.ts_validator.walk_forward_split(X, min_train_size, test_size, step_size)
                )
            elif ts_strategy == "expanding":
                splits = list(
                    self.ts_validator.expanding_window_split(
                        X, min_train_size, test_size, step_size
                    )
                )
            elif ts_strategy == "sliding":
                train_size = min_train_size
                splits = list(
                    self.ts_validator.sliding_window_split(X, train_size, test_size, step_size)
                )
            else:
                raise ValidationError(f"Unknown time series strategy: {ts_strategy}")

            # Perform validation
            fold_scores = []
            fold_predictions = []

            for fold, (train_idx, test_idx) in enumerate(splits):
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Create fresh model instance
                model_copy = type(model)(self.config, model.model_name)

                # Train model
                model_copy.train(X_train, y_train)

                # Make predictions
                y_pred = model_copy.predict(X_test)

                # Calculate score
                score = self._calculate_score(y_test, y_pred, scoring)
                fold_scores.append(score)

                # Store predictions with timestamps
                fold_predictions.append(
                    {
                        "fold": fold,
                        "timestamps": X_test.index,
                        "actual": y_test.values,
                        "predicted": y_pred,
                        "score": score,
                    }
                )

            # Calculate statistics
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            # Prepare results
            ts_validation_result = {
                "model_name": model.model_name,
                "model_type": model.model_type,
                "ts_strategy": ts_strategy,
                "scoring": scoring,
                "n_splits": len(splits),
                "min_train_size": min_train_size,
                "test_size": test_size,
                "step_size": step_size,
                "fold_scores": fold_scores,
                "mean_score": mean_score,
                "std_score": std_score,
                "fold_predictions": fold_predictions,
                "timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Time series validation completed",
                model_name=model.model_name,
                mean_score=mean_score,
                std_score=std_score,
                n_splits=len(splits),
            )

            return ts_validation_result

        except Exception as e:
            logger.error("Time series validation failed", model_name=model.model_name, error=str(e))
            raise ModelError(f"Time series validation failed: {e}") from e

    @time_execution
    @log_calls
    def nested_cross_validation(
        self,
        model_class: type,
        X: pd.DataFrame,
        y: pd.Series,
        parameter_grid: dict[str, list[Any]],
        outer_cv_folds: int = 5,
        inner_cv_folds: int = 3,
        scoring: str = "accuracy",
    ) -> dict[str, Any]:
        """
        Perform nested cross-validation for unbiased model evaluation.

        Args:
            model_class: Model class to evaluate
            X: Feature data
            y: Target data
            parameter_grid: Grid of parameters to search
            outer_cv_folds: Number of outer CV folds
            inner_cv_folds: Number of inner CV folds
            scoring: Scoring metric

        Returns:
            Nested cross-validation results
        """
        try:
            from sklearn.model_selection import GridSearchCV

            logger.info(
                "Starting nested cross-validation",
                model_class=model_class.__name__,
                outer_folds=outer_cv_folds,
                inner_folds=inner_cv_folds,
            )

            # Create outer CV splitter
            outer_cv = self._create_cv_splitter("kfold", outer_cv_folds, y)
            inner_cv = self._create_cv_splitter("kfold", inner_cv_folds, y)

            outer_scores = []
            best_params_per_fold = []

            for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                # Split data for outer fold
                X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
                y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]

                # Create model instance
                model = model_class(self.config)

                # Grid search on inner folds
                if hasattr(model, "model") and model.model is not None:
                    grid_search = GridSearchCV(
                        model.model, parameter_grid, cv=inner_cv, scoring=scoring, n_jobs=1
                    )
                    grid_search.fit(X_train_outer, y_train_outer)

                    # Get best model
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_

                    # Evaluate on outer test set
                    y_pred = best_model.predict(X_test_outer)
                    score = self._calculate_score(y_test_outer, y_pred, scoring)
                else:
                    # For custom models, do manual grid search
                    best_score = float("-inf")
                    best_params = None

                    # Simple grid search
                    from itertools import product

                    param_combinations = list(product(*parameter_grid.values()))

                    for params in param_combinations:
                        param_dict = dict(zip(parameter_grid.keys(), params, strict=False))

                        # Create and evaluate model
                        test_model = model_class(self.config, **param_dict)

                        # Cross-validate on inner folds
                        inner_scores = []
                        for inner_train_idx, inner_val_idx in inner_cv.split(
                            X_train_outer, y_train_outer
                        ):
                            X_inner_train = X_train_outer.iloc[inner_train_idx]
                            y_inner_train = y_train_outer.iloc[inner_train_idx]
                            X_inner_val = X_train_outer.iloc[inner_val_idx]
                            y_inner_val = y_train_outer.iloc[inner_val_idx]

                            test_model.train(X_inner_train, y_inner_train)
                            y_inner_pred = test_model.predict(X_inner_val)
                            inner_score = self._calculate_score(y_inner_val, y_inner_pred, scoring)
                            inner_scores.append(inner_score)

                        avg_inner_score = np.mean(inner_scores)
                        if avg_inner_score > best_score:
                            best_score = avg_inner_score
                            best_params = param_dict

                    # Train best model on full outer training set
                    final_model = model_class(self.config, **best_params)
                    final_model.train(X_train_outer, y_train_outer)

                    # Evaluate on outer test set
                    y_pred = final_model.predict(X_test_outer)
                    score = self._calculate_score(y_test_outer, y_pred, scoring)

                outer_scores.append(score)
                best_params_per_fold.append(best_params)

                logger.info(f"Outer fold {fold + 1} completed, score: {score}")

            # Calculate final statistics
            mean_score = np.mean(outer_scores)
            std_score = np.std(outer_scores)

            nested_cv_result = {
                "model_class": model_class.__name__,
                "scoring": scoring,
                "outer_cv_folds": outer_cv_folds,
                "inner_cv_folds": inner_cv_folds,
                "parameter_grid": parameter_grid,
                "outer_scores": outer_scores,
                "mean_score": mean_score,
                "std_score": std_score,
                "best_params_per_fold": best_params_per_fold,
                "timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Nested cross-validation completed",
                model_class=model_class.__name__,
                mean_score=mean_score,
                std_score=std_score,
            )

            return nested_cv_result

        except Exception as e:
            logger.error(
                "Nested cross-validation failed", model_class=model_class.__name__, error=str(e)
            )
            raise ModelError(f"Nested cross-validation failed: {e}") from e

    def _create_cv_splitter(self, cv_strategy: str, cv_folds: int, y: pd.Series, **kwargs):
        """Create cross-validation splitter based on strategy."""
        if cv_strategy == "kfold":
            return KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        elif cv_strategy == "stratified":
            return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        elif cv_strategy == "timeseries":
            return TimeSeriesSplit(n_splits=cv_folds)
        else:
            raise ValidationError(f"Unknown CV strategy: {cv_strategy}")

    def _manual_cross_validation(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        cv_splitter,
        scoring: str,
        return_train_score: bool,
    ) -> dict[str, np.ndarray]:
        """Perform manual cross-validation for custom models."""
        test_scores = []
        train_scores = []

        for train_idx, test_idx in cv_splitter.split(X, y):
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Create fresh model instance
            model_copy = type(model)(self.config, model.model_name)

            # Train model
            model_copy.train(X_train, y_train)

            # Calculate test score
            y_test_pred = model_copy.predict(X_test)
            test_score = self._calculate_score(y_test, y_test_pred, scoring)
            test_scores.append(test_score)

            # Calculate train score if requested
            if return_train_score:
                y_train_pred = model_copy.predict(X_train)
                train_score = self._calculate_score(y_train, y_train_pred, scoring)
                train_scores.append(train_score)

        results = {"test_score": np.array(test_scores)}
        if return_train_score:
            results["train_score"] = np.array(train_scores)

        return results

    def _calculate_score(self, y_true, y_pred, scoring: str) -> float:
        """Calculate score based on scoring metric."""
        if scoring == "accuracy":
            return accuracy_score(y_true, y_pred)
        elif scoring == "f1":
            return f1_score(y_true, y_pred, average="weighted")
        elif scoring == "f1_macro":
            return f1_score(y_true, y_pred, average="macro")
        elif scoring == "roc_auc":
            return roc_auc_score(y_true, y_pred)
        elif scoring == "neg_mean_squared_error":
            return -mean_squared_error(y_true, y_pred)
        elif scoring == "r2":
            return r2_score(y_true, y_pred)
        elif scoring == "neg_root_mean_squared_error":
            return -np.sqrt(mean_squared_error(y_true, y_pred))
        else:
            # Default to r2 for regression, accuracy for classification
            if len(np.unique(y_true)) < 10:
                return accuracy_score(y_true, y_pred)
            else:
                return r2_score(y_true, y_pred)

    def _process_cv_results(
        self,
        cv_results: dict[str, Any],
        model: BaseModel,
        cv_strategy: str,
        scoring: str | list[str],
        cv_folds: int,
    ) -> dict[str, Any]:
        """Process cross-validation results."""
        if isinstance(scoring, str):
            # Single metric
            test_scores = cv_results.get("test_score", [])
            train_scores = cv_results.get("train_score", [])

            result = {
                "model_name": model.model_name,
                "model_type": model.model_type,
                "cv_strategy": cv_strategy,
                "cv_folds": cv_folds,
                "scoring": scoring,
                "test_scores": test_scores,
                "mean_test_score": np.mean(test_scores),
                "std_test_score": np.std(test_scores),
                "timestamp": datetime.utcnow().isoformat(),
            }

            if len(train_scores) > 0:
                result.update(
                    {
                        "train_scores": train_scores,
                        "mean_train_score": np.mean(train_scores),
                        "std_train_score": np.std(train_scores),
                    }
                )
        else:
            # Multiple metrics
            result = {
                "model_name": model.model_name,
                "model_type": model.model_type,
                "cv_strategy": cv_strategy,
                "cv_folds": cv_folds,
                "scoring": scoring,
                "timestamp": datetime.utcnow().isoformat(),
            }

            for metric in scoring:
                metric_results = cv_results.get(metric, {})
                test_scores = metric_results.get("test_score", [])
                train_scores = metric_results.get("train_score", [])

                result[f"{metric}_test_scores"] = test_scores
                result[f"{metric}_mean_test_score"] = np.mean(test_scores)
                result[f"{metric}_std_test_score"] = np.std(test_scores)

                if len(train_scores) > 0:
                    result[f"{metric}_train_scores"] = train_scores
                    result[f"{metric}_mean_train_score"] = np.mean(train_scores)
                    result[f"{metric}_std_train_score"] = np.std(train_scores)

        return result

    def get_validation_history(self) -> list[dict[str, Any]]:
        """Get validation history."""
        return self.validation_history.copy()

    def clear_history(self) -> None:
        """Clear validation history."""
        self.validation_history.clear()
        logger.info("Validation history cleared")
