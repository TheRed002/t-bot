"""
Cross-Validation for ML Models.

This module provides comprehensive cross-validation capabilities including
time series cross-validation, nested cross-validation, and custom validation strategies.
"""

from collections.abc import Generator
from datetime import datetime, timezone
from decimal import Decimal
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

from src.core.base.service import BaseService
from src.core.exceptions import ModelError, ValidationError
from src.core.types.base import ConfigDict
from src.ml.models.base_model import BaseMLModel
from src.utils.decorators import UnifiedDecorator

# Initialize decorator instance
dec = UnifiedDecorator()


class TimeSeriesValidator:
    """
    Time series specific cross-validation strategies.

    This class provides validation strategies that respect the temporal nature
    of financial time series data, including purged cross-validation to prevent
    lookahead bias and data leakage.
    """

    @staticmethod
    def purged_walk_forward_split(
        data: pd.DataFrame,
        min_train_size: int,
        test_size: int,
        embargo_period: int = 0,
        step_size: int = 1,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Purged walk-forward split with embargo periods to prevent lookahead bias.

        This method implements proper time series cross-validation for financial data
        by ensuring no overlap between training and test sets and adding embargo
        periods to account for autocorrelation and delayed market reactions.

        Args:
            data: Time series data with datetime index
            min_train_size: Minimum training set size
            test_size: Test set size
            embargo_period: Number of periods to skip after training (default: 0)
            step_size: Step size for walking forward

        Yields:
            Tuple of (train_indices, test_indices) with proper purging
        """
        n_samples = len(data)

        for i in range(min_train_size, n_samples - test_size - embargo_period + 1, step_size):
            # Training set: from start to i
            train_indices = np.arange(0, i)

            # Embargo period: skip embargo_period samples after training
            test_start = i + embargo_period
            test_end = min(test_start + test_size, n_samples)

            # Test set: after embargo period
            test_indices = np.arange(test_start, test_end)

            if len(test_indices) == test_size:
                yield train_indices, test_indices

    @staticmethod
    def combinatorial_purged_cross_validation(
        data: pd.DataFrame,
        n_splits: int = 5,
        test_size_ratio: float = 0.2,
        embargo_ratio: float = 0.01,
        purge_ratio: float = 0.02,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Combinatorial Purged Cross-Validation (CPCV) for financial time series.

        This advanced cross-validation technique prevents data leakage by:
        1. Purging overlapping observations based on feature generation windows
        2. Adding embargo periods to account for non-instantaneous information incorporation
        3. Using combinatorial approach to maximize data usage while maintaining independence

        Args:
            data: Time series data with datetime index
            n_splits: Number of CV splits
            test_size_ratio: Ratio of data for test set
            embargo_ratio: Ratio of data for embargo period
            purge_ratio: Ratio of data to purge around test set

        Yields:
            Tuple of (train_indices, test_indices) with purging and embargo
        """
        n_samples = len(data)
        test_size = int(n_samples * test_size_ratio)
        embargo_size = int(n_samples * embargo_ratio)
        purge_size = int(n_samples * purge_ratio)

        # Generate test set start points
        test_starts = np.linspace(
            purge_size, n_samples - test_size - embargo_size - purge_size, n_splits, dtype=int
        )

        for test_start in test_starts:
            # Test set indices
            test_end = test_start + test_size
            test_indices = np.arange(test_start, test_end)

            # Purged training set: remove data around test period
            purge_start = test_start - purge_size
            purge_end = test_end + embargo_size + purge_size

            # Create training indices (everything except purged region)
            all_indices = np.arange(n_samples)
            purged_indices = np.arange(max(0, purge_start), min(n_samples, purge_end))
            train_indices = np.setdiff1d(all_indices, purged_indices)

            # Ensure minimum training size
            if len(train_indices) >= len(data) // 4:  # At least 25% for training
                yield train_indices, test_indices

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


class CrossValidationService(BaseService):
    """
    Cross-validation service for ML models.

    This service provides comprehensive cross-validation capabilities including
    standard CV, time series CV, nested CV, and custom validation strategies.

    Attributes:
        validation_history: History of validation runs
        ts_validator: Time series validator instance
    """

    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None):
        """
        Initialize the cross-validation service.

        Args:
            config: Service configuration
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="CrossValidationService",
            config=config,
            correlation_id=correlation_id,
        )

        # Service state
        self.validation_history: list[dict[str, Any]] = []
        self.ts_validator = TimeSeriesValidator()

        # Configuration with defaults
        self.cv_folds = self._config.get("ml", {}).get("cross_validation_folds", 5)

        # Dependencies that will be resolved during startup
        self.add_dependency("ModelFactory")

        self._logger.info("Cross-validation service initialized", cv_folds=self.cv_folds)

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def validate_model(
        self,
        model: BaseMLModel,
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

            self._logger.info(
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

            self._logger.info(
                "Cross-validation completed",
                model_name=model.model_name,
                mean_score=validation_result.get("mean_test_score"),
                std_score=validation_result.get("std_test_score"),
            )

            return validation_result

        except Exception as e:
            self._logger.error("Cross-validation failed", model_name=model.model_name, error=str(e))
            raise ModelError(f"Cross-validation failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def time_series_validation(
        self,
        model: BaseMLModel,
        X: pd.DataFrame,
        y: pd.Series,
        ts_strategy: str = "purged_walk_forward",
        min_train_size: int | None = None,
        test_size: int | None = None,
        step_size: int = 1,
        embargo_period: int = 0,
        scoring: str = "sharpe_ratio",
    ) -> dict[str, Any]:
        """
        Perform time series cross-validation.

        Args:
            model: Model to validate
            X: Feature data with datetime index
            y: Target data with datetime index
            ts_strategy: Time series strategy ('purged_walk_forward', 'combinatorial_purged', 'walk_forward', 'expanding', 'sliding')
            min_train_size: Minimum training size
            test_size: Test set size
            step_size: Step size for validation
            embargo_period: Number of periods to skip after training to prevent lookahead bias
            scoring: Scoring metric (supports trading-specific metrics)

        Returns:
            Time series validation results with trading performance metrics
        """
        try:
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValidationError("Time series validation requires datetime index")

            # Default sizes
            if min_train_size is None:
                min_train_size = max(100, len(X) // 4)
            if test_size is None:
                test_size = max(10, len(X) // 20)

            self._logger.info(
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
            elif ts_strategy == "purged_walk_forward":
                splits = list(
                    self.ts_validator.purged_walk_forward_split(
                        X, min_train_size, test_size, embargo_period, step_size
                    )
                )
            elif ts_strategy == "combinatorial_purged":
                splits = list(
                    self.ts_validator.combinatorial_purged_cross_validation(
                        X, n_splits=5, test_size_ratio=0.2
                    )
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
                model_factory = self.resolve_dependency("ModelFactory")
                model_copy = model_factory.create_model(type(model).__name__, model.model_name)

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
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self._logger.info(
                "Time series validation completed",
                model_name=model.model_name,
                mean_score=mean_score,
                std_score=std_score,
                n_splits=len(splits),
            )

            return ts_validation_result

        except Exception as e:
            self._logger.error(
                "Time series validation failed", model_name=model.model_name, error=str(e)
            )
            raise ModelError(f"Time series validation failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def nested_cross_validation(
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

            self._logger.info(
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
                model_factory = self.resolve_dependency("ModelFactory")
                model = model_factory.create_model(model_class.__name__)

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
                        model_factory = self.resolve_dependency("ModelFactory")
                        test_model = model_factory.create_model(model_class.__name__, **param_dict)

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
                    model_factory = self.resolve_dependency("ModelFactory")
                    final_model = model_factory.create_model(model_class.__name__, **best_params)
                    final_model.train(X_train_outer, y_train_outer)

                    # Evaluate on outer test set
                    y_pred = final_model.predict(X_test_outer)
                    score = self._calculate_score(y_test_outer, y_pred, scoring)

                outer_scores.append(score)
                best_params_per_fold.append(best_params)

                self._logger.info(f"Outer fold {fold + 1} completed, score: {score}")

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
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self._logger.info(
                "Nested cross-validation completed",
                model_class=model_class.__name__,
                mean_score=mean_score,
                std_score=std_score,
            )

            return nested_cv_result

        except Exception as e:
            self._logger.error(
                "Nested cross-validation failed", model_class=model_class.__name__, error=str(e)
            )
            raise ModelError(f"Nested cross-validation failed: {e}")

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
        model: BaseMLModel,
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
            model_factory = self.resolve_dependency("ModelFactory")
            model_copy = model_factory.create_model(type(model).__name__, model.model_name)

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
        """Calculate score based on scoring metric, including trading-specific metrics."""
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
        elif scoring == "sharpe_ratio":
            return self._calculate_sharpe_ratio(y_true, y_pred)
        elif scoring == "information_ratio":
            return self._calculate_information_ratio(y_true, y_pred)
        elif scoring == "calmar_ratio":
            return self._calculate_calmar_ratio(y_true, y_pred)
        elif scoring == "max_drawdown":
            return -self._calculate_max_drawdown(y_true, y_pred)  # Negative for minimization
        elif scoring == "hit_ratio":
            return self._calculate_hit_ratio(y_true, y_pred)
        elif scoring == "profit_factor":
            return self._calculate_profit_factor(y_true, y_pred)
        else:
            # Default to r2 for regression, accuracy for classification
            if len(np.unique(y_true)) < 10:
                return accuracy_score(y_true, y_pred)
            else:
                return r2_score(y_true, y_pred)

    def _calculate_sharpe_ratio(
        self, y_true: np.ndarray, y_pred: np.ndarray, risk_free_rate: float = 0.02
    ) -> Decimal:
        """
        Calculate Sharpe ratio for trading predictions.

        Args:
            y_true: Actual returns
            y_pred: Predicted returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio (annualized)
        """
        # Convert predictions to trading signals with Decimal precision
        returns_decimal = []
        for true_val, pred_val in zip(y_true, y_pred, strict=False):
            return_val = Decimal(str(true_val)) * Decimal("1" if pred_val >= 0 else "-1")
            returns_decimal.append(return_val)

        if len(returns_decimal) == 0:
            return Decimal("0")

        # Calculate standard deviation with Decimal precision
        mean_return_decimal = sum(returns_decimal) / len(returns_decimal)
        variance = sum((r - mean_return_decimal) ** 2 for r in returns_decimal) / len(returns_decimal)
        std_return_decimal = variance.sqrt() if variance > 0 else Decimal("0")

        if std_return_decimal == 0:
            return Decimal("0")

        # Annualize assuming daily returns
        annual_mean = mean_return_decimal * Decimal("252")  # 252 trading days
        annual_std = std_return_decimal * Decimal("252").sqrt()

        return (annual_mean - Decimal(str(risk_free_rate))) / annual_std if annual_std != 0 else Decimal("0")

    def _calculate_information_ratio(self, y_true: np.ndarray, y_pred: np.ndarray) -> Decimal:
        """
        Calculate Information Ratio (excess return / tracking error).

        Args:
            y_true: Actual returns
            y_pred: Predicted returns

        Returns:
            Information ratio
        """
        # Calculate excess returns with Decimal precision
        excess_returns_decimal = []
        for true_val, pred_val in zip(y_true, y_pred, strict=False):
            excess_return = Decimal(str(true_val)) * Decimal("1" if pred_val >= 0 else "-1")
            excess_returns_decimal.append(excess_return)

        if len(excess_returns_decimal) == 0:
            return Decimal("0")

        mean_excess = sum(excess_returns_decimal) / len(excess_returns_decimal)
        variance = sum((r - mean_excess) ** 2 for r in excess_returns_decimal) / len(excess_returns_decimal)
        std_excess = variance.sqrt() if variance > 0 else Decimal("0")

        if std_excess == 0:
            return Decimal("0")

        return mean_excess / std_excess * Decimal("252").sqrt()

    def _calculate_calmar_ratio(self, y_true: np.ndarray, y_pred: np.ndarray) -> Decimal:
        """
        Calculate Calmar ratio (annual return / max drawdown).

        Args:
            y_true: Actual returns
            y_pred: Predicted returns

        Returns:
            Calmar ratio
        """
        # Calculate returns with Decimal precision
        returns_decimal = []
        for true_val, pred_val in zip(y_true, y_pred, strict=False):
            return_val = Decimal(str(true_val)) * Decimal("1" if pred_val >= 0 else "-1")
            returns_decimal.append(return_val)

        if len(returns_decimal) == 0:
            return Decimal("0")

        annual_return = sum(returns_decimal) / len(returns_decimal) * Decimal("252")
        max_dd = self._calculate_max_drawdown(y_true, y_pred)

        return annual_return / max_dd if max_dd != 0 else Decimal("0")

    def _calculate_max_drawdown(self, y_true: np.ndarray, y_pred: np.ndarray) -> Decimal:
        """
        Calculate maximum drawdown for trading strategy.

        Args:
            y_true: Actual returns
            y_pred: Predicted returns

        Returns:
            Maximum drawdown (positive value)
        """
        # Calculate returns with Decimal precision
        returns_decimal = []
        for true_val, pred_val in zip(y_true, y_pred, strict=False):
            return_val = Decimal(str(true_val)) * Decimal("1" if pred_val >= 0 else "-1")
            returns_decimal.append(return_val)

        if len(returns_decimal) == 0:
            return Decimal("0")

        # Calculate cumulative returns with Decimal precision
        cumulative_returns = []
        cumulative_value = Decimal("1")
        for r in returns_decimal:
            cumulative_value *= (Decimal("1") + r)
            cumulative_returns.append(cumulative_value)

        # Calculate maximum drawdown with Decimal precision
        max_dd = Decimal("0")
        running_max = Decimal("0")

        for cum_return in cumulative_returns:
            if cum_return > running_max:
                running_max = cum_return

            if running_max > 0:
                drawdown = (cum_return - running_max) / running_max
                if drawdown < max_dd:
                    max_dd = drawdown

        return abs(max_dd)

    def _calculate_hit_ratio(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate hit ratio (percentage of correct directional predictions).

        Args:
            y_true: Actual returns
            y_pred: Predicted returns

        Returns:
            Hit ratio (0 to 1)
        """
        if len(y_true) == 0:
            return 0.0

        # Check if prediction and actual have same sign
        correct_direction = np.sign(y_true) == np.sign(y_pred)

        return np.mean(correct_direction)

    def _calculate_profit_factor(self, y_true: np.ndarray, y_pred: np.ndarray) -> Decimal:
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            y_true: Actual returns
            y_pred: Predicted returns

        Returns:
            Profit factor
        """
        # Calculate returns with Decimal precision
        returns_decimal = []
        for true_val, pred_val in zip(y_true, y_pred, strict=False):
            return_val = Decimal(str(true_val)) * Decimal("1" if pred_val >= 0 else "-1")
            returns_decimal.append(return_val)

        if len(returns_decimal) == 0:
            return Decimal("0")

        gross_profit = sum(r for r in returns_decimal if r > 0)
        gross_loss = abs(sum(r for r in returns_decimal if r < 0))

        return gross_profit / gross_loss if gross_loss != 0 else Decimal("inf")

    def _process_cv_results(
        self,
        cv_results: dict[str, Any],
        model: BaseMLModel,
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
                "timestamp": datetime.now(timezone.utc).isoformat(),
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
                "timestamp": datetime.now(timezone.utc).isoformat(),
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

    async def _do_start(self) -> None:
        """Start the service and resolve dependencies."""
        await super()._do_start()
        self._logger.info("Cross-validation service started successfully")

    async def _do_stop(self) -> None:
        """Stop the service and cleanup resources."""
        await super()._do_stop()
        self._logger.info("Cross-validation service stopped")

    async def _service_health_check(self) -> "HealthStatus":
        """Check service-specific health."""
        from src.core.base.interfaces import HealthStatus

        # Check if we have reasonable validation history size
        if len(self.validation_history) > 10000:  # Too many entries might indicate memory issues
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def get_validation_history(self) -> list[dict[str, Any]]:
        """Get validation history."""
        return self.validation_history.copy()

    def clear_history(self) -> None:
        """Clear validation history."""
        self.validation_history.clear()
        self._logger.info("Validation history cleared")
