"""
Hyperparameter Optimization using Optuna.

This module provides comprehensive hyperparameter optimization for ML models
using Optuna with pruning, parallel execution, and result tracking.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score

from src.core.config import Config
from src.core.exceptions import ModelError, ValidationError
from src.core.logging import get_logger
from src.ml.models.base_model import BaseModel
from src.utils.decorators import log_calls, time_execution

logger = get_logger(__name__)


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna.

    This class provides comprehensive hyperparameter optimization capabilities
    including study management, pruning, parallel execution, and result analysis.

    Attributes:
        config: Application configuration
        studies: Dictionary of active studies
        optimization_history: History of optimization runs
    """

    def __init__(self, config: Config):
        """
        Initialize the hyperparameter optimizer.

        Args:
            config: Application configuration
        """
        self.config = config
        self.studies: dict[str, optuna.Study] = {}
        self.optimization_history: list[dict[str, Any]] = []

        # Configuration from ML config
        self.n_trials = config.ml.optuna_n_trials
        self.timeout_hours = config.ml.optuna_timeout_hours
        self.pruning_enabled = config.ml.optuna_pruning_enabled

        logger.info(
            "Hyperparameter optimizer initialized",
            n_trials=self.n_trials,
            timeout_hours=self.timeout_hours,
            pruning_enabled=self.pruning_enabled,
        )

    @time_execution
    @log_calls
    def optimize_model(
        self,
        model_class: type,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        parameter_space: dict[str, Any] | None = None,
        scoring: str = "accuracy",
        cv_folds: int = 5,
        study_name: str | None = None,
        direction: str = "maximize",
        n_trials: int | None = None,
        timeout: int | None = None,
        pruner_type: str = "median",
        sampler_type: str = "tpe",
    ) -> dict[str, Any]:
        """
        Optimize hyperparameters for a model.

        Args:
            model_class: Model class to optimize
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            parameter_space: Custom parameter space definition
            scoring: Scoring metric for optimization
            cv_folds: Number of cross-validation folds
            study_name: Name for the optimization study
            direction: Optimization direction ('maximize' or 'minimize')
            n_trials: Number of trials to run
            timeout: Timeout in seconds
            pruner_type: Type of pruner ('median', 'successive_halving')
            sampler_type: Type of sampler ('tpe', 'random')

        Returns:
            Optimization results dictionary

        Raises:
            ModelError: If optimization fails
            ValidationError: If input validation fails
        """
        try:
            # Validate inputs
            if X_train.empty or y_train.empty:
                raise ValidationError("Training data cannot be empty")

            if X_val is not None and (X_val.empty or y_val.empty):
                raise ValidationError("Validation data cannot be empty if provided")

            # Use default values if not provided
            n_trials = n_trials or self.n_trials
            timeout = timeout or (self.timeout_hours * 3600)

            # Generate study name if not provided
            if study_name is None:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                study_name = f"{model_class.__name__}_{timestamp}"

            logger.info(
                "Starting hyperparameter optimization",
                model_class=model_class.__name__,
                study_name=study_name,
                n_trials=n_trials,
                timeout=timeout,
                scoring=scoring,
            )

            # Create study
            study = self._create_study(study_name, direction, pruner_type, sampler_type)

            # Get parameter space
            if parameter_space is None:
                parameter_space = self._get_default_parameter_space(model_class)

            # Create objective function
            objective_func = self._create_objective_function(
                model_class, X_train, y_train, X_val, y_val, parameter_space, scoring, cv_folds
            )

            # Run optimization
            study.optimize(
                objective_func,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=1,  # Use single job for stability
            )

            # Get results
            best_params = study.best_params
            best_value = study.best_value
            best_trial = study.best_trial

            # Create optimized model
            optimized_model = self._create_optimized_model(
                model_class, best_params, X_train, y_train, X_val, y_val
            )

            # Prepare results
            optimization_result = {
                "study_name": study_name,
                "model_class": model_class.__name__,
                "best_params": best_params,
                "best_value": best_value,
                "best_trial_number": best_trial.number,
                "n_trials": len(study.trials),
                "scoring": scoring,
                "cv_folds": cv_folds,
                "direction": direction,
                "optimization_time": study.trials[-1].datetime_complete
                - study.trials[0].datetime_start,
                "study": study,
                "optimized_model": optimized_model,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Store study and history
            self.studies[study_name] = study
            self.optimization_history.append(optimization_result)

            logger.info(
                "Hyperparameter optimization completed",
                study_name=study_name,
                best_value=best_value,
                n_trials=len(study.trials),
                best_params=best_params,
            )

            return optimization_result

        except Exception as e:
            logger.error(
                "Hyperparameter optimization failed",
                model_class=model_class.__name__ if model_class else "unknown",
                error=str(e),
            )
            raise ModelError(f"Hyperparameter optimization failed: {e}") from e

    @time_execution
    @log_calls
    def multi_model_optimization(
        self,
        model_classes: list[type],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        parameter_spaces: dict[str, dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Optimize multiple models in parallel.

        Args:
            model_classes: List of model classes to optimize
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            parameter_spaces: Custom parameter spaces for each model
            **kwargs: Additional optimization parameters

        Returns:
            List of optimization results
        """
        results = []

        for i, model_class in enumerate(model_classes):
            try:
                logger.info(
                    f"Optimizing model {i + 1}/{len(model_classes)}",
                    model_class=model_class.__name__,
                )

                # Get parameter space for this model
                param_space = None
                if parameter_spaces and model_class.__name__ in parameter_spaces:
                    param_space = parameter_spaces[model_class.__name__]

                # Optimize model
                result = self.optimize_model(
                    model_class,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    parameter_space=param_space,
                    **kwargs,
                )

                results.append(result)

            except Exception as e:
                logger.error(f"Failed to optimize {model_class.__name__}", error=str(e))
                results.append(
                    {"model_class": model_class.__name__, "error": str(e), "success": False}
                )

        # Rank results by best value
        successful_results = [r for r in results if r.get("success", True)]
        if successful_results:
            successful_results.sort(key=lambda x: x.get("best_value", float("-inf")), reverse=True)

        logger.info(
            "Multi-model optimization completed",
            total_models=len(model_classes),
            successful_models=len(successful_results),
        )

        return results

    def _create_study(
        self, study_name: str, direction: str, pruner_type: str, sampler_type: str
    ) -> optuna.Study:
        """Create an Optuna study with specified configuration."""
        # Create pruner
        if self.pruning_enabled:
            if pruner_type == "median":
                pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            elif pruner_type == "successive_halving":
                pruner = SuccessiveHalvingPruner()
            else:
                pruner = MedianPruner()
        else:
            pruner = optuna.pruners.NopPruner()

        # Create sampler
        if sampler_type == "tpe":
            sampler = TPESampler(seed=42)
        else:
            sampler = optuna.samplers.RandomSampler(seed=42)

        # Create study
        study = optuna.create_study(
            study_name=study_name, direction=direction, pruner=pruner, sampler=sampler
        )

        return study

    def _create_objective_function(
        self,
        model_class: type,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        parameter_space: dict[str, Any],
        scoring: str,
        cv_folds: int,
    ) -> Callable:
        """Create objective function for Optuna optimization."""

        def objective(trial: optuna.Trial) -> float:
            try:
                # Sample hyperparameters
                params = {}
                for param_name, param_config in parameter_space.items():
                    param_type = param_config["type"]

                    if param_type == "int":
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config["low"],
                            param_config["high"],
                            step=param_config.get("step", 1),
                        )
                    elif param_type == "float":
                        if param_config.get("log", False):
                            params[param_name] = trial.suggest_float(
                                param_name, param_config["low"], param_config["high"], log=True
                            )
                        else:
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_config["low"],
                                param_config["high"],
                                step=param_config.get("step"),
                            )
                    elif param_type == "categorical":
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_config["choices"]
                        )
                    elif param_type == "discrete_uniform":
                        params[param_name] = trial.suggest_discrete_uniform(
                            param_name, param_config["low"], param_config["high"], param_config["q"]
                        )

                # Create model with sampled parameters
                model = model_class(self.config, **params)

                # Evaluate model
                if X_val is not None and y_val is not None:
                    # Use validation set
                    model.train(X_train, y_train)
                    y_pred = model.predict(X_val)

                    # Calculate score based on problem type
                    if scoring == "accuracy":
                        from sklearn.metrics import accuracy_score

                        score = accuracy_score(y_val, y_pred)
                    elif scoring == "f1":
                        from sklearn.metrics import f1_score

                        score = f1_score(y_val, y_pred, average="weighted")
                    elif scoring == "roc_auc":
                        from sklearn.metrics import roc_auc_score

                        score = roc_auc_score(y_val, y_pred)
                    elif scoring == "neg_mean_squared_error":
                        from sklearn.metrics import mean_squared_error

                        score = -mean_squared_error(y_val, y_pred)
                    elif scoring == "r2":
                        from sklearn.metrics import r2_score

                        score = r2_score(y_val, y_pred)
                    else:
                        # Default to R2 for regression, accuracy for classification
                        if y_train.dtype in ["object", "category"] or y_train.nunique() < 10:
                            from sklearn.metrics import accuracy_score

                            score = accuracy_score(y_val, y_pred)
                        else:
                            from sklearn.metrics import r2_score

                            score = r2_score(y_val, y_pred)
                else:
                    # Use cross-validation
                    if hasattr(model, "model") and model.model is not None:
                        # If model has underlying sklearn model
                        scores = cross_val_score(
                            model.model, X_train, y_train, cv=cv_folds, scoring=scoring
                        )
                    else:
                        # Train model and use simple train score
                        model.train(X_train, y_train)
                        y_pred = model.predict(X_train)

                        if scoring == "accuracy":
                            from sklearn.metrics import accuracy_score

                            scores = [accuracy_score(y_train, y_pred)]
                        else:
                            from sklearn.metrics import r2_score

                            scores = [r2_score(y_train, y_pred)]

                    score = np.mean(scores)

                # Report intermediate value for pruning
                trial.report(score, step=0)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return score

            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                # Return worst possible score
                return float("-inf") if trial.study.direction.name == "MAXIMIZE" else float("inf")

        return objective

    def _get_default_parameter_space(self, model_class: type) -> dict[str, Any]:
        """Get default parameter space for a model class."""
        # This is a basic implementation - in practice, you'd have
        # model-specific parameter spaces

        model_name = model_class.__name__.lower()

        if "random" in model_name or "forest" in model_name:
            return {
                "n_estimators": {"type": "int", "low": 10, "high": 200},
                "max_depth": {"type": "int", "low": 3, "high": 20},
                "min_samples_split": {"type": "int", "low": 2, "high": 20},
                "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            }
        elif "gradient" in model_name or "xgb" in model_name or "lgb" in model_name:
            return {
                "n_estimators": {"type": "int", "low": 50, "high": 300},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            }
        elif "svm" in model_name or "svc" in model_name:
            return {
                "C": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
                "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
                "kernel": {"type": "categorical", "choices": ["rbf", "linear", "poly"]},
            }
        elif "linear" in model_name or "logistic" in model_name:
            return {
                "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
                "penalty": {"type": "categorical", "choices": ["l1", "l2", "elasticnet"]},
                "solver": {"type": "categorical", "choices": ["liblinear", "saga"]},
            }
        else:
            # Generic parameter space
            return {
                "learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
                "regularization": {"type": "float", "low": 0.0001, "high": 0.1, "log": True},
            }

    def _create_optimized_model(
        self,
        model_class: type,
        best_params: dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
    ) -> BaseModel:
        """Create and train model with optimized parameters."""
        # Create model with best parameters
        model = model_class(self.config, **best_params)

        # Train model
        if X_val is not None and y_val is not None:
            model.train(X_train, y_train, validation_data=(X_val, y_val))
        else:
            model.train(X_train, y_train)

        return model

    def get_study_results(self, study_name: str) -> dict[str, Any] | None:
        """Get results for a specific study."""
        if study_name not in self.studies:
            return None

        study = self.studies[study_name]

        return {
            "study_name": study_name,
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "trials_dataframe": study.trials_dataframe(),
            "optimization_history": [t.value for t in study.trials if t.value is not None],
        }

    def plot_optimization_history(self, study_name: str) -> Any | None:
        """Plot optimization history for a study."""
        if study_name not in self.studies:
            return None

        try:
            import matplotlib.pyplot as plt

            study = self.studies[study_name]

            # Plot optimization history
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.title(f"Optimization History - {study_name}")
            plt.show()

            return plt.gcf()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None

    def plot_param_importances(self, study_name: str) -> Any | None:
        """Plot parameter importances for a study."""
        if study_name not in self.studies:
            return None

        try:
            import matplotlib.pyplot as plt

            study = self.studies[study_name]

            # Plot parameter importances
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.title(f"Parameter Importances - {study_name}")
            plt.show()

            return plt.gcf()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None

    def get_optimization_history(self) -> list[dict[str, Any]]:
        """Get complete optimization history."""
        return self.optimization_history.copy()

    def clear_studies(self) -> None:
        """Clear all studies and history."""
        self.studies.clear()
        self.optimization_history.clear()
        logger.info("All studies and optimization history cleared")
