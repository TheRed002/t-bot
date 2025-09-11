"""
Hyperparameter Optimization using Optuna.

Simple hyperparameter optimization for ML models using Optuna.
"""

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score

from src.core.base.service import BaseService
from src.core.exceptions import ModelError, ValidationError
from src.core.types.base import ConfigDict


class HyperparameterOptimizationService(BaseService):
    """Simple hyperparameter optimization service using Optuna."""

    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None) -> None:
        """Initialize the hyperparameter optimization service."""
        super().__init__(
            name="HyperparameterOptimizationService",
            config=config,
            correlation_id=correlation_id,
        )

        # Configuration
        ml_config = self._config.get("ml", {})
        self.n_trials = ml_config.get("optuna_n_trials", 100)

        # Dependencies
        self.add_dependency("ModelFactory")

        self._logger.info("Hyperparameter optimization service initialized")

    async def optimize_model(
        self,
        model_class: type,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        parameter_space: dict[str, Any] | None = None,
        scoring: str = "accuracy",
        n_trials: int | None = None,
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
            n_trials: Number of trials to run

        Returns:
            Optimization results dictionary
        """
        if X_train.empty or y_train.empty:
            raise ValidationError("Training data cannot be empty")

        try:
            n_trials = n_trials or self.n_trials

            # Get parameter space
            if parameter_space is None:
                parameter_space = self._get_default_parameter_space(model_class)

            # Create study
            study = optuna.create_study(direction="maximize")

            # Create objective function
            objective_func = self._create_objective_function(
                model_class, X_train, y_train, X_val, y_val, parameter_space, scoring
            )

            # Run optimization
            study.optimize(objective_func, n_trials=n_trials)

            # Create optimized model
            model_factory = self.resolve_dependency("ModelFactory")
            optimized_model = model_factory.create_model(model_class.__name__, **study.best_params)

            # Train the optimized model
            if X_val is not None and y_val is not None:
                optimized_model.train(X_train, y_train, validation_data=(X_val, y_val))
            else:
                optimized_model.train(X_train, y_train)

            result = {
                "model_class": model_class.__name__,
                "best_params": study.best_params,
                "best_value": study.best_value,
                "n_trials": len(study.trials),
                "optimized_model": optimized_model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self._logger.info(
                "Model optimization completed",
                model_class=model_class.__name__,
                best_value=study.best_value,
                n_trials=len(study.trials),
            )

            return result

        except Exception as e:
            self._logger.error("Model optimization failed", error=str(e))
            raise ModelError(f"Hyperparameter optimization failed: {e}")

    def _create_objective_function(
        self,
        model_class: type,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        parameter_space: dict[str, Any],
        scoring: str,
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
                            param_name, param_config["low"], param_config["high"]
                        )
                    elif param_type == "float":
                        params[param_name] = trial.suggest_float(
                            param_name, param_config["low"], param_config["high"]
                        )
                    elif param_type == "categorical":
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_config["choices"]
                        )

                # Create model with sampled parameters
                model_factory = self.resolve_dependency("ModelFactory")
                model = model_factory.create_model(model_class.__name__, **params)

                # Evaluate model
                if X_val is not None and y_val is not None:
                    # Use validation set
                    model.train(X_train, y_train)
                    y_pred = model.predict(X_val)

                    # Calculate score
                    if scoring == "accuracy":
                        from sklearn.metrics import accuracy_score
                        score = accuracy_score(y_val, y_pred)
                    elif scoring == "r2":
                        from sklearn.metrics import r2_score
                        score = r2_score(y_val, y_pred)
                    else:
                        from sklearn.metrics import accuracy_score
                        score = accuracy_score(y_val, y_pred)
                else:
                    # Use cross-validation
                    if hasattr(model, "model") and model.model is not None:
                        scores = cross_val_score(model.model, X_train, y_train, cv=3, scoring=scoring)
                        score = np.mean(scores)
                    else:
                        # Simple train score
                        model.train(X_train, y_train)
                        y_pred = model.predict(X_train)
                        from sklearn.metrics import accuracy_score
                        score = accuracy_score(y_train, y_pred)

                return score

            except Exception as e:
                self._logger.warning(f"Trial failed: {e}")
                return 0.0

        return objective

    def _get_default_parameter_space(self, model_class: type) -> dict[str, Any]:
        """Get default parameter space for a model class."""
        model_name = model_class.__name__.lower()

        if "random" in model_name or "forest" in model_name:
            return {
                "n_estimators": {"type": "int", "low": 10, "high": 100},
                "max_depth": {"type": "int", "low": 3, "high": 10},
            }
        elif "svm" in model_name or "svc" in model_name:
            return {
                "C": {"type": "float", "low": 0.1, "high": 10.0},
                "kernel": {"type": "categorical", "choices": ["rbf", "linear"]},
            }
        else:
            return {
                "learning_rate": {"type": "float", "low": 0.001, "high": 0.1},
            }

    async def _do_start(self) -> None:
        """Start the service."""
        await super()._do_start()
        self._logger.info("Hyperparameter optimization service started")

    async def _do_stop(self) -> None:
        """Stop the service."""
        await super()._do_stop()
        self._logger.info("Hyperparameter optimization service stopped")

    async def _service_health_check(self) -> "HealthStatus":
        """Check service health."""
        from src.core.base.interfaces import HealthStatus
        return HealthStatus.HEALTHY
