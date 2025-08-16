"""
Training Orchestration System for ML Models.

This module provides comprehensive training orchestration for ML models including
data preparation, training pipeline management, and result tracking.
"""

from datetime import datetime
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.config import Config
from src.core.exceptions import ModelError, ValidationError
from src.core.logging import get_logger
from src.ml.feature_engineering import FeatureEngineer
from src.ml.models.base_model import BaseModel
from src.ml.registry.artifact_store import ArtifactStore
from src.ml.registry.model_registry import ModelRegistry
from src.utils.decorators import log_calls, memory_usage, time_execution

logger = get_logger(__name__)


class TrainingPipeline:
    """
    Training pipeline for managing data preparation and model training flow.

    Attributes:
        steps: List of pipeline steps
        fitted: Whether the pipeline has been fitted
    """

    def __init__(self, steps: list[tuple[str, Any]]):
        """
        Initialize training pipeline.

        Args:
            steps: List of (name, transformer) tuples
        """
        self.steps = steps
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TrainingPipeline":
        """Fit all pipeline steps."""
        X_current = X.copy()

        for _name, transformer in self.steps:
            if hasattr(transformer, "fit"):
                transformer.fit(X_current, y)
            if hasattr(transformer, "transform"):
                X_current = transformer.transform(X_current)

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data through all pipeline steps."""
        if not self.fitted:
            raise ModelError("Pipeline must be fitted before transform")

        X_current = X.copy()

        for _name, transformer in self.steps:
            if hasattr(transformer, "transform"):
                X_current = transformer.transform(X_current)

        return X_current

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit pipeline and transform data."""
        return self.fit(X, y).transform(X)


class Trainer:
    """
    Training orchestration system for ML models.

    This class provides comprehensive training orchestration including data preparation,
    feature engineering, model training, validation, and result tracking with integration
    to the model registry and artifact store.

    Attributes:
        config: Application configuration
        feature_engineer: Feature engineering instance
        model_registry: Model registry instance
        artifact_store: Artifact store instance
        training_history: History of training runs
    """

    def __init__(self, config: Config):
        """
        Initialize the trainer.

        Args:
            config: Application configuration
        """
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.model_registry = ModelRegistry(config)
        self.artifact_store = ArtifactStore(config)

        # Training state
        self.training_history: list[dict[str, Any]] = []
        self.current_run_id: str | None = None

        logger.info("Trainer initialized successfully")

    @time_execution
    @memory_usage
    @log_calls
    def train_model(
        self,
        model: BaseModel,
        market_data: pd.DataFrame,
        target_column: str,
        symbol: str,
        train_size: float = 0.8,
        validation_size: float = 0.2,
        feature_types: list[str] | None = None,
        feature_selection: bool = True,
        preprocessing: bool = True,
        save_artifacts: bool = True,
        register_model: bool = True,
        **training_kwargs,
    ) -> dict[str, Any]:
        """
        Train a model with comprehensive orchestration.

        Args:
            model: Model instance to train
            market_data: Market data for training
            target_column: Name of target column
            symbol: Trading symbol
            train_size: Proportion of data for training
            validation_size: Proportion of training data for validation
            feature_types: Types of features to create
            feature_selection: Whether to perform feature selection
            preprocessing: Whether to preprocess features
            save_artifacts: Whether to save training artifacts
            register_model: Whether to register the trained model
            **training_kwargs: Additional training parameters

        Returns:
            Training results dictionary

        Raises:
            ModelError: If training fails
            ValidationError: If input validation fails
        """
        # Generate unique run ID
        self.current_run_id = f"train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{symbol}"

        try:
            # Validate inputs
            if market_data.empty:
                raise ValidationError("Market data cannot be empty")

            if target_column not in market_data.columns:
                raise ValidationError(f"Target column '{target_column}' not found in data")

            logger.info(
                "Starting model training",
                model_name=model.model_name,
                symbol=symbol,
                run_id=self.current_run_id,
                data_shape=market_data.shape,
            )

            # Step 1: Feature Engineering
            features_df = self._prepare_features(
                market_data, symbol, feature_types, self.current_run_id
            )

            # Step 2: Prepare targets
            targets = market_data[target_column].copy()

            # Step 3: Align features and targets
            features_df, targets = self._align_data(features_df, targets)

            # Step 4: Split data
            train_data, val_data, test_data = self._split_data(
                features_df, targets, train_size, validation_size
            )

            # Step 5: Feature selection and preprocessing
            if feature_selection or preprocessing:
                train_data, val_data, test_data = self._process_features(
                    train_data,
                    val_data,
                    test_data,
                    feature_selection,
                    preprocessing,
                    self.current_run_id,
                )

            # Step 6: Train model
            training_metrics = self._train_model(model, train_data, val_data, **training_kwargs)

            # Step 7: Evaluate model
            test_metrics = self._evaluate_model(model, test_data)

            # Step 8: Combine results
            all_metrics = {**training_metrics, **test_metrics}

            # Step 9: Save artifacts
            if save_artifacts:
                self._save_training_artifacts(
                    model, train_data, val_data, test_data, all_metrics, symbol
                )

            # Step 10: Register model
            model_id = None
            if register_model:
                model_id = self._register_trained_model(model, all_metrics, symbol)

            # Step 11: Update training history
            training_result = {
                "run_id": self.current_run_id,
                "model_name": model.model_name,
                "model_type": model.model_type,
                "symbol": symbol,
                "model_id": model_id,
                "metrics": all_metrics,
                "data_shape": market_data.shape,
                "feature_count": len(train_data[0].columns),
                "train_size": train_size,
                "validation_size": validation_size,
                "feature_selection": feature_selection,
                "preprocessing": preprocessing,
                "timestamp": datetime.utcnow().isoformat(),
                "training_kwargs": training_kwargs,
            }

            self.training_history.append(training_result)

            logger.info(
                "Model training completed successfully",
                model_name=model.model_name,
                symbol=symbol,
                run_id=self.current_run_id,
                metrics=all_metrics,
            )

            return training_result

        except Exception as e:
            logger.error(
                "Model training failed",
                model_name=model.model_name,
                symbol=symbol,
                run_id=self.current_run_id,
                error=str(e),
            )
            raise ModelError(f"Training failed for {model.model_name}: {e}") from e

    @time_execution
    @log_calls
    def batch_train_models(
        self,
        models: list[BaseModel],
        market_data: pd.DataFrame,
        target_column: str,
        symbol: str,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Train multiple models in batch.

        Args:
            models: List of models to train
            market_data: Market data for training
            target_column: Name of target column
            symbol: Trading symbol
            **kwargs: Additional training parameters

        Returns:
            List of training results
        """
        results = []

        for i, model in enumerate(models):
            try:
                logger.info(f"Training model {i+1}/{len(models)}", model_name=model.model_name)

                result = self.train_model(model, market_data, target_column, symbol, **kwargs)
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to train model {model.model_name}", error=str(e))
                # Continue with other models
                results.append({"model_name": model.model_name, "error": str(e), "success": False})

        logger.info(
            "Batch training completed",
            total_models=len(models),
            successful_models=sum(1 for r in results if r.get("success", True)),
        )

        return results

    def _prepare_features(
        self, market_data: pd.DataFrame, symbol: str, feature_types: list[str] | None, run_id: str
    ) -> pd.DataFrame:
        """Prepare features for training."""
        logger.info("Preparing features", symbol=symbol, run_id=run_id)

        features_df = self.feature_engineer.create_features(market_data, symbol, feature_types)

        logger.info(
            "Features prepared",
            feature_count=len(features_df.columns),
            data_points=len(features_df),
        )

        return features_df

    def _align_data(
        self, features_df: pd.DataFrame, targets: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Align features and targets by index."""
        # Find common index
        common_index = features_df.index.intersection(targets.index)

        if len(common_index) == 0:
            raise ValidationError("No common index between features and targets")

        # Align data
        features_aligned = features_df.loc[common_index]
        targets_aligned = targets.loc[common_index]

        # Remove any remaining NaN values
        valid_mask = ~(features_aligned.isna().any(axis=1) | targets_aligned.isna())
        features_aligned = features_aligned[valid_mask]
        targets_aligned = targets_aligned[valid_mask]

        logger.info(
            "Data aligned",
            aligned_samples=len(features_aligned),
            original_features=len(features_df),
            original_targets=len(targets),
        )

        return features_aligned, targets_aligned

    def _split_data(
        self,
        features_df: pd.DataFrame,
        targets: pd.Series,
        train_size: float,
        validation_size: float,
    ) -> tuple[
        tuple[pd.DataFrame, pd.Series],
        tuple[pd.DataFrame, pd.Series],
        tuple[pd.DataFrame, pd.Series],
    ]:
        """Split data into train, validation, and test sets."""
        # First split: train vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features_df, targets, train_size=train_size, random_state=42, stratify=None
        )

        # Second split: train vs validation
        val_size_adjusted = validation_size / train_size
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42, stratify=None
        )

        logger.info(
            "Data split completed",
            train_samples=len(X_train),
            val_samples=len(X_val),
            test_samples=len(X_test),
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def _process_features(
        self,
        train_data: tuple[pd.DataFrame, pd.Series],
        val_data: tuple[pd.DataFrame, pd.Series],
        test_data: tuple[pd.DataFrame, pd.Series],
        feature_selection: bool,
        preprocessing: bool,
        run_id: str,
    ) -> tuple[
        tuple[pd.DataFrame, pd.Series],
        tuple[pd.DataFrame, pd.Series],
        tuple[pd.DataFrame, pd.Series],
    ]:
        """Process features with selection and preprocessing."""
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        # Feature selection
        if feature_selection:
            X_train_selected, selected_features = self.feature_engineer.select_features(
                X_train, y_train, k_features=min(self.config.ml.max_features, len(X_train.columns))
            )
            X_val_selected = X_val[selected_features]
            X_test_selected = X_test[selected_features]

            logger.info(
                "Feature selection completed",
                original_features=len(X_train.columns),
                selected_features=len(selected_features),
            )
        else:
            X_train_selected, X_val_selected, X_test_selected = X_train, X_val, X_test

        # Preprocessing
        if preprocessing:
            X_train_processed = self.feature_engineer.preprocess_features(
                X_train_selected, fit_scalers=True
            )
            X_val_processed = self.feature_engineer.preprocess_features(
                X_val_selected, fit_scalers=False
            )
            X_test_processed = self.feature_engineer.preprocess_features(
                X_test_selected, fit_scalers=False
            )

            logger.info("Feature preprocessing completed")
        else:
            X_train_processed, X_val_processed, X_test_processed = (
                X_train_selected,
                X_val_selected,
                X_test_selected,
            )

        return (X_train_processed, y_train), (X_val_processed, y_val), (X_test_processed, y_test)

    def _train_model(
        self,
        model: BaseModel,
        train_data: tuple[pd.DataFrame, pd.Series],
        val_data: tuple[pd.DataFrame, pd.Series],
        **kwargs,
    ) -> dict[str, float]:
        """Train the model."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        logger.info("Training model", model_name=model.model_name)

        # Train model
        metrics = model.train(X_train, y_train, validation_data=(X_val, y_val), **kwargs)

        return metrics

    def _evaluate_model(
        self, model: BaseModel, test_data: tuple[pd.DataFrame, pd.Series]
    ) -> dict[str, float]:
        """Evaluate the trained model."""
        X_test, y_test = test_data

        logger.info("Evaluating model", model_name=model.model_name)

        # Evaluate model
        test_metrics = model.evaluate(X_test, y_test)

        return test_metrics

    def _save_training_artifacts(
        self,
        model: BaseModel,
        train_data: tuple[pd.DataFrame, pd.Series],
        val_data: tuple[pd.DataFrame, pd.Series],
        test_data: tuple[pd.DataFrame, pd.Series],
        metrics: dict[str, float],
        symbol: str,
    ) -> None:
        """Save training artifacts."""
        try:
            # Save training data
            X_train, y_train = train_data
            train_df = X_train.copy()
            train_df["target"] = y_train

            self.artifact_store.store_artifact(
                train_df, "data", f"{symbol}_train_data", self.current_run_id, "1.0.0"
            )

            # Save validation data
            X_val, y_val = val_data
            val_df = X_val.copy()
            val_df["target"] = y_val

            self.artifact_store.store_artifact(
                val_df, "data", f"{symbol}_val_data", self.current_run_id, "1.0.0"
            )

            # Save test data
            X_test, y_test = test_data
            test_df = X_test.copy()
            test_df["target"] = y_test

            self.artifact_store.store_artifact(
                test_df, "data", f"{symbol}_test_data", self.current_run_id, "1.0.0"
            )

            # Save metrics
            self.artifact_store.store_artifact(
                metrics, "report", f"{symbol}_training_metrics", self.current_run_id, "1.0.0"
            )

            # Save feature transformers if available
            if hasattr(self.feature_engineer, "scalers") and self.feature_engineer.scalers:
                self.artifact_store.store_artifact(
                    self.feature_engineer.scalers,
                    "transformer",
                    f"{symbol}_scalers",
                    self.current_run_id,
                    "1.0.0",
                )

            if hasattr(self.feature_engineer, "selectors") and self.feature_engineer.selectors:
                self.artifact_store.store_artifact(
                    self.feature_engineer.selectors,
                    "transformer",
                    f"{symbol}_selectors",
                    self.current_run_id,
                    "1.0.0",
                )

            logger.info("Training artifacts saved successfully")

        except Exception as e:
            logger.warning(f"Failed to save training artifacts: {e}")

    def _register_trained_model(
        self, model: BaseModel, metrics: dict[str, float], symbol: str
    ) -> str | None:
        """Register the trained model."""
        try:
            model_id = self.model_registry.register_model(
                model,
                description=f"Model trained on {symbol} data",
                tags={"symbol": symbol, "run_id": self.current_run_id},
                stage="development",
            )

            logger.info(
                "Model registered successfully", model_id=model_id, model_name=model.model_name
            )

            return model_id

        except Exception as e:
            logger.warning(f"Failed to register model: {e}")
            return None

    def get_training_history(self) -> list[dict[str, Any]]:
        """Get training history."""
        return self.training_history.copy()

    def get_best_model_by_metric(
        self, metric_name: str, model_type: str | None = None, higher_is_better: bool = True
    ) -> dict[str, Any] | None:
        """
        Get the best model from training history by a specific metric.

        Args:
            metric_name: Name of the metric to optimize
            model_type: Optional model type filter
            higher_is_better: Whether higher values are better

        Returns:
            Best model training result or None
        """
        if not self.training_history:
            return None

        # Filter by model type if specified
        candidates = self.training_history
        if model_type:
            candidates = [h for h in candidates if h.get("model_type") == model_type]

        if not candidates:
            return None

        # Find best model
        best_result = None
        best_score = None

        for result in candidates:
            metrics = result.get("metrics", {})
            if metric_name not in metrics:
                continue

            score = metrics[metric_name]

            if best_score is None:
                best_score = score
                best_result = result
            elif (higher_is_better and score > best_score) or (
                not higher_is_better and score < best_score
            ):
                best_score = score
                best_result = result

        return best_result

    def clear_history(self) -> None:
        """Clear training history."""
        self.training_history.clear()
        logger.info("Training history cleared")
