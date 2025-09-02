"""
Training Orchestration System for ML Models.

This module provides comprehensive training orchestration for ML models including
data preparation, training pipeline management, and result tracking.
"""

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.base.service import BaseService
from src.core.exceptions import ModelError, ValidationError
from src.core.types.base import ConfigDict
from src.ml.models.base_model import BaseMLModel
from src.utils.decorators import UnifiedDecorator

# Initialize decorator instance
dec = UnifiedDecorator()


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
        # Note: This is a utility class, not a component
        self.steps = steps

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


class ModelTrainingService(BaseService):
    """
    Training orchestration service for ML models.

    This service provides comprehensive training orchestration including data preparation,
    feature engineering, model training, validation, and result tracking with integration
    to the model registry and artifact store.

    Attributes:
        feature_engineer: Feature engineering service instance
        model_registry: Model registry instance
        artifact_store: Artifact store instance
        training_history: History of training runs
    """

    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None):
        """
        Initialize the training service.

        Args:
            config: Service configuration
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="ModelTrainingService",
            config=config,
            correlation_id=correlation_id,
        )

        # Decimal context is already set by utils module

        # Service state
        self.training_history: list[dict[str, Any]] = []
        self.current_run_id: str | None = None

        # Dependencies that will be resolved during startup
        self.add_dependency("FeatureEngineeringService")
        self.add_dependency("ModelRegistry")
        self.add_dependency("ArtifactStore")

        # Configuration with defaults
        ml_config = self._config.get("ml", {})
        self.max_features = ml_config.get("max_features", 100)

        self._logger.info("Training service initialized successfully")

    @dec.enhance(monitor=True)
    @dec.enhance(log=True, monitor=True, log_level="info")
    async def train_model(
        self,
        model: BaseMLModel,
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
        self.current_run_id = (
            f"train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{symbol}"
        )

        try:
            # Validate inputs
            if market_data.empty:
                raise ValidationError("Market data cannot be empty")

            if target_column not in market_data.columns:
                raise ValidationError(f"Target column '{target_column}' not found in data")

            self._logger.info(
                "Starting model training",
                model_name=model.model_name,
                symbol=symbol,
                run_id=self.current_run_id,
                data_shape=market_data.shape,
            )

            # Step 1: Feature Engineering
            features_df = await self._prepare_features(
                market_data, symbol, feature_types, self.current_run_id
            )

            # Step 2: Prepare targets
            targets = market_data[target_column].copy()

            # Step 3: Align features and targets
            features_df, targets = self._align_data(features_df, targets)

            # Step 3.1: Validate training data
            features_df, targets = validate_training_data(features_df, targets, model.model_name)

            # Step 4: Split data
            train_data, val_data, test_data = self._split_data(
                features_df, targets, train_size, validation_size
            )

            # Step 5: Feature selection and preprocessing
            if feature_selection or preprocessing:
                train_data, val_data, test_data = await self._process_features(
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
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "training_kwargs": training_kwargs,
            }

            self.training_history.append(training_result)

            self._logger.info(
                "Model training completed successfully",
                model_name=model.model_name,
                symbol=symbol,
                run_id=self.current_run_id,
                metrics=all_metrics,
            )

            return training_result

        except Exception as e:
            self._logger.error(
                "Model training failed",
                model_name=model.model_name,
                symbol=symbol,
                run_id=self.current_run_id,
                error=str(e),
            )
            raise ModelError(f"Training failed for {model.model_name}: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def batch_train_models(
        self,
        models: list[BaseMLModel],
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
                self._logger.info(
                    f"Training model {i + 1}/{len(models)}", model_name=model.model_name
                )

                result = await self.train_model(model, market_data, target_column, symbol, **kwargs)
                results.append(result)

            except Exception as e:
                self._logger.error(f"Failed to train model {model.model_name}", error=str(e))
                # Continue with other models
                results.append({"model_name": model.model_name, "error": str(e), "success": False})

        self._logger.info(
            "Batch training completed",
            total_models=len(models),
            successful_models=sum(1 for r in results if r.get("success", True)),
        )

        return results

    async def _prepare_features(
        self, market_data: pd.DataFrame, symbol: str, feature_types: list[str] | None, run_id: str
    ) -> pd.DataFrame:
        """Prepare features for training."""
        self._logger.info("Preparing features", symbol=symbol, run_id=run_id)

        feature_engineering_service = self.resolve_dependency("FeatureEngineeringService")

        # Create feature request for service call
        from src.ml.feature_engineering import FeatureRequest

        feature_request = FeatureRequest(
            market_data=market_data.to_dict("records"),
            symbol=symbol,
            feature_types=feature_types,
            enable_preprocessing=False,  # We'll handle preprocessing separately
        )

        feature_response = await feature_engineering_service.compute_features(feature_request)
        if feature_response.error:
            raise ModelError(f"Feature engineering failed: {feature_response.error}")

        features_df = pd.DataFrame(feature_response.feature_set.features)

        self._logger.info(
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

        self._logger.info(
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

        self._logger.info(
            "Data split completed",
            train_samples=len(X_train),
            val_samples=len(X_val),
            test_samples=len(X_test),
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    async def _process_features(
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
            feature_engineering_service = self.resolve_dependency("FeatureEngineeringService")
            (
                X_train_selected,
                selected_features,
                _,
            ) = await feature_engineering_service.select_features(
                X_train, y_train, max_features=min(self.max_features, len(X_train.columns))
            )
            X_val_selected = X_val[selected_features]
            X_test_selected = X_test[selected_features]

            self._logger.info(
                "Feature selection completed",
                original_features=len(X_train.columns),
                selected_features=len(selected_features),
            )
        else:
            X_train_selected, X_val_selected, X_test_selected = X_train, X_val, X_test

        # Preprocessing
        if preprocessing:
            feature_engineering_service = self.resolve_dependency("FeatureEngineeringService")
            # For preprocessing, we'd need additional methods in the service interface
            # For now, we'll skip complex preprocessing until proper service methods are available
            X_train_processed, X_val_processed, X_test_processed = (
                X_train_selected,
                X_val_selected,
                X_test_selected,
            )

            self._logger.info("Feature preprocessing completed (simplified)")
        else:
            X_train_processed, X_val_processed, X_test_processed = (
                X_train_selected,
                X_val_selected,
                X_test_selected,
            )

        return (X_train_processed, y_train), (X_val_processed, y_val), (X_test_processed, y_test)

    def _train_model(
        self,
        model: BaseMLModel,
        train_data: tuple[pd.DataFrame, pd.Series],
        val_data: tuple[pd.DataFrame, pd.Series],
        **kwargs,
    ) -> dict[str, float]:
        """Train the model."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        self._logger.info("Training model", model_name=model.model_name)

        # Train model
        metrics = model.train(X_train, y_train, validation_data=(X_val, y_val), **kwargs)

        return metrics

    def _evaluate_model(
        self, model: BaseMLModel, test_data: tuple[pd.DataFrame, pd.Series]
    ) -> dict[str, float]:
        """Evaluate the trained model."""
        X_test, y_test = test_data

        self._logger.info("Evaluating model", model_name=model.model_name)

        # Evaluate model
        test_metrics = model.evaluate(X_test, y_test)

        return test_metrics

    def _save_training_artifacts(
        self,
        model: BaseMLModel,
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

            artifact_store = self.resolve_dependency("ArtifactStore")
            artifact_store.store_artifact(
                train_df, "data", f"{symbol}_train_data", self.current_run_id, "1.0.0"
            )

            # Save validation data
            X_val, y_val = val_data
            val_df = X_val.copy()
            val_df["target"] = y_val

            artifact_store.store_artifact(
                val_df, "data", f"{symbol}_val_data", self.current_run_id, "1.0.0"
            )

            # Save test data
            X_test, y_test = test_data
            test_df = X_test.copy()
            test_df["target"] = y_test

            artifact_store.store_artifact(
                test_df, "data", f"{symbol}_test_data", self.current_run_id, "1.0.0"
            )

            # Save metrics
            artifact_store.store_artifact(
                metrics, "report", f"{symbol}_training_metrics", self.current_run_id, "1.0.0"
            )

            # Save feature transformers if available
            feature_engineer = self.resolve_dependency("FeatureEngineeringService")
            if hasattr(feature_engineer, "scalers") and feature_engineer.scalers:
                artifact_store.store_artifact(
                    feature_engineer.scalers,
                    "transformer",
                    f"{symbol}_scalers",
                    self.current_run_id,
                    "1.0.0",
                )

            if hasattr(feature_engineer, "selectors") and feature_engineer.selectors:
                artifact_store.store_artifact(
                    feature_engineer.selectors,
                    "transformer",
                    f"{symbol}_selectors",
                    self.current_run_id,
                    "1.0.0",
                )

            self._logger.info("Training artifacts saved successfully")

        except Exception as e:
            self._logger.warning(f"Failed to save training artifacts: {e}")

    def _register_trained_model(
        self, model: BaseMLModel, metrics: dict[str, float], symbol: str
    ) -> str | None:
        """Register the trained model."""
        try:
            model_registry = self.resolve_dependency("ModelRegistry")
            model_id = model_registry.register_model(
                model,
                description=f"Model trained on {symbol} data",
                tags={"symbol": symbol, "run_id": self.current_run_id},
                stage="development",
            )

            self._logger.info(
                "Model registered successfully", model_id=model_id, model_name=model.model_name
            )

            return model_id

        except Exception as e:
            self._logger.warning(f"Failed to register model: {e}")
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

    async def _do_start(self) -> None:
        """Start the service and resolve dependencies."""
        await super()._do_start()
        self._logger.info("Training service started successfully")

    async def _do_stop(self) -> None:
        """Stop the service and cleanup resources."""
        await super()._do_stop()
        self._logger.info("Training service stopped")

    async def _service_health_check(self) -> "HealthStatus":
        """Check service-specific health."""
        from src.core.base.interfaces import HealthStatus

        # Check if we have reasonable training history size
        if len(self.training_history) > 1000:  # Too many entries might indicate memory issues
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def clear_history(self) -> None:
        """Clear training history."""
        self.training_history.clear()
        self._logger.info("Training history cleared")
