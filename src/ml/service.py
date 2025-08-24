"""
ML Service - Coordinating Machine Learning Operations.

This module provides the main ML service that coordinates feature engineering,
model registry, inference services, and provides a unified interface for all
ML operations in the trading system.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from src.core.base.service import BaseService
from src.core.exceptions import ModelError, ValidationError
from src.core.types.base import ConfigDict
from src.ml.feature_engineering import FeatureRequest
from src.ml.registry.model_registry import ModelLoadRequest, ModelRegistrationRequest
from src.utils.decorators import UnifiedDecorator

# Initialize decorator instance
dec = UnifiedDecorator()


class MLServiceConfig(BaseModel):
    """Configuration for ML service."""

    enable_feature_engineering: bool = Field(default=True, description="Enable feature engineering")
    enable_model_registry: bool = Field(default=True, description="Enable model registry")
    enable_inference: bool = Field(default=True, description="Enable inference service")
    enable_feature_store: bool = Field(default=True, description="Enable feature store")
    enable_batch_processing: bool = Field(default=True, description="Enable batch processing")
    enable_pipeline_caching: bool = Field(default=True, description="Enable pipeline caching")
    max_concurrent_operations: int = Field(
        default=10, description="Maximum concurrent ML operations"
    )
    pipeline_timeout_seconds: int = Field(default=300, description="Pipeline timeout in seconds")
    cache_ttl_minutes: int = Field(default=30, description="Cache TTL in minutes")
    enable_performance_monitoring: bool = Field(
        default=True, description="Enable performance monitoring"
    )


class MLPipelineRequest(BaseModel):
    """Request for ML pipeline processing."""

    request_id: str = Field(
        default_factory=lambda: f"ml_{int(datetime.utcnow().timestamp() * 1000)}"
    )
    symbol: str
    market_data: dict[str, Any] | Any
    model_id: str | None = None
    model_name: str | None = None
    model_type: str | None = None
    feature_types: list[str] | None = None
    return_probabilities: bool = False
    use_cache: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class MLPipelineResponse(BaseModel):
    """Response from ML pipeline processing."""

    request_id: str
    symbol: str
    predictions: list[float] = Field(default_factory=list)
    probabilities: list[list[float]] | None = None
    confidence_scores: list[float] | None = None
    feature_set_id: str | None = None
    model_id: str | None = None
    processing_stages: dict[str, float] = Field(
        default_factory=dict
    )  # Stage name -> processing time
    total_processing_time_ms: float
    pipeline_success: bool
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MLTrainingRequest(BaseModel):
    """Request for ML model training."""

    request_id: str = Field(
        default_factory=lambda: f"train_{int(datetime.utcnow().timestamp() * 1000)}"
    )
    training_data: dict[str, Any] | Any
    target_data: list | Any
    model_type: str
    model_name: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    feature_types: list[str] | None = None
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    enable_feature_selection: bool = True
    stage: str = "development"
    tags: dict[str, str] = Field(default_factory=dict)
    description: str = ""


class MLTrainingResponse(BaseModel):
    """Response from ML model training."""

    request_id: str
    model_id: str | None = None
    training_metrics: dict[str, Any] = Field(default_factory=dict)
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    feature_importance: dict[str, float] = Field(default_factory=dict)
    selected_features: list[str] = Field(default_factory=list)
    training_time_ms: float
    success: bool
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class MLService(BaseService):
    """
    Main ML service coordinating all machine learning operations.

    This service provides a unified interface for all ML operations including
    feature engineering, model training, inference, and registry management.
    It orchestrates the interaction between different ML services and provides
    end-to-end ML pipelines for trading strategies.

    All operations go through proper service dependencies and avoid direct database access.
    """

    def __init__(
        self,
        config: ConfigDict | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize the ML service.

        Args:
            config: Service configuration
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="MLService",
            config=config,
            correlation_id=correlation_id,
        )

        # Parse ML service configuration
        ml_config_dict = (config or {}).get("ml_service", {})
        self.ml_config = MLServiceConfig(**ml_config_dict)

        # Service dependencies - resolved during startup
        self.data_service: Any = None
        self.feature_engineering_service: Any = None
        self.model_registry_service: Any = None
        self.inference_service: Any = None
        self.feature_store_service: Any = None

        # Internal state
        self._pipeline_cache: dict[str, tuple[MLPipelineResponse, datetime]] = {}
        self._active_operations: dict[str, asyncio.Task] = {}

        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Semaphore to limit concurrent operations
        self._operation_semaphore = asyncio.Semaphore(self.ml_config.max_concurrent_operations)

        # Add required dependencies
        self.add_dependency("DataService")
        if self.ml_config.enable_feature_engineering:
            self.add_dependency("FeatureEngineeringService")
        if self.ml_config.enable_model_registry:
            self.add_dependency("ModelRegistryService")
        if self.ml_config.enable_inference:
            self.add_dependency("InferenceService")
        if self.ml_config.enable_feature_store:
            self.add_dependency("FeatureStoreService")

    async def _do_start(self) -> None:
        """Start the ML service."""
        await super()._do_start()

        # Resolve dependencies
        self.data_service = self.resolve_dependency("DataService")

        if self.ml_config.enable_feature_engineering:
            self.feature_engineering_service = self.resolve_dependency("FeatureEngineeringService")

        if self.ml_config.enable_model_registry:
            self.model_registry_service = self.resolve_dependency("ModelRegistryService")

        if self.ml_config.enable_inference:
            self.inference_service = self.resolve_dependency("InferenceService")

        if self.ml_config.enable_feature_store:
            try:
                self.feature_store_service = self.resolve_dependency("FeatureStoreService")
            except Exception:
                # Feature store is optional
                self.feature_store_service = None

        self._logger.info(
            "ML service started successfully",
            config=self.ml_config.dict(),
            dependencies_resolved=sum(
                [
                    bool(self.data_service),
                    bool(self.feature_engineering_service),
                    bool(self.model_registry_service),
                    bool(self.inference_service),
                    bool(self.feature_store_service),
                ]
            ),
        )

    async def _do_stop(self) -> None:
        """Stop the ML service."""
        # Cancel active operations
        for _operation_id, task in self._active_operations.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Shutdown thread pool
        self._executor.shutdown(wait=True)

        await super()._do_stop()

    # Core ML Pipeline Operations
    @dec.enhance(log=True, monitor=True, log_level="info")
    async def process_pipeline(self, request: MLPipelineRequest) -> MLPipelineResponse:
        """
        Process a complete ML pipeline from data to prediction.

        Args:
            request: ML pipeline request

        Returns:
            ML pipeline response with predictions and metadata

        Raises:
            ModelError: If pipeline processing fails
        """
        return await self.execute_with_monitoring(
            "process_pipeline",
            self._process_pipeline_impl,
            request,
        )

    async def _process_pipeline_impl(self, request: MLPipelineRequest) -> MLPipelineResponse:
        """Internal ML pipeline implementation."""
        async with self._operation_semaphore:
            pipeline_start = datetime.utcnow()
            processing_stages = {}
            warnings = []

            try:
                # Check cache first
                if request.use_cache and self.ml_config.enable_pipeline_caching:
                    cache_key = self._generate_pipeline_cache_key(request)
                    cached_response = await self._get_cached_pipeline(cache_key)
                    if cached_response is not None:
                        cached_response.request_id = request.request_id  # Update request ID
                        return cached_response

                # Track the operation
                self._active_operations[request.request_id] = asyncio.current_task()

                # Convert market data to DataFrame if needed
                if isinstance(request.market_data, dict):
                    market_data_df = pd.DataFrame(request.market_data)
                else:
                    market_data_df = request.market_data

                # Stage 1: Feature Engineering
                stage_start = datetime.utcnow()
                feature_response = None

                if self.feature_engineering_service:
                    feature_request = FeatureRequest(
                        market_data=market_data_df.to_dict("records"),
                        symbol=request.symbol,
                        feature_types=request.feature_types,
                        enable_preprocessing=True,
                    )

                    feature_response = await self.feature_engineering_service.compute_features(
                        feature_request
                    )

                    if feature_response.error:
                        raise ModelError(f"Feature engineering failed: {feature_response.error}")

                    processing_stages["feature_engineering"] = (
                        datetime.utcnow() - stage_start
                    ).total_seconds() * 1000
                else:
                    warnings.append("Feature engineering service not available")

                # Stage 2: Feature Store (optional)
                if self.feature_store_service and feature_response:
                    stage_start = datetime.utcnow()
                    try:
                        await self.feature_store_service.store_features(
                            request.symbol, feature_response.feature_set
                        )
                        processing_stages["feature_store"] = (
                            datetime.utcnow() - stage_start
                        ).total_seconds() * 1000
                    except Exception as e:
                        warnings.append(f"Feature store operation failed: {e}")

                # Stage 3: Model Loading and Inference
                stage_start = datetime.utcnow()
                predictions = []
                probabilities = None
                confidence_scores = None
                model_id = None

                if self.inference_service and feature_response:
                    # Convert features to DataFrame
                    features_df = pd.DataFrame(feature_response.feature_set.features)

                    # Determine model to use
                    model_id = request.model_id
                    if not model_id and (request.model_name or request.model_type):
                        # Find model by name/type
                        load_request = ModelLoadRequest(
                            model_name=request.model_name,
                            model_type=request.model_type,
                            stage="production",  # Default to production models
                            use_cache=request.use_cache,
                        )

                        try:
                            model_info = await self.model_registry_service.load_model(load_request)
                            model_id = model_info["model_id"]
                        except Exception as e:
                            raise ModelError(f"Model loading failed: {e}")

                    if model_id:
                        # Make prediction
                        prediction_response = await self.inference_service.predict(
                            model_id=model_id,
                            features=features_df,
                            return_probabilities=request.return_probabilities,
                            use_cache=request.use_cache,
                            request_id=request.request_id,
                        )

                        if prediction_response.error:
                            raise ModelError(f"Inference failed: {prediction_response.error}")

                        predictions = prediction_response.predictions
                        probabilities = prediction_response.probabilities
                        confidence_scores = prediction_response.confidence_scores
                        model_id = prediction_response.model_info.get("model_id", model_id)
                    else:
                        warnings.append("No model specified for inference")

                    processing_stages["inference"] = (
                        datetime.utcnow() - stage_start
                    ).total_seconds() * 1000
                else:
                    warnings.append("Inference service not available")

                # Calculate total processing time
                total_processing_time = (datetime.utcnow() - pipeline_start).total_seconds() * 1000

                # Create response
                response = MLPipelineResponse(
                    request_id=request.request_id,
                    symbol=request.symbol,
                    predictions=predictions,
                    probabilities=probabilities,
                    confidence_scores=confidence_scores,
                    feature_set_id=(
                        feature_response.feature_set.feature_set_id if feature_response else None
                    ),
                    model_id=model_id,
                    processing_stages=processing_stages,
                    total_processing_time_ms=total_processing_time,
                    pipeline_success=True,
                    warnings=warnings,
                    metadata={
                        "feature_count": (
                            len(feature_response.feature_set.features) if feature_response else 0
                        ),
                        "prediction_count": len(predictions),
                    },
                )

                # Cache the response
                if request.use_cache and self.ml_config.enable_pipeline_caching:
                    await self._cache_pipeline(cache_key, response)

                self._logger.info(
                    "ML pipeline processed successfully",
                    request_id=request.request_id,
                    symbol=request.symbol,
                    model_id=model_id,
                    prediction_count=len(predictions),
                    total_time_ms=total_processing_time,
                    stages=len(processing_stages),
                )

                return response

            except Exception as e:
                total_processing_time = (datetime.utcnow() - pipeline_start).total_seconds() * 1000

                error_response = MLPipelineResponse(
                    request_id=request.request_id,
                    symbol=request.symbol,
                    processing_stages=processing_stages,
                    total_processing_time_ms=total_processing_time,
                    pipeline_success=False,
                    error=str(e),
                    warnings=warnings,
                )

                self._logger.error(
                    "ML pipeline processing failed",
                    request_id=request.request_id,
                    symbol=request.symbol,
                    error=str(e),
                    processing_stages=processing_stages,
                )

                return error_response

            finally:
                # Clean up tracking
                if request.request_id in self._active_operations:
                    del self._active_operations[request.request_id]

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def train_model(self, request: MLTrainingRequest) -> MLTrainingResponse:
        """
        Train a new ML model.

        Args:
            request: ML training request

        Returns:
            ML training response

        Raises:
            ModelError: If training fails
        """
        return await self.execute_with_monitoring(
            "train_model",
            self._train_model_impl,
            request,
        )

    async def _train_model_impl(self, request: MLTrainingRequest) -> MLTrainingResponse:
        """Internal model training implementation."""
        async with self._operation_semaphore:
            training_start = datetime.utcnow()
            warnings = []

            try:
                # Track the operation
                self._active_operations[request.request_id] = asyncio.current_task()

                # Convert data to DataFrames if needed
                if isinstance(request.training_data, dict):
                    training_data_df = pd.DataFrame(request.training_data)
                else:
                    training_data_df = request.training_data

                if isinstance(request.target_data, list):
                    target_series = pd.Series(request.target_data)
                else:
                    target_series = request.target_data

                # Validate data
                if training_data_df.empty or target_series.empty:
                    raise ValidationError("Training data and target data cannot be empty")

                if len(training_data_df) != len(target_series):
                    raise ValidationError("Training data and target data must have the same length")

                # Feature engineering
                features_df = training_data_df
                feature_importance = {}
                selected_features = list(features_df.columns)

                if self.feature_engineering_service:
                    # Generate features
                    feature_request = FeatureRequest(
                        market_data=training_data_df.to_dict("records"),
                        symbol="TRAINING",  # Placeholder for training
                        feature_types=request.feature_types,
                        enable_preprocessing=True,
                    )

                    feature_response = await self.feature_engineering_service.compute_features(
                        feature_request
                    )

                    if feature_response.error:
                        warnings.append(f"Feature engineering failed: {feature_response.error}")
                    else:
                        features_df = pd.DataFrame(feature_response.feature_set.features)

                    # Feature selection if enabled
                    if request.enable_feature_selection and len(features_df.columns) > 10:
                        try:
                            (
                                selected_features_df,
                                selected_features,
                                feature_importance,
                            ) = await self.feature_engineering_service.select_features(
                                features_df, target_series, method="mutual_info", max_features=50
                            )
                            features_df = selected_features_df
                        except Exception as e:
                            warnings.append(f"Feature selection failed: {e}")

                # Train model using thread pool to avoid blocking
                model, training_metrics, validation_metrics = await self._train_model_async(
                    features_df, target_series, request
                )

                # Register model if registry service is available
                model_id = None
                if self.model_registry_service:
                    try:
                        # Add feature names to model
                        if hasattr(model, "feature_names_"):
                            model.feature_names_ = list(features_df.columns)

                        registration_request = ModelRegistrationRequest(
                            model=model,
                            name=request.model_name,
                            model_type=request.model_type,
                            description=request.description,
                            tags=request.tags,
                            stage=request.stage,
                            metadata={
                                "hyperparameters": request.hyperparameters,
                                "training_data": {
                                    "shape": features_df.shape,
                                    "features": list(features_df.columns),
                                    "target_type": str(target_series.dtype),
                                },
                                "training_metrics": training_metrics,
                                "validation_metrics": validation_metrics,
                                "feature_importance": feature_importance,
                            },
                        )

                        model_id = await self.model_registry_service.register_model(
                            registration_request
                        )

                    except Exception as e:
                        warnings.append(f"Model registration failed: {e}")

                training_time = (datetime.utcnow() - training_start).total_seconds() * 1000

                response = MLTrainingResponse(
                    request_id=request.request_id,
                    model_id=model_id,
                    training_metrics=training_metrics,
                    validation_metrics=validation_metrics,
                    feature_importance=feature_importance,
                    selected_features=selected_features,
                    training_time_ms=training_time,
                    success=True,
                    warnings=warnings,
                )

                self._logger.info(
                    "Model training completed successfully",
                    request_id=request.request_id,
                    model_name=request.model_name,
                    model_id=model_id,
                    training_time_ms=training_time,
                    features_used=len(selected_features),
                )

                return response

            except Exception as e:
                training_time = (datetime.utcnow() - training_start).total_seconds() * 1000

                self._logger.error(
                    "Model training failed",
                    request_id=request.request_id,
                    model_name=request.model_name,
                    error=str(e),
                )

                return MLTrainingResponse(
                    request_id=request.request_id,
                    training_time_ms=training_time,
                    success=False,
                    error=str(e),
                    warnings=warnings,
                )

            finally:
                # Clean up tracking
                if request.request_id in self._active_operations:
                    del self._active_operations[request.request_id]

    async def _train_model_async(
        self, features_df: pd.DataFrame, target_series: pd.Series, request: MLTrainingRequest
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        """Train model asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._train_model_sync, features_df, target_series, request
        )

    def _train_model_sync(
        self, features_df: pd.DataFrame, target_series: pd.Series, request: MLTrainingRequest
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        """Synchronous model training."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.model_selection import cross_val_score, train_test_split

        # Determine if this is classification or regression
        is_classification = (
            target_series.dtype == "object"
            or target_series.nunique() < 10
            or request.model_type.lower().endswith("classifier")
        )

        # Select model based on type
        if request.model_type.lower() == "random_forest":
            if is_classification:
                model = RandomForestClassifier(**request.hyperparameters)
            else:
                model = RandomForestRegressor(**request.hyperparameters)
        elif request.model_type.lower() == "logistic_regression":
            model = LogisticRegression(**request.hyperparameters)
        elif request.model_type.lower() == "linear_regression":
            model = LinearRegression(**request.hyperparameters)
        else:
            raise ModelError(f"Unsupported model type: {request.model_type}")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features_df, target_series, test_size=request.validation_split, random_state=42
        )

        # Train model
        model.fit(X_train, y_train)

        # Training metrics
        train_predictions = model.predict(X_train)
        training_metrics = self._calculate_metrics(y_train, train_predictions, is_classification)

        # Validation metrics
        val_predictions = model.predict(X_val)
        validation_metrics = self._calculate_metrics(y_val, val_predictions, is_classification)

        # Cross-validation score
        if request.cross_validation_folds > 1:
            cv_scores = cross_val_score(
                model, features_df, target_series, cv=request.cross_validation_folds
            )
            validation_metrics["cv_score_mean"] = float(cv_scores.mean())
            validation_metrics["cv_score_std"] = float(cv_scores.std())

        return model, training_metrics, validation_metrics

    def _calculate_metrics(self, y_true, y_pred, is_classification: bool) -> dict[str, Any]:
        """Calculate appropriate metrics."""
        if is_classification:
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(
                    precision_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            }
        else:
            return {
                "mse": float(mean_squared_error(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2_score": float(r2_score(y_true, y_pred)),
            }

    # Batch Processing
    @dec.enhance(log=True, monitor=True, log_level="info")
    async def process_batch_pipeline(
        self, requests: list[MLPipelineRequest]
    ) -> list[MLPipelineResponse]:
        """
        Process multiple ML pipeline requests in batch.

        Args:
            requests: List of ML pipeline requests

        Returns:
            List of ML pipeline responses
        """
        return await self.execute_with_monitoring(
            "process_batch_pipeline",
            self._process_batch_pipeline_impl,
            requests,
        )

    async def _process_batch_pipeline_impl(
        self, requests: list[MLPipelineRequest]
    ) -> list[MLPipelineResponse]:
        """Internal batch pipeline processing implementation."""
        if not requests:
            return []

        batch_start = datetime.utcnow()

        try:
            # Process requests concurrently, but respect the semaphore limit
            tasks = []
            for request in requests:
                task = asyncio.create_task(self.process_pipeline(request))
                tasks.append(task)

            # Wait for all to complete
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions
            final_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    error_response = MLPipelineResponse(
                        request_id=requests[i].request_id,
                        symbol=requests[i].symbol,
                        total_processing_time_ms=0.0,
                        pipeline_success=False,
                        error=str(response),
                    )
                    final_responses.append(error_response)
                else:
                    final_responses.append(response)

            batch_processing_time = (datetime.utcnow() - batch_start).total_seconds() * 1000

            self._logger.info(
                "Batch pipeline processing completed",
                batch_size=len(requests),
                successful_pipelines=sum(1 for r in final_responses if r.pipeline_success),
                failed_pipelines=sum(1 for r in final_responses if not r.pipeline_success),
                total_batch_time_ms=batch_processing_time,
            )

            return final_responses

        except Exception as e:
            self._logger.error(
                "Batch pipeline processing failed", batch_size=len(requests), error=str(e)
            )

            # Return error responses for all requests
            error_responses = []
            for request in requests:
                error_response = MLPipelineResponse(
                    request_id=request.request_id,
                    symbol=request.symbol,
                    total_processing_time_ms=0.0,
                    pipeline_success=False,
                    error=f"Batch processing failed: {e}",
                )
                error_responses.append(error_response)

            return error_responses

    # Cache Management
    def _generate_pipeline_cache_key(self, request: MLPipelineRequest) -> str:
        """Generate cache key for pipeline request."""
        import hashlib

        # Create hash from key request parameters
        if isinstance(request.market_data, dict):
            data_hash = str(hash(frozenset(request.market_data.items())))
        else:
            data_hash = str(hash(tuple(request.market_data.iloc[0].values)))

        cache_str = f"{request.symbol}_{request.model_id or request.model_name}_{data_hash}_{request.return_probabilities}"
        return hashlib.md5(cache_str.encode()).hexdigest()[:16]

    async def _get_cached_pipeline(self, cache_key: str) -> MLPipelineResponse | None:
        """Get cached pipeline response."""
        if cache_key in self._pipeline_cache:
            response, timestamp = self._pipeline_cache[cache_key]
            ttl_minutes = self.ml_config.cache_ttl_minutes

            if datetime.utcnow() - timestamp < timedelta(minutes=ttl_minutes):
                return response
            else:
                del self._pipeline_cache[cache_key]

        return None

    async def _cache_pipeline(self, cache_key: str, response: MLPipelineResponse) -> None:
        """Cache pipeline response."""
        self._pipeline_cache[cache_key] = (response, datetime.utcnow())

        # Clean old cache entries
        await self._clean_pipeline_cache()

    async def _clean_pipeline_cache(self) -> None:
        """Clean expired pipeline cache entries."""
        ttl_minutes = self.ml_config.cache_ttl_minutes
        cutoff_time = datetime.utcnow() - timedelta(minutes=ttl_minutes)

        expired_keys = [
            key for key, (_, timestamp) in self._pipeline_cache.items() if timestamp < cutoff_time
        ]

        for key in expired_keys:
            del self._pipeline_cache[key]

        if expired_keys:
            self._logger.debug(f"Cleaned {len(expired_keys)} expired pipeline cache entries")

    # Model Management Operations
    async def list_available_models(
        self, model_type: str | None = None, stage: str | None = None
    ) -> list[dict[str, Any]]:
        """List available models."""
        if not self.model_registry_service:
            raise ModelError("Model registry service not available")

        return await self.model_registry_service.list_models(
            model_type=model_type, stage=stage, active_only=True
        )

    async def promote_model(self, model_id: str, stage: str, description: str = "") -> bool:
        """Promote a model to a different stage."""
        if not self.model_registry_service:
            raise ModelError("Model registry service not available")

        return await self.model_registry_service.promote_model(model_id, stage, description)

    async def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get detailed model information."""
        if not self.model_registry_service:
            raise ModelError("Model registry service not available")

        return await self.model_registry_service.get_model_metrics(model_id)

    # Service Health and Metrics
    async def _service_health_check(self) -> Any:
        """ML service specific health check."""
        from src.core.base.interfaces import HealthStatus

        try:
            # Check required dependencies
            if not self.data_service:
                return HealthStatus.UNHEALTHY

            # Check optional dependencies based on configuration
            unhealthy_services = 0
            total_services = 1  # data_service is always required

            if self.ml_config.enable_feature_engineering:
                total_services += 1
                if not self.feature_engineering_service:
                    unhealthy_services += 1

            if self.ml_config.enable_model_registry:
                total_services += 1
                if not self.model_registry_service:
                    unhealthy_services += 1

            if self.ml_config.enable_inference:
                total_services += 1
                if not self.inference_service:
                    unhealthy_services += 1

            # Check active operations
            if len(self._active_operations) > self.ml_config.max_concurrent_operations:
                return HealthStatus.DEGRADED

            # Determine overall health
            if unhealthy_services == 0:
                return HealthStatus.HEALTHY
            elif unhealthy_services < total_services * 0.5:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.UNHEALTHY

        except Exception as e:
            self._logger.error("ML service health check failed", error=str(e))
            return HealthStatus.UNHEALTHY

    def get_ml_service_metrics(self) -> dict[str, Any]:
        """Get ML service metrics."""
        return {
            "active_operations": len(self._active_operations),
            "cached_pipelines": len(self._pipeline_cache),
            "max_concurrent_operations": self.ml_config.max_concurrent_operations,
            "feature_engineering_enabled": self.ml_config.enable_feature_engineering,
            "model_registry_enabled": self.ml_config.enable_model_registry,
            "inference_enabled": self.ml_config.enable_inference,
            "feature_store_enabled": self.ml_config.enable_feature_store,
            "services_available": {
                "data_service": bool(self.data_service),
                "feature_engineering_service": bool(self.feature_engineering_service),
                "model_registry_service": bool(self.model_registry_service),
                "inference_service": bool(self.inference_service),
                "feature_store_service": bool(self.feature_store_service),
            },
        }

    async def clear_cache(self) -> dict[str, int]:
        """Clear ML service caches."""
        pipeline_cache_size = len(self._pipeline_cache)

        self._pipeline_cache.clear()

        self._logger.info(
            "ML service caches cleared",
            pipelines_removed=pipeline_cache_size,
        )

        return {
            "pipelines_removed": pipeline_cache_size,
        }

    # Configuration validation
    def _validate_service_config(self, config: ConfigDict) -> bool:
        """Validate ML service configuration."""
        try:
            ml_config_dict = config.get("ml_service", {})
            MLServiceConfig(**ml_config_dict)
            return True
        except Exception as e:
            self._logger.error("ML service configuration validation failed", error=str(e))
            return False
