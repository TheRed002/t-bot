"""
Real-time Inference Service for ML Models.

This module provides high-performance real-time inference capabilities for trained
ML models with caching, batch processing, and performance monitoring using proper
service patterns without direct database access.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.core.base.service import BaseService
from src.core.exceptions import ModelError, ValidationError
from src.core.types.base import ConfigDict
from src.utils.decorators import UnifiedDecorator

# Initialize decorator instance
dec = UnifiedDecorator()


class InferenceConfig(BaseModel):
    """Configuration for inference service."""

    max_cpu_cores: int = Field(default=4, description="Maximum CPU cores for inference")
    inference_batch_size: int = Field(default=32, description="Inference batch size")
    max_queue_size: int = Field(default=1000, description="Maximum queue size for batch processing")
    model_cache_ttl_hours: int = Field(default=24, description="Model cache TTL in hours")
    enable_model_warmup: bool = Field(default=True, description="Enable model warmup")
    enable_batch_processing: bool = Field(default=True, description="Enable batch processing")
    batch_timeout_ms: int = Field(
        default=100, description="Batch processing timeout in milliseconds"
    )
    enable_performance_monitoring: bool = Field(
        default=True, description="Enable performance monitoring"
    )


class InferencePredictionRequest(BaseModel):
    """Request object for predictions."""

    request_id: str
    model_id: str
    features: dict[str, Any]
    return_probabilities: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class InferencePredictionResponse(BaseModel):
    """Response object for predictions."""

    request_id: str
    predictions: list[float]
    probabilities: list[list[float]] | None = None
    confidence_scores: list[float] | None = None
    processing_time_ms: float
    model_info: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class InferenceMetrics(BaseModel):
    """Inference service performance metrics."""

    total_requests: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    batch_requests_processed: int = 0
    models_warmed_up: int = 0


class InferenceService(BaseService):
    """
    Real-time inference service for ML models.

    This service provides high-performance real-time inference capabilities including
    model caching, batch processing, async processing, and performance monitoring
    using proper service patterns without direct database access.

    All data and model access goes through service dependencies.
    """

    def __init__(
        self,
        config: ConfigDict | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize the inference service.

        Args:
            config: Service configuration
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="InferenceService",
            config=config,
            correlation_id=correlation_id,
        )

        # Parse inference configuration
        inference_config_dict = (config or {}).get("inference", {})
        self.inference_config = InferenceConfig(**inference_config_dict)

        # Service dependencies - resolved during startup
        self.model_registry_service: Any = None
        self.feature_engineering_service: Any = None

        # Internal state
        self._model_cache: dict[str, tuple[Any, datetime]] = {}
        self._prediction_cache: dict[str, tuple[InferencePredictionResponse, datetime]] = {}

        # Performance metrics
        self.metrics = InferenceMetrics()

        # Async processing components
        self._executor = ThreadPoolExecutor(max_workers=self.inference_config.max_cpu_cores)
        self._batch_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.inference_config.max_queue_size
        )
        self._batch_processor_task: asyncio.Task | None = None
        self._batch_results: dict[str, asyncio.Future] = {}

        # Add required dependencies
        self.add_dependency("ModelRegistryService")
        self.add_dependency("FeatureEngineeringService")

    async def _do_start(self) -> None:
        """Start the inference service."""
        await super()._do_start()

        # Resolve dependencies
        self.model_registry_service = self.resolve_dependency("ModelRegistryService")
        self.feature_engineering_service = self.resolve_dependency("FeatureEngineeringService")

        # Start batch processor if enabled
        if self.inference_config.enable_batch_processing:
            self._batch_processor_task = asyncio.create_task(self._batch_processor_loop())

        self._logger.info(
            "Inference service started successfully",
            config=self.inference_config.dict(),
            batch_processing_enabled=self.inference_config.enable_batch_processing,
        )

    async def _do_stop(self) -> None:
        """Stop the inference service."""
        # Cancel batch processor
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass

        # Shutdown thread pool
        self._executor.shutdown(wait=True)

        await super()._do_stop()

    # Core Inference Operations
    @dec.enhance(log=True, monitor=True, log_level="info")
    async def predict(
        self,
        model_id: str,
        features: pd.DataFrame,
        return_probabilities: bool = False,
        use_cache: bool = True,
        request_id: str | None = None,
    ) -> InferencePredictionResponse:
        """
        Make a single prediction.

        Args:
            model_id: Model ID to use for prediction
            features: Feature data
            return_probabilities: Whether to return probabilities
            use_cache: Whether to use model cache
            request_id: Optional request ID

        Returns:
            Prediction response
        """
        return await self.execute_with_monitoring(
            "predict",
            self._predict_impl,
            model_id,
            features,
            return_probabilities,
            use_cache,
            request_id,
        )

    async def _predict_impl(
        self,
        model_id: str,
        features: pd.DataFrame,
        return_probabilities: bool,
        use_cache: bool,
        request_id: str | None,
    ) -> InferencePredictionResponse:
        """Internal prediction implementation."""
        start_time = time.time()

        if request_id is None:
            request_id = f"pred_{int(time.time() * 1000)}"

        try:
            self.metrics.total_requests += 1

            # Validate features
            if features.empty:
                raise ValidationError("Features cannot be empty")

            # Check prediction cache first
            if use_cache:
                cache_key = self._generate_prediction_cache_key(
                    model_id, features, return_probabilities
                )
                cached_response = await self._get_cached_prediction(cache_key)
                if cached_response is not None:
                    self.metrics.cache_hits += 1
                    cached_response.request_id = request_id  # Update request ID
                    return cached_response

            self.metrics.cache_misses += 1

            # Get model
            model = await self._get_model(model_id, use_cache)

            # Make prediction
            predictions, probabilities, confidence_scores = await self._make_prediction(
                model, features, return_probabilities
            )

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Create response
            response = InferencePredictionResponse(
                request_id=request_id,
                predictions=(
                    predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
                ),
                probabilities=probabilities.tolist() if probabilities is not None else None,
                confidence_scores=(
                    confidence_scores.tolist() if confidence_scores is not None else None
                ),
                processing_time_ms=processing_time_ms,
                model_info={
                    "model_id": model_id,
                    "model_name": getattr(model, "model_name", "unknown"),
                    "model_type": getattr(model, "model_type", "unknown"),
                    "version": getattr(model, "version", "unknown"),
                },
            )

            # Cache the response
            if use_cache:
                await self._cache_prediction(cache_key, response)

            # Update metrics
            self.metrics.successful_predictions += 1
            self.metrics.total_processing_time += processing_time_ms
            self.metrics.average_processing_time = (
                self.metrics.total_processing_time / self.metrics.total_requests
            )

            self._logger.debug(
                "Prediction completed",
                request_id=request_id,
                model_id=model_id,
                processing_time_ms=processing_time_ms,
                features_shape=features.shape,
            )

            return response

        except Exception as e:
            self.metrics.failed_predictions += 1
            processing_time_ms = (time.time() - start_time) * 1000

            error_msg = f"Prediction failed: {e}"

            self._logger.error(
                "Prediction failed",
                request_id=request_id,
                model_id=model_id,
                error=str(e),
            )

            return InferencePredictionResponse(
                request_id=request_id,
                predictions=[],
                processing_time_ms=processing_time_ms,
                error=error_msg,
            )

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def predict_async(
        self,
        model_id: str,
        features: pd.DataFrame,
        return_probabilities: bool = False,
        use_cache: bool = True,
        request_id: str | None = None,
    ) -> InferencePredictionResponse:
        """
        Make an async prediction (same as predict, but explicitly async).

        Args:
            model_id: Model ID to use for prediction
            features: Feature data
            return_probabilities: Whether to return probabilities
            use_cache: Whether to use model cache
            request_id: Optional request ID

        Returns:
            Prediction response
        """
        return await self.predict(model_id, features, return_probabilities, use_cache, request_id)

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def predict_batch(
        self, requests: list[InferencePredictionRequest]
    ) -> list[InferencePredictionResponse]:
        """
        Process a batch of prediction requests.

        Args:
            requests: List of prediction requests

        Returns:
            List of prediction responses
        """
        return await self.execute_with_monitoring(
            "predict_batch",
            self._predict_batch_impl,
            requests,
        )

    async def _predict_batch_impl(
        self, requests: list[InferencePredictionRequest]
    ) -> list[InferencePredictionResponse]:
        """Internal batch prediction implementation."""
        if not requests:
            return []

        start_time = time.time()
        responses = []

        try:
            # Group requests by model_id for efficiency
            requests_by_model = {}
            for request in requests:
                if request.model_id not in requests_by_model:
                    requests_by_model[request.model_id] = []
                requests_by_model[request.model_id].append(request)

            # Process each model group
            for model_id, model_requests in requests_by_model.items():
                try:
                    # Get model once for all requests
                    model = await self._get_model(model_id, use_cache=True)

                    # Process requests for this model concurrently
                    model_tasks = []
                    for request in model_requests:
                        task = asyncio.create_task(
                            self._process_single_batch_request(model, request)
                        )
                        model_tasks.append(task)

                    # Wait for all requests for this model
                    model_responses = await asyncio.gather(*model_tasks, return_exceptions=True)

                    # Collect responses
                    for response in model_responses:
                        if isinstance(response, Exception):
                            # Create error response
                            error_response = InferencePredictionResponse(
                                request_id="batch_error",
                                predictions=[],
                                error=str(response),
                            )
                            responses.append(error_response)
                            self.metrics.failed_predictions += 1
                        else:
                            responses.append(response)
                            self.metrics.successful_predictions += 1

                except Exception as e:
                    # Model loading failed - create error responses for all requests
                    for request in model_requests:
                        error_response = InferencePredictionResponse(
                            request_id=request.request_id,
                            predictions=[],
                            error=f"Model loading failed: {e}",
                        )
                        responses.append(error_response)
                        self.metrics.failed_predictions += 1

            batch_processing_time = (time.time() - start_time) * 1000
            self.metrics.batch_requests_processed += len(requests)

            self._logger.info(
                "Batch prediction completed",
                batch_size=len(requests),
                successful_predictions=sum(1 for r in responses if not r.error),
                failed_predictions=sum(1 for r in responses if r.error),
                batch_processing_time_ms=batch_processing_time,
            )

            return responses

        except Exception as e:
            self._logger.error("Batch prediction failed", batch_size=len(requests), error=str(e))

            # Return error responses for all requests
            error_responses = []
            for request in requests:
                error_response = InferencePredictionResponse(
                    request_id=request.request_id,
                    predictions=[],
                    error=f"Batch prediction failed: {e}",
                )
                error_responses.append(error_response)

            return error_responses

    async def _process_single_batch_request(
        self, model: Any, request: InferencePredictionRequest
    ) -> InferencePredictionResponse:
        """Process a single request within a batch."""
        request_start = time.time()

        try:
            # Convert features to DataFrame
            features_df = pd.DataFrame([request.features])

            # Make prediction
            predictions, probabilities, confidence_scores = await self._make_prediction(
                model, features_df, request.return_probabilities
            )

            processing_time_ms = (time.time() - request_start) * 1000

            return InferencePredictionResponse(
                request_id=request.request_id,
                predictions=(
                    predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
                ),
                probabilities=probabilities.tolist() if probabilities is not None else None,
                confidence_scores=(
                    confidence_scores.tolist() if confidence_scores is not None else None
                ),
                processing_time_ms=processing_time_ms,
                model_info={
                    "model_id": request.model_id,
                    "batch_processed": True,
                },
            )

        except Exception as e:
            processing_time_ms = (time.time() - request_start) * 1000
            return InferencePredictionResponse(
                request_id=request.request_id,
                predictions=[],
                processing_time_ms=processing_time_ms,
                error=str(e),
            )

    # Prediction with automatic feature engineering
    @dec.enhance(cache=True, cache_ttl=300)  # 5-minute cache
    async def predict_with_features(
        self,
        model_id: str,
        market_data: pd.DataFrame,
        symbol: str,
        return_probabilities: bool = False,
    ) -> InferencePredictionResponse:
        """
        Make prediction with automatic feature engineering.

        Args:
            model_id: Model ID to use
            market_data: Raw market data
            symbol: Trading symbol
            return_probabilities: Whether to return probabilities

        Returns:
            Prediction response
        """
        return await self.execute_with_monitoring(
            "predict_with_features",
            self._predict_with_features_impl,
            model_id,
            market_data,
            symbol,
            return_probabilities,
        )

    async def _predict_with_features_impl(
        self,
        model_id: str,
        market_data: pd.DataFrame,
        symbol: str,
        return_probabilities: bool,
    ) -> InferencePredictionResponse:
        """Internal predict with features implementation."""
        try:
            # Import the feature request model
            from src.ml.feature_engineering import FeatureRequest

            # Create features using feature engineering service
            feature_request = FeatureRequest(
                market_data=market_data.to_dict("records"),
                symbol=symbol,
                enable_preprocessing=True,
            )

            feature_response = await self.feature_engineering_service.compute_features(
                feature_request
            )

            if feature_response.error:
                return InferencePredictionResponse(
                    request_id=f"feat_pred_{int(time.time() * 1000)}",
                    predictions=[],
                    error=f"Feature engineering failed: {feature_response.error}",
                )

            # Convert features to DataFrame
            features_df = pd.DataFrame(feature_response.feature_set.features)

            # Make prediction
            prediction_response = await self.predict(model_id, features_df, return_probabilities)

            return prediction_response

        except Exception as e:
            error_msg = f"Prediction with features failed: {e}"

            self._logger.error(
                "Prediction with features failed",
                model_id=model_id,
                symbol=symbol,
                error=str(e),
            )

            return InferencePredictionResponse(
                request_id=f"feat_pred_{int(time.time() * 1000)}",
                predictions=[],
                error=error_msg,
            )

    # Model Management
    async def _get_model(self, model_id: str, use_cache: bool = True) -> Any:
        """Get model from cache or registry."""
        if use_cache:
            # Try cache first
            cached_model = await self._get_cached_model(model_id)
            if cached_model is not None:
                self.metrics.cache_hits += 1
                return cached_model

            self.metrics.cache_misses += 1

        # Load from registry service
        model_info = await self.model_registry_service.load_model(model_id)

        # Extract model from model_info
        # ModelRegistryService should return a dict with 'model' key
        if isinstance(model_info, dict) and "model" in model_info:
            model = model_info["model"]
        else:
            # If model_info is the model itself or doesn't have 'model' key
            model = model_info
            
        if model is None:
            raise ModelError(f"Failed to load model {model_id}: No model returned")

        # Cache the model if caching is enabled
        if use_cache:
            await self._cache_model(model_id, model)

        return model

    async def _make_prediction(
        self, model: Any, features: pd.DataFrame, return_probabilities: bool
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Make prediction using model."""
        loop = asyncio.get_event_loop()

        if return_probabilities:
            # Run prediction in thread pool
            result = await loop.run_in_executor(
                self._executor, self._predict_with_probabilities, model, features
            )
            predictions, probabilities = result

            # Calculate confidence scores
            confidence_scores = None
            if probabilities is not None:
                confidence_scores = np.max(probabilities, axis=1)

            return predictions, probabilities, confidence_scores
        else:
            # Run prediction in thread pool
            predictions = await loop.run_in_executor(
                self._executor, self._predict_without_probabilities, model, features
            )

            return predictions, None, None

    def _predict_with_probabilities(
        self, model: Any, features: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synchronous prediction with probabilities."""
        try:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)
                predictions = model.predict(features)
                return predictions, probabilities
            else:
                predictions = model.predict(features)
                return predictions, None
        except Exception as e:
            raise ModelError(f"Model prediction failed: {e}")

    def _predict_without_probabilities(self, model: Any, features: pd.DataFrame) -> np.ndarray:
        """Synchronous prediction without probabilities."""
        try:
            return model.predict(features)
        except Exception as e:
            raise ModelError(f"Model prediction failed: {e}")

    # Caching Operations
    def _generate_prediction_cache_key(
        self, model_id: str, features: pd.DataFrame, return_probabilities: bool
    ) -> str:
        """Generate cache key for prediction."""
        import hashlib

        # Create hash from model_id, features shape, and options
        features_hash = str(hash(tuple(features.iloc[0].values))) if not features.empty else "empty"
        cache_str = f"{model_id}_{features.shape}_{features_hash}_{return_probabilities}"

        return hashlib.md5(cache_str.encode()).hexdigest()[:16]

    async def _get_cached_model(self, model_id: str) -> Any | None:
        """Get cached model."""
        if model_id in self._model_cache:
            model, timestamp = self._model_cache[model_id]

            # Check if expired
            ttl_hours = self.inference_config.model_cache_ttl_hours
            if (datetime.utcnow() - timestamp).total_seconds() < ttl_hours * 3600:
                return model
            else:
                # Remove expired model
                del self._model_cache[model_id]

        return None

    async def _cache_model(self, model_id: str, model: Any) -> None:
        """Cache model."""
        self._model_cache[model_id] = (model, datetime.utcnow())

        # Clean old cache entries
        await self._clean_model_cache()

    async def _get_cached_prediction(self, cache_key: str) -> InferencePredictionResponse | None:
        """Get cached prediction."""
        if cache_key in self._prediction_cache:
            response, timestamp = self._prediction_cache[cache_key]

            # Check if expired (5 minute TTL for predictions)
            if (datetime.utcnow() - timestamp).total_seconds() < 300:
                return response
            else:
                del self._prediction_cache[cache_key]

        return None

    async def _cache_prediction(
        self, cache_key: str, response: InferencePredictionResponse
    ) -> None:
        """Cache prediction response."""
        self._prediction_cache[cache_key] = (response, datetime.utcnow())

        # Clean old cache entries to prevent memory issues
        await self._clean_prediction_cache()

    async def _clean_model_cache(self) -> None:
        """Clean expired model cache entries."""
        ttl_hours = self.inference_config.model_cache_ttl_hours
        cutoff_time = datetime.utcnow() - timedelta(hours=ttl_hours)

        expired_keys = [
            key for key, (_, timestamp) in self._model_cache.items() if timestamp < cutoff_time
        ]

        for key in expired_keys:
            del self._model_cache[key]

        if expired_keys:
            self._logger.debug(f"Cleaned {len(expired_keys)} expired model cache entries")

    async def _clean_prediction_cache(self) -> None:
        """Clean expired prediction cache entries."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=5)  # 5 minute TTL

        expired_keys = [
            key for key, (_, timestamp) in self._prediction_cache.items() if timestamp < cutoff_time
        ]

        for key in expired_keys:
            del self._prediction_cache[key]

        if expired_keys:
            self._logger.debug(f"Cleaned {len(expired_keys)} expired prediction cache entries")

    # Batch Processing
    async def _batch_processor_loop(self) -> None:
        """Background loop for processing batched requests."""
        while True:
            try:
                # Collect batch of requests
                batch = []
                deadline = time.time() + (self.inference_config.batch_timeout_ms / 1000)

                while (
                    len(batch) < self.inference_config.inference_batch_size
                    and time.time() < deadline
                ):
                    try:
                        # Wait for request with timeout
                        timeout = max(0.001, deadline - time.time())
                        request_future_pair = await asyncio.wait_for(
                            self._batch_queue.get(), timeout=timeout
                        )
                        batch.append(request_future_pair)
                    except asyncio.TimeoutError:
                        break

                # Process batch if we have requests
                if batch:
                    await self._process_batch(batch)

                # Small sleep to prevent busy waiting
                await asyncio.sleep(0.001)

            except asyncio.CancelledError:
                # Handle remaining requests
                if batch:
                    await self._process_batch(batch)
                break
            except Exception as e:
                self._logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(0.1)

    async def _process_batch(
        self, batch: list[tuple[InferencePredictionRequest, asyncio.Future]]
    ) -> None:
        """Process a batch of requests."""
        try:
            requests = [req for req, _ in batch]
            futures = [fut for _, fut in batch]

            # Process batch
            responses = await self._predict_batch_impl(requests)

            # Set results
            for future, response in zip(futures, responses):
                if not future.done():
                    future.set_result(response)

        except Exception as e:
            # Set error for all futures
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)

    # Model Warmup
    async def warm_up_models(self, model_ids: list[str]) -> dict[str, bool]:
        """
        Warm up models by loading them into cache.

        Args:
            model_ids: List of model IDs to warm up

        Returns:
            Dictionary of model_id -> success status
        """
        if not self.inference_config.enable_model_warmup:
            self._logger.info("Model warmup is disabled")
            return {}

        return await self.execute_with_monitoring(
            "warm_up_models",
            self._warm_up_models_impl,
            model_ids,
        )

    async def _warm_up_models_impl(self, model_ids: list[str]) -> dict[str, bool]:
        """Internal model warmup implementation."""
        self._logger.info(f"Warming up {len(model_ids)} models")

        warmup_results = {}

        # Process warmup concurrently
        warmup_tasks = []
        for model_id in model_ids:
            task = asyncio.create_task(self._warm_up_single_model(model_id))
            warmup_tasks.append((model_id, task))

        # Wait for all warmup tasks
        results = await asyncio.gather(*[task for _, task in warmup_tasks], return_exceptions=True)

        # Collect results
        for i, result in enumerate(results):
            model_id = warmup_tasks[i][0]
            if isinstance(result, Exception):
                warmup_results[model_id] = False
                self._logger.warning(f"Failed to warm up model {model_id}: {result}")
            else:
                warmup_results[model_id] = result
                if result:
                    self.metrics.models_warmed_up += 1

        successful_warmups = sum(warmup_results.values())
        self._logger.info(f"Warmed up {successful_warmups}/{len(model_ids)} models successfully")

        return warmup_results

    async def _warm_up_single_model(self, model_id: str) -> bool:
        """Warm up a single model."""
        try:
            # Load model into cache
            await self._get_model(model_id, use_cache=True)
            self._logger.debug(f"Model {model_id} warmed up successfully")
            return True
        except Exception as e:
            self._logger.warning(f"Failed to warm up model {model_id}: {e}")
            return False

    # Service Health and Metrics
    async def _service_health_check(self) -> Any:
        """Inference service specific health check."""
        from src.core.types import HealthStatus

        try:
            # Check dependencies
            if not all([self.model_registry_service, self.feature_engineering_service]):
                return HealthStatus.UNHEALTHY

            # Check if success rate is acceptable
            if self.metrics.total_requests > 100:
                success_rate = self.metrics.successful_predictions / self.metrics.total_requests
                if success_rate < 0.9:
                    return HealthStatus.DEGRADED
                elif success_rate < 0.5:
                    return HealthStatus.UNHEALTHY

            # Check cache sizes
            if len(self._model_cache) > 100 or len(self._prediction_cache) > 10000:
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            self._logger.error("Inference service health check failed", error=str(e))
            return HealthStatus.UNHEALTHY

    def get_inference_metrics(self) -> dict[str, Any]:
        """Get inference service metrics."""
        total_requests = self.metrics.total_requests

        return {
            **self.metrics.dict(),
            "success_rate": (
                self.metrics.successful_predictions / total_requests if total_requests > 0 else 0.0
            ),
            "cache_hit_rate": (
                self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses)
                if (self.metrics.cache_hits + self.metrics.cache_misses) > 0
                else 0.0
            ),
            "cached_models": len(self._model_cache),
            "cached_predictions": len(self._prediction_cache),
            "batch_queue_size": self._batch_queue.qsize() if self._batch_queue else 0,
        }

    async def clear_cache(self) -> dict[str, int]:
        """Clear inference caches."""
        model_cache_size = len(self._model_cache)
        prediction_cache_size = len(self._prediction_cache)

        self._model_cache.clear()
        self._prediction_cache.clear()

        # Reset metrics related to cache
        self.metrics.cache_hits = 0
        self.metrics.cache_misses = 0

        self._logger.info(
            "Inference caches cleared",
            models_removed=model_cache_size,
            predictions_removed=prediction_cache_size,
        )

        return {
            "models_removed": model_cache_size,
            "predictions_removed": prediction_cache_size,
        }

    def reset_metrics(self) -> None:
        """Reset inference metrics."""
        self.metrics = InferenceMetrics()
        self._logger.info("Inference metrics reset")

    # Configuration validation
    def _validate_service_config(self, config: ConfigDict) -> bool:
        """Validate inference service configuration."""
        try:
            inference_config_dict = config.get("inference", {})
            InferenceConfig(**inference_config_dict)
            return True
        except Exception as e:
            self._logger.error("Inference service configuration validation failed", error=str(e))
            return False
