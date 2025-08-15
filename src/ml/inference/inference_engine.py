"""
Real-time Inference Engine for ML Models.

This module provides high-performance real-time inference capabilities for trained
ML models with caching, batch processing, and performance monitoring.
"""

import asyncio
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.ml.feature_engineering import FeatureEngineer
from src.ml.inference.model_cache import ModelCache
from src.ml.models.base_model import BaseModel
from src.ml.registry.model_registry import ModelRegistry
from src.utils.decorators import cache_result, log_calls, time_execution

logger = get_logger(__name__)


class PredictionRequest:
    """Request object for predictions."""

    def __init__(
        self,
        request_id: str,
        model_id: str,
        features: pd.DataFrame,
        return_probabilities: bool = False,
        metadata: dict[str, Any] | None = None,
    ):
        self.request_id = request_id
        self.model_id = model_id
        self.features = features
        self.return_probabilities = return_probabilities
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()


class PredictionResponse:
    """Response object for predictions."""

    def __init__(
        self,
        request_id: str,
        predictions: np.ndarray,
        probabilities: np.ndarray | None = None,
        confidence_scores: np.ndarray | None = None,
        processing_time_ms: float = 0.0,
        model_info: dict[str, Any] | None = None,
        error: str | None = None,
    ):
        self.request_id = request_id
        self.predictions = predictions
        self.probabilities = probabilities
        self.confidence_scores = confidence_scores
        self.processing_time_ms = processing_time_ms
        self.model_info = model_info
        self.error = error
        self.timestamp = datetime.utcnow()


class InferenceEngine:
    """
    Real-time inference engine for ML models.

    This class provides high-performance real-time inference capabilities including
    model caching, batch processing, async processing, and performance monitoring.

    Attributes:
        config: Application configuration
        model_registry: Model registry instance
        model_cache: Model cache instance
        feature_engineer: Feature engineering instance
        executor: Thread pool executor for async processing
        request_queue: Queue for batching requests
        processing_stats: Statistics tracking
    """

    def __init__(self, config: Config):
        """
        Initialize the inference engine.

        Args:
            config: Application configuration
        """
        self.config = config
        self.model_registry = ModelRegistry(config)
        self.model_cache = ModelCache(config)
        self.feature_engineer = FeatureEngineer(config)

        # Async processing
        self.executor = ThreadPoolExecutor(max_workers=config.ml.max_cpu_cores)
        self.request_queue = queue.Queue(maxsize=config.ml.inference_batch_size * 2)

        # Processing statistics
        self.processing_stats = {
            "total_requests": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Background processing
        self._batch_processor_running = False
        self._batch_processor_thread = None

        logger.info(
            "Inference engine initialized",
            max_workers=config.ml.max_cpu_cores,
            batch_size=config.ml.inference_batch_size,
        )

    def start_batch_processor(self) -> None:
        """Start the background batch processor."""
        if not self._batch_processor_running:
            self._batch_processor_running = True
            self._batch_processor_thread = threading.Thread(
                target=self._batch_processor_loop, daemon=True
            )
            self._batch_processor_thread.start()
            logger.info("Batch processor started")

    def stop_batch_processor(self) -> None:
        """Stop the background batch processor."""
        if self._batch_processor_running:
            self._batch_processor_running = False
            if self._batch_processor_thread:
                self._batch_processor_thread.join(timeout=5.0)
            logger.info("Batch processor stopped")

    @time_execution
    @log_calls
    def predict(
        self,
        model_id: str,
        features: pd.DataFrame,
        return_probabilities: bool = False,
        use_cache: bool = True,
        request_id: str | None = None,
    ) -> PredictionResponse:
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

        Raises:
            ModelError: If prediction fails
        """
        start_time = time.time()

        if request_id is None:
            request_id = f"pred_{int(time.time() * 1000)}"

        try:
            self.processing_stats["total_requests"] += 1

            # Get model
            model = self._get_model(model_id, use_cache)

            # Validate features
            if features.empty:
                raise ValidationError("Features cannot be empty")

            # Make prediction
            if return_probabilities:
                predictions, probabilities = model.predict(features, return_probabilities=True)

                # Calculate confidence scores
                if probabilities is not None:
                    confidence_scores = np.max(probabilities, axis=1)
                else:
                    confidence_scores = None
            else:
                predictions = model.predict(features, return_probabilities=False)
                probabilities = None
                confidence_scores = None

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Update statistics
            self.processing_stats["successful_predictions"] += 1
            self.processing_stats["total_processing_time"] += processing_time_ms
            self.processing_stats["average_processing_time"] = (
                self.processing_stats["total_processing_time"]
                / self.processing_stats["total_requests"]
            )

            # Create response
            response = PredictionResponse(
                request_id=request_id,
                predictions=predictions,
                probabilities=probabilities,
                confidence_scores=confidence_scores,
                processing_time_ms=processing_time_ms,
                model_info={
                    "model_id": model_id,
                    "model_name": model.model_name,
                    "model_type": model.model_type,
                    "version": model.version,
                },
            )

            logger.debug(
                "Prediction completed",
                request_id=request_id,
                model_id=model_id,
                processing_time_ms=processing_time_ms,
                features_shape=features.shape,
            )

            return response

        except Exception as e:
            self.processing_stats["failed_predictions"] += 1

            processing_time_ms = (time.time() - start_time) * 1000

            logger.error(
                "Prediction failed", request_id=request_id, model_id=model_id, error=str(e)
            )

            return PredictionResponse(
                request_id=request_id,
                predictions=np.array([]),
                processing_time_ms=processing_time_ms,
                error=str(e),
            )

    @time_execution
    @log_calls
    async def predict_async(
        self,
        model_id: str,
        features: pd.DataFrame,
        return_probabilities: bool = False,
        use_cache: bool = True,
        request_id: str | None = None,
    ) -> PredictionResponse:
        """
        Make an async prediction.

        Args:
            model_id: Model ID to use for prediction
            features: Feature data
            return_probabilities: Whether to return probabilities
            use_cache: Whether to use model cache
            request_id: Optional request ID

        Returns:
            Prediction response
        """
        loop = asyncio.get_event_loop()

        # Run prediction in thread pool
        response = await loop.run_in_executor(
            self.executor,
            self.predict,
            model_id,
            features,
            return_probabilities,
            use_cache,
            request_id,
        )

        return response

    @time_execution
    @log_calls
    def predict_batch(self, requests: list[PredictionRequest]) -> list[PredictionResponse]:
        """
        Process a batch of prediction requests.

        Args:
            requests: List of prediction requests

        Returns:
            List of prediction responses
        """
        if not requests:
            return []

        start_time = time.time()
        responses = []

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
                model = self._get_model(model_id, use_cache=True)

                # Process requests for this model
                for request in model_requests:
                    try:
                        request_start = time.time()

                        # Make prediction
                        if request.return_probabilities:
                            predictions, probabilities = model.predict(
                                request.features, return_probabilities=True
                            )
                            confidence_scores = (
                                np.max(probabilities, axis=1) if probabilities is not None else None
                            )
                        else:
                            predictions = model.predict(request.features)
                            probabilities = None
                            confidence_scores = None

                        processing_time_ms = (time.time() - request_start) * 1000

                        response = PredictionResponse(
                            request_id=request.request_id,
                            predictions=predictions,
                            probabilities=probabilities,
                            confidence_scores=confidence_scores,
                            processing_time_ms=processing_time_ms,
                            model_info={
                                "model_id": model_id,
                                "model_name": model.model_name,
                                "model_type": model.model_type,
                            },
                        )

                        responses.append(response)
                        self.processing_stats["successful_predictions"] += 1

                    except Exception as e:
                        response = PredictionResponse(
                            request_id=request.request_id, predictions=np.array([]), error=str(e)
                        )
                        responses.append(response)
                        self.processing_stats["failed_predictions"] += 1

            except Exception as e:
                # Model loading failed - create error responses for all requests
                for request in model_requests:
                    response = PredictionResponse(
                        request_id=request.request_id,
                        predictions=np.array([]),
                        error=f"Model loading failed: {e}",
                    )
                    responses.append(response)
                    self.processing_stats["failed_predictions"] += 1

        batch_processing_time = (time.time() - start_time) * 1000

        logger.info(
            "Batch prediction completed",
            batch_size=len(requests),
            successful_predictions=sum(1 for r in responses if r.error is None),
            failed_predictions=sum(1 for r in responses if r.error is not None),
            batch_processing_time_ms=batch_processing_time,
        )

        return responses

    @cache_result(ttl_seconds=300)  # 5-minute cache
    def predict_with_features(
        self,
        model_id: str,
        market_data: pd.DataFrame,
        symbol: str,
        return_probabilities: bool = False,
    ) -> PredictionResponse:
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
        try:
            # Create features
            features = self.feature_engineer.create_features(market_data, symbol)

            # Make prediction
            response = self.predict(model_id, features, return_probabilities)

            return response

        except Exception as e:
            logger.error(
                "Prediction with features failed", model_id=model_id, symbol=symbol, error=str(e)
            )

            return PredictionResponse(
                request_id=f"feat_pred_{int(time.time() * 1000)}",
                predictions=np.array([]),
                error=str(e),
            )

    def _get_model(self, model_id: str, use_cache: bool = True) -> BaseModel:
        """Get model from cache or registry."""
        if use_cache:
            # Try cache first
            model = self.model_cache.get_model(model_id)
            if model is not None:
                self.processing_stats["cache_hits"] += 1
                return model

            self.processing_stats["cache_misses"] += 1

        # Load from registry
        model = self.model_registry.get_model(model_id=model_id)

        # Cache the model if caching is enabled
        if use_cache:
            self.model_cache.cache_model(model_id, model)

        return model

    def _batch_processor_loop(self) -> None:
        """Background loop for processing batched requests."""
        while self._batch_processor_running:
            try:
                # Collect batch of requests
                batch = []
                deadline = time.time() + 0.1  # 100ms deadline

                while len(batch) < self.config.ml.inference_batch_size and time.time() < deadline:
                    try:
                        request = self.request_queue.get(timeout=0.01)
                        batch.append(request)
                    except queue.Empty:
                        break

                # Process batch if we have requests
                if batch:
                    responses = self.predict_batch(batch)

                    # Here you would typically send responses back
                    # This is a simplified implementation
                    logger.debug(f"Processed batch of {len(batch)} requests")

                # Small sleep to prevent busy waiting
                time.sleep(0.001)

            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                time.sleep(0.1)

    def warm_up_models(self, model_ids: list[str]) -> dict[str, bool]:
        """
        Warm up models by loading them into cache.

        Args:
            model_ids: List of model IDs to warm up

        Returns:
            Dictionary of model_id -> success status
        """
        if not self.config.ml.model_warmup_enabled:
            logger.info("Model warmup is disabled")
            return {}

        logger.info(f"Warming up {len(model_ids)} models")

        warmup_results = {}

        for model_id in model_ids:
            try:
                # Load model into cache
                model = self._get_model(model_id, use_cache=True)
                warmup_results[model_id] = True

                logger.debug(f"Model {model_id} warmed up successfully")

            except Exception as e:
                warmup_results[model_id] = False
                logger.warning(f"Failed to warm up model {model_id}: {e}")

        successful_warmups = sum(warmup_results.values())
        logger.info(f"Warmed up {successful_warmups}/{len(model_ids)} models successfully")

        return warmup_results

    def get_processing_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        stats = self.processing_stats.copy()

        # Add cache stats
        cache_stats = self.model_cache.get_cache_stats()
        stats.update(cache_stats)

        # Calculate additional metrics
        if stats["total_requests"] > 0:
            stats["success_rate"] = stats["successful_predictions"] / stats["total_requests"]
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_requests"]
        else:
            stats["success_rate"] = 0.0
            stats["cache_hit_rate"] = 0.0

        return stats

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            "total_requests": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        self.model_cache.clear_stats()
        logger.info("Processing statistics reset")

    def health_check(self) -> dict[str, Any]:
        """Perform health check of the inference engine."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "batch_processor_running": self._batch_processor_running,
            "executor_active": not self.executor._shutdown,
            "queue_size": self.request_queue.qsize(),
            "processing_stats": self.get_processing_stats(),
        }

        # Check if there are any critical issues
        if self.processing_stats["total_requests"] > 100:
            if self.processing_stats["success_rate"] < 0.9:
                health_status["status"] = "degraded"
                health_status["warning"] = "Low success rate"

        return health_status

    def __enter__(self):
        """Context manager entry."""
        self.start_batch_processor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_batch_processor()
        self.executor.shutdown(wait=True)
