"""
ML Caching Utilities.

This module provides common caching functionality for ML services to eliminate
duplicate caching logic across the ML module.
"""

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Generic, TypeVar

from src.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class TTLCache(Generic[T]):
    """Time-To-Live cache implementation."""

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        """
        Initialize TTL cache.

        Args:
            ttl_seconds: Time-to-live in seconds
            max_size: Maximum cache size
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: dict[str, tuple[T, datetime]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> T | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if datetime.now(timezone.utc) - timestamp < timedelta(seconds=self.ttl_seconds):
                    return value
                else:
                    # Remove expired entry
                    del self._cache[key]
            return None

    async def set(self, key: str, value: T) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        async with self._lock:
            # Check size limit
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Remove oldest entry
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[key] = (value, datetime.now(timezone.utc))

    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    async def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    async def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.

        Returns:
            Number of expired entries removed
        """
        async with self._lock:
            cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.ttl_seconds)
            expired_keys = [
                key for key, (_, timestamp) in self._cache.items() if timestamp < cutoff_time
            ]

            for key in expired_keys:
                del self._cache[key]

            return len(expired_keys)


class ModelCache(TTLCache[Any]):
    """Specialized cache for ML models."""

    def __init__(self, ttl_hours: int = 24, max_models: int = 100):
        """
        Initialize model cache.

        Args:
            ttl_hours: Time-to-live in hours
            max_models: Maximum number of cached models
        """
        super().__init__(ttl_seconds=ttl_hours * 3600, max_size=max_models)

    async def get_model(self, model_id: str) -> Any | None:
        """Get model from cache."""
        return await self.get(model_id)

    async def cache_model(self, model_id: str, model: Any) -> None:
        """Cache a model."""
        await self.set(model_id, model)

    async def remove_model(self, model_id: str) -> bool:
        """Remove model from cache."""
        return await self.delete(model_id)


class PredictionCache(TTLCache[dict[str, Any]]):
    """Specialized cache for ML predictions."""

    def __init__(self, ttl_minutes: int = 5, max_predictions: int = 10000):
        """
        Initialize prediction cache.

        Args:
            ttl_minutes: Time-to-live in minutes
            max_predictions: Maximum number of cached predictions
        """
        super().__init__(ttl_seconds=ttl_minutes * 60, max_size=max_predictions)

    async def get_prediction(self, cache_key: str) -> dict[str, Any] | None:
        """Get prediction from cache."""
        return await self.get(cache_key)

    async def cache_prediction(self, cache_key: str, prediction: dict[str, Any]) -> None:
        """Cache a prediction."""
        await self.set(cache_key, prediction)


class FeatureCache(TTLCache[dict[str, Any]]):
    """Specialized cache for feature sets."""

    def __init__(self, ttl_hours: int = 4, max_feature_sets: int = 1000):
        """
        Initialize feature cache.

        Args:
            ttl_hours: Time-to-live in hours
            max_feature_sets: Maximum number of cached feature sets
        """
        super().__init__(ttl_seconds=ttl_hours * 3600, max_size=max_feature_sets)

    async def get_features(self, cache_key: str) -> dict[str, Any] | None:
        """Get features from cache."""
        return await self.get(cache_key)

    async def cache_features(self, cache_key: str, features: dict[str, Any]) -> None:
        """Cache features."""
        await self.set(cache_key, features)


def generate_cache_key(*args: Any) -> str:
    """
    Generate a cache key from arguments.

    Args:
        *args: Arguments to create cache key from

    Returns:
        MD5 hash of the arguments
    """
    # Convert arguments to string representation
    key_parts = []
    for arg in args:
        if hasattr(arg, "shape"):  # Handle pandas/numpy objects
            key_parts.append(f"{type(arg).__name__}_{arg.shape}")
            if hasattr(arg, "iloc") and len(arg) > 0:  # DataFrame/Series
                key_parts.append(
                    str(
                        hash(tuple(arg.iloc[0].values if hasattr(arg, "values") else [arg.iloc[0]]))
                    )
                )
        elif isinstance(arg, (dict, list)):
            key_parts.append(str(hash(str(arg))))
        else:
            key_parts.append(str(arg))

    cache_str = "_".join(key_parts)
    return hashlib.md5(cache_str.encode()).hexdigest()[:16]


def generate_model_cache_key(model_id: str, version: str | None = None) -> str:
    """
    Generate cache key for models.

    Args:
        model_id: Model identifier
        version: Optional model version

    Returns:
        Cache key for the model
    """
    if version:
        return f"model_{model_id}_{version}"
    return f"model_{model_id}"


def generate_prediction_cache_key(
    model_id: str, features_hash: str, return_probabilities: bool = False
) -> str:
    """
    Generate cache key for predictions.

    Args:
        model_id: Model identifier
        features_hash: Hash of feature data
        return_probabilities: Whether probabilities were requested

    Returns:
        Cache key for the prediction
    """
    return f"pred_{model_id}_{features_hash}_{return_probabilities}"


def generate_feature_cache_key(symbol: str, feature_types: list[str], data_hash: str) -> str:
    """
    Generate cache key for features.

    Args:
        symbol: Trading symbol
        feature_types: List of feature types
        data_hash: Hash of input data

    Returns:
        Cache key for the features
    """
    feature_types_str = "_".join(sorted(feature_types))
    return f"feat_{symbol}_{feature_types_str}_{data_hash}"


class CacheManager:
    """Centralized cache manager for ML operations."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize cache manager.

        Args:
            config: Cache configuration
        """
        config = config or {}

        # Initialize caches
        self.model_cache = ModelCache(
            ttl_hours=config.get("model_cache_ttl_hours", 24),
            max_models=config.get("max_cached_models", 100),
        )

        self.prediction_cache = PredictionCache(
            ttl_minutes=config.get("prediction_cache_ttl_minutes", 5),
            max_predictions=config.get("max_cached_predictions", 10000),
        )

        self.feature_cache = FeatureCache(
            ttl_hours=config.get("feature_cache_ttl_hours", 4),
            max_feature_sets=config.get("max_cached_feature_sets", 1000),
        )

        # Background cleanup task
        self._cleanup_task: asyncio.Task | None = None
        self._cleanup_interval = config.get("cleanup_interval_minutes", 30)

    async def start(self) -> None:
        """Start the cache manager and background cleanup."""
        if self._cleanup_interval > 0:
            self._cleanup_task = asyncio.create_task(self._background_cleanup())
        logger.info("Cache manager started")

    async def stop(self) -> None:
        """Stop the cache manager and cleanup tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Cache manager stopped")

    async def clear_all(self) -> dict[str, int]:
        """
        Clear all caches.

        Returns:
            Dictionary with counts of cleared entries
        """
        results = {
            "models_cleared": await self.model_cache.clear(),
            "predictions_cleared": await self.prediction_cache.clear(),
            "features_cleared": await self.feature_cache.clear(),
        }

        logger.info("All caches cleared", **results)
        return results

    async def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "model_cache_size": await self.model_cache.size(),
            "prediction_cache_size": await self.prediction_cache.size(),
            "feature_cache_size": await self.feature_cache.size(),
            "model_cache_ttl_hours": self.model_cache.ttl_seconds // 3600,
            "prediction_cache_ttl_minutes": self.prediction_cache.ttl_seconds // 60,
            "feature_cache_ttl_hours": self.feature_cache.ttl_seconds // 3600,
        }

    async def _background_cleanup(self) -> None:
        """Background task for cache cleanup."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval * 60)  # Convert to seconds

                # Clean up expired entries
                model_cleaned = await self.model_cache.cleanup_expired()
                prediction_cleaned = await self.prediction_cache.cleanup_expired()
                feature_cleaned = await self.feature_cache.cleanup_expired()

                if model_cleaned + prediction_cleaned + feature_cleaned > 0:
                    logger.debug(
                        "Cache cleanup completed",
                        models_cleaned=model_cleaned,
                        predictions_cleaned=prediction_cleaned,
                        features_cleaned=feature_cleaned,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")


# Global cache manager instance
_cache_manager: CacheManager | None = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


async def init_cache_manager(config: dict[str, Any] | None = None) -> CacheManager:
    """
    Initialize the global cache manager.

    Args:
        config: Cache configuration

    Returns:
        Initialized cache manager
    """
    global _cache_manager
    _cache_manager = CacheManager(config)
    await _cache_manager.start()
    return _cache_manager
