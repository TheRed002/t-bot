"""
Model Cache for ML Models.

This module provides high-performance caching for ML models with LRU eviction,
memory monitoring, and cache statistics.
"""

import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any

import psutil
from pydantic import BaseModel, Field

from src.core.base.service import BaseService
from src.core.types.base import ConfigDict
from src.utils.decorators import UnifiedDecorator

# Initialize decorator instance
dec = UnifiedDecorator()


class ModelCacheConfig(BaseModel):
    """Configuration for model cache service."""

    model_cache_size: int = Field(default=10, description="Maximum number of cached models")
    max_memory_gb: float = Field(default=2.0, description="Maximum memory usage in GB")
    prediction_cache_ttl_minutes: int = Field(default=60, description="Cache TTL in minutes")
    enable_memory_monitoring: bool = Field(default=True, description="Enable memory monitoring")
    cleanup_interval_seconds: int = Field(default=30, description="Cleanup interval in seconds")
    memory_pressure_threshold: float = Field(
        default=85.0, description="Memory pressure threshold %"
    )


class ModelCacheService(BaseService):
    """
    High-performance cache for ML models.

    This service provides LRU-based caching with memory monitoring, automatic eviction,
    and comprehensive statistics tracking using proper service patterns.

    All cached models are managed through this service without direct database access.
    """

    def __init__(
        self,
        config: ConfigDict | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize the model cache service.

        Args:
            config: Service configuration
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="ModelCacheService",
            config=config,
            correlation_id=correlation_id,
        )

        # Parse model cache configuration
        mc_config_dict = (config or {}).get("model_cache", {})
        self.mc_config = ModelCacheConfig(**mc_config_dict)

        # Cache data structures
        self.cache: dict[str, Any] = OrderedDict()
        self.access_times: dict[str, datetime] = {}
        self.memory_usage: dict[str, float] = {}

        # Cache statistics
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0,
            "memory_evictions": 0,
            "current_size": 0,
            "current_memory_mb": 0.0,
        }

        # Thread safety
        self.lock = threading.RLock()

        # Configuration properties
        self.max_size = self.mc_config.model_cache_size
        self.max_memory_mb = self.mc_config.max_memory_gb * 1024  # Convert GB to MB
        self.ttl_minutes = self.mc_config.prediction_cache_ttl_minutes

        # Background cleanup
        self._cleanup_running = False
        self._cleanup_thread = None

    async def _do_start(self) -> None:
        """Start the model cache service."""
        await super()._do_start()

        # Start cleanup thread if memory monitoring is enabled
        if self.mc_config.enable_memory_monitoring:
            self.start_cleanup_thread()

        self._logger.info(
            "Model cache service started successfully",
            config=self.mc_config.dict(),
            max_size=self.max_size,
            max_memory_mb=self.max_memory_mb,
            ttl_minutes=self.ttl_minutes,
        )

    async def _do_stop(self) -> None:
        """Stop the model cache service."""
        self.stop_cleanup_thread()
        await super()._do_stop()

    def start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        if not self._cleanup_running:
            self._cleanup_running = True
            self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self._cleanup_thread.start()
            self._logger.info("Cache cleanup thread started")

    def stop_cleanup_thread(self) -> None:
        """Stop background cleanup thread."""
        if self._cleanup_running:
            self._cleanup_running = False
            if self._cleanup_thread:
                self._cleanup_thread.join(timeout=5.0)
            self._logger.info("Cache cleanup thread stopped")

    @dec.enhance(log=True, monitor=True, log_level="info")
    def cache_model(self, model_id: str, model: Any) -> bool:
        """
        Cache a model.

        Args:
            model_id: Unique model identifier
            model: Model instance to cache

        Returns:
            True if successfully cached, False otherwise
        """
        with self.lock:
            try:
                # Check if model is already cached
                if model_id in self.cache:
                    # Move to end (most recent)
                    self.cache.move_to_end(model_id)
                    self.access_times[model_id] = datetime.utcnow()
                    return True

                # Estimate model memory usage
                model_memory_mb = self._estimate_model_memory(model)

                # Check if we need to make space
                self._make_space_for_model(model_memory_mb)

                # Add model to cache
                self.cache[model_id] = model
                self.access_times[model_id] = datetime.utcnow()
                self.memory_usage[model_id] = model_memory_mb

                # Update statistics
                self.cache_stats["current_size"] = len(self.cache)
                self.cache_stats["current_memory_mb"] = sum(self.memory_usage.values())

                self._logger.debug(
                    "Model cached successfully",
                    model_id=model_id,
                    model_memory_mb=model_memory_mb,
                    cache_size=len(self.cache),
                )

                return True

            except Exception as e:
                self._logger.error("Failed to cache model", model_id=model_id, error=str(e))
                return False

    @dec.enhance(monitor=True)
    def get_model(self, model_id: str) -> Any | None:
        """
        Retrieve a model from cache.

        Args:
            model_id: Model identifier

        Returns:
            Model instance if found, None otherwise
        """
        with self.lock:
            self.cache_stats["total_requests"] += 1

            if model_id in self.cache:
                # Check TTL
                access_time = self.access_times[model_id]
                if datetime.utcnow() - access_time > timedelta(minutes=self.ttl_minutes):
                    # Model has expired
                    self._remove_model(model_id, reason="TTL expired")
                    self.cache_stats["misses"] += 1
                    return None

                # Move to end (most recent) and update access time
                model = self.cache[model_id]
                self.cache.move_to_end(model_id)
                self.access_times[model_id] = datetime.utcnow()

                self.cache_stats["hits"] += 1

                self._logger.debug(
                    "Cache hit", model_id=model_id, hit_rate=self._calculate_hit_rate()
                )

                return model
            else:
                self.cache_stats["misses"] += 1

                self._logger.debug(
                    "Cache miss", model_id=model_id, hit_rate=self._calculate_hit_rate()
                )

                return None

    def remove_model(self, model_id: str) -> bool:
        """
        Remove a model from cache.

        Args:
            model_id: Model identifier

        Returns:
            True if removed, False if not found
        """
        with self.lock:
            return self._remove_model(model_id, reason="Manual removal")

    def clear_cache(self) -> None:
        """Clear all models from cache."""
        with self.lock:
            model_count = len(self.cache)

            self.cache.clear()
            self.access_times.clear()
            self.memory_usage.clear()

            self.cache_stats["current_size"] = 0
            self.cache_stats["current_memory_mb"] = 0.0

            self._logger.info(f"Cache cleared, removed {model_count} models")

    def get_cached_models(self) -> dict[str, dict[str, Any]]:
        """
        Get information about currently cached models.

        Returns:
            Dictionary with model information
        """
        with self.lock:
            models_info = {}

            for model_id, model in self.cache.items():
                models_info[model_id] = {
                    "model_name": getattr(model, "model_name", "unknown"),
                    "model_type": getattr(model, "model_type", "unknown"),
                    "version": getattr(model, "version", "unknown"),
                    "is_trained": getattr(model, "is_trained", False),
                    "last_access": self.access_times[model_id].isoformat(),
                    "memory_mb": self.memory_usage.get(model_id, 0.0),
                    "feature_count": len(getattr(model, "feature_names", [])),
                }

            return models_info

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            stats = self.cache_stats.copy()

            # Calculate derived metrics
            stats["hit_rate"] = self._calculate_hit_rate()
            stats["memory_usage_pct"] = (
                stats["current_memory_mb"] / self.max_memory_mb * 100
                if self.max_memory_mb > 0
                else 0
            )
            stats["capacity_usage_pct"] = (
                stats["current_size"] / self.max_size * 100 if self.max_size > 0 else 0
            )

            return stats

    def clear_stats(self) -> None:
        """Clear cache statistics."""
        with self.lock:
            self.cache_stats.update(
                {"hits": 0, "misses": 0, "evictions": 0, "total_requests": 0, "memory_evictions": 0}
            )

            self._logger.info("Cache statistics cleared")

    def _make_space_for_model(self, required_memory_mb: float) -> None:
        """Make space for a new model by evicting old ones."""
        # Check size limit
        while len(self.cache) >= self.max_size:
            self._evict_lru_model("Size limit reached")

        # Check memory limit
        current_memory = sum(self.memory_usage.values())
        while current_memory + required_memory_mb > self.max_memory_mb:
            if not self.cache:
                break

            self._evict_lru_model("Memory limit reached")
            current_memory = sum(self.memory_usage.values())

    def _evict_lru_model(self, reason: str) -> None:
        """Evict the least recently used model."""
        if not self.cache:
            return

        # Get LRU model (first item in OrderedDict)
        lru_model_id = next(iter(self.cache))
        self._remove_model(lru_model_id, reason=reason)

        self.cache_stats["evictions"] += 1
        if "memory" in reason.lower():
            self.cache_stats["memory_evictions"] += 1

    def _remove_model(self, model_id: str, reason: str = "Unknown") -> bool:
        """Remove a model from cache."""
        if model_id not in self.cache:
            return False

        self.cache[model_id]
        memory_mb = self.memory_usage.get(model_id, 0.0)

        # Remove from all tracking structures
        del self.cache[model_id]
        del self.access_times[model_id]
        if model_id in self.memory_usage:
            del self.memory_usage[model_id]

        # Update statistics
        self.cache_stats["current_size"] = len(self.cache)
        self.cache_stats["current_memory_mb"] = sum(self.memory_usage.values())

        self._logger.debug(
            "Model removed from cache",
            model_id=model_id,
            reason=reason,
            memory_mb=memory_mb,
            remaining_models=len(self.cache),
        )

        return True

    def _estimate_model_memory(self, model: Any) -> float:
        """
        Estimate memory usage of a model in MB.

        Args:
            model: Model to estimate

        Returns:
            Estimated memory usage in MB
        """
        try:
            # Get current process memory
            process = psutil.Process()
            process.memory_info().rss

            # This is a rough estimation
            # In practice, you'd want more sophisticated memory estimation
            base_memory_mb = 10.0  # Base overhead

            # Estimate based on model complexity
            if hasattr(model, "model") and model.model is not None:
                # For sklearn models, estimate based on parameters
                if hasattr(model.model, "n_features_in_"):
                    feature_memory = model.model.n_features_in_ * 0.001  # 1KB per feature
                    base_memory_mb += feature_memory

                # Add memory for different model types
                model_class = model.model.__class__.__name__.lower()
                if "forest" in model_class or "tree" in model_class:
                    base_memory_mb += 20.0  # Tree-based models are memory intensive
                elif "svm" in model_class:
                    base_memory_mb += 15.0
                elif "neural" in model_class or "mlp" in model_class:
                    base_memory_mb += 25.0
                else:
                    base_memory_mb += 5.0

            return base_memory_mb

        except Exception as e:
            self._logger.warning(f"Failed to estimate model memory: {e}")
            return 10.0  # Default estimate

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        return self.cache_stats["hits"] / total if total > 0 else 0.0

    def _cleanup_expired_models(self) -> int:
        """Clean up expired models and return count of removed models."""
        if self.ttl_minutes <= 0:
            return 0

        expired_models = []
        cutoff_time = datetime.utcnow() - timedelta(minutes=self.ttl_minutes)

        with self.lock:
            for model_id, access_time in self.access_times.items():
                if access_time < cutoff_time:
                    expired_models.append(model_id)

            for model_id in expired_models:
                self._remove_model(model_id, reason="TTL cleanup")

        if expired_models:
            self._logger.debug(f"Cleaned up {len(expired_models)} expired models")

        return len(expired_models)

    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._cleanup_running:
            try:
                # Clean up expired models
                self._cleanup_expired_models()

                # Check memory pressure
                self._check_memory_pressure()

                # Sleep for configured interval
                time.sleep(self.mc_config.cleanup_interval_seconds)

            except Exception as e:
                self._logger.error(f"Cache cleanup error: {e}")
                time.sleep(10)

    def _check_memory_pressure(self) -> None:
        """Check system memory pressure and evict models if needed."""
        try:
            # Check system memory
            system_memory = psutil.virtual_memory()

            # If system memory usage exceeds threshold, start aggressive eviction
            if system_memory.percent > self.mc_config.memory_pressure_threshold:
                with self.lock:
                    models_to_evict = max(1, len(self.cache) // 4)  # Evict 25% of models

                    for _ in range(models_to_evict):
                        if self.cache:
                            self._evict_lru_model("System memory pressure")

                self._logger.warning(
                    "High system memory pressure, evicted models",
                    system_memory_pct=system_memory.percent,
                    models_evicted=models_to_evict,
                )

        except Exception as e:
            self._logger.warning(f"Failed to check memory pressure: {e}")

    def health_check(self) -> dict[str, Any]:
        """Perform health check of the cache."""
        with self.lock:
            health = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "memory_usage_mb": sum(self.memory_usage.values()),
                "max_memory_mb": self.max_memory_mb,
                "hit_rate": self._calculate_hit_rate(),
                "cleanup_running": self._cleanup_running,
            }

            # Check for issues
            if len(self.cache) >= self.max_size:
                health["status"] = "warning"
                health["warning"] = "Cache at maximum capacity"

            if sum(self.memory_usage.values()) > self.max_memory_mb * 0.9:
                health["status"] = "warning"
                health["warning"] = "High memory usage"

            if self._calculate_hit_rate() < 0.5 and self.cache_stats["total_requests"] > 100:
                health["status"] = "warning"
                health["warning"] = "Low cache hit rate"

            return health

    def __enter__(self):
        """Context manager entry."""
        self.start_cleanup_thread()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_cleanup_thread()

    # Service Health and Metrics
    async def _service_health_check(self) -> "HealthStatus":
        """Model cache service specific health check."""
        from src.core.base.interfaces import HealthStatus

        try:
            # Check cache state
            with self.lock:
                # Check if cache is at maximum capacity
                if len(self.cache) >= self.max_size:
                    return HealthStatus.DEGRADED

                # Check memory usage
                if sum(self.memory_usage.values()) > self.max_memory_mb * 0.9:
                    return HealthStatus.DEGRADED

                # Check hit rate
                if self._calculate_hit_rate() < 0.5 and self.cache_stats["total_requests"] > 100:
                    return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            self._logger.error("Model cache service health check failed", error=str(e))
            return HealthStatus.UNHEALTHY

    def get_model_cache_metrics(self) -> dict[str, Any]:
        """Get model cache service metrics."""
        with self.lock:
            stats = self.get_cache_stats()
            stats.update(
                {
                    "cleanup_thread_running": self._cleanup_running,
                    "memory_monitoring_enabled": self.mc_config.enable_memory_monitoring,
                    "cleanup_interval_seconds": self.mc_config.cleanup_interval_seconds,
                    "memory_pressure_threshold": self.mc_config.memory_pressure_threshold,
                }
            )
            return stats

    # Configuration validation
    def _validate_service_config(self, config: ConfigDict) -> bool:
        """Validate model cache service configuration."""
        try:
            mc_config_dict = config.get("model_cache", {})
            ModelCacheConfig(**mc_config_dict)
            return True
        except Exception as e:
            self._logger.error("Model cache service configuration validation failed", error=str(e))
            return False
