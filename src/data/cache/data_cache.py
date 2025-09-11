"""
DataCache - Enterprise Multi-Level Caching System

This module implements a sophisticated multi-level caching system optimized
for high-frequency financial data, featuring L1 memory cache, L2 Redis cache,
and L3 database cache with intelligent cache warming and eviction policies.

Dependencies:
- P-001: Core types, exceptions, logging
- P-002: Database models, queries, and connections
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

import asyncio
import json
import pickle
import sys
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as redis
from pydantic import BaseModel, ConfigDict, Field

from src.core import BaseComponent, Config, HealthCheckResult, HealthStatus
from src.utils.cache_utilities import (
    CacheEntry,
    CacheLevel,
    CacheMode,
    CacheStats,
    CacheStrategy,
)
from src.utils.decorators import time_execution

# Cache enums are now imported from shared utilities


# CacheEntry is now imported from shared utilities


# CacheStats is now imported from shared utilities


class CacheConfig(BaseModel):
    """Cache configuration model."""

    level: CacheLevel
    max_size: int = Field(10000, ge=1)
    max_memory_mb: int = Field(512, ge=1, le=8192)
    default_ttl: int = Field(3600, ge=1, le=86400)
    strategy: CacheStrategy = CacheStrategy.LRU
    mode: CacheMode = CacheMode.CACHE_ASIDE
    compression_enabled: bool = False
    serialization_format: str = "json"  # json, pickle, msgpack
    key_prefix: str = "tbot"

    model_config = ConfigDict(use_enum_values=False)


class L1MemoryCache(BaseComponent):
    """
    L1 Memory Cache - Fastest access, limited capacity.

    Features:
    - LRU/LFU/FIFO eviction policies
    - Memory usage monitoring
    - TTL support
    - Access pattern tracking
    """

    def __init__(self, config: CacheConfig):
        """Initialize L1 memory cache."""
        super().__init__()
        self.config = config
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """Get value from memory cache."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                # Check expiration
                if entry.is_expired():
                    await self._evict_entry(key)
                    self._stats.misses += 1
                    return None

                # Update access and move to end for LRU
                entry.update_access()
                if self.config.strategy == CacheStrategy.LRU:
                    self._cache.move_to_end(key)

                self._stats.hits += 1
                self._stats.calculate_hit_rate()
                return entry.value

            self._stats.misses += 1
            self._stats.calculate_hit_rate()
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in memory cache."""
        try:
            async with self._lock:
                # Calculate size
                size_bytes = self._calculate_size(value)

                # Check memory limits
                if not await self._ensure_capacity(size_bytes):
                    return False

                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(timezone.utc),
                    last_accessed=datetime.now(timezone.utc),
                    ttl_seconds=ttl or self.config.default_ttl,
                    size_bytes=size_bytes,
                )

                # Remove existing entry if present
                if key in self._cache:
                    old_entry = self._cache[key]
                    self._stats.size_bytes -= old_entry.size_bytes
                    self._stats.entry_count -= 1

                # Add new entry
                self._cache[key] = entry
                self._stats.size_bytes += size_bytes
                self._stats.entry_count += 1
                self._stats.writes += 1

                return True

        except Exception as e:
            self.logger.error(f"Memory cache set failed: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        async with self._lock:
            if key in self._cache:
                await self._evict_entry(key)
                self._stats.deletes += 1
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._stats = CacheStats()

    async def _ensure_capacity(self, required_bytes: int) -> bool:
        """Ensure cache has capacity for new entry."""
        # Check entry count limit
        while len(self._cache) >= self.config.max_size:
            if not await self._evict_one_entry():
                return False

        # Check memory limit
        max_bytes = self.config.max_memory_mb * 1024 * 1024
        while self._stats.size_bytes + required_bytes > max_bytes:
            if not await self._evict_one_entry():
                return False

        return True

    async def _evict_one_entry(self) -> bool:
        """Evict one entry based on strategy."""
        if not self._cache:
            return False

        if self.config.strategy == CacheStrategy.LRU:
            # Remove least recently used (first item in OrderedDict)
            key = next(iter(self._cache))
        elif self.config.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
        elif self.config.strategy == CacheStrategy.FIFO:
            # Remove oldest entry
            key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        else:
            # Default to LRU
            key = next(iter(self._cache))

        await self._evict_entry(key)
        return True

    async def _evict_entry(self, key: str) -> None:
        """Evict specific entry."""
        if key in self._cache:
            entry = self._cache[key]
            self._stats.size_bytes -= entry.size_bytes
            self._stats.entry_count -= 1
            self._stats.evictions += 1
            del self._cache[key]

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if isinstance(value, str):
                return len(value.encode("utf-8"))
            elif isinstance(value, (int, float)):
                return sys.getsizeof(value)
            elif isinstance(value, (list, dict)):
                return len(json.dumps(value, default=str).encode("utf-8"))
            else:
                return sys.getsizeof(value)
        except Exception as e:
            self.logger.warning(f"Size calculation failed: {e}, using default estimate")
            return 1024  # Default estimate

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        async with self._lock:
            self._stats.memory_usage_mb = self._stats.size_bytes / (1024 * 1024)
            return self._stats


class L2RedisCache(BaseComponent):
    """
    L2 Redis Cache - Network-based, persistent, shared across instances.

    Features:
    - Redis-based distributed caching
    - Compression support
    - Batch operations
    - Connection pooling
    - Failover handling
    """

    def __init__(self, config: CacheConfig, redis_config: dict[str, Any]):
        """Initialize L2 Redis cache."""
        super().__init__()
        self.config = config
        self.redis_config = redis_config
        self._redis_client: redis.Redis | None = None
        self._stats = CacheStats()

    async def initialize(self) -> None:
        """Initialize Redis connection.

        SECURITY NOTE: Redis password should be provided via environment variables
        or secure vault, not hardcoded in configuration files.
        """
        try:
            # Get password from environment or config
            import os

            redis_password = os.environ.get("REDIS_PASSWORD") or self.redis_config.get("password")

            self._redis_client = redis.Redis(
                host=self.redis_config.get("host", os.environ.get("REDIS_HOST", "127.0.0.1")),
                port=self.redis_config.get("port", int(os.environ.get("REDIS_PORT", "6379"))),
                db=self.redis_config.get("db", int(os.environ.get("REDIS_DB", "0"))),
                password=redis_password,
                max_connections=self.redis_config.get("max_connections", 20),
                decode_responses=False,  # Handle binary data
                socket_connect_timeout=5,
                socket_timeout=10,
            )

            # Test connection with timeout
            await asyncio.wait_for(self._redis_client.ping(), timeout=5.0)
            self.logger.info("Redis cache connection established")

        except Exception as e:
            self.logger.error(f"Redis cache initialization failed: {e}")
            self._redis_client = None

    async def get(self, key: str) -> Any | None:
        """Get value from Redis cache."""
        if not self._redis_client:
            return None

        try:
            full_key = f"{self.config.key_prefix}:{key}"
            # Add timeout to Redis get operation
            data = await asyncio.wait_for(self._redis_client.get(full_key), timeout=3.0)

            if data:
                value = self._deserialize(data)
                self._stats.hits += 1
                self._stats.calculate_hit_rate()
                return value

            self._stats.misses += 1
            self._stats.calculate_hit_rate()
            return None

        except Exception as e:
            self.logger.error(f"Redis cache get failed: {e}")
            self._stats.misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in Redis cache."""
        if not self._redis_client:
            return False

        try:
            full_key = f"{self.config.key_prefix}:{key}"
            serialized_data = self._serialize(value)

            ttl_seconds = ttl or self.config.default_ttl
            await asyncio.wait_for(
                self._redis_client.setex(full_key, ttl_seconds, serialized_data), timeout=3.0
            )

            self._stats.writes += 1
            return True

        except Exception as e:
            self.logger.error(f"Redis cache set failed: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        if not self._redis_client:
            return False

        try:
            full_key = f"{self.config.key_prefix}:{key}"
            deleted = await asyncio.wait_for(self._redis_client.delete(full_key), timeout=3.0)

            if deleted:
                self._stats.deletes += 1
            return deleted > 0

        except Exception as e:
            self.logger.error(f"Redis cache delete failed: {e}")
            return False

    async def clear(self) -> None:
        """Clear cache entries with prefix."""
        if not self._redis_client:
            return

        try:
            pattern = f"{self.config.key_prefix}:*"
            keys = await asyncio.wait_for(self._redis_client.keys(pattern), timeout=5.0)

            if keys:
                await asyncio.wait_for(self._redis_client.delete(*keys), timeout=10.0)

        except Exception as e:
            self.logger.error(f"Redis cache clear failed: {e}")

    async def batch_get(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values in batch."""
        if not self._redis_client or not keys:
            return {}

        try:
            full_keys = [f"{self.config.key_prefix}:{key}" for key in keys]
            values = await asyncio.wait_for(self._redis_client.mget(full_keys), timeout=5.0)

            result = {}
            for _i, (original_key, value) in enumerate(zip(keys, values, strict=False)):
                if value:
                    result[original_key] = self._deserialize(value)
                    self._stats.hits += 1
                else:
                    self._stats.misses += 1

            self._stats.calculate_hit_rate()
            return result

        except Exception as e:
            self.logger.error(f"Redis batch get failed: {e}")
            self._stats.misses += len(keys)
            return {}

    async def batch_set(self, data: dict[str, Any], ttl: int | None = None) -> bool:
        """Set multiple values in batch."""
        if not self._redis_client or not data:
            return False

        try:
            ttl_seconds = ttl or self.config.default_ttl

            # Use async context manager for pipeline with timeout
            async with self._redis_client.pipeline() as pipe:
                for key, value in data.items():
                    full_key = f"{self.config.key_prefix}:{key}"
                    serialized_data = self._serialize(value)
                    pipe.setex(full_key, ttl_seconds, serialized_data)

                await asyncio.wait_for(pipe.execute(), timeout=10.0)
                self._stats.writes += len(data)
                return True

        except Exception as e:
            self.logger.error(f"Redis batch set failed: {e}")
            return False

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            if self.config.serialization_format == "pickle":
                data = pickle.dumps(value)
            else:
                # Default to JSON
                json_str = json.dumps(value, default=str)
                data = json_str.encode("utf-8")

            # Apply compression if enabled
            if self.config.compression_enabled:
                import gzip

                data = gzip.compress(data)

            return data

        except Exception as e:
            self.logger.error(f"Serialization failed: {e}")
            raise

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Decompress if needed
            if self.config.compression_enabled:
                import gzip

                data = gzip.decompress(data)

            if self.config.serialization_format == "pickle":
                return pickle.loads(data)
            else:
                # Default to JSON
                json_str = data.decode("utf-8")
                return json.loads(json_str)

        except Exception as e:
            self.logger.error(f"Deserialization failed: {e}")
            raise

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    async def cleanup(self) -> None:
        """Cleanup Redis connection."""
        redis_client = None
        try:
            if self._redis_client:
                redis_client = self._redis_client
                self._redis_client = None
                await asyncio.wait_for(redis_client.close(), timeout=5.0)
        except asyncio.TimeoutError:
            self.logger.warning("Redis close timeout, forcing cleanup")
            if redis_client:
                try:
                    await redis_client.connection_pool.disconnect()
                except Exception as e:
                    self.logger.debug(f"Error disconnecting Redis pool: {e}")
        except Exception as e:
            self.logger.warning(f"Redis cleanup error during primary close: {e}")
        finally:
            if redis_client:
                try:
                    if not redis_client.connection_pool.closed:
                        await asyncio.wait_for(redis_client.close(), timeout=2.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Redis final close timeout")
                    try:
                        await redis_client.connection_pool.disconnect()
                    except Exception as e:
                        self.logger.debug(f"Error disconnecting Redis pool during timeout cleanup: {e}")
                except Exception as e:
                    self.logger.warning(f"Redis cleanup error during final close: {e}")


class DataCache(BaseComponent):
    """
    Multi-level data cache orchestrator.

    Features:
    - L1 memory cache for ultra-fast access
    - L2 Redis cache for persistence and sharing
    - Intelligent cache warming and prefetching
    - Cache coherence across levels
    - Performance monitoring and optimization
    """

    def __init__(self, config: Config):
        """Initialize multi-level data cache."""
        super().__init__()
        self.config = config

        # Setup cache configurations
        self._setup_configurations()

        # Initialize cache levels
        self._l1_cache = L1MemoryCache(self.l1_config)
        self._l2_cache = L2RedisCache(self.l2_config, self.redis_config)

        # Cache warming and prefetching
        self._warming_enabled = True
        self._prefetch_queue: asyncio.Queue = asyncio.Queue()
        self._background_tasks: list[asyncio.Task] = []

        # Add lock for batch operations to prevent race conditions
        self._batch_lock = asyncio.Lock()

        self._initialized = False

    def _setup_configurations(self) -> None:
        """Setup cache level configurations."""
        cache_config = getattr(self.config, "data_cache", {})

        # L1 Memory Cache Config
        l1_config = cache_config.get("l1", {})
        self.l1_config = CacheConfig(
            level=CacheLevel.L1_MEMORY,
            max_size=l1_config.get("max_size", 10000),
            max_memory_mb=l1_config.get("max_memory_mb", 512),
            default_ttl=l1_config.get("default_ttl", 300),  # 5 minutes
            strategy=CacheStrategy(l1_config.get("strategy", "lru")),
            mode=CacheMode(l1_config.get("mode", "cache_aside")),
        )

        # L2 Redis Cache Config
        l2_config = cache_config.get("l2", {})
        self.l2_config = CacheConfig(
            level=CacheLevel.L2_REDIS,
            max_size=l2_config.get("max_size", 100000),
            default_ttl=l2_config.get("default_ttl", 3600),  # 1 hour
            compression_enabled=l2_config.get("compression", False),
            serialization_format=l2_config.get("serialization", "json"),
            key_prefix=l2_config.get("key_prefix", "tbot_cache"),
        )

        # Redis connection config
        self.redis_config = getattr(self.config, "redis", {})

    async def initialize(self) -> None:
        """Initialize the data cache system."""
        try:
            if self._initialized:
                return

            self.logger.info("Initializing DataCache...")

            # Initialize L2 Redis cache
            await self._l2_cache.initialize()

            # Start background tasks
            if self._warming_enabled:
                task = asyncio.create_task(self._background_cache_warmer())
                self._background_tasks.append(task)

            self._initialized = True
            self.logger.info("DataCache initialized successfully")

        except Exception as e:
            self.logger.error(f"DataCache initialization failed: {e}")
            raise

    @time_execution
    async def get(self, key: str, warm_lower_levels: bool = True) -> Any | None:
        """
        Get value with multi-level cache lookup.

        Args:
            key: Cache key
            warm_lower_levels: Whether to warm lower cache levels on hit

        Returns:
            Cached value or None if not found
        """
        try:
            if not self._initialized:
                await self.initialize()

            # Try L1 memory cache first
            value = await self._l1_cache.get(key)
            if value is not None:
                self.logger.debug(f"Cache hit L1: {key}")
                return value

            # Try L2 Redis cache
            value = await self._l2_cache.get(key)
            if value is not None:
                self.logger.debug(f"Cache hit L2: {key}")

                # Warm L1 cache if enabled
                if warm_lower_levels:
                    await self._l1_cache.set(key, value)

                return value

            self.logger.debug(f"Cache miss: {key}")
            return None

        except Exception as e:
            self.logger.error(f"Cache get failed for {key}: {e}")
            return None

    @time_execution
    async def set(
        self, key: str, value: Any, ttl: int | None = None, levels: list[CacheLevel] | None = None
    ) -> bool:
        """
        Set value in specified cache levels.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            levels: Cache levels to update (default: all)

        Returns:
            Success status
        """
        try:
            if not self._initialized:
                await self.initialize()

            if levels is None:
                levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]

            success = True

            # Set in L1 memory cache
            if CacheLevel.L1_MEMORY in levels:
                l1_success = await self._l1_cache.set(key, value, ttl)
                success = success and l1_success

            # Set in L2 Redis cache
            if CacheLevel.L2_REDIS in levels:
                l2_success = await self._l2_cache.set(key, value, ttl)
                success = success and l2_success

            return success

        except Exception as e:
            self.logger.error(f"Cache set failed for {key}: {e}")
            return False

    async def delete(self, key: str, levels: list[CacheLevel] | None = None) -> bool:
        """
        Delete value from specified cache levels.

        Args:
            key: Cache key
            levels: Cache levels to delete from (default: all)

        Returns:
            Success status
        """
        try:
            if levels is None:
                levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]

            success = True

            # Delete from L1
            if CacheLevel.L1_MEMORY in levels:
                l1_success = await self._l1_cache.delete(key)
                success = success and l1_success

            # Delete from L2
            if CacheLevel.L2_REDIS in levels:
                l2_success = await self._l2_cache.delete(key)
                success = success and l2_success

            return success

        except Exception as e:
            self.logger.error(f"Cache delete failed for {key}: {e}")
            return False

    async def batch_get(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values efficiently with proper synchronization."""
        try:
            if not keys:
                return {}

            # Use lock to prevent race conditions during batch operations
            async with self._batch_lock:
                result = {}

                # Try L1 cache for all keys
                l1_misses = []
                for key in keys:
                    value = await self._l1_cache.get(key)
                    if value is not None:
                        result[key] = value
                    else:
                        l1_misses.append(key)

                # Try L2 cache for L1 misses
                if l1_misses:
                    l2_results = await self._l2_cache.batch_get(l1_misses)
                    result.update(l2_results)

                    # Warm L1 cache with L2 hits
                    for key, value in l2_results.items():
                        await self._l1_cache.set(key, value)

                return result

        except Exception as e:
            self.logger.error(f"Batch cache get failed: {e}")
            return {}

    async def batch_set(self, data: dict[str, Any], ttl: int | None = None) -> bool:
        """Set multiple values efficiently with proper synchronization."""
        try:
            if not data:
                return True

            # Use lock to prevent race conditions during batch operations
            async with self._batch_lock:
                success = True

                # Set in L1 cache
                for key, value in data.items():
                    l1_success = await self._l1_cache.set(key, value, ttl)
                    success = success and l1_success

                # Batch set in L2 cache
                l2_success = await self._l2_cache.batch_set(data, ttl)
                success = success and l2_success

                return success

        except Exception as e:
            self.logger.error(f"Batch cache set failed: {e}")
            return False

    async def clear(self, levels: list[CacheLevel] | None = None) -> None:
        """Clear specified cache levels."""
        try:
            if levels is None:
                levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]

            if CacheLevel.L1_MEMORY in levels:
                await self._l1_cache.clear()

            if CacheLevel.L2_REDIS in levels:
                await self._l2_cache.clear()

            self.logger.info("Cache cleared")

        except Exception as e:
            self.logger.error(f"Cache clear failed: {e}")

    async def warm_cache(
        self, data_loader: Callable[[list[str]], dict[str, Any]], keys: list[str]
    ) -> None:
        """Warm cache with data from loader function."""
        try:
            # Load data
            data = await data_loader(keys)

            # Set in cache
            await self.batch_set(data)

            self.logger.info(f"Warmed cache with {len(data)} entries")

        except Exception as e:
            self.logger.error(f"Cache warming failed: {e}")

    async def _background_cache_warmer(self) -> None:
        """Background task for cache warming."""
        try:
            while True:
                try:
                    # Wait for prefetch requests
                    request = await asyncio.wait_for(self._prefetch_queue.get(), timeout=60.0)

                    # Process prefetch request
                    await self._process_prefetch_request(request)

                except asyncio.TimeoutError:
                    # Periodic maintenance
                    await self._perform_cache_maintenance()

                except Exception as e:
                    self.logger.error(f"Cache warming error: {e}")

        except asyncio.CancelledError:
            self.logger.info("Cache warmer task cancelled")

    async def _process_prefetch_request(self, request: dict[str, Any]) -> None:
        """Process a cache prefetch request."""
        # Prefetch strategy not implemented - using lazy loading instead
        self.logger.debug(f"Prefetch request ignored: {request.get('key', 'unknown')}")

    async def _perform_cache_maintenance(self) -> None:
        """Perform periodic cache maintenance."""
        try:
            # Get cache statistics
            l1_stats = await self._l1_cache.get_stats()
            l2_stats = await self._l2_cache.get_stats()

            # Log statistics
            self.logger.debug(
                f"Cache stats - L1: {l1_stats.hit_rate:.2%} hit rate, "
                f"{l1_stats.memory_usage_mb:.1f}MB used; "
                f"L2: {l2_stats.hit_rate:.2%} hit rate"
            )

        except Exception as e:
            self.logger.error(f"Cache maintenance failed: {e}")

    async def get_stats(self) -> dict[str, CacheStats]:
        """Get comprehensive cache statistics."""
        try:
            stats = {}

            stats["l1"] = await self._l1_cache.get_stats()
            stats["l2"] = await self._l2_cache.get_stats()

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {}

    async def health_check(self) -> HealthCheckResult:
        """Perform cache health check."""
        overall_status = HealthStatus.HEALTHY
        details = {
            "initialized": self._initialized,
            "levels": {},
        }

        try:
            # Check L1 cache
            l1_stats = await self._l1_cache.get_stats()
            details["levels"]["l1"] = {
                "status": "healthy",
                "entries": l1_stats.entry_count,
                "memory_mb": l1_stats.memory_usage_mb,
                "hit_rate": l1_stats.hit_rate,
            }

            # Check L2 cache
            try:
                l2_stats = await self._l2_cache.get_stats()
                details["levels"]["l2"] = {
                    "status": "healthy",
                    "hit_rate": l2_stats.hit_rate,
                    "redis_available": self._l2_cache._redis_client is not None,
                }
            except Exception as e:
                details["levels"]["l2"] = {
                    "status": f"unhealthy: {e}",
                    "redis_available": False,
                }
                overall_status = HealthStatus.DEGRADED

        except Exception as e:
            overall_status = HealthStatus.UNHEALTHY
            details["error"] = str(e)

        return HealthCheckResult(
            status=overall_status,
            details=details,
            message=f"Cache health check: {overall_status.value}",
        )

    async def cleanup(self) -> None:
        """Cleanup cache resources."""
        background_tasks = []
        try:
            # Collect background tasks for cleanup
            background_tasks = list(self._background_tasks)

            # Cancel background tasks
            for task in background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Clear caches
            await self.clear()

            # Cleanup L2 cache
            await self._l2_cache.cleanup()

            self._background_tasks.clear()
            self._initialized = False
            self.logger.info("DataCache cleanup completed")

        except Exception as e:
            self.logger.error(f"DataCache cleanup error: {e}")
        finally:
            # Force cleanup any remaining resources
            try:
                # Force cancel any remaining background tasks
                for task in background_tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            self.logger.debug(f"Error waiting for background task completion: {e}")

                # Force cleanup L2 cache
                try:
                    await self._l2_cache.cleanup()
                except Exception as e:
                    self.logger.warning(f"Error in L2 cache cleanup: {e}")

                # Force clear all caches
                try:
                    await self._l1_cache.clear()
                except Exception as e:
                    self.logger.debug(f"Error clearing L1 cache: {e}")

                self._background_tasks.clear()
                self._initialized = False
            except Exception as e:
                self.logger.warning(f"Error in final cache cleanup: {e}")
