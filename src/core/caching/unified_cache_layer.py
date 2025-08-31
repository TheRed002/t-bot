"""
Unified Caching Layer for T-Bot Trading System

This module provides a comprehensive, multi-level caching system that integrates
all caching strategies across the trading system for optimal performance:

- L1 CPU Cache: Ultra-fast in-memory caching for hot data
- L2 Application Cache: Application-level memory caching with LRU eviction
- L3 Redis Cache: Distributed caching for shared data across instances
- L4 Database Cache: Query result caching with intelligent invalidation
- Smart Cache Warming: Predictive cache population based on trading patterns
- Cache Coherence: Automatic invalidation and synchronization across levels

Performance targets:
- L1 cache access: < 1ms
- L2 cache access: < 5ms
- L3 cache access: < 10ms
- Cache hit ratio: > 90% for trading data
- Cache coherence latency: < 50ms
"""

import asyncio
import json
import pickle
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pickle import PicklingError
from typing import Any

import redis.asyncio as redis
from cachetools import TTLCache  # type: ignore

from src.core.base.component import BaseComponent
from src.core.config import Config
from src.core.exceptions import CacheError
from src.core.logging import get_logger
from src.data.cache.data_cache import DataCache
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class CacheLevel(Enum):
    """Cache levels in the hierarchy."""

    L1_CPU = "l1_cpu"  # CPU cache-friendly data structures
    L2_MEMORY = "l2_memory"  # Application memory cache
    L3_REDIS = "l3_redis"  # Distributed Redis cache
    L4_DATABASE = "l4_database"  # Database query cache


class CacheStrategy(Enum):
    """Cache management strategies."""

    WRITE_THROUGH = "write_through"  # Write to cache and storage simultaneously
    WRITE_BEHIND = "write_behind"  # Write to cache immediately, storage asynchronously
    WRITE_AROUND = "write_around"  # Write only to storage, bypass cache
    READ_THROUGH = "read_through"  # Read from cache, load from storage on miss
    CACHE_ASIDE = "cache_aside"  # Application manages cache explicitly


class DataCategory(Enum):
    """Categories of data for cache optimization."""

    MARKET_DATA = "market_data"  # Real-time market prices, orderbooks
    TRADING_DATA = "trading_data"  # Orders, positions, trades
    USER_DATA = "user_data"  # User profiles, preferences, sessions
    ANALYTICS_DATA = "analytics_data"  # Performance metrics, reports
    STATIC_DATA = "static_data"  # Configuration, symbols, exchange info
    ML_DATA = "ml_data"  # Model predictions, features


@dataclass
class CachePolicy:
    """Caching policy for different data categories."""

    category: DataCategory
    levels: list[CacheLevel]
    ttl_seconds: int
    max_size: int
    strategy: CacheStrategy
    compression: bool = False
    encryption: bool = False
    warm_on_startup: bool = False
    invalidate_on_write: bool = True


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata."""

    key: str
    value: Any
    level: CacheLevel
    category: DataCategory
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    ttl_seconds: int = 3600
    size_bytes: int = 0
    version: int = 1
    dependencies: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds <= 0:
            return False
        return (datetime.now(timezone.utc) - self.created_at).total_seconds() > self.ttl_seconds

    def touch(self) -> None:
        """Update access information."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1


@dataclass
class CacheStats:
    """Comprehensive cache statistics."""

    level: CacheLevel
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    avg_access_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    error_count: int = 0


class CacheInterface(ABC):
    """Abstract interface for cache implementations."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


class L1CPUCache(CacheInterface):
    """
    L1 CPU Cache - Ultra-fast cache optimized for CPU cache efficiency.

    Uses memory-efficient data structures and access patterns optimized
    for modern CPU cache hierarchies.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats(level=CacheLevel.L1_CPU)

        # CPU cache optimization: pre-allocate arrays for hot keys
        self._hot_keys: list[str] = [""] * 100  # Top 100 hot keys
        self._hot_values: list[Any] = [None] * 100
        self._hot_timestamps: list[float] = [0.0] * 100
        self._hot_index = 0

    async def get(self, key: str) -> Any | None:
        """Get from CPU-optimized cache."""
        start_time = time.perf_counter()

        try:
            # Check hot keys first (CPU cache friendly)
            for i in range(100):
                if self._hot_keys[i] == key:
                    current_time = time.perf_counter()
                    if current_time - self._hot_timestamps[i] < 10:  # 10 second TTL for hot data
                        self._stats.hits += 1
                        self._update_access_time(start_time)
                        return self._hot_values[i]
                    else:
                        # Expired hot key
                        self._hot_keys[i] = ""
                        self._hot_values[i] = None
                        break

            # Check main cache
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired():
                    entry.touch()
                    self._cache.move_to_end(key)  # LRU

                    # Promote to hot cache if frequently accessed
                    if entry.access_count > 10:
                        self._promote_to_hot_cache(key, entry.value)

                    self._stats.hits += 1
                    self._update_access_time(start_time)
                    return entry.value
                else:
                    del self._cache[key]
                    self._stats.evictions += 1

            self._stats.misses += 1
            self._update_access_time(start_time)
            return None

        except Exception as e:
            self._stats.error_count += 1
            logger.error(f"L1 cache get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set in CPU-optimized cache."""
        try:
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                level=CacheLevel.L1_CPU,
                category=DataCategory.MARKET_DATA,  # Default for L1
                ttl_seconds=ttl or 60,  # Short TTL for L1
                size_bytes=self._estimate_size(value),
            )

            # Evict if necessary
            while len(self._cache) >= self.max_size:
                self._evict_lru()

            self._cache[key] = entry
            self._update_stats()

            return True

        except Exception as e:
            self._stats.error_count += 1
            logger.error(f"L1 cache set error: {e}")
            return False

    def _promote_to_hot_cache(self, key: str, value: Any) -> None:
        """Promote frequently accessed data to hot cache."""
        self._hot_keys[self._hot_index] = key
        self._hot_values[self._hot_index] = value
        self._hot_timestamps[self._hot_index] = time.perf_counter()
        self._hot_index = (self._hot_index + 1) % 100

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            self._cache.popitem(last=False)
            self._stats.evictions += 1

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value."""
        if isinstance(value, str | bytes):
            return len(value)
        elif isinstance(value, int | float):
            return 8
        else:
            return len(str(value))

    def _update_access_time(self, start_time: float) -> None:
        """Update average access time."""
        access_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        self._stats.avg_access_time_ms = 0.1 * access_time + 0.9 * self._stats.avg_access_time_ms

    def _update_stats(self) -> None:
        """Update cache statistics."""
        self._stats.entry_count = len(self._cache)
        self._stats.size_bytes = sum(entry.size_bytes for entry in self._cache.values())
        self._stats.memory_usage_mb = self._stats.size_bytes / (1024 * 1024)

        total_requests = self._stats.hits + self._stats.misses
        self._stats.hit_rate = self._stats.hits / total_requests if total_requests > 0 else 0

    async def delete(self, key: str) -> bool:
        """Delete from cache."""
        # Remove from hot cache
        for i in range(100):
            if self._hot_keys[i] == key:
                self._hot_keys[i] = ""
                self._hot_values[i] = None
                break

        # Remove from main cache
        if key in self._cache:
            del self._cache[key]
            self._update_stats()
            return True
        return False

    async def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._hot_keys = [""] * 100
        self._hot_values = [None] * 100
        self._hot_timestamps = [0.0] * 100
        self._update_stats()

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self._update_stats()
        return self._stats


class L2MemoryCache(CacheInterface):
    """
    L2 Memory Cache - Application-level memory cache with advanced features.

    Features:
    - TTL-based expiration
    - Size-based eviction
    - Category-aware management
    - Dependency tracking
    """

    def __init__(self, max_size: int = 10000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._caches: dict[DataCategory, TTLCache] = {}
        self._stats = CacheStats(level=CacheLevel.L2_MEMORY)
        self._dependencies: dict[str, set[str]] = defaultdict(set)

        # Initialize category-specific caches
        for category in DataCategory:
            cache_size = self._get_category_cache_size(category)
            self._caches[category] = TTLCache(
                maxsize=cache_size, ttl=self._get_category_ttl(category)
            )

    def _get_category_cache_size(self, category: DataCategory) -> int:
        """Get cache size for category."""
        sizes = {
            DataCategory.MARKET_DATA: 5000,  # High-frequency updates
            DataCategory.TRADING_DATA: 2000,
            DataCategory.USER_DATA: 1000,
            DataCategory.ANALYTICS_DATA: 1500,
            DataCategory.STATIC_DATA: 500,
            DataCategory.ML_DATA: 1000,
        }
        return sizes.get(category, 1000)

    def _get_category_ttl(self, category: DataCategory) -> int:
        """Get TTL for category."""
        ttls = {
            DataCategory.MARKET_DATA: 30,  # 30 seconds for market data
            DataCategory.TRADING_DATA: 300,  # 5 minutes for trading data
            DataCategory.USER_DATA: 3600,  # 1 hour for user data
            DataCategory.ANALYTICS_DATA: 1800,  # 30 minutes for analytics
            DataCategory.STATIC_DATA: 86400,  # 24 hours for static data
            DataCategory.ML_DATA: 600,  # 10 minutes for ML data
        }
        return ttls.get(category, 300)

    async def get(self, key: str) -> Any | None:
        """Get from L2 cache."""
        start_time = time.perf_counter()

        try:
            # Try to find in any category cache
            for _category, cache in self._caches.items():
                if key in cache:
                    value = cache[key]
                    self._stats.hits += 1
                    self._update_access_time(start_time)
                    return value

            self._stats.misses += 1
            self._update_access_time(start_time)
            return None

        except Exception as e:
            self._stats.error_count += 1
            logger.error(f"L2 cache get error: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        category: DataCategory = DataCategory.TRADING_DATA,
    ) -> bool:
        """Set in L2 cache."""
        try:
            cache = self._caches[category]

            # Check memory usage
            estimated_size = self._estimate_size(value)
            if self._stats.size_bytes + estimated_size > self.max_memory_bytes:
                await self._evict_by_memory()

            # Set TTL if provided
            if ttl:
                cache.ttl = ttl

            cache[key] = value
            self._stats.size_bytes += estimated_size
            self._update_stats()

            return True

        except Exception as e:
            self._stats.error_count += 1
            logger.error(f"L2 cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete from L2 cache."""
        deleted = False
        for _category, cache in self._caches.items():
            if key in cache:
                del cache[key]
                deleted = True

        # Remove dependencies
        if key in self._dependencies:
            for dependent_key in self._dependencies[key]:
                await self.delete(dependent_key)
            del self._dependencies[key]

        if deleted:
            self._stats.invalidations += 1
            self._update_stats()

        return deleted

    async def clear(self) -> None:
        """Clear all caches."""
        for cache in self._caches.values():
            cache.clear()
        self._dependencies.clear()
        self._update_stats()

    async def _evict_by_memory(self) -> None:
        """Evict entries to free memory."""
        # Simple strategy: clear 20% of least recently used entries
        for cache in self._caches.values():
            if len(cache) > 0:
                items_to_remove = max(1, len(cache) // 5)
                for _ in range(items_to_remove):
                    try:
                        cache.popitem()
                        self._stats.evictions += 1
                    except KeyError:
                        break

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            if isinstance(value, str):
                return len(value.encode("utf-8"))
            elif isinstance(value, bytes):
                return len(value)
            elif isinstance(value, int | float):
                return 8
            elif isinstance(value, list | dict):
                return len(pickle.dumps(value))
            else:
                return 100  # Default estimate
        except (UnicodeDecodeError, PicklingError, MemoryError, ValueError) as e:
            # Log the specific error for debugging but don't let it propagate
            import logging

            logging.getLogger(__name__).debug(f"Failed to estimate cache value size: {e}")
            return 100

    def _update_access_time(self, start_time: float) -> None:
        """Update average access time."""
        access_time = (time.perf_counter() - start_time) * 1000
        self._stats.avg_access_time_ms = 0.1 * access_time + 0.9 * self._stats.avg_access_time_ms

    def _update_stats(self) -> None:
        """Update statistics."""
        self._stats.entry_count = sum(len(cache) for cache in self._caches.values())
        self._stats.memory_usage_mb = self._stats.size_bytes / (1024 * 1024)

        total_requests = self._stats.hits + self._stats.misses
        self._stats.hit_rate = self._stats.hits / total_requests if total_requests > 0 else 0

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self._update_stats()
        return self._stats


class L3RedisCache(CacheInterface):
    """
    L3 Redis Cache - Distributed cache for sharing data across instances.

    Features:
    - Distributed caching across multiple instances
    - Pub/sub for cache invalidation
    - Compression for large values
    - Atomic operations for consistency
    """

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self._stats = CacheStats(level=CacheLevel.L3_REDIS)
        self._compression_threshold = 1024  # Compress values > 1KB
        self._key_prefix = "tbot:cache:"

    async def get(self, key: str) -> Any | None:
        """Get from Redis cache."""
        start_time = time.perf_counter()

        try:
            full_key = self._key_prefix + key
            data = await self.redis.get(full_key)

            if data:
                value = self._deserialize(data)
                self._stats.hits += 1
                self._update_access_time(start_time)
                return value

            self._stats.misses += 1
            self._update_access_time(start_time)
            return None

        except Exception as e:
            self._stats.error_count += 1
            logger.error(f"L3 cache get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set in Redis cache."""
        try:
            full_key = self._key_prefix + key
            serialized_data = self._serialize(value)

            if ttl:
                await self.redis.setex(full_key, ttl, serialized_data)
            else:
                await self.redis.set(full_key, serialized_data)

            self._stats.size_bytes += len(serialized_data)
            return True

        except Exception as e:
            self._stats.error_count += 1
            logger.error(f"L3 cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete from Redis cache."""
        try:
            full_key = self._key_prefix + key
            result = await self.redis.delete(full_key)

            if result:
                self._stats.invalidations += 1
                # Publish invalidation event
                await self.redis.publish("cache_invalidation", key)

            return result > 0

        except Exception as e:
            self._stats.error_count += 1
            logger.error(f"L3 cache delete error: {e}")
            return False

    async def clear(self) -> None:
        """Clear Redis cache."""
        try:
            pattern = self._key_prefix + "*"
            async for key in self.redis.scan_iter(match=pattern):
                await self.redis.delete(key)
        except Exception as e:
            logger.error(f"L3 cache clear error: {e}")

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for Redis storage."""
        try:
            json_str = json.dumps(value, default=str)
            data = json_str.encode("utf-8")

            # Compress if large
            if len(data) > self._compression_threshold:
                import gzip

                data = gzip.compress(data)
                # Add compression marker
                data = b"\x01" + data
            else:
                data = b"\x00" + data

            return data
        except Exception as e:
            logger.error(
                "Serialization error",
                error=str(e),
                error_type=type(e).__name__,
                value_type=type(value).__name__,
            )
            raise

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from Redis storage."""
        try:
            # Check compression marker
            if data[0:1] == b"\x01":
                import gzip

                data = gzip.decompress(data[1:])
            else:
                data = data[1:]

            json_str = data.decode("utf-8")
            return json.loads(json_str)
        except Exception as e:
            logger.error(
                "Deserialization error",
                error=str(e),
                error_type=type(e).__name__,
                data_size=len(data) if data else 0,
            )
            raise

    def _update_access_time(self, start_time: float) -> None:
        """Update average access time."""
        access_time = (time.perf_counter() - start_time) * 1000
        self._stats.avg_access_time_ms = 0.1 * access_time + 0.9 * self._stats.avg_access_time_ms

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class UnifiedCacheLayer(BaseComponent):
    """
    Unified caching layer that coordinates all cache levels for optimal performance.

    This class provides:
    - Multi-level cache hierarchy (L1 -> L2 -> L3 -> Database)
    - Intelligent cache warming and prefetching
    - Cache coherence across all levels
    - Category-aware caching policies
    - Performance monitoring and optimization
    """

    def __init__(self, config: Config):
        """Initialize unified cache layer."""
        super().__init__()
        self.config = config

        # Cache level instances
        self.l1_cache: L1CPUCache | None = None
        self.l2_cache: L2MemoryCache | None = None
        self.l3_cache: L3RedisCache | None = None
        self.data_cache: DataCache | None = None

        # Cache policies
        self.policies: dict[DataCategory, CachePolicy] = self._define_cache_policies()

        # Performance monitoring
        self.global_stats = {
            "total_requests": 0,
            "cache_hierarchy_hits": defaultdict(int),
            "cache_hierarchy_misses": defaultdict(int),
            "avg_response_time_ms": 0.0,
            "cache_warming_events": 0,
            "cache_invalidation_events": 0,
        }

        # Cache warming
        self._warming_enabled = True
        self._warming_patterns: dict[str, list[str]] = defaultdict(list)
        self._warming_queue: asyncio.Queue = asyncio.Queue()

        # Cache invalidation
        self._invalidation_subscribers: list[Callable] = []

        # Background tasks
        self._background_tasks: list[asyncio.Task] = []

    def _define_cache_policies(self) -> dict[DataCategory, CachePolicy]:
        """Define caching policies for different data categories."""
        return {
            DataCategory.MARKET_DATA: CachePolicy(
                category=DataCategory.MARKET_DATA,
                levels=[CacheLevel.L1_CPU, CacheLevel.L2_MEMORY, CacheLevel.L3_REDIS],
                ttl_seconds=30,
                max_size=5000,
                strategy=CacheStrategy.WRITE_THROUGH,
                compression=False,
                warm_on_startup=True,
            ),
            DataCategory.TRADING_DATA: CachePolicy(
                category=DataCategory.TRADING_DATA,
                levels=[CacheLevel.L2_MEMORY, CacheLevel.L3_REDIS],
                ttl_seconds=300,
                max_size=2000,
                strategy=CacheStrategy.WRITE_BEHIND,
                compression=True,
                warm_on_startup=False,
            ),
            DataCategory.USER_DATA: CachePolicy(
                category=DataCategory.USER_DATA,
                levels=[CacheLevel.L2_MEMORY, CacheLevel.L3_REDIS],
                ttl_seconds=3600,
                max_size=1000,
                strategy=CacheStrategy.READ_THROUGH,
                compression=True,
                encryption=True,
            ),
            DataCategory.ANALYTICS_DATA: CachePolicy(
                category=DataCategory.ANALYTICS_DATA,
                levels=[CacheLevel.L2_MEMORY, CacheLevel.L3_REDIS],
                ttl_seconds=1800,
                max_size=1500,
                strategy=CacheStrategy.CACHE_ASIDE,
                compression=True,
            ),
            DataCategory.STATIC_DATA: CachePolicy(
                category=DataCategory.STATIC_DATA,
                levels=[CacheLevel.L2_MEMORY, CacheLevel.L3_REDIS],
                ttl_seconds=86400,
                max_size=500,
                strategy=CacheStrategy.READ_THROUGH,
                compression=True,
                warm_on_startup=True,
            ),
            DataCategory.ML_DATA: CachePolicy(
                category=DataCategory.ML_DATA,
                levels=[CacheLevel.L2_MEMORY, CacheLevel.L3_REDIS],
                ttl_seconds=600,
                max_size=1000,
                strategy=CacheStrategy.WRITE_BEHIND,
                compression=True,
            ),
        }

    async def initialize(self) -> None:
        """Initialize the unified cache layer."""
        try:
            self.logger.info("Initializing unified cache layer...")

            # Initialize cache levels
            await self._initialize_cache_levels()

            # Start background tasks
            await self._start_background_tasks()

            # Warm critical caches
            await self._warm_startup_caches()

            self.logger.info("Unified cache layer initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize unified cache layer: {e}")
            raise CacheError(f"Initialization failed: {e}") from e

    async def _initialize_cache_levels(self) -> None:
        """Initialize all cache levels."""
        # L1 CPU Cache
        self.l1_cache = L1CPUCache(max_size=1000)

        # L2 Memory Cache
        self.l2_cache = L2MemoryCache(max_size=10000, max_memory_mb=100)

        # L3 Redis Cache (requires Redis connection)
        redis_config = getattr(self.config, "redis", {})
        if redis_config:
            redis_client = redis.Redis.from_url(
                redis_config.get("url", "redis://localhost:6379"), decode_responses=False
            )
            self.l3_cache = L3RedisCache(redis_client)

        # L4 Data Cache
        self.data_cache = DataCache(self.config)
        await self.data_cache.initialize()

    async def _start_background_tasks(self) -> None:
        """Start background tasks for cache management."""
        # Cache warming task
        warming_task = asyncio.create_task(self._cache_warming_loop())
        self._background_tasks.append(warming_task)

        # Statistics collection task
        stats_task = asyncio.create_task(self._statistics_collection_loop())
        self._background_tasks.append(stats_task)

        # Cache maintenance task
        maintenance_task = asyncio.create_task(self._cache_maintenance_loop())
        self._background_tasks.append(maintenance_task)

    @time_execution
    async def get(
        self,
        key: str,
        category: DataCategory = DataCategory.TRADING_DATA,
        loader: Callable | None = None,
    ) -> Any | None:
        """
        Get value from unified cache hierarchy.

        Args:
            key: Cache key
            category: Data category for policy lookup
            loader: Optional function to load data on cache miss
        """
        start_time = time.perf_counter()
        self.global_stats["total_requests"] += 1

        try:
            policy = self.policies.get(category)
            if not policy:
                return None

            # Try cache levels in order
            for level in policy.levels:
                cache = self._get_cache_by_level(level)
                if cache:
                    value = await cache.get(key)
                    if value is not None:
                        self.global_stats["cache_hierarchy_hits"][level] += 1
                        await self._promote_to_higher_levels(key, value, level, policy)
                        self._update_global_response_time(start_time)
                        return value
                    else:
                        self.global_stats["cache_hierarchy_misses"][level] += 1

            # Cache miss across all levels
            if loader and policy.strategy in [
                CacheStrategy.READ_THROUGH,
                CacheStrategy.CACHE_ASIDE,
            ]:
                value = await loader(key)
                if value is not None:
                    await self.set(key, value, category=category)
                    self._update_global_response_time(start_time)
                    return value

            self._update_global_response_time(start_time)
            return None

        except Exception as e:
            self.logger.error(f"Cache get error for key {key}: {e}")
            return None

    @time_execution
    async def set(
        self,
        key: str,
        value: Any,
        category: DataCategory = DataCategory.TRADING_DATA,
        ttl: int | None = None,
    ) -> bool:
        """
        Set value in unified cache hierarchy.

        Args:
            key: Cache key
            value: Value to cache
            category: Data category for policy lookup
            ttl: Time to live override
        """
        try:
            policy = self.policies.get(category)
            if not policy:
                return False

            effective_ttl = ttl or policy.ttl_seconds
            success = True

            # Set in all configured levels
            for level in policy.levels:
                cache = self._get_cache_by_level(level)
                if cache:
                    result = await cache.set(key, value, effective_ttl)
                    success = success and result

            # Handle write strategies
            if policy.strategy == CacheStrategy.WRITE_BEHIND:
                # Queue for background persistence
                await self._queue_write_behind(key, value, category)

            # Track warming patterns
            self._track_warming_pattern(key, category)

            return success

        except Exception as e:
            self.logger.error(f"Cache set error for key {key}: {e}")
            return False

    async def delete(self, key: str, category: DataCategory = DataCategory.TRADING_DATA) -> bool:
        """Delete value from all cache levels."""
        try:
            policy = self.policies.get(category)
            if not policy:
                return False

            success = True

            # Delete from all levels
            for level in policy.levels:
                cache = self._get_cache_by_level(level)
                if cache:
                    result = await cache.delete(key)
                    success = success and result

            # Publish invalidation event
            await self._publish_invalidation(key, category)
            self.global_stats["cache_invalidation_events"] += 1

            return success

        except Exception as e:
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False

    async def invalidate_pattern(self, pattern: str, category: DataCategory) -> int:
        """Invalidate all keys matching pattern."""
        invalidated_count = 0

        try:
            # This is a simplified implementation
            # In practice, you'd need more sophisticated pattern matching
            policy = self.policies.get(category)
            if policy:
                for level in policy.levels:
                    cache = self._get_cache_by_level(level)
                    if cache and hasattr(cache, "invalidate_pattern"):
                        count = await cache.invalidate_pattern(pattern)
                        invalidated_count += count

            return invalidated_count

        except Exception as e:
            self.logger.error(f"Pattern invalidation error for {pattern}: {e}")
            return 0

    async def warm_cache(self, keys: list[str], category: DataCategory, loader: Callable) -> int:
        """Warm cache with data for specified keys."""
        warmed_count = 0

        try:
            # Load data in batches
            batch_size = 100
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i : i + batch_size]

                # Load data
                data = await loader(batch_keys)

                # Set in cache
                for key, value in data.items():
                    if await self.set(key, value, category=category):
                        warmed_count += 1

            self.global_stats["cache_warming_events"] += 1
            self.logger.info(f"Warmed {warmed_count} cache entries for category {category.value}")

            return warmed_count

        except Exception as e:
            self.logger.error(f"Cache warming error: {e}")
            return 0

    def _get_cache_by_level(self, level: CacheLevel) -> CacheInterface | None:
        """Get cache instance by level."""
        if level == CacheLevel.L1_CPU:
            return self.l1_cache
        elif level == CacheLevel.L2_MEMORY:
            return self.l2_cache
        elif level == CacheLevel.L3_REDIS:
            return self.l3_cache
        elif level == CacheLevel.L4_DATABASE:
            return self.data_cache
        return None

    async def _promote_to_higher_levels(
        self, key: str, value: Any, hit_level: CacheLevel, policy: CachePolicy
    ) -> None:
        """Promote cache hit to higher levels."""
        try:
            level_order = [
                CacheLevel.L1_CPU,
                CacheLevel.L2_MEMORY,
                CacheLevel.L3_REDIS,
                CacheLevel.L4_DATABASE,
            ]
            hit_index = level_order.index(hit_level)

            # Promote to all higher levels configured in policy
            for i in range(hit_index):
                level = level_order[i]
                if level in policy.levels:
                    cache = self._get_cache_by_level(level)
                    if cache:
                        await cache.set(key, value, policy.ttl_seconds)

        except Exception as e:
            self.logger.error(f"Cache promotion error: {e}")

    async def _warm_startup_caches(self) -> None:
        """Warm caches that should be populated on startup."""
        for category, policy in self.policies.items():
            if policy.warm_on_startup:
                # Queue for warming
                await self._warming_queue.put(
                    {
                        "type": "startup_warm",
                        "category": category,
                        "timestamp": datetime.now(timezone.utc),
                    }
                )

    async def _cache_warming_loop(self) -> None:
        """Background loop for cache warming."""
        while True:
            try:
                # Wait for warming requests
                request = await asyncio.wait_for(self._warming_queue.get(), timeout=60.0)
                await self._process_warming_request(request)

            except asyncio.TimeoutError:
                # Periodic predictive warming
                await self._predictive_warming()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cache warming error: {e}")

    async def _process_warming_request(self, request: dict[str, Any]) -> None:
        """Process a cache warming request."""
        # Implementation depends on specific warming strategies
        # This is a placeholder for the warming logic
        category = request.get("category")
        if category:
            self.logger.debug(f"Processing warming request for {category}")

    async def _predictive_warming(self) -> None:
        """Perform predictive cache warming based on patterns."""
        # Analyze access patterns and pre-warm likely cache misses
        # This is a placeholder for predictive warming logic
        pass

    def _track_warming_pattern(self, key: str, category: DataCategory) -> None:
        """Track access patterns for predictive warming."""
        pattern_key = f"{category.value}_{key[:10]}"  # Use key prefix for pattern
        self._warming_patterns[pattern_key].append(key)

        # Keep only recent patterns
        if len(self._warming_patterns[pattern_key]) > 100:
            self._warming_patterns[pattern_key] = self._warming_patterns[pattern_key][-50:]

    async def _queue_write_behind(self, key: str, value: Any, category: DataCategory) -> None:
        """Queue write-behind operation."""
        # Implementation for asynchronous write-behind caching
        pass

    async def _publish_invalidation(self, key: str, category: DataCategory) -> None:
        """Publish cache invalidation event."""
        for subscriber in self._invalidation_subscribers:
            try:
                await subscriber(key, category)
            except Exception as e:
                self.logger.error(f"Invalidation notification error: {e}")

    def _update_global_response_time(self, start_time: float) -> None:
        """Update global average response time."""
        response_time = (time.perf_counter() - start_time) * 1000
        self.global_stats["avg_response_time_ms"] = (
            0.1 * response_time + 0.9 * self.global_stats["avg_response_time_ms"]
        )

    async def _statistics_collection_loop(self) -> None:
        """Background loop for statistics collection."""
        while True:
            try:
                await asyncio.sleep(60)  # Collect stats every minute
                await self._collect_statistics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Statistics collection error: {e}")

    async def _collect_statistics(self) -> None:
        """Collect comprehensive cache statistics."""
        try:
            stats = await self.get_comprehensive_stats()

            # Log key metrics
            self.logger.info(
                "Cache performance summary",
                extra={
                    "total_requests": self.global_stats["total_requests"],
                    "avg_response_time_ms": self.global_stats["avg_response_time_ms"],
                    "l1_hit_rate": stats.get("l1_stats", {}).get("hit_rate", 0),
                    "l2_hit_rate": stats.get("l2_stats", {}).get("hit_rate", 0),
                    "l3_hit_rate": stats.get("l3_stats", {}).get("hit_rate", 0),
                    "cache_warming_events": self.global_stats["cache_warming_events"],
                    "invalidation_events": self.global_stats["cache_invalidation_events"],
                },
            )

        except Exception as e:
            self.logger.error(f"Statistics collection failed: {e}")

    async def _cache_maintenance_loop(self) -> None:
        """Background loop for cache maintenance."""
        while True:
            try:
                await asyncio.sleep(300)  # Run maintenance every 5 minutes
                await self._perform_maintenance()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cache maintenance error: {e}")

    async def _perform_maintenance(self) -> None:
        """Perform cache maintenance tasks."""
        try:
            # Clean up expired entries, optimize memory usage, etc.
            for cache in [self.l1_cache, self.l2_cache, self.l3_cache]:
                if cache and hasattr(cache, "maintenance"):
                    await cache.maintenance()

        except Exception as e:
            self.logger.error(f"Maintenance failed: {e}")

    async def get_comprehensive_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics from all cache levels."""
        stats = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "global_stats": self.global_stats.copy(),
        }

        # Collect stats from each level
        if self.l1_cache:
            stats["l1_stats"] = (await self.l1_cache.get_stats()).__dict__

        if self.l2_cache:
            stats["l2_stats"] = (await self.l2_cache.get_stats()).__dict__

        if self.l3_cache:
            stats["l3_stats"] = (await self.l3_cache.get_stats()).__dict__

        if self.data_cache:
            stats["data_cache_stats"] = await self.data_cache.get_stats()

        return stats

    async def cleanup(self) -> None:
        """Cleanup cache resources."""
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Cleanup cache levels
            if self.l1_cache:
                await self.l1_cache.clear()

            if self.l2_cache:
                await self.l2_cache.clear()

            if self.l3_cache:
                await self.l3_cache.clear()

            if self.data_cache:
                await self.data_cache.cleanup()

            self.logger.info("Unified cache layer cleaned up")

        except Exception as e:
            self.logger.error(f"Cache cleanup error: {e}")
