"""Test suite for data cache components."""

import asyncio
import pytest
import sys
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from src.data.cache.data_cache import (
    CacheConfig,
    CacheEntry,
    CacheLevel,
    CacheMode,
    CacheStats,
    CacheStrategy,
    L1MemoryCache,
)


class TestCacheEntry:
    """Test suite for CacheEntry."""

    def test_initialization(self):
        """Test cache entry initialization."""
        key = "test_key"
        value = {"data": "test"}
        created_at = datetime.now(timezone.utc)
        last_accessed = datetime.now(timezone.utc)
        metadata = {"source": "test"}
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=created_at,
            last_accessed=last_accessed,
            access_count=5,
            ttl_seconds=3600,
            size_bytes=100,
            metadata=metadata
        )
        
        assert entry.key == key
        assert entry.value == value
        assert entry.created_at == created_at
        assert entry.last_accessed == last_accessed
        assert entry.access_count == 5
        assert entry.ttl_seconds == 3600
        assert entry.size_bytes == 100
        assert entry.metadata == metadata

    def test_initialization_defaults(self):
        """Test cache entry initialization with defaults."""
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc)
        )
        
        assert entry.access_count == 0
        assert entry.ttl_seconds is None
        assert entry.size_bytes == 0
        assert entry.metadata == {}

    def test_is_expired_with_ttl(self):
        """Test expiration check with TTL."""
        past_time = datetime.now(timezone.utc) - timedelta(seconds=7200)  # 2 hours ago
        
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=past_time,
            last_accessed=past_time,
            ttl_seconds=3600  # 1 hour TTL
        )
        
        assert entry.is_expired() is True

    def test_is_expired_without_ttl(self):
        """Test expiration check without TTL."""
        past_time = datetime.now(timezone.utc) - timedelta(seconds=7200)
        
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=past_time,
            last_accessed=past_time,
            ttl_seconds=None
        )
        
        assert entry.is_expired() is False

    def test_is_expired_not_expired(self):
        """Test expiration check for non-expired entry."""
        recent_time = datetime.now(timezone.utc) - timedelta(seconds=1800)  # 30 minutes ago
        
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=recent_time,
            last_accessed=recent_time,
            ttl_seconds=3600  # 1 hour TTL
        )
        
        assert entry.is_expired() is False

    def test_update_access(self):
        """Test access update."""
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
            access_count=5
        )
        
        original_access_time = entry.last_accessed
        original_count = entry.access_count
        
        # Wait a small amount to ensure time difference
        import time
        time.sleep(0.01)
        
        entry.update_access()
        
        assert entry.access_count == original_count + 1
        assert entry.last_accessed > original_access_time


class TestCacheStats:
    """Test suite for CacheStats."""

    def test_initialization_defaults(self):
        """Test cache stats initialization with defaults."""
        stats = CacheStats()
        
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.writes == 0
        assert stats.deletes == 0
        assert stats.size_bytes == 0
        assert stats.entry_count == 0
        assert stats.hit_rate == 0.0
        assert stats.memory_usage_mb == 0.0

    def test_initialization_with_values(self):
        """Test cache stats initialization with custom values."""
        stats = CacheStats(
            hits=100,
            misses=25,
            evictions=5,
            writes=120,
            deletes=15,
            size_bytes=1048576,
            entry_count=150,
            hit_rate=0.8,
            memory_usage_mb=64.5
        )
        
        assert stats.hits == 100
        assert stats.misses == 25
        assert stats.evictions == 5
        assert stats.writes == 120
        assert stats.deletes == 15
        assert stats.size_bytes == 1048576
        assert stats.entry_count == 150
        assert stats.hit_rate == 0.8
        assert stats.memory_usage_mb == 64.5

    def test_calculate_hit_rate_with_data(self):
        """Test hit rate calculation with data."""
        stats = CacheStats(hits=80, misses=20)
        stats.calculate_hit_rate()
        
        assert stats.hit_rate == 0.8

    def test_calculate_hit_rate_no_data(self):
        """Test hit rate calculation with no data."""
        stats = CacheStats(hits=0, misses=0)
        stats.calculate_hit_rate()
        
        assert stats.hit_rate == 0.0

    def test_calculate_hit_rate_only_misses(self):
        """Test hit rate calculation with only misses."""
        stats = CacheStats(hits=0, misses=50)
        stats.calculate_hit_rate()
        
        assert stats.hit_rate == 0.0


class TestCacheConfig:
    """Test suite for CacheConfig."""

    def test_initialization_defaults(self):
        """Test cache config initialization with defaults."""
        config = CacheConfig(level=CacheLevel.L1_MEMORY)
        
        assert config.level == CacheLevel.L1_MEMORY
        assert config.max_size == 10000
        assert config.max_memory_mb == 512
        assert config.default_ttl == 3600
        assert config.strategy == CacheStrategy.LRU
        assert config.mode == CacheMode.CACHE_ASIDE
        assert config.compression_enabled is False
        assert config.serialization_format == "json"
        assert config.key_prefix == "tbot"

    def test_initialization_custom_values(self):
        """Test cache config initialization with custom values."""
        config = CacheConfig(
            level=CacheLevel.L2_REDIS,
            max_size=50000,
            max_memory_mb=1024,
            default_ttl=7200,
            strategy=CacheStrategy.LFU,
            mode=CacheMode.WRITE_THROUGH,
            compression_enabled=True,
            serialization_format="pickle",
            key_prefix="trading"
        )
        
        assert config.level == CacheLevel.L2_REDIS
        assert config.max_size == 50000
        assert config.max_memory_mb == 1024
        assert config.default_ttl == 7200
        assert config.strategy == CacheStrategy.LFU
        assert config.mode == CacheMode.WRITE_THROUGH
        assert config.compression_enabled is True
        assert config.serialization_format == "pickle"
        assert config.key_prefix == "trading"

    def test_validation_max_size(self):
        """Test max_size validation."""
        with pytest.raises(ValueError):
            CacheConfig(level=CacheLevel.L1_MEMORY, max_size=0)

    def test_validation_max_memory_mb(self):
        """Test max_memory_mb validation."""
        with pytest.raises(ValueError):
            CacheConfig(level=CacheLevel.L1_MEMORY, max_memory_mb=0)
        
        with pytest.raises(ValueError):
            CacheConfig(level=CacheLevel.L1_MEMORY, max_memory_mb=10000)

    def test_validation_default_ttl(self):
        """Test default_ttl validation."""
        with pytest.raises(ValueError):
            CacheConfig(level=CacheLevel.L1_MEMORY, default_ttl=0)
        
        with pytest.raises(ValueError):
            CacheConfig(level=CacheLevel.L1_MEMORY, default_ttl=100000)


class TestL1MemoryCache:
    """Test suite for L1MemoryCache."""

    @pytest.fixture
    def cache_config(self):
        """Create cache configuration."""
        return CacheConfig(
            level=CacheLevel.L1_MEMORY,
            max_size=100,
            max_memory_mb=10,
            default_ttl=3600,
            strategy=CacheStrategy.LRU
        )

    @pytest.fixture
    def memory_cache(self, cache_config):
        """Create L1 memory cache instance."""
        return L1MemoryCache(config=cache_config)

    def test_initialization(self, cache_config):
        """Test L1 memory cache initialization."""
        cache = L1MemoryCache(config=cache_config)
        
        assert cache.config is cache_config
        assert len(cache._cache) == 0
        assert isinstance(cache._stats, CacheStats)
        assert isinstance(cache._lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, memory_cache):
        """Test getting nonexistent key."""
        result = await memory_cache.get("nonexistent")
        
        assert result is None
        assert memory_cache._stats.misses == 1
        assert memory_cache._stats.hits == 0

    @pytest.mark.asyncio
    async def test_set_and_get_success(self, memory_cache):
        """Test successful set and get operations."""
        key = "test_key"
        value = {"data": "test_value"}
        
        # Mock the methods that require implementation
        with patch.object(memory_cache, '_calculate_size', return_value=100):
            with patch.object(memory_cache, '_ensure_capacity', return_value=True):
                set_result = await memory_cache.set(key, value)
        
        assert set_result is True
        assert memory_cache._stats.writes == 1
        assert memory_cache._stats.entry_count == 1
        assert memory_cache._stats.size_bytes == 100
        
        # Get the value
        get_result = await memory_cache.get(key)
        
        assert get_result == value
        assert memory_cache._stats.hits == 1
        assert memory_cache._stats.misses == 0

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self, memory_cache):
        """Test set operation with custom TTL."""
        key = "test_key"
        value = "test_value"
        custom_ttl = 7200
        
        with patch.object(memory_cache, '_calculate_size', return_value=50):
            with patch.object(memory_cache, '_ensure_capacity', return_value=True):
                await memory_cache.set(key, value, ttl=custom_ttl)
        
        entry = memory_cache._cache[key]
        assert entry.ttl_seconds == custom_ttl

    @pytest.mark.asyncio
    async def test_set_capacity_exceeded(self, memory_cache):
        """Test set operation when capacity is exceeded."""
        key = "test_key"
        value = "test_value"
        
        with patch.object(memory_cache, '_calculate_size', return_value=100):
            with patch.object(memory_cache, '_ensure_capacity', return_value=False):
                result = await memory_cache.set(key, value)
        
        assert result is False
        assert len(memory_cache._cache) == 0

    @pytest.mark.asyncio
    async def test_get_expired_entry(self, memory_cache):
        """Test getting expired cache entry."""
        key = "test_key"
        value = "test_value"
        
        # Create expired entry manually
        expired_time = datetime.now(timezone.utc) - timedelta(seconds=7200)
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=expired_time,
            last_accessed=expired_time,
            ttl_seconds=3600  # 1 hour TTL, but created 2 hours ago
        )
        memory_cache._cache[key] = entry
        
        with patch.object(memory_cache, '_evict_entry') as mock_evict:
            result = await memory_cache.get(key)
        
        assert result is None
        assert memory_cache._stats.misses == 1
        mock_evict.assert_called_once_with(key)

    @pytest.mark.asyncio
    async def test_set_replace_existing(self, memory_cache):
        """Test replacing existing cache entry."""
        key = "test_key"
        old_value = "old_value"
        new_value = "new_value"
        
        # Set initial value
        with patch.object(memory_cache, '_calculate_size', return_value=50):
            with patch.object(memory_cache, '_ensure_capacity', return_value=True):
                await memory_cache.set(key, old_value)
        
        assert memory_cache._stats.size_bytes == 50
        assert memory_cache._stats.entry_count == 1
        
        # Replace with new value
        with patch.object(memory_cache, '_calculate_size', return_value=100):
            with patch.object(memory_cache, '_ensure_capacity', return_value=True):
                await memory_cache.set(key, new_value)
        
        assert memory_cache._stats.size_bytes == 100  # Old size removed, new added
        assert memory_cache._stats.entry_count == 1   # Still one entry
        assert memory_cache._stats.writes == 2        # Two write operations

    @pytest.mark.asyncio
    async def test_lru_strategy_access_ordering(self, memory_cache):
        """Test LRU strategy maintains access ordering."""
        # Set up cache with LRU strategy
        assert memory_cache.config.strategy == CacheStrategy.LRU
        
        key1 = "key1"
        key2 = "key2"
        
        with patch.object(memory_cache, '_calculate_size', return_value=50):
            with patch.object(memory_cache, '_ensure_capacity', return_value=True):
                await memory_cache.set(key1, "value1")
                await memory_cache.set(key2, "value2")
        
        # Access key1 to move it to end
        await memory_cache.get(key1)
        
        # Check that key1 was moved to end (most recent)
        keys = list(memory_cache._cache.keys())
        assert keys[-1] == key1  # key1 should be last (most recently accessed)


class TestEnums:
    """Test suite for cache enums."""

    def test_cache_level_values(self):
        """Test cache level enum values."""
        assert CacheLevel.L1_MEMORY.value == "l1_memory"
        assert CacheLevel.L2_REDIS.value == "l2_redis"
        assert CacheLevel.L3_DATABASE.value == "l3_database"

    def test_cache_strategy_values(self):
        """Test cache strategy enum values."""
        assert CacheStrategy.LRU.value == "lru"
        assert CacheStrategy.LFU.value == "lfu"
        assert CacheStrategy.FIFO.value == "fifo"
        assert CacheStrategy.TTL.value == "ttl"

    def test_cache_mode_values(self):
        """Test cache mode enum values."""
        assert CacheMode.READ_THROUGH.value == "read_through"
        assert CacheMode.WRITE_THROUGH.value == "write_through"
        assert CacheMode.WRITE_BEHIND.value == "write_behind"
        assert CacheMode.CACHE_ASIDE.value == "cache_aside"