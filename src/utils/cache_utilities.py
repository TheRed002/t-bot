"""
Shared Cache Utilities

This module consolidates common caching functionality to eliminate duplication
between data_cache.py and redis_cache.py, providing shared cache operations,
statistics tracking, and serialization/deserialization utilities.
"""

import json
import pickle
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.core.exceptions import CacheError
from src.core.logging import get_logger

logger = get_logger(__name__)


class CacheLevel(Enum):
    """Cache level enumeration."""

    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DATABASE = "l3_database"


class CacheStrategy(Enum):
    """Cache strategy enumeration."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live based


class CacheMode(Enum):
    """Cache operation mode."""

    READ_THROUGH = "read_through"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    CACHE_ASIDE = "cache_aside"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: int | None = None
    size_bytes: int = 0
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if not self.ttl_seconds:
            return False
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def update_access(self) -> None:
        """Update access metadata."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    writes: int = 0
    deletes: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    memory_usage_mb: float = 0.0

    def calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.hits + self.misses
        if total_requests == 0:
            return 0.0
        self.hit_rate = self.hits / total_requests
        return self.hit_rate

    def calculate_memory_usage_mb(self) -> float:
        """Calculate memory usage in MB."""
        self.memory_usage_mb = self.size_bytes / (1024 * 1024)
        return self.memory_usage_mb


class CacheSerializationUtils:
    """Shared serialization utilities for cache implementations."""

    @staticmethod
    def serialize_json(value: Any) -> str:
        """
        Serialize value to JSON string with proper error handling.

        Args:
            value: Value to serialize

        Returns:
            JSON string

        Raises:
            CacheError: If serialization fails
        """
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization failed: {e}")
            raise CacheError(f"Failed to serialize value to JSON: {e}") from e

    @staticmethod
    def deserialize_json(data: str) -> Any:
        """
        Deserialize JSON string to Python object.

        Args:
            data: JSON string to deserialize

        Returns:
            Deserialized object

        Raises:
            CacheError: If deserialization fails
        """
        try:
            return json.loads(data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON deserialization failed: {e}")
            raise CacheError(f"Failed to deserialize JSON data: {e}") from e

    @staticmethod
    def serialize_pickle(value: Any) -> bytes:
        """
        Serialize value to pickle bytes with proper error handling.

        Args:
            value: Value to serialize

        Returns:
            Pickle bytes

        Raises:
            CacheError: If serialization fails
        """
        try:
            return pickle.dumps(value)
        except (TypeError, pickle.PickleError) as e:
            logger.error(f"Pickle serialization failed: {e}")
            raise CacheError(f"Failed to serialize value to pickle: {e}") from e

    @staticmethod
    def deserialize_pickle(data: bytes) -> Any:
        """
        Deserialize pickle bytes to Python object.

        Args:
            data: Pickle bytes to deserialize

        Returns:
            Deserialized object

        Raises:
            CacheError: If deserialization fails
        """
        try:
            return pickle.loads(data)
        except (TypeError, pickle.PickleError) as e:
            logger.error(f"Pickle deserialization failed: {e}")
            raise CacheError(f"Failed to deserialize pickle data: {e}") from e

    @staticmethod
    def calculate_size_bytes(value: Any) -> int:
        """
        Calculate approximate size of an object in bytes.

        Args:
            value: Object to calculate size for

        Returns:
            Size in bytes
        """
        try:
            return sys.getsizeof(value)
        except (TypeError, AttributeError):
            # Fallback for complex objects
            try:
                return len(pickle.dumps(value))
            except Exception:
                return 0


class CacheKeyUtils:
    """Shared cache key utilities."""

    @staticmethod
    def generate_key(prefix: str, *args: Any) -> str:
        """
        Generate cache key from prefix and arguments.

        Args:
            prefix: Key prefix
            *args: Arguments to include in key

        Returns:
            Cache key string
        """
        key_parts = [str(prefix)]
        for arg in args:
            if isinstance(arg, (str, int, float)):
                key_parts.append(str(arg))
            else:
                # Use hash for complex objects
                key_parts.append(str(hash(str(arg))))

        return ":".join(key_parts)

    @staticmethod
    def validate_key(key: str) -> bool:
        """
        Validate cache key format.

        Args:
            key: Cache key to validate

        Returns:
            True if valid
        """
        if not key or not isinstance(key, str):
            return False

        # Check for invalid characters
        invalid_chars = {"\n", "\r", "\t", "\0"}
        if any(char in key for char in invalid_chars):
            return False

        # Check length (Redis has max key length of 512MB, but we use 1KB for practicality)
        if len(key.encode("utf-8")) > 1024:
            return False

        return True


class CacheLRUUtils:
    """Shared LRU cache utilities."""

    @staticmethod
    def update_lru_order(cache: OrderedDict, key: str) -> None:
        """
        Update LRU order by moving key to end.

        Args:
            cache: OrderedDict cache
            key: Key to move to end
        """
        if key in cache:
            cache.move_to_end(key)

    @staticmethod
    def evict_lru(cache: OrderedDict, stats: CacheStats) -> str | None:
        """
        Evict least recently used item from cache.

        Args:
            cache: OrderedDict cache
            stats: Cache statistics to update

        Returns:
            Evicted key or None
        """
        if not cache:
            return None

        # Remove first item (least recently used)
        evicted_key, evicted_entry = cache.popitem(last=False)

        # Update statistics
        stats.evictions += 1
        stats.entry_count -= 1
        if hasattr(evicted_entry, "size_bytes"):
            stats.size_bytes -= evicted_entry.size_bytes

        return evicted_key


class CacheValidationUtils:
    """Shared cache validation utilities."""

    @staticmethod
    def validate_ttl(ttl: int | None) -> bool:
        """
        Validate TTL value.

        Args:
            ttl: TTL in seconds

        Returns:
            True if valid
        """
        if ttl is None:
            return True

        return isinstance(ttl, int) and ttl > 0

    @staticmethod
    def validate_cache_size(max_size: int) -> bool:
        """
        Validate cache size configuration.

        Args:
            max_size: Maximum cache size

        Returns:
            True if valid
        """
        return isinstance(max_size, int) and max_size > 0

    @staticmethod
    def validate_cache_config(config: dict[str, Any]) -> list[str]:
        """
        Validate cache configuration and return list of errors.

        Args:
            config: Cache configuration dictionary

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields
        required_fields = ["max_size", "default_ttl"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Validate max_size
        if "max_size" in config and not CacheValidationUtils.validate_cache_size(
            config["max_size"]
        ):
            errors.append("Invalid max_size: must be positive integer")

        # Validate default_ttl
        if "default_ttl" in config and not CacheValidationUtils.validate_ttl(config["default_ttl"]):
            errors.append("Invalid default_ttl: must be positive integer or None")

        return errors
