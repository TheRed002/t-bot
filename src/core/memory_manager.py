"""
High-Performance Memory Manager

This module implements advanced memory management optimizations including
object pooling, memory leak detection, and automatic cleanup for stable
memory usage under 2GB during normal operations.

Key Features:
- Object pools for frequently created/destroyed objects
- Memory leak detection and automatic cleanup
- Weak references to prevent circular references
- Memory profiling and monitoring
- Cache-friendly data structures
- Automatic garbage collection optimization
- Memory-mapped files for large datasets

Performance Targets:
- Memory usage: Stable under 2GB for normal operations
- Object allocation: <1μs for pooled objects
- Memory leak detection: <5MB growth per hour
- GC pause time: <10ms for minor collections
"""

import asyncio
import gc
import mmap
import os
import threading
import time
import tracemalloc
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import psutil

from src.core.config import Config
from src.core.logging import get_logger

T = TypeVar("T")


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    heap_mb: float  # Heap memory
    available_mb: float  # Available system memory
    gc_collections: dict[int, int] = field(default_factory=dict)
    gc_time_ms: float = 0.0
    object_counts: dict[str, int] = field(default_factory=dict)
    pool_utilization: dict[str, float] = field(default_factory=dict)

    @property
    def memory_pressure(self) -> float:
        """Calculate memory pressure (0.0 to 1.0)."""
        if self.available_mb <= 0:
            return 1.0
        return min(1.0, self.rss_mb / (self.rss_mb + self.available_mb))


class ObjectPool(Generic[T]):
    """High-performance object pool for frequently used objects."""

    def __init__(
        self,
        create_func: callable,
        reset_func: optional_callable = None,
        max_size: int = 1000,
        name: str = "ObjectPool",
    ):
        """
        Initialize object pool.

        Args:
            create_func: Function to create new objects
            reset_func: Function to reset objects before reuse
            max_size: Maximum pool size
            name: Pool name for monitoring
        """
        self.create_func = create_func
        self.reset_func = reset_func or (lambda obj: None)
        self.max_size = max_size
        self.name = name

        # Thread-safe pool using deque
        self._pool = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self._in_use = set()  # Track objects in use

        # Statistics
        self.created_count = 0
        self.borrowed_count = 0
        self.returned_count = 0
        self.discarded_count = 0

        # Pre-populate pool
        self._populate_pool()

    def _populate_pool(self, initial_size: int | None = None):
        """Pre-populate pool with objects."""
        size = initial_size or min(10, self.max_size // 4)

        with self._lock:
            for _ in range(size):
                if len(self._pool) < self.max_size:
                    obj = self.create_func()
                    self._pool.append(obj)
                    self.created_count += 1

    def borrow(self) -> T:
        """Borrow object from pool."""
        with self._lock:
            if self._pool:
                obj = self._pool.popleft()
                self._in_use.add(id(obj))
                self.borrowed_count += 1
                return obj
            else:
                # Pool empty, create new object
                obj = self.create_func()
                self._in_use.add(id(obj))
                self.created_count += 1
                self.borrowed_count += 1
                return obj

    def return_object(self, obj: T):
        """Return object to pool."""
        obj_id = id(obj)

        with self._lock:
            if obj_id not in self._in_use:
                return  # Object not from this pool

            self._in_use.remove(obj_id)
            self.returned_count += 1

            # Reset object
            try:
                self.reset_func(obj)
            except Exception:
                # If reset fails, discard object
                self.discarded_count += 1
                return

            # Return to pool if not full
            if len(self._pool) < self.max_size:
                self._pool.append(obj)
            else:
                self.discarded_count += 1

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "name": self.name,
                "pool_size": len(self._pool),
                "in_use": len(self._in_use),
                "max_size": self.max_size,
                "created": self.created_count,
                "borrowed": self.borrowed_count,
                "returned": self.returned_count,
                "discarded": self.discarded_count,
                "utilization": len(self._in_use) / max(1, self.borrowed_count),
                "efficiency": len(self._pool) / max(1, self.created_count),
            }

    def clear(self):
        """Clear all objects from pool."""
        with self._lock:
            self._pool.clear()
            self._in_use.clear()


class MemoryLeakDetector:
    """Detect and track memory leaks."""

    def __init__(self, check_interval: int = 300):  # 5 minutes
        self.check_interval = check_interval
        self.snapshots = deque(maxlen=100)  # Keep last 100 snapshots
        self.is_running = False
        self.logger = get_logger(f"{__name__}.LeakDetector")

        # Enable tracemalloc for detailed tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start(10)  # Keep 10 frames

    async def start(self):
        """Start memory leak detection."""
        self.is_running = True

        while self.is_running:
            try:
                await self._take_snapshot()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error("Memory leak detection error", error=str(e))
                await asyncio.sleep(30)  # Retry in 30 seconds

    async def _take_snapshot(self):
        """Take memory snapshot."""
        if not tracemalloc.is_tracing():
            return

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        # Get top memory users
        memory_usage = {}
        for stat in top_stats[:20]:  # Top 20 allocations
            key = f"{stat.traceback.format()}"
            memory_usage[key] = {
                "size_mb": stat.size / (1024 * 1024),
                "count": stat.count,
            }

        snapshot_data = {
            "timestamp": time.time(),
            "total_mb": sum(stat.size for stat in top_stats) / (1024 * 1024),
            "allocations": memory_usage,
        }

        self.snapshots.append(snapshot_data)

        # Check for leaks
        await self._analyze_leaks()

    async def _analyze_leaks(self):
        """Analyze snapshots for memory leaks."""
        if len(self.snapshots) < 5:
            return

        # Compare current with 5 snapshots ago
        current = self.snapshots[-1]
        previous = self.snapshots[-5]

        memory_growth = current["total_mb"] - previous["total_mb"]
        time_diff = current["timestamp"] - previous["timestamp"]

        # Growth rate in MB per hour
        growth_rate = (memory_growth / max(1, time_diff)) * 3600

        if growth_rate > 5.0:  # More than 5MB per hour
            self.logger.warning(
                "Potential memory leak detected",
                growth_rate_mb_per_hour=round(growth_rate, 2),
                current_usage_mb=round(current["total_mb"], 2),
            )

            # Log top growing allocations
            await self._log_top_growers(current, previous)

    async def _log_top_growers(self, current: dict, previous: dict):
        """Log top growing allocations."""
        growers = []

        for key, curr_data in current["allocations"].items():
            prev_data = previous["allocations"].get(key, {"size_mb": 0})
            growth = curr_data["size_mb"] - prev_data["size_mb"]

            if growth > 0.1:  # More than 100KB growth
                growers.append((key, growth))

        # Sort by growth
        growers.sort(key=lambda x: x[1], reverse=True)

        for location, growth in growers[:5]:  # Top 5 growers
            self.logger.warning(
                "Memory growth detected",
                location=location[:200],  # Truncate long traces
                growth_mb=round(growth, 3),
            )

    def stop(self):
        """Stop leak detection."""
        self.is_running = False

    def get_leak_report(self) -> dict[str, Any]:
        """Get comprehensive leak report."""
        if len(self.snapshots) < 2:
            return {"status": "insufficient_data"}

        latest = self.snapshots[-1]
        oldest = self.snapshots[0]

        total_growth = latest["total_mb"] - oldest["total_mb"]
        time_span = latest["timestamp"] - oldest["timestamp"]
        growth_rate = (total_growth / max(1, time_span)) * 3600

        return {
            "status": "leak_detected" if growth_rate > 5.0 else "normal",
            "total_growth_mb": round(total_growth, 2),
            "time_span_hours": round(time_span / 3600, 2),
            "growth_rate_mb_per_hour": round(growth_rate, 2),
            "snapshots_analyzed": len(self.snapshots),
            "current_usage_mb": round(latest["total_mb"], 2),
        }


class CacheOptimizedList:
    """Cache-friendly list implementation for better performance."""

    def __init__(self, initial_capacity: int = 1000, growth_factor: float = 1.5):
        self.data = [None] * initial_capacity
        self.size = 0
        self.capacity = initial_capacity
        self.growth_factor = growth_factor

    def append(self, item):
        """Add item to list."""
        if self.size >= self.capacity:
            self._grow()

        self.data[self.size] = item
        self.size += 1

    def _grow(self):
        """Grow internal storage."""
        new_capacity = int(self.capacity * self.growth_factor)
        new_data = [None] * new_capacity

        # Copy existing data
        for i in range(self.size):
            new_data[i] = self.data[i]

        self.data = new_data
        self.capacity = new_capacity

    def get(self, index: int):
        """Get item at index."""
        if 0 <= index < self.size:
            return self.data[index]
        raise IndexError("Index out of range")

    def clear(self):
        """Clear all items."""
        # Set to None to help GC
        for i in range(self.size):
            self.data[i] = None
        self.size = 0

    def __len__(self):
        return self.size


class MemoryMappedCache:
    """Memory-mapped cache for large datasets."""

    def __init__(self, file_path: str, max_size: int = 100 * 1024 * 1024):  # 100MB default
        self.file_path = file_path
        self.max_size = max_size
        self.mmap_file = None
        self.file_handle = None
        self.current_size = 0

        # Create file if it doesn't exist
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(b"\x00" * max_size)

        self._open_mmap()

    def _open_mmap(self):
        """Open memory-mapped file."""
        try:
            self.file_handle = open(self.file_path, "r+b")
            self.mmap_file = mmap.mmap(self.file_handle.fileno(), self.max_size)
        except Exception as e:
            if self.file_handle:
                self.file_handle.close()
            raise e

    def write_data(self, offset: int, data: bytes) -> bool:
        """Write data at offset."""
        if offset + len(data) > self.max_size:
            return False

        try:
            self.mmap_file[offset : offset + len(data)] = data
            self.current_size = max(self.current_size, offset + len(data))
            return True
        except Exception:
            return False

    def read_data(self, offset: int, length: int) -> bytes | None:
        """Read data from offset."""
        if offset + length > self.current_size:
            return None

        try:
            return self.mmap_file[offset : offset + length]
        except Exception:
            return None

    def close(self):
        """Close memory-mapped file."""
        if self.mmap_file:
            self.mmap_file.close()
        if self.file_handle:
            self.file_handle.close()


class HighPerformanceMemoryManager:
    """Comprehensive memory management system."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)

        # Object pools for common objects
        self.pools = {}
        self._initialize_pools()

        # Memory monitoring
        self.process = psutil.Process()
        self.leak_detector = MemoryLeakDetector()

        # Weak reference tracking
        self.weak_refs = weakref.WeakSet()

        # Memory statistics
        self.stats_history = deque(maxlen=1000)  # Keep last 1000 readings

        # Background monitoring
        self.monitoring_task = None
        self.is_running = False

        # GC optimization
        self.gc_thresholds = gc.get_threshold()
        self._optimize_gc()

        self.logger.info(
            "Memory manager initialized",
            initial_memory_mb=round(self._get_memory_usage().rss_mb, 2),
        )

    def _initialize_pools(self):
        """Initialize object pools for common objects."""

        # Dictionary pool
        self.pools["dict"] = ObjectPool(
            create_func=dict, reset_func=lambda d: d.clear(), max_size=1000, name="DictPool"
        )

        # List pool
        self.pools["list"] = ObjectPool(
            create_func=list, reset_func=lambda l: l.clear(), max_size=1000, name="ListPool"
        )

        # Set pool
        self.pools["set"] = ObjectPool(
            create_func=set, reset_func=lambda s: s.clear(), max_size=500, name="SetPool"
        )

        # Cache-optimized list pool
        self.pools["cache_list"] = ObjectPool(
            create_func=CacheOptimizedList,
            reset_func=lambda l: l.clear(),
            max_size=100,
            name="CacheListPool",
        )

    def _optimize_gc(self):
        """Optimize garbage collection settings."""
        # Increase thresholds to reduce GC frequency
        # (generation0, generation1, generation2)
        new_thresholds = (
            self.gc_thresholds[0] * 2,  # 1400 -> 2800
            self.gc_thresholds[1] * 3,  # 20 -> 60
            self.gc_thresholds[2] * 3,  # 20 -> 60
        )

        gc.set_threshold(*new_thresholds)

        # Disable GC during critical sections (can be controlled by config)
        if hasattr(self.config, "memory") and getattr(
            self.config.memory, "disable_gc_during_trading", False
        ):
            gc.disable()

        self.logger.info(
            "GC optimized", old_thresholds=self.gc_thresholds, new_thresholds=new_thresholds
        )

    async def start_monitoring(self):
        """Start background memory monitoring."""
        self.is_running = True

        # Start leak detection
        asyncio.create_task(self.leak_detector.start())

        # Start memory monitoring
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        self.logger.info("Memory monitoring started")

    async def _monitoring_loop(self):
        """Background memory monitoring loop."""
        while self.is_running:
            try:
                # Collect memory statistics
                stats = self._get_memory_usage()
                self.stats_history.append(stats)

                # Check for memory pressure
                if stats.memory_pressure > 0.8:
                    self.logger.warning(
                        "High memory pressure detected",
                        pressure=round(stats.memory_pressure, 3),
                        rss_mb=round(stats.rss_mb, 2),
                    )

                    # Trigger cleanup
                    await self._emergency_cleanup()

                # Periodic GC if memory usage is high
                if stats.rss_mb > 1500:  # More than 1.5GB
                    await self._perform_gc()

                # Wait for next check
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error("Memory monitoring error", error=str(e))
                await asyncio.sleep(30)

    def _get_memory_usage(self) -> MemoryStats:
        """Get current memory usage statistics."""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()

        # Get GC stats
        gc_stats = {i: gc.get_count()[i] for i in range(3)}

        # Get object counts
        object_counts = {}
        for obj_type in [dict, list, set, str, int, float]:
            object_counts[obj_type.__name__] = len(gc.get_objects())

        # Get pool utilization
        pool_utilization = {}
        for name, pool in self.pools.items():
            stats = pool.get_stats()
            pool_utilization[name] = stats["utilization"]

        return MemoryStats(
            rss_mb=memory_info.rss / (1024 * 1024),
            vms_mb=memory_info.vms / (1024 * 1024),
            heap_mb=0,  # Would need additional profiling
            available_mb=system_memory.available / (1024 * 1024),
            gc_collections=gc_stats,
            object_counts=object_counts,
            pool_utilization=pool_utilization,
        )

    async def _emergency_cleanup(self):
        """Perform emergency cleanup when memory pressure is high."""
        self.logger.info("Performing emergency cleanup")

        # Force GC
        collected = gc.collect()
        self.logger.info(f"Emergency GC collected {collected} objects")

        # Clear pools partially
        for pool in self.pools.values():
            with pool._lock:
                # Keep only 25% of pooled objects
                while len(pool._pool) > pool.max_size // 4:
                    pool._pool.popleft()
                    pool.discarded_count += 1

        # Clear weak references
        self.weak_refs.clear()

        # Trim statistics history
        while len(self.stats_history) > 500:
            self.stats_history.popleft()

    async def _perform_gc(self):
        """Perform garbage collection."""
        start_time = time.perf_counter()

        # Run GC in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        collected = await loop.run_in_executor(None, gc.collect)

        gc_time = (time.perf_counter() - start_time) * 1000

        self.logger.debug("GC performed", collected=collected, time_ms=round(gc_time, 2))

    def get_pool(self, pool_name: str) -> ObjectPool | None:
        """Get object pool by name."""
        return self.pools.get(pool_name)

    def borrow_object(self, pool_name: str):
        """Borrow object from pool."""
        pool = self.pools.get(pool_name)
        if pool:
            return pool.borrow()
        return None

    def return_object(self, pool_name: str, obj):
        """Return object to pool."""
        pool = self.pools.get(pool_name)
        if pool:
            pool.return_object(obj)

    def track_object(self, obj):
        """Track object with weak reference."""
        self.weak_refs.add(obj)

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        return self._get_memory_usage()

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        current_stats = self._get_memory_usage()

        # Calculate trends if we have history
        trend_info = {}
        if len(self.stats_history) >= 10:
            recent_avg = sum(s.rss_mb for s in list(self.stats_history)[-10:]) / 10
            older_avg = sum(s.rss_mb for s in list(self.stats_history)[-20:-10]) / 10
            trend_info = {
                "memory_trend_mb_per_min": (recent_avg - older_avg) / 10,
                "trend_direction": "increasing" if recent_avg > older_avg else "stable",
            }

        # Pool statistics
        pool_stats = {name: pool.get_stats() for name, pool in self.pools.items()}

        # Leak detection report
        leak_report = self.leak_detector.get_leak_report()

        return {
            "current_memory": {
                "rss_mb": round(current_stats.rss_mb, 2),
                "vms_mb": round(current_stats.vms_mb, 2),
                "available_mb": round(current_stats.available_mb, 2),
                "memory_pressure": round(current_stats.memory_pressure, 3),
            },
            "trends": trend_info,
            "pools": pool_stats,
            "leak_detection": leak_report,
            "gc_info": {
                "collections": current_stats.gc_collections,
                "thresholds": gc.get_threshold(),
                "enabled": gc.isenabled(),
            },
            "tracked_objects": len(self.weak_refs),
            "recommendations": self._generate_recommendations(current_stats),
        }

    def _generate_recommendations(self, stats: MemoryStats) -> list[str]:
        """Generate memory optimization recommendations."""
        recommendations = []

        if stats.memory_pressure > 0.7:
            recommendations.append("High memory pressure - consider increasing system RAM")

        if stats.rss_mb > 1800:  # Close to 2GB limit
            recommendations.append("Memory usage approaching limit - enable emergency cleanup")

        # Check pool efficiency
        for name, utilization in stats.pool_utilization.items():
            if utilization < 0.3:
                recommendations.append(f"Low utilization in {name} pool - consider reducing size")
            elif utilization > 0.9:
                recommendations.append(
                    f"High utilization in {name} pool - consider increasing size"
                )

        if not recommendations:
            recommendations.append("Memory usage is within optimal parameters")

        return recommendations

    async def stop_monitoring(self):
        """Stop memory monitoring."""
        self.is_running = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        self.leak_detector.stop()

        self.logger.info("Memory monitoring stopped")

    def cleanup(self):
        """Clean up memory manager."""
        # Clear all pools
        for pool in self.pools.values():
            pool.clear()

        # Clear tracking
        self.weak_refs.clear()
        self.stats_history.clear()

        # Restore original GC settings
        gc.set_threshold(*self.gc_thresholds)
        if not gc.isenabled():
            gc.enable()

        self.logger.info("Memory manager cleaned up")


# Global memory manager instance
_memory_manager: HighPerformanceMemoryManager | None = None


def initialize_memory_manager(config: Config) -> HighPerformanceMemoryManager:
    """Initialize global memory manager."""
    global _memory_manager
    _memory_manager = HighPerformanceMemoryManager(config)
    return _memory_manager


def get_memory_manager() -> HighPerformanceMemoryManager | None:
    """Get global memory manager instance."""
    return _memory_manager


def borrow_dict() -> dict:
    """Convenience function to borrow dictionary from pool."""
    if _memory_manager:
        return _memory_manager.borrow_object("dict") or {}
    return {}


def return_dict(obj: dict):
    """Convenience function to return dictionary to pool."""
    if _memory_manager:
        _memory_manager.return_object("dict", obj)


def borrow_list() -> list:
    """Convenience function to borrow list from pool."""
    if _memory_manager:
        return _memory_manager.borrow_object("list") or []
    return []


def return_list(obj: list):
    """Convenience function to return list to pool."""
    if _memory_manager:
        _memory_manager.return_object("list", obj)
