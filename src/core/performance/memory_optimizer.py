"""
Memory Optimization and Garbage Collection Management

This module provides comprehensive memory optimization for the T-Bot trading system,
focusing on minimizing memory footprint while maintaining high-performance trading operations.

Features:
- Intelligent garbage collection tuning
- Memory pool management for frequent allocations
- Object lifecycle tracking and optimization
- Memory leak detection and prevention
- Trading-specific memory patterns optimization
- Real-time memory monitoring and alerts

Performance targets:
- Memory usage < 2GB per bot instance
- Garbage collection pause < 10ms for critical operations
- Memory allocation rate optimized for trading frequency
- Memory leak detection within 5 minutes
"""

import asyncio
import gc
import time
import tracemalloc
import weakref
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import psutil

from src.base import BaseComponent
from src.core.config import Config
from src.core.exceptions import PerformanceError
from src.core.logging import get_logger
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class MemoryCategory(Enum):
    """Categories of memory usage for optimization."""

    MARKET_DATA = "market_data"  # Real-time market data buffers
    TRADING_OBJECTS = "trading_objects"  # Orders, positions, trades
    CACHE_DATA = "cache_data"  # Various cache levels
    ML_MODELS = "ml_models"  # Machine learning models and features
    NETWORK_BUFFERS = "network_buffers"  # WebSocket and HTTP buffers
    TEMPORARY_DATA = "temporary_data"  # Short-lived computational data


class GCStrategy(Enum):
    """Garbage collection strategies."""

    AGGRESSIVE = "aggressive"  # Frequent, small collections
    BALANCED = "balanced"  # Default Python behavior
    CONSERVATIVE = "conservative"  # Less frequent, larger collections
    TRADING_OPTIMIZED = "trading_optimized"  # Optimized for trading patterns


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_memory_mb: float = 0.0
    available_memory_mb: float = 0.0
    process_memory_mb: float = 0.0
    heap_size_mb: float = 0.0
    gc_collections: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0})
    gc_time_ms: float = 0.0
    memory_growth_rate_mb_min: float = 0.0
    leak_suspects: list[str] = field(default_factory=list)
    large_objects_count: int = 0
    fragmentation_ratio: float = 0.0


@dataclass
class ObjectPoolStats:
    """Statistics for object pools."""

    pool_name: str
    total_objects: int = 0
    active_objects: int = 0
    available_objects: int = 0
    peak_usage: int = 0
    allocation_count: int = 0
    deallocation_count: int = 0
    pool_hits: int = 0
    pool_misses: int = 0


class ObjectPool:
    """
    High-performance object pool for frequently allocated objects.

    Reduces memory allocation pressure by reusing objects for trading operations.
    """

    def __init__(self, factory: Callable, reset_func: Callable | None = None, max_size: int = 1000):
        """
        Initialize object pool.

        Args:
            factory: Function to create new objects
            reset_func: Function to reset objects before reuse
            max_size: Maximum pool size
        """
        self.factory = factory
        self.reset_func = reset_func
        self.max_size = max_size
        self._pool: deque = deque()
        self._active_objects: set[weakref.ref] = set()
        self.stats = ObjectPoolStats(
            pool_name=factory.__name__ if hasattr(factory, "__name__") else "unknown"
        )
        self._lock = asyncio.Lock()

    async def acquire(self) -> Any:
        """Acquire object from pool."""
        async with self._lock:
            if self._pool:
                obj = self._pool.popleft()
                if self.reset_func:
                    self.reset_func(obj)
                self.stats.pool_hits += 1
            else:
                obj = self.factory()
                self.stats.pool_misses += 1
                self.stats.allocation_count += 1

            # Track active objects with weak references
            weak_ref = weakref.ref(obj, self._object_finalizer)
            self._active_objects.add(weak_ref)

            self.stats.active_objects = len(self._active_objects)
            self.stats.available_objects = len(self._pool)
            self.stats.total_objects = self.stats.active_objects + self.stats.available_objects

            if self.stats.active_objects > self.stats.peak_usage:
                self.stats.peak_usage = self.stats.active_objects

            return obj

    async def release(self, obj: Any) -> None:
        """Release object back to pool."""
        async with self._lock:
            if len(self._pool) < self.max_size:
                self._pool.append(obj)
                self.stats.deallocation_count += 1

            # Remove from active tracking (will be cleaned up by weak reference callback)
            self.stats.active_objects = len(self._active_objects)
            self.stats.available_objects = len(self._pool)
            self.stats.total_objects = self.stats.active_objects + self.stats.available_objects

    def _object_finalizer(self, weak_ref: weakref.ref) -> None:
        """Called when tracked object is garbage collected."""
        self._active_objects.discard(weak_ref)

    def get_stats(self) -> ObjectPoolStats:
        """Get pool statistics."""
        return self.stats

    def clear(self) -> None:
        """Clear the pool."""
        self._pool.clear()
        self._active_objects.clear()
        self.stats.active_objects = 0
        self.stats.available_objects = 0
        self.stats.total_objects = 0


class MemoryProfiler:
    """
    Advanced memory profiler for detecting leaks and optimization opportunities.
    """

    def __init__(self):
        self.enabled = False
        self._snapshots: list[tuple[datetime, tracemalloc.Snapshot]] = []
        self._leak_threshold_mb = 50  # Alert if growth > 50MB
        self._monitoring_interval = 60  # seconds

    def start_profiling(self) -> None:
        """Start memory profiling."""
        if not self.enabled:
            tracemalloc.start()
            self.enabled = True
            logger.info("Memory profiling started")

    def stop_profiling(self) -> None:
        """Stop memory profiling."""
        if self.enabled:
            tracemalloc.stop()
            self.enabled = False
            logger.info("Memory profiling stopped")

    def take_snapshot(self, label: str | None = None) -> None:
        """Take memory snapshot."""
        if self.enabled:
            snapshot = tracemalloc.take_snapshot()
            self._snapshots.append((datetime.now(timezone.utc), snapshot))

            # Keep only recent snapshots
            if len(self._snapshots) > 50:
                self._snapshots = self._snapshots[-25:]

            if label:
                logger.debug(f"Memory snapshot taken: {label}")

    def detect_leaks(self) -> list[dict[str, Any]]:
        """Detect potential memory leaks."""
        if len(self._snapshots) < 2:
            return []

        leaks = []
        current_snapshot = self._snapshots[-1][1]
        previous_snapshot = self._snapshots[-2][1]

        # Compare snapshots
        top_stats = current_snapshot.compare_to(previous_snapshot, "lineno")

        for stat in top_stats[:10]:  # Top 10 growing allocations
            if stat.size_diff > self._leak_threshold_mb * 1024 * 1024:  # Convert MB to bytes
                leaks.append(
                    {
                        "location": str(stat.traceback),
                        "size_diff_mb": stat.size_diff / (1024 * 1024),
                        "count_diff": stat.count_diff,
                        "current_size_mb": stat.size / (1024 * 1024),
                    }
                )

        return leaks

    def get_top_allocations(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get top memory allocations."""
        if not self._snapshots:
            return []

        snapshot = self._snapshots[-1][1]
        top_stats = snapshot.statistics("lineno")

        allocations = []
        for stat in top_stats[:limit]:
            allocations.append(
                {
                    "location": str(stat.traceback),
                    "size_mb": stat.size / (1024 * 1024),
                    "count": stat.count,
                    "average_size_bytes": stat.size / stat.count if stat.count > 0 else 0,
                }
            )

        return allocations


class GarbageCollectionOptimizer:
    """
    Garbage collection optimizer for trading workloads.

    Optimizes GC timing and parameters to minimize impact on trading operations.
    """

    def __init__(self):
        self.strategy = GCStrategy.TRADING_OPTIMIZED
        self._gc_stats: list[dict[str, Any]] = []
        self._original_thresholds = gc.get_threshold()
        self._last_collection_time = time.time()
        self._trading_active = False
        self._gc_disabled_start = None

    def set_strategy(self, strategy: GCStrategy) -> None:
        """Set garbage collection strategy."""
        self.strategy = strategy
        self._apply_strategy()

    def _apply_strategy(self) -> None:
        """Apply current GC strategy."""
        if self.strategy == GCStrategy.AGGRESSIVE:
            # More frequent, smaller collections
            gc.set_threshold(500, 8, 8)
        elif self.strategy == GCStrategy.CONSERVATIVE:
            # Less frequent, larger collections
            gc.set_threshold(2000, 20, 20)
        elif self.strategy == GCStrategy.TRADING_OPTIMIZED:
            # Optimized for trading patterns
            gc.set_threshold(1000, 15, 10)
        else:
            # Balanced (default)
            gc.set_threshold(*self._original_thresholds)

        logger.info(f"GC strategy set to {self.strategy.value}")

    def disable_gc_during_trading(self) -> None:
        """Temporarily disable GC during critical trading operations."""
        if not self._trading_active:
            gc.disable()
            self._trading_active = True
            self._gc_disabled_start = time.time()
            logger.debug("GC disabled for trading operation")

    def enable_gc_after_trading(self) -> None:
        """Re-enable GC after trading operations."""
        if self._trading_active:
            gc.enable()
            self._trading_active = False

            # Force collection if disabled for too long
            if self._gc_disabled_start and (time.time() - self._gc_disabled_start) > 30:
                self.force_collection()

            self._gc_disabled_start = None
            logger.debug("GC re-enabled after trading operation")

    def force_collection(self) -> dict[str, Any]:
        """Force garbage collection and return statistics."""
        start_time = time.time()

        # Collect statistics before
        before_stats = {
            "gen0": gc.get_count()[0],
            "gen1": gc.get_count()[1],
            "gen2": gc.get_count()[2],
        }

        # Force collection
        collected = []
        for generation in range(3):
            collected.append(gc.collect(generation))

        collection_time = (time.time() - start_time) * 1000  # Convert to ms

        stats = {
            "timestamp": datetime.now(timezone.utc),
            "collection_time_ms": collection_time,
            "objects_collected": sum(collected),
            "before_counts": before_stats,
            "after_counts": gc.get_count(),
            "collections_by_gen": collected,
        }

        self._gc_stats.append(stats)

        # Keep only recent stats
        if len(self._gc_stats) > 1000:
            self._gc_stats = self._gc_stats[-500:]

        logger.debug(
            f"Forced GC completed in {collection_time:.2f}ms, collected {sum(collected)} objects"
        )

        return stats

    def get_gc_stats(self) -> dict[str, Any]:
        """Get garbage collection statistics."""
        recent_stats = self._gc_stats[-10:] if self._gc_stats else []

        if recent_stats:
            avg_time = sum(s["collection_time_ms"] for s in recent_stats) / len(recent_stats)
            total_collected = sum(s["objects_collected"] for s in recent_stats)
        else:
            avg_time = 0
            total_collected = 0

        return {
            "strategy": self.strategy.value,
            "current_thresholds": gc.get_threshold(),
            "current_counts": gc.get_count(),
            "gc_enabled": gc.isenabled(),
            "trading_active": self._trading_active,
            "recent_collections": len(recent_stats),
            "avg_collection_time_ms": avg_time,
            "total_objects_collected": total_collected,
            "detailed_stats": recent_stats,
        }


class MemoryOptimizer(BaseComponent):
    """
    Comprehensive memory optimizer for the T-Bot trading system.

    Provides intelligent memory management, garbage collection optimization,
    and leak detection for high-performance trading operations.
    """

    def __init__(self, config: Config):
        """Initialize memory optimizer."""
        super().__init__()
        self.config = config

        # Memory monitoring
        self.stats_history: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.memory_alert_threshold_mb = 1500  # Alert at 1.5GB
        self.memory_critical_threshold_mb = 1800  # Critical at 1.8GB

        # Object pools for common trading objects
        self.object_pools: dict[str, ObjectPool] = {}
        self._initialize_object_pools()

        # Memory profiler
        self.profiler = MemoryProfiler()

        # Garbage collection optimizer
        self.gc_optimizer = GarbageCollectionOptimizer()

        # Memory categories tracking
        self.category_usage: dict[MemoryCategory, float] = defaultdict(float)

        # Background monitoring
        self._monitoring_task: asyncio.Task | None = None
        self._monitoring_interval = 60  # seconds

        # Memory alerts
        self._alert_callbacks: list[Callable] = []
        self._last_alert_time = 0
        self._alert_cooldown = 300  # 5 minutes

    def _initialize_object_pools(self) -> None:
        """Initialize object pools for common trading objects."""
        # Dictionary pool for market data
        self.object_pools["market_data_dict"] = ObjectPool(
            factory=dict, reset_func=lambda d: d.clear(), max_size=1000
        )

        # List pool for order book levels
        self.object_pools["orderbook_levels"] = ObjectPool(
            factory=list, reset_func=lambda lst: lst.clear(), max_size=500
        )

        # Float list pool for price calculations
        self.object_pools["price_calculations"] = ObjectPool(
            factory=lambda: [0.0] * 100,  # Pre-sized float list
            reset_func=lambda price_list: [0.0] * len(price_list),
            max_size=200,
        )

    async def initialize(self) -> None:
        """Initialize memory optimizer."""
        try:
            self.logger.info("Initializing memory optimizer...")

            # Start memory profiling in development
            if getattr(self.config, "development_mode", False):
                self.profiler.start_profiling()

            # Set optimal GC strategy
            self.gc_optimizer.set_strategy(GCStrategy.TRADING_OPTIMIZED)

            # Start background monitoring
            await self._start_monitoring()

            # Take initial memory snapshot
            await self._collect_memory_stats()

            self.logger.info("Memory optimizer initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize memory optimizer: {e}")
            raise PerformanceError(f"Memory optimizer initialization failed: {e}") from e

    async def _start_monitoring(self) -> None:
        """Start background memory monitoring."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Memory monitoring started")

    async def _monitoring_loop(self) -> None:
        """Background loop for memory monitoring."""
        while True:
            try:
                await asyncio.sleep(self._monitoring_interval)

                # Collect memory statistics
                await self._collect_memory_stats()

                # Check for alerts
                await self._check_memory_alerts()

                # Detect memory leaks
                await self._check_memory_leaks()

                # Optimize if needed
                await self._auto_optimize()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")

    @time_execution
    async def _collect_memory_stats(self) -> MemoryStats:
        """Collect comprehensive memory statistics."""
        try:
            # System memory
            system_memory = psutil.virtual_memory()

            # Process memory
            process = psutil.Process()
            process_memory = process.memory_info()

            # Garbage collection stats
            gc_stats = self.gc_optimizer.get_gc_stats()

            # Create stats object
            stats = MemoryStats(
                total_memory_mb=system_memory.total / (1024 * 1024),
                available_memory_mb=system_memory.available / (1024 * 1024),
                process_memory_mb=process_memory.rss / (1024 * 1024),
                gc_collections=gc_stats["current_counts"],
                gc_time_ms=gc_stats.get("avg_collection_time_ms", 0),
            )

            # Calculate memory growth rate
            if len(self.stats_history) > 0:
                prev_stats = self.stats_history[-1]
                time_diff = (stats.timestamp - prev_stats.timestamp).total_seconds() / 60  # minutes
                if time_diff > 0:
                    memory_diff = stats.process_memory_mb - prev_stats.process_memory_mb
                    stats.memory_growth_rate_mb_min = memory_diff / time_diff

            # Check for potential leaks
            if self.profiler.enabled:
                stats.leak_suspects = [leak["location"] for leak in self.profiler.detect_leaks()]

            # Add to history
            self.stats_history.append(stats)

            return stats

        except Exception as e:
            self.logger.error(f"Failed to collect memory stats: {e}")
            return MemoryStats()

    async def _check_memory_alerts(self) -> None:
        """Check for memory usage alerts."""
        if not self.stats_history:
            return

        current_stats = self.stats_history[-1]
        current_time = time.time()

        # Check if we need to alert (with cooldown)
        if current_time - self._last_alert_time < self._alert_cooldown:
            return

        alert_level = None
        message = None

        if current_stats.process_memory_mb > self.memory_critical_threshold_mb:
            alert_level = "critical"
            message = f"Critical memory usage: {current_stats.process_memory_mb:.1f}MB"
        elif current_stats.process_memory_mb > self.memory_alert_threshold_mb:
            alert_level = "warning"
            message = f"High memory usage: {current_stats.process_memory_mb:.1f}MB"
        elif current_stats.memory_growth_rate_mb_min > 5:  # Growing > 5MB/min
            alert_level = "warning"
            message = (
                f"High memory growth rate: {current_stats.memory_growth_rate_mb_min:.1f}MB/min"
            )

        if alert_level:
            self.logger.warning(
                message,
                extra={
                    "alert_level": alert_level,
                    "memory_mb": current_stats.process_memory_mb,
                    "growth_rate_mb_min": current_stats.memory_growth_rate_mb_min,
                },
            )

            # Notify alert callbacks
            for callback in self._alert_callbacks:
                try:
                    await callback(alert_level, message, current_stats)
                except Exception as e:
                    self.logger.error(f"Alert callback error: {e}")

            self._last_alert_time = current_time

    async def _check_memory_leaks(self) -> None:
        """Check for potential memory leaks."""
        if not self.profiler.enabled:
            return

        # Take periodic snapshots
        self.profiler.take_snapshot()

        # Detect leaks
        leaks = self.profiler.detect_leaks()

        if leaks:
            self.logger.warning(
                f"Detected {len(leaks)} potential memory leaks",
                extra={
                    "leak_count": len(leaks),
                    "top_leak_size_mb": leaks[0]["size_diff_mb"] if leaks else 0,
                    "leak_locations": [
                        leak["location"][:100] for leak in leaks[:3]
                    ],  # Top 3, truncated
                },
            )

    async def _auto_optimize(self) -> None:
        """Perform automatic memory optimization."""
        if not self.stats_history:
            return

        current_stats = self.stats_history[-1]

        # Auto-optimize if memory usage is high
        if current_stats.process_memory_mb > self.memory_alert_threshold_mb:
            self.logger.info("Auto-optimizing memory due to high usage")

            # Force garbage collection
            gc_stats = self.gc_optimizer.force_collection()

            # Clear unused object pools
            await self._optimize_object_pools()

            # Log optimization results
            self.logger.info(
                "Memory optimization completed",
                extra={
                    "objects_collected": gc_stats["objects_collected"],
                    "collection_time_ms": gc_stats["collection_time_ms"],
                },
            )

    async def acquire_pooled_object(self, pool_name: str) -> Any:
        """Acquire object from specified pool."""
        if pool_name in self.object_pools:
            return await self.object_pools[pool_name].acquire()
        else:
            raise ValueError(f"Unknown object pool: {pool_name}")

    async def release_pooled_object(self, pool_name: str, obj: Any) -> None:
        """Release object back to specified pool."""
        if pool_name in self.object_pools:
            await self.object_pools[pool_name].release(obj)
        else:
            raise ValueError(f"Unknown object pool: {pool_name}")

    async def _optimize_object_pools(self) -> None:
        """Optimize object pools by clearing unused objects."""
        for pool_name, pool in self.object_pools.items():
            # Clear pool if utilization is low
            stats = pool.get_stats()
            if stats.total_objects > 0:
                utilization = stats.active_objects / stats.total_objects
                if utilization < 0.1:  # Less than 10% utilization
                    pool.clear()
                    self.logger.debug(f"Cleared under-utilized object pool: {pool_name}")

    def track_category_usage(self, category: MemoryCategory, size_mb: float) -> None:
        """Track memory usage by category."""
        self.category_usage[category] = size_mb

    def add_alert_callback(self, callback: Callable) -> None:
        """Add callback for memory alerts."""
        self._alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable) -> None:
        """Remove alert callback."""
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)

    async def optimize_for_trading_operation(self) -> None:
        """Optimize memory for upcoming trading operation."""
        try:
            # Disable GC during critical operations
            self.gc_optimizer.disable_gc_during_trading()

            # Pre-allocate common objects
            for pool_name in ["market_data_dict", "orderbook_levels"]:
                if pool_name in self.object_pools:
                    pool = self.object_pools[pool_name]
                    # Pre-warm pool
                    for _ in range(10):
                        obj = await pool.acquire()
                        await pool.release(obj)

            self.logger.debug("Memory optimized for trading operation")

        except Exception as e:
            self.logger.error(f"Trading memory optimization failed: {e}")

    async def cleanup_after_trading_operation(self) -> None:
        """Cleanup memory after trading operation."""
        try:
            # Re-enable GC
            self.gc_optimizer.enable_gc_after_trading()

            # Clear temporary data
            gc.collect(0)  # Quick generation 0 collection

            self.logger.debug("Memory cleaned up after trading operation")

        except Exception as e:
            self.logger.error(f"Trading memory cleanup failed: {e}")

    async def get_memory_report(self) -> dict[str, Any]:
        """Get comprehensive memory usage report."""
        current_stats = self.stats_history[-1] if self.stats_history else MemoryStats()

        # Object pool statistics
        pool_stats = {}
        for pool_name, pool in self.object_pools.items():
            pool_stats[pool_name] = pool.get_stats().__dict__

        # Top memory allocations
        top_allocations = self.profiler.get_top_allocations() if self.profiler.enabled else []

        # Memory growth trend
        growth_trend = []
        if len(self.stats_history) >= 10:
            recent_stats = list(self.stats_history)[-10:]
            for i in range(1, len(recent_stats)):
                time_diff = (
                    recent_stats[i].timestamp - recent_stats[i - 1].timestamp
                ).total_seconds()
                memory_diff = (
                    recent_stats[i].process_memory_mb - recent_stats[i - 1].process_memory_mb
                )
                growth_rate = (memory_diff / time_diff) * 60 if time_diff > 0 else 0  # MB/min
                growth_trend.append(growth_rate)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_stats": current_stats.__dict__,
            "memory_alerts": {
                "alert_threshold_mb": self.memory_alert_threshold_mb,
                "critical_threshold_mb": self.memory_critical_threshold_mb,
                "current_level": self._get_current_alert_level(current_stats),
            },
            "object_pools": pool_stats,
            "gc_stats": self.gc_optimizer.get_gc_stats(),
            "category_usage": {cat.value: usage for cat, usage in self.category_usage.items()},
            "top_allocations": top_allocations,
            "memory_growth_trend": growth_trend,
            "profiling_enabled": self.profiler.enabled,
            "stats_history_length": len(self.stats_history),
        }

    def _get_current_alert_level(self, stats: MemoryStats) -> str:
        """Get current alert level based on memory usage."""
        if stats.process_memory_mb > self.memory_critical_threshold_mb:
            return "critical"
        elif stats.process_memory_mb > self.memory_alert_threshold_mb:
            return "warning"
        else:
            return "normal"

    async def force_memory_optimization(self) -> dict[str, Any]:
        """Force comprehensive memory optimization."""
        self.logger.info("Starting forced memory optimization")

        optimization_results = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "initial_memory_mb": 0,
            "final_memory_mb": 0,
            "memory_freed_mb": 0,
            "gc_stats": {},
            "pools_optimized": 0,
        }

        try:
            # Get initial memory usage
            initial_stats = await self._collect_memory_stats()
            optimization_results["initial_memory_mb"] = initial_stats.process_memory_mb

            # Force garbage collection
            gc_stats = self.gc_optimizer.force_collection()
            optimization_results["gc_stats"] = gc_stats

            # Optimize object pools
            await self._optimize_object_pools()
            optimization_results["pools_optimized"] = len(self.object_pools)

            # Additional manual cleanup
            import sys

            sys.intern.clear() if hasattr(sys, "intern") else None

            # Final memory measurement
            final_stats = await self._collect_memory_stats()
            optimization_results["final_memory_mb"] = final_stats.process_memory_mb
            optimization_results["memory_freed_mb"] = (
                initial_stats.process_memory_mb - final_stats.process_memory_mb
            )

            freed_mb = optimization_results["memory_freed_mb"]
            self.logger.info(
                f"Memory optimization completed, freed {freed_mb:.1f}MB",
                extra=optimization_results,
            )

            return optimization_results

        except Exception as e:
            self.logger.error(f"Forced memory optimization failed: {e}")
            optimization_results["error"] = str(e)
            return optimization_results

    async def cleanup(self) -> None:
        """Cleanup memory optimizer resources."""
        try:
            # Cancel monitoring task
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass

            # Stop profiling
            self.profiler.stop_profiling()

            # Clear object pools
            for pool in self.object_pools.values():
                pool.clear()

            # Clear statistics history
            self.stats_history.clear()

            # Re-enable GC if disabled
            self.gc_optimizer.enable_gc_after_trading()

            self.logger.info("Memory optimizer cleaned up")

        except Exception as e:
            self.logger.error(f"Memory optimizer cleanup error: {e}")


# Context managers for trading operations
class TradingMemoryContext:
    """Context manager for trading operations with memory optimization."""

    def __init__(self, memory_optimizer: MemoryOptimizer):
        self.memory_optimizer = memory_optimizer

    async def __aenter__(self):
        await self.memory_optimizer.optimize_for_trading_operation()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.memory_optimizer.cleanup_after_trading_operation()


# Decorators for memory-optimized functions
def memory_optimized_trading_operation(memory_optimizer: MemoryOptimizer):
    """Decorator for trading operations that need memory optimization."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with TradingMemoryContext(memory_optimizer):
                return await func(*args, **kwargs)

        return wrapper

    return decorator
