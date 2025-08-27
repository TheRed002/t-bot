"""Cache metrics and monitoring for performance tracking."""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from src.base import BaseComponent


@dataclass
class CacheStats:
    """Cache statistics for monitoring with memory tracking."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_time: float = 0.0
    avg_response_time: float = 0.0

    # Memory tracking
    memory_used_bytes: int = 0
    memory_allocated_bytes: int = 0
    cache_entries: int = 0

    # Cleanup tracking
    cleanups_performed: int = 0
    last_cleanup_time: float = field(default_factory=time.time)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate


class CacheMetrics(BaseComponent):
    """Cache metrics collector and reporter with memory accounting."""

    def __init__(self):
        super().__init__()
        self._stats: dict[str, CacheStats] = defaultdict(CacheStats)
        self._recent_operations: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )  # Reduced from 1000
        self._start_time = time.time()

        # Memory tracking
        self._memory_lock = threading.Lock()
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
        self._max_memory_bytes = 50 * 1024 * 1024  # 50MB memory limit

        # Register cleanup callback to prevent memory leaks
        self._cleanup_timer = None
        self._shutdown = False

    def record_hit(self, namespace: str, response_time: float = 0.0):
        """Record cache hit with memory accounting."""
        with self._memory_lock:
            stats = self._stats[namespace]
            stats.hits += 1
            stats.total_time += response_time
            total_ops = stats.hits + stats.misses + stats.sets
            if total_ops > 0:
                stats.avg_response_time = stats.total_time / total_ops

            self._record_operation(namespace, "hit", response_time)
            self._cleanup_if_needed()

    def _record_operation(self, namespace: str, op_type: str, response_time: float = 0.0, **kwargs):
        """Record operation with memory-bounded storage."""
        operation = {
            "type": op_type,
            "timestamp": time.time(),
            "response_time": response_time,
            **kwargs,
        }

        # Add to bounded deque - automatically removes oldest when full
        self._recent_operations[namespace].append(operation)

    def _cleanup_if_needed(self):
        """Perform cleanup if memory limits are approaching."""
        current_time = time.time()

        if current_time - self._last_cleanup > self._cleanup_interval:
            self._perform_cleanup()

    def _perform_cleanup(self):
        """Perform memory cleanup."""
        current_time = time.time()
        cleanup_count = 0

        # Clean old operations (older than 1 hour)
        cutoff_time = current_time - 3600

        for namespace in list(self._recent_operations.keys()):
            operations = self._recent_operations[namespace]

            # Remove old operations
            while operations and operations[0]["timestamp"] < cutoff_time:
                operations.popleft()
                cleanup_count += 1

        # Update cleanup stats
        for stats in self._stats.values():
            stats.cleanups_performed += 1
            stats.last_cleanup_time = current_time

        self._last_cleanup = current_time

        if cleanup_count > 0:
            self.logger.debug(f"Cleaned up {cleanup_count} old cache operations")

    def record_miss(self, namespace: str, response_time: float = 0.0):
        """Record cache miss with memory accounting."""
        with self._memory_lock:
            stats = self._stats[namespace]
            stats.misses += 1
            stats.total_time += response_time
            total_ops = stats.hits + stats.misses + stats.sets
            if total_ops > 0:
                stats.avg_response_time = stats.total_time / total_ops

            self._record_operation(namespace, "miss", response_time)
            self._cleanup_if_needed()

    def record_set(self, namespace: str, response_time: float = 0.0, memory_bytes: int = 0):
        """Record cache set operation with memory tracking."""
        with self._memory_lock:
            stats = self._stats[namespace]
            stats.sets += 1
            stats.total_time += response_time
            stats.cache_entries += 1
            stats.memory_used_bytes += memory_bytes

            total_ops = stats.hits + stats.misses + stats.sets
            if total_ops > 0:
                stats.avg_response_time = stats.total_time / total_ops

            self._record_operation(namespace, "set", response_time, memory_bytes=memory_bytes)
            self._cleanup_if_needed()

    def record_delete(self, namespace: str, response_time: float = 0.0, memory_freed: int = 0):
        """Record cache delete operation with memory tracking."""
        with self._memory_lock:
            stats = self._stats[namespace]
            stats.deletes += 1
            if stats.cache_entries > 0:
                stats.cache_entries -= 1
            stats.memory_used_bytes = max(0, stats.memory_used_bytes - memory_freed)

            self._record_operation(namespace, "delete", response_time, memory_freed=memory_freed)
            self._cleanup_if_needed()

    def record_error(self, namespace: str, error_type: str = "unknown"):
        """Record cache error with memory accounting."""
        with self._memory_lock:
            stats = self._stats[namespace]
            stats.errors += 1

            self._record_operation(namespace, "error", error_type=error_type)
            self._cleanup_if_needed()

    def get_stats(self, namespace: str | None = None) -> dict[str, Any]:
        """Get cache statistics."""
        if namespace:
            return {
                namespace: {
                    "hits": self._stats[namespace].hits,
                    "misses": self._stats[namespace].misses,
                    "sets": self._stats[namespace].sets,
                    "deletes": self._stats[namespace].deletes,
                    "errors": self._stats[namespace].errors,
                    "hit_rate": self._stats[namespace].hit_rate,
                    "miss_rate": self._stats[namespace].miss_rate,
                    "avg_response_time": self._stats[namespace].avg_response_time,
                }
            }

        # Return all stats
        result = {}
        for ns, stats in self._stats.items():
            result[ns] = {
                "hits": stats.hits,
                "misses": stats.misses,
                "sets": stats.sets,
                "deletes": stats.deletes,
                "errors": stats.errors,
                "hit_rate": stats.hit_rate,
                "miss_rate": stats.miss_rate,
                "avg_response_time": stats.avg_response_time,
            }

        # Add overall stats
        total_stats = CacheStats()
        for stats in self._stats.values():
            total_stats.hits += stats.hits
            total_stats.misses += stats.misses
            total_stats.sets += stats.sets
            total_stats.deletes += stats.deletes
            total_stats.errors += stats.errors
            total_stats.total_time += stats.total_time

        if total_stats.hits + total_stats.misses + total_stats.sets > 0:
            total_stats.avg_response_time = total_stats.total_time / (
                total_stats.hits + total_stats.misses + total_stats.sets
            )

        result["total"] = {
            "hits": total_stats.hits,
            "misses": total_stats.misses,
            "sets": total_stats.sets,
            "deletes": total_stats.deletes,
            "errors": total_stats.errors,
            "hit_rate": total_stats.hit_rate,
            "miss_rate": total_stats.miss_rate,
            "avg_response_time": total_stats.avg_response_time,
            "uptime_seconds": time.time() - self._start_time,
        }

        return result

    def get_recent_operations(self, namespace: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent cache operations for debugging."""
        operations = list(self._recent_operations.get(namespace, []))
        return operations[-limit:] if operations else []

    def get_performance_summary(self, time_window_minutes: int = 5) -> dict[str, Any]:
        """Get performance summary for the last N minutes."""
        cutoff_time = time.time() - (time_window_minutes * 60)
        summary = {}

        for namespace, operations in self._recent_operations.items():
            recent_ops = [op for op in operations if op["timestamp"] >= cutoff_time]

            if not recent_ops:
                continue

            hits = sum(1 for op in recent_ops if op["type"] == "hit")
            misses = sum(1 for op in recent_ops if op["type"] == "miss")
            sets = sum(1 for op in recent_ops if op["type"] == "set")
            errors = sum(1 for op in recent_ops if op["type"] == "error")

            response_times = [
                op.get("response_time", 0) for op in recent_ops if op.get("response_time", 0) > 0
            ]

            summary[namespace] = {
                "operations_count": len(recent_ops),
                "hits": hits,
                "misses": misses,
                "sets": sets,
                "errors": errors,
                "hit_rate": hits / (hits + misses) if (hits + misses) > 0 else 0,
                "avg_response_time": (
                    sum(response_times) / len(response_times) if response_times else 0
                ),
                "max_response_time": max(response_times) if response_times else 0,
                "min_response_time": min(response_times) if response_times else 0,
            }

        return summary

    def shutdown(self):
        """Shutdown metrics collector and cleanup resources."""
        with self._memory_lock:
            self._shutdown = True

            # Cancel any pending cleanup timer
            if self._cleanup_timer:
                self._cleanup_timer.cancel()
                self._cleanup_timer = None

            # Clear all data to free memory
            self._stats.clear()
            self._recent_operations.clear()

            self.logger.info("Cache metrics shutdown completed")

    def reset_stats(self, namespace: str | None = None):
        """Reset statistics."""
        if namespace:
            self._stats[namespace] = CacheStats()
            self._recent_operations[namespace].clear()
        else:
            self._stats.clear()
            self._recent_operations.clear()
            self._start_time = time.time()

    def export_metrics_for_monitoring(self) -> dict[str, Any]:
        """Export metrics in format suitable for monitoring systems."""
        stats = self.get_stats()
        metrics = {}

        for namespace, ns_stats in stats.items():
            if namespace == "total":
                prefix = "cache"
            else:
                prefix = f"cache_{namespace}"

            metrics.update(
                {
                    f"{prefix}_hits_total": ns_stats["hits"],
                    f"{prefix}_misses_total": ns_stats["misses"],
                    f"{prefix}_sets_total": ns_stats["sets"],
                    f"{prefix}_deletes_total": ns_stats["deletes"],
                    f"{prefix}_errors_total": ns_stats["errors"],
                    f"{prefix}_hit_rate": ns_stats["hit_rate"],
                    f"{prefix}_miss_rate": ns_stats["miss_rate"],
                    f"{prefix}_avg_response_time_seconds": ns_stats["avg_response_time"],
                }
            )

        return metrics


# Global metrics instance
_metrics_instance: CacheMetrics | None = None


def get_cache_metrics() -> CacheMetrics:
    """Get or create global cache metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = CacheMetrics()
    return _metrics_instance
