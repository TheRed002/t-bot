"""Cache metrics and monitoring for performance tracking."""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from src.base import BaseComponent


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_time: float = 0.0
    avg_response_time: float = 0.0

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
    """Cache metrics collector and reporter."""

    def __init__(self):
        super().__init__()
        self._stats: dict[str, CacheStats] = defaultdict(CacheStats)
        self._recent_operations: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._start_time = time.time()

    def record_hit(self, namespace: str, response_time: float = 0.0):
        """Record cache hit."""
        stats = self._stats[namespace]
        stats.hits += 1
        stats.total_time += response_time
        stats.avg_response_time = stats.total_time / (stats.hits + stats.misses + stats.sets)

        self._recent_operations[namespace].append(
            {"type": "hit", "timestamp": time.time(), "response_time": response_time}
        )

    def record_miss(self, namespace: str, response_time: float = 0.0):
        """Record cache miss."""
        stats = self._stats[namespace]
        stats.misses += 1
        stats.total_time += response_time
        stats.avg_response_time = stats.total_time / (stats.hits + stats.misses + stats.sets)

        self._recent_operations[namespace].append(
            {"type": "miss", "timestamp": time.time(), "response_time": response_time}
        )

    def record_set(self, namespace: str, response_time: float = 0.0):
        """Record cache set operation."""
        stats = self._stats[namespace]
        stats.sets += 1
        stats.total_time += response_time
        stats.avg_response_time = stats.total_time / (stats.hits + stats.misses + stats.sets)

        self._recent_operations[namespace].append(
            {"type": "set", "timestamp": time.time(), "response_time": response_time}
        )

    def record_delete(self, namespace: str, response_time: float = 0.0):
        """Record cache delete operation."""
        stats = self._stats[namespace]
        stats.deletes += 1

        self._recent_operations[namespace].append(
            {"type": "delete", "timestamp": time.time(), "response_time": response_time}
        )

    def record_error(self, namespace: str, error_type: str = "unknown"):
        """Record cache error."""
        stats = self._stats[namespace]
        stats.errors += 1

        self._recent_operations[namespace].append(
            {"type": "error", "timestamp": time.time(), "error_type": error_type}
        )

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
