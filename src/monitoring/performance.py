"""
Performance optimization and profiling tools for T-Bot Trading System.

This module implements P-032: Performance Optimization with:
- Performance profiling tools and endpoints
- Query optimization and analysis
- Caching strategies and optimization
- Resource optimization and monitoring
- Bottleneck identification and resolution

Key Features:
- Real-time performance monitoring
- Code profiling and analysis
- Database query optimization
- Memory usage tracking
- Cache performance optimization
- Async operation optimization
"""

import asyncio
import cProfile
import gc
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Any

import psutil

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""

    function_name: str
    module_name: str
    execution_time: float
    memory_usage: int
    cpu_time: float
    call_count: int
    timestamp: datetime
    thread_id: int
    additional_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryMetrics:
    """Database query performance metrics."""

    query: str
    execution_time: float
    rows_affected: int
    database: str
    timestamp: datetime
    trace_id: str | None = None
    slow_query: bool = False


@dataclass
class CacheMetrics:
    """Cache performance metrics."""

    cache_name: str
    operation: str  # get, set, delete, clear
    hit: bool
    execution_time: float
    key_size: int
    value_size: int
    timestamp: datetime


class PerformanceProfiler:
    """
    Advanced performance profiler for T-Bot trading system.

    Provides comprehensive performance monitoring with minimal overhead,
    focusing on trading-critical operations.
    """

    def __init__(self, enable_memory_tracking: bool = True, enable_cpu_profiling: bool = True):
        """
        Initialize performance profiler.

        Args:
            enable_memory_tracking: Enable memory usage tracking
            enable_cpu_profiling: Enable CPU profiling
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_cpu_profiling = enable_cpu_profiling

        # Performance data storage
        self._performance_data: deque = deque(maxlen=10000)
        self._query_data: deque = deque(maxlen=5000)
        self._cache_data: deque = deque(maxlen=5000)

        # Profiling state
        self._active_profiles: dict[str, cProfile.Profile] = {}
        self._memory_snapshots: dict[str, Any] = {}

        # Threading
        self._lock = threading.RLock()

        # Performance thresholds
        self.slow_query_threshold = 1.0  # seconds
        self.memory_growth_threshold = 100 * 1024 * 1024  # 100MB
        self.cpu_time_threshold = 0.5  # seconds

        # Background monitoring
        self._monitoring_active = False
        self._monitoring_task: asyncio.Task | None = None

        # Initialize memory tracking
        if self.enable_memory_tracking:
            tracemalloc.start()

        logger.info("PerformanceProfiler initialized")

    async def start_monitoring(self) -> None:
        """Start background performance monitoring."""
        if self._monitoring_active:
            logger.warning("Performance monitoring already active")
            return

        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop background performance monitoring."""
        self._monitoring_active = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        logger.info("Performance monitoring stopped")

    @contextmanager
    def profile_function(self, function_name: str, module_name: str = ""):
        """
        Context manager for profiling function execution.

        Args:
            function_name: Name of the function being profiled
            module_name: Module name

        Yields:
            Profiling context
        """
        start_time = time.time()
        start_cpu = time.process_time()

        # Memory tracking
        memory_before = 0
        if self.enable_memory_tracking:
            try:
                memory_before = tracemalloc.get_traced_memory()[0]
            except Exception:
                pass

        # CPU profiling
        profiler = None
        if self.enable_cpu_profiling:
            profiler = cProfile.Profile()
            profiler.enable()

        try:
            yield
        finally:
            # Stop profiling
            if profiler:
                profiler.disable()

            # Calculate metrics
            execution_time = time.time() - start_time
            cpu_time = time.process_time() - start_cpu

            memory_after = 0
            if self.enable_memory_tracking:
                try:
                    memory_after = tracemalloc.get_traced_memory()[0]
                except Exception:
                    pass

            memory_usage = max(0, memory_after - memory_before)

            # Store metrics
            metrics = PerformanceMetrics(
                function_name=function_name,
                module_name=module_name,
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_time=cpu_time,
                call_count=1,
                timestamp=datetime.now(),
                thread_id=threading.get_ident(),
            )

            with self._lock:
                self._performance_data.append(metrics)

            # Log slow operations
            if execution_time > self.cpu_time_threshold:
                logger.warning(
                    f"Slow operation detected: {function_name} took {execution_time:.3f}s"
                )

    def profile_async_function(self, function_name: str = "", module_name: str = ""):
        """
        Decorator for profiling async functions.

        Args:
            function_name: Override function name
            module_name: Module name

        Returns:
            Decorated function
        """

        def decorator(func: Callable):
            actual_function_name = function_name or func.__name__
            actual_module_name = module_name or func.__module__

            @wraps(func)
            async def wrapper(*args, **kwargs):
                with self.profile_function(actual_function_name, actual_module_name):
                    return await func(*args, **kwargs)

            return wrapper

        return decorator

    def profile_sync_function(self, function_name: str = "", module_name: str = ""):
        """
        Decorator for profiling synchronous functions.

        Args:
            function_name: Override function name
            module_name: Module name

        Returns:
            Decorated function
        """

        def decorator(func: Callable):
            actual_function_name = function_name or func.__name__
            actual_module_name = module_name or func.__module__

            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile_function(actual_function_name, actual_module_name):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def record_query_performance(
        self,
        query: str,
        execution_time: float,
        rows_affected: int,
        database: str,
        trace_id: str | None = None,
    ) -> None:
        """
        Record database query performance.

        Args:
            query: SQL query or operation description
            execution_time: Query execution time in seconds
            rows_affected: Number of rows affected
            database: Database name
            trace_id: Optional trace ID for correlation
        """
        is_slow = execution_time > self.slow_query_threshold

        metrics = QueryMetrics(
            query=query[:1000],  # Truncate long queries
            execution_time=execution_time,
            rows_affected=rows_affected,
            database=database,
            timestamp=datetime.now(),
            trace_id=trace_id,
            slow_query=is_slow,
        )

        with self._lock:
            self._query_data.append(metrics)

        if is_slow:
            logger.warning(f"Slow query detected: {execution_time:.3f}s - {query[:100]}...")

    def record_cache_performance(
        self,
        cache_name: str,
        operation: str,
        hit: bool,
        execution_time: float,
        key_size: int = 0,
        value_size: int = 0,
    ) -> None:
        """
        Record cache performance metrics.

        Args:
            cache_name: Name of the cache
            operation: Cache operation (get, set, delete, clear)
            hit: Whether it was a cache hit
            execution_time: Operation execution time
            key_size: Size of the key in bytes
            value_size: Size of the value in bytes
        """
        metrics = CacheMetrics(
            cache_name=cache_name,
            operation=operation,
            hit=hit,
            execution_time=execution_time,
            key_size=key_size,
            value_size=value_size,
            timestamp=datetime.now(),
        )

        with self._lock:
            self._cache_data.append(metrics)

    def get_performance_summary(self, timeframe_minutes: int = 60) -> dict[str, Any]:
        """
        Get performance summary for the specified timeframe.

        Args:
            timeframe_minutes: Timeframe in minutes

        Returns:
            Performance summary dictionary
        """
        cutoff_time = datetime.now() - timedelta(minutes=timeframe_minutes)

        with self._lock:
            # Filter recent data
            recent_performance = [d for d in self._performance_data if d.timestamp >= cutoff_time]
            recent_queries = [d for d in self._query_data if d.timestamp >= cutoff_time]
            recent_cache = [d for d in self._cache_data if d.timestamp >= cutoff_time]

        # Calculate function performance stats
        function_stats = defaultdict(
            lambda: {
                "call_count": 0,
                "total_time": 0.0,
                "total_memory": 0,
                "avg_time": 0.0,
                "max_time": 0.0,
            }
        )

        for perf in recent_performance:
            key = f"{perf.module_name}.{perf.function_name}"
            stats = function_stats[key]
            stats["call_count"] += 1
            stats["total_time"] += perf.execution_time
            stats["total_memory"] += perf.memory_usage
            stats["max_time"] = max(stats["max_time"], perf.execution_time)

        # Calculate averages
        for stats in function_stats.values():
            if stats["call_count"] > 0:
                stats["avg_time"] = stats["total_time"] / stats["call_count"]

        # Query performance stats
        total_queries = len(recent_queries)
        slow_queries = len([q for q in recent_queries if q.slow_query])
        avg_query_time = (
            sum(q.execution_time for q in recent_queries) / total_queries
            if total_queries > 0
            else 0
        )

        # Cache performance stats
        total_cache_ops = len(recent_cache)
        cache_hits = len([c for c in recent_cache if c.hit])
        hit_rate = (cache_hits / total_cache_ops * 100) if total_cache_ops > 0 else 0

        return {
            "timeframe_minutes": timeframe_minutes,
            "function_performance": dict(function_stats),
            "query_performance": {
                "total_queries": total_queries,
                "slow_queries": slow_queries,
                "slow_query_percentage": (
                    (slow_queries / total_queries * 100) if total_queries > 0 else 0
                ),
                "average_query_time": avg_query_time,
            },
            "cache_performance": {
                "total_operations": total_cache_ops,
                "cache_hits": cache_hits,
                "hit_rate_percentage": hit_rate,
            },
            "system_resources": self._get_system_resources(),
        }

    def get_slow_queries(self, limit: int = 10) -> list[QueryMetrics]:
        """
        Get slowest queries.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of slow query metrics
        """
        with self._lock:
            slow_queries = [q for q in self._query_data if q.slow_query]

        # Sort by execution time (descending)
        slow_queries.sort(key=lambda x: x.execution_time, reverse=True)
        return slow_queries[:limit]

    def get_memory_usage_report(self) -> dict[str, Any]:
        """
        Get detailed memory usage report.

        Returns:
            Memory usage report
        """
        if not self.enable_memory_tracking:
            return {"error": "Memory tracking not enabled"}

        try:
            current, peak = tracemalloc.get_traced_memory()
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")

            # Process info
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                "traced_memory": {
                    "current_bytes": current,
                    "peak_bytes": peak,
                    "current_mb": current / 1024 / 1024,
                    "peak_mb": peak / 1024 / 1024,
                },
                "process_memory": {
                    "rss_bytes": memory_info.rss,
                    "vms_bytes": memory_info.vms,
                    "rss_mb": memory_info.rss / 1024 / 1024,
                    "vms_mb": memory_info.vms / 1024 / 1024,
                },
                "top_allocations": [
                    {
                        "filename": stat.traceback.format()[0],
                        "size_bytes": stat.size,
                        "size_mb": stat.size / 1024 / 1024,
                        "count": stat.count,
                    }
                    for stat in top_stats[:10]
                ],
                "gc_stats": {"collections": gc.get_count(), "objects": len(gc.get_objects())},
            }

        except Exception as e:
            logger.error(f"Error generating memory report: {e}")
            return {"error": str(e)}

    def optimize_memory(self) -> dict[str, Any]:
        """
        Perform memory optimization operations.

        Returns:
            Optimization results
        """
        try:
            # Get memory before optimization
            memory_before = 0
            if self.enable_memory_tracking:
                memory_before = tracemalloc.get_traced_memory()[0]

            process = psutil.Process()
            rss_before = process.memory_info().rss

            # Perform garbage collection
            collected = gc.collect()

            # Clear internal caches if they're too large
            if len(self._performance_data) > 5000:
                with self._lock:
                    # Keep only recent 50% of data
                    keep_count = len(self._performance_data) // 2
                    self._performance_data = deque(
                        list(self._performance_data)[-keep_count:], maxlen=10000
                    )

            # Get memory after optimization
            memory_after = 0
            if self.enable_memory_tracking:
                memory_after = tracemalloc.get_traced_memory()[0]

            rss_after = process.memory_info().rss

            memory_freed = max(0, memory_before - memory_after)
            rss_freed = max(0, rss_before - rss_after)

            result = {
                "gc_collected": collected,
                "memory_freed_bytes": memory_freed,
                "memory_freed_mb": memory_freed / 1024 / 1024,
                "rss_freed_bytes": rss_freed,
                "rss_freed_mb": rss_freed / 1024 / 1024,
                "performance_data_size": len(self._performance_data),
                "query_data_size": len(self._query_data),
                "cache_data_size": len(self._cache_data),
            }

            logger.info(
                f"Memory optimization completed: freed {memory_freed / 1024 / 1024:.2f}MB, "
                f"collected {collected} objects"
            )

            return result

        except Exception as e:
            logger.error(f"Error during memory optimization: {e}")
            return {"error": str(e)}

    def _get_system_resources(self) -> dict[str, Any]:
        """Get current system resource usage."""
        try:
            process = psutil.Process()

            # CPU usage
            cpu_percent = process.cpu_percent()

            # Memory usage
            memory_info = process.memory_info()

            # File descriptors (Unix only)
            fd_count = 0
            try:
                fd_count = process.num_fds()
            except AttributeError:
                pass  # Windows doesn't have num_fds

            # Thread count
            thread_count = process.num_threads()

            return {
                "cpu_percent": cpu_percent,
                "memory_rss_mb": memory_info.rss / 1024 / 1024,
                "memory_vms_mb": memory_info.vms / 1024 / 1024,
                "file_descriptors": fd_count,
                "thread_count": thread_count,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {"error": str(e)}

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                # Check for memory growth
                if self.enable_memory_tracking:
                    current_memory = tracemalloc.get_traced_memory()[0]
                    # Log if memory usage is high
                    if current_memory > self.memory_growth_threshold:
                        logger.warning(
                            f"High memory usage detected: {current_memory / 1024 / 1024:.2f}MB"
                        )

                # Periodic cleanup
                if len(self._performance_data) > 8000:
                    self.optimize_memory()

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(30)


class QueryOptimizer:
    """Database query optimization utilities."""

    def __init__(self, profiler: PerformanceProfiler):
        """
        Initialize query optimizer.

        Args:
            profiler: Performance profiler instance
        """
        self.profiler = profiler
        self._query_cache: dict[str, Any] = {}
        self._prepared_statements: dict[str, Any] = {}

    def analyze_slow_queries(self) -> list[dict[str, Any]]:
        """
        Analyze slow queries and provide optimization suggestions.

        Returns:
            List of query analysis results
        """
        slow_queries = self.profiler.get_slow_queries(limit=20)

        analysis_results = []
        for query_metric in slow_queries:
            suggestions = self._analyze_query(query_metric.query)

            analysis_results.append(
                {
                    "query": (
                        query_metric.query[:200] + "..."
                        if len(query_metric.query) > 200
                        else query_metric.query
                    ),
                    "execution_time": query_metric.execution_time,
                    "rows_affected": query_metric.rows_affected,
                    "database": query_metric.database,
                    "optimization_suggestions": suggestions,
                }
            )

        return analysis_results

    def _analyze_query(self, query: str) -> list[str]:
        """
        Analyze a query and provide optimization suggestions.

        Args:
            query: SQL query to analyze

        Returns:
            List of optimization suggestions
        """
        suggestions = []
        query_lower = query.lower()

        # Check for common anti-patterns
        if "select *" in query_lower:
            suggestions.append("Avoid SELECT *, specify only needed columns")

        if "where" not in query_lower and any(
            keyword in query_lower for keyword in ["update", "delete"]
        ):
            suggestions.append("Consider adding WHERE clause to limit affected rows")

        if "order by" in query_lower and "limit" not in query_lower:
            suggestions.append("Consider adding LIMIT clause with ORDER BY")

        if query_lower.count("join") > 3:
            suggestions.append("Consider reducing number of JOINs or using temporary tables")

        if "like '%%" in query_lower:
            suggestions.append("Avoid leading wildcards in LIKE clauses, consider full-text search")

        if "or" in query_lower:
            suggestions.append(
                "Consider rewriting OR conditions using UNION for better performance"
            )

        if not suggestions:
            suggestions.append("Query appears optimized, consider checking indexes")

        return suggestions


class CacheOptimizer:
    """Cache performance optimization utilities."""

    def __init__(self, profiler: PerformanceProfiler):
        """
        Initialize cache optimizer.

        Args:
            profiler: Performance profiler instance
        """
        self.profiler = profiler

    def analyze_cache_performance(self) -> dict[str, Any]:
        """
        Analyze cache performance and provide optimization recommendations.

        Returns:
            Cache performance analysis
        """
        # Get recent cache data
        cutoff_time = datetime.now() - timedelta(hours=1)

        with self.profiler._lock:
            recent_cache = [d for d in self.profiler._cache_data if d.timestamp >= cutoff_time]

        if not recent_cache:
            return {"error": "No cache data available"}

        # Analyze by cache name
        cache_stats = defaultdict(
            lambda: {
                "total_operations": 0,
                "hits": 0,
                "misses": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
            }
        )

        for cache_metric in recent_cache:
            stats = cache_stats[cache_metric.cache_name]
            stats["total_operations"] += 1
            stats["total_time"] += cache_metric.execution_time

            if cache_metric.hit:
                stats["hits"] += 1
            else:
                stats["misses"] += 1

        # Calculate derived metrics
        for cache_name, stats in cache_stats.items():
            if stats["total_operations"] > 0:
                stats["hit_rate"] = stats["hits"] / stats["total_operations"] * 100
                stats["avg_time"] = stats["total_time"] / stats["total_operations"]

        # Generate recommendations
        recommendations = []
        for cache_name, stats in cache_stats.items():
            if stats["hit_rate"] < 70:
                recommendations.append(
                    f"{cache_name}: Low hit rate ({stats['hit_rate']:.1f}%), "
                    "consider adjusting cache size or TTL"
                )

            if stats["avg_time"] > 0.01:  # 10ms
                recommendations.append(
                    f"{cache_name}: High average access time ({stats['avg_time'] * 1000:.1f}ms), "
                    "consider cache optimization"
                )

        return {
            "cache_statistics": dict(cache_stats),
            "recommendations": recommendations,
            "overall_metrics": {
                "total_operations": sum(s["total_operations"] for s in cache_stats.values()),
                "overall_hit_rate": (
                    sum(s["hits"] for s in cache_stats.values())
                    / sum(s["total_operations"] for s in cache_stats.values())
                    * 100
                    if sum(s["total_operations"] for s in cache_stats.values()) > 0
                    else 0
                ),
            },
        }


# Global profiler instance
_global_profiler: PerformanceProfiler | None = None


def get_performance_profiler() -> PerformanceProfiler | None:
    """
    Get the global performance profiler instance.

    Returns:
        Global PerformanceProfiler instance or None if not initialized
    """
    return _global_profiler


def set_global_profiler(profiler: PerformanceProfiler) -> None:
    """
    Set the global performance profiler instance.

    Args:
        profiler: PerformanceProfiler instance
    """
    global _global_profiler
    _global_profiler = profiler


def profile_async(function_name: str = "", module_name: str = ""):
    """
    Decorator for profiling async functions with global profiler.

    Args:
        function_name: Override function name
        module_name: Module name

    Returns:
        Decorated function
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            profiler = get_performance_profiler()
            if profiler:
                actual_function_name = function_name or func.__name__
                actual_module_name = module_name or func.__module__
                with profiler.profile_function(actual_function_name, actual_module_name):
                    return await func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def profile_sync(function_name: str = "", module_name: str = ""):
    """
    Decorator for profiling synchronous functions with global profiler.

    Args:
        function_name: Override function name
        module_name: Module name

    Returns:
        Decorated function
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_performance_profiler()
            if profiler:
                actual_function_name = function_name or func.__name__
                actual_module_name = module_name or func.__module__
                with profiler.profile_function(actual_function_name, actual_module_name):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator
