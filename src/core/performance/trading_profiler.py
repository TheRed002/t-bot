"""
Critical Trading Operations Profiler and Optimizer

This module provides specialized profiling and optimization for time-critical
trading operations in the T-Bot system. It focuses on achieving ultra-low latency
(<100ms) for order execution, market data processing, and risk calculations.

Features:
- Microsecond-precision timing for trading operations
- Hot path identification and optimization
- Order execution pipeline profiling
- Market data processing optimization
- Risk calculation performance tuning
- Memory allocation tracking for trading paths
- CPU cache optimization recommendations
- Real-time optimization during trading hours
- Performance regression detection
- Trading-specific benchmarking

Performance targets:
- Order placement: < 50ms (99th percentile)
- Market data processing: < 10ms (99th percentile)
- Risk calculations: < 25ms (99th percentile)
- Position updates: < 15ms (99th percentile)
- Portfolio calculations: < 30ms (99th percentile)
"""

import asyncio
import cProfile
import functools
import io
import pstats
import time
import tracemalloc
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from src.core.base.component import BaseComponent
from src.core.exceptions import PerformanceError
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.core.config import Config
    from src.core.performance.performance_monitor import PerformanceMonitor

logger = get_logger(__name__)


class TradingOperation(Enum):
    """Specific trading operations that require optimization."""

    ORDER_PLACEMENT = "order_placement"
    ORDER_CANCELLATION = "order_cancellation"
    ORDER_MODIFICATION = "order_modification"
    MARKET_DATA_TICK = "market_data_tick"
    ORDERBOOK_UPDATE = "orderbook_update"
    POSITION_UPDATE = "position_update"
    PORTFOLIO_CALCULATION = "portfolio_calculation"
    RISK_CHECK = "risk_check"
    PNL_CALCULATION = "pnl_calculation"
    STOP_LOSS_TRIGGER = "stop_loss_trigger"
    ARBITRAGE_CALCULATION = "arbitrage_calculation"
    ML_PREDICTION = "ml_prediction"


class OptimizationLevel(Enum):
    """Levels of optimization to apply."""

    DEVELOPMENT = "development"  # Comprehensive profiling, all optimizations
    PRODUCTION = "production"  # Lightweight profiling, critical optimizations only
    HIGH_FREQUENCY = "high_frequency"  # Minimal overhead, maximum performance
    BENCHMARK = "benchmark"  # Full profiling for performance testing


@dataclass
class OperationProfile:
    """Detailed profile of a trading operation."""

    operation: TradingOperation
    execution_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    memory_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    cpu_instructions: deque = field(default_factory=lambda: deque(maxlen=100))
    hot_paths: dict[str, int] = field(default_factory=dict)
    bottlenecks: list[dict[str, Any]] = field(default_factory=list)
    optimization_suggestions: list[str] = field(default_factory=list)
    last_profiled: datetime | None = None
    profile_count: int = 0


@dataclass
class TradingBenchmark:
    """Benchmark results for trading operations."""

    operation: TradingOperation
    target_latency_ms: float
    current_p50_ms: float
    current_p95_ms: float
    current_p99_ms: float
    performance_score: float  # 0-100, higher is better
    meets_target: bool
    improvement_potential_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TradingProfiler:
    """
    High-precision profiler for individual trading operations.

    Uses cProfile and custom timing to identify performance bottlenecks
    in critical trading paths.
    """

    def __init__(self, operation: TradingOperation):
        self.operation = operation
        self.profile = OperationProfile(operation=operation)
        self._profiler: cProfile.Optional[Profile] = None
        self._memory_tracer_active = False
        self._start_time = 0.0
        self._start_memory = 0

    def start_profiling(self, enable_memory_tracing: bool = False) -> None:
        """Start profiling the operation."""
        self._start_time = time.perf_counter()

        # Start CPU profiling
        self._profiler = cProfile.Profile()
        self._profiler.enable()

        # Start memory tracing if requested
        if enable_memory_tracing and not self._memory_tracer_active:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            self._start_memory = tracemalloc.get_traced_memory()[0]
            self._memory_tracer_active = True

    def stop_profiling(self) -> dict[str, Any]:
        """Stop profiling and return results."""
        end_time = time.perf_counter()
        execution_time_ms = (end_time - self._start_time) * 1000

        results = {
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.now(timezone.utc),
            "memory_usage_bytes": 0,
            "hot_functions": [],
            "bottlenecks": [],
        }

        # Stop CPU profiling
        if self._profiler:
            self._profiler.disable()
            results.update(self._analyze_cpu_profile())

        # Stop memory tracing
        if self._memory_tracer_active:
            current_memory = tracemalloc.get_traced_memory()[0]
            results["memory_usage_bytes"] = current_memory - self._start_memory
            self._memory_tracer_active = False

        # Update profile data
        self.profile.execution_times.append(execution_time_ms)
        self.profile.memory_usage.append(results["memory_usage_bytes"])
        self.profile.last_profiled = datetime.now(timezone.utc)
        self.profile.profile_count += 1

        return results

    def _analyze_cpu_profile(self) -> dict[str, Any]:
        """Analyze CPU profiling results."""
        if not self._profiler:
            return {"hot_functions": [], "bottlenecks": []}

        # Capture profile statistics
        s = io.StringIO()
        ps = pstats.Stats(self._profiler, stream=s)
        ps.sort_stats("cumulative")
        ps.print_stats(20)  # Top 20 functions

        profile_output = s.getvalue()

        # Parse hot functions and bottlenecks
        hot_functions = self._parse_hot_functions(ps)
        bottlenecks = self._identify_bottlenecks(ps)

        return {
            "hot_functions": hot_functions,
            "bottlenecks": bottlenecks,
            "profile_output": profile_output,
        }

    def _parse_hot_functions(self, stats: pstats.Stats) -> list[dict[str, Any]]:
        """Parse hot functions from profile statistics."""
        hot_functions = []

        for func, (cc, _nc, tt, ct, _callers) in stats.stats.items():
            if ct > 0.001:  # Functions taking more than 1ms
                filename, line_number, function_name = func
                hot_functions.append(
                    {
                        "function": function_name,
                        "filename": filename,
                        "line_number": line_number,
                        "call_count": cc,
                        "total_time": tt,
                        "cumulative_time": ct,
                        "per_call_time": ct / cc if cc > 0 else 0,
                    }
                )

        # Sort by cumulative time
        hot_functions.sort(key=lambda x: x["cumulative_time"], reverse=True)
        return hot_functions[:10]  # Top 10

    def _identify_bottlenecks(self, stats: pstats.Stats) -> list[dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        for func, (cc, _nc, _tt, ct, _callers) in stats.stats.items():
            # Identify functions that are called frequently or take long time
            if cc > 100 or ct > 0.01:  # Called 100+ times or takes 10ms+
                filename, line_number, function_name = func

                # Calculate bottleneck score
                bottleneck_score = (ct * 1000) + (cc * 0.1)  # Weight time heavily

                bottlenecks.append(
                    {
                        "function": function_name,
                        "filename": filename,
                        "line_number": line_number,
                        "bottleneck_score": bottleneck_score,
                        "call_count": cc,
                        "cumulative_time": ct,
                        "suggestion": self._generate_optimization_suggestion(function_name, cc, ct),
                    }
                )

        # Sort by bottleneck score
        bottlenecks.sort(key=lambda x: x["bottleneck_score"], reverse=True)
        return bottlenecks[:5]  # Top 5 bottlenecks

    def _generate_optimization_suggestion(
        self, function_name: str, call_count: int, cumulative_time: float
    ) -> str:
        """Generate optimization suggestions based on profiling data."""
        suggestions = []

        if call_count > 1000:
            suggestions.append("Consider caching or memoization")

        if cumulative_time > 0.05:  # > 50ms
            suggestions.append("Optimize algorithm or use async/await")

        if "json" in function_name.lower():
            suggestions.append("Consider faster serialization (e.g., msgpack)")

        if "sql" in function_name.lower() or "query" in function_name.lower():
            suggestions.append("Add database indexes or optimize query")

        if "sort" in function_name.lower():
            suggestions.append("Consider using heapq or optimized sorting")

        return "; ".join(suggestions) if suggestions else "No specific suggestions"

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary for this operation."""
        if not self.profile.execution_times:
            return {"error": "No profiling data available"}

        times = list(self.profile.execution_times)
        times.sort()

        return {
            "operation": self.operation.value,
            "sample_count": len(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "avg_time_ms": sum(times) / len(times),
            "p50_ms": times[len(times) // 2],
            "p95_ms": times[int(len(times) * 0.95)],
            "p99_ms": times[int(len(times) * 0.99)],
            "last_profiled": (
                self.profile.last_profiled.isoformat() if self.profile.last_profiled else None
            ),
            "profile_count": self.profile.profile_count,
        }


class TradingOperationOptimizer(BaseComponent):
    """
    Comprehensive optimizer for critical trading operations.

    Provides real-time optimization, profiling, and performance tuning
    for trading-critical code paths.
    """

    def __init__(self, config: "Config", performance_monitor: "PerformanceMonitor"):
        """Initialize trading operation optimizer."""
        super().__init__()
        self.config = config
        self.performance_monitor = performance_monitor

        # Profilers for each trading operation
        self.profilers: dict[TradingOperation, TradingProfiler] = {}
        for operation in TradingOperation:
            self.profilers[operation] = TradingProfiler(operation)

        # Performance targets for each operation (in milliseconds)
        self.performance_targets = {
            TradingOperation.ORDER_PLACEMENT: 50.0,
            TradingOperation.ORDER_CANCELLATION: 30.0,
            TradingOperation.ORDER_MODIFICATION: 40.0,
            TradingOperation.MARKET_DATA_TICK: 10.0,
            TradingOperation.ORDERBOOK_UPDATE: 15.0,
            TradingOperation.POSITION_UPDATE: 15.0,
            TradingOperation.PORTFOLIO_CALCULATION: 30.0,
            TradingOperation.RISK_CHECK: 25.0,
            TradingOperation.PNL_CALCULATION: 20.0,
            TradingOperation.STOP_LOSS_TRIGGER: 5.0,
            TradingOperation.ARBITRAGE_CALCULATION: 10.0,
            TradingOperation.ML_PREDICTION: 100.0,
        }

        # Optimization level
        self.optimization_level = OptimizationLevel.PRODUCTION

        # Benchmarking data
        self.benchmarks: dict[TradingOperation, TradingBenchmark] = {}

        # Hot path tracking
        self.hot_paths: dict[str, int] = defaultdict(int)

        # Optimization recommendations
        self.optimization_queue: deque = deque(maxlen=100)

        # Background optimization
        self._optimization_task: asyncio.Task | None = None
        self._optimization_interval = 300  # 5 minutes

        # Function call frequency tracking
        self.function_call_counts: dict[str, int] = defaultdict(int)
        self.function_timing: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    async def initialize(self) -> None:
        """Initialize the trading operation optimizer."""
        try:
            self.logger.info("Initializing trading operation optimizer...")

            # Set optimization level based on config
            if hasattr(self.config, "trading_optimization_level"):
                self.optimization_level = OptimizationLevel(self.config.trading_optimization_level)

            # Start background optimization
            await self._start_background_optimization()

            # Run initial benchmarks
            await self._run_initial_benchmarks()

            level_name = self.optimization_level.value
            self.logger.info(f"Trading operation optimizer initialized with {level_name} level")

        except Exception as e:
            self.logger.error(f"Failed to initialize trading optimizer: {e}")
            raise PerformanceError(f"Trading optimizer initialization failed: {e}") from e

    async def _start_background_optimization(self) -> None:
        """Start background optimization task."""
        if self._optimization_task is None:
            self._optimization_task = asyncio.create_task(self._optimization_loop())
            self.logger.info("Background optimization started")

    async def _optimization_loop(self) -> None:
        """Background loop for continuous optimization."""
        while True:
            try:
                await asyncio.sleep(self._optimization_interval)

                # Run performance analysis
                await self._analyze_performance_trends()

                # Generate optimization recommendations
                await self._generate_optimization_recommendations()

                # Update benchmarks
                await self._update_benchmarks()

                # Log optimization summary
                await self._log_optimization_summary()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background optimization error: {e}")

    async def profile_operation(
        self,
        operation: TradingOperation,
        func: Callable,
        *args,
        enable_memory_tracing: bool = False,
        **kwargs,
    ) -> tuple[Any, dict[str, Any]]:
        """
        Profile a trading operation and return results with profiling data.

        Args:
            operation: Type of trading operation
            func: Function to profile
            enable_memory_tracing: Whether to trace memory allocations
            *args, **kwargs: Arguments for the function

        Returns:
            Tuple of (function_result, profiling_data)
        """
        profiler = self.profilers[operation]

        try:
            # Start profiling based on optimization level
            if self.optimization_level in [
                OptimizationLevel.DEVELOPMENT,
                OptimizationLevel.BENCHMARK,
            ]:
                profiler.start_profiling(enable_memory_tracing=enable_memory_tracing)
                enable_profiling = True
            else:
                # Lightweight timing for production
                enable_profiling = False
                start_time = time.perf_counter()

            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Stop profiling and get results
            if enable_profiling:
                profiling_data = profiler.stop_profiling()
            else:
                execution_time = (time.perf_counter() - start_time) * 1000
                profiling_data = {
                    "execution_time_ms": execution_time,
                    "timestamp": datetime.now(timezone.utc),
                    "lightweight_mode": True,
                }
                profiler.profile.execution_times.append(execution_time)

            # Record with performance monitor
            from src.core.performance.performance_monitor import OperationType
            await self.performance_monitor.record_simple_latency(
                OperationType.TRADING_ORDER,  # Map to generic operation type
                profiling_data["execution_time_ms"],
                {"trading_operation": operation.value},
            )

            # Track hot paths
            if hasattr(func, "__name__"):
                self.hot_paths[func.__name__] += 1

            return result, profiling_data

        except Exception as e:
            self.logger.error(f"Error profiling {operation.value}: {e}")
            # Return error in profiling data but re-raise the original exception
            profiling_data = {
                "error": str(e),
                "execution_time_ms": 0,
                "timestamp": datetime.now(timezone.utc),
            }
            raise

    def optimize_function(self, operation: TradingOperation):
        """Decorator for automatic function optimization."""

        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                result, profiling_data = await self.profile_operation(
                    operation, func, *args, **kwargs
                )

                # Track function performance
                func_name = f"{func.__module__}.{func.__name__}"
                self.function_call_counts[func_name] += 1
                self.function_timing[func_name].append(profiling_data["execution_time_ms"])

                # Check if optimization is needed
                if profiling_data["execution_time_ms"] > self.performance_targets.get(
                    operation, 100.0
                ):
                    await self._queue_optimization_recommendation(
                        operation, func_name, profiling_data
                    )

                return result

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For synchronous functions, run in event loop if possible
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If we're in an async context, we can't use this wrapper
                        # Fall back to direct execution
                        start_time = time.perf_counter()
                        result = func(*args, **kwargs)
                        execution_time = (time.perf_counter() - start_time) * 1000

                        # Track performance
                        func_name = f"{func.__module__}.{func.__name__}"
                        self.function_call_counts[func_name] += 1
                        self.function_timing[func_name].append(execution_time)

                        return result
                    else:
                        # Run async version
                        return loop.run_until_complete(async_wrapper(*args, **kwargs))
                except RuntimeError:
                    # No event loop, run synchronously
                    start_time = time.perf_counter()
                    result = func(*args, **kwargs)
                    execution_time = (time.perf_counter() - start_time) * 1000

                    # Track performance
                    func_name = f"{func.__module__}.{func.__name__}"
                    self.function_call_counts[func_name] += 1
                    self.function_timing[func_name].append(execution_time)

                    return result

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    async def _queue_optimization_recommendation(
        self, operation: TradingOperation, func_name: str, profiling_data: dict[str, Any]
    ) -> None:
        """Queue an optimization recommendation."""
        recommendation = {
            "timestamp": datetime.now(timezone.utc),
            "operation": operation.value,
            "function": func_name,
            "current_latency_ms": profiling_data["execution_time_ms"],
            "target_latency_ms": self.performance_targets.get(operation, 100.0),
            "priority": (
                "high"
                if profiling_data["execution_time_ms"]
                > 2 * self.performance_targets.get(operation, 100.0)
                else "medium"
            ),
            "suggestions": self._generate_specific_optimizations(operation, profiling_data),
        }

        self.optimization_queue.append(recommendation)

        # Log high-priority recommendations immediately
        if recommendation["priority"] == "high":
            self.logger.warning(
                f"High-priority optimization needed for {operation.value}", extra=recommendation
            )

    def _generate_specific_optimizations(
        self, operation: TradingOperation, profiling_data: dict[str, Any]
    ) -> list[str]:
        """Generate specific optimization suggestions based on operation type and profiling data."""
        suggestions = []

        # General suggestions based on execution time
        execution_time = profiling_data["execution_time_ms"]

        if execution_time > 100:
            suggestions.append("Consider async/await for I/O operations")
            suggestions.append("Profile for CPU-bound bottlenecks")

        if execution_time > 50:
            suggestions.append("Implement caching for repeated calculations")
            suggestions.append("Use more efficient data structures")

        # Operation-specific suggestions
        if operation in [TradingOperation.ORDER_PLACEMENT, TradingOperation.ORDER_CANCELLATION]:
            suggestions.extend(
                [
                    "Optimize network requests with connection pooling",
                    "Pre-validate orders to reduce rejection latency",
                    "Use batch operations where possible",
                ]
            )

        elif operation == TradingOperation.MARKET_DATA_TICK:
            suggestions.extend(
                [
                    "Use memory pools for tick objects",
                    "Implement lock-free data structures",
                    "Optimize serialization/deserialization",
                ]
            )

        elif operation == TradingOperation.ORDERBOOK_UPDATE:
            suggestions.extend(
                [
                    "Use sorted containers for price levels",
                    "Implement incremental updates",
                    "Cache frequently accessed data",
                ]
            )

        elif operation in [TradingOperation.RISK_CHECK, TradingOperation.PNL_CALCULATION]:
            suggestions.extend(
                [
                    "Cache risk parameters",
                    "Use vectorized calculations",
                    "Pre-compute common risk metrics",
                ]
            )

        elif operation == TradingOperation.ML_PREDICTION:
            suggestions.extend(
                ["Use model caching", "Implement feature caching", "Consider model quantization"]
            )

        return suggestions

    async def _analyze_performance_trends(self) -> None:
        """Analyze performance trends across all operations."""
        performance_summary = {}

        for operation, profiler in self.profilers.items():
            summary = profiler.get_performance_summary()
            if "error" not in summary:
                performance_summary[operation.value] = summary

                # Check for performance degradation
                target = self.performance_targets.get(operation, 100.0)
                if summary.get("p95_ms", 0) > target * 1.5:  # 50% worse than target
                    self.logger.warning(
                        f"Performance degradation detected for {operation.value}",
                        extra={
                            "operation": operation.value,
                            "current_p95": summary.get("p95_ms"),
                            "target": target,
                            "degradation_factor": summary.get("p95_ms", 0) / target,
                        },
                    )

        return performance_summary

    async def _generate_optimization_recommendations(self) -> None:
        """Generate optimization recommendations based on accumulated data."""
        # Analyze function call patterns
        for func_name, call_count in self.function_call_counts.items():
            if call_count > 1000:  # High-frequency function
                if func_name in self.function_timing:
                    timings = list(self.function_timing[func_name])
                    if timings:
                        avg_time = sum(timings) / len(timings)
                        if avg_time > 10:  # Average > 10ms for high-frequency function
                            recommendation = {
                                "type": "high_frequency_optimization",
                                "function": func_name,
                                "call_count": call_count,
                                "avg_time_ms": avg_time,
                                "suggestion": (
                                    "High-frequency function with significant execution time - "
                                    "consider optimization"
                                ),
                                "priority": "high",
                            }
                            self.optimization_queue.append(recommendation)

        # Analyze hot paths
        for path, count in self.hot_paths.items():
            if count > 500:  # Hot path
                recommendation = {
                    "type": "hot_path_optimization",
                    "path": path,
                    "hit_count": count,
                    "suggestion": "Hot path detected - consider caching or optimization",
                    "priority": "medium",
                }
                self.optimization_queue.append(recommendation)

    async def _update_benchmarks(self) -> None:
        """Update benchmark data for all trading operations."""
        for operation, profiler in self.profilers.items():
            summary = profiler.get_performance_summary()
            if "error" not in summary and summary.get("sample_count", 0) > 0:
                target = self.performance_targets.get(operation, 100.0)
                current_p99 = summary.get("p99_ms", 0)

                # Calculate performance score (0-100)
                if current_p99 <= target:
                    performance_score = 100
                else:
                    # Degrade score based on how much we exceed target
                    score = max(0, 100 - ((current_p99 - target) / target) * 50)
                    performance_score = min(100, max(0, score))

                benchmark = TradingBenchmark(
                    operation=operation,
                    target_latency_ms=target,
                    current_p50_ms=summary.get("p50_ms", 0),
                    current_p95_ms=summary.get("p95_ms", 0),
                    current_p99_ms=current_p99,
                    performance_score=performance_score,
                    meets_target=current_p99 <= target,
                    improvement_potential_ms=max(0, current_p99 - target),
                )

                self.benchmarks[operation] = benchmark

    async def _log_optimization_summary(self) -> None:
        """Log optimization summary."""
        total_operations = len(self.profilers)
        operations_meeting_target = sum(1 for b in self.benchmarks.values() if b.meets_target)
        pending_optimizations = len(self.optimization_queue)

        avg_performance_score = (
            sum(b.performance_score for b in self.benchmarks.values()) / len(self.benchmarks)
            if self.benchmarks
            else 0
        )

        self.logger.info(
            "Trading optimization summary",
            extra={
                "total_operations": total_operations,
                "operations_meeting_target": operations_meeting_target,
                "target_compliance_rate": (
                    operations_meeting_target / total_operations if total_operations > 0 else 0
                ),
                "avg_performance_score": avg_performance_score,
                "pending_optimizations": pending_optimizations,
                "optimization_level": self.optimization_level.value,
            },
        )

    async def _run_initial_benchmarks(self) -> None:
        """Run initial benchmarks for all operations."""
        # This would run synthetic benchmarks for each operation type
        # For now, just log that we're ready for benchmarking
        self.logger.info("Trading operation benchmarks initialized")

    async def get_optimization_report(self) -> dict[str, Any]:
        """Get comprehensive optimization report."""
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "optimization_level": self.optimization_level.value,
            "performance_targets": {
                op.value: target for op, target in self.performance_targets.items()
            },
            "benchmarks": {},
            "optimization_recommendations": list(self.optimization_queue),
            "hot_paths": dict(self.hot_paths),
            "function_performance": {},
            "summary": {},
        }

        # Add benchmark data
        for operation, benchmark in self.benchmarks.items():
            report["benchmarks"][operation.value] = benchmark.__dict__

        # Add function performance data
        for func_name, timings in self.function_timing.items():
            if timings:
                timings_list = list(timings)
                report["function_performance"][func_name] = {
                    "call_count": self.function_call_counts[func_name],
                    "avg_time_ms": sum(timings_list) / len(timings_list),
                    "max_time_ms": max(timings_list),
                    "min_time_ms": min(timings_list),
                }

        # Add summary statistics
        if self.benchmarks:
            report["summary"] = {
                "operations_meeting_targets": sum(
                    1 for b in self.benchmarks.values() if b.meets_target
                ),
                "total_operations": len(self.benchmarks),
                "avg_performance_score": sum(b.performance_score for b in self.benchmarks.values())
                / len(self.benchmarks),
                "worst_performing_operation": min(
                    self.benchmarks.items(), key=lambda x: x[1].performance_score
                )[0].value,
                "best_performing_operation": max(
                    self.benchmarks.items(), key=lambda x: x[1].performance_score
                )[0].value,
            }

        return report

    async def force_optimization_analysis(self) -> dict[str, Any]:
        """Force immediate optimization analysis."""
        self.logger.info("Starting forced optimization analysis")

        analysis_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis_type": "forced",
            "performance_trends": await self._analyze_performance_trends(),
            "recommendations_generated": 0,
            "benchmarks_updated": 0,
        }

        # Generate recommendations
        await self._generate_optimization_recommendations()
        analysis_results["recommendations_generated"] = len(self.optimization_queue)

        # Update benchmarks
        await self._update_benchmarks()
        analysis_results["benchmarks_updated"] = len(self.benchmarks)

        self.logger.info("Forced optimization analysis completed", extra=analysis_results)

        return analysis_results

    async def cleanup(self) -> None:
        """Cleanup optimizer resources with guaranteed cleanup."""
        cleanup_errors = []

        # Cancel background optimization task with timeout
        if self._optimization_task:
            try:
                self._optimization_task.cancel()
                # Give the task a chance to cleanup gracefully with timeout
                try:
                    await asyncio.wait_for(self._optimization_task, timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.warning(
                        "Optimization task did not complete within timeout, forcing termination"
                    )
            except asyncio.CancelledError:
                pass  # Expected when task is cancelled
            except Exception as e:
                cleanup_errors.append(f"optimization_task: {e}")
                self.logger.error(f"Error cancelling optimization task: {e}")
            finally:
                self._optimization_task = None

        # Clear profiling data with error handling
        try:
            for operation, profiler in self.profilers.items():
                try:
                    profiler.profile.execution_times.clear()
                    profiler.profile.memory_usage.clear()
                    profiler.profile.hot_paths.clear()
                    profiler.profile.bottlenecks.clear()
                    profiler.profile.optimization_suggestions.clear()
                except Exception as e:
                    cleanup_errors.append(f"profiler_{operation.value}: {e}")
                    self.logger.error(f"Error clearing profiler for {operation.value}: {e}")
        except Exception as e:
            cleanup_errors.append(f"profilers: {e}")
            self.logger.error(f"Error clearing profilers: {e}")

        # Clear optimization data with error handling
        try:
            self.optimization_queue.clear()
        except Exception as e:
            cleanup_errors.append(f"optimization_queue: {e}")
            self.logger.error(f"Error clearing optimization queue: {e}")

        try:
            self.hot_paths.clear()
        except Exception as e:
            cleanup_errors.append(f"hot_paths: {e}")
            self.logger.error(f"Error clearing hot paths: {e}")

        try:
            self.function_call_counts.clear()
        except Exception as e:
            cleanup_errors.append(f"function_call_counts: {e}")
            self.logger.error(f"Error clearing function call counts: {e}")

        try:
            self.function_timing.clear()
        except Exception as e:
            cleanup_errors.append(f"function_timing: {e}")
            self.logger.error(f"Error clearing function timing: {e}")

        # Clear benchmarks
        try:
            self.benchmarks.clear()
        except Exception as e:
            cleanup_errors.append(f"benchmarks: {e}")
            self.logger.error(f"Error clearing benchmarks: {e}")

        # Stop any active tracemalloc if running
        try:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
        except Exception as e:
            cleanup_errors.append(f"tracemalloc: {e}")
            self.logger.error(f"Error stopping tracemalloc: {e}")

        # Log final cleanup status
        if cleanup_errors:
            self.logger.warning(
                f"Trading operation optimizer cleanup completed with {len(cleanup_errors)} errors: "
                f"{cleanup_errors}"
            )
        else:
            self.logger.info("Trading operation optimizer cleaned up successfully")


# Context manager for trading operation profiling
class TradingOperationContext:
    """Context manager for automatic trading operation profiling."""

    def __init__(self, optimizer: TradingOperationOptimizer, operation: TradingOperation):
        self.optimizer = optimizer
        self.operation = operation
        self.profiler = optimizer.profilers[operation]
        self.start_time = None

    async def __aenter__(self):
        self.start_time = time.perf_counter()
        if self.optimizer.optimization_level in [
            OptimizationLevel.DEVELOPMENT,
            OptimizationLevel.BENCHMARK,
        ]:
            self.profiler.start_profiling(enable_memory_tracing=True)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            execution_time = (time.perf_counter() - self.start_time) * 1000

            if self.optimizer.optimization_level in [
                OptimizationLevel.DEVELOPMENT,
                OptimizationLevel.BENCHMARK,
            ]:
                self.profiler.stop_profiling()
            else:
                self.profiler.profile.execution_times.append(execution_time)

            # Record with performance monitor
            from src.core.performance.performance_monitor import OperationType
            await self.optimizer.performance_monitor.record_simple_latency(
                OperationType.TRADING_ORDER,
                execution_time,
                {"trading_operation": self.operation.value, "success": exc_type is None},
            )
