"""
T-Bot Performance Optimization Suite

This package provides comprehensive performance optimization for the T-Bot trading system,
designed to achieve <100ms latency for critical trading operations.

Modules:
- performance_monitor: Real-time performance monitoring and metrics
- memory_optimizer: Memory management and garbage collection optimization
- trading_profiler: Critical trading operations profiling and optimization
- unified_cache_layer: Multi-level caching system integration

Usage:
    from src.core.performance import PerformanceOptimizer

    # Initialize with config
    optimizer = PerformanceOptimizer(config)
    await optimizer.initialize()

    # Use performance decorators
    @optimizer.optimize_trading_operation(TradingOperation.ORDER_PLACEMENT)
    async def place_order(order_data):
        # Trading logic here
        pass
"""

from ..caching.unified_cache_layer import CacheLevel, CacheStrategy, DataCategory, UnifiedCacheLayer
from .memory_optimizer import (
    GCStrategy,
    MemoryCategory,
    MemoryOptimizer,
    TradingMemoryContext,
    memory_optimized_trading_operation,
)
from .performance_monitor import (
    AlertLevel,
    MetricType,
    OperationTracker,
    OperationType,
    PerformanceMonitor,
    track_performance,
)
from .trading_profiler import (
    OptimizationLevel,
    TradingOperation,
    TradingOperationContext,
    TradingOperationOptimizer,
)

__all__ = [
    "AlertLevel",
    "CacheLevel",
    "CacheStrategy",
    "DataCategory",
    "GCStrategy",
    "MemoryCategory",
    "MemoryOptimizer",
    "MetricType",
    "OperationTracker",
    "OperationType",
    "OptimizationLevel",
    "PerformanceMonitor",
    "PerformanceOptimizer",
    "TradingMemoryContext",
    "TradingOperation",
    "TradingOperationContext",
    "TradingOperationOptimizer",
    "UnifiedCacheLayer",
    "memory_optimized_trading_operation",
    "track_performance",
]
