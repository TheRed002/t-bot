"""Core caching module with Redis infrastructure for the T-Bot trading system."""

from .cache_decorators import (
    cache_bot_status,
    cache_invalidate,
    cache_market_data,
    cache_risk_metrics,
    cache_strategy_signals,
    cache_warm,
    cached,
)
from .cache_keys import CacheKeys
from .cache_manager import CacheManager, get_cache_manager
from .cache_metrics import CacheMetrics, get_cache_metrics
from .cache_monitoring import CacheHealthReport, CacheMonitor, get_cache_monitor
from .cache_warming import CacheWarmer, WarmingStrategy, WarmingTask, get_cache_warmer

__all__ = [
    "CacheHealthReport",
    "CacheKeys",
    "CacheManager",
    "CacheMetrics",
    "CacheMonitor",
    "CacheWarmer",
    "WarmingStrategy",
    "WarmingTask",
    "cache_bot_status",
    "cache_invalidate",
    "cache_market_data",
    "cache_risk_metrics",
    "cache_strategy_signals",
    "cache_warm",
    "cached",
    "get_cache_manager",
    "get_cache_metrics",
    "get_cache_monitor",
    "get_cache_warmer",
]
