"""Cache decorators for easy caching integration."""

import asyncio
import hashlib
import json
from collections.abc import Callable
from functools import wraps

from .cache_keys import CacheKeys
from .cache_manager import get_cache_manager


def _create_cache_key(*args, **kwargs) -> str:
    """Create cache key from function arguments."""
    # Convert args and kwargs to a consistent string representation
    key_data = {
        "args": [str(arg) for arg in args],
        "kwargs": {str(k): str(v) for k, v in sorted(kwargs.items())},
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def cached(
    ttl: int | None = None,
    namespace: str = "cache",
    data_type: str = "default",
    key_generator: Callable | None = None,
    invalidate_on_error: bool = False,
    use_lock: bool = False,
    lock_timeout: int = 30,
):
    """
    Decorator for caching function results.

    Args:
        ttl: Time to live in seconds
        namespace: Cache namespace
        data_type: Data type for TTL selection
        key_generator: Custom key generation function
        invalidate_on_error: Invalidate cache on function error
        use_lock: Use distributed locking for cache updates
        lock_timeout: Lock timeout in seconds
    """

    def decorator(func: Callable):
        cache_config = {
            "ttl": ttl,
            "namespace": namespace,
            "data_type": data_type,
            "key_generator": key_generator,
            "invalidate_on_error": invalidate_on_error,
            "use_lock": use_lock,
            "lock_timeout": lock_timeout,
        }

        if asyncio.iscoroutinefunction(func):
            return _create_async_cached_wrapper(func, cache_config)
        else:
            return _create_sync_cached_wrapper(func, cache_config)

    return decorator


def _create_async_cached_wrapper(func: Callable, config: dict) -> Callable:
    """Create async wrapper for cached function."""

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        cache_manager = get_cache_manager()
        cache_key = _generate_cache_key(func, config["key_generator"], *args, **kwargs)

        # Try cache first
        cached_result = await _try_get_from_cache(cache_manager, cache_key, config["namespace"])
        if cached_result is not None:
            return cached_result

        # Execute function and handle result
        return await _execute_and_cache_async(
            func, cache_manager, cache_key, config, *args, **kwargs
        )

    return async_wrapper


def _create_sync_cached_wrapper(func: Callable, config: dict) -> Callable:
    """Create sync wrapper for cached function."""

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        async_wrapper = _create_async_cached_wrapper(_make_func_async(func), config)
        return _run_in_event_loop(async_wrapper, *args, **kwargs)

    return sync_wrapper


def _generate_cache_key(func: Callable, key_generator: Callable | None, *args, **kwargs) -> str:
    """Generate cache key for function call."""
    if key_generator:
        return key_generator(*args, **kwargs)
    else:
        func_name = f"{func.__module__}.{func.__name__}"
        return f"{func_name}:{_create_cache_key(*args, **kwargs)}"


async def _try_get_from_cache(cache_manager, cache_key: str, namespace: str):
    """Try to get result from cache."""
    try:
        cached_result = await cache_manager.get(cache_key, namespace)
        return cached_result
    except Exception as e:
        cache_manager.logger.warning(f"Cache get failed for {cache_key}: {e}")
        return None


async def _execute_and_cache_async(
    func: Callable, cache_manager, cache_key: str, config: dict, *args, **kwargs
):
    """Execute function and cache result."""
    try:
        # Execute function
        if config["use_lock"]:
            result = await _execute_with_lock(
                func, cache_manager, cache_key, config["lock_timeout"], *args, **kwargs
            )
        else:
            result = await func(*args, **kwargs)

        # Cache result
        await _store_result_in_cache(
            cache_manager,
            cache_key,
            result,
            config["namespace"],
            config["ttl"],
            config["data_type"],
        )

        return result

    except Exception:
        if config["invalidate_on_error"]:
            await _safe_invalidate_cache(cache_manager, cache_key, config["namespace"])
        raise


async def _execute_with_lock(
    func: Callable, cache_manager, cache_key: str, lock_timeout: int, *args, **kwargs
):
    """Execute function with distributed lock."""
    lock_resource = f"func_exec:{cache_key}"

    async def execute_function():
        return await func(*args, **kwargs)

    return await cache_manager.with_lock(lock_resource, execute_function, timeout=lock_timeout)


async def _store_result_in_cache(
    cache_manager, cache_key: str, result, namespace: str, ttl: int | None, data_type: str
):
    """Store result in cache with error handling."""
    try:
        await cache_manager.set(
            cache_key, result, namespace=namespace, ttl=ttl, data_type=data_type
        )
    except Exception as e:
        cache_manager.logger.warning(f"Cache set failed for {cache_key}: {e}")


async def _safe_invalidate_cache(cache_manager, cache_key: str, namespace: str):
    """Safely invalidate cache entry."""
    try:
        await cache_manager.delete(cache_key, namespace)
    except Exception:
        pass  # Ignore invalidation errors


def _make_func_async(func: Callable) -> Callable:
    """Convert sync function to async for uniform handling."""

    async def async_func(*args, **kwargs):
        return func(*args, **kwargs)

    return async_func


def _run_in_event_loop(async_func: Callable, *args, **kwargs):
    """Run async function in event loop."""
    loop = None
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(async_func(*args, **kwargs))


def cache_invalidate(
    patterns: str | list[str] | None = None,
    namespace: str = "cache",
    keys: str | list[str] | None = None,
):
    """
    Decorator for cache invalidation after function execution.

    Args:
        patterns: Pattern(s) to invalidate
        namespace: Cache namespace
        keys: Specific key(s) to invalidate
    """

    def decorator(func: Callable):
        invalidation_config = {"patterns": patterns, "namespace": namespace, "keys": keys}

        if asyncio.iscoroutinefunction(func):
            return _create_async_invalidation_wrapper(func, invalidation_config)
        else:
            return _create_sync_invalidation_wrapper(func, invalidation_config)

    return decorator


def _create_async_invalidation_wrapper(func: Callable, config: dict) -> Callable:
    """Create async wrapper for cache invalidation."""

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Execute function first
        result = await func(*args, **kwargs)

        # Invalidate cache after successful execution
        await _perform_cache_invalidation(config)

        return result

    return async_wrapper


def _create_sync_invalidation_wrapper(func: Callable, config: dict) -> Callable:
    """Create sync wrapper for cache invalidation."""

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        async_wrapper = _create_async_invalidation_wrapper(_make_func_async(func), config)
        return _run_in_event_loop(async_wrapper, *args, **kwargs)

    return sync_wrapper


async def _perform_cache_invalidation(config: dict):
    """Perform cache invalidation based on configuration."""
    cache_manager = get_cache_manager()
    namespace = config["namespace"]

    try:
        # Invalidate specific keys
        if config["keys"]:
            await _invalidate_specific_keys(cache_manager, config["keys"], namespace)

        # Invalidate by patterns
        if config["patterns"]:
            await _invalidate_by_patterns(cache_manager, config["patterns"], namespace)

    except Exception as e:
        cache_manager.logger.warning(f"Cache invalidation failed: {e}")


async def _invalidate_specific_keys(cache_manager, keys, namespace: str):
    """Invalidate specific cache keys."""
    key_list = [keys] if isinstance(keys, str) else keys
    for key in key_list:
        await cache_manager.delete(key, namespace)


async def _invalidate_by_patterns(cache_manager, patterns, namespace: str):
    """Invalidate cache keys by patterns."""
    pattern_list = [patterns] if isinstance(patterns, str) else patterns
    for pattern in pattern_list:
        await cache_manager.invalidate_pattern(pattern, namespace)


def cache_warm(
    warming_keys: str | list[str], namespace: str = "warm", batch_size: int = 10, delay: float = 0.1
):
    """
    Decorator for cache warming on application startup.

    Args:
        warming_keys: Key(s) to warm
        namespace: Cache namespace for warming
        batch_size: Batch size for warming
        delay: Delay between batches
    """

    def decorator(func: Callable):
        warming_config = {
            "warming_keys": warming_keys,
            "namespace": namespace,
            "batch_size": batch_size,
            "delay": delay,
        }

        if asyncio.iscoroutinefunction(func):
            return _create_async_warming_wrapper(func, warming_config)
        else:
            return _create_sync_warming_wrapper(func, warming_config)

    return decorator


def _create_async_warming_wrapper(func: Callable, config: dict) -> Callable:
    """Create async wrapper for cache warming."""

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Execute function to get warming data
        warming_data = await func(*args, **kwargs)

        # Warm cache with the data
        await _perform_cache_warming(warming_data, config)

        return warming_data

    return async_wrapper


def _create_sync_warming_wrapper(func: Callable, config: dict) -> Callable:
    """Create sync wrapper for cache warming."""

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        async_wrapper = _create_async_warming_wrapper(_make_func_async(func), config)
        return _run_in_event_loop(async_wrapper, *args, **kwargs)

    return sync_wrapper


async def _perform_cache_warming(warming_data, config: dict):
    """Perform cache warming with the provided data."""
    if not isinstance(warming_data, dict):
        return

    cache_manager = get_cache_manager()
    warming_functions = _build_warming_functions(warming_data, config["warming_keys"])

    if warming_functions:
        await cache_manager.warm_cache(
            warming_functions,
            batch_size=config["batch_size"],
            delay_between_batches=config["delay"],
        )


def _build_warming_functions(warming_data: dict, warming_keys) -> dict:
    """Build warming functions from data and keys."""
    warming_functions = {}
    key_list = [warming_keys] if isinstance(warming_keys, str) else warming_keys

    for key in key_list:
        if key in warming_data:
            warming_functions[key] = lambda data=warming_data[key]: data

    return warming_functions


# Specialized decorators for trading system components
def cache_market_data(symbol_arg_name: str = "symbol", ttl: int = 5):
    """Specialized decorator for market data caching."""

    def key_gen(*args, **kwargs):
        symbol = kwargs.get(symbol_arg_name) or (args[0] if args else "unknown")
        exchange = kwargs.get("exchange", "all")
        return CacheKeys.market_price(symbol, exchange)

    return cached(ttl=ttl, namespace="market_data", data_type="market_data", key_generator=key_gen)


def cache_risk_metrics(bot_id_arg_name: str = "bot_id", ttl: int = 60):
    """Specialized decorator for risk metrics caching."""

    def key_gen(*args, **kwargs):
        bot_id = kwargs.get(bot_id_arg_name) or (args[0] if args else "unknown")
        timeframe = kwargs.get("timeframe", "1h")
        return CacheKeys.risk_metrics(bot_id, timeframe)

    return cached(ttl=ttl, namespace="risk", data_type="risk_metrics", key_generator=key_gen)


def cache_strategy_signals(strategy_id_arg_name: str = "strategy_id", ttl: int = 300):
    """Specialized decorator for strategy signals caching."""

    def key_gen(*args, **kwargs):
        strategy_id = kwargs.get(strategy_id_arg_name) or (args[0] if args else "unknown")
        symbol = kwargs.get("symbol", "all")
        return CacheKeys.strategy_signals(strategy_id, symbol)

    return cached(
        ttl=ttl,
        namespace="strategy",
        data_type="strategy",
        key_generator=key_gen,
        use_lock=True,  # Strategy calculations might be expensive
    )


def cache_bot_status(bot_id_arg_name: str = "bot_id", ttl: int = 30):
    """Specialized decorator for bot status caching."""

    def key_gen(*args, **kwargs):
        bot_id = kwargs.get(bot_id_arg_name) or (args[0] if args else "unknown")
        return CacheKeys.bot_status(bot_id)

    return cached(ttl=ttl, namespace="bot", data_type="default", key_generator=key_gen)
