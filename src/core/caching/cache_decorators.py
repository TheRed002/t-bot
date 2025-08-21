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
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()

            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                func_name = f"{func.__module__}.{func.__name__}"
                cache_key = f"{func_name}:{_create_cache_key(*args, **kwargs)}"

            # Try to get from cache first
            try:
                cached_result = await cache_manager.get(cache_key, namespace)
                if cached_result is not None:
                    return cached_result
            except Exception as e:
                cache_manager.logger.warning(f"Cache get failed for {cache_key}: {e}")

            # Function to execute the actual function
            async def execute_function():
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            # Execute with or without lock
            try:
                if use_lock:
                    lock_resource = f"func_exec:{cache_key}"
                    result = await cache_manager.with_lock(
                        lock_resource, execute_function, timeout=lock_timeout
                    )
                else:
                    result = await execute_function()

                # Cache the result
                try:
                    await cache_manager.set(
                        cache_key, result, namespace=namespace, ttl=ttl, data_type=data_type
                    )
                except Exception as e:
                    cache_manager.logger.warning(f"Cache set failed for {cache_key}: {e}")

                return result

            except Exception:
                if invalidate_on_error:
                    try:
                        await cache_manager.delete(cache_key, namespace)
                    except Exception:
                        pass  # Ignore invalidation errors
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to run the async operations in event loop
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(async_wrapper(*args, **kwargs))

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


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
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Execute the function first
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Invalidate cache after successful execution
            cache_manager = get_cache_manager()

            try:
                # Invalidate specific keys
                if keys:
                    key_list = [keys] if isinstance(keys, str) else keys
                    for key in key_list:
                        await cache_manager.delete(key, namespace)

                # Invalidate by patterns
                if patterns:
                    pattern_list = [patterns] if isinstance(patterns, str) else patterns
                    for pattern in pattern_list:
                        await cache_manager.invalidate_pattern(pattern, namespace)

            except Exception as e:
                cache_manager.logger.warning(f"Cache invalidation failed: {e}")

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


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
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()

            # Execute function to get warming data
            if asyncio.iscoroutinefunction(func):
                warming_data = await func(*args, **kwargs)
            else:
                warming_data = func(*args, **kwargs)

            # Warm cache with the data
            if isinstance(warming_data, dict):
                warming_functions = {}

                key_list = [warming_keys] if isinstance(warming_keys, str) else warming_keys

                for key in key_list:
                    if key in warming_data:
                        warming_functions[key] = lambda data=warming_data[key]: data

                if warming_functions:
                    await cache_manager.warm_cache(
                        warming_functions, batch_size=batch_size, delay_between_batches=delay
                    )

            return warming_data

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


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
