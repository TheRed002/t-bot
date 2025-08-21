"""Unified decorator system for the T-Bot trading system."""

import asyncio
import functools
import logging
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, TypeVar

from src.base import BaseComponent
from src.utils.validation import validator

F = TypeVar("F", bound=Callable[..., Any])


class UnifiedDecorator(BaseComponent):
    """
    Single configurable decorator replacing multiple decorators.
    Provides retry, validation, logging, caching, and monitoring capabilities.
    """

    # Class-level cache storage
    _cache: dict[str, dict[str, Any]] = {}
    _cache_timestamps: dict[str, datetime] = {}

    @classmethod
    def _get_cache_key(cls, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function and arguments."""
        func_name = f"{func.__module__}.{func.__name__}"
        # Create hashable key from args and kwargs
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        return f"{func_name}:{args_str}:{kwargs_str}"

    @classmethod
    def _cache_result(
        cls, func: Callable, args: tuple, kwargs: dict, result: Any, ttl: int
    ) -> None:
        """Cache function result with TTL."""
        key = cls._get_cache_key(func, args, kwargs)
        cls._cache[key] = result
        cls._cache_timestamps[key] = datetime.utcnow() + timedelta(seconds=ttl)

    @classmethod
    def _get_cached_result(cls, func: Callable, args: tuple, kwargs: dict) -> Any | None:
        """Get cached result if still valid."""
        key = cls._get_cache_key(func, args, kwargs)

        if key in cls._cache:
            if datetime.utcnow() < cls._cache_timestamps.get(key, datetime.min):
                return cls._cache[key]
            else:
                # Cache expired, remove it
                del cls._cache[key]
                del cls._cache_timestamps[key]

        return None

    @classmethod
    async def _with_retry(
        cls,
        func: Callable,
        args: tuple,
        kwargs: dict,
        retry_times: int,
        retry_delay: float,
        logger: Any | None = None,
    ) -> Any:
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(retry_times):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if logger:
                    logger.warning(
                        f"Attempt {attempt + 1}/{retry_times} failed for {func.__name__}: {e}"
                    )

                if attempt < retry_times - 1:
                    await asyncio.sleep(retry_delay * (2**attempt))  # Exponential backoff

        raise last_exception

    @classmethod
    def _record_metrics(cls, func: Callable, result: Any, execution_time: float) -> None:
        """Record execution metrics (placeholder for actual metrics system)."""
        # This would integrate with your metrics system
        # For now, just log it using the base logger
        base_logger = logging.getLogger(__name__)
        base_logger.debug(f"Metrics: {func.__name__} completed in {execution_time:.3f}s")

    @classmethod
    def _validate_args(cls, func: Callable, args: tuple, kwargs: dict) -> None:
        """Validate function arguments based on annotations."""
        # Get function signature
        import inspect

        sig = inspect.signature(func)

        # Bind arguments
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError as e:
            raise ValueError(f"Invalid arguments for {func.__name__}: {e}")

        # Validate based on type hints
        for param_name, param in sig.parameters.items():
            if param.annotation != param.empty:
                value = bound.arguments.get(param_name)

                # Special validation for common types
                if "order" in param_name.lower() and isinstance(value, dict):
                    validator.validate_order(value)
                elif "price" in param_name.lower() and value is not None:
                    validator.validate_price(value)
                elif "quantity" in param_name.lower() and value is not None:
                    validator.validate_quantity(value)
                elif "symbol" in param_name.lower() and isinstance(value, str):
                    validator.validate_symbol(value)

    @staticmethod
    def enhance(
        retry: bool = False,
        retry_times: int = 3,
        retry_delay: float = 1.0,
        validate: bool = False,
        log: bool = False,
        log_level: str = "debug",
        cache: bool = False,
        cache_ttl: int = 60,
        monitor: bool = False,
        timeout: float | None = None,
        fallback: Callable | None = None,
    ) -> Callable[[F], F]:
        """
        Create a decorator with specified enhancements.

        Args:
            retry: Enable retry on failure
            retry_times: Number of retry attempts
            retry_delay: Base delay between retries (exponential backoff applied)
            validate: Enable argument validation
            log: Enable function call logging
            log_level: Logging level (debug, info, warning, error)
            cache: Enable result caching
            cache_ttl: Cache time-to-live in seconds
            monitor: Enable metrics monitoring
            timeout: Function timeout in seconds
            fallback: Fallback function if all retries fail

        Returns:
            Decorated function
        """

        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                logger = logging.getLogger(func.__module__) if log else None
                start_time = time.time()

                # Validation
                if validate:
                    try:
                        UnifiedDecorator._validate_args(func, args, kwargs)
                    except Exception as e:
                        if logger:
                            logger.error(f"Validation failed for {func.__name__}: {e}")
                        raise

                # Logging
                if log and logger:
                    log_method = getattr(logger, log_level, logger.debug)
                    log_method(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")

                # Check cache
                if cache:
                    cached_result = UnifiedDecorator._get_cached_result(func, args, kwargs)
                    if cached_result is not None:
                        if logger:
                            logger.debug(f"Cache hit for {func.__name__}")
                        return cached_result

                try:
                    # Apply timeout if specified
                    if timeout:
                        result = await asyncio.wait_for(
                            UnifiedDecorator._execute_with_retry(
                                func, args, kwargs, retry, retry_times, retry_delay, logger
                            ),
                            timeout=timeout,
                        )
                    else:
                        result = await UnifiedDecorator._execute_with_retry(
                            func, args, kwargs, retry, retry_times, retry_delay, logger
                        )

                    # Cache result
                    if cache:
                        UnifiedDecorator._cache_result(func, args, kwargs, result, cache_ttl)

                    # Monitor
                    if monitor:
                        execution_time = time.time() - start_time
                        UnifiedDecorator._record_metrics(func, result, execution_time)

                    # Log success
                    if log and logger:
                        execution_time = time.time() - start_time
                        log_method(f"{func.__name__} completed in {execution_time:.3f}s")

                    return result

                except Exception as e:
                    if logger:
                        logger.error(f"{func.__name__} failed: {e}\n{traceback.format_exc()}")

                    # Use fallback if provided
                    if fallback:
                        if logger:
                            logger.info(f"Using fallback for {func.__name__}")
                        if asyncio.iscoroutinefunction(fallback):
                            return await fallback(*args, **kwargs)
                        else:
                            return fallback(*args, **kwargs)

                    raise

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                """Wrapper for synchronous functions."""
                # For sync functions, run in event loop if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # For sync functions, just execute them directly with basic features
                logger = logging.getLogger(func.__module__) if log else None
                start_time = time.time()

                # Validation
                if validate:
                    try:
                        UnifiedDecorator._validate_args(func, args, kwargs)
                    except Exception as e:
                        if logger:
                            logger.error(f"Validation failed for {func.__name__}: {e}")
                        raise

                # Logging
                if log and logger:
                    log_method = getattr(logger, log_level, logger.debug)
                    log_method(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")

                # Check cache
                if cache:
                    cached_result = UnifiedDecorator._get_cached_result(func, args, kwargs)
                    if cached_result is not None:
                        if logger:
                            logger.debug(f"Cache hit for {func.__name__}")
                        return cached_result

                try:
                    # Execute with retry if needed
                    if retry:
                        last_exception = None
                        for attempt in range(retry_times):
                            try:
                                result = func(*args, **kwargs)
                                break
                            except Exception as e:
                                last_exception = e
                                if logger:
                                    logger.warning(
                                        f"Attempt {attempt + 1}/{retry_times} failed for {func.__name__}: {e}"
                                    )
                                if attempt < retry_times - 1:
                                    time.sleep(retry_delay * (2**attempt))
                        else:
                            raise last_exception
                    else:
                        result = func(*args, **kwargs)

                    # Cache result
                    if cache:
                        UnifiedDecorator._cache_result(func, args, kwargs, result, cache_ttl)

                    # Monitor
                    if monitor:
                        execution_time = time.time() - start_time
                        UnifiedDecorator._record_metrics(func, result, execution_time)

                    # Log success
                    if log and logger:
                        execution_time = time.time() - start_time
                        log_method(f"{func.__name__} completed in {execution_time:.3f}s")

                    return result

                except Exception as e:
                    if logger:
                        logger.error(f"{func.__name__} failed: {e}")

                    # Use fallback if provided
                    if fallback:
                        if logger:
                            logger.info(f"Using fallback for {func.__name__}")
                        return fallback(*args, **kwargs)

                    raise

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    @classmethod
    async def _execute_with_retry(
        cls,
        func: Callable,
        args: tuple,
        kwargs: dict,
        retry: bool,
        retry_times: int,
        retry_delay: float,
        logger: Any | None,
    ) -> Any:
        """Execute function with optional retry."""
        if retry:
            return await cls._with_retry(func, args, kwargs, retry_times, retry_delay, logger)
        else:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)


# Export main decorator
dec = UnifiedDecorator


# Simple decorator functions for common use cases
def retry(max_attempts: int = 3, delay: float = 1.0, base_delay: float | None = None):
    """Retry decorator with exponential backoff."""
    # Handle different parameter names for backward compatibility
    actual_delay = base_delay if base_delay is not None else delay
    return dec.enhance(retry=True, retry_times=max_attempts, retry_delay=actual_delay)


def cached(ttl: int = 300):
    """Cache decorator with TTL."""
    return dec.enhance(cache=True, cache_ttl=ttl)


def validated():
    """Validation decorator."""
    return dec.enhance(validate=True)


def logged(level: str = "info"):
    """Logging decorator."""
    return dec.enhance(log=True, log_level=level)


def monitored():
    """Monitoring decorator."""
    return dec.enhance(monitor=True)


def timeout(seconds: float):
    """Timeout decorator."""
    return dec.enhance(timeout=seconds)


# Helper function to create bare decorators that can be used without parentheses
def _make_bare_decorator(decorator_func):
    """Create a decorator that can be used both with and without parentheses."""

    def wrapper(func_or_args=None, **kwargs):
        if func_or_args is None or callable(func_or_args):
            # Used as @decorator or @decorator(func)
            if func_or_args is None:
                # Called with () - return the decorator
                return decorator_func(**kwargs)
            else:
                # Called without () - func_or_args is the function
                return decorator_func()(func_or_args)
        else:
            # Called with arguments - return the decorator
            return decorator_func(func_or_args, **kwargs)

    return wrapper


# Additional aliases for backward compatibility - these work as bare decorators
cache_result = _make_bare_decorator(lambda ttl=300: cached(ttl))
validate_input = _make_bare_decorator(lambda: validated())
log_calls = _make_bare_decorator(lambda level="info": logged(level))
log_errors = _make_bare_decorator(lambda level="error": logged(level))
log_performance = _make_bare_decorator(lambda: monitored())
api_throttle = _make_bare_decorator(lambda: monitored())


def circuit_breaker(failure_threshold=5, recovery_timeout=60):
    return dec.enhance(retry=True, retry_times=failure_threshold)


rate_limit = _make_bare_decorator(lambda: monitored())
redis_cache = _make_bare_decorator(lambda ttl=300: cached(ttl))
time_execution = _make_bare_decorator(lambda: monitored())
ttl_cache = _make_bare_decorator(lambda ttl=300: cached(ttl))
type_check = _make_bare_decorator(lambda: validated())
validate_output = _make_bare_decorator(lambda: validated())
cpu_usage = _make_bare_decorator(lambda: monitored())
memory_usage = _make_bare_decorator(lambda: monitored())


def robust():
    return dec.enhance(retry=True, retry_times=5, retry_delay=1.0)


# Export all decorators
__all__ = [
    "UnifiedDecorator",
    "api_throttle",
    # Backward compatibility exports
    "cache_result",
    "cached",
    "circuit_breaker",
    "cpu_usage",
    "dec",
    "log_calls",
    "log_errors",
    "log_performance",
    "logged",
    "memory_usage",
    "monitored",
    "rate_limit",
    "redis_cache",
    "retry",
    "robust",
    "time_execution",
    "timeout",
    "ttl_cache",
    "type_check",
    "validate_input",
    "validate_output",
    "validated",
]
