"""
Common Decorators for Cross-Cutting Concerns

This module provides reusable decorators for performance monitoring, error
handling, caching, logging, validation, and rate limiting that are used across
all components of the trading bot system.

Key Decorators:
- Performance Monitoring: @time_execution, @memory_usage, @cpu_usage
- Error Handling: @retry, @circuit_breaker, @timeout
- Caching: @cache_result, @redis_cache, @ttl_cache
- Logging: @log_calls, @log_performance, @log_errors
- Validation: @validate_input, @validate_output, @type_check
- Rate Limiting: @rate_limit, @api_throttle

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
"""

import asyncio
import functools
import inspect
import threading
import time
from collections.abc import Callable
from typing import Any

import psutil

from src.core.exceptions import TimeoutError, ValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)

# Thread-safe cache for decorators
_cache_lock = threading.Lock()
_memory_cache: dict[str, Any] = {}


def time_execution(func: Callable) -> Callable:
    """
    Decorator to measure and log execution time for both sync and async functions.

    This is the primary performance monitoring decorator used throughout the system.
    It measures execution time and logs performance metrics for monitoring and
    optimization.

    Args:
        func: The function to decorate

    Returns:
        Decorated function with execution time logging

    Example:
        @time_execution
        async def api_call():
            # Function implementation
            pass
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            logger.info(
                "Function executed successfully",
                function=func.__name__,
                execution_time_ms=round(execution_time * 1000, 2),
                success=True,
                module=func.__module__,
            )
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(
                "Function execution failed",
                function=func.__name__,
                execution_time_ms=round(execution_time * 1000, 2),
                error=str(e),
                error_type=type(e).__name__,
                success=False,
                module=func.__module__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()) if kwargs else [],
            )
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            logger.info(
                "Function executed successfully",
                function=func.__name__,
                execution_time_ms=round(execution_time * 1000, 2),
                success=True,
                module=func.__module__,
            )
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(
                "Function execution failed",
                function=func.__name__,
                execution_time_ms=round(execution_time * 1000, 2),
                error=str(e),
                error_type=type(e).__name__,
                success=False,
                module=func.__module__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()) if kwargs else [],
            )
            raise

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def memory_usage(func: Callable) -> Callable:
    """
    Decorator to monitor memory usage during function execution.

    Args:
        func: The function to decorate

    Returns:
        Decorated function with memory usage logging
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        process = psutil.Process()
        start_memory = process.memory_info().rss
        try:
            result = await func(*args, **kwargs)
            end_memory = process.memory_info().rss
            memory_used = end_memory - start_memory
            logger.info(
                "Memory usage tracked",
                function=func.__name__,
                memory_used_mb=round(memory_used / 1024 / 1024, 2),
                total_memory_mb=round(end_memory / 1024 / 1024, 2),
            )
            return result
        except Exception as e:
            end_memory = process.memory_info().rss
            memory_used = end_memory - start_memory
            logger.error(
                "Memory usage tracked (error)",
                function=func.__name__,
                memory_used_mb=round(memory_used / 1024 / 1024, 2),
                error=str(e),
            )
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        process = psutil.Process()
        start_memory = process.memory_info().rss
        try:
            result = func(*args, **kwargs)
            end_memory = process.memory_info().rss
            memory_used = end_memory - start_memory
            logger.info(
                "Memory usage tracked",
                function=func.__name__,
                memory_used_mb=round(memory_used / 1024 / 1024, 2),
                total_memory_mb=round(end_memory / 1024 / 1024, 2),
            )
            return result
        except Exception as e:
            end_memory = process.memory_info().rss
            memory_used = end_memory - start_memory
            logger.error(
                "Memory usage tracked (error)",
                function=func.__name__,
                memory_used_mb=round(memory_used / 1024 / 1024, 2),
                error=str(e),
            )
            raise

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def cpu_usage(func: Callable) -> Callable:
    """
    Decorator to monitor CPU usage during function execution.

    Args:
        func: The function to decorate

    Returns:
        Decorated function with CPU usage logging
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        process = psutil.Process()
        start_cpu = process.cpu_percent()
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            end_time = time.time()
            end_cpu = process.cpu_percent()
            duration = end_time - start_time
            avg_cpu = (start_cpu + end_cpu) / 2
            logger.info(
                "CPU usage tracked",
                function=func.__name__,
                duration_seconds=round(duration, 3),
                avg_cpu_percent=round(avg_cpu, 2),
            )
            return result
        except Exception as e:
            end_time = time.time()
            end_cpu = process.cpu_percent()
            duration = end_time - start_time
            avg_cpu = (start_cpu + end_cpu) / 2
            logger.error(
                "CPU usage tracked (error)",
                function=func.__name__,
                duration_seconds=round(duration, 3),
                avg_cpu_percent=round(avg_cpu, 2),
                error=str(e),
            )
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        process = psutil.Process()
        start_cpu = process.cpu_percent()
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            end_cpu = process.cpu_percent()
            duration = end_time - start_time
            avg_cpu = (start_cpu + end_cpu) / 2
            logger.info(
                "CPU usage tracked",
                function=func.__name__,
                duration_seconds=round(duration, 3),
                avg_cpu_percent=round(avg_cpu, 2),
            )
            return result
        except Exception as e:
            end_time = time.time()
            end_cpu = process.cpu_percent()
            duration = end_time - start_time
            avg_cpu = (start_cpu + end_cpu) / 2
            logger.error(
                "CPU usage tracked (error)",
                function=func.__name__,
                duration_seconds=round(duration, 3),
                avg_cpu_percent=round(avg_cpu, 2),
                error=str(e),
            )
            raise

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator to retry function execution with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for exponential backoff
        exceptions: Tuple of exceptions to retry on

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        logger.error(
                            "Function failed after all retry attempts",
                            function=func.__name__,
                            attempts=max_attempts,
                            final_error=str(e),
                        )
                        raise

                    delay = min(base_delay * (backoff_factor**attempt), max_delay)
                    logger.warning(
                        "Function failed, retrying",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        delay_seconds=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)

            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        logger.error(
                            "Function failed after all retry attempts",
                            function=func.__name__,
                            attempts=max_attempts,
                            final_error=str(e),
                        )
                        raise

                    delay = min(base_delay * (backoff_factor**attempt), max_delay)
                    logger.warning(
                        "Function failed, retrying",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        delay_seconds=delay,
                        error=str(e),
                    )
                    time.sleep(delay)

            raise last_exception

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def circuit_breaker(
    failure_threshold: int = 5, recovery_timeout: float = 30.0, expected_exception: type = Exception
) -> Callable:
    """
    Circuit breaker decorator to prevent cascading failures.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery
        expected_exception: Exception type that triggers circuit breaker

    Returns:
        Decorated function with circuit breaker logic
    """

    def decorator(func: Callable) -> Callable:
        # Circuit breaker state with thread safety
        failure_count = 0
        last_failure_time = 0
        circuit_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        lock = threading.Lock()

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            nonlocal failure_count, last_failure_time, circuit_state

            current_time = time.time()

            # Check if circuit is open and recovery timeout has passed
            with lock:
                if circuit_state == "OPEN" and current_time - last_failure_time > recovery_timeout:
                    circuit_state = "HALF_OPEN"
                    logger.info(
                        "Circuit breaker transitioning to HALF_OPEN", function=func.__name__
                    )

                # If circuit is open, reject the call
                if circuit_state == "OPEN":
                    raise TimeoutError(f"Circuit breaker is OPEN for {func.__name__}")

            try:
                result = await func(*args, **kwargs)

                # Success - close circuit if it was half-open
                with lock:
                    if circuit_state == "HALF_OPEN":
                        circuit_state = "CLOSED"
                        failure_count = 0
                        logger.info(
                            "Circuit breaker CLOSED after successful recovery",
                            function=func.__name__,
                        )

                return result

            except expected_exception:
                with lock:
                    failure_count += 1
                    last_failure_time = current_time

                    # Check if we should open the circuit
                    if failure_count >= failure_threshold:
                        circuit_state = "OPEN"
                        logger.error(
                            "Circuit breaker OPENED due to repeated failures",
                            function=func.__name__,
                            failure_count=failure_count,
                            threshold=failure_threshold,
                        )

                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            nonlocal failure_count, last_failure_time, circuit_state

            current_time = time.time()

            # Check if circuit is open and recovery timeout has passed
            with lock:
                if circuit_state == "OPEN" and current_time - last_failure_time > recovery_timeout:
                    circuit_state = "HALF_OPEN"
                    logger.info(
                        "Circuit breaker transitioning to HALF_OPEN", function=func.__name__
                    )

                # If circuit is open, reject the call
                if circuit_state == "OPEN":
                    raise TimeoutError(f"Circuit breaker is OPEN for {func.__name__}")

            try:
                result = func(*args, **kwargs)

                # Success - close circuit if it was half-open
                with lock:
                    if circuit_state == "HALF_OPEN":
                        circuit_state = "CLOSED"
                        failure_count = 0
                        logger.info(
                            "Circuit breaker CLOSED after successful recovery",
                            function=func.__name__,
                        )

                return result

            except expected_exception:
                with lock:
                    failure_count += 1
                    last_failure_time = current_time

                    # Check if we should open the circuit
                    if failure_count >= failure_threshold:
                        circuit_state = "OPEN"
                        logger.error(
                            "Circuit breaker OPENED due to repeated failures",
                            function=func.__name__,
                            failure_count=failure_count,
                            threshold=failure_threshold,
                        )

                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def timeout(seconds: float) -> Callable:
    """
    Decorator to add timeout to function execution.

    Args:
        seconds: Timeout duration in seconds

    Returns:
        Decorated function with timeout logic
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(
                    "Function execution timed out", function=func.__name__, timeout_seconds=seconds
                )
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            import concurrent.futures

            # Use ThreadPoolExecutor to implement timeout for sync functions
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timeout=seconds)
                    return result
                except concurrent.futures.TimeoutError:
                    logger.error(
                        "Function execution timed out",
                        function=func.__name__,
                        timeout_seconds=seconds,
                    )
                    raise TimeoutError(
                        f"Function {func.__name__} timed out after {seconds} seconds"
                    )
                except Exception as e:
                    logger.error("Function execution failed", function=func.__name__, error=str(e))
                    raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# Thread-safe in-memory cache
_cache: dict[str, Any] = {}
_cache_timestamps: dict[str, float] = {}


def cache_result(ttl_seconds: float = 300) -> Callable:
    """
    Decorator to cache function results in memory.

    Args:
        ttl_seconds: Time to live for cached results in seconds

    Returns:
        Decorated function with caching logic
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

            current_time = time.time()

            # Check if cached result exists and is still valid
            with _cache_lock:
                if (
                    cache_key in _cache
                    and current_time - _cache_timestamps[cache_key] < ttl_seconds
                ):
                    logger.debug(
                        "Returning cached result", function=func.__name__, cache_key=cache_key
                    )
                    return _cache[cache_key]

            # Execute function and cache result
            result = await func(*args, **kwargs)

            with _cache_lock:
                _cache[cache_key] = result
                _cache_timestamps[cache_key] = current_time

            logger.debug(
                "Cached function result",
                function=func.__name__,
                cache_key=cache_key,
                ttl_seconds=ttl_seconds,
            )

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

            current_time = time.time()

            # Check if cached result exists and is still valid
            with _cache_lock:
                if (
                    cache_key in _cache
                    and current_time - _cache_timestamps[cache_key] < ttl_seconds
                ):
                    logger.debug(
                        "Returning cached result", function=func.__name__, cache_key=cache_key
                    )
                    return _cache[cache_key]

            # Execute function and cache result
            result = func(*args, **kwargs)

            with _cache_lock:
                _cache[cache_key] = result
                _cache_timestamps[cache_key] = current_time

            logger.debug(
                "Cached function result",
                function=func.__name__,
                cache_key=cache_key,
                ttl_seconds=ttl_seconds,
            )

            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def redis_cache(ttl_seconds: float = 300) -> Callable:
    """
    Decorator to cache function results in Redis.

    Args:
        ttl_seconds: Time to live for cached results in seconds

    Returns:
        Decorated function with Redis caching logic
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # TODO: Implement Redis caching
            # This is a placeholder that falls back to memory cache
            logger.warning(
                "Redis cache not implemented, falling back to memory cache", function=func.__name__
            )
            # Use the same cache mechanism as cache_result
            cache_key = f"redis:{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

            current_time = time.time()

            # Check if cached result exists and is still valid
            if cache_key in _cache and current_time - _cache_timestamps[cache_key] < ttl_seconds:
                logger.debug(
                    "Returning cached result (Redis fallback)",
                    function=func.__name__,
                    cache_key=cache_key,
                )
                return _cache[cache_key]

            # Execute function and cache result
            result = await func(*args, **kwargs)
            _cache[cache_key] = result
            _cache_timestamps[cache_key] = current_time

            logger.debug(
                "Cached function result (Redis fallback)",
                function=func.__name__,
                cache_key=cache_key,
                ttl_seconds=ttl_seconds,
            )

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # TODO: Implement Redis caching
            # This is a placeholder that falls back to memory cache
            logger.warning(
                "Redis cache not implemented, falling back to memory cache", function=func.__name__
            )
            # Use the same cache mechanism as cache_result
            cache_key = f"redis:{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

            current_time = time.time()

            # Check if cached result exists and is still valid
            if cache_key in _cache and current_time - _cache_timestamps[cache_key] < ttl_seconds:
                logger.debug(
                    "Returning cached result (Redis fallback)",
                    function=func.__name__,
                    cache_key=cache_key,
                )
                return _cache[cache_key]

            # Execute function and cache result
            result = func(*args, **kwargs)
            _cache[cache_key] = result
            _cache_timestamps[cache_key] = current_time

            logger.debug(
                "Cached function result (Redis fallback)",
                function=func.__name__,
                cache_key=cache_key,
                ttl_seconds=ttl_seconds,
            )

            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def ttl_cache(ttl_seconds: float = 300) -> Callable:
    """
    Alias for cache_result with TTL.

    Args:
        ttl_seconds: Time to live for cached results in seconds

    Returns:
        Decorated function with TTL caching logic
    """
    return cache_result(ttl_seconds)


def log_calls(func: Callable) -> Callable:
    """
    Decorator to log all function calls with parameters.

    Args:
        func: The function to decorate

    Returns:
        Decorated function with call logging
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        logger.info(
            "Function call started",
            function=func.__name__,
            args_count=len(args),
            kwargs_count=len(kwargs),
            module=func.__module__,
        )
        try:
            result = await func(*args, **kwargs)
            logger.info("Function call completed", function=func.__name__, success=True)
            return result
        except Exception as e:
            logger.error(
                "Function call failed", function=func.__name__, error=str(e), success=False
            )
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        logger.info(
            "Function call started",
            function=func.__name__,
            args_count=len(args),
            kwargs_count=len(kwargs),
            module=func.__module__,
        )
        try:
            result = func(*args, **kwargs)
            logger.info("Function call completed", function=func.__name__, success=True)
            return result
        except Exception as e:
            logger.error(
                "Function call failed", function=func.__name__, error=str(e), success=False
            )
            raise

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def log_performance(func: Callable) -> Callable:
    """
    Decorator to log performance metrics for function execution.

    Args:
        func: The function to decorate

    Returns:
        Decorated function with performance logging
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss

        try:
            result = await func(*args, **kwargs)

            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss

            execution_time = end_time - start_time
            memory_used = end_memory - start_memory

            logger.info(
                "Performance metrics",
                function=func.__name__,
                execution_time_ms=round(execution_time * 1000, 2),
                memory_used_mb=round(memory_used / 1024 / 1024, 2),
                success=True,
            )

            return result
        except Exception as e:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss

            execution_time = end_time - start_time
            memory_used = end_memory - start_memory

            logger.error(
                "Performance metrics (error)",
                function=func.__name__,
                execution_time_ms=round(execution_time * 1000, 2),
                memory_used_mb=round(memory_used / 1024 / 1024, 2),
                error=str(e),
                success=False,
            )
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss

        try:
            result = func(*args, **kwargs)

            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss

            execution_time = end_time - start_time
            memory_used = end_memory - start_memory

            logger.info(
                "Performance metrics",
                function=func.__name__,
                execution_time_ms=round(execution_time * 1000, 2),
                memory_used_mb=round(memory_used / 1024 / 1024, 2),
                success=True,
            )

            return result
        except Exception as e:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss

            execution_time = end_time - start_time
            memory_used = end_memory - start_memory

            logger.error(
                "Performance metrics (error)",
                function=func.__name__,
                execution_time_ms=round(execution_time * 1000, 2),
                memory_used_mb=round(memory_used / 1024 / 1024, 2),
                error=str(e),
                success=False,
            )
            raise

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def log_errors(func: Callable) -> Callable:
    """
    Decorator to log all errors with detailed context.

    Args:
        func: The function to decorate

    Returns:
        Decorated function with error logging
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(
                "Function error occurred",
                function=func.__name__,
                error=str(e),
                error_type=type(e).__name__,
                module=func.__module__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()) if kwargs else [],
            )
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(
                "Function error occurred",
                function=func.__name__,
                error=str(e),
                error_type=type(e).__name__,
                module=func.__module__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()) if kwargs else [],
            )
            raise

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def validate_input(validation_func: Callable) -> Callable:
    """
    Decorator to validate function input parameters.

    Args:
        validation_func: Function that validates the input parameters

    Returns:
        Decorated function with input validation
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                validation_func(*args, **kwargs)
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "Input validation failed", function=func.__name__, validation_error=str(e)
                )
                raise ValidationError(f"Input validation failed for {func.__name__}: {e!s}")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                validation_func(*args, **kwargs)
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "Input validation failed", function=func.__name__, validation_error=str(e)
                )
                raise ValidationError(f"Input validation failed for {func.__name__}: {e!s}")

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def validate_output(validation_func: Callable) -> Callable:
    """
    Decorator to validate function output.

    Args:
        validation_func: Function that validates the output

    Returns:
        Decorated function with output validation
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            result = await func(*args, **kwargs)
            try:
                validation_func(result)
                return result
            except Exception as e:
                logger.error(
                    "Output validation failed", function=func.__name__, validation_error=str(e)
                )
                raise ValidationError(f"Output validation failed for {func.__name__}: {e!s}")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            result = func(*args, **kwargs)
            try:
                validation_func(result)
                return result
            except Exception as e:
                logger.error(
                    "Output validation failed", function=func.__name__, validation_error=str(e)
                )
                raise ValidationError(f"Output validation failed for {func.__name__}: {e!s}")

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def type_check(func: Callable) -> Callable:
    """
    Decorator to perform runtime type checking of function parameters and return value.

    Args:
        func: The function to decorate

    Returns:
        Decorated function with type checking
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        # Get function signature
        sig = inspect.signature(func)

        # Check argument types
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for param_name, param_value in bound_args.arguments.items():
            param = sig.parameters[param_name]
            if param.annotation != inspect.Parameter.empty:
                if not isinstance(param_value, param.annotation):
                    raise ValidationError(
                        f"Type mismatch for parameter '{param_name}': expected {param.annotation.__name__}, got {type(param_value).__name__}"
                    )

        result = await func(*args, **kwargs)

        # Check return type
        if sig.return_annotation != inspect.Parameter.empty:
            if not isinstance(result, sig.return_annotation):
                raise ValidationError(
                    f"Return type mismatch: expected {sig.return_annotation.__name__}, got {type(result).__name__}"
                )

        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        # Get function signature
        sig = inspect.signature(func)

        # Check argument types
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for param_name, param_value in bound_args.arguments.items():
            param = sig.parameters[param_name]
            if param.annotation != inspect.Parameter.empty:
                if not isinstance(param_value, param.annotation):
                    raise ValidationError(
                        f"Type mismatch for parameter '{param_name}': expected {param.annotation.__name__}, got {type(param_value).__name__}"
                    )

        result = func(*args, **kwargs)

        # Check return type
        if sig.return_annotation != inspect.Parameter.empty:
            if not isinstance(result, sig.return_annotation):
                raise ValidationError(
                    f"Return type mismatch: expected {sig.return_annotation.__name__}, got {type(result).__name__}"
                )

        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def rate_limit(max_calls: int, time_window: float) -> Callable:
    """
    Decorator to implement rate limiting for function calls.

    Args:
        max_calls: Maximum number of calls allowed in the time window
        time_window: Time window in seconds

    Returns:
        Decorated function with rate limiting
    """

    def decorator(func: Callable) -> Callable:
        # Rate limiting state
        call_times: list[float] = []

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            nonlocal call_times
            current_time = time.time()

            # Remove old calls outside the time window
            call_times = [t for t in call_times if current_time - t < time_window]

            # Check if we're at the rate limit
            if len(call_times) >= max_calls:
                oldest_call = min(call_times)
                wait_time = time_window - (current_time - oldest_call)
                logger.warning(
                    "Rate limit exceeded, waiting",
                    function=func.__name__,
                    wait_time_seconds=wait_time,
                    max_calls=max_calls,
                    time_window=time_window,
                )
                await asyncio.sleep(wait_time)

            # Record this call
            call_times.append(current_time)

            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            nonlocal call_times
            current_time = time.time()

            # Remove old calls outside the time window
            call_times = [t for t in call_times if current_time - t < time_window]

            # Check if we're at the rate limit
            if len(call_times) >= max_calls:
                oldest_call = min(call_times)
                wait_time = time_window - (current_time - oldest_call)
                logger.warning(
                    "Rate limit exceeded, waiting",
                    function=func.__name__,
                    wait_time_seconds=wait_time,
                    max_calls=max_calls,
                    time_window=time_window,
                )
                time.sleep(wait_time)

            # Record this call
            call_times.append(current_time)

            return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def api_throttle(max_calls: int, time_window: float) -> Callable:
    """
    Alias for rate_limit specifically for API calls.

    Args:
        max_calls: Maximum number of API calls allowed in the time window
        time_window: Time window in seconds

    Returns:
        Decorated function with API rate limiting
    """
    return rate_limit(max_calls, time_window)
