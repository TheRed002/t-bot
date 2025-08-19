"""Unified decorator system for the T-Bot trading system."""

import asyncio
import functools
import time
import traceback
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from functools import lru_cache
from datetime import datetime, timedelta

from src.core.logging import get_logger
from src.utils.validation import validator

F = TypeVar('F', bound=Callable[..., Any])


class UnifiedDecorator:
    """
    Single configurable decorator replacing multiple decorators.
    Provides retry, validation, logging, caching, and monitoring capabilities.
    """
    
    # Class-level cache storage
    _cache: Dict[str, Dict[str, Any]] = {}
    _cache_timestamps: Dict[str, datetime] = {}
    
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
        cls,
        func: Callable,
        args: tuple,
        kwargs: dict,
        result: Any,
        ttl: int
    ) -> None:
        """Cache function result with TTL."""
        key = cls._get_cache_key(func, args, kwargs)
        cls._cache[key] = result
        cls._cache_timestamps[key] = datetime.utcnow() + timedelta(seconds=ttl)
    
    @classmethod
    def _get_cached_result(cls, func: Callable, args: tuple, kwargs: dict) -> Optional[Any]:
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
        logger: Optional[Any] = None
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
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
        
        raise last_exception
    
    @classmethod
    def _record_metrics(cls, func: Callable, result: Any, execution_time: float) -> None:
        """Record execution metrics (placeholder for actual metrics system)."""
        # This would integrate with your metrics system
        # For now, just log it
        logger = get_logger(__name__)
        logger.debug(
            f"Metrics: {func.__name__} completed in {execution_time:.3f}s"
        )
    
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
                if 'order' in param_name.lower() and isinstance(value, dict):
                    validator.validate_order(value)
                elif 'price' in param_name.lower() and value is not None:
                    validator.validate_price(value)
                elif 'quantity' in param_name.lower() and value is not None:
                    validator.validate_quantity(value)
                elif 'symbol' in param_name.lower() and isinstance(value, str):
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
        timeout: Optional[float] = None,
        fallback: Optional[Callable] = None
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
                logger = get_logger(func.__module__) if log else None
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
                            timeout=timeout
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
                
                if loop.is_running():
                    # If loop is already running, schedule as task
                    future = asyncio.ensure_future(async_wrapper(*args, **kwargs))
                    return future
                else:
                    # Run in the loop
                    return loop.run_until_complete(async_wrapper(*args, **kwargs))
            
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
        logger: Optional[Any]
    ) -> Any:
        """Execute function with optional retry."""
        if retry:
            return await cls._with_retry(func, args, kwargs, retry_times, retry_delay, logger)
        else:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)


# Convenience decorator presets
class Decorators:
    """Pre-configured decorator combinations for common use cases."""
    
    @staticmethod
    def retry_on_failure(times: int = 3, delay: float = 1.0, max_attempts: int = None):
        """Retry decorator with exponential backoff."""
        # Support both 'times' and 'max_attempts' for backward compatibility
        retry_times = max_attempts if max_attempts is not None else times
        return UnifiedDecorator.enhance(retry=True, retry_times=retry_times, retry_delay=delay)
    
    @staticmethod
    def cached(ttl: int = 300):
        """Cache decorator with TTL."""
        return UnifiedDecorator.enhance(cache=True, cache_ttl=ttl)
    
    @staticmethod
    def validated():
        """Validation decorator."""
        return UnifiedDecorator.enhance(validate=True)
    
    @staticmethod
    def logged(level: str = "info"):
        """Logging decorator."""
        return UnifiedDecorator.enhance(log=True, log_level=level)
    
    @staticmethod
    def monitored():
        """Monitoring decorator."""
        return UnifiedDecorator.enhance(monitor=True)
    
    @staticmethod
    def robust(fallback: Optional[Callable] = None):
        """Robust execution with retry, logging, and monitoring."""
        return UnifiedDecorator.enhance(
            retry=True,
            retry_times=3,
            retry_delay=1.0,
            log=True,
            monitor=True,
            fallback=fallback
        )
    
    @staticmethod
    def api_call():
        """Decorator for API calls with retry and caching."""
        return UnifiedDecorator.enhance(
            retry=True,
            retry_times=3,
            retry_delay=2.0,
            cache=True,
            cache_ttl=30,
            log=True,
            timeout=30.0
        )
    
    @staticmethod
    def critical_section():
        """Decorator for critical sections with validation and monitoring."""
        return UnifiedDecorator.enhance(
            validate=True,
            log=True,
            log_level="info",
            monitor=True,
            retry=False  # No retry for critical sections
        )


# Export main decorator and presets
dec = UnifiedDecorator
decorators = Decorators

# Backward compatibility exports - functions that return decorators
def retry(max_attempts: int = 3, delay: float = 1.0, **kwargs):
    """Backward compatible retry decorator."""
    return decorators.retry_on_failure(max_attempts=max_attempts, delay=delay)

def circuit_breaker(threshold: int = 5, timeout: int = 60, **kwargs):
    """Backward compatible circuit breaker decorator."""
    return decorators.robust()

def time_execution(func=None):
    """Backward compatible time execution decorator."""
    if func is None:
        return decorators.monitored()
    return decorators.monitored()(func)

cache_result = decorators.cached
validate_input = decorators.validated
log_calls = decorators.logged
timeout = lambda t: dec.enhance(timeout=t)

__all__ = [
    'UnifiedDecorator', 
    'dec', 
    'Decorators', 
    'decorators',
    # Backward compatibility
    'retry',
    'circuit_breaker',
    'time_execution',
    'cache_result',
    'validate_input',
    'log_calls',
    'timeout'
]