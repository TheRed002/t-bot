"""Unified decorator system for the T-Bot trading system."""

import asyncio
import functools
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, ClassVar, ParamSpec, TypeVar, cast

from src.core.exceptions import ValidationError
from src.core.logging import get_logger

# Note: Validation should be done via dependency injection, not global import

# Logger for module-level functions
logger = get_logger(__name__)


# Exception classification for retry logic
class ExceptionCategory:
    """Classification of exceptions for intelligent retry behavior."""

    # Permanent failures - should not be retried
    PERMANENT = {
        ValidationError,
        TypeError,
        KeyError,
        AttributeError,
        NotImplementedError,
        PermissionError,
        FileNotFoundError,
    }

    # Authentication/Authorization errors - usually permanent
    AUTH_ERRORS = {
        "Unauthorized",
        "Forbidden",
        "Invalid API key",
        "Authentication failed",
        "Access denied",
        "Token expired",
    }

    # Network errors - usually transient, good for retry
    NETWORK_ERRORS = {
        "ConnectionError",
        "TimeoutError",
        "ConnectTimeout",
        "ReadTimeout",
        "HTTPError",
        "NetworkError",
        "SocketError",
        "DNSError",
        "WebSocketError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "ConnectionRefusedError",
    }

    # Resource errors - may be transient or permanent
    RESOURCE_ERRORS = {
        "MemoryError",
        "DiskSpaceError",
        "ResourceExhausted",
        "RateLimitExceeded",
        "ServiceUnavailable",
    }

    # Trading-specific errors
    TRADING_TRANSIENT = {
        "MarketClosed",
        "InsufficientLiquidity",
        "PriceOutOfBounds",
        "ExchangeUnavailable",
        "OrderBookEmpty",
    }

    TRADING_PERMANENT = {
        "InvalidSymbol",
        "UnsupportedOrderType",
        "InsufficientBalance",
        "MarketNotFound",
        "InvalidOrderSize",
    }

    @classmethod
    def should_retry(cls, exception: Exception) -> bool:
        """Determine if an exception should be retried."""
        exc_type = type(exception)
        exc_message = str(exception)

        # Check for permanent exception types
        if exc_type in cls.PERMANENT:
            return False

        # Check for trading permanent errors
        if any(pattern in exc_message for pattern in cls.TRADING_PERMANENT):
            return False

        # Check for authentication errors
        if any(pattern in exc_message for pattern in cls.AUTH_ERRORS):
            return False

        # Network errors are generally retryable
        if any(pattern in exc_type.__name__ for pattern in cls.NETWORK_ERRORS):
            return True

        # Trading transient errors are retryable
        if any(pattern in exc_message for pattern in cls.TRADING_TRANSIENT):
            return True

        # Resource errors - depends on the specific error
        if any(pattern in exc_type.__name__ for pattern in cls.RESOURCE_ERRORS):
            # Rate limits and service unavailable are retryable
            if "RateLimit" in exc_message or "ServiceUnavailable" in exc_message:
                return True
            # Memory errors are usually not retryable
            if "MemoryError" in exc_type.__name__:
                return False
            # Default to retryable for other resource errors
            return True

        # Default: retry for unknown exceptions (conservative approach)
        return True

    @classmethod
    def get_retry_delay(cls, exception: Exception, attempt: int, base_delay: float) -> float:
        """Get appropriate retry delay based on exception type and attempt."""
        exc_type = type(exception)
        exc_message = str(exception)

        # Rate limiting errors - use longer delays
        if "RateLimit" in exc_message or "TooManyRequests" in exc_message:
            # Extract rate limit reset time if available
            if "retry after" in exc_message.lower():
                try:
                    import re

                    match = re.search(r"retry after (\d+)", exc_message.lower())
                    if match:
                        return float(match.group(1))
                except (ImportError, AttributeError, ValueError) as e:
                    logger.debug(f"Failed to extract retry delay from rate limit message: {e}")
                    pass
            # Default rate limit backoff
            return base_delay * (3**attempt)  # More aggressive backoff

        # Network errors - standard exponential backoff with jitter
        if any(pattern in exc_type.__name__ for pattern in cls.NETWORK_ERRORS):
            import random

            jitter = random.uniform(0.1, 0.3) * base_delay
            return (base_delay * (2**attempt)) + jitter

        # Service unavailable - moderate backoff
        if "ServiceUnavailable" in exc_message:
            return base_delay * (2.5**attempt)

        # Default exponential backoff
        return base_delay * (2**attempt)


# Define proper type variables
P = ParamSpec("P")
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class UnifiedDecorator:
    """
    Single configurable decorator replacing multiple decorators.
    Provides retry, validation, logging, caching, and monitoring capabilities.
    """

    # Class-level cache storage with proper memory management
    _cache: ClassVar[dict[str, Any]] = {}
    _cache_timestamps: ClassVar[dict[str, datetime]] = {}
    _last_cache_cleanup: ClassVar[datetime] = datetime.now(timezone.utc)
    _max_cache_entries: ClassVar[int] = 1000  # Reduced from 10000 to prevent unbounded growth
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()  # Thread-safe access
    _cleanup_in_progress: ClassVar[bool] = False  # Prevent concurrent cleanups

    @classmethod
    def _get_cache_key(
        cls, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> str:
        """Generate cache key from function and arguments."""
        func_name = f"{func.__module__}.{func.__name__}"
        # Create hashable key from args and kwargs
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        return f"{func_name}:{args_str}:{kwargs_str}"

    @classmethod
    def _cache_result(
        cls,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
        ttl: int,
    ) -> None:
        """Cache function result with TTL and automatic cleanup."""
        key = cls._get_cache_key(func, args, kwargs)

        with cls._cache_lock:
            # Force cleanup if cache is getting too large
            if len(cls._cache) >= cls._max_cache_entries:
                cls._force_cache_cleanup()

            cls._cache[key] = result
            cls._cache_timestamps[key] = datetime.now(timezone.utc) + timedelta(seconds=ttl)

    @classmethod
    def _get_cached_result(
        cls, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any | None:
        """Get cached result if still valid."""
        key = cls._get_cache_key(func, args, kwargs)

        with cls._cache_lock:
            # Periodically cleanup expired entries (non-blocking)
            cls._cleanup_cache_if_needed()

            if key in cls._cache:
                expiry_time = cls._cache_timestamps.get(key, datetime.min)
                if datetime.now(timezone.utc) < expiry_time:
                    return cls._cache[key]
                else:
                    # Cache expired, remove it immediately
                    cls._cache.pop(key, None)
                    cls._cache_timestamps.pop(key, None)

        return None

    @classmethod
    def _cleanup_cache_if_needed(cls) -> None:
        """Clean up expired cache entries if needed (non-blocking)."""
        # Only run if not already in progress and enough time has passed
        if cls._cleanup_in_progress:
            return

        now = datetime.now(timezone.utc)

        # Cleanup every 30 minutes or when cache is getting large
        should_cleanup = (
            now - cls._last_cache_cleanup > timedelta(minutes=30)
            or len(cls._cache) > cls._max_cache_entries * 0.8  # 80% threshold
        )

        if should_cleanup:
            cls._cleanup_in_progress = True
            try:
                # Remove expired entries
                expired_keys = [
                    key for key, timestamp in cls._cache_timestamps.items() if now >= timestamp
                ]

                for key in expired_keys:
                    try:
                        cls._cache.pop(key, None)
                        cls._cache_timestamps.pop(key, None)
                    except KeyError as e:
                        logger.debug(f"Key removed during cache cleanup: {e}")
                        pass

                # If still too many entries, remove oldest 20%
                if len(cls._cache) > cls._max_cache_entries * 0.8:
                    sorted_items = sorted(
                        cls._cache_timestamps.items(), key=lambda x: x[1], reverse=True
                    )
                    # Keep 80% of max entries
                    keep_count = int(cls._max_cache_entries * 0.8)
                    keep_keys = {k for k, _ in sorted_items[:keep_count]}

                    # Remove entries not in keep_keys
                    try:
                        cls._cache = {k: v for k, v in cls._cache.items() if k in keep_keys}
                        cls._cache_timestamps = {
                            k: v for k, v in cls._cache_timestamps.items() if k in keep_keys
                        }
                    except RuntimeError as e:
                        logger.debug(f"Dictionary changed during cleanup: {e}")
                        pass

                cls._last_cache_cleanup = now
            finally:
                cls._cleanup_in_progress = False

    @classmethod
    def _force_cache_cleanup(cls) -> None:
        """Force immediate cache cleanup when at capacity."""
        now = datetime.now(timezone.utc)

        # Remove expired entries first
        expired_keys = [key for key, timestamp in cls._cache_timestamps.items() if now >= timestamp]

        for key in expired_keys:
            try:
                cls._cache.pop(key, None)
                cls._cache_timestamps.pop(key, None)
            except KeyError as e:
                logger.debug(f"Key removed during force cleanup: {e}")
                pass

        # If still at capacity, remove oldest 30%
        if len(cls._cache) >= cls._max_cache_entries:
            sorted_items = sorted(cls._cache_timestamps.items(), key=lambda x: x[1], reverse=True)
            # Keep only 70% of entries
            keep_count = int(cls._max_cache_entries * 0.7)
            keep_keys = {k for k, _ in sorted_items[:keep_count]}

            try:
                cls._cache = {k: v for k, v in cls._cache.items() if k in keep_keys}
                cls._cache_timestamps = {
                    k: v for k, v in cls._cache_timestamps.items() if k in keep_keys
                }
            except RuntimeError as e:
                logger.debug(f"Dictionary changed during force cleanup: {e}")
                pass

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached results - useful for testing and cleanup."""
        with cls._cache_lock:
            cls._cache.clear()
            cls._cache_timestamps.clear()
            cls._last_cache_cleanup = datetime.now(timezone.utc)

    @classmethod
    async def _with_retry(
        cls,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        retry_times: int,
        retry_delay: float,
        logger: Any | None = None,
    ) -> Any:
        """Execute function with intelligent retry logic and exception classification."""
        last_exception = None
        correlation_id = str(uuid.uuid4())[:8]

        for attempt in range(retry_times):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                # Enhanced exception classification
                should_retry = ExceptionCategory.should_retry(e)

                if not should_retry:
                    if logger:
                        logger.error(
                            f"Permanent failure in {func.__name__} (not retrying): "
                            f"{type(e).__name__}: {e}. Correlation: {correlation_id}"
                        )
                    # Add correlation ID to exception for tracking
                    if not hasattr(e, "correlation_id"):
                        e.correlation_id = correlation_id  # type: ignore
                    raise

                if logger:
                    logger.warning(
                        f"Attempt {attempt + 1}/{retry_times} failed for {func.__name__}: "
                        f"{type(e).__name__}: {e}. Correlation: {correlation_id}"
                    )

                # Check if we should continue retrying
                if attempt < retry_times - 1:
                    # Calculate intelligent retry delay
                    delay = ExceptionCategory.get_retry_delay(e, attempt, retry_delay)

                    if logger:
                        logger.debug(
                            f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt + 1}/{retry_times}). "
                            f"Correlation: {correlation_id}"
                        )

                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    if logger:
                        logger.error(
                            f"All {retry_times} retry attempts failed for {func.__name__}. "
                            f"Final error: {type(e).__name__}: {e}. Correlation: {correlation_id}"
                        )

        if last_exception:
            # Add correlation ID for final tracking
            if not hasattr(last_exception, "correlation_id"):
                last_exception.correlation_id = correlation_id  # type: ignore
            raise last_exception
        else:
            final_error = RuntimeError(
                f"Failed to execute {func.__name__} after {retry_times} attempts. "
                f"Correlation: {correlation_id}"
            )
            final_error.correlation_id = correlation_id  # type: ignore
            raise final_error

    @classmethod
    def _record_metrics(cls, func: Callable[..., Any], result: Any, execution_time: float) -> None:
        """Record execution metrics (placeholder for actual metrics system)."""
        # This would integrate with your metrics system
        # For now, just log it using the base logger
        base_logger = get_logger(__name__)
        base_logger.debug(f"Metrics: {func.__name__} completed in {execution_time:.3f}s")

    @classmethod
    def _bind_function_arguments(
        cls, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Bind function arguments and apply defaults."""
        import inspect

        sig = inspect.signature(func)
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return bound
        except TypeError as e:
            raise ValidationError(f"Invalid arguments for {func.__name__}: {e}") from e

    @classmethod
    def _validate_numeric_param(cls, param_name: str, value: Any) -> None:
        """Validate numeric parameters like price and quantity."""
        if not isinstance(value, int | float | Decimal):
            raise ValidationError(f"{param_name} must be numeric, got {type(value).__name__}")
        # Use Decimal for comparison to maintain precision
        if Decimal(str(value)) <= 0:
            raise ValidationError(f"{param_name} must be positive")

    @classmethod
    def _validate_symbol_param(cls, param_name: str, value: Any) -> None:
        """Validate symbol parameters."""
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"{param_name} must be a non-empty string")

    @classmethod
    def _validate_single_param(cls, param_name: str, value: Any) -> None:
        """Validate a single parameter based on its name and value."""
        if value is None:
            return

        try:
            # Basic type validation only
            if "price" in param_name.lower() or "quantity" in param_name.lower():
                cls._validate_numeric_param(param_name, value)
            elif "symbol" in param_name.lower():
                cls._validate_symbol_param(param_name, value)
        except Exception as validation_error:
            raise ValidationError(
                f"Validation failed for {param_name}: {validation_error}"
            ) from validation_error

    @classmethod
    def _validate_args(
        cls, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        """Validate function arguments based on annotations."""
        import inspect

        sig = inspect.signature(func)
        bound = cls._bind_function_arguments(func, args, kwargs)

        # Validate based on type hints
        for param_name, param in sig.parameters.items():
            if param.annotation != param.empty:
                value = bound.arguments.get(param_name)
                # Special validation for common types - lazy import to avoid circular deps
                # NOTE: This is kept simple to avoid circular dependencies
                # For full validation, use the ValidationService via dependency injection
                cls._validate_single_param(param_name, value)

    @classmethod
    def _create_enhancement_config(
        cls,
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
        fallback: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Create configuration for function enhancement."""
        return {
            "retry": retry,
            "retry_times": retry_times,
            "retry_delay": retry_delay,
            "validate": validate,
            "log": log,
            "log_level": log_level,
            "cache": cache,
            "cache_ttl": cache_ttl,
            "monitor": monitor,
            "timeout": timeout,
            "fallback": fallback,
        }

    @classmethod
    def _prepare_execution_context(
        cls, func: Callable[..., Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare execution context for enhanced function."""
        logger = get_logger(func.__module__) if config["log"] else None
        start_time = time.time()
        return {"logger": logger, "start_time": start_time}

    @classmethod
    def _handle_validation(
        cls,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        """Handle argument validation if enabled."""
        if config["validate"]:
            try:
                cls._validate_args(func, args, kwargs)
            except Exception as e:
                if context["logger"]:
                    context["logger"].error(f"Validation failed for {func.__name__}: {e}")
                raise

    @classmethod
    def _handle_pre_execution_logging(
        cls,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        """Handle logging before function execution."""
        if config["log"] and context["logger"]:
            log_method = getattr(context["logger"], config["log_level"], context["logger"].debug)
            log_method(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")

    @classmethod
    def _handle_cache_check(
        cls,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:
        """Check cache for existing result."""
        if config["cache"]:
            cached_result = cls._get_cached_result(func, args, kwargs)
            if cached_result is not None:
                if context["logger"]:
                    context["logger"].debug(f"Cache hit for {func.__name__}")
                return cached_result
        return None

    @classmethod
    def _handle_post_execution(
        cls,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        """Handle post-execution tasks like caching, monitoring, and logging."""
        # Cache result
        if config["cache"]:
            cls._cache_result(func, args, kwargs, result, config["cache_ttl"])

        # Monitor
        if config["monitor"]:
            execution_time = time.time() - context["start_time"]
            cls._record_metrics(func, result, execution_time)

        # Log success
        if config["log"] and context["logger"]:
            execution_time = time.time() - context["start_time"]
            log_method = getattr(context["logger"], config["log_level"], context["logger"].debug)
            log_method(f"{func.__name__} completed in {execution_time:.3f}s")

    @classmethod
    def enhance(
        cls,
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
        fallback: Callable[..., Any] | None = None,
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
            config = cls._create_enhancement_config(
                retry,
                retry_times,
                retry_delay,
                validate,
                log,
                log_level,
                cache,
                cache_ttl,
                monitor,
                timeout,
                fallback,
            )

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await cls._execute_async_enhanced(func, args, kwargs, config)

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return cls._execute_sync_enhanced(func, args, kwargs, config)

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return cast(F, async_wrapper)
            else:
                return cast(F, sync_wrapper)

        return decorator

    @classmethod
    async def _execute_async_enhanced(
        cls,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        config: dict[str, Any],
    ) -> Any:
        """Execute async function with enhancements."""
        context = cls._prepare_execution_context(func, config)

        cls._handle_validation(func, args, kwargs, config, context)
        cls._handle_pre_execution_logging(func, args, kwargs, config, context)

        cached_result = cls._handle_cache_check(func, args, kwargs, config, context)
        if cached_result is not None:
            return cached_result

        try:
            # Execute function with timeout handling
            if config["timeout"]:
                try:
                    result = await asyncio.wait_for(
                        cls._execute_with_retry(
                            func,
                            args,
                            kwargs,
                            config["retry"],
                            config["retry_times"],
                            config["retry_delay"],
                            context["logger"],
                        ),
                        timeout=config["timeout"],
                    )
                except asyncio.TimeoutError as timeout_error:
                    # Add timeout context for better error tracking
                    timeout_error.correlation_id = getattr(  # type: ignore[attr-defined]
                        timeout_error, "correlation_id", str(uuid.uuid4())[:8]
                    )
                    raise timeout_error
            else:
                result = await cls._execute_with_retry(
                    func,
                    args,
                    kwargs,
                    config["retry"],
                    config["retry_times"],
                    config["retry_delay"],
                    context["logger"],
                )

            cls._handle_post_execution(func, args, kwargs, result, config, context)
            return result

        except Exception as e:
            return await cls._handle_execution_error_async(func, args, kwargs, e, config, context)

    @classmethod
    def _execute_sync_enhanced(
        cls,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        config: dict[str, Any],
    ) -> Any:
        """Execute sync function with enhancements."""
        context = cls._prepare_execution_context(func, config)

        cls._handle_validation(func, args, kwargs, config, context)
        cls._handle_pre_execution_logging(func, args, kwargs, config, context)

        cached_result = cls._handle_cache_check(func, args, kwargs, config, context)
        if cached_result is not None:
            return cached_result

        try:
            result = cls._execute_sync_with_retry(func, args, kwargs, config, context)
            cls._handle_post_execution(func, args, kwargs, result, config, context)
            return result

        except Exception as e:
            return cls._handle_execution_error_sync(func, args, kwargs, e, config, context)

    @classmethod
    def _execute_sync_with_retry(
        cls,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:
        """Execute sync function with intelligent retry logic."""
        if not config["retry"]:
            return func(*args, **kwargs)

        last_exception = None
        correlation_id = str(uuid.uuid4())[:8]

        for attempt in range(config["retry_times"]):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                # Enhanced exception classification
                should_retry = ExceptionCategory.should_retry(e)

                if not should_retry:
                    if context["logger"]:
                        context["logger"].error(
                            f"Permanent failure in {func.__name__} (not retrying): "
                            f"{type(e).__name__}: {e}. Correlation: {correlation_id}"
                        )
                    # Add correlation ID to exception
                    if not hasattr(e, "correlation_id"):
                        e.correlation_id = correlation_id  # type: ignore
                    raise

                if context["logger"]:
                    context["logger"].warning(
                        f"Attempt {attempt + 1}/{config['retry_times']} failed for "
                        f"{func.__name__}: {type(e).__name__}: {e}. Correlation: {correlation_id}"
                    )

                if attempt < config["retry_times"] - 1:
                    # Calculate intelligent retry delay
                    delay = ExceptionCategory.get_retry_delay(e, attempt, config["retry_delay"])

                    if context["logger"]:
                        context["logger"].debug(
                            f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt + 1}/{config['retry_times']}). "
                            f"Correlation: {correlation_id}"
                        )

                    time.sleep(delay)
                else:
                    # Final attempt failed
                    if context["logger"]:
                        context["logger"].error(
                            f"All {config['retry_times']} retry attempts failed for {func.__name__}. "
                            f"Final error: {type(e).__name__}: {e}. Correlation: {correlation_id}"
                        )

        if last_exception:
            # Add correlation ID for final tracking
            if not hasattr(last_exception, "correlation_id"):
                last_exception.correlation_id = correlation_id  # type: ignore
            raise last_exception
        else:
            final_error = RuntimeError(
                f"Failed to execute {func.__name__} after {config['retry_times']} attempts. "
                f"Correlation: {correlation_id}"
            )
            final_error.correlation_id = correlation_id  # type: ignore
            raise final_error

    @classmethod
    async def _handle_execution_error_async(
        cls,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        error: Exception,
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:
        """Handle execution errors and fallbacks for async functions with correlation tracking."""
        correlation_id = getattr(error, "correlation_id", str(uuid.uuid4())[:8])

        if context["logger"]:
            context["logger"].error(
                f"{func.__name__} failed: {type(error).__name__}: {error}. "
                f"Correlation: {correlation_id}"
            )

        # Use fallback if provided
        if config["fallback"]:
            if context["logger"]:
                context["logger"].info(
                    f"Using fallback for {func.__name__}. Correlation: {correlation_id}"
                )

            try:
                fallback_result = config["fallback"](*args, **kwargs)
                # If fallback is async, await it
                if asyncio.iscoroutine(fallback_result):
                    return await fallback_result
                return fallback_result
            except Exception as fallback_error:
                if context["logger"]:
                    context["logger"].error(
                        f"Fallback for {func.__name__} also failed: {type(fallback_error).__name__}: {fallback_error}. "
                        f"Correlation: {correlation_id}"
                    )
                # Chain the original error with fallback error
                fallback_error.__cause__ = error
                if not hasattr(fallback_error, "correlation_id"):
                    fallback_error.correlation_id = correlation_id  # type: ignore
                raise fallback_error

        # Ensure error has correlation ID before raising
        if not hasattr(error, "correlation_id"):
            error.correlation_id = correlation_id  # type: ignore
        raise error

    @classmethod
    def _handle_execution_error_sync(
        cls,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        error: Exception,
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:
        """Handle execution errors and fallbacks for sync functions with correlation tracking."""
        correlation_id = getattr(error, "correlation_id", str(uuid.uuid4())[:8])

        if context["logger"]:
            context["logger"].error(
                f"{func.__name__} failed: {type(error).__name__}: {error}. "
                f"Correlation: {correlation_id}"
            )

        # Use fallback if provided
        if config["fallback"]:
            if context["logger"]:
                context["logger"].info(
                    f"Using fallback for {func.__name__}. Correlation: {correlation_id}"
                )
            try:
                return config["fallback"](*args, **kwargs)
            except Exception as fallback_error:
                if context["logger"]:
                    context["logger"].error(
                        f"Fallback for {func.__name__} also failed: {type(fallback_error).__name__}: {fallback_error}. "
                        f"Correlation: {correlation_id}"
                    )
                # Chain the original error with fallback error
                fallback_error.__cause__ = error
                if not hasattr(fallback_error, "correlation_id"):
                    fallback_error.correlation_id = correlation_id  # type: ignore
                raise fallback_error

        # Ensure error has correlation ID before raising
        if not hasattr(error, "correlation_id"):
            error.correlation_id = correlation_id  # type: ignore
        raise error

    @classmethod
    async def _execute_with_retry(
        cls,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
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
def retry(
    max_attempts: int = 3, delay: float = 1.0, base_delay: float | None = None
) -> Callable[[F], F]:
    """Retry decorator with exponential backoff."""
    # Handle different parameter names for backward compatibility
    actual_delay = base_delay if base_delay is not None else delay
    return dec.enhance(retry=True, retry_times=max_attempts, retry_delay=actual_delay)


def cached(ttl: int = 300) -> Callable[[F], F]:
    """Cache decorator with TTL."""
    return dec.enhance(cache=True, cache_ttl=ttl)


def validated() -> Callable[[F], F]:
    """Validation decorator."""
    return dec.enhance(validate=True)


def logged(level: str = "info") -> Callable[[F], F]:
    """Logging decorator."""
    return dec.enhance(log=True, log_level=level)


def monitored() -> Callable[[F], F]:
    """Monitoring decorator."""
    return dec.enhance(monitor=True, log=True, log_level="debug")


def timeout(seconds: float) -> Callable[[F], F]:
    """Timeout decorator."""
    return dec.enhance(timeout=seconds)


# Hybrid factory functions supporting both bare and parameterized usage
def _make_hybrid_decorator(
    base_decorator: Callable[..., Callable[[F], F]],
    default_args: tuple = (),
    default_kwargs: dict | None = None,
) -> Callable[..., Any]:
    """Create a hybrid decorator that works with or without parameters."""
    default_kwargs = default_kwargs or {}

    def hybrid(func_or_param: F | Any = None, **kwargs: Any) -> F | Callable[[F], F]:
        if callable(func_or_param):
            # Used as bare decorator: @decorator
            return base_decorator(*default_args, **default_kwargs)(cast(F, func_or_param))
        else:
            # Used with parameters: @decorator(param) or @decorator(param=value)
            if func_or_param is not None:
                # Positional parameter provided
                if default_args:
                    # Replace first default arg with provided value
                    args = (func_or_param, *default_args[1:])
                else:
                    args = (func_or_param,)
                # Merge kwargs, giving priority to explicitly passed kwargs
                merged_kwargs = {**default_kwargs, **kwargs}
                return base_decorator(*args, **merged_kwargs)
            else:
                # Only keyword arguments provided, don't use default_args to avoid conflicts
                merged_kwargs = {**default_kwargs, **kwargs}
                return base_decorator(**merged_kwargs)

    return hybrid


# Factory functions using hybrid pattern for backward compatibility
cache_result = _make_hybrid_decorator(cached, (300,))
validate_input = _make_hybrid_decorator(validated, ())
log_calls = _make_hybrid_decorator(logged, ("info",))
log_errors = _make_hybrid_decorator(logged, ("error",))
log_performance = _make_hybrid_decorator(monitored, ())
api_throttle = _make_hybrid_decorator(monitored, ())
rate_limit = _make_hybrid_decorator(monitored, ())
redis_cache = _make_hybrid_decorator(cached, (300,))
time_execution = _make_hybrid_decorator(monitored, ())
ttl_cache = _make_hybrid_decorator(cached, (300,))
type_check = _make_hybrid_decorator(validated, ())
validate_output = _make_hybrid_decorator(validated, ())
cpu_usage = _make_hybrid_decorator(monitored, ())
memory_usage = _make_hybrid_decorator(monitored, ())


def circuit_breaker(failure_threshold: int = 5, recovery_timeout: int = 60) -> Callable[[F], F]:
    """Circuit breaker decorator factory."""
    return dec.enhance(retry=True, retry_times=failure_threshold)


def robust() -> Callable[[F], F]:
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


# Aliases for compatibility with web interface
with_cache = cached
with_monitoring = monitored
with_error_handler = robust
with_audit_trail = logged
