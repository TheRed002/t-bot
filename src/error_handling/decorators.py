"""
Enhanced error handling decorators for comprehensive coverage.

This module provides advanced error handling decorators that eliminate the need
for repetitive try-catch blocks throughout the codebase. Features include:
- Universal error decorator that handles all error types
- Circuit breaker patterns with automatic recovery
- Retry mechanisms with exponential backoff and jitter
- Error categorization and routing
- Structured error logging with correlation IDs
- Performance metrics and monitoring
- Graceful degradation patterns
- Context-aware error handling

Replace repetitive try-catch blocks with intelligent decorators.
"""

import asyncio
import atexit
import functools
import random
import sys
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, ClassVar, TypeVar

# Import consolidated classes from context module
from src.core.exceptions import (
    ConfigurationError,
    DatabaseConnectionError,
    DataError,
    ErrorCategory,
    ExchangeError,
    ExecutionError,
    ModelError,
    NetworkError,
    RiskManagementError,
    SecurityError,
    ServiceError,
    StateConsistencyError,
    TradingBotError,
    ValidationError,
)
from src.core.logging import correlation_context, get_logger
from src.error_handling.context import ErrorContext
from src.error_handling.security_sanitizer import (
    SensitivityLevel,
)
from src.utils.error_categorization import (
    categorize_by_keywords,
    get_fallback_type_keywords,
    is_sensitive_key,
)

# Import existing decorators to avoid duplication

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Configuration Constants
DEFAULT_CACHE_SIZE = 1000
DEFAULT_CACHE_TTL_SECONDS = 3600  # 1 hour
DEFAULT_CACHE_MEMORY_LIMIT_MB = 100
DEFAULT_EVICTION_PERCENTAGE = 0.3  # 30%
DEFAULT_CLEANUP_INTERVAL_MINUTES = 30
DEFAULT_HEALTH_CHECK_INTERVAL_SECONDS = 30
DEFAULT_MAX_OPERATION_TIMES = 1000
DEFAULT_TASK_TIMEOUT_SECONDS = 30
DEFAULT_FALLBACK_CACHE_TTL_SECONDS = 1800  # 30 min
DEFAULT_RESULT_CACHE_TTL_SECONDS = 300  # 5 min

# Memory usage thresholds
HIGH_MEMORY_WARNING_THRESHOLD_MB = 50
MEMORY_CLEANUP_SLEEP_SECONDS = 30
PERIODIC_CLEANUP_SLEEP_SECONDS = 1
PERIODIC_CLEANUP_ITERATIONS = 300  # 5 minutes


class LRUCache:
    """Thread-safe LRU cache with TTL support and memory limits."""

    def __init__(
        self, max_size: int = DEFAULT_CACHE_SIZE, ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS
    ) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, datetime]] = OrderedDict()
        self._lock = threading.RLock()
        self._memory_usage = 0
        self._max_memory_mb = DEFAULT_CACHE_MEMORY_LIMIT_MB

    def get(self, key: str) -> Any | None:
        """Get value from cache if not expired."""
        with self._lock:
            if key not in self._cache:
                return None

            value, timestamp = self._cache[key]

            # Check TTL expiration
            if (datetime.now(timezone.utc) - timestamp).total_seconds() > self.ttl_seconds:
                del self._cache[key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return value

    def put(self, key: str, value: Any) -> None:
        """Put value in cache with size and memory limits."""
        with self._lock:
            current_time = datetime.now(timezone.utc)

            # Remove expired entries first
            self._cleanup_expired()

            # Estimate memory usage more accurately
            estimated_size = self._estimate_object_size(key, value)

            # Check memory limit
            if self._memory_usage + estimated_size > self._max_memory_mb * 1024 * 1024:
                self._evict_lru_entries(DEFAULT_EVICTION_PERCENTAGE)

            # Remove LRU entries if at max size
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # Remove least recently used

            self._cache[key] = (value, current_time)
            self._memory_usage += estimated_size

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = datetime.now(timezone.utc)
        expired_keys = []

        for key, (_, timestamp) in self._cache.items():
            if (current_time - timestamp).total_seconds() > self.ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]

    def cleanup_expired(self) -> None:
        """Public method to remove expired entries."""
        self._cleanup_expired()

    def _evict_lru_entries(self, percentage: float) -> None:
        """Evict a percentage of LRU entries."""
        entries_to_remove = int(len(self._cache) * percentage)
        for _ in range(entries_to_remove):
            if self._cache:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._memory_usage = 0

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def memory_usage_mb(self) -> float:
        """Get estimated memory usage in MB."""
        return self._memory_usage / (1024 * 1024)

    def _estimate_object_size(self, key: str, value: Any) -> int:
        """Estimate memory size of an object more accurately."""
        # Base size for key
        size = sys.getsizeof(key)

        # Estimate value size based on type
        if value is None:
            return size + 16
        elif isinstance(value, int | float | bool):
            size += sys.getsizeof(value)
        elif isinstance(value, str):
            size += sys.getsizeof(value)
        elif isinstance(value, list | tuple):
            size += sys.getsizeof(value)
            # Add size of elements (simplified)
            for item in value[:10]:  # Sample first 10 items
                size += self._estimate_object_size("", item) // 2
        elif isinstance(value, dict):
            size += sys.getsizeof(value)
            # Add size of key-value pairs (simplified)
            for k, v in list(value.items())[:10]:  # Sample first 10 items
                size += self._estimate_object_size(str(k), v) // 2
        else:
            # For other objects, use string representation as fallback
            size += len(str(value)) * 2  # Unicode chars

        return size


class FallbackStrategy(Enum):
    """Fallback strategies for error handling."""

    RETURN_NONE = "return_none"
    RETURN_EMPTY = "return_empty"
    RETURN_DEFAULT = "return_default"
    RAISE_DEGRADED = "raise_degraded"
    RETRY_ALTERNATIVE = "retry_alternative"
    USE_CACHE = "use_cache"


# ErrorContext is now imported from src.error_handling.context


@dataclass
class RetryConfig:
    """Configuration for retry behavior with production-grade validation.

    Attributes:
        max_attempts: Maximum number of retry attempts (1-10)
        base_delay: Base delay between retries in seconds (0.1-30.0)
        max_delay: Maximum delay cap in seconds (1.0-300.0)
        backoff_factor: Exponential backoff multiplier (1.1-10.0)
        jitter: Enable jitter to prevent thundering herd
        exponential: Use exponential backoff vs linear
        retriable_errors: Tuple of exception types that should be retried
    """

    max_attempts: int = 3
    base_delay: Decimal = Decimal("1.0")
    max_delay: Decimal = Decimal("60.0")
    backoff_factor: Decimal = Decimal("2.0")
    jitter: bool = True
    exponential: bool = True
    retriable_errors: tuple = (
        NetworkError,
        DatabaseConnectionError,
        ExchangeError,
        ServiceError,
    )

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate retry configuration parameters."""
        if not isinstance(self.max_attempts, int) or not (1 <= self.max_attempts <= 10):
            raise ValueError(
                f"max_attempts must be an integer between 1 and 10, got {self.max_attempts}"
            )

        if not isinstance(self.base_delay, int | float | Decimal) or not (
            Decimal("0.1") <= Decimal(str(self.base_delay)) <= Decimal("30.0")
        ):
            raise ValueError(
                f"base_delay must be between 0.1 and 30.0 seconds, got {self.base_delay}"
            )

        if not isinstance(self.max_delay, int | float | Decimal) or not (
            Decimal("1.0") <= Decimal(str(self.max_delay)) <= Decimal("300.0")
        ):
            raise ValueError(
                f"max_delay must be between 1.0 and 300.0 seconds, got {self.max_delay}"
            )

        if not isinstance(self.backoff_factor, int | float | Decimal) or not (
            Decimal("1.1") <= Decimal(str(self.backoff_factor)) <= Decimal("10.0")
        ):
            raise ValueError(
                f"backoff_factor must be between 1.1 and 10.0, got {self.backoff_factor}"
            )

        if self.base_delay > self.max_delay:
            raise ValueError(
                f"base_delay ({self.base_delay}) cannot be greater than "
                f"max_delay ({self.max_delay})"
            )

        if not isinstance(self.retriable_errors, tuple):
            raise ValueError(f"retriable_errors must be a tuple, got {type(self.retriable_errors)}")

        # Validate all errors are exception classes
        for error_type in self.retriable_errors:
            if not (isinstance(error_type, type) and issubclass(error_type, Exception)):
                raise ValueError(
                    f"All items in retriable_errors must be Exception subclasses, got {error_type}"
                )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior with validation.

    Attributes:
        failure_threshold: Number of failures before opening circuit (1-50)
        recovery_timeout: Time to wait before attempting recovery in seconds (1.0-3600.0)
        half_open_max_calls: Maximum calls allowed in half-open state (1-20)
        expected_exception: Exception type that triggers circuit breaker
    """

    failure_threshold: int = 5
    recovery_timeout: Decimal = Decimal("60.0")
    half_open_max_calls: int = 3
    expected_exception: type = Exception

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate circuit breaker configuration."""
        if not isinstance(self.failure_threshold, int) or not (1 <= self.failure_threshold <= 50):
            raise ValueError(
                f"failure_threshold must be between 1 and 50, got {self.failure_threshold}"
            )

        if not isinstance(self.recovery_timeout, int | float | Decimal) or not (
            Decimal("1.0") <= Decimal(str(self.recovery_timeout)) <= Decimal("3600.0")
        ):
            raise ValueError(
                f"recovery_timeout must be between 1.0 and 3600.0 seconds, "
                f"got {self.recovery_timeout}"
            )

        if not isinstance(self.half_open_max_calls, int) or not (
            1 <= self.half_open_max_calls <= 20
        ):
            raise ValueError(
                f"half_open_max_calls must be between 1 and 20, got {self.half_open_max_calls}"
            )

        if not (
            isinstance(self.expected_exception, type)
            and issubclass(self.expected_exception, Exception)
        ):
            raise ValueError(
                f"expected_exception must be an Exception subclass, got {self.expected_exception}"
            )


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior with validation.

    Attributes:
        strategy: Fallback strategy to use when handling errors
        default_value: Default value to return for RETURN_DEFAULT strategy
        fallback_function: Function to call for RETRY_ALTERNATIVE strategy
        use_cache: Whether to use cached values as fallback
        cache_key_func: Function to generate cache keys for arguments
    """

    strategy: FallbackStrategy = FallbackStrategy.RETURN_NONE
    default_value: Any = None
    fallback_function: Callable[..., Any] | None = None
    use_cache: bool = False
    cache_key_func: Callable[..., str] | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate fallback configuration."""
        if not isinstance(self.strategy, FallbackStrategy):
            raise ValueError(f"strategy must be a FallbackStrategy enum, got {type(self.strategy)}")

        if self.strategy == FallbackStrategy.RETRY_ALTERNATIVE and self.fallback_function is None:
            raise ValueError(
                "fallback_function must be provided when using RETRY_ALTERNATIVE strategy"
            )

        if self.fallback_function is not None and not callable(self.fallback_function):
            raise ValueError(
                f"fallback_function must be callable, got {type(self.fallback_function)}"
            )

        if self.cache_key_func is not None and not callable(self.cache_key_func):
            raise ValueError(f"cache_key_func must be callable, got {type(self.cache_key_func)}")


class ErrorMetrics:
    """Production-grade error metrics tracking with automatic cleanup and health monitoring.

    Attributes:
        _metrics: Dictionary storing error metrics by function name
        _lock: Thread-safe access lock
        _max_entries: Maximum number of function metrics to retain
        _retention_hours: Hours to retain metrics data
        _last_cleanup: Timestamp of last cleanup operation
        _health_status: Current health status of the metrics system
    """

    def __init__(self, max_entries: int = 1000, retention_hours: int = 24) -> None:
        self._metrics: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._max_entries = max_entries
        self._retention_hours = retention_hours
        self._last_cleanup = datetime.now(timezone.utc)
        self._health_status: dict[str, Any] = {
            "status": "healthy",
            "last_health_check": datetime.now(timezone.utc),
            "total_memory_usage_mb": 0.0,
            "cleanup_failures": 0,
            "last_cleanup_duration_ms": 0.0,
        }

    async def record_error(
        self,
        function_name: str,
        error_category: ErrorCategory,
        error_type: str,
        resolved: bool = False,
        duration_ms: float | None = None,
        severity: str = "medium",
    ) -> None:
        """Record error occurrence with enhanced metrics.

        Args:
            function_name: Name of the function where error occurred
            error_category: Category of the error for classification
            error_type: Type/class name of the error
            resolved: Whether the error was successfully resolved
            duration_ms: Duration of the operation in milliseconds
            severity: Severity level of the error
        """
        # Check if cleanup is needed
        if self._should_cleanup():
            await self._cleanup_old_metrics_async()

        with self._lock:
            if function_name not in self._metrics:
                self._metrics[function_name] = {
                    "total_errors": 0,
                    "resolved_errors": 0,
                    "error_types": {},
                    "error_categories": {},
                    "severity_counts": {},
                    "avg_resolution_time_ms": 0.0,
                    "last_error": None,
                    "first_error": datetime.now(timezone.utc),
                    "error_rate_per_hour": 0.0,
                    "success_rate": 0.0,
                }

            metrics = self._metrics[function_name]
            metrics["total_errors"] += 1
            metrics["last_error"] = datetime.now(timezone.utc)

            if resolved:
                metrics["resolved_errors"] += 1
                if duration_ms is not None:
                    # Update average resolution time
                    current_avg = metrics["avg_resolution_time_ms"]
                    resolved_count = metrics["resolved_errors"]
                    metrics["avg_resolution_time_ms"] = (
                        current_avg * (resolved_count - 1) + duration_ms
                    ) / resolved_count

            # Track error types
            if error_type not in metrics["error_types"]:
                metrics["error_types"][error_type] = 0
            metrics["error_types"][error_type] += 1

            # Track error categories
            category_name = error_category.value
            if category_name not in metrics["error_categories"]:
                metrics["error_categories"][category_name] = 0
            metrics["error_categories"][category_name] += 1

            # Track severity levels
            if severity not in metrics["severity_counts"]:
                metrics["severity_counts"][severity] = 0
            metrics["severity_counts"][severity] += 1

            # Calculate success rate
            total_errors = metrics["total_errors"]
            resolved_errors = metrics["resolved_errors"]
            metrics["success_rate"] = (
                (resolved_errors / total_errors) * 100 if total_errors > 0 else 0.0
            )

            # Calculate error rate per hour
            first_error = metrics["first_error"]
            now = datetime.now(timezone.utc)
            hours_elapsed = max(1, (now - first_error).total_seconds() / 3600)
            metrics["error_rate_per_hour"] = total_errors / hours_elapsed

    def get_metrics(self, function_name: str | None = None) -> dict[str, Any]:
        """Get error metrics with optional function filtering.

        Args:
            function_name: Optional function name to filter metrics

        Returns:
            Dictionary containing error metrics
        """
        with self._lock:
            if function_name:
                return self._metrics.get(function_name, {}).copy()
            return {k: v.copy() for k, v in self._metrics.items()}

    def get_health_status(self) -> dict[str, Any]:
        """Get current health status of the metrics system.

        Returns:
            Dictionary containing health metrics and status
        """
        with self._lock:
            # Update memory usage estimate
            self._health_status["total_memory_usage_mb"] = self._estimate_memory_usage()
            self._health_status["total_functions_tracked"] = len(self._metrics)
            self._health_status["last_health_check"] = datetime.now(timezone.utc)

            # Determine overall health status
            memory_usage = self._health_status["total_memory_usage_mb"]
            cleanup_failures = self._health_status["cleanup_failures"]

            if memory_usage > 500 or cleanup_failures > 5:
                self._health_status["status"] = "unhealthy"
            elif memory_usage > 200 or cleanup_failures > 2:
                self._health_status["status"] = "degraded"
            else:
                self._health_status["status"] = "healthy"

            return self._health_status.copy()

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of metrics data in MB.

        Returns:
            Estimated memory usage in megabytes
        """
        try:
            # Simple estimation based on string representation size
            total_size = sys.getsizeof(self._metrics)
            for func_metrics in self._metrics.values():
                total_size += sys.getsizeof(func_metrics)
                for key, value in func_metrics.items():
                    total_size += sys.getsizeof(key) + sys.getsizeof(value)
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.error(f"Failed to calculate metrics memory size: {e}")
            return 0.0

    def reset_metrics(self, function_name: str | None = None) -> None:
        """Reset error metrics with logging.

        Args:
            function_name: Optional function name to reset, if None resets all
        """
        with self._lock:
            if function_name:
                if function_name in self._metrics:
                    logger.info(f"Resetting metrics for function: {function_name}")
                    self._metrics.pop(function_name, None)
            else:
                logger.info(f"Resetting all metrics for {len(self._metrics)} functions")
                self._metrics.clear()
                self._health_status["cleanup_failures"] = 0

    def _cleanup_old_metrics(self) -> None:
        """Remove old metrics to prevent memory growth (synchronous version)."""
        with self._lock:
            self._perform_cleanup()

    async def _cleanup_old_metrics_async(self) -> None:
        """Remove old metrics to prevent memory growth (async version with error handling)."""
        start_time = time.time()
        try:
            with self._lock:
                self._perform_cleanup()

            # Update health status
            cleanup_duration_ms = (time.time() - start_time) * 1000
            self._health_status["last_cleanup_duration_ms"] = cleanup_duration_ms

        except Exception as e:
            self._health_status["cleanup_failures"] += 1
            logger.error(f"Metrics cleanup failed: {e}")

    def _perform_cleanup(self) -> None:
        """Perform the actual cleanup operations."""
        now = datetime.now(timezone.utc)
        initial_count = len(self._metrics)

        # Remove metrics older than retention period
        retention_cutoff = now - timedelta(hours=self._retention_hours)
        to_remove = []

        for func_name, metrics in self._metrics.items():
            last_error = metrics.get("last_error")
            if last_error and last_error < retention_cutoff:
                to_remove.append(func_name)

        for func_name in to_remove:
            del self._metrics[func_name]

        # If still too many entries, remove oldest
        if len(self._metrics) > self._max_entries:
            # Sort by last_error time and keep only the most recent
            sorted_funcs = sorted(
                self._metrics.items(),
                key=lambda x: x[1].get("last_error", datetime.min.replace(tzinfo=timezone.utc)),
                reverse=True,
            )
            self._metrics = dict(sorted_funcs[: self._max_entries])

        self._last_cleanup = now
        removed_count = initial_count - len(self._metrics)

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old metrics entries")

    def _should_cleanup(self) -> bool:
        """Check if cleanup is needed based on time and size thresholds.

        Returns:
            True if cleanup should be performed
        """
        now = datetime.now(timezone.utc)
        # Run cleanup every hour or when reaching 80% of max entries
        time_threshold = now - self._last_cleanup > timedelta(hours=1)
        size_threshold = len(self._metrics) > (self._max_entries * 0.8)

        return time_threshold or size_threshold


# Global error metrics instance
error_metrics = ErrorMetrics()


# Global handler registry for proper lifecycle management
class HandlerRegistry:
    """Thread-safe registry for tracking active UniversalErrorHandler instances."""

    def __init__(self) -> None:
        self._handlers: weakref.WeakSet = weakref.WeakSet()
        self._lock = threading.RLock()

    def register(self, handler: "UniversalErrorHandler") -> None:
        """Register a handler instance."""
        with self._lock:
            self._handlers.add(handler)

    def unregister(self, handler: "UniversalErrorHandler") -> None:
        """Unregister a handler instance."""
        with self._lock:
            self._handlers.discard(handler)

    async def shutdown_all(self) -> None:
        """Shutdown all registered handlers."""
        with self._lock:
            handlers = list(self._handlers)

        if handlers:
            logger.info(f"Shutting down {len(handlers)} error handlers")
            shutdown_tasks = [handler.shutdown() for handler in handlers]
            if shutdown_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*shutdown_tasks, return_exceptions=True), timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Handler shutdown timed out after 30 seconds")

    def get_active_count(self) -> int:
        """Get count of active handlers."""
        with self._lock:
            return len(self._handlers)


# Global handler registry instance
_handler_registry = HandlerRegistry()


def categorize_error(error: Exception) -> ErrorCategory:
    """Categorize error for intelligent handling."""
    error_type = type(error)

    # Check if error_type is actually a class before using issubclass
    if not isinstance(error_type, type):
        return ErrorCategory.SYSTEM

    if issubclass(error_type, NetworkError | ConnectionError | TimeoutError):
        return ErrorCategory.NETWORK
    elif issubclass(error_type, DatabaseConnectionError | DataError):
        return ErrorCategory.DATA_QUALITY
    elif issubclass(error_type, ValidationError | ValueError | TypeError):
        return ErrorCategory.VALIDATION
    elif issubclass(error_type, ExchangeError):
        return ErrorCategory.NETWORK  # Exchange errors are network-related
    elif issubclass(error_type, ConfigurationError):
        return ErrorCategory.CONFIGURATION
    elif issubclass(error_type, ServiceError | TradingBotError):
        return ErrorCategory.BUSINESS_LOGIC
    elif issubclass(error_type, OSError | MemoryError | SystemError):
        return ErrorCategory.SYSTEM
    else:
        return ErrorCategory.SYSTEM


def calculate_retry_delay(attempt: int, config: RetryConfig) -> Decimal:
    """Calculate retry delay with backoff and jitter."""
    if config.exponential:
        delay = config.base_delay * (config.backoff_factor ** (attempt - 1))
    else:
        delay = config.base_delay * attempt

    # Apply maximum delay limit
    delay = min(delay, config.max_delay)

    # Apply jitter to avoid thundering herd
    if config.jitter:
        jitter_factor = Decimal(str(random.uniform(0.1, 0.9)))
        jitter = jitter_factor * delay
        delay = delay * Decimal("0.5") + jitter

    return delay


class UniversalErrorHandler:
    """Production-grade universal error handler with intelligent routing and monitoring.

    This handler provides:
    - Context manager support for resource cleanup
    - Health monitoring and diagnostics
    - Production-grade metrics collection
    - Structured logging with correlation IDs
    - Graceful degradation patterns
    - Thread-safe operations

    Attributes:
        retry_config: Configuration for retry behavior
        circuit_breaker_config: Configuration for circuit breaker
        fallback_config: Configuration for fallback strategies
        enable_metrics: Whether to collect and track metrics
        enable_logging: Whether to enable structured logging
    """

    # Class-level constants
    HEALTH_CHECK_INTERVAL_SECONDS: ClassVar[Decimal] = Decimal(
        str(DEFAULT_HEALTH_CHECK_INTERVAL_SECONDS)
    )
    MAX_CIRCUIT_BREAKER_FAILURES: ClassVar[int] = 100
    MEMORY_USAGE_THRESHOLD_MB: ClassVar[Decimal] = Decimal(str(DEFAULT_CACHE_MEMORY_LIMIT_MB))

    def __init__(
        self,
        retry_config: RetryConfig | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        fallback_config: FallbackConfig | None = None,
        enable_metrics: bool = True,
        enable_logging: bool = True,
        handler_id: str | None = None,
    ) -> None:
        """Initialize the UniversalErrorHandler with production-grade configurations.

        Args:
            retry_config: Retry behavior configuration
            circuit_breaker_config: Circuit breaker configuration
            fallback_config: Fallback strategy configuration
            enable_metrics: Enable metrics collection
            enable_logging: Enable structured logging
            handler_id: Unique identifier for this handler instance

        Raises:
            ValueError: If any configuration is invalid
        """
        # Validate and store configurations
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self.fallback_config = fallback_config or FallbackConfig()
        self.enable_metrics = enable_metrics
        self.enable_logging = enable_logging

        # Handler identification
        self.handler_id = handler_id or f"handler_{id(self)}"
        self._creation_time = datetime.now(timezone.utc)

        # Circuit breaker state with enhanced tracking
        self._circuit_breaker_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: datetime | None = None
        self._last_success_time: datetime | None = None
        self._half_open_calls = 0
        self._circuit_breaker_lock = threading.RLock()

        # Health monitoring
        self._health_status: dict[str, Any] = {
            "status": "healthy",
            "last_health_check": datetime.now(timezone.utc),
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "avg_operation_time_ms": 0.0,
            "cache_hit_rate": 0.0,
            "memory_usage_mb": 0.0,
        }

        # Performance tracking
        self._operation_times: list[float] = []
        self._max_operation_times = DEFAULT_MAX_OPERATION_TIMES
        self._performance_lock = threading.RLock()

        # LRU cache for fallback values with memory limits
        self._fallback_cache = LRUCache(
            max_size=DEFAULT_CACHE_SIZE // 2, ttl_seconds=DEFAULT_FALLBACK_CACHE_TTL_SECONDS
        )
        self._cache_hits = 0
        self._cache_requests = 0

        # Cleanup task for periodic maintenance
        self._cleanup_task: asyncio.Task | None = None
        self._health_check_task: asyncio.Task | None = None
        self._shutdown_flag = False
        self._cleanup_task_lock = threading.RLock()
        self._context_manager_active = False

        # Enhanced logging setup
        self.logger = get_logger(f"{__name__}.{self.handler_id}")
        if self.enable_logging:
            self.logger.info(
                "UniversalErrorHandler initialized",
                handler_id=self.handler_id,
                retry_config={
                    "max_attempts": self.retry_config.max_attempts,
                    "base_delay": self.retry_config.base_delay,
                },
                circuit_breaker_config={
                    "failure_threshold": self.circuit_breaker_config.failure_threshold,
                    "recovery_timeout": self.circuit_breaker_config.recovery_timeout,
                },
                fallback_strategy=self.fallback_config.strategy.value,
            )

        # Register with global handler registry
        _handler_registry.register(self)

        self._start_cleanup_task()

    def _start_cleanup_task(self) -> None:
        """Start periodic cleanup task."""
        with self._cleanup_task_lock:
            try:
                # Check if there's an event loop running
                asyncio.get_running_loop()
                if self._cleanup_task is None or self._cleanup_task.done():
                    self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
                    # Add done callback to handle any exceptions
                    self._cleanup_task.add_done_callback(self._cleanup_task_done_callback)
            except RuntimeError:
                # No event loop running, this is expected in sync contexts
                # Cleanup will be started when an async context is available
                pass

    def _cleanup_task_done_callback(self, task: asyncio.Task) -> None:
        """Handle cleanup task completion."""
        try:
            exception = task.exception()
            if exception and not isinstance(exception, asyncio.CancelledError):
                self.logger.error(f"Cleanup task failed: {exception}")
        except asyncio.CancelledError:
            # Task was cancelled, this is expected during shutdown
            pass
        # Reset task reference so it can be restarted if needed
        with self._cleanup_task_lock:
            if self._cleanup_task is task:
                self._cleanup_task = None

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of caches and state."""
        try:
            while not self._shutdown_flag:
                try:
                    # Clean up fallback cache
                    if self._fallback_cache.size() > 0:
                        # Force cleanup of expired entries
                        self._fallback_cache.cleanup_expired()

                        # Log memory usage
                        memory_usage = self._fallback_cache.memory_usage_mb()
                        if memory_usage > HIGH_MEMORY_WARNING_THRESHOLD_MB:
                            logger.warning(
                                "High fallback cache memory usage",
                                memory_mb=memory_usage,
                                entries=self._fallback_cache.size(),
                                threshold_mb=HIGH_MEMORY_WARNING_THRESHOLD_MB,
                            )

                    # Check for shutdown every second for 5 minutes
                    for _ in range(PERIODIC_CLEANUP_ITERATIONS):
                        if self._shutdown_flag:
                            return  # Exit cleanly on shutdown
                        await asyncio.sleep(PERIODIC_CLEANUP_SLEEP_SECONDS)

                except asyncio.CancelledError:
                    logger.debug("Cleanup task cancelled during operation")
                    return
                except Exception as e:
                    logger.error("Error in periodic cleanup", error=str(e))
                    if not self._shutdown_flag:
                        try:
                            await asyncio.sleep(MEMORY_CLEANUP_SLEEP_SECONDS)
                        except asyncio.CancelledError:
                            logger.debug("Cleanup task cancelled during error recovery")
                            return
        except asyncio.CancelledError:
            logger.debug("Cleanup task cancelled")
            # Don't re-raise CancelledError, just exit cleanly
        finally:
            logger.debug("Periodic cleanup task finished")

    async def shutdown(self) -> None:
        """Shutdown the error handler and cleanup resources."""
        try:
            logger.debug("Shutting down UniversalErrorHandler")

            # Signal shutdown to cleanup task
            self._shutdown_flag = True

            # Cancel and wait for cleanup task
            with self._cleanup_task_lock:
                if self._cleanup_task and not self._cleanup_task.done():
                    self._cleanup_task.cancel()
                    cleanup_task = self._cleanup_task
                else:
                    cleanup_task = None

            # Wait for cleanup task to complete with timeout
            if cleanup_task:
                try:
                    await asyncio.wait_for(cleanup_task, timeout=5.0)
                except asyncio.CancelledError:
                    logger.debug("Cleanup task cancelled successfully")
                except asyncio.TimeoutError:
                    logger.warning("Cleanup task shutdown timed out after 5 seconds")
                except Exception as e:
                    logger.warning(f"Error during cleanup task shutdown: {e}")

            # Clear cache resources
            self._fallback_cache.clear()

            # Unregister from global registry
            _handler_registry.unregister(self)

            logger.debug("UniversalErrorHandler shutdown complete")

        except Exception as e:
            logger.error(f"Error during handler shutdown: {e}")
            # Ensure we still unregister even if other cleanup fails
            _handler_registry.unregister(self)
            raise

    # Context Manager Support
    def __enter__(self) -> "UniversalErrorHandler":
        """Enter context manager - start monitoring and health checks.

        Returns:
            Self for use in with statements
        """
        self._context_manager_active = True
        self._start_health_monitoring()
        if self.enable_logging:
            self.logger.info("ErrorHandler context entered", handler_id=self.handler_id)
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any | None
    ) -> None:
        """Exit context manager - cleanup resources synchronously.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self._context_manager_active = False
        try:
            self.shutdown_sync()
            if self.enable_logging:
                self.logger.info("ErrorHandler context exited cleanly", handler_id=self.handler_id)
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Error during context exit: {e}", handler_id=self.handler_id)

    @asynccontextmanager
    async def async_context(self):
        """Async context manager for proper resource cleanup.

        Usage:
            async with handler.async_context():
                # Use handler for error handling
                pass
        """
        self._context_manager_active = True
        self._start_health_monitoring()
        try:
            if self.enable_logging:
                self.logger.info("ErrorHandler async context entered", handler_id=self.handler_id)
            yield self
        finally:
            self._context_manager_active = False
            await self.shutdown()
            if self.enable_logging:
                self.logger.info("ErrorHandler async context exited", handler_id=self.handler_id)

    # Health Monitoring Methods
    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status of the error handler.

        Returns:
            Dictionary containing detailed health metrics
        """
        with self._circuit_breaker_lock, self._performance_lock:
            # Update health metrics
            self._update_health_metrics()

            # Add circuit breaker status
            health_status = self._health_status.copy()
            health_status.update(
                {
                    "handler_id": self.handler_id,
                    "creation_time": self._creation_time.isoformat(),
                    "uptime_seconds": (
                        datetime.now(timezone.utc) - self._creation_time
                    ).total_seconds(),
                    "circuit_breaker_state": self._circuit_breaker_state,
                    "circuit_breaker_failure_count": self._failure_count,
                    "circuit_breaker_success_count": self._success_count,
                    "cache_size": self._fallback_cache.size(),
                    "cache_memory_usage_mb": self._fallback_cache.memory_usage_mb(),
                    "is_context_manager_active": self._context_manager_active,
                    "is_shutdown": self._shutdown_flag,
                }
            )

            return health_status

    def _update_health_metrics(self) -> None:
        """Update health metrics with current system state."""
        total_ops = self._health_status["total_operations"]
        successful_ops = self._health_status["successful_operations"]
        self._health_status["failed_operations"]

        # Calculate success rate
        success_rate = (successful_ops / total_ops * 100) if total_ops > 0 else 100.0

        # Calculate average operation time
        avg_time = (
            sum(self._operation_times) / len(self._operation_times)
            if self._operation_times
            else 0.0
        )

        # Calculate cache hit rate
        cache_hit_rate = (
            (self._cache_hits / self._cache_requests * 100) if self._cache_requests > 0 else 0.0
        )

        # Estimate memory usage
        memory_usage = self._fallback_cache.memory_usage_mb() + (
            len(self._operation_times) * 8 / 1024 / 1024
        )

        # Update health status
        self._health_status.update(
            {
                "last_health_check": datetime.now(timezone.utc),
                "success_rate": success_rate,
                "avg_operation_time_ms": avg_time,
                "cache_hit_rate": cache_hit_rate,
                "memory_usage_mb": memory_usage,
            }
        )

        # Determine overall health
        if (
            success_rate < 50
            or memory_usage > self.MEMORY_USAGE_THRESHOLD_MB
            or self._failure_count > self.MAX_CIRCUIT_BREAKER_FAILURES
        ):
            self._health_status["status"] = "unhealthy"
        elif success_rate < 80 or memory_usage > self.MEMORY_USAGE_THRESHOLD_MB * Decimal("0.7"):
            self._health_status["status"] = "degraded"
        else:
            self._health_status["status"] = "healthy"

    def is_healthy(self) -> bool:
        """Quick health check for the error handler.

        Returns:
            True if handler is in healthy state
        """
        return self.get_health_status()["status"] == "healthy"

    def _start_health_monitoring(self) -> None:
        """Start periodic health monitoring task."""
        with self._cleanup_task_lock:
            try:
                asyncio.get_running_loop()
                if self._health_check_task is None or self._health_check_task.done():
                    self._health_check_task = asyncio.create_task(self._periodic_health_check())
            except RuntimeError:
                # No event loop running
                pass

    async def _periodic_health_check(self) -> None:
        """Periodic health check and maintenance."""
        try:
            while not self._shutdown_flag:
                try:
                    # Update health metrics
                    self._update_health_metrics()

                    # Log health status periodically
                    if self.enable_logging:
                        health_status = self.get_health_status()
                        if health_status["status"] != "healthy":
                            self.logger.warning(
                                "Handler health check",
                                handler_id=self.handler_id,
                                status=health_status["status"],
                                success_rate=health_status["success_rate"],
                                memory_usage_mb=health_status["memory_usage_mb"],
                            )

                    # Wait for next check
                    await asyncio.sleep(float(self.HEALTH_CHECK_INTERVAL_SECONDS))

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if self.enable_logging:
                        self.logger.error(f"Health check error: {e}", handler_id=self.handler_id)
                    await asyncio.sleep(30)  # Back off on error

        except asyncio.CancelledError:
            pass
        finally:
            if self.enable_logging:
                self.logger.debug("Health monitoring stopped", handler_id=self.handler_id)

    def shutdown_sync(self) -> None:
        """Synchronous shutdown for compatibility - use shutdown() when possible."""
        with self._cleanup_task_lock:
            self._shutdown_flag = True

            # Cancel tasks
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
            if self._health_check_task and not self._health_check_task.done():
                self._health_check_task.cancel()

            # Clear cache resources
            self._fallback_cache.clear()

            # Unregister from global registry
            _handler_registry.unregister(self)

            if self.enable_logging:
                self.logger.info("Handler shutdown completed", handler_id=self.handler_id)

    def __del__(self) -> None:
        """Cleanup when the object is garbage collected."""
        try:
            # Use synchronous cleanup in destructor
            if not self._shutdown_flag:
                self.shutdown_sync()
        except Exception:
            # Ignore all errors in destructor to avoid issues during shutdown
            # Don't log during shutdown as logging infrastructure may be unavailable
            pass

    def _should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if error should be retried."""
        if attempt >= self.retry_config.max_attempts:
            return False

        return isinstance(error, self.retry_config.retriable_errors)

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows operation."""
        with self._circuit_breaker_lock:
            now = datetime.now(timezone.utc)

            if self._circuit_breaker_state == "OPEN":
                if (
                    self._last_failure_time
                    and (now - self._last_failure_time).total_seconds()
                    >= self.circuit_breaker_config.recovery_timeout
                ):
                    self._circuit_breaker_state = "HALF_OPEN"
                    self._half_open_calls = 0
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    return False

            elif self._circuit_breaker_state == "HALF_OPEN":
                if self._half_open_calls >= self.circuit_breaker_config.half_open_max_calls:
                    return False

            return True

    def _record_success(self, operation_time_ms: float | None = None) -> None:
        """Record successful operation with enhanced metrics.

        Args:
            operation_time_ms: Duration of the successful operation in milliseconds
        """
        with self._circuit_breaker_lock, self._performance_lock:
            # Update circuit breaker state
            if self._circuit_breaker_state == "HALF_OPEN":
                self._circuit_breaker_state = "CLOSED"
                self._failure_count = 0
                if self.enable_logging:
                    self.logger.info(
                        "Circuit breaker reset to CLOSED",
                        handler_id=self.handler_id,
                        success_count=self._success_count + 1,
                    )
            elif self._circuit_breaker_state == "CLOSED":
                self._failure_count = 0

            # Update success metrics
            self._success_count += 1
            self._last_success_time = datetime.now(timezone.utc)
            self._health_status["successful_operations"] += 1
            self._health_status["total_operations"] += 1

            # Record operation time
            if operation_time_ms is not None:
                self._operation_times.append(operation_time_ms)
                if len(self._operation_times) > self._max_operation_times:
                    self._operation_times.pop(0)  # Remove oldest time

    def _record_failure(
        self, operation_time_ms: float | None = None, error_severity: str = "medium"
    ) -> None:
        """Record failed operation with enhanced tracking.

        Args:
            operation_time_ms: Duration of the failed operation in milliseconds
            error_severity: Severity level of the error
        """
        with self._circuit_breaker_lock, self._performance_lock:
            # Update circuit breaker state
            self._failure_count += 1
            self._last_failure_time = datetime.now(timezone.utc)

            # Update failure metrics
            self._health_status["failed_operations"] += 1
            self._health_status["total_operations"] += 1

            # Record operation time even for failures
            if operation_time_ms is not None:
                self._operation_times.append(operation_time_ms)
                if len(self._operation_times) > self._max_operation_times:
                    self._operation_times.pop(0)

            if self._circuit_breaker_state == "HALF_OPEN":
                self._circuit_breaker_state = "OPEN"
                if self.enable_logging:
                    self.logger.warning(
                        "Circuit breaker tripped to OPEN from HALF_OPEN",
                        handler_id=self.handler_id,
                        failure_count=self._failure_count,
                        error_severity=error_severity,
                    )
            elif (
                self._circuit_breaker_state == "CLOSED"
                and self._failure_count >= self.circuit_breaker_config.failure_threshold
            ):
                self._circuit_breaker_state = "OPEN"
                if self.enable_logging:
                    self.logger.warning(
                        "Circuit breaker tripped to OPEN",
                        handler_id=self.handler_id,
                        failure_count=self._failure_count,
                        threshold=self.circuit_breaker_config.failure_threshold,
                        error_severity=error_severity,
                    )

    async def _apply_fallback(self, context: ErrorContext, cache_key: str | None = None) -> Any:
        """Apply fallback strategy with comprehensive error handling and logging.

        Args:
            context: Error context containing details about the failure
            cache_key: Optional cache key for retrieving cached values

        Returns:
            Fallback value based on the configured strategy

        Raises:
            ServiceError: When RAISE_DEGRADED strategy is used
        """
        strategy = self.fallback_config.strategy
        fallback_start_time = time.time()

        try:
            if strategy == FallbackStrategy.RETURN_NONE:
                if self.enable_logging:
                    self.logger.debug(
                        "Applying RETURN_NONE fallback",
                        handler_id=self.handler_id,
                        function=context.function_name,
                    )
                return None

            elif strategy == FallbackStrategy.RETURN_EMPTY:
                # Enhanced empty collection detection
                empty_value = self._determine_empty_value(context)
                if self.enable_logging:
                    self.logger.debug(
                        "Applying RETURN_EMPTY fallback",
                        handler_id=self.handler_id,
                        function=context.function_name,
                        empty_type=type(empty_value).__name__,
                    )
                return empty_value

            elif strategy == FallbackStrategy.RETURN_DEFAULT:
                if self.enable_logging:
                    self.logger.debug(
                        "Applying RETURN_DEFAULT fallback",
                        handler_id=self.handler_id,
                        function=context.function_name,
                        default_type=type(self.fallback_config.default_value).__name__,
                    )
                return self.fallback_config.default_value

            elif strategy == FallbackStrategy.USE_CACHE and cache_key:
                self._cache_requests += 1
                cached_value = self._fallback_cache.get(cache_key)

                if cached_value is not None:
                    self._cache_hits += 1
                    if self.enable_logging:
                        self.logger.debug(
                            "Cache hit for fallback",
                            handler_id=self.handler_id,
                            function=context.function_name,
                            cache_key=cache_key[:50] + "..." if len(cache_key) > 50 else cache_key,
                        )
                else:
                    if self.enable_logging:
                        self.logger.debug(
                            "Cache miss for fallback",
                            handler_id=self.handler_id,
                            function=context.function_name,
                        )

                return cached_value

            elif strategy == FallbackStrategy.RETRY_ALTERNATIVE:
                if self.fallback_config.fallback_function:
                    try:
                        # Check if the fallback function is async
                        if asyncio.iscoroutinefunction(self.fallback_config.fallback_function):
                            result = await self.fallback_config.fallback_function(
                                *context.args, **context.kwargs
                            )
                        else:
                            # Run sync function in executor to avoid blocking
                            loop = asyncio.get_running_loop()
                            result = await loop.run_in_executor(
                                None,
                                self.fallback_config.fallback_function,
                                *context.args,
                                **context.kwargs,
                            )

                        if self.enable_logging:
                            self.logger.info(
                                "Fallback function succeeded",
                                handler_id=self.handler_id,
                                function=context.function_name,
                                fallback_duration_ms=(time.time() - fallback_start_time) * 1000,
                            )
                        return result

                    except Exception as fallback_error:
                        if self.enable_logging:
                            self.logger.error(
                                "Fallback function failed",
                                handler_id=self.handler_id,
                                function=context.function_name,
                                error=str(fallback_error),
                                error_type=type(fallback_error).__name__,
                            )
                        # Return None instead of raising to provide graceful degradation
                        return None
                else:
                    if self.enable_logging:
                        self.logger.warning(
                            "RETRY_ALTERNATIVE strategy configured but no fallback function "
                            "provided",
                            handler_id=self.handler_id,
                            function=context.function_name,
                        )
                    return None

            elif strategy == FallbackStrategy.RAISE_DEGRADED:
                if self.enable_logging:
                    self.logger.warning(
                        "Raising degraded service error",
                        handler_id=self.handler_id,
                        function=context.function_name,
                        original_error=str(context.error),
                    )
                raise ServiceError(
                    f"Service degraded due to {context.category.value} error in "
                    f"{context.function_name}",
                    details={
                        "original_error": str(context.error),
                        "error_category": context.category.value,
                        "handler_id": self.handler_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

            # Default fallback
            if self.enable_logging:
                self.logger.warning(
                    "Unknown fallback strategy, returning None",
                    handler_id=self.handler_id,
                    strategy=strategy.value if hasattr(strategy, "value") else str(strategy),
                )
            return None

        except Exception as e:
            # Ensure fallback application never fails catastrophically
            if self.enable_logging:
                self.logger.error(
                    "Critical error in fallback application",
                    handler_id=self.handler_id,
                    function=context.function_name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
            # Return None as last resort
            return None

    def _determine_empty_value(self, context: ErrorContext) -> Any:
        """Determine appropriate empty value based on function context.

        Args:
            context: Error context to analyze

        Returns:
            Appropriate empty value (list, dict, set, tuple, or None)
        """
        function_name = context.function_name or ""

        # Analyze function name for collection type hints using shared utilities
        type_map = {
            "list": [],
            "dict": {},
            "set": set(),
            "tuple": tuple(),
            "str": "",
            "int": 0,
            "bool": False,
        }

        detected_type = categorize_by_keywords(function_name, get_fallback_type_keywords())
        return type_map.get(detected_type) if detected_type is not None else None

    def _log_error(
        self, context: ErrorContext, resolved: bool = False, operation_time_ms: float | None = None
    ) -> None:
        """Log error with comprehensive structured information and consistent propagation patterns.

        Args:
            context: Error context containing error details
            resolved: Whether the error was successfully resolved
            operation_time_ms: Duration of the operation in milliseconds
        """
        if not self.enable_logging:
            return

        # Note: sensitivity level could be used for future sanitization features

        # Prepare log data with enhanced information and error propagation metadata
        log_data = {
            "handler_id": self.handler_id,
            "function": context.function_name,
            "error_type": type(context.error).__name__,
            "error_message": str(context.error),
            "category": context.category.value,
            "attempt": context.attempt_number,
            "correlation_id": context.correlation_id,
            "resolved": resolved,
            "circuit_breaker_state": self._circuit_breaker_state,
            "operation_time_ms": operation_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": self._get_error_severity(context.error),
            # Error propagation metadata for consistent flow between modules
            "error_propagation": {
                "source_module": "error_handling",
                "target_modules": ["core", "monitoring", "database"],
                "propagation_method": "structured_logging",
                "data_format": "error_context_v1",
                "processing_stage": "error_logging",
                "boundary_crossed": True
            }
        }

        # Add metadata if available, with sanitization
        if context.metadata:
            # Only include non-sensitive metadata in logs
            safe_metadata = {}
            for key, value in context.metadata.items():
                if not is_sensitive_key(key):
                    safe_metadata[key] = value
            if safe_metadata:
                log_data["metadata"] = str(safe_metadata)

        # Add performance context if available
        with self._performance_lock:
            if self._operation_times:
                log_data["avg_operation_time_ms"] = sum(self._operation_times) / len(
                    self._operation_times
                )

        # Log with appropriate level based on resolution status
        if resolved:
            self.logger.info("Error resolved", **log_data)
        else:
            # Use different log levels based on error severity
            error_severity = log_data["severity"]
            if error_severity == "critical":
                self.logger.critical("Critical error occurred", **log_data)
            elif error_severity == "high":
                self.logger.error("High severity error occurred", **log_data)
            else:
                self.logger.warning("Error occurred", **log_data)

    async def handle_error(
        self,
        error: Exception,
        function_name: str,
        args: tuple,
        kwargs: dict,
        attempt: int = 1,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        operation_start_time: float | None = None,
    ) -> tuple[bool, Any]:
        """
        Handle error with intelligent routing and comprehensive recovery.

        Args:
            error: Exception that occurred
            function_name: Name of the function where error occurred
            args: Function arguments
            kwargs: Function keyword arguments
            attempt: Current attempt number
            correlation_id: Optional correlation ID for tracking
            metadata: Additional metadata for context
            operation_start_time: Start time of the operation for duration calculation

        Returns:
            Tuple of (was_handled: bool, result_value: Any)

        Raises:
            ServiceError: When fallback strategy is RAISE_DEGRADED
        """
        handle_start_time = time.time()
        operation_time_ms = None

        if operation_start_time is not None:
            operation_time_ms = (handle_start_time - operation_start_time) * 1000

        try:
            # Create error context using the consolidated class
            context = ErrorContext.from_decorator_context(
                error=error,
                function_name=function_name,
                args=args,
                kwargs=kwargs,
                attempt_number=attempt,
                correlation_id=correlation_id,
                metadata=metadata or {},
            )
            context.category = categorize_error(error)

            # Determine error severity
            error_severity = self._get_error_severity(error)

            # Log error with timing information
            self._log_error(context, resolved=False, operation_time_ms=operation_time_ms)

            # Record failure metrics
            self._record_failure(operation_time_ms=operation_time_ms, error_severity=error_severity)

            # Record error metrics
            if self.enable_metrics:
                await error_metrics.record_error(
                    function_name,
                    context.category,
                    type(error).__name__,
                    resolved=False,
                    duration_ms=operation_time_ms,
                    severity=error_severity,
                )

            # Check if we should apply fallback (no more retries)
            if not self._should_retry(error, attempt):
                fallback_start_time = time.time()

                try:
                    # Generate cache key if configured
                    cache_key = None
                    if self.fallback_config.cache_key_func:
                        try:
                            cache_key = self.fallback_config.cache_key_func(*args, **kwargs)
                        except Exception as cache_key_error:
                            if self.enable_logging:
                                self.logger.warning(
                                    "Failed to generate cache key",
                                    handler_id=self.handler_id,
                                    function=function_name,
                                    error=str(cache_key_error),
                                )

                    # Apply fallback strategy
                    fallback_result = await self._apply_fallback(context, cache_key)

                    # Calculate fallback timing
                    fallback_time_ms = (time.time() - fallback_start_time) * 1000
                    total_handle_time_ms = (time.time() - handle_start_time) * 1000

                    # Log resolved error
                    self._log_error(context, resolved=True, operation_time_ms=total_handle_time_ms)

                    # Record resolved metrics
                    if self.enable_metrics:
                        await error_metrics.record_error(
                            function_name,
                            context.category,
                            type(error).__name__,
                            resolved=True,
                            duration_ms=fallback_time_ms,
                            severity=error_severity,
                        )

                    # Record partial success (error was handled)
                    self._record_success(operation_time_ms=total_handle_time_ms)

                    return True, fallback_result

                except ServiceError:
                    # Re-raise ServiceError from RAISE_DEGRADED strategy
                    raise
                except Exception as fallback_error:
                    # Log fallback failure but don't let it crash the system
                    if self.enable_logging:
                        self.logger.error(
                            "Critical fallback failure",
                            handler_id=self.handler_id,
                            function=function_name,
                            original_error=str(error),
                            fallback_error=str(fallback_error),
                        )
                    # Return handled with None to prevent crash
                    return True, None

            # Error should be retried
            return False, None

        except Exception as handler_error:
            # Critical: error handler itself failed
            if self.enable_logging:
                self.logger.critical(
                    "Error handler failure",
                    handler_id=self.handler_id,
                    function=function_name,
                    original_error=str(error),
                    handler_error=str(handler_error),
                    handler_error_type=type(handler_error).__name__,
                )

            # Update health status to reflect handler failure
            self._health_status["status"] = "unhealthy"

            # Return False to indicate handler couldn't process the error
            return False, None

    def _get_sensitivity_level(self, context: ErrorContext) -> SensitivityLevel:
        """Determine sensitivity level based on error context."""
        # Financial/trading functions always get high sensitivity
        financial_keywords = [
            "trading",
            "exchange",
            "wallet",
            "payment",
            "order",
            "position",
            "trade",
        ]

        if context.function_name and any(
            keyword in context.function_name.lower() for keyword in financial_keywords
        ):
            return SensitivityLevel.CRITICAL

        # Map error categories to sensitivity levels
        category_mapping = {
            ErrorCategory.NETWORK: SensitivityLevel.MEDIUM,
            ErrorCategory.DATA_QUALITY: SensitivityLevel.HIGH,
            ErrorCategory.VALIDATION: SensitivityLevel.LOW,
            ErrorCategory.CONFIGURATION: SensitivityLevel.MEDIUM,
            ErrorCategory.BUSINESS_LOGIC: SensitivityLevel.HIGH,
            ErrorCategory.SYSTEM: SensitivityLevel.MEDIUM,
            ErrorCategory.PERMISSION: SensitivityLevel.CRITICAL,
            ErrorCategory.RATE_LIMIT: SensitivityLevel.MEDIUM,
            ErrorCategory.RETRYABLE: SensitivityLevel.LOW,
            ErrorCategory.FATAL: SensitivityLevel.CRITICAL,
        }

        return category_mapping.get(context.category, SensitivityLevel.MEDIUM)

    def _get_error_severity(self, error: Exception) -> str:
        """Map exception to severity string for rate limiting."""
        if isinstance(error, SecurityError | StateConsistencyError):
            return "critical"
        elif isinstance(error, RiskManagementError | ExchangeError | ExecutionError):
            return "high"
        elif isinstance(error, DataError | ModelError):
            return "medium"
        else:
            return "low"


def enhanced_error_handler(
    retry_config: RetryConfig | None = None,
    circuit_breaker_config: CircuitBreakerConfig | None = None,
    fallback_config: FallbackConfig | None = None,
    enable_metrics: bool = True,
    enable_logging: bool = True,
    cache_results: bool = False,
    cache_ttl: int = 300,
) -> Callable[[F], F]:
    """
    Enhanced error handling decorator with comprehensive features.

    This decorator replaces the need for try-catch blocks by providing:
    - Intelligent error categorization
    - Automatic retry with backoff
    - Circuit breaker patterns
    - Fallback strategies
    - Performance metrics
    - Structured logging

    Args:
        retry_config: Retry behavior configuration
        circuit_breaker_config: Circuit breaker configuration
        fallback_config: Fallback strategy configuration
        enable_metrics: Enable error metrics tracking
        enable_logging: Enable structured error logging
        cache_results: Cache successful results
        cache_ttl: Cache time-to-live in seconds

    Returns:
        Decorated function with error handling
    """

    def decorator(func: F) -> F:
        handler = UniversalErrorHandler(
            retry_config=retry_config,
            circuit_breaker_config=circuit_breaker_config,
            fallback_config=fallback_config,
            enable_metrics=enable_metrics,
            enable_logging=enable_logging,
        )

        # LRU cache for successful results with memory management
        result_cache = LRUCache(
            max_size=DEFAULT_CACHE_SIZE if cache_results else (DEFAULT_CACHE_SIZE // 10),
            ttl_seconds=cache_ttl,
        )

        def _create_cache_key(*args, **kwargs) -> str:
            """Create cache key from arguments."""
            return f"{func.__name__}_{hash(str(args))}_{hash(str(kwargs))}"

        def _get_cached_result(cache_key: str) -> Any | None:
            """Get cached result if not expired."""
            if not cache_results:
                return None
            return result_cache.get(cache_key)

        def _cache_result(cache_key: str, result: Any) -> None:
            """Cache successful result."""
            if cache_results:
                result_cache.put(cache_key, result)

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                operation_start_time = time.time()
                correlation_id = correlation_context.get_correlation_id()

                # Check circuit breaker first
                if not handler._check_circuit_breaker():
                    if handler.enable_logging:
                        handler.logger.warning(
                            "Circuit breaker OPEN - applying fallback",
                            handler_id=handler.handler_id,
                            function=func.__name__,
                            circuit_state=handler._circuit_breaker_state,
                        )

                    # Create context for circuit breaker failure
                    context = ErrorContext.from_decorator_context(
                        error=ServiceError("Circuit breaker is OPEN"),
                        function_name=func.__name__,
                        args=args,
                        kwargs=kwargs,
                        attempt_number=1,
                        correlation_id=correlation_id,
                    )
                    context.category = ErrorCategory.SYSTEM

                    # Apply fallback for circuit breaker
                    cache_key = _create_cache_key(*args, **kwargs)
                    fallback_result = await handler._apply_fallback(context, cache_key)
                    return fallback_result

                # Check result cache first
                cache_key = _create_cache_key(*args, **kwargs)
                cached_result = _get_cached_result(cache_key)
                if cached_result is not None:
                    if handler.enable_logging:
                        handler.logger.debug(
                            "Result cache hit",
                            handler_id=handler.handler_id,
                            function=func.__name__,
                        )
                    # Record cache hit as successful operation
                    cache_time_ms = (time.time() - operation_start_time) * 1000
                    handler._record_success(operation_time_ms=cache_time_ms)
                    return cached_result

                attempt = 1
                last_exception = None

                while attempt <= handler.retry_config.max_attempts:
                    attempt_start_time = time.time()

                    try:
                        # Track half-open calls for circuit breaker
                        with handler._circuit_breaker_lock:
                            if handler._circuit_breaker_state == "HALF_OPEN":
                                handler._half_open_calls += 1

                        # Execute the function
                        if handler.enable_logging and attempt > 1:
                            handler.logger.info(
                                "Retrying function execution",
                                handler_id=handler.handler_id,
                                function=func.__name__,
                                attempt=attempt,
                                correlation_id=correlation_id,
                            )

                        # Ensure proper await for async functions
                        result = await func(*args, **kwargs)

                        # Calculate operation time
                        operation_time_ms = (time.time() - attempt_start_time) * 1000

                        # Record success
                        handler._record_success(operation_time_ms=operation_time_ms)

                        # Cache successful result
                        _cache_result(cache_key, result)

                        # Store in fallback cache if configured
                        if handler.fallback_config.use_cache:
                            try:
                                handler._fallback_cache.put(cache_key, result)
                            except Exception as cache_error:
                                # Don't let cache failures break the flow
                                if handler.enable_logging:
                                    handler.logger.warning(
                                        "Failed to store result in fallback cache",
                                        handler_id=handler.handler_id,
                                        error=str(cache_error),
                                    )

                        # Log successful execution on first success or after retries
                        if handler.enable_logging and (attempt > 1 or operation_time_ms > 1000):
                            handler.logger.info(
                                "Function executed successfully",
                                handler_id=handler.handler_id,
                                function=func.__name__,
                                attempt=attempt,
                                duration_ms=operation_time_ms,
                                correlation_id=correlation_id,
                            )

                        return result

                    except Exception as e:
                        last_exception = e
                        attempt_time_ms = (time.time() - attempt_start_time) * 1000

                        # Handle the error
                        handled, fallback_result = await handler.handle_error(
                            error=e,
                            function_name=func.__name__,
                            args=args,
                            kwargs=kwargs,
                            attempt=attempt,
                            correlation_id=correlation_id,
                            metadata={
                                "cache_key": cache_key,
                                "attempt_duration_ms": attempt_time_ms,
                            },
                            operation_start_time=operation_start_time,
                        )

                        if handled:
                            return fallback_result

                        # Check if we should retry
                        if not handler._should_retry(e, attempt):
                            break

                        # Calculate and apply retry delay
                        if attempt < handler.retry_config.max_attempts:
                            delay = calculate_retry_delay(attempt, handler.retry_config)
                            if handler.enable_logging:
                                handler.logger.info(
                                    "Scheduling retry with backoff",
                                    handler_id=handler.handler_id,
                                    function=func.__name__,
                                    delay_seconds=delay,
                                    next_attempt=attempt + 1,
                                    correlation_id=correlation_id,
                                )
                            await asyncio.sleep(delay)

                        attempt += 1

                # All retries exhausted - log and raise
                total_time_ms = (time.time() - operation_start_time) * 1000
                if handler.enable_logging:
                    handler.logger.error(
                        "All retries exhausted - operation failed",
                        handler_id=handler.handler_id,
                        function=func.__name__,
                        total_attempts=attempt - 1,
                        total_duration_ms=total_time_ms,
                        final_error=str(last_exception),
                        correlation_id=correlation_id,
                    )

                raise last_exception

            return async_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                """Synchronous wrapper with comprehensive error handling."""
                operation_start_time = time.time()
                correlation_id = correlation_context.get_correlation_id()

                # Check circuit breaker
                if not handler._check_circuit_breaker():
                    if handler.enable_logging:
                        handler.logger.warning(
                            "Circuit breaker OPEN - applying fallback (sync)",
                            handler_id=handler.handler_id,
                            function=func.__name__,
                            circuit_state=handler._circuit_breaker_state,
                        )

                    # For sync functions, we can't use async fallback easily
                    # Return a simple fallback value based on strategy
                    strategy = handler.fallback_config.strategy
                    if strategy == FallbackStrategy.RETURN_NONE:
                        return None
                    elif strategy == FallbackStrategy.RETURN_DEFAULT:
                        return handler.fallback_config.default_value
                    elif strategy == FallbackStrategy.RETURN_EMPTY:
                        # Simple empty value detection for sync
                        func_name = func.__name__.lower()
                        if "list" in func_name or "array" in func_name:
                            return []
                        elif "dict" in func_name or "map" in func_name:
                            return {}
                        else:
                            return None
                    else:
                        return None

                # Check result cache
                cache_key = _create_cache_key(*args, **kwargs)
                cached_result = _get_cached_result(cache_key)
                if cached_result is not None:
                    if handler.enable_logging:
                        handler.logger.debug(
                            "Result cache hit (sync)",
                            handler_id=handler.handler_id,
                            function=func.__name__,
                        )
                    cache_time_ms = (time.time() - operation_start_time) * 1000
                    handler._record_success(operation_time_ms=cache_time_ms)
                    return cached_result

                attempt = 1
                last_exception = None

                while attempt <= handler.retry_config.max_attempts:
                    attempt_start_time = time.time()

                    try:
                        # Track half-open calls
                        with handler._circuit_breaker_lock:
                            if handler._circuit_breaker_state == "HALF_OPEN":
                                handler._half_open_calls += 1

                        # Execute function
                        if handler.enable_logging and attempt > 1:
                            handler.logger.info(
                                "Retrying function execution (sync)",
                                handler_id=handler.handler_id,
                                function=func.__name__,
                                attempt=attempt,
                                correlation_id=correlation_id,
                            )

                        result = func(*args, **kwargs)

                        # Calculate timing
                        operation_time_ms = (time.time() - attempt_start_time) * 1000

                        # Record success
                        handler._record_success(operation_time_ms=operation_time_ms)

                        # Cache result
                        _cache_result(cache_key, result)

                        # Store in fallback cache
                        if handler.fallback_config.use_cache:
                            try:
                                handler._fallback_cache.put(cache_key, result)
                            except Exception as cache_error:
                                if handler.enable_logging:
                                    handler.logger.warning(
                                        "Failed to store result in fallback cache (sync)",
                                        handler_id=handler.handler_id,
                                        error=str(cache_error),
                                    )

                        return result

                    except Exception as e:
                        last_exception = e
                        attempt_time_ms = (time.time() - attempt_start_time) * 1000

                        # Record failure
                        error_severity = handler._get_error_severity(e)
                        handler._record_failure(
                            operation_time_ms=attempt_time_ms, error_severity=error_severity
                        )

                        # For sync functions, we handle fallback directly here
                        if not handler._should_retry(e, attempt):
                            # Apply fallback strategy
                            strategy = handler.fallback_config.strategy

                            if strategy == FallbackStrategy.RETURN_NONE:
                                if handler.enable_logging:
                                    handler.logger.info(
                                        "Applying RETURN_NONE fallback (sync)",
                                        handler_id=handler.handler_id,
                                        function=func.__name__,
                                    )
                                return None
                            elif strategy == FallbackStrategy.RETURN_DEFAULT:
                                if handler.enable_logging:
                                    handler.logger.info(
                                        "Applying RETURN_DEFAULT fallback (sync)",
                                        handler_id=handler.handler_id,
                                        function=func.__name__,
                                    )
                                return handler.fallback_config.default_value
                            elif strategy == FallbackStrategy.RAISE_DEGRADED:
                                if handler.enable_logging:
                                    handler.logger.warning(
                                        "Raising degraded service error (sync)",
                                        handler_id=handler.handler_id,
                                        function=func.__name__,
                                    )
                                raise ServiceError(
                                    f"Service degraded due to error in {func.__name__}",
                                    details={
                                        "original_error": str(e),
                                        "handler_id": handler.handler_id,
                                    },
                                )
                            else:
                                # Default fallback
                                return None

                        if not handler._should_retry(e, attempt):
                            break

                        if attempt < handler.retry_config.max_attempts:
                            delay = calculate_retry_delay(attempt, handler.retry_config)
                            if handler.enable_logging:
                                handler.logger.info(
                                    "Scheduling retry with backoff (sync)",
                                    handler_id=handler.handler_id,
                                    function=func.__name__,
                                    delay_seconds=delay,
                                    next_attempt=attempt + 1,
                                )
                            time.sleep(delay)

                        attempt += 1

                # All retries exhausted
                total_time_ms = (time.time() - operation_start_time) * 1000
                if handler.enable_logging:
                    handler.logger.error(
                        "All retries exhausted - operation failed (sync)",
                        handler_id=handler.handler_id,
                        function=func.__name__,
                        total_attempts=attempt - 1,
                        total_duration_ms=total_time_ms,
                        final_error=str(last_exception),
                    )

                raise last_exception

            return sync_wrapper  # type: ignore[return-value]

    return decorator


# Specialized decorators for common patterns


def handle_database_errors(
    fallback_value: Any = None, enable_cache: bool = True
) -> Callable[[F], F]:
    """Decorator specifically for database operations."""
    return enhanced_error_handler(
        retry_config=RetryConfig(
            max_attempts=3,
            base_delay=Decimal("0.5"),
            retriable_errors=(DatabaseConnectionError, DataError),
        ),
        fallback_config=FallbackConfig(
            strategy=FallbackStrategy.RETURN_DEFAULT,
            default_value=fallback_value,
            use_cache=enable_cache,
        ),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=5, recovery_timeout=Decimal("30.0")
        ),
    )


def handle_network_errors(
    max_retries: int = 5, fallback_function: Callable | None = None
) -> Callable[[F], F]:
    """Decorator specifically for network operations."""
    return enhanced_error_handler(
        retry_config=RetryConfig(
            max_attempts=max_retries,
            base_delay=Decimal("1.0"),
            max_delay=Decimal("60.0"),
            backoff_factor=Decimal("2.0"),
            retriable_errors=(NetworkError, ExchangeError, ConnectionError, TimeoutError),
        ),
        fallback_config=FallbackConfig(
            strategy=(
                FallbackStrategy.RETRY_ALTERNATIVE
                if fallback_function
                else FallbackStrategy.RETURN_NONE
            ),
            fallback_function=fallback_function,
        ),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=3, recovery_timeout=Decimal("60.0")
        ),
    )


def handle_validation_errors(strict: bool = True) -> Callable[[F], F]:
    """Decorator specifically for validation operations."""
    strategy = FallbackStrategy.RAISE_DEGRADED if strict else FallbackStrategy.RETURN_NONE

    return enhanced_error_handler(
        retry_config=RetryConfig(
            max_attempts=1,  # Don't retry validation errors
            retriable_errors=(),
        ),
        fallback_config=FallbackConfig(strategy=strategy),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=10, recovery_timeout=Decimal("5.0")
        ),
    )


def handle_trading_errors(emergency_stop: bool = True) -> Callable[[F], F]:
    """Decorator specifically for trading operations."""
    return enhanced_error_handler(
        retry_config=RetryConfig(
            max_attempts=2,  # Minimal retries for trading
            base_delay=Decimal("0.1"),
            retriable_errors=(NetworkError, ExchangeError),
        ),
        fallback_config=FallbackConfig(
            strategy=(
                FallbackStrategy.RAISE_DEGRADED if emergency_stop else FallbackStrategy.RETURN_NONE
            )
        ),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=2,  # Low threshold for trading
            recovery_timeout=Decimal("120.0"),
        ),
    )


# Utility functions for error handling


def get_error_metrics(function_name: str | None = None) -> dict[str, Any]:
    """Get error metrics for analysis."""
    return error_metrics.get_metrics(function_name)


def reset_error_metrics(function_name: str | None = None) -> None:
    """Reset error metrics."""
    error_metrics.reset_metrics(function_name)


def create_custom_retry_config(
    max_attempts: int = 3,
    base_delay: Decimal = Decimal("1.0"),
    max_delay: Decimal = Decimal("60.0"),
    backoff_factor: Decimal = Decimal("2.0"),
    retriable_errors: tuple = (Exception,),
) -> RetryConfig:
    """Create custom retry configuration."""
    return RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
        retriable_errors=retriable_errors,
    )


def create_custom_fallback_config(
    strategy: FallbackStrategy = FallbackStrategy.RETURN_NONE,
    default_value: Any = None,
    fallback_function: Callable[..., Any] | None = None,
) -> FallbackConfig:
    """Create custom fallback configuration."""
    return FallbackConfig(
        strategy=strategy, default_value=default_value, fallback_function=fallback_function
    )


# Compatibility aliases for commonly used decorators


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: Decimal = Decimal("1.0"),
    max_delay: Decimal = Decimal("60.0"),
    backoff_factor: Decimal = Decimal("2.0"),
    exceptions: tuple = (Exception,),
    **kwargs,
):
    """Legacy compatibility wrapper for retry_with_backoff."""
    retry_config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
        retriable_errors=exceptions,
    )
    return enhanced_error_handler(retry_config=retry_config, **kwargs)


def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type[Exception] = Exception,
):
    """Circuit breaker decorator - wrapper around enhanced_error_handler."""
    circuit_config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=Decimal(str(recovery_timeout)),
        expected_exception=expected_exception,
    )
    return enhanced_error_handler(circuit_breaker_config=circuit_config)


def with_retry(
    max_retries: int | None = None,
    max_attempts: int | None = None,
    base_delay: Decimal = Decimal("1.0"),
    max_delay: Decimal = Decimal("60.0"),
    exponential_base: Decimal | None = None,
    backoff_factor: Decimal | None = None,
    exceptions: tuple | None = None,
):
    """Retry decorator - wrapper around enhanced_error_handler."""
    # Support both parameter names for backward compatibility
    if max_attempts is not None:
        max_retries = max_attempts
    elif max_retries is None:
        max_retries = 3

    if backoff_factor is not None:
        exponential_base = backoff_factor
    elif exponential_base is None:
        exponential_base = Decimal("2.0")

    # At this point exponential_base is guaranteed to be Decimal
    assert exponential_base is not None, "exponential_base should not be None"

    # Handle exceptions parameter
    retriable_errors = (
        exceptions
        if exceptions is not None
        else (
            NetworkError,
            DatabaseConnectionError,
            ExchangeError,
            ServiceError,
        )
    )

    retry_config = RetryConfig(
        max_attempts=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_factor=exponential_base,
        retriable_errors=retriable_errors,
    )
    return enhanced_error_handler(retry_config=retry_config)


def with_fallback(
    strategy: FallbackStrategy = FallbackStrategy.RETURN_NONE,
    default_value: Any = None,
    fallback_function: Callable[..., Any] | None = None,
):
    """Fallback decorator - wrapper around enhanced_error_handler."""
    fallback_config = FallbackConfig(
        strategy=strategy, default_value=default_value, fallback_function=fallback_function
    )
    return enhanced_error_handler(fallback_config=fallback_config)


def with_error_context(
    component: str | None = None,
    operation: str | None = None,
):
    """Add error context to decorated function."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Add context to the exception
                context = ErrorContext.from_exception(
                    error=e,
                    component=component or func.__module__,
                    operation=operation or func.__name__,
                )
                # Store context in exception args instead of attribute
                # This is safer and doesn't require modifying exception type
                if hasattr(e, "args") and isinstance(e.args, tuple):
                    e.args = (*e.args, {"error_context": context})
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Add context to the exception
                context = ErrorContext.from_exception(
                    error=e,
                    component=component or func.__module__,
                    operation=operation or func.__name__,
                )
                # Store context in exception args instead of attribute
                # This is safer and doesn't require modifying exception type
                if hasattr(e, "args") and isinstance(e.args, tuple):
                    e.args = (*e.args, {"error_context": context})
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Module-level cleanup functions for proper resource management


async def shutdown_all_error_handlers() -> None:
    """
    Shutdown all registered UniversalErrorHandler instances.

    This function should be called during application shutdown to ensure
    all cleanup tasks are properly cancelled and resources are freed.
    """
    await _handler_registry.shutdown_all()


def get_active_handler_count() -> int:
    """
    Get the count of active UniversalErrorHandler instances.

    Returns:
        int: Number of active error handlers
    """
    return _handler_registry.get_active_count()


# Compatibility alias for module cleanup
cleanup_all_error_handlers = shutdown_all_error_handlers


# Add weakref finalizer to handle emergency cleanup
def _emergency_cleanup() -> None:
    """Emergency cleanup when module is being unloaded."""
    try:
        # Try to get the current event loop and schedule cleanup
        try:
            loop = asyncio.get_running_loop()
            if not loop.is_closed():
                # Create a task for emergency cleanup if there's still a loop
                # Create cleanup task
                cleanup_task = loop.create_task(_handler_registry.shutdown_all())
                # Store task reference to prevent garbage collection
                _emergency_cleanup._last_task = cleanup_task  # type: ignore[attr-defined]
        except RuntimeError:
            # No event loop available, just do synchronous cleanup
            pass
    except Exception:
        # Ignore all errors during emergency cleanup
        # Don't log during shutdown as logging infrastructure may be unavailable
        pass


# Register emergency cleanup
atexit.register(_emergency_cleanup)
