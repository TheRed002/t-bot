"""
Rate limiting framework for exchange APIs.

This module implements token bucket rate limiting to prevent API violations
and ensure compliance with exchange-specific rate limits.

CRITICAL: This integrates with P-001 (core types, exceptions, config)
and P-002A (error handling) components.
"""

import asyncio
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from src.core.config import Config

# MANDATORY: Import from P-001
from src.core.exceptions import ExchangeRateLimitError

# Logger is provided by BaseExchange (via BaseComponent)
# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import APIRateLimitRecovery

# MANDATORY: Import monitoring from P-030
from src.monitoring.metrics import MetricDefinition, MetricsCollector
from src.utils import RATE_LIMITS

# MANDATORY: Import from P-007A (utils)
from src.utils.decorators import log_calls, retry


class TokenBucket:
    """
    Token bucket implementation for rate limiting.

    This class implements a token bucket algorithm for rate limiting
    with configurable capacity and refill rate.
    """

    def __init__(self, capacity: int, refill_rate: float, refill_time: float = 1.0):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens per refill_time
            refill_time: Time interval for refill in seconds
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.refill_time = refill_time
        self.tokens = capacity
        self.last_refill = time.time()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        time_passed = now - self.last_refill

        if time_passed >= self.refill_time:
            tokens_to_add = (time_passed / self.refill_time) * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + int(tokens_to_add))
            self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            bool: True if tokens available, False otherwise
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Calculate wait time for token consumption.

        Args:
            tokens: Number of tokens needed

        Returns:
            float: Wait time in seconds
        """
        self._refill()

        if self.tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self.tokens
        return (tokens_needed / self.refill_rate) * self.refill_time


class RateLimiter:
    """
    Rate limiter for exchange API calls.

    This class manages rate limiting for different types of API calls
    and provides automatic retry with exponential backoff.
    """

    def __init__(
        self,
        config: Config,
        exchange_name: str,
        metrics_collector: MetricsCollector | None = None,
    ):
        """
        Initialize rate limiter.

        Args:
            config: Application configuration
            exchange_name: Name of the exchange
            metrics_collector: Optional metrics collector instance
        """
        self.config = config
        self.exchange_name = exchange_name
        # Initialize error handler if available
        if hasattr(config, "error_handling"):
            self.error_handler = ErrorHandler(config.error_handling)
        else:
            self.error_handler = None

        # Initialize metrics collector
        self.metrics_collector = metrics_collector
        if self.metrics_collector:
            self._register_metrics()

        # Get rate limits from config or use constants as defaults
        if hasattr(config, "exchanges") and hasattr(config.exchanges, "rate_limits"):
            rate_limits = config.exchanges.rate_limits.get(exchange_name, {})
        else:
            rate_limits = {}
        default_limits = RATE_LIMITS.get(exchange_name, RATE_LIMITS.get("default", {}))

        # Initialize token buckets for different rate limits
        self.buckets: dict[str, TokenBucket] = {}

        # Requests per minute bucket
        requests_per_minute = rate_limits.get(
            "requests_per_minute", default_limits.get("requests_per_minute", 1200)
        )
        self.buckets["requests_per_minute"] = TokenBucket(
            capacity=requests_per_minute, refill_rate=requests_per_minute, refill_time=60.0
        )

        # Orders per second bucket
        orders_per_second = rate_limits.get(
            "orders_per_second", default_limits.get("orders_per_second", 10)
        )
        self.buckets["orders_per_second"] = TokenBucket(
            capacity=orders_per_second, refill_rate=orders_per_second, refill_time=1.0
        )

        # WebSocket connections bucket
        websocket_connections = rate_limits.get(
            "websocket_connections", default_limits.get("websocket_connections", 5)
        )
        self.buckets["websocket_connections"] = TokenBucket(
            capacity=websocket_connections,
            refill_rate=websocket_connections,
            refill_time=300.0,  # 5 minutes
        )

        # Request tracking
        self.request_history: dict[str, list] = defaultdict(list)
        self.last_request_time: dict[str, float] = defaultdict(lambda: 0.0)

        # Create logger - fallback if BaseComponent not properly initialized
        if not hasattr(self, "logger"):
            import logging

            self.logger = logging.getLogger(
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )

        self.logger.info(f"Initialized rate limiter for {exchange_name}")

    def _register_metrics(self) -> None:
        """
        Register rate limiting metrics.
        """
        if not self.metrics_collector:
            return

        metrics = [
            # Total requests metric
            MetricDefinition(
                "rate_limit_requests_total",
                "Total rate limit requests by exchange and bucket",
                "counter",
                ["exchange", "bucket"],
            ),
            # Rate limit hits metric
            MetricDefinition(
                "rate_limit_hits_total",
                "Total rate limit hits by exchange and bucket",
                "counter",
                ["exchange", "bucket"],
            ),
            # Wait time metric
            MetricDefinition(
                "rate_limit_wait_time_seconds",
                "Wait times when rate limited",
                "histogram",
                ["exchange", "bucket"],
                [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            ),
            # Bucket tokens available metric
            MetricDefinition(
                "bucket_tokens_available",
                "Current tokens available in each bucket",
                "gauge",
                ["exchange", "bucket"],
            ),
            # Rate limit violations metric
            MetricDefinition(
                "rate_limit_violations_total",
                "Total rate limit violations",
                "counter",
                ["exchange", "bucket", "violation_type"],
            ),
        ]

        for metric_def in metrics:
            try:
                self.metrics_collector.register_metric(metric_def)
            except Exception as e:
                self.logger.warning(f"Failed to register metric {metric_def.name}: {e}")

    async def _handle_rate_limit_error(
        self, error: Exception, operation: str, bucket_name: str | None = None
    ) -> None:
        """
        Handle rate limit errors using the error handler.

        Args:
            error: The exception that occurred
            operation: The operation being performed
            bucket_name: The rate limit bucket name
        """
        try:
            if self.error_handler:
                # Create error context
                error_context = self.error_handler.create_error_context(
                    error=error,
                    component="exchange_rate_limiter",
                    operation=operation,
                    details={
                        "exchange_name": self.exchange_name,
                        "operation": operation,
                        "bucket_name": bucket_name,
                    },
                )

                # Use API rate limit recovery for rate limit violations
                recovery_scenario = APIRateLimitRecovery(self.config)

                # Handle the error
                await self.error_handler.handle_error(error, error_context, recovery_scenario)
            else:
                # Fallback if no error handler available
                self.logger.error(f"Rate limit error in {operation}: {error}")

        except Exception as e:
            # Fallback to basic logging if error handling fails
            self.logger.error(f"Error handling failed for {operation}: {e!s}")

    @retry(max_attempts=3, base_delay=1.0)
    @log_calls
    async def acquire(
        self, bucket_name: str = "requests_per_minute", tokens: int = 1, timeout: float = 30.0
    ) -> bool:
        """
        Acquire tokens from a rate limit bucket.

        Args:
            bucket_name: Name of the bucket to acquire from
            tokens: Number of tokens to acquire
            timeout: Maximum wait time in seconds

        Returns:
            bool: True if tokens acquired, False if timeout

        Raises:
            ExchangeRateLimitError: If rate limit exceeded
        """
        if bucket_name not in self.buckets:
            self.logger.warning(f"Unknown rate limit bucket: {bucket_name}")
            return True

        bucket = self.buckets[bucket_name]
        start_time = time.time()

        # Track the request
        self._track_rate_limit_request(bucket_name)

        while time.time() - start_time < timeout:
            if bucket.consume(tokens):
                return self._handle_successful_acquisition(bucket_name, tokens)

            # Record rate limit hit
            self._track_rate_limit_hit(bucket_name)

            # Calculate wait time and check timeout
            wait_time = bucket.get_wait_time(tokens)
            if wait_time > timeout:
                await self._handle_wait_timeout(bucket_name, wait_time, timeout)

            # Record wait time and sleep
            self._track_wait_time(bucket_name, wait_time)
            await asyncio.sleep(min(wait_time, 0.1))

        # Handle final timeout
        self._track_acquisition_timeout(bucket_name)
        raise ExchangeRateLimitError(f"Rate limit timeout for {bucket_name}")

    def _track_rate_limit_request(self, bucket_name: str) -> None:
        """Track rate limit request metric."""
        if self.metrics_collector:
            self.metrics_collector.increment_counter(
                "rate_limit_requests_total", {"exchange": self.exchange_name, "bucket": bucket_name}
            )

    def _handle_successful_acquisition(self, bucket_name: str, tokens: int) -> bool:
        """Handle successful token acquisition."""
        self._record_request(bucket_name, tokens)
        # Update bucket tokens metric
        if self.metrics_collector:
            self.metrics_collector.set_gauge(
                "bucket_tokens_available",
                self.buckets[bucket_name].tokens,
                {"exchange": self.exchange_name, "bucket": bucket_name},
            )
        return True

    def _track_rate_limit_hit(self, bucket_name: str) -> None:
        """Track rate limit hit metric."""
        if self.metrics_collector:
            self.metrics_collector.increment_counter(
                "rate_limit_hits_total", {"exchange": self.exchange_name, "bucket": bucket_name}
            )

    async def _handle_wait_timeout(
        self, bucket_name: str, wait_time: float, timeout: float
    ) -> None:
        """Handle when wait time exceeds timeout."""
        if self.metrics_collector:
            self.metrics_collector.increment_counter(
                "rate_limit_violations_total",
                {
                    "exchange": self.exchange_name,
                    "bucket": bucket_name,
                    "violation_type": "timeout_exceeded",
                },
            )
        rate_limit_error = ExchangeRateLimitError(
            f"Rate limit exceeded for {bucket_name}. "
            f"Wait time {wait_time:.2f}s exceeds timeout {timeout}s"
        )
        await self._handle_rate_limit_error(rate_limit_error, "acquire", bucket_name)
        raise rate_limit_error

    def _track_wait_time(self, bucket_name: str, wait_time: float) -> None:
        """Track wait time metric."""
        if self.metrics_collector:
            self.metrics_collector.observe_histogram(
                "rate_limit_wait_time_seconds",
                wait_time,
                {"exchange": self.exchange_name, "bucket": bucket_name},
            )

    def _track_acquisition_timeout(self, bucket_name: str) -> None:
        """Track acquisition timeout metric."""
        if self.metrics_collector:
            self.metrics_collector.increment_counter(
                "rate_limit_violations_total",
                {
                    "exchange": self.exchange_name,
                    "bucket": bucket_name,
                    "violation_type": "acquire_timeout",
                },
            )

    def _record_request(self, bucket_name: str, tokens: int) -> None:
        """
        Record a request for monitoring purposes.

        Args:
            bucket_name: Name of the bucket
            tokens: Number of tokens consumed
        """
        now = time.time()
        self.request_history[bucket_name].append({"timestamp": now, "tokens": tokens})

        # Clean up old history (keep last 1000 requests)
        if len(self.request_history[bucket_name]) > 1000:
            self.request_history[bucket_name] = self.request_history[bucket_name][-1000:]

        self.last_request_time[bucket_name] = now

        # Update metrics if collector is available
        if self.metrics_collector and bucket_name in self.buckets:
            bucket = self.buckets[bucket_name]
            self.metrics_collector.set_gauge(
                "bucket_tokens_available",
                bucket.tokens,
                {"exchange": self.exchange_name, "bucket": bucket_name},
            )

    def get_bucket_status(self, bucket_name: str) -> dict[str, Any]:
        """
        Get status of a rate limit bucket.

        Args:
            bucket_name: Name of the bucket

        Returns:
            Dict[str, Any]: Bucket status information
        """
        if bucket_name not in self.buckets:
            return {"error": f"Unknown bucket: {bucket_name}"}

        bucket = self.buckets[bucket_name]

        # Update metrics when status is requested
        if self.metrics_collector:
            self.metrics_collector.set_gauge(
                "bucket_tokens_available",
                bucket.tokens,
                {"exchange": self.exchange_name, "bucket": bucket_name},
            )

        return {
            "tokens_available": bucket.tokens,
            "capacity": bucket.capacity,
            "refill_rate": bucket.refill_rate,
            "refill_time": bucket.refill_time,
            "last_request": self.last_request_time.get(bucket_name, 0),
            "request_count": len(self.request_history.get(bucket_name, [])),
        }

    def get_all_bucket_status(self) -> dict[str, dict[str, Any]]:
        """
        Get status of all rate limit buckets.

        Returns:
            Dict[str, Dict[str, Any]]: Status of all buckets
        """
        return {
            bucket_name: self.get_bucket_status(bucket_name) for bucket_name in self.buckets.keys()
        }

    async def wait_for_capacity(self, bucket_name: str, tokens: int = 1) -> float:
        """
        Wait for capacity to become available.

        Args:
            bucket_name: Name of the bucket
            tokens: Number of tokens needed

        Returns:
            float: Wait time in seconds
        """
        if bucket_name not in self.buckets:
            return 0.0

        bucket = self.buckets[bucket_name]
        wait_time = bucket.get_wait_time(tokens)

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        return wait_time

    def is_rate_limited(self, bucket_name: str) -> bool:
        """
        Check if a bucket is currently rate limited.

        Args:
            bucket_name: Name of the bucket

        Returns:
            bool: True if rate limited, False otherwise
        """
        if bucket_name not in self.buckets:
            return False

        bucket = self.buckets[bucket_name]
        is_limited = bucket.tokens < 1

        # Update metrics when limit status is checked
        if self.metrics_collector:
            self.metrics_collector.set_gauge(
                "bucket_tokens_available",
                bucket.tokens,
                {"exchange": self.exchange_name, "bucket": bucket_name},
            )

        return is_limited

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass


class RateLimitDecorator:
    """
    Decorator for applying rate limiting to exchange API methods.

    This decorator automatically applies rate limiting to exchange API
    calls and handles retries with exponential backoff.
    """

    def __init__(
        self, bucket_name: str = "requests_per_minute", tokens: int = 1, timeout: float = 30.0
    ):
        """
        Initialize rate limit decorator.

        Args:
            bucket_name: Name of the rate limit bucket
            tokens: Number of tokens to consume
            timeout: Maximum wait time
        """
        self.bucket_name = bucket_name
        self.tokens = tokens
        self.timeout = timeout

    def __call__(self, func: Callable) -> Callable:
        """
        Apply rate limiting to a function.

        Args:
            func: Function to decorate

        Returns:
            Callable: Decorated function
        """

        async def wrapper(*args, **kwargs):
            # Get rate limiter from the first argument (should be self)
            if args and hasattr(args[0], "rate_limiter"):
                rate_limiter = args[0].rate_limiter

                # Check if rate_limiter has acquire method and is awaitable
                if hasattr(rate_limiter, "acquire") and asyncio.iscoroutinefunction(
                    rate_limiter.acquire
                ):
                    # Acquire tokens
                    await rate_limiter.acquire(
                        bucket_name=self.bucket_name, tokens=self.tokens, timeout=self.timeout
                    )

            # Execute the function
            return await func(*args, **kwargs)

        return wrapper
