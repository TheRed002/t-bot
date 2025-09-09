"""
Handler Pool for efficient error handler management.

This module provides a pooling mechanism for UniversalErrorHandler instances
to reduce memory overhead and improve performance.
"""

import hashlib
import threading
from typing import Any, Optional

from src.core.logging import get_logger
from src.error_handling.decorators import (
    CircuitBreakerConfig,
    FallbackConfig,
    RetryConfig,
)

logger = get_logger(__name__)


class UniversalErrorHandler:
    """Simple universal error handler stub for backward compatibility."""

    def __init__(
        self,
        retry_config: RetryConfig | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        fallback_config: FallbackConfig | None = None,
        enable_metrics: bool = True,
        enable_logging: bool = True,
    ):
        self.retry_config = retry_config
        self.circuit_breaker_config = circuit_breaker_config
        self.fallback_config = fallback_config
        self.enable_metrics = enable_metrics
        self.enable_logging = enable_logging

    def handle_error(self, error: Exception, context: dict | None = None) -> bool:
        """Simple error handling stub."""
        logger.error(f"Error handled: {error}")
        return True


class HandlerPool:
    """
    Singleton pool for managing UniversalErrorHandler instances.

    Instead of creating a new handler for each decorated function,
    we reuse handlers with identical configurations.
    """

    _instance: Optional["HandlerPool"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "HandlerPool":
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the handler pool."""
        if self._initialized:
            return

        self._handlers: dict[str, UniversalErrorHandler] = {}
        self._usage_count: dict[str, int] = {}
        self._lock = threading.Lock()
        self._initialized = True

        logger.info("HandlerPool initialized")

    def _create_config_key(
        self,
        retry_config: RetryConfig | None,
        circuit_breaker_config: CircuitBreakerConfig | None,
        fallback_config: FallbackConfig | None,
        enable_metrics: bool,
        enable_logging: bool,
    ) -> str:
        """
        Create a unique key for a handler configuration.

        Args:
            retry_config: Retry configuration
            circuit_breaker_config: Circuit breaker configuration
            fallback_config: Fallback configuration
            enable_metrics: Whether to enable metrics
            enable_logging: Whether to enable logging

        Returns:
            Unique key for the configuration
        """
        # Create a tuple of all config values
        config_tuple = (
            # Retry config
            (
                retry_config.max_attempts if retry_config else None,
                str(retry_config.base_delay) if retry_config else None,
                str(retry_config.max_delay) if retry_config else None,
                (
                    str(retry_config.exponential)
                    if retry_config and hasattr(retry_config, "exponential")
                    else None
                ),
                (
                    str(retry_config.retriable_errors)
                    if retry_config and hasattr(retry_config, "retriable_errors")
                    else None
                ),
            ),
            # Circuit breaker config
            (
                circuit_breaker_config.failure_threshold if circuit_breaker_config else None,
                str(circuit_breaker_config.recovery_timeout) if circuit_breaker_config else None,
                str(circuit_breaker_config.expected_exception) if circuit_breaker_config else None,
            ),
            # Fallback config
            (
                fallback_config.strategy if fallback_config else None,
                str(fallback_config.default_value) if fallback_config else None,
                (
                    fallback_config.fallback_function.__name__
                    if fallback_config and fallback_config.fallback_function
                    else None
                ),
            ),
            # Other settings
            enable_metrics,
            enable_logging,
        )

        # Create hash of the configuration
        config_str = str(config_tuple)
        return hashlib.md5(config_str.encode()).hexdigest()

    def get_handler(
        self,
        retry_config: RetryConfig | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        fallback_config: FallbackConfig | None = None,
        enable_metrics: bool = True,
        enable_logging: bool = True,
    ) -> UniversalErrorHandler:
        """
        Get or create a handler with the specified configuration.

        Args:
            retry_config: Retry configuration
            circuit_breaker_config: Circuit breaker configuration
            fallback_config: Fallback configuration
            enable_metrics: Whether to enable metrics
            enable_logging: Whether to enable logging

        Returns:
            UniversalErrorHandler instance
        """
        # Create configuration key
        config_key = self._create_config_key(
            retry_config,
            circuit_breaker_config,
            fallback_config,
            enable_metrics,
            enable_logging,
        )

        with self._lock:
            # Check if handler exists
            if config_key in self._handlers:
                self._usage_count[config_key] += 1
                usage_count = self._usage_count[config_key]
                logger.debug(f"Reusing handler {config_key[:8]}... (usage count: {usage_count})")
                return self._handlers[config_key]

            # Create new handler
            handler = UniversalErrorHandler(
                retry_config=retry_config,
                circuit_breaker_config=circuit_breaker_config,
                fallback_config=fallback_config,
                enable_metrics=enable_metrics,
                enable_logging=enable_logging,
            )

            self._handlers[config_key] = handler
            self._usage_count[config_key] = 1

            logger.info(
                f"Created new handler {config_key[:8]}... (total handlers: {len(self._handlers)})"
            )

            return handler

    def get_stats(self) -> dict[str, Any]:
        """
        Get pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        with self._lock:
            total_handlers = len(self._handlers)
            total_usage = sum(self._usage_count.values())
            avg_usage = total_usage / total_handlers if total_handlers > 0 else 0

            return {
                "total_handlers": total_handlers,
                "total_usage_count": total_usage,
                "average_usage_per_handler": avg_usage,
                "handler_configs": list(self._handlers.keys()),
                "usage_distribution": dict(self._usage_count),
            }

    def shutdown(self):
        """Shutdown all handlers in the pool."""
        import asyncio
        import os

        with self._lock:
            if not self._handlers:
                return  # Nothing to shutdown

            logger.info(f"Shutting down {len(self._handlers)} pooled handlers")

            # In testing mode or when event loop is running, do synchronous cleanup
            # to avoid creating unawaited coroutines
            if os.environ.get("TESTING") or self._is_event_loop_running():
                self._sync_shutdown_handlers()
                return

            # Only do async shutdown when we can properly run it
            loop = None
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    # Loop is closed, create a new one for shutdown
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self._async_shutdown_impl())
                    finally:
                        if loop:
                            loop.close()
                        asyncio.set_event_loop(None)
                else:
                    # Loop exists and is not closed, and not running
                    loop.run_until_complete(self._async_shutdown_impl())
            except Exception as e:
                logger.warning(f"Could not perform async shutdown: {e}")
                self._sync_shutdown_handlers()

    def _is_event_loop_running(self) -> bool:
        """Check if there's a running event loop."""
        import asyncio

        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    def _sync_shutdown_handlers(self):
        """Synchronous handler shutdown to avoid unawaited coroutines."""
        # Since UniversalErrorHandler doesn't have shutdown methods, just clear the pool
        self._handlers.clear()
        self._usage_count.clear()
        logger.info("HandlerPool synchronous shutdown complete")

    async def _async_shutdown_impl(self):
        """Internal async shutdown implementation."""
        # Since UniversalErrorHandler doesn't have shutdown methods, just clear the pool
        self._handlers.clear()
        self._usage_count.clear()
        logger.info("HandlerPool async shutdown implementation complete")

    async def async_shutdown(self):
        """Async shutdown method that properly awaits all handlers."""
        with self._lock:
            if not self._handlers:
                return  # Nothing to shutdown

            logger.info(f"Shutting down {len(self._handlers)} pooled handlers (async)")

        await self._async_shutdown_impl()

    def clear(self):
        """Clear the handler pool (useful for testing)."""
        self.shutdown()
        self._initialized = False
        HandlerPool._instance = None


_handler_pool = None


def get_pooled_handler(
    retry_config: RetryConfig | None = None,
    circuit_breaker_config: CircuitBreakerConfig | None = None,
    fallback_config: FallbackConfig | None = None,
    enable_metrics: bool = True,
    enable_logging: bool = True,
) -> UniversalErrorHandler:
    """
    Get a pooled handler instance.

    This is the main interface for getting handlers from the pool.

    Args:
        retry_config: Retry configuration
        circuit_breaker_config: Circuit breaker configuration
        fallback_config: Fallback configuration
        enable_metrics: Whether to enable metrics
        enable_logging: Whether to enable logging

    Returns:
        UniversalErrorHandler instance from the pool
    """
    global _handler_pool
    if _handler_pool is None:
        _handler_pool = HandlerPool()
    return _handler_pool.get_handler(
        retry_config=retry_config,
        circuit_breaker_config=circuit_breaker_config,
        fallback_config=fallback_config,
        enable_metrics=enable_metrics,
        enable_logging=enable_logging,
    )


def get_pool_stats() -> dict[str, Any]:
    """Get statistics about the handler pool."""
    pool = HandlerPool()
    return pool.get_stats()


def shutdown_handler_pool():
    """Shutdown the handler pool."""
    if _handler_pool is not None:
        _handler_pool.shutdown()


async def async_shutdown_handler_pool():
    """Async shutdown the handler pool."""
    if _handler_pool is None:
        return
    await _handler_pool.async_shutdown()
