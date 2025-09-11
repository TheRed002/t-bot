"""Advanced Redis cache manager with distributed locking and warming strategies."""

import asyncio
import hashlib
import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from src.core.base.component import BaseComponent
from src.core.base.interfaces import CacheClientInterface
from src.core.exceptions import CacheError


# Simple mixins to avoid circular imports
class DependencyInjectionMixin:
    """Simple mixin for dependency injection."""

    def __init__(self):
        self._injector = None

    def get_injector(self):
        if self._injector is None:
            from src.core.dependency_injection import get_global_injector
            self._injector = get_global_injector()
        return self._injector


class ConnectionManagerMixin:
    """Simple connection manager mixin."""

    def __init__(self):
        self._connections = {}


class ResourceCleanupMixin:
    """Simple resource cleanup mixin."""

    def __init__(self):
        self._resources = []

    async def cleanup_resources(self):
        """Clean up tracked resources."""
        pass


class LoggingHelperMixin:
    """Simple logging helper mixin."""

    def __init__(self):
        pass

from .cache_keys import CacheKeys
from .cache_metrics import get_cache_metrics


class CacheManager(
    BaseComponent,
    DependencyInjectionMixin,
    ConnectionManagerMixin,
    ResourceCleanupMixin,
    LoggingHelperMixin,
):
    """
    Advanced cache manager with:
    - Distributed locking for critical operations
    - Cache warming strategies
    - Automatic invalidation patterns
    - Performance monitoring
    - Fallback mechanisms
    """

    def __init__(self, redis_client: CacheClientInterface | None = None, config: Any | None = None):
        super().__init__()
        DependencyInjectionMixin.__init__(self)
        ConnectionManagerMixin.__init__(self)
        ResourceCleanupMixin.__init__(self)
        LoggingHelperMixin.__init__(self)

        self.redis_client = redis_client
        self.config = config
        self.metrics = get_cache_metrics()

        self.default_ttls = {
            "market_data": 5,
            "risk_metrics": 60,
            "state": 300,
            "orders": 30,
            "strategy": 600,
            "bot_config": 3600,
            "api_response": 300,
            "default": 3600,
        }

        self.lock_timeout = 30
        self.lock_retry_delay = 0.1

    async def _ensure_client(self):
        """Ensure Redis client is available with proper connection management."""
        if self._shutdown_requested:
            raise CacheError("Cache manager is shutting down")

        async with self._connection_lock:
            if not self.redis_client or self.redis_client is None:
                if self._dependency_container:
                    try:
                        self.redis_client = self._dependency_container.resolve("CacheClientInterface")
                    except (AttributeError, KeyError, TypeError) as e:
                        self.logger.debug(f"Could not resolve CacheClientInterface from DI container: {e}")

                if not self.redis_client:
                    raise CacheError("Cache client not available and no dependency injection configured")

            # Check if client is connected and healthy
            if (
                self.redis_client is None
                or self.redis_client.client is None
                or not await self._is_client_healthy()
            ):
                await self._reconnect_with_retries()

    async def _is_client_healthy(self) -> bool:
        """Check if Redis client is healthy."""
        try:
            if self.redis_client is None or self.redis_client.client is None:
                return False
            # Simple ping to check connectivity
            await asyncio.wait_for(self.redis_client.ping(), timeout=2.0)
            return True
        except (ConnectionError, TimeoutError, OSError, asyncio.TimeoutError) as e:
            self.logger.debug(f"Redis health check failed: {e}")
            return False

    async def _reconnect_with_retries(self):
        """Reconnect to Redis with exponential backoff."""
        for attempt in range(self._max_connection_retries):
            client_disconnected = False
            try:
                if self.redis_client.client is not None:
                    try:
                        await self.redis_client.disconnect()
                        client_disconnected = True
                    except (ConnectionError, OSError) as e:
                        # Log disconnection errors for debugging
                        # but don't let them block reconnection
                        self.logger.debug(
                            f"Failed to disconnect Redis client during reconnect: {e}"
                        )

                await self.redis_client.connect()

                # Test the connection
                await asyncio.wait_for(self.redis_client.ping(), timeout=2.0)

                self._connection_retries = 0  # Reset on success
                self.logger.info("Successfully reconnected to Redis")
                return

            except Exception as e:
                self._connection_retries += 1
                wait_time = self._connection_retry_delay * (2**attempt)

                if attempt < self._max_connection_retries - 1:
                    self.logger.warning(
                        f"Redis connection attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"All Redis connection attempts failed: {e}")
                    # Ensure disconnection on final failure
                    if self.redis_client.client is not None and not client_disconnected:
                        try:
                            await self.redis_client.disconnect()
                        except Exception as disconnect_e:
                            self.logger.warning(
                                "Failed to disconnect during cleanup",
                                error=str(disconnect_e),
                                error_type=type(disconnect_e).__name__,
                            )
                    raise CacheError(
                        f"Failed to connect to Redis after {self._max_connection_retries} "
                        f"attempts: {e}"
                    ) from e

    def _get_ttl(self, data_type: str = "default") -> int:
        """Get TTL for specific data type."""
        return self.default_ttls.get(data_type, self.default_ttls["default"])

    def _hash_key(self, key: str) -> str:
        """Create hash of key for consistent caching."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    async def get(
        self,
        key: str,
        namespace: str = "cache",
        fallback: Callable | None = None,
        fallback_args: tuple = (),
        fallback_kwargs: dict | None = None,
    ) -> Any:
        """
        Get value from cache with optional fallback.

        Args:
            key: Cache key
            namespace: Cache namespace
            fallback: Function to call if cache miss
            fallback_args: Args for fallback function
            fallback_kwargs: Kwargs for fallback function
        """
        start_time = time.time()
        fallback_kwargs = fallback_kwargs or {}

        try:
            await self._ensure_client()
            if self.redis_client is None:
                raise CacheError("Redis client not available")
            value = await self.redis_client.get(key, namespace)

            if value is not None:
                response_time = time.time() - start_time
                self.metrics.record_hit(namespace, response_time)
                return value

            # Cache miss - try fallback
            response_time = time.time() - start_time
            self.metrics.record_miss(namespace, response_time)

            if fallback:
                try:
                    if asyncio.iscoroutinefunction(fallback):
                        result = await fallback(*fallback_args, **fallback_kwargs)
                    else:
                        result = fallback(*fallback_args, **fallback_kwargs)

                    # Store result in cache for next time
                    data_type = namespace.split("_")[0] if "_" in namespace else namespace
                    await self.set(key, result, namespace=namespace, ttl=self._get_ttl(data_type))
                    return result
                except Exception as e:
                    self.logger.warning(f"Fallback function failed for key {key}: {e}")
                    self.metrics.record_error(namespace, "fallback_error")

            return None

        except Exception as e:
            self.logger.error(f"Cache get failed for key {key}: {e}")
            self.metrics.record_error(namespace, "get_error")
            raise CacheError(f"Failed to get from cache: {e}") from e

    async def set(
        self,
        key: str,
        value: Any,
        namespace: str = "cache",
        ttl: int | None = None,
        data_type: str = "default",
    ) -> bool:
        """Set value in cache with TTL."""
        start_time = time.time()

        try:
            await self._ensure_client()

            if self.redis_client is None:
                return False

            if ttl is None:
                ttl = self._get_ttl(data_type)

            result = await self.redis_client.set(key, value, ttl, namespace)

            response_time = time.time() - start_time
            self.metrics.record_set(namespace, response_time)

            return result

        except Exception as e:
            self.logger.error(f"Cache set failed for key {key}: {e}")
            self.metrics.record_error(namespace, "set_error")
            raise CacheError(f"Failed to set cache: {e}") from e

    async def delete(self, key: str, namespace: str = "cache") -> bool:
        """Delete key from cache."""
        start_time = time.time()

        try:
            await self._ensure_client()
            if self.redis_client:
                result = await self.redis_client.delete(key, namespace)
            else:
                result = False

            response_time = time.time() - start_time
            self.metrics.record_delete(namespace, response_time)

            return result

        except Exception as e:
            self.logger.error(f"Cache delete failed for key {key}: {e}")
            self.metrics.record_error(namespace, "delete_error")
            raise CacheError(f"Failed to delete from cache: {e}") from e

    async def exists(self, key: str, namespace: str = "cache") -> bool:
        """Check if key exists in cache."""
        try:
            await self._ensure_client()
            return await self.redis_client.exists(key, namespace) if self.redis_client else False
        except Exception as e:
            self.logger.error(
                "Cache exists check failed",
                key=key,
                namespace=namespace,
                error=str(e),
                error_type=type(e).__name__,
            )
            self.metrics.record_error(namespace, "exists_error")
            return False

    async def expire(self, key: str, ttl: int, namespace: str = "cache") -> bool:
        """Set expiration for existing key."""
        try:
            await self._ensure_client()
            return await self.redis_client.expire(key, ttl, namespace) if self.redis_client else False
        except Exception as e:
            self.logger.error(
                "Cache expire failed",
                key=key,
                namespace=namespace,
                ttl=ttl,
                error=str(e),
                error_type=type(e).__name__,
            )
            self.metrics.record_error(namespace, "expire_error")
            return False

    # Advanced operations
    async def get_many(self, keys: list[str], namespace: str = "cache") -> dict[str, Any]:
        """Get multiple values efficiently."""
        results = {}
        pipeline = None

        try:
            await self._ensure_client()

            # Use pipeline for efficiency
            if not self.redis_client or not self.redis_client.client:
                return {}
            pipeline = self.redis_client.client.pipeline()

            for key in keys:
                namespaced_key = self.redis_client._get_namespaced_key(key, namespace)
                pipeline.get(namespaced_key)

            values = await pipeline.execute()

            for i, key in enumerate(keys):
                value = values[i]
                if value is not None:
                    try:
                        results[key] = json.loads(value)
                        self.metrics.record_hit(namespace)
                    except json.JSONDecodeError:
                        results[key] = value
                        self.metrics.record_hit(namespace)
                else:
                    self.metrics.record_miss(namespace)

            return results

        except Exception as e:
            self.logger.error(f"Cache get_many failed: {e}")
            self.metrics.record_error(namespace, "get_many_error")
            return {}
        finally:
            if pipeline:
                try:
                    pipeline.reset()
                except Exception as cleanup_e:
                    self.logger.debug(f"Failed to cleanup pipeline: {cleanup_e}")

    async def set_many(
        self,
        mapping: dict[str, Any],
        namespace: str = "cache",
        ttl: int | None = None,
        data_type: str = "default",
    ) -> bool:
        """Set multiple values efficiently."""
        pipeline = None
        try:
            await self._ensure_client()

            if ttl is None:
                ttl = self._get_ttl(data_type)

            if not self.redis_client or not self.redis_client.client:
                return False
            pipeline = self.redis_client.client.pipeline()

            for key, value in mapping.items():
                namespaced_key = self.redis_client._get_namespaced_key(key, namespace)

                # Serialize value
                if isinstance(value, dict | list):
                    serialized_value = json.dumps(value, default=str)
                else:
                    serialized_value = str(value)

                pipeline.setex(namespaced_key, ttl, serialized_value)

            results = await pipeline.execute()

            for _ in results:
                self.metrics.record_set(namespace)

            return all(results)

        except Exception as e:
            self.logger.error(f"Cache set_many failed: {e}")
            self.metrics.record_error(namespace, "set_many_error")
            return False
        finally:
            if pipeline:
                try:
                    pipeline.reset()
                except Exception as cleanup_e:
                    self.logger.debug(f"Failed to cleanup pipeline: {cleanup_e}")

    # Distributed locking
    async def acquire_lock(
        self, resource: str, timeout: int | None = None, namespace: str = "locks"
    ) -> str | None:
        """Acquire distributed lock."""
        if timeout is None:
            timeout = self.lock_timeout

        lock_key = CacheKeys._build_key(namespace, "lock", resource)
        lock_value = f"{datetime.now(timezone.utc).isoformat()}_{id(self)}"

        try:
            await self._ensure_client()

            # Try to acquire lock
            if not self.redis_client or not self.redis_client.client:
                return None
            result = await self.redis_client.client.set(lock_key, lock_value, nx=True, ex=timeout)

            if result:
                self.logger.debug(f"Acquired lock for resource: {resource}")
                return lock_value

            return None

        except Exception as e:
            self.logger.error(f"Lock acquisition failed for {resource}: {e}")
            return None

    async def release_lock(self, resource: str, lock_value: str, namespace: str = "locks") -> bool:
        """Release distributed lock."""
        lock_key = CacheKeys._build_key(namespace, "lock", resource)

        try:
            await self._ensure_client()

            # Lua script for atomic lock release
            release_script = """
            if redis.call("GET", KEYS[1]) == ARGV[1] then
                return redis.call("DEL", KEYS[1])
            else
                return 0
            end
            """

            if not self.redis_client or not self.redis_client.client:
                return False
            result = await self.redis_client.client.eval(release_script, 1, lock_key, lock_value)

            if result:
                self.logger.debug(f"Released lock for resource: {resource}")

            return bool(result)

        except Exception as e:
            self.logger.error(f"Lock release failed for {resource}: {e}")
            return False

    async def with_lock(
        self,
        resource: str,
        func: Callable,
        *args,
        timeout: int | None = None,
        max_retries: int = 5,
        **kwargs,
    ):
        """Execute function with distributed lock."""
        for attempt in range(max_retries):
            lock_value = await self.acquire_lock(resource, timeout)

            if lock_value:
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    return result
                finally:
                    await self.release_lock(resource, lock_value)

            if attempt < max_retries - 1:
                await asyncio.sleep(self.lock_retry_delay * (2**attempt))

        raise CacheError(f"Could not acquire lock for resource: {resource}")

    # Cache warming strategies
    async def warm_cache(
        self,
        warming_functions: dict[str, Callable],
        batch_size: int = 10,
        delay_between_batches: float = 0.1,
    ):
        """Warm cache with preloaded data."""
        self.logger.info(f"Starting cache warming with {len(warming_functions)} functions")

        # Process warming functions in batches
        functions_list = list(warming_functions.items())

        for i in range(0, len(functions_list), batch_size):
            batch = functions_list[i : i + batch_size]

            # Execute batch concurrently
            tasks = []
            for key, func in batch:
                if asyncio.iscoroutinefunction(func):
                    tasks.append(self._warm_single_async(key, func))
                else:
                    tasks.append(self._warm_single_sync(key, func))

            await asyncio.gather(*tasks, return_exceptions=True)

            # Delay between batches to avoid overwhelming
            if delay_between_batches > 0 and i + batch_size < len(functions_list):
                await asyncio.sleep(delay_between_batches)

        self.logger.info("Cache warming completed")

    async def _warm_single_async(self, key: str, func: Callable):
        """Warm single cache entry with async function."""
        try:
            result = await func()
            if result is not None:
                await self.set(key, result, namespace="warm")
                self.logger.debug(f"Warmed cache key: {key}")
        except Exception as e:
            self.logger.warning(f"Failed to warm cache key {key}: {e}")

    async def _warm_single_sync(self, key: str, func: Callable):
        """Warm single cache entry with sync function."""
        try:
            result = func()
            if result is not None:
                await self.set(key, result, namespace="warm")
                self.logger.debug(f"Warmed cache key: {key}")
        except Exception as e:
            self.logger.warning(f"Failed to warm cache key {key}: {e}")

    # Pattern-based invalidation
    async def invalidate_pattern(self, pattern: str, namespace: str = "cache"):
        """Invalidate cache keys matching pattern."""
        try:
            await self._ensure_client()

            search_pattern = f"{namespace}:{pattern}*"
            if not self.redis_client or not self.redis_client.client:
                return
            keys = await self.redis_client.client.keys(search_pattern)

            if keys:
                await self.redis_client.client.delete(*keys)
                self.logger.info(f"Invalidated {len(keys)} keys matching pattern: {pattern}")

                for _ in keys:
                    self.metrics.record_delete(namespace)

        except Exception as e:
            self.logger.error(f"Pattern invalidation failed for {pattern}: {e}")
            self.metrics.record_error(namespace, "invalidation_error")

    # Health and diagnostics
    async def health_check(self) -> Any:
        """Comprehensive cache health check."""
        try:
            await self._ensure_client()

            # Basic connectivity test
            start_time = time.time()
            if not self.redis_client:
                raise Exception("Redis client not available")
            await self.redis_client.ping()
            ping_time = time.time() - start_time

            # Get Redis info
            if not self.redis_client:
                raise Exception("Redis client not available")
            info = await self.redis_client.info()

            # Test basic operations
            test_key = f"health_check_{int(time.time())}"
            await self.set(test_key, "test", ttl=10)
            test_get = await self.get(test_key)
            await self.delete(test_key)

            return {
                "status": "healthy",
                "ping_time": ping_time,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "operations_test": test_get == "test",
                "cache_stats": self.metrics.get_stats(),
            }

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e), "cache_stats": self.metrics.get_stats()}

    async def cleanup(self) -> None:
        """Clean up cache manager resources with proper connection handling."""
        async with self._connection_lock:
            try:
                self._shutdown_requested = True

                if self.redis_client:
                    self.logger.info("Cleaning up cache manager resources")

                    # Properly close the connection with timeout
                    if self.redis_client.client:
                        try:
                            # Use async context manager pattern for proper cleanup
                            await self.redis_client.disconnect()
                        except asyncio.TimeoutError:
                            self.logger.warning("Redis disconnection timed out")
                            # Force close connection with proper error handling
                            try:
                                if hasattr(self.redis_client.client, "close"):
                                    await self.redis_client.client.close()
                                if hasattr(self.redis_client.client, "wait_closed"):
                                    await self.redis_client.client.wait_closed()
                            except Exception as force_close_error:
                                self.logger.error(
                                    f"Failed to force close Redis client: {force_close_error}"
                                )
                        except Exception as e:
                            self.logger.warning(f"Error disconnecting Redis client: {e}")
                            # Attempt graceful cleanup even on error
                            try:
                                if hasattr(self.redis_client.client, "close"):
                                    await self.redis_client.client.close()
                            except (ConnectionError, OSError, AttributeError) as close_error:
                                self.logger.debug(f"Best effort cleanup failed: {close_error}")

                    self.redis_client = None
                    self.logger.info("Cache manager cleanup completed")

            except Exception as e:
                self.logger.error(f"Error during cache manager cleanup: {e}")
                # Don't re-raise - we're in cleanup mode
            finally:
                self._shutdown_requested = True

    async def shutdown(self) -> None:
        """Shutdown cache manager and release all resources."""
        try:
            self.logger.info("Shutting down cache manager")
            self._shutdown_requested = True

            # Invalidate any active locks with timeout
            try:
                await asyncio.wait_for(
                    self.invalidate_pattern("lock", namespace="locks"), timeout=5.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("Lock invalidation timed out during shutdown")
            except Exception as e:
                self.logger.warning(f"Failed to invalidate locks during shutdown: {e}")

            # Perform cleanup
            await self.cleanup()

            # Clear global instance
            global _cache_manager
            if _cache_manager is self:
                _cache_manager = None

            self.logger.info("Cache manager shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during cache manager shutdown: {e}")
            # Don't re-raise during shutdown

    def get_dependencies(self) -> list[str]:
        """Get list of required dependencies."""
        return ["CacheClientInterface"]

    def __del__(self):
        """Ensure cleanup on deletion."""
        if self.redis_client and self.redis_client.client:
            self.logger.warning("Cache manager deleted without proper cleanup - resources may leak")


# Global cache manager instance
_cache_manager: CacheManager | None = None


def get_cache_manager(
    redis_client: CacheClientInterface | None = None, config: Any | None = None
) -> CacheManager:
    """Get or create global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(redis_client, config)
    return _cache_manager


def create_cache_manager_factory(config: Any | None = None) -> Callable[[], CacheManager]:
    """Create a factory function for CacheManager to use with DI container."""

    def factory() -> CacheManager:
        # Create instance without hard-coded dependencies - DI will handle redis_client
        instance = CacheManager(redis_client=None, config=config)
        return instance

    return factory
