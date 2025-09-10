"""
Redis client for the trading bot framework.

This module provides async Redis client with connection pooling, serialization,
and utilities for real-time state management and caching.

CRITICAL: This module integrates with P-001 core framework and will be
used by all subsequent prompts for real-time state management.
"""

import asyncio
import json
import os
import re
from typing import Any

import redis.asyncio as redis

from src.core.base import BaseComponent

# Import core components from P-001
from src.core.exceptions import DataError

# Import utils from P-007A
from src.error_handling.decorators import with_circuit_breaker, with_retry

# Error handling is provided by decorators
from src.utils.constants import DEFAULT_VALUES, LIMITS, TIMEOUTS
from src.utils.decorators import time_execution


def substitute_env_vars(url: str) -> str:
    """
    Substitute environment variables in URLs with shell-style syntax.

    Supports patterns like:
    - ${VAR}: Simple substitution
    - ${VAR:default}: Substitution with default value

    Args:
        url: URL template with environment variables

    Returns:
        URL with environment variables substituted
    """

    def replace_var(match):
        var_expr = match.group(1)
        if ":" in var_expr:
            var_name, default_value = var_expr.split(":", 1)
        else:
            var_name = var_expr
            default_value = ""

        return os.environ.get(var_name, default_value)

    # Pattern to match ${VAR} or ${VAR:default}
    pattern = r"\$\{([^}]+)\}"
    return re.sub(pattern, replace_var, url)


class RedisClient(BaseComponent):
    """Async Redis client with utilities for trading bot data."""

    def __init__(self, config_or_url: Any | str, *, auto_close: bool = False) -> None:
        super().__init__()  # Initialize BaseComponent
        # Accept both config object and direct URL
        if hasattr(config_or_url, "redis"):
            self.redis_url = (
                getattr(config_or_url.redis, "url", DEFAULT_VALUES["redis_default_url"])
                if hasattr(config_or_url, "redis")
                else DEFAULT_VALUES["redis_default_url"]
            )
            self.config = config_or_url
        elif isinstance(config_or_url, str):
            self.redis_url = config_or_url
            self.config = None
        else:
            self.redis_url = DEFAULT_VALUES["redis_default_url"]
            self.config = None

        # Apply environment variable substitution to Redis URL
        self.redis_url = substitute_env_vars(str(self.redis_url))

        self.client: redis.Redis | None = None
        # Use utils constants for default TTL
        self._default_ttl = TIMEOUTS.get("REDIS_DEFAULT_TTL", 3600)
        self.auto_close = auto_close
        self._close_lock = asyncio.Lock()

        # Backpressure handling - Use constants from utils
        max_concurrent_ops = LIMITS.get("REDIS_MAX_CONCURRENT_OPS", 100)
        self._operation_semaphore = asyncio.Semaphore(max_concurrent_ops)
        self._operation_queue_size = 0
        self._max_queue_size = LIMITS.get("REDIS_MAX_QUEUE_SIZE", 1000)

        # Connection timeout and heartbeat - Use constants from utils
        self._last_heartbeat: float | None = None
        self._heartbeat_interval = TIMEOUTS.get("REDIS_HEARTBEAT_INTERVAL", 30)
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._connection_timeout = TIMEOUTS.get("REDIS_CONNECTION_TIMEOUT", 10)

    @time_execution
    @with_retry(max_attempts=3)
    @with_circuit_breaker(failure_threshold=3, recovery_timeout=30)
    async def connect(self) -> None:
        """Connect to Redis with proper configuration."""
        # Use constants for Redis connection configuration
        max_connections = LIMITS.get("REDIS_MAX_CONNECTIONS", 100)
        health_check_interval = TIMEOUTS.get("REDIS_HEALTH_CHECK_INTERVAL", 30)
        socket_connect_timeout = TIMEOUTS.get("REDIS_SOCKET_CONNECT_TIMEOUT", 5)
        socket_timeout = TIMEOUTS.get("REDIS_SOCKET_TIMEOUT", 10)

        # Validate URL before attempting connection
        if "}" in self.redis_url and "${" not in self.redis_url:
            # URL appears malformed - try to fix common issues
            self.logger.warning(f"Malformed Redis URL detected: {self.redis_url}")
            # Use configured fallback URL
            fallback_url = DEFAULT_VALUES.get("redis_fallback_url", "redis://localhost:6379/0")
            self.redis_url = fallback_url
            self.logger.info(f"Using fallback Redis URL: {self.redis_url}")

        self.client = redis.Redis.from_url(
            self.redis_url,
            decode_responses=True,
            max_connections=max_connections,
            retry_on_timeout=True,
            health_check_interval=health_check_interval,
            socket_connect_timeout=socket_connect_timeout,
            socket_timeout=socket_timeout,
            retry_on_error=[redis.BusyLoadingError, redis.ConnectionError, redis.TimeoutError],
            socket_keepalive=True,
            socket_keepalive_options={
                "TCP_KEEPIDLE": 1,
                "TCP_KEEPINTVL": 3,
                "TCP_KEEPCNT": 5,
            },
        )

        # Test connection with timeout
        await asyncio.wait_for(self.client.ping(), timeout=5.0)
        self._last_heartbeat = asyncio.get_event_loop().time()

        # Start heartbeat monitoring
        self._start_heartbeat_monitoring()

        self.logger.info("Redis connection established")

    def _start_heartbeat_monitoring(self) -> None:
        """Start heartbeat monitoring task."""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

    async def _heartbeat_monitor(self) -> None:
        """Monitor connection with periodic heartbeat."""
        while self.client is not None:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                if self.client:
                    await asyncio.wait_for(self.client.ping(), timeout=5.0)
                    self._last_heartbeat = asyncio.get_event_loop().time()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"Heartbeat failed: {e}")
                # Connection is unhealthy - will be handled by _ensure_connected
                break

    async def _ensure_connected(self) -> None:
        if self.client is None:
            await self.connect()
        else:
            # Check if heartbeat is recent enough
            current_time = asyncio.get_event_loop().time()
            if (
                self._last_heartbeat
                and current_time - self._last_heartbeat > self._heartbeat_interval * 2
            ):
                self.logger.warning("Redis connection heartbeat stale, reconnecting...")
                await self._cleanup_connection()
                await self.connect()
            else:
                # Test connection health periodically with timeout
                try:
                    await asyncio.wait_for(self.client.ping(), timeout=2.0)
                    self._last_heartbeat = current_time
                except (asyncio.TimeoutError, Exception):
                    self.logger.warning("Redis connection unhealthy, reconnecting...")
                    await self._cleanup_connection()
                    await self.connect()

    async def _cleanup_connection(self) -> None:
        """Clean up connection resources."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        self.client = None

    async def _with_backpressure(self, operation):
        """Execute Redis operation with backpressure control."""
        # Check queue size to prevent memory issues
        if self._operation_queue_size >= self._max_queue_size:
            raise DataError("Redis operation queue full - backpressure applied")

        self._operation_queue_size += 1
        try:
            async with self._operation_semaphore:
                # Apply operation timeout to prevent hanging
                return await asyncio.wait_for(operation(), timeout=self._connection_timeout)
        except asyncio.TimeoutError as e:
            self.logger.error(f"Redis operation timed out after {self._connection_timeout}s")
            raise DataError(f"Redis operation timed out after {self._connection_timeout}s") from e
        finally:
            self._operation_queue_size -= 1

    async def _maybe_autoclose(self) -> None:
        if self.auto_close and self.client is not None:
            async with self._close_lock:
                # Double-check pattern to avoid race conditions
                if self.client is not None:
                    try:
                        # Check if aclose method exists (newer redis versions)
                        if hasattr(self.client, "aclose") and callable(self.client.aclose):
                            await self.client.aclose()
                        else:
                            # Fall back to close() for older versions
                            await self.client.close()
                    except Exception as e:
                        self.logger.warning(f"Error during auto-close: {e}")
                    finally:
                        self.client = None

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        client = None
        try:
            # Cancel heartbeat monitoring first
            if self._heartbeat_task and not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass

            client = self.client
            if client:
                # Check if aclose method exists (newer redis versions)
                if hasattr(client, "aclose") and callable(client.aclose):
                    await asyncio.wait_for(client.aclose(), timeout=5.0)
                else:
                    # Fall back to close() for older versions
                    await asyncio.wait_for(client.close(), timeout=5.0)
                self.logger.info("Redis connection closed")
        except asyncio.TimeoutError:
            self.logger.warning("Redis disconnect timed out")
        except Exception as e:
            self.logger.warning(f"Error during disconnect: {e}")
        finally:
            # Ensure references are cleared even if close fails
            self.client = None
            self._heartbeat_task = None

    def _get_namespaced_key(self, key: str, namespace: str = "trading_bot") -> str:
        """Get namespaced key for organization."""
        return f"{namespace}:{key}"

    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def set(
        self, key: str, value: Any, ttl: int | None = None, namespace: str = "trading_bot"
    ) -> bool:
        """Set a key-value pair with optional TTL."""

        async def _set_operation():
            await self._ensure_connected()
            namespaced_key = self._get_namespaced_key(key, namespace)

            # Serialize value
            if isinstance(value, dict | list):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)

            # Set with TTL
            if ttl is not None:
                result = await self.client.setex(namespaced_key, ttl, serialized_value)
            else:
                result = await self.client.setex(
                    namespaced_key, self._default_ttl, serialized_value
                )
            return result

        try:
            result = await self._with_backpressure(_set_operation)
            return result

        except Exception as e:
            self.logger.error("Redis set operation failed", key=key, error=str(e))
            raise DataError(f"Redis set operation failed: {e!s}")
        finally:
            await self._maybe_autoclose()

    async def get(self, key: str, namespace: str = "trading_bot") -> Any | None:
        """Get a value by key."""

        async def _get_operation():
            await self._ensure_connected()
            namespaced_key = self._get_namespaced_key(key, namespace)
            value = await self.client.get(namespaced_key)

            if value is None:
                return None

            # Try to deserialize JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

        try:
            return await self._with_backpressure(_get_operation)

        except Exception as e:
            self.logger.error("Redis get operation failed", key=key, error=str(e))
            raise DataError(f"Redis get operation failed: {e!s}")
        finally:
            await self._maybe_autoclose()

    async def delete(self, key: str, namespace: str = "trading_bot") -> bool:
        """Delete a key."""
        try:
            await self._ensure_connected()
            namespaced_key = self._get_namespaced_key(key, namespace)
            result = await self.client.delete(namespaced_key)
            return result > 0

        except Exception as e:
            self.logger.error("Redis delete operation failed", key=key, error=str(e))
            raise DataError(f"Redis delete operation failed: {e!s}")
        finally:
            await self._maybe_autoclose()

    async def exists(self, key: str, namespace: str = "trading_bot") -> bool:
        """Check if a key exists."""
        try:
            await self._ensure_connected()
            namespaced_key = self._get_namespaced_key(key, namespace)
            result = await self.client.exists(namespaced_key)
            return result > 0

        except Exception as e:
            self.logger.error("Redis exists operation failed", key=key, error=str(e))
            raise DataError(f"Redis exists operation failed: {e!s}")
        finally:
            await self._maybe_autoclose()

    async def expire(self, key: str, ttl: int, namespace: str = "trading_bot") -> bool:
        """Set expiration for a key."""
        try:
            await self._ensure_connected()
            namespaced_key = self._get_namespaced_key(key, namespace)
            result = await self.client.expire(namespaced_key, ttl)
            return result

        except Exception as e:
            self.logger.error("Redis expire operation failed", key=key, error=str(e))
            raise DataError(f"Redis expire operation failed: {e!s}")
        finally:
            await self._maybe_autoclose()

    async def ttl(self, key: str, namespace: str = "trading_bot") -> int:
        """Get TTL for a key."""
        try:
            await self._ensure_connected()
            namespaced_key = self._get_namespaced_key(key, namespace)
            result = await self.client.ttl(namespaced_key)
            return result

        except Exception as e:
            self.logger.error("Redis TTL operation failed", key=key, error=str(e))
            raise DataError(f"Redis TTL operation failed: {e!s}")
        finally:
            await self._maybe_autoclose()

    # Hash operations
    async def hset(self, key: str, field: str, value: Any, namespace: str = "trading_bot") -> bool:
        """Set a field in a hash."""
        try:
            await self._ensure_connected()
            namespaced_key = self._get_namespaced_key(key, namespace)

            # Serialize value
            if isinstance(value, dict | list):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)

            result = await self.client.hset(namespaced_key, field, serialized_value)
            return result >= 0

        except Exception as e:
            self.logger.error("Redis hset operation failed", key=key, field=field, error=str(e))
            raise DataError(f"Redis hset operation failed: {e!s}")
        finally:
            await self._maybe_autoclose()

    async def hget(self, key: str, field: str, namespace: str = "trading_bot") -> Any | None:
        """Get a field from a hash."""
        try:
            await self._ensure_connected()
            namespaced_key = self._get_namespaced_key(key, namespace)
            value = await self.client.hget(namespaced_key, field)

            if value is None:
                return None

            # Try to deserialize JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

        except Exception as e:
            self.logger.error("Redis hget operation failed", key=key, field=field, error=str(e))
            raise DataError(f"Redis hget operation failed: {e!s}")
        finally:
            await self._maybe_autoclose()

    async def hgetall(self, key: str, namespace: str = "trading_bot") -> dict[str, Any]:
        """Get all fields from a hash."""
        try:
            await self._ensure_connected()
            namespaced_key = self._get_namespaced_key(key, namespace)
            result = await self.client.hgetall(namespaced_key)

            # Deserialize values
            deserialized = {}
            for field, value in result.items():
                try:
                    deserialized[field] = json.loads(value)
                except json.JSONDecodeError:
                    deserialized[field] = value

            return deserialized

        except Exception as e:
            self.logger.error("Redis hgetall operation failed", key=key, error=str(e))
            raise DataError(f"Redis hgetall operation failed: {e!s}")
        finally:
            await self._maybe_autoclose()

    async def hdel(self, key: str, field: str, namespace: str = "trading_bot") -> bool:
        """Delete a field from a hash."""
        try:
            await self._ensure_connected()
            namespaced_key = self._get_namespaced_key(key, namespace)
            result = await self.client.hdel(namespaced_key, field)
            return result > 0

        except Exception as e:
            self.logger.error("Redis hdel operation failed", key=key, field=field, error=str(e))
            raise DataError(f"Redis hdel operation failed: {e!s}")
        finally:
            await self._maybe_autoclose()

    # List operations
    async def lpush(self, key: str, value: Any, namespace: str = "trading_bot") -> int:
        """Push a value to the left of a list."""
        try:
            await self._ensure_connected()
            namespaced_key = self._get_namespaced_key(key, namespace)

            # Serialize value
            if isinstance(value, dict | list):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)

            result = await self.client.lpush(namespaced_key, serialized_value)
            return result

        except Exception as e:
            self.logger.error("Redis lpush operation failed", key=key, error=str(e))
            raise DataError(f"Redis lpush operation failed: {e!s}")
        finally:
            await self._maybe_autoclose()

    async def rpush(self, key: str, value: Any, namespace: str = "trading_bot") -> int:
        """Push a value to the right of a list."""
        try:
            await self._ensure_connected()
            namespaced_key = self._get_namespaced_key(key, namespace)

            # Serialize value
            if isinstance(value, dict | list):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)

            result = await self.client.rpush(namespaced_key, serialized_value)
            return result

        except Exception as e:
            self.logger.error("Redis rpush operation failed", key=key, error=str(e))
            raise DataError(f"Redis rpush operation failed: {e!s}")
        finally:
            await self._maybe_autoclose()

    async def lrange(
        self, key: str, start: int = 0, end: int = -1, namespace: str = "trading_bot"
    ) -> list[Any]:
        """Get a range of elements from a list."""
        try:
            await self._ensure_connected()
            namespaced_key = self._get_namespaced_key(key, namespace)
            result = await self.client.lrange(namespaced_key, start, end)

            # Deserialize values
            deserialized = []
            for value in result:
                try:
                    deserialized.append(json.loads(value))
                except json.JSONDecodeError:
                    deserialized.append(value)

            return deserialized

        except Exception as e:
            self.logger.error("Redis lrange operation failed", key=key, error=str(e))
            raise DataError(f"Redis lrange operation failed: {e!s}")
        finally:
            await self._maybe_autoclose()

    # Set operations
    async def sadd(self, key: str, value: Any, namespace: str = "trading_bot") -> bool:
        """Add a value to a set."""
        try:
            await self._ensure_connected()
            namespaced_key = self._get_namespaced_key(key, namespace)

            # Serialize value
            if isinstance(value, dict | list):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)

            result = await self.client.sadd(namespaced_key, serialized_value)
            return result > 0

        except Exception as e:
            self.logger.error("Redis sadd operation failed", key=key, error=str(e))
            raise DataError(f"Redis sadd operation failed: {e!s}")
        finally:
            await self._maybe_autoclose()

    async def smembers(self, key: str, namespace: str = "trading_bot") -> list[Any]:
        """Get all members of a set."""
        try:
            await self._ensure_connected()
            namespaced_key = self._get_namespaced_key(key, namespace)
            result = await self.client.smembers(namespaced_key)

            # Deserialize values
            deserialized = []
            for value in result:
                try:
                    deserialized.append(json.loads(value))
                except json.JSONDecodeError:
                    deserialized.append(value)

            return list(deserialized)

        except Exception as e:
            self.logger.error("Redis smembers operation failed", key=key, error=str(e))
            raise DataError(f"Redis smembers operation failed: {e!s}")
        finally:
            await self._maybe_autoclose()

    # Trading-specific utilities
    async def store_market_data(self, symbol: str, data: dict[str, Any], ttl: int = 300) -> bool:
        """Store market data with TTL."""
        key = f"market_data:{symbol}"
        return await self.set(key, data, ttl, "market_data")

    async def get_market_data(self, symbol: str) -> dict[str, Any] | None:
        """Get market data for a symbol."""
        key = f"market_data:{symbol}"
        return await self.get(key, "market_data")

    async def store_position(self, bot_id: str, position: dict[str, Any]) -> bool:
        """Store bot position data."""
        return await self.hset("positions", bot_id, position, "bot_state")

    async def get_position(self, bot_id: str) -> dict[str, Any] | None:
        """Get bot position data."""
        return await self.hget("positions", bot_id, "bot_state")

    async def store_balance(self, user_id: str, exchange: str, balance: dict[str, Any]) -> bool:
        """Store user balance data."""
        key = f"balance:{user_id}:{exchange}"
        return await self.set(key, balance, 3600, "balances")

    async def get_balance(self, user_id: str, exchange: str) -> dict[str, Any] | None:
        """Get user balance data."""
        key = f"balance:{user_id}:{exchange}"
        return await self.get(key, "balances")

    async def store_cache(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Store cached data."""
        return await self.set(key, value, ttl, "cache")

    async def get_cache(self, key: str) -> Any | None:
        """Get cached data."""
        return await self.get(key, "cache")

    # Health check and basic operations
    async def ping(self) -> bool:
        """Ping Redis server."""
        try:
            await self._ensure_connected()
            result = await self.client.ping()
            return result
        except Exception as e:
            self.logger.error("Redis ping failed", error=str(e))
            raise DataError(f"Redis ping failed: {e!s}")
        finally:
            await self._maybe_autoclose()

    async def info(self) -> dict[str, Any]:
        """Get Redis info."""
        try:
            await self._ensure_connected()
            info = await self.client.info()
            return info
        except Exception as e:
            self.logger.error("Redis info failed", error=str(e))
            raise DataError(f"Redis info failed: {e!s}")
        finally:
            await self._maybe_autoclose()

    async def health_check(self) -> bool:
        """Check Redis health."""
        try:
            await self.ping()
            return True
        except Exception as e:
            self.logger.error("Redis health check failed", error=str(e))
            return False
