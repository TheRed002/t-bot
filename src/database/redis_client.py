"""
Redis client for the trading bot framework.

This module provides async Redis client with connection pooling, serialization,
and utilities for real-time state management and caching.

CRITICAL: This module integrates with P-001 core framework and will be
used by all subsequent prompts for real-time state management.
"""

import json
from typing import Any

import redis.asyncio as redis

# Import core components from P-001
from src.core.exceptions import DataError, DataSourceError
from src.core.logging import get_logger
from src.utils.constants import TIMEOUTS

# Import utils from P-007A
from src.utils.formatters import format_api_response

logger = get_logger(__name__)


class RedisClient:
    """Async Redis client with utilities for trading bot data."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client: redis.Redis | None = None
        # Use utils constants for default TTL
        self._default_ttl = getattr(TIMEOUTS, "REDIS_DEFAULT_TTL", 3600)

    async def connect(self) -> None:
        """Connect to Redis with proper configuration."""
        try:
            self.client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                max_connections=100,
                retry_on_timeout=True,
                health_check_interval=30,
            )

            # Test connection
            await self.client.ping()
            logger.info("Redis connection established")

        except Exception as e:
            logger.error("Redis connection failed", error=str(e))
            raise DataSourceError(f"Redis connection failed: {e!s}")

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.client:
            await self.client.close()
            logger.info("Redis connection closed")

    def _get_namespaced_key(self, key: str, namespace: str = "trading_bot") -> str:
        """Get namespaced key for organization."""
        return f"{namespace}:{key}"

    async def set(
        self, key: str, value: Any, ttl: int | None = None, namespace: str = "trading_bot"
    ) -> bool:
        """Set a key-value pair with optional TTL."""
        try:
            namespaced_key = self._get_namespaced_key(key, namespace)

            # Serialize value
            if isinstance(value, (dict, list)):
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

        except Exception as e:
            logger.error("Redis set operation failed", key=key, error=str(e))
            raise DataError(f"Redis set operation failed: {e!s}")

    async def get(self, key: str, namespace: str = "trading_bot") -> Any | None:
        """Get a value by key."""
        try:
            namespaced_key = self._get_namespaced_key(key, namespace)
            value = await self.client.get(namespaced_key)

            if value is None:
                return None

            # Try to deserialize JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

        except Exception as e:
            logger.error("Redis get operation failed", key=key, error=str(e))
            raise DataError(f"Redis get operation failed: {e!s}")

    async def delete(self, key: str, namespace: str = "trading_bot") -> bool:
        """Delete a key."""
        try:
            namespaced_key = self._get_namespaced_key(key, namespace)
            result = await self.client.delete(namespaced_key)
            return result > 0

        except Exception as e:
            logger.error("Redis delete operation failed", key=key, error=str(e))
            raise DataError(f"Redis delete operation failed: {e!s}")

    async def exists(self, key: str, namespace: str = "trading_bot") -> bool:
        """Check if a key exists."""
        try:
            namespaced_key = self._get_namespaced_key(key, namespace)
            result = await self.client.exists(namespaced_key)
            return result > 0

        except Exception as e:
            logger.error("Redis exists operation failed", key=key, error=str(e))
            raise DataError(f"Redis exists operation failed: {e!s}")

    async def expire(self, key: str, ttl: int, namespace: str = "trading_bot") -> bool:
        """Set expiration for a key."""
        try:
            namespaced_key = self._get_namespaced_key(key, namespace)
            result = await self.client.expire(namespaced_key, ttl)
            return result

        except Exception as e:
            logger.error("Redis expire operation failed", key=key, error=str(e))
            raise DataError(f"Redis expire operation failed: {e!s}")

    async def ttl(self, key: str, namespace: str = "trading_bot") -> int:
        """Get TTL for a key."""
        try:
            namespaced_key = self._get_namespaced_key(key, namespace)
            result = await self.client.ttl(namespaced_key)
            return result

        except Exception as e:
            logger.error("Redis TTL operation failed", key=key, error=str(e))
            raise DataError(f"Redis TTL operation failed: {e!s}")

    # Hash operations
    async def hset(self, key: str, field: str, value: Any, namespace: str = "trading_bot") -> bool:
        """Set a field in a hash."""
        try:
            namespaced_key = self._get_namespaced_key(key, namespace)

            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)

            result = await self.client.hset(namespaced_key, field, serialized_value)
            return result >= 0

        except Exception as e:
            logger.error("Redis hset operation failed", key=key, field=field, error=str(e))
            raise DataError(f"Redis hset operation failed: {e!s}")

    async def hget(self, key: str, field: str, namespace: str = "trading_bot") -> Any | None:
        """Get a field from a hash."""
        try:
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
            logger.error("Redis hget operation failed", key=key, field=field, error=str(e))
            raise DataError(f"Redis hget operation failed: {e!s}")

    async def hgetall(self, key: str, namespace: str = "trading_bot") -> dict[str, Any]:
        """Get all fields from a hash."""
        try:
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
            logger.error("Redis hgetall operation failed", key=key, error=str(e))
            raise DataError(f"Redis hgetall operation failed: {e!s}")

    async def hdel(self, key: str, field: str, namespace: str = "trading_bot") -> bool:
        """Delete a field from a hash."""
        try:
            namespaced_key = self._get_namespaced_key(key, namespace)
            result = await self.client.hdel(namespaced_key, field)
            return result > 0

        except Exception as e:
            logger.error("Redis hdel operation failed", key=key, field=field, error=str(e))
            raise DataError(f"Redis hdel operation failed: {e!s}")

    # List operations
    async def lpush(self, key: str, value: Any, namespace: str = "trading_bot") -> int:
        """Push a value to the left of a list."""
        try:
            namespaced_key = self._get_namespaced_key(key, namespace)

            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)

            result = await self.client.lpush(namespaced_key, serialized_value)
            return result

        except Exception as e:
            logger.error("Redis lpush operation failed", key=key, error=str(e))
            raise DataError(f"Redis lpush operation failed: {e!s}")

    async def rpush(self, key: str, value: Any, namespace: str = "trading_bot") -> int:
        """Push a value to the right of a list."""
        try:
            namespaced_key = self._get_namespaced_key(key, namespace)

            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)

            result = await self.client.rpush(namespaced_key, serialized_value)
            return result

        except Exception as e:
            logger.error("Redis rpush operation failed", key=key, error=str(e))
            raise DataError(f"Redis rpush operation failed: {e!s}")

    async def lrange(
        self, key: str, start: int = 0, end: int = -1, namespace: str = "trading_bot"
    ) -> list[Any]:
        """Get a range of elements from a list."""
        try:
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
            logger.error("Redis lrange operation failed", key=key, error=str(e))
            raise DataError(f"Redis lrange operation failed: {e!s}")

    # Set operations
    async def sadd(self, key: str, value: Any, namespace: str = "trading_bot") -> bool:
        """Add a value to a set."""
        try:
            namespaced_key = self._get_namespaced_key(key, namespace)

            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)

            result = await self.client.sadd(namespaced_key, serialized_value)
            return result > 0

        except Exception as e:
            logger.error("Redis sadd operation failed", key=key, error=str(e))
            raise DataError(f"Redis sadd operation failed: {e!s}")

    async def smembers(self, key: str, namespace: str = "trading_bot") -> list[Any]:
        """Get all members of a set."""
        try:
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
            logger.error("Redis smembers operation failed", key=key, error=str(e))
            raise DataError(f"Redis smembers operation failed: {e!s}")

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
        key = f"position:{bot_id}"
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

    # Health check
    async def health_check(self) -> bool:
        """Check Redis health."""
        try:
            await self.client.ping()
            return True
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            return False

    # TODO: Remove in production - Debug functions
    async def debug_info(self) -> dict[str, Any]:
        """Get debug information about Redis."""
        try:
            info = await self.client.info()
            debug_data = {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0),
            }
            # Use utils formatter for consistent API response
            return format_api_response(
                debug_data, success=True, message="Redis debug info retrieved"
            )
        except Exception as e:
            return format_api_response(
                {}, success=False, message=f"Failed to get Redis info: {e!s}"
            )
