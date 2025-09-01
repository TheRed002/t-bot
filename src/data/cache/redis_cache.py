"""
Redis Cache Implementation

Concrete implementation of DataCacheInterface for Redis operations.
"""

import asyncio
from typing import Any

import redis.asyncio as redis

from src.core.base.component import BaseComponent
from src.core.config import Config
from src.data.interfaces import DataCacheInterface
from src.utils.cache_utilities import CacheSerializationUtils


class RedisCache(BaseComponent, DataCacheInterface):
    """Redis cache implementation."""

    def __init__(self, config: Config):
        """Initialize Redis cache."""
        super().__init__()
        self.config = config
        self._redis_client: redis.Redis | None = None
        self._initialized = False

        # Redis configuration with environment variable fallback
        redis_config = getattr(config, "redis", {})
        import os

        self.redis_config = {
            "host": redis_config.get("host", os.environ.get("REDIS_HOST", "127.0.0.1")),
            "port": redis_config.get("port", int(os.environ.get("REDIS_PORT", "6379"))),
            "db": redis_config.get("db", int(os.environ.get("REDIS_DB", "0"))),
            "password": redis_config.get("password") or os.environ.get("REDIS_PASSWORD"),
            "max_connections": redis_config.get("max_connections", 10),
        }

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        if self._initialized:
            return

        try:
            self._redis_client = redis.Redis(
                host=self.redis_config["host"],
                port=self.redis_config["port"],
                db=self.redis_config["db"],
                password=self.redis_config["password"],
                max_connections=self.redis_config["max_connections"],
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )

            # Test connection with timeout
            await asyncio.wait_for(self._redis_client.ping(), timeout=5.0)
            self.logger.info("Redis cache connection established")
            self._initialized = True

        except Exception as e:
            self.logger.warning(f"Redis connection failed, cache disabled: {e}")
            self._redis_client = None

    async def get(self, key: str) -> Any | None:
        """Get data from Redis cache."""
        if not self._redis_client:
            return None

        try:
            # Add timeout to Redis get operation
            cached_data = await asyncio.wait_for(self._redis_client.get(key), timeout=3.0)
            if cached_data:
                return CacheSerializationUtils.deserialize_json(cached_data)
            return None

        except Exception as e:
            self.logger.error(f"Redis get failed for key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set data in Redis cache."""
        if not self._redis_client:
            return

        try:
            serialized_value = CacheSerializationUtils.serialize_json(value)
            if ttl:
                await asyncio.wait_for(
                    self._redis_client.setex(key, ttl, serialized_value), timeout=3.0
                )
            else:
                await asyncio.wait_for(self._redis_client.set(key, serialized_value), timeout=3.0)

        except Exception as e:
            self.logger.error(f"Redis set failed for key {key}: {e}")

    async def delete(self, key: str) -> bool:
        """Delete data from Redis cache."""
        if not self._redis_client:
            return False

        try:
            result = await asyncio.wait_for(self._redis_client.delete(key), timeout=3.0)
            return result > 0

        except Exception as e:
            self.logger.error(f"Redis delete failed for key {key}: {e}")
            return False

    async def clear(self) -> None:
        """Clear all cached data."""
        if not self._redis_client:
            return

        try:
            await asyncio.wait_for(self._redis_client.flushdb(), timeout=10.0)
            self.logger.info("Redis cache cleared")

        except Exception as e:
            self.logger.error(f"Redis clear failed: {e}")

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        if not self._redis_client:
            return False

        try:
            result = await asyncio.wait_for(self._redis_client.exists(key), timeout=3.0)
            return result > 0

        except Exception as e:
            self.logger.error(f"Redis exists failed for key {key}: {e}")
            return False

    async def health_check(self) -> dict[str, Any]:
        """Perform Redis cache health check."""
        if not self._redis_client:
            return {
                "status": "unhealthy",
                "component": "redis_cache",
                "message": "Redis client not initialized",
            }

        try:
            await asyncio.wait_for(self._redis_client.ping(), timeout=3.0)
            return {
                "status": "healthy",
                "component": "redis_cache",
                "message": "Redis connection healthy",
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "redis_cache",
                "error": str(e),
                "message": f"Redis connection failed: {e}",
            }

    async def cleanup(self) -> None:
        """Cleanup Redis cache resources."""
        redis_client = None
        try:
            if self._redis_client:
                redis_client = self._redis_client
                self._redis_client = None
                await asyncio.wait_for(redis_client.close(), timeout=5.0)
                self.logger.info("Redis cache cleanup completed")
        except Exception as e:
            self.logger.error(f"Redis cleanup error: {e}")
        finally:
            if redis_client:
                try:
                    if hasattr(redis_client, "aclose") and not redis_client.connection_pool.closed:
                        await asyncio.wait_for(redis_client.aclose(), timeout=2.0)
                    elif not redis_client.connection_pool.closed:
                        await asyncio.wait_for(redis_client.close(), timeout=2.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Redis cleanup timeout")
                except Exception as e:
                    self.logger.warning(f"Failed to close Redis client in finally block: {e}")
            self._initialized = False
