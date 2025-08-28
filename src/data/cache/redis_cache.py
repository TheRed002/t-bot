"""
Redis Cache Implementation

Concrete implementation of DataCacheInterface for Redis operations.
"""

import json
from typing import Any

import redis.asyncio as redis

from src.core.base.component import BaseComponent
from src.core.config import Config
from src.data.interfaces import DataCacheInterface


class RedisCache(BaseComponent, DataCacheInterface):
    """Redis cache implementation."""

    def __init__(self, config: Config):
        """Initialize Redis cache."""
        super().__init__()
        self.config = config
        self._redis_client: redis.Redis | None = None
        self._initialized = False

        # Redis configuration
        redis_config = getattr(config, "redis", {})
        self.redis_config = {
            "host": redis_config.get("host", "localhost"),
            "port": redis_config.get("port", 6379),
            "db": redis_config.get("db", 0),
            "password": redis_config.get("password"),
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

            # Test connection
            await self._redis_client.ping()
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
            cached_data = await self._redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)
            return None

        except Exception as e:
            self.logger.error(f"Redis get failed for key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set data in Redis cache."""
        if not self._redis_client:
            return

        try:
            serialized_value = json.dumps(value, default=str)
            if ttl:
                await self._redis_client.setex(key, ttl, serialized_value)
            else:
                await self._redis_client.set(key, serialized_value)

        except Exception as e:
            self.logger.error(f"Redis set failed for key {key}: {e}")

    async def delete(self, key: str) -> bool:
        """Delete data from Redis cache."""
        if not self._redis_client:
            return False

        try:
            result = await self._redis_client.delete(key)
            return result > 0

        except Exception as e:
            self.logger.error(f"Redis delete failed for key {key}: {e}")
            return False

    async def clear(self) -> None:
        """Clear all cached data."""
        if not self._redis_client:
            return

        try:
            await self._redis_client.flushdb()
            self.logger.info("Redis cache cleared")

        except Exception as e:
            self.logger.error(f"Redis clear failed: {e}")

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        if not self._redis_client:
            return False

        try:
            result = await self._redis_client.exists(key)
            return result > 0

        except Exception as e:
            self.logger.error(f"Redis exists failed for key {key}: {e}")
            return False

    async def health_check(self) -> dict[str, Any]:
        """Perform Redis cache health check."""
        if not self._redis_client:
            return {"status": "disabled", "component": "redis_cache"}

        try:
            await self._redis_client.ping()
            return {"status": "healthy", "component": "redis_cache"}

        except Exception as e:
            return {"status": "unhealthy", "component": "redis_cache", "error": str(e)}

    async def cleanup(self) -> None:
        """Cleanup Redis cache resources."""
        if self._redis_client:
            try:
                await self._redis_client.close()
                self._redis_client = None
                self.logger.info("Redis cache cleanup completed")
            except Exception as e:
                self.logger.error(f"Redis cleanup error: {e}")

        self._initialized = False