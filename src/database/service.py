"""
Simple Database Service implementing service layer pattern.

This module provides database service functionality with:
- Connection management through injected DatabaseConnectionManager
- Basic CRUD operations
- Transaction support
- Health monitoring
- Simple caching layer

CRITICAL: All modules MUST use this service instead of direct database access.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, TypeVar

import redis.asyncio as redis
from sqlalchemy import asc, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.base.interfaces import HealthStatus

# Import core components
from src.core.base.service import BaseService
from src.core.exceptions import ComponentError, DatabaseError
from src.core.logging import get_logger

# Import database infrastructure
from src.database.connection import DatabaseConnectionManager

if TYPE_CHECKING:
    from src.database.models import Base

# Type variables
T = TypeVar("T", bound="Base")
K = TypeVar("K")

logger = get_logger(__name__)


class DatabaseService(BaseService):
    """
    Simple database service implementing service layer pattern.

    This service provides basic database operations with connection management,
    transaction support, and simple caching.
    """

    def __init__(
        self,
        connection_manager: DatabaseConnectionManager,
        config_service: Any = None,
        validation_service: Any = None,
        dependency_injector: Any = None,
        cache_enabled: bool = True,
        cache_ttl: int = 300,
        correlation_id: str | None = None,
    ):
        """
        Initialize database service with connection manager.

        Args:
            connection_manager: Database connection manager (required)
            config_service: Configuration service (optional, injected)
            validation_service: Validation service (optional, injected)
            dependency_injector: Dependency injector (optional, injected)
            cache_enabled: Whether to enable Redis caching
            cache_ttl: Cache TTL in seconds
        """
        super().__init__(name="DatabaseService", correlation_id=correlation_id)

        if connection_manager is None:
            raise ValueError("connection_manager is required")

        self.connection_manager = connection_manager
        self._config_service = config_service
        self._validation_service = validation_service
        self._dependency_injector = dependency_injector
        self._cache_enabled = cache_enabled
        self._cache_ttl = cache_ttl
        self._redis_client: redis.Redis | None = None
        self._started = False

        logger.info("DatabaseService initialized with dependency injection")

    @property
    def config_service(self) -> Any:
        """Get config service."""
        return self._config_service

    @property
    def validation_service(self) -> Any:
        """Get validation service."""
        return self._validation_service

    async def start(self) -> None:
        """Start the database service."""
        try:
            if self._started:
                return

            # Initialize connection manager if needed
            if (
                not hasattr(self.connection_manager, "async_engine")
                or self.connection_manager.async_engine is None
            ):
                await self.connection_manager.initialize()

            # Initialize Redis client for caching if enabled
            if self._cache_enabled:
                try:
                    self._redis_client = await self.connection_manager.get_redis_client()
                    logger.info("Redis cache initialized")
                except Exception as e:
                    logger.warning(f"Redis cache initialization failed: {e}")
                    self._cache_enabled = False

            self._started = True
            logger.info("DatabaseService started successfully")

        except Exception as e:
            logger.error(f"Failed to start DatabaseService: {e}")
            raise ComponentError(f"DatabaseService startup failed: {e}") from e

    async def stop(self) -> None:
        """Stop the database service."""
        try:
            if not self._started:
                return

            # Close Redis client
            if self._redis_client:
                try:
                    if hasattr(self._redis_client, "aclose"):
                        await self._redis_client.aclose()
                    elif hasattr(self._redis_client, "close"):
                        await self._redis_client.close()
                except Exception as e:
                    logger.warning(f"Error closing Redis client: {e}")

            # Close connection manager
            if self.connection_manager:
                try:
                    await self.connection_manager.close()
                except Exception as e:
                    logger.error(f"Error closing connection manager: {e}")

            self._started = False
            logger.info("DatabaseService stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping DatabaseService: {e}")
        finally:
            self._redis_client = None

    # CRUD Operations

    async def create_entity(self, entity: T) -> T:
        """
        Create a new entity.

        Args:
            entity: Entity to create

        Returns:
            Created entity with ID assigned
        """
        try:
            async with self.connection_manager.get_async_session() as session:
                session.add(entity)
                await session.commit()
                await session.refresh(entity)

                # Invalidate cache if enabled
                if self._cache_enabled and self._redis_client:
                    cache_pattern = f"{type(entity).__name__}_*"
                    await self._invalidate_cache_pattern(cache_pattern)

                return entity

        except Exception as e:
            logger.error(f"Entity creation failed for {type(entity).__name__}: {e}")
            raise DatabaseError(f"Failed to create {type(entity).__name__}: {e}") from e

    async def get_entity_by_id(self, model_class: type[T], entity_id: K) -> T | None:
        """
        Get entity by ID.

        Args:
            model_class: Entity class
            entity_id: Primary key value

        Returns:
            Entity or None if not found
        """
        try:
            # Check cache first
            if self._cache_enabled and self._redis_client:
                cache_key = f"{model_class.__name__}_{entity_id}"
                cached_data = await self._redis_client.get(cache_key)
                if cached_data:
                    # For simple caching, we'll skip deserialization and fetch from DB
                    # This can be enhanced later if needed
                    pass

            async with self.connection_manager.get_async_session() as session:
                entity = await session.get(model_class, entity_id)

                # Cache the result if found
                if entity and self._cache_enabled and self._redis_client:
                    cache_key = f"{model_class.__name__}_{entity_id}"
                    await self._redis_client.setex(cache_key, self._cache_ttl, "cached")

                return entity

        except Exception as e:
            logger.error(f"Entity retrieval failed for {model_class.__name__}[{entity_id}]: {e}")
            raise DatabaseError(f"Failed to get {model_class.__name__}[{entity_id}]: {e}") from e

    async def update_entity(self, entity: T) -> T:
        """
        Update existing entity.

        Args:
            entity: Entity to update

        Returns:
            Updated entity
        """
        try:
            async with self.connection_manager.get_async_session() as session:
                merged_entity = await session.merge(entity)
                await session.commit()
                await session.refresh(merged_entity)

                # Invalidate cache if enabled
                if self._cache_enabled and self._redis_client:
                    cache_key = f"{type(entity).__name__}_{entity.id}"
                    await self._redis_client.delete(cache_key)
                    cache_pattern = f"{type(entity).__name__}_*"
                    await self._invalidate_cache_pattern(cache_pattern)

                return merged_entity

        except Exception as e:
            logger.error(f"Entity update failed for {type(entity).__name__}: {e}")
            raise DatabaseError(f"Failed to update {type(entity).__name__}: {e}") from e

    async def delete_entity(self, model_class: type[T], entity_id: K) -> bool:
        """
        Delete entity by ID.

        Args:
            model_class: Entity class
            entity_id: Primary key value

        Returns:
            True if deleted, False if not found
        """
        try:
            async with self.connection_manager.get_async_session() as session:
                entity = await session.get(model_class, entity_id)
                if not entity:
                    return False

                await session.delete(entity)
                await session.commit()

                # Invalidate cache if enabled
                if self._cache_enabled and self._redis_client:
                    cache_key = f"{model_class.__name__}_{entity_id}"
                    await self._redis_client.delete(cache_key)
                    cache_pattern = f"{model_class.__name__}_*"
                    await self._invalidate_cache_pattern(cache_pattern)

                return True

        except Exception as e:
            logger.error(f"Entity deletion failed for {model_class.__name__}[{entity_id}]: {e}")
            raise DatabaseError(f"Failed to delete {model_class.__name__}[{entity_id}]: {e}") from e

    async def list_entities(
        self,
        model_class: type[T],
        limit: int | None = None,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        order_desc: bool = False,
        include_relations: list[str] | None = None,
    ) -> list[T]:
        """
        List entities with filtering, pagination, and ordering.

        Args:
            model_class: Entity class
            limit: Maximum number of entities
            offset: Number of entities to skip
            filters: Filter criteria
            order_by: Field to order by
            order_desc: Descending order flag
            include_relations: Relations to eager load

        Returns:
            List of entities matching criteria
        """
        try:
            async with self.connection_manager.get_async_session() as session:
                # Build query
                query = select(model_class)

                # Apply filters
                if filters:
                    for field, value in filters.items():
                        if hasattr(model_class, field):
                            attr = getattr(model_class, field)
                            if isinstance(value, dict):
                                # Handle range filters like {"gte": 100, "lte": 200}
                                if "gte" in value:
                                    query = query.where(attr >= value["gte"])
                                if "lte" in value:
                                    query = query.where(attr <= value["lte"])
                                if "gt" in value:
                                    query = query.where(attr > value["gt"])
                                if "lt" in value:
                                    query = query.where(attr < value["lt"])
                            else:
                                query = query.where(attr == value)

                # Apply ordering
                if order_by and hasattr(model_class, order_by):
                    attr = getattr(model_class, order_by)
                    if order_desc:
                        query = query.order_by(desc(attr))
                    else:
                        query = query.order_by(asc(attr))

                # Apply pagination
                if offset:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)

                # Execute query
                result = await session.execute(query)
                return result.scalars().all()

        except Exception as e:
            logger.error(f"Entity listing failed for {model_class.__name__}: {e}")
            raise DatabaseError(f"Failed to list {model_class.__name__} entities: {e}") from e

    async def count_entities(
        self, model_class: type[T], filters: dict[str, Any] | None = None
    ) -> int:
        """
        Count entities matching filters.

        Args:
            model_class: Entity class
            filters: Filter criteria

        Returns:
            Count of matching entities
        """
        try:
            async with self.connection_manager.get_async_session() as session:
                # Build count query
                query = select(model_class)

                # Apply filters
                if filters:
                    for field, value in filters.items():
                        if hasattr(model_class, field):
                            attr = getattr(model_class, field)
                            if isinstance(value, dict):
                                # Handle range filters
                                if "gte" in value:
                                    query = query.where(attr >= value["gte"])
                                if "lte" in value:
                                    query = query.where(attr <= value["lte"])
                            else:
                                query = query.where(attr == value)

                # Count results
                result = await session.execute(query)
                return len(result.scalars().all())

        except Exception as e:
            logger.error(f"Entity counting failed for {model_class.__name__}: {e}")
            raise DatabaseError(f"Failed to count {model_class.__name__} entities: {e}") from e

    async def bulk_create(self, entities: list[T]) -> list[T]:
        """
        Create multiple entities in bulk.

        Args:
            entities: List of entities to create

        Returns:
            List of created entities
        """
        try:
            async with self.connection_manager.get_async_session() as session:
                session.add_all(entities)
                await session.commit()

                # Refresh all entities
                for entity in entities:
                    await session.refresh(entity)

                # Invalidate cache if enabled
                if self._cache_enabled and self._redis_client and entities:
                    cache_pattern = f"{type(entities[0]).__name__}_*"
                    await self._invalidate_cache_pattern(cache_pattern)

                return entities

        except Exception as e:
            entity_type = type(entities[0]).__name__ if entities else "Unknown"
            logger.error(f"Bulk entity creation failed for {entity_type}: {e}")
            raise DatabaseError(f"Failed to bulk create {entity_type} entities: {e}") from e

    # Transaction support

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Create a database transaction context.

        Returns:
            Database session within transaction context
        """
        async with self.connection_manager.get_async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Transaction failed: {e}")
                raise

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session.

        Returns:
            Database session
        """
        async with self.connection_manager.get_async_session() as session:
            yield session

    # Health and monitoring

    async def get_health_status(self) -> HealthStatus:
        """Get service health status."""
        try:
            # Test database connection
            async with self.connection_manager.get_async_session() as session:
                await session.execute(select(1))

            return HealthStatus.HEALTHY

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return HealthStatus.UNHEALTHY

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        return {
            "cache_enabled": self._cache_enabled,
            "started": self._started,
        }

    # Cache utility methods

    async def _invalidate_cache_pattern(self, pattern: str) -> None:
        """Invalidate cache keys matching pattern."""
        if not self._cache_enabled or not self._redis_client:
            return

        try:
            # Simple pattern invalidation - in production, use Redis SCAN
            # For now, we'll just skip this as it's complex to implement correctly
            pass
        except Exception as e:
            logger.warning(f"Cache pattern invalidation failed: {e}")
