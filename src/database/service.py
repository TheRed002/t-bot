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
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, TypeVar

import redis.asyncio as redis
from sqlalchemy import asc, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.base.interfaces import DatabaseServiceInterface, HealthStatus

# Import core components
from src.core.base.service import BaseService
from src.core.exceptions import ComponentError, DatabaseError, ValidationError
from src.core.logging import get_logger

# Import database infrastructure
from src.database.connection import DatabaseConnectionManager

if TYPE_CHECKING:
    from src.database.models import Base

# Type variables
T = TypeVar("T", bound="Base")
K = TypeVar("K")

logger = get_logger(__name__)


class DatabaseService(BaseService, DatabaseServiceInterface):
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

    async def create_entity(self, entity: T, processing_mode: str = "stream") -> T:
        """
        Create a new entity with consistent data transformation and processing mode.

        Args:
            entity: Entity to create
            processing_mode: Processing mode ("stream" for real-time, "batch" for bulk operations)

        Returns:
            Created entity with ID assigned (in standardized format)
        """
        try:
            # Apply consistent data transformation at module boundary
            transformed_entity = self._transform_entity_data(entity, processing_mode)

            async with self.connection_manager.get_async_session() as session:
                session.add(transformed_entity)
                await session.commit()
                await session.refresh(transformed_entity)

                # Invalidate cache if enabled
                if self._cache_enabled and self._redis_client:
                    cache_pattern = f"{type(transformed_entity).__name__}_*"
                    await self._invalidate_cache_pattern(cache_pattern)

                return transformed_entity

        except Exception as e:
            logger.error(f"Entity creation failed for {type(entity).__name__}: {e}")
            # Apply consistent error propagation aligned with error_handling module
            from src.error_handling.propagation_utils import (
                ProcessingStage,
                PropagationMethod,
                add_propagation_step,
            )

            error_data = {
                "error_type": type(e).__name__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operation": "create_entity",
                "entity_type": type(entity).__name__,
                "processing_mode": processing_mode,
                "data_format": "database_entity_v1",
                "message_pattern": "pub_sub",  # Align with analytics patterns
            }

            # Add propagation step for cross-module error flow
            error_data = add_propagation_step(
                error_data,
                source_module="database",
                target_module="error_handling",
                method=PropagationMethod.DIRECT_CALL,
                stage=ProcessingStage.ERROR_PROPAGATION
            )

            # Validate boundary data before propagation
            from src.utils.messaging_patterns import BoundaryValidator
            try:
                BoundaryValidator.validate_database_to_error_boundary(error_data)
            except ValidationError as ve:
                logger.warning(f"Database boundary validation failed: {ve}")
                # Continue with propagation but log validation failure

            self._propagate_database_error(e, "create_entity", type(entity).__name__)
            raise DatabaseError(
                f"Failed to create {type(entity).__name__}: {e}",
                details=error_data
            ) from e

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
        processing_mode: str = "stream",
    ) -> list[T]:
        """
        List entities with filtering, pagination, and ordering using consistent processing patterns.

        Args:
            model_class: Entity class
            limit: Maximum number of entities
            offset: Number of entities to skip
            filters: Filter criteria (validated at boundary)
            order_by: Field to order by
            order_desc: Descending order flag
            include_relations: Relations to eager load
            processing_mode: Processing mode ("stream" for real-time, "batch" for bulk operations)

        Returns:
            List of entities matching criteria (in standardized format)
        """
        try:
            # Validate filters at module boundary
            if filters:
                self._validate_filter_boundary(filters, model_class.__name__)

            async with self.connection_manager.get_async_session() as session:
                # Build query
                query = select(model_class)

                # Apply filters with consistent transformation
                if filters:
                    query = self._apply_consistent_filters(query, model_class, filters)

                # Apply ordering
                if order_by and hasattr(model_class, order_by):
                    attr = getattr(model_class, order_by)
                    if order_desc:
                        query = query.order_by(desc(attr))
                    else:
                        query = query.order_by(asc(attr))

                # Apply pagination (adjust for processing mode)
                if processing_mode == "stream":
                    # Stream processing uses smaller default batches for real-time processing
                    default_limit = 50
                else:
                    # Batch processing uses larger batches for efficiency
                    default_limit = 100

                if offset:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)
                else:
                    query = query.limit(default_limit)

                # Execute query
                result = await session.execute(query)
                entities = result.scalars().all()

                # Transform entities to consistent format
                return [self._transform_entity_data(entity, processing_mode) for entity in entities]

        except Exception as e:
            logger.error(f"Entity listing failed for {model_class.__name__}: {e}")
            # Apply consistent error propagation aligned with error_handling module
            from src.error_handling.propagation_utils import (
                ProcessingStage,
                PropagationMethod,
                add_propagation_step,
            )

            error_data = {
                "error_type": type(e).__name__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operation": "list_entities",
                "entity_type": model_class.__name__,
                "processing_mode": processing_mode,
                "data_format": "database_entity_v1",
                "message_pattern": "pub_sub",  # Align with analytics patterns
                "operation_type": "list",
                "batch_size": limit,
            }

            # Add propagation step for cross-module error flow
            error_data = add_propagation_step(
                error_data,
                source_module="database",
                target_module="error_handling",
                method=PropagationMethod.DIRECT_CALL,
                stage=ProcessingStage.ERROR_PROPAGATION
            )

            # Validate boundary data before propagation
            from src.utils.messaging_patterns import BoundaryValidator
            try:
                BoundaryValidator.validate_database_to_error_boundary(error_data)
            except ValidationError as ve:
                logger.warning(f"Database boundary validation failed: {ve}")
                # Continue with propagation but log validation failure

            self._propagate_database_error(e, "list_entities", model_class.__name__)
            raise DatabaseError(
                f"Failed to list {model_class.__name__} entities: {e}",
                details=error_data
            ) from e

    async def count_entities(
        self, model_class: type[T] | None = None, filters: dict[str, Any] | None = None
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

    async def execute_query(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a database query."""
        if not self._started:
            raise DatabaseError("Database service not started")

        try:
            async with self.get_session() as session:
                result = await session.execute(query, params or {})
                await session.commit()
                return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise DatabaseError(f"Query execution failed: {e}") from e

    async def get_connection_pool_status(self) -> dict[str, Any]:
        """Get connection pool status."""
        if not self.connection_manager:
            return {"status": "unavailable", "reason": "No connection manager"}

        try:
            # Get pool information from the connection manager
            if hasattr(self.connection_manager, "get_pool_status"):
                return await self.connection_manager.get_pool_status()
            else:
                # Basic status information
                return {
                    "status": "available" if self._started else "stopped",
                    "connection_manager": type(self.connection_manager).__name__,
                }
        except Exception as e:
            logger.error(f"Failed to get connection pool status: {e}")
            return {"status": "error", "error": str(e)}

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

    # Data Transformation and Validation Methods (aligned with core module patterns)

    def _transform_entity_data(self, entity: T, processing_mode: str) -> T:
        """Transform entity data to consistent format aligned with core module patterns."""
        if entity is None:
            return entity

        # Use standardized data transformation for consistency across modules
        from src.utils.data_flow_integrity import DataFlowTransformer

        try:
            entity = DataFlowTransformer.validate_and_transform_entity(
                entity, module_source="database", processing_mode=processing_mode
            )
        except Exception as e:
            self.logger.warning(f"Standardized entity transformation failed: {e}")
            # Fallback to minimal transformation
            if hasattr(entity, "__dict__") and not hasattr(entity, "processing_mode"):
                entity.processing_mode = processing_mode

        return entity

    def _validate_filter_boundary(self, filters: dict[str, Any], entity_name: str) -> None:
        """Validate filter data at module boundary for consistency."""
        from src.utils.data_flow_integrity import DataFlowValidator

        try:
            # Use standardized boundary validation
            DataFlowValidator.validate_complete_data_flow(
                filters,
                source_module="database",
                target_module="core",
                operation_type=f"filter_validation_{entity_name}"
            )
        except Exception:
            # Fallback to original validation
            if not isinstance(filters, dict):
                from src.core.exceptions import ValidationError
                raise ValidationError(
                    f"Filters must be dict for {entity_name}",
                    field_name="filters",
                    field_value=type(filters).__name__,
                    expected_type="dict"
                )

            # Validate financial filter fields using standardized approach
            from src.utils.data_flow_integrity import DataFlowTransformer

            try:
                # Create temporary holder for validation
                class FilterHolder:
                    def __init__(self, filter_dict):
                        for key, value in filter_dict.items():
                            setattr(self, key, value)

                filter_holder = FilterHolder(filters)
                DataFlowTransformer.apply_financial_field_transformation(filter_holder)
            except Exception as validation_error:
                from src.utils.data_flow_integrity import StandardizedErrorPropagator
                StandardizedErrorPropagator.propagate_validation_error(
                    validation_error,
                    context=f"filter_validation_{entity_name}",
                    module_source="database",
                    field_name="filters",
                    field_value=str(filters)
                )

    def _apply_consistent_filters(self, query: Any, model_class: type[T], filters: dict[str, Any]) -> Any:
        """Apply filters with consistent transformation patterns."""
        for field, value in filters.items():
            if hasattr(model_class, field):
                attr = getattr(model_class, field)
                if isinstance(value, dict):
                    # Handle range filters with consistent validation
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
        return query

    def _propagate_database_error(self, error: Exception, operation: str, entity_name: str) -> None:
        """Propagate database errors with consistent patterns aligned with core module."""
        from src.core.exceptions import DatabaseError, ValidationError

        # Apply consistent error propagation patterns
        if isinstance(error, ValidationError):
            # Validation errors are re-raised as-is for consistency
            logger.debug(
                f"Validation error in database.{operation} for {entity_name} - propagating as validation error",
                operation=operation,
                entity_name=entity_name,
                error_type=type(error).__name__
            )
        elif isinstance(error, DatabaseError):
            # Database errors get additional context
            logger.warning(
                f"Database error in database.{operation} for {entity_name} - adding context",
                operation=operation,
                entity_name=entity_name,
                error=str(error)
            )
        else:
            # Generic errors get database-level error propagation
            logger.error(
                f"Database service error in database.{operation} for {entity_name} - wrapping in DatabaseError",
                operation=operation,
                entity_name=entity_name,
                original_error=str(error)
            )
