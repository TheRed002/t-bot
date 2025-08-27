"""
Base repository implementation for the repository pattern.

This module provides the foundation for all data access layer components
in the trading bot system, implementing CRUD operations, transaction
management, and connection pooling.
"""

import asyncio
import builtins
from abc import abstractmethod
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timezone
from typing import (
    Any,
    Generic,
    TypeVar,
)

from src.core.base.component import BaseComponent
from src.core.base.interfaces import HealthStatus, RepositoryComponent
from src.core.exceptions import (
    DatabaseConnectionError,
    DataValidationError,
    EntityNotFoundError,
    RepositoryError,
)
from src.core.types.base import ConfigDict

# Type variables for repository operations
T = TypeVar("T")  # Entity type
K = TypeVar("K")  # Primary key type


class BaseRepository(BaseComponent, RepositoryComponent, Generic[T, K]):
    """
    Base repository implementing the repository pattern.

    Provides:
    - CRUD operations with type safety
    - Connection management and pooling
    - Transaction support
    - Query optimization
    - Caching layer
    - Audit logging
    - Performance monitoring
    - Data validation

    Example:
        ```python
        @dataclass
        class Order:
            id: int
            symbol: str
            quantity: Decimal
            price: Decimal
            created_at: datetime


        class OrderRepository(BaseRepository[Order, int]):
            def __init__(self, connection_pool):
                super().__init__(name="OrderRepository")
                self.connection_pool = connection_pool

            async def _create_entity(self, entity: Order) -> Order:
                # Database-specific implementation
                async with self.get_connection() as conn:
                    # SQL execution logic
                    pass
        ```
    """

    def __init__(
        self,
        entity_type: type[T],
        key_type: type[K],
        name: str | None = None,
        config: ConfigDict | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize base repository.

        Args:
            entity_type: Type of entities this repository manages
            key_type: Type of primary key for entities
            name: Repository name for identification
            config: Repository configuration
            correlation_id: Request correlation ID
        """
        super().__init__(name, config, correlation_id)

        self._entity_type = entity_type
        self._key_type = key_type
        self._table_name = getattr(entity_type, "__tablename__", None)

        # Connection management
        self._connection_pool: Any | None = None
        self._connection_timeout = 30.0
        self._max_connections = 10

        # Transaction support
        self._transaction_manager: Any | None = None
        self._auto_commit = True

        # Caching
        self._cache_enabled = False
        self._cache_ttl = 300  # 5 minutes
        self._cache_store: dict[str, Any] = {}
        self._cache_timestamps: dict[str, datetime] = {}

        # Performance tracking
        self._query_metrics: dict[str, Any] = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_query_time": 0.0,
            "slow_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Query optimization
        self._slow_query_threshold = 1.0  # seconds
        self._query_timeout = 30.0  # seconds

        self._logger.debug(
            "Repository initialized",
            repository=self._name,
            entity_type=entity_type.__name__,
            key_type=key_type.__name__,
        )

    @property
    def entity_type(self) -> type[T]:
        """Get entity type managed by this repository."""
        return self._entity_type

    @property
    def key_type(self) -> type[K]:
        """Get primary key type for entities."""
        return self._key_type

    @property
    def query_metrics(self) -> dict[str, Any]:
        """Get query performance metrics."""
        return self._query_metrics.copy()

    # Connection Management
    def set_connection_pool(self, connection_pool: Any) -> None:
        """
        Set database connection pool.

        Args:
            connection_pool: Database connection pool instance
        """
        self._connection_pool = connection_pool
        self._logger.debug(
            "Connection pool configured",
            repository=self._name,
        )

    async def get_connection(self) -> AbstractAsyncContextManager[Any]:
        """
        Get database connection from pool.

        Returns:
            AsyncContextManager for database connection

        Raises:
            DatabaseConnectionError: If connection cannot be obtained
        """
        if not self._connection_pool:
            raise DatabaseConnectionError("No connection pool configured")

        try:
            return self._connection_pool.get_connection(timeout=self._connection_timeout)
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to get database connection: {e}") from e

    def set_transaction_manager(self, transaction_manager: Any) -> None:
        """
        Set transaction manager for this repository.

        Args:
            transaction_manager: Transaction manager instance
        """
        self._transaction_manager = transaction_manager
        self._logger.debug(
            "Transaction manager configured",
            repository=self._name,
        )

    # CRUD Operations
    async def create(self, entity: T) -> T:
        """
        Create new entity.

        Args:
            entity: Entity to create

        Returns:
            Created entity with updated fields (e.g., generated ID)

        Raises:
            RepositoryError: If creation fails
            DataValidationError: If entity data is invalid
        """
        start_time = datetime.now(timezone.utc)

        try:
            self._logger.debug(
                "Creating entity",
                repository=self._name,
                entity_type=self._entity_type.__name__,
            )

            # Validate entity before creation
            self._validate_entity(entity)

            # Execute creation with monitoring
            result = await self._execute_with_monitoring("create", self._create_entity, entity)

            # Clear relevant cache entries
            if self._cache_enabled:
                self._invalidate_cache_pattern("list_")

            self._logger.info(
                "Entity created successfully",
                repository=self._name,
                entity_type=self._entity_type.__name__,
            )

            return result

        except DataValidationError:
            raise
        except Exception as e:
            raise RepositoryError(f"Failed to create entity in {self._name}: {e}") from e
        finally:
            self._record_query_metrics("create", start_time)

    async def get_by_id(self, entity_id: K) -> T | None:
        """
        Get entity by primary key.

        Args:
            entity_id: Primary key value

        Returns:
            Entity if found, None otherwise

        Raises:
            RepositoryError: If query fails
        """
        start_time = datetime.now(timezone.utc)
        cache_key = f"get_by_id_{entity_id}"

        try:
            # Check cache first
            if self._cache_enabled:
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    self._query_metrics["cache_hits"] += 1
                    return cached_result
                self._query_metrics["cache_misses"] += 1

            self._logger.debug(
                "Getting entity by ID",
                repository=self._name,
                entity_id=entity_id,
            )

            # Execute query with monitoring
            result = await self._execute_with_monitoring(
                "get_by_id", self._get_entity_by_id, entity_id
            )

            # Cache the result
            if self._cache_enabled and result is not None:
                self._set_cache(cache_key, result)

            return result

        except Exception as e:
            raise RepositoryError(f"Failed to get entity by ID in {self._name}: {e}") from e
        finally:
            self._record_query_metrics("get_by_id", start_time)

    async def update(self, entity: T) -> T:
        """
        Update existing entity.

        Args:
            entity: Entity with updated data

        Returns:
            Updated entity

        Raises:
            RepositoryError: If update fails
            EntityNotFoundError: If entity doesn't exist
            DataValidationError: If entity data is invalid
        """
        start_time = datetime.now(timezone.utc)

        try:
            self._logger.debug(
                "Updating entity",
                repository=self._name,
                entity_type=self._entity_type.__name__,
            )

            # Validate entity before update
            self._validate_entity(entity)

            # Execute update with monitoring
            result = await self._execute_with_monitoring("update", self._update_entity, entity)

            if result is None:
                raise EntityNotFoundError(f"Entity not found for update in {self._name}")

            # Invalidate cache entries
            if self._cache_enabled:
                entity_id = self._extract_entity_id(entity)
                if entity_id:
                    self._invalidate_cache(f"get_by_id_{entity_id}")
                self._invalidate_cache_pattern("list_")

            self._logger.info(
                "Entity updated successfully",
                repository=self._name,
                entity_type=self._entity_type.__name__,
            )

            return result

        except (EntityNotFoundError, DataValidationError):
            raise
        except Exception as e:
            raise RepositoryError(f"Failed to update entity in {self._name}: {e}") from e
        finally:
            self._record_query_metrics("update", start_time)

    async def delete(self, entity_id: K) -> bool:
        """
        Delete entity by primary key.

        Args:
            entity_id: Primary key value

        Returns:
            True if entity was deleted, False if not found

        Raises:
            RepositoryError: If deletion fails
        """
        start_time = datetime.now(timezone.utc)

        try:
            self._logger.debug(
                "Deleting entity",
                repository=self._name,
                entity_id=entity_id,
            )

            # Execute deletion with monitoring
            result = await self._execute_with_monitoring("delete", self._delete_entity, entity_id)

            # Invalidate cache entries
            if self._cache_enabled and result:
                self._invalidate_cache(f"get_by_id_{entity_id}")
                self._invalidate_cache_pattern("list_")

            if result:
                self._logger.info(
                    "Entity deleted successfully",
                    repository=self._name,
                    entity_id=entity_id,
                )
            else:
                self._logger.warning(
                    "Entity not found for deletion",
                    repository=self._name,
                    entity_id=entity_id,
                )

            return result

        except Exception as e:
            raise RepositoryError(f"Failed to delete entity in {self._name}: {e}") from e
        finally:
            self._record_query_metrics("delete", start_time)

    async def list(
        self,
        limit: int | None = None,
        offset: int | None = None,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        order_desc: bool = False,
    ) -> list[T]:
        """
        List entities with optional pagination, filtering, and ordering.

        Args:
            limit: Maximum number of entities to return
            offset: Number of entities to skip
            filters: Filter criteria as key-value pairs
            order_by: Field name to order by
            order_desc: Whether to order in descending order

        Returns:
            List of entities matching criteria

        Raises:
            RepositoryError: If query fails
        """
        start_time = datetime.now(timezone.utc)

        # Create cache key from parameters
        cache_key = self._create_list_cache_key(limit, offset, filters, order_by, order_desc)

        try:
            # Check cache first
            if self._cache_enabled:
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    self._query_metrics["cache_hits"] += 1
                    return cached_result
                self._query_metrics["cache_misses"] += 1

            self._logger.debug(
                "Listing entities",
                repository=self._name,
                limit=limit,
                offset=offset,
                filters=filters,
                order_by=order_by,
            )

            # Execute query with monitoring
            result = await self._execute_with_monitoring(
                "list", self._list_entities, limit, offset, filters, order_by, order_desc
            )

            # Cache the result
            if self._cache_enabled:
                self._set_cache(cache_key, result)

            self._logger.debug(
                "Listed entities",
                repository=self._name,
                count=len(result),
            )

            return result

        except Exception as e:
            raise RepositoryError(f"Failed to list entities in {self._name}: {e}") from e
        finally:
            self._record_query_metrics("list", start_time)

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """
        Count entities with optional filtering.

        Args:
            filters: Filter criteria as key-value pairs

        Returns:
            Number of entities matching criteria

        Raises:
            RepositoryError: If query fails
        """
        start_time = datetime.now(timezone.utc)
        cache_key = f"count_{hash(str(filters))}"

        try:
            # Check cache first
            if self._cache_enabled:
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    self._query_metrics["cache_hits"] += 1
                    return cached_result
                self._query_metrics["cache_misses"] += 1

            self._logger.debug(
                "Counting entities",
                repository=self._name,
                filters=filters,
            )

            # Execute count with monitoring
            result = await self._execute_with_monitoring("count", self._count_entities, filters)

            # Cache the result
            if self._cache_enabled:
                self._set_cache(cache_key, result)

            return result

        except Exception as e:
            raise RepositoryError(f"Failed to count entities in {self._name}: {e}") from e
        finally:
            self._record_query_metrics("count", start_time)

    # Batch Operations
    async def bulk_create(self, entities: builtins.list[T]) -> builtins.list[T]:
        """
        Create multiple entities in a batch operation.

        Args:
            entities: List of entities to create

        Returns:
            List of created entities

        Raises:
            RepositoryError: If batch creation fails
        """
        if not entities:
            return []

        start_time = datetime.now(timezone.utc)

        try:
            self._logger.debug(
                "Bulk creating entities",
                repository=self._name,
                count=len(entities),
            )

            # Validate all entities
            for entity in entities:
                self._validate_entity(entity)

            # Execute bulk creation
            result = await self._execute_with_monitoring(
                "bulk_create", self._bulk_create_entities, entities
            )

            # Invalidate cache
            if self._cache_enabled:
                self._invalidate_cache_pattern("list_")

            self._logger.info(
                "Bulk entity creation completed",
                repository=self._name,
                count=len(result),
            )

            return result

        except Exception as e:
            raise RepositoryError(f"Failed to bulk create entities in {self._name}: {e}") from e
        finally:
            self._record_query_metrics("bulk_create", start_time)

    async def bulk_update(self, entities: builtins.list[T]) -> builtins.list[T]:
        """
        Update multiple entities in a batch operation.

        Args:
            entities: List of entities to update

        Returns:
            List of updated entities

        Raises:
            RepositoryError: If batch update fails
        """
        if not entities:
            return []

        start_time = datetime.now(timezone.utc)

        try:
            self._logger.debug(
                "Bulk updating entities",
                repository=self._name,
                count=len(entities),
            )

            # Validate all entities
            for entity in entities:
                self._validate_entity(entity)

            # Execute bulk update
            result = await self._execute_with_monitoring(
                "bulk_update", self._bulk_update_entities, entities
            )

            # Invalidate cache
            if self._cache_enabled:
                self._clear_cache()

            self._logger.info(
                "Bulk entity update completed",
                repository=self._name,
                count=len(result),
            )

            return result

        except Exception as e:
            raise RepositoryError(f"Failed to bulk update entities in {self._name}: {e}") from e
        finally:
            self._record_query_metrics("bulk_update", start_time)

    # Transaction Support
    async def execute_in_transaction(
        self, operation_func: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """
        Execute operation within a database transaction.

        Args:
            operation_func: Function to execute in transaction
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            RepositoryError: If transaction fails
        """
        if not self._transaction_manager:
            # Execute without transaction if no manager available
            return await operation_func(*args, **kwargs)

        try:
            async with self._transaction_manager.transaction():
                return await operation_func(*args, **kwargs)
        except Exception as e:
            self._logger.error(
                "Transaction failed",
                repository=self._name,
                error=str(e),
            )
            raise RepositoryError(f"Transaction failed in {self._name}: {e}") from e

    # Cache Management
    def configure_cache(
        self,
        enabled: bool = True,
        ttl: int = 300,
    ) -> None:
        """
        Configure repository caching.

        Args:
            enabled: Enable/disable caching
            ttl: Cache time-to-live in seconds
        """
        self._cache_enabled = enabled
        self._cache_ttl = ttl

        if not enabled:
            self._clear_cache()

        self._logger.info(
            "Cache configured",
            repository=self._name,
            enabled=enabled,
            ttl=ttl,
        )

    def _get_from_cache(self, key: str) -> Any | None:
        """Get value from cache if not expired."""
        if not self._cache_enabled or key not in self._cache_store:
            return None

        timestamp = self._cache_timestamps.get(key)
        if not timestamp:
            return None

        # Check expiration
        age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        if age > self._cache_ttl:
            self._invalidate_cache(key)
            return None

        return self._cache_store[key]

    def _set_cache(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if not self._cache_enabled:
            return

        self._cache_store[key] = value
        self._cache_timestamps[key] = datetime.now(timezone.utc)

    def _invalidate_cache(self, key: str) -> None:
        """Remove specific key from cache."""
        self._cache_store.pop(key, None)
        self._cache_timestamps.pop(key, None)

    def _invalidate_cache_pattern(self, pattern: str) -> None:
        """Remove all keys matching pattern from cache."""
        keys_to_remove = [key for key in self._cache_store.keys() if pattern in key]
        for key in keys_to_remove:
            self._invalidate_cache(key)

    def _clear_cache(self) -> None:
        """Clear all cache entries."""
        self._cache_store.clear()
        self._cache_timestamps.clear()

    # Health Check
    async def _health_check_internal(self) -> HealthStatus:
        """Repository-specific health check."""
        try:
            # Check connection pool health
            if self._connection_pool:
                # Test connection
                async with self.get_connection() as conn:
                    # Simple connectivity test
                    await asyncio.wait_for(self._test_connection(conn), timeout=5.0)

            # Check query performance
            if self._query_metrics["total_queries"] > 0:
                error_rate = (
                    self._query_metrics["failed_queries"] / self._query_metrics["total_queries"]
                )

                if error_rate > 0.1:  # More than 10% errors
                    return HealthStatus.DEGRADED

            # Repository-specific health check
            return await self._repository_health_check()

        except asyncio.TimeoutError:
            return HealthStatus.UNHEALTHY
        except Exception as e:
            self._logger.error(
                "Repository health check failed",
                repository=self._name,
                error=str(e),
            )
            return HealthStatus.UNHEALTHY

    # Metrics and Monitoring
    async def _execute_with_monitoring(
        self, operation_name: str, operation_func: Callable, *args, **kwargs
    ) -> Any:
        """Execute repository operation with monitoring."""
        start_time = datetime.now(timezone.utc)

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                operation_func(*args, **kwargs), timeout=self._query_timeout
            )

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Track slow queries
            if execution_time > self._slow_query_threshold:
                self._query_metrics["slow_queries"] += 1
                self._logger.warning(
                    "Slow query detected",
                    repository=self._name,
                    operation=operation_name,
                    execution_time=execution_time,
                    threshold=self._slow_query_threshold,
                )

            self._query_metrics["successful_queries"] += 1
            return result

        except Exception as e:
            self._query_metrics["failed_queries"] += 1
            self._logger.error(
                "Repository operation failed",
                repository=self._name,
                operation=operation_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def _record_query_metrics(self, operation: str, start_time: datetime) -> None:
        """Record query execution metrics."""
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        self._query_metrics["total_queries"] += 1

        # Update average query time
        current_avg = self._query_metrics["average_query_time"]
        total_queries = self._query_metrics["total_queries"]
        self._query_metrics["average_query_time"] = (
            current_avg * (total_queries - 1) + execution_time
        ) / total_queries

    def get_metrics(self) -> dict[str, Any]:
        """Get combined component and repository metrics."""
        metrics = super().get_metrics()
        metrics.update(self.query_metrics)
        return metrics

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        super().reset_metrics()
        self._query_metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_query_time": 0.0,
            "slow_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    # Utility Methods
    def _create_list_cache_key(
        self,
        limit: int | None,
        offset: int | None,
        filters: dict[str, Any] | None,
        order_by: str | None,
        order_desc: bool,
    ) -> str:
        """Create cache key for list operations."""
        key_parts = [
            "list",
            str(limit or ""),
            str(offset or ""),
            str(hash(str(filters or {}))),
            str(order_by or ""),
            str(order_desc),
        ]
        return "_".join(key_parts)

    def _validate_entity(self, entity: T) -> None:
        """
        Validate entity data.

        Args:
            entity: Entity to validate

        Raises:
            DataValidationError: If entity is invalid
        """
        if not isinstance(entity, self._entity_type):
            raise DataValidationError(f"Entity must be of type {self._entity_type.__name__}")

        # Additional validation can be implemented in subclasses
        self._validate_entity_data(entity)

    def _extract_entity_id(self, entity: T) -> K | None:
        """Extract primary key from entity."""
        # Default implementation - override in subclasses
        return getattr(entity, "id", None)

    # Abstract methods for subclass implementation
    @abstractmethod
    async def _create_entity(self, entity: T) -> T:
        """Create entity in the database."""
        pass

    @abstractmethod
    async def _get_entity_by_id(self, entity_id: K) -> T | None:
        """Get entity from database by ID."""
        pass

    @abstractmethod
    async def _update_entity(self, entity: T) -> T | None:
        """Update entity in the database."""
        pass

    @abstractmethod
    async def _delete_entity(self, entity_id: K) -> bool:
        """Delete entity from database."""
        pass

    @abstractmethod
    async def _list_entities(
        self,
        limit: int | None,
        offset: int | None,
        filters: dict[str, Any] | None,
        order_by: str | None,
        order_desc: bool,
    ) -> builtins.list[T]:
        """List entities from database."""
        pass

    @abstractmethod
    async def _count_entities(self, filters: dict[str, Any] | None) -> int:
        """Count entities in database."""
        pass

    # Optional methods for subclass customization
    async def _bulk_create_entities(self, entities: builtins.list[T]) -> builtins.list[T]:
        """Bulk create entities - default implementation uses individual creates."""
        results = []
        for entity in entities:
            result = await self._create_entity(entity)
            results.append(result)
        return results

    async def _bulk_update_entities(self, entities: builtins.list[T]) -> builtins.list[T]:
        """Bulk update entities - default implementation uses individual updates."""
        results = []
        for entity in entities:
            result = await self._update_entity(entity)
            if result:
                results.append(result)
        return results

    async def _test_connection(self, connection: Any) -> bool:
        """Test database connection - override in subclasses."""
        return True

    async def _repository_health_check(self) -> HealthStatus:
        """Repository-specific health check - override in subclasses."""
        return HealthStatus.HEALTHY

    def _validate_entity_data(self, entity: T) -> None:
        """Additional entity validation - override in subclasses."""
        pass
