"""Database repository base that complies with core interfaces."""

from datetime import datetime, timezone
from typing import Any, TypeVar

from sqlalchemy import asc, desc, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.base.interfaces import HealthCheckResult, HealthStatus
from src.core.base.repository import BaseRepository as CoreBaseRepository
from src.core.exceptions import DatabaseError
from src.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")  # Entity type
K = TypeVar("K")  # Key type


class DatabaseRepository(CoreBaseRepository[T, K]):
    """Database repository that bridges SQLAlchemy with core interfaces."""

    def __init__(
        self,
        session: AsyncSession,
        model: type[T],
        entity_type: type[T],
        key_type: type[K],
        name: str | None = None,
    ):
        """Initialize database repository with core compliance.

        Args:
            session: AsyncSession for database operations
            model: SQLAlchemy model class
            entity_type: Entity type for core repository
            key_type: Key type for core repository
            name: Repository name
        """
        # Initialize core base repository
        super().__init__(entity_type=entity_type, key_type=key_type, name=name)

        # Database-specific attributes
        self.session = session
        self.model = model
        self._logger = logger

    # Implement core abstract methods
    async def _create_entity(self, entity: T) -> T:
        """Create entity in database."""
        try:
            self.session.add(entity)
            await self.session.flush()
            return entity
        except IntegrityError as e:
            self._logger.error(f"Integrity error creating {self.model.__name__}: {e}")
            await self.session.rollback()
            raise DatabaseError(f"Integrity error: {e}")
        except Exception as e:
            self._logger.error(f"Error creating {self.model.__name__}: {e}")
            await self.session.rollback()
            raise DatabaseError(f"Failed to create entity: {e}")

    async def _get_entity_by_id(self, entity_id: K) -> T | None:
        """Get entity by ID from database."""
        try:
            stmt = select(self.model).where(self.model.id == entity_id)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            self._logger.error(f"Error getting {self.model.__name__} by ID {entity_id}: {e}")
            raise DatabaseError(f"Failed to get entity: {e}")

    async def _update_entity(self, entity: T) -> T | None:
        """Update entity in database."""
        try:
            # Update timestamp if model has it
            if hasattr(entity, "updated_at"):
                entity.updated_at = datetime.now(timezone.utc)

            # Increment version if model has it
            if hasattr(entity, "version"):
                entity.version = (entity.version or 0) + 1

            # Use merge for detached instances
            merged = await self.session.merge(entity)
            await self.session.flush()
            return merged
        except Exception as e:
            self._logger.error(f"Error updating {self.model.__name__}: {e}")
            await self.session.rollback()
            raise DatabaseError(f"Failed to update entity: {e}")

    async def _delete_entity(self, entity_id: K) -> bool:
        """Delete entity from database."""
        try:
            entity = await self._get_entity_by_id(entity_id)
            if entity:
                await self.session.delete(entity)
                await self.session.flush()
                return True
            return False
        except Exception as e:
            self._logger.error(f"Error deleting {self.model.__name__} {entity_id}: {e}")
            await self.session.rollback()
            raise DatabaseError(f"Failed to delete entity: {e}")

    async def _list_entities(
        self,
        limit: int | None,
        offset: int | None,
        filters: dict[str, Any] | None,
        order_by: str | None,
        order_desc: bool,
    ) -> list[T]:
        """List entities from database."""
        try:
            stmt = select(self.model)

            # Apply filters
            if filters:
                for key, value in filters.items():
                    if hasattr(self.model, key):
                        if isinstance(value, list):
                            stmt = stmt.where(getattr(self.model, key).in_(value))
                        elif isinstance(value, dict):
                            # Handle complex filters
                            column = getattr(self.model, key)
                            if "gt" in value:
                                stmt = stmt.where(column > value["gt"])
                            if "gte" in value:
                                stmt = stmt.where(column >= value["gte"])
                            if "lt" in value:
                                stmt = stmt.where(column < value["lt"])
                            if "lte" in value:
                                stmt = stmt.where(column <= value["lte"])
                            if "like" in value:
                                stmt = stmt.where(column.like(f"%{value['like']}%"))
                        else:
                            stmt = stmt.where(getattr(self.model, key) == value)

            # Apply ordering
            if order_by:
                if order_desc:
                    stmt = stmt.order_by(desc(getattr(self.model, order_by)))
                else:
                    stmt = stmt.order_by(asc(getattr(self.model, order_by)))

            # Apply pagination
            if offset:
                stmt = stmt.offset(offset)
            if limit:
                stmt = stmt.limit(limit)

            result = await self.session.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            self._logger.error(f"Error listing {self.model.__name__}: {e}")
            raise DatabaseError(f"Failed to list entities: {e}")

    async def _count_entities(self, filters: dict[str, Any] | None) -> int:
        """Count entities in database."""
        try:
            from sqlalchemy import func

            stmt = select(func.count()).select_from(self.model)

            if filters:
                for key, value in filters.items():
                    if hasattr(self.model, key):
                        stmt = stmt.where(getattr(self.model, key) == value)

            result = await self.session.execute(stmt)
            return result.scalar() or 0
        except Exception as e:
            self._logger.error(f"Error counting {self.model.__name__}: {e}")
            raise DatabaseError(f"Failed to count entities: {e}")

    async def _bulk_create_entities(self, entities: list[T]) -> list[T]:
        """Bulk create entities in database."""
        try:
            self.session.add_all(entities)
            await self.session.flush()
            return entities
        except Exception as e:
            self._logger.error(f"Error creating multiple {self.model.__name__}: {e}")
            await self.session.rollback()
            raise DatabaseError(f"Failed to bulk create entities: {e}")

    async def _test_connection(self, connection: Any) -> bool:
        """Test database connection."""
        try:
            # Simple connectivity test
            stmt = select(1)
            await self.session.execute(stmt)
            return True
        except Exception:
            return False

    async def _repository_health_check(self) -> HealthStatus:
        """Repository-specific health check."""
        try:
            # Test database connectivity
            if await self._test_connection(None):
                return HealthStatus.HEALTHY
            return HealthStatus.UNHEALTHY
        except Exception:
            return HealthStatus.UNHEALTHY

    # Health check interface methods
    async def ready_check(self) -> HealthCheckResult:
        """Check if repository is ready to serve requests."""
        try:
            # Test database connectivity
            if await self._test_connection(None):
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message="Repository is ready",
                    details={"repository": self._name, "connected": True},
                )
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message="Repository not ready - database connection failed",
                details={"repository": self._name, "connected": False},
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Repository readiness check failed: {e!s}",
                details={"repository": self._name, "error": str(e)},
            )

    async def live_check(self) -> HealthCheckResult:
        """Check if repository is alive and functioning."""
        try:
            # Quick connectivity check
            stmt = select(1)
            await self.session.execute(stmt)

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Repository is live",
                details={
                    "repository": self._name,
                    "alive": True,
                    "query_metrics": self.query_metrics,
                },
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Repository liveness check failed: {e!s}",
                details={"repository": self._name, "error": str(e)},
            )

    # Additional helper methods for backward compatibility
    async def get(self, id: Any) -> T | None:
        """Get entity by ID (backward compatibility)."""
        return await self.get_by_id(id)

    async def get_all(
        self,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[T]:
        """Get all entities (backward compatibility)."""
        # Map old order_by format to new format
        order_desc = False
        if order_by and order_by.startswith("-"):
            order_desc = True
            order_by = order_by[1:]

        return await self.list(
            limit=limit, offset=offset, filters=filters, order_by=order_by, order_desc=order_desc
        )

    async def get_by(self, **kwargs) -> T | None:
        """Get entity by attributes."""
        try:
            stmt = select(self.model)
            for key, value in kwargs.items():
                stmt = stmt.where(getattr(self.model, key) == value)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            self._logger.error(f"Error getting {self.model.__name__} by {kwargs}: {e}")
            raise DatabaseError(f"Failed to get entity by attributes: {e}")

    async def create_many(self, entities: list[T]) -> list[T]:
        """Create multiple entities (backward compatibility)."""
        return await self.bulk_create(entities)

    async def soft_delete(self, id: Any, deleted_by: str | None = None) -> bool:
        """Soft delete entity if it supports it."""
        try:
            entity = await self.get_by_id(id)
            if entity and hasattr(entity, "soft_delete"):
                entity.soft_delete(deleted_by)
                await self.update(entity)
                return True
            return False
        except Exception as e:
            self._logger.error(f"Error soft deleting {self.model.__name__} {id}: {e}")
            return False

    async def exists(self, id: Any) -> bool:
        """Check if entity exists."""
        try:
            stmt = select(self.model.id).where(self.model.id == id)
            result = await self.session.execute(stmt)
            return result.scalar() is not None
        except Exception as e:
            self._logger.error(f"Error checking existence of {self.model.__name__} {id}: {e}")
            return False

    async def begin(self):
        """Begin transaction."""
        return self.session.begin()

    async def commit(self):
        """Commit transaction."""
        await self.session.commit()

    async def rollback(self):
        """Rollback transaction."""
        await self.session.rollback()

    async def refresh(self, entity: T):
        """Refresh entity from database."""
        await self.session.refresh(entity)
