"""Base repository pattern implementation."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from sqlalchemy import asc, desc, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class RepositoryInterface(ABC, Generic[T]):
    """Repository interface."""

    @abstractmethod
    async def get(self, id: Any) -> T | None:
        """Get entity by ID."""
        pass

    @abstractmethod
    async def get_all(
        self,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[T]:
        """Get all entities with optional filtering."""
        pass

    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create new entity."""
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update existing entity."""
        pass

    @abstractmethod
    async def delete(self, id: Any) -> bool:
        """Delete entity by ID."""
        pass

    @abstractmethod
    async def exists(self, id: Any) -> bool:
        """Check if entity exists."""
        pass


class BaseRepository(RepositoryInterface[T]):
    """Base repository with common CRUD operations."""

    def __init__(self, session: AsyncSession, model: type[T]):
        """
        Initialize repository.

        Args:
            session: Async database session
            model: Model class
        """
        self.session = session
        self.model = model
        self._logger = logger

    async def get(self, id: Any) -> T | None:
        """Get entity by ID."""
        try:
            stmt = select(self.model).where(self.model.id == id)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            self._logger.error(f"Error getting {self.model.__name__} by ID {id}: {e}")
            raise

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
            raise

    async def get_all(
        self,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[T]:
        """Get all entities with optional filtering."""
        try:
            stmt = select(self.model)

            # Apply filters
            if filters:
                for key, value in filters.items():
                    if hasattr(self.model, key):
                        if isinstance(value, list):
                            stmt = stmt.where(getattr(self.model, key).in_(value))
                        elif isinstance(value, dict):
                            # Handle complex filters like {'gt': 100, 'lt': 200}
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
                if order_by.startswith("-"):
                    stmt = stmt.order_by(desc(getattr(self.model, order_by[1:])))
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
            self._logger.error(f"Error getting all {self.model.__name__}: {e}")
            raise

    async def create(self, entity: T) -> T:
        """Create new entity."""
        try:
            self.session.add(entity)
            await self.session.flush()
            return entity
        except IntegrityError as e:
            self._logger.error(f"Integrity error creating {self.model.__name__}: {e}")
            await self.session.rollback()
            raise
        except Exception as e:
            self._logger.error(f"Error creating {self.model.__name__}: {e}")
            await self.session.rollback()
            raise

    async def create_many(self, entities: list[T]) -> list[T]:
        """Create multiple entities."""
        try:
            self.session.add_all(entities)
            await self.session.flush()
            return entities
        except Exception as e:
            self._logger.error(f"Error creating multiple {self.model.__name__}: {e}")
            await self.session.rollback()
            raise

    async def update(self, entity: T) -> T:
        """Update existing entity."""
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
            raise

    async def delete(self, id: Any) -> bool:
        """Delete entity by ID."""
        try:
            entity = await self.get(id)
            if entity:
                self.session.delete(entity)
                await self.session.flush()
                return True
            return False
        except Exception as e:
            self._logger.error(f"Error deleting {self.model.__name__} {id}: {e}")
            await self.session.rollback()
            raise

    async def soft_delete(self, id: Any, deleted_by: str | None = None) -> bool:
        """Soft delete entity if it supports it."""
        try:
            entity = await self.get(id)
            if entity and hasattr(entity, "soft_delete"):
                entity.soft_delete(deleted_by)
                await self.update(entity)
                return True
            return False
        except Exception as e:
            self._logger.error(f"Error soft deleting {self.model.__name__} {id}: {e}")
            raise

    async def exists(self, id: Any) -> bool:
        """Check if entity exists."""
        try:
            stmt = select(self.model.id).where(self.model.id == id)
            result = await self.session.execute(stmt)
            return result.scalar() is not None
        except Exception as e:
            self._logger.error(f"Error checking existence of {self.model.__name__} {id}: {e}")
            raise

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count entities."""
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
            raise

    async def begin(self):
        """Begin transaction."""
        return await self.session.begin()

    async def commit(self):
        """Commit transaction."""
        await self.session.commit()

    async def rollback(self):
        """Rollback transaction."""
        await self.session.rollback()

    # Methods to comply with core Repository protocol
    async def list(
        self,
        limit: int | None = None,
        offset: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[T]:
        """
        List entities with optional pagination and filtering.

        This is an alias for get_all() to match core protocol interface.
        """
        return await self.get_all(filters=filters, order_by=None, limit=limit, offset=offset)

    async def get_by_id(self, entity_id: Any) -> T | None:
        """
        Get entity by ID.

        This is an alias for get() to match core protocol interface.
        """
        return await self.get(entity_id)

    async def refresh(self, entity: T):
        """Refresh entity from database."""
        await self.session.refresh(entity)
