"""Base repository pattern implementation using core BaseRepository."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import asc, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.base.repository import BaseRepository as CoreBaseRepository
from src.core.exceptions import RepositoryError
from src.core.logging import get_logger
from src.core.types.base import ConfigDict

logger = get_logger(__name__)


class DatabaseRepository(CoreBaseRepository):
    """Database repository implementation using core BaseRepository."""

    def __init__(
        self,
        session: AsyncSession,
        model: type[Any],
        entity_type: type[Any],
        key_type: type[Any] = str,
        name: str | None = None,
        config: ConfigDict | None = None,
    ) -> None:
        """
        Initialize repository with injected session.

        Args:
            session: Injected async database session
            model: SQLAlchemy model class
            entity_type: Entity type for core repository
            key_type: Primary key type
            name: Repository name
            config: Repository configuration
        """
        # Initialize core repository
        super().__init__(
            entity_type=entity_type,
            key_type=key_type,
            name=name or f"{model.__name__}Repository",
            config=config,
        )

        self.session = session
        self.model = model

    # Core repository abstract method implementations
    async def _create_entity(self, entity) -> Any:
        """Create entity in database with boundary validation."""
        try:
            # Apply consistent boundary validation at database module boundary
            self._validate_entity_at_boundary(entity, "create")

            self.session.add(entity)
            await self.session.flush()
            return entity
        except Exception as e:
            await self.session.rollback()
            # Use consistent error propagation patterns
            try:
                from src.utils.messaging_patterns import ErrorPropagationMixin

                error_propagator = ErrorPropagationMixin()
                error_propagator.propagate_database_error(
                    e, f"repository_{self.model.__name__.lower()}_create"
                )
            except ImportError:
                # Fallback to direct raise if pattern not available
                raise RepositoryError(f"Failed to create {self.model.__name__}: {e}") from e

    async def _get_entity_by_id(self, entity_id: Any) -> Any | None:
        """Get entity by ID from database."""
        try:
            stmt: Any = select(self.model).where(self.model.id == entity_id)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            raise RepositoryError(f"Failed to get {self.model.__name__} by ID: {e}") from e

    async def _update_entity(self, entity: Any) -> Any | None:
        """Update entity in database with consistent patterns."""
        try:
            # Apply consistent boundary validation at database module boundary
            self._validate_entity_at_boundary(entity, "update")

            # Apply consistent data transformation for updates
            self._apply_update_transforms(entity)

            # Use merge for detached instances
            merged = await self.session.merge(entity)
            await self.session.flush()
            return merged
        except Exception as e:
            await self.session.rollback()
            # Use consistent error propagation patterns
            try:
                from src.utils.messaging_patterns import ErrorPropagationMixin

                error_propagator = ErrorPropagationMixin()
                error_propagator.propagate_database_error(
                    e, f"repository_{self.model.__name__.lower()}_update"
                )
            except ImportError:
                # Fallback to direct raise if pattern not available
                raise RepositoryError(f"Failed to update {self.model.__name__}: {e}") from e

    async def _delete_entity(self, entity_id) -> bool:
        """Delete entity from database."""
        try:
            entity = await self._get_entity_by_id(entity_id)
            if entity:
                self.session.delete(entity)
                await self.session.flush()
                return True
            return False
        except Exception as e:
            await self.session.rollback()
            # Use consistent error propagation patterns
            try:
                from src.utils.messaging_patterns import ErrorPropagationMixin

                error_propagator = ErrorPropagationMixin()
                error_propagator.propagate_database_error(
                    e, f"repository_{self.model.__name__.lower()}_delete"
                )
            except ImportError:
                # Fallback to direct raise if pattern not available
                raise RepositoryError(f"Failed to delete {self.model.__name__}: {e}") from e

    async def _list_entities(
        self,
        limit: int | None,
        offset: int | None,
        filters: dict[str, Any] | None,
        order_by: str | None,
        order_desc: bool,
    ) -> list:
        """List entities from database."""
        try:
            stmt: Any = select(self.model)

            # Apply filtering with consistent patterns and validation
            if filters:
                stmt = self._apply_consistent_filters(stmt, filters)

            # Apply ordering
            if order_by and hasattr(self.model, order_by):
                order_field = getattr(self.model, order_by)
                if order_desc:
                    stmt = stmt.order_by(desc(order_field))
                else:
                    stmt = stmt.order_by(asc(order_field))

            # Apply pagination
            if offset:
                stmt = stmt.offset(offset)
            if limit:
                stmt = stmt.limit(limit)

            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            raise RepositoryError(f"Failed to list {self.model.__name__}: {e}") from e

    async def _count_entities(self, filters: dict[str, Any] | None) -> int:
        """Count entities in database."""
        try:
            stmt = select(func.count()).select_from(self.model)

            if filters:
                for key, value in filters.items():
                    if hasattr(self.model, key):
                        if isinstance(value, list):
                            stmt = stmt.where(getattr(self.model, key).in_(value))
                        else:
                            stmt = stmt.where(getattr(self.model, key) == value)

            result = await self.session.execute(stmt)
            return result.scalar() or 0
        except Exception as e:
            raise RepositoryError(f"Failed to count {self.model.__name__}: {e}") from e

    # Additional utility methods for backward compatibility
    async def get_by(self, **kwargs):
        """Get entity by attributes."""
        try:
            stmt = select(self.model)
            for key, value in kwargs.items():
                if hasattr(self.model, key):
                    stmt = stmt.where(getattr(self.model, key) == value)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            raise RepositoryError(f"Failed to get {self.model.__name__} by attributes: {e}") from e

    async def get(self, entity_id: Any):
        """Get entity by ID - alias for get_by_id for backward compatibility."""
        return await self.get_by_id(entity_id)

    async def get_all(
        self,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list:
        """Get all entities with optional filtering - backward compatibility."""
        # Map old order_by format to new format
        order_desc = False
        if order_by and order_by.startswith("-"):
            order_desc = True
            order_by = order_by[1:]

        return await self.list(
            limit=limit,
            offset=offset,
            filters=filters,
            order_by=order_by,
            order_desc=order_desc,
        )

    async def exists(self, entity_id: Any) -> bool:
        """Check if entity exists."""
        try:
            stmt: Any = select(self.model.id).where(self.model.id == entity_id)
            result = await self.session.execute(stmt)
            return result.scalar() is not None
        except Exception as e:
            raise RepositoryError(f"Failed to check existence of {self.model.__name__}: {e}") from e

    async def soft_delete(self, entity_id: Any, deleted_by: str | None = None) -> bool:
        """Soft delete entity if it supports it."""
        try:
            entity = await self._get_entity_by_id(entity_id)
            if entity and hasattr(entity, "soft_delete"):
                entity.soft_delete(deleted_by)
                await self._update_entity(entity)
                return True
            return False
        except Exception as e:
            raise RepositoryError(f"Failed to soft delete {self.model.__name__}: {e}") from e

    # Transaction management
    async def begin(self):
        """Begin transaction."""
        return await self.session.begin()

    async def commit(self):
        """Commit transaction."""
        await self.session.commit()

    async def rollback(self):
        """Rollback transaction."""
        await self.session.rollback()

    async def refresh(self, entity):
        """Refresh entity from database."""
        await self.session.refresh(entity)

    def _apply_update_transforms(self, entity) -> None:
        """Apply consistent update transformations matching error_handling module patterns."""
        # Update timestamp consistently
        if hasattr(entity, "updated_at"):
            entity.updated_at = datetime.now(timezone.utc)

        # Increment version for optimistic locking
        if hasattr(entity, "version"):
            entity.version = (entity.version or 0) + 1

        # Core already handles data transformation - no need for duplicate logic

    def _apply_consistent_filters(self, stmt, filters: dict[str, Any]):
        """Apply consistent filtering patterns across all queries."""
        for key, value in filters.items():
            if hasattr(self.model, key):
                column = getattr(self.model, key)
                stmt = self._apply_single_filter(stmt, column, value)
        return stmt

    def _apply_single_filter(self, stmt, column, value):
        """Apply a single filter condition to the statement."""
        if isinstance(value, list):
            return stmt.where(column.in_(value))
        elif isinstance(value, dict):
            return self._apply_complex_filter(stmt, column, value)
        else:
            return stmt.where(column == value)

    def _apply_complex_filter(self, stmt, column, value_dict: dict[str, Any]):
        """Apply complex filter conditions from dictionary."""
        if "gt" in value_dict:
            stmt = stmt.where(column > value_dict["gt"])
        if "gte" in value_dict:
            stmt = stmt.where(column >= value_dict["gte"])
        if "lt" in value_dict:
            stmt = stmt.where(column < value_dict["lt"])
        if "lte" in value_dict:
            stmt = stmt.where(column <= value_dict["lte"])
        if "like" in value_dict:
            stmt = stmt.where(column.like(f"%{value_dict['like']}%"))
        if self._has_valid_between_filter(value_dict):
            stmt = stmt.where(column.between(value_dict["between"][0], value_dict["between"][1]))
        return stmt

    def _has_valid_between_filter(self, value_dict: dict[str, Any]) -> bool:
        """Check if the dictionary has a valid 'between' filter."""
        return (
            "between" in value_dict
            and isinstance(value_dict["between"], (list, tuple))
            and len(value_dict["between"]) == 2
        )

    def _validate_entity_at_boundary(self, entity: Any, operation: str) -> None:
        """Validate entity at database module boundary."""
        try:
            # Use consistent boundary validation patterns
            from src.utils.messaging_patterns import BoundaryValidator

            # Convert entity to dict for validation if needed
            if hasattr(entity, "__dict__"):
                entity_dict = {k: v for k, v in entity.__dict__.items() if not k.startswith("_")}
            else:
                entity_dict = entity

            # Apply database-specific boundary validation
            BoundaryValidator.validate_database_entity(entity_dict, operation)

        except ImportError:
            # Fallback: basic validation if pattern not available
            if entity is None:
                raise ValueError(f"Cannot {operation} None entity")

            # Basic validation for financial fields if present
            financial_fields = ["price", "quantity", "volume", "value"]
            for field in financial_fields:
                if hasattr(entity, field):
                    value = getattr(entity, field)
                    if value is not None and value < 0:
                        raise ValueError(f"Financial field {field} cannot be negative")

        except Exception as e:
            logger.error(f"Entity boundary validation failed for {operation}: {e}")
            # Allow operation to continue with warning for non-critical validation failures
            logger.warning(f"Proceeding with {operation} despite validation warning")


# Backward compatibility aliases
BaseRepository = DatabaseRepository


# For interface compatibility with tests
class RepositoryInterface:
    """Interface for repository pattern."""

    pass
