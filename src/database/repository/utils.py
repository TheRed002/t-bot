"""Common repository utilities to eliminate code duplication."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Generic, TypeVar

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import RepositoryError
from src.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class RepositoryUtils(Generic[T]):
    """Common utilities for repository operations."""

    @staticmethod
    async def update_entity_status(
        repository: Any, entity_id: str, status: str, entity_name: str, **additional_fields: Any
    ) -> bool:
        """
        Update entity status with consistent pattern.

        Args:
            repository: Repository instance
            entity_id: Entity ID to update
            status: New status value
            entity_name: Entity name for logging
            **additional_fields: Additional fields to update

        Returns:
            bool: True if updated, False if entity not found
        """
        try:
            entity = await repository.get(entity_id)
            if entity:
                entity.status = status
                for field, value in additional_fields.items():
                    if hasattr(entity, field):
                        setattr(entity, field, value)
                await repository.update(entity)
                logger.debug(f"Updated {entity_name} {entity_id} status to {status}")
                return True
            logger.warning(f"{entity_name} {entity_id} not found for status update")
            return False
        except Exception as e:
            logger.error(f"Failed to update {entity_name} {entity_id} status: {e}")
            raise RepositoryError(f"Failed to update {entity_name} status: {e}") from e

    @staticmethod
    async def update_entity_fields(
        repository: Any, entity_id: str, entity_name: str, **fields: Any
    ) -> bool:
        """
        Update entity fields with consistent pattern.

        Args:
            repository: Repository instance
            entity_id: Entity ID to update
            entity_name: Entity name for logging
            **fields: Fields to update

        Returns:
            bool: True if updated, False if entity not found
        """
        try:
            entity = await repository.get(entity_id)
            if entity:
                for field, value in fields.items():
                    if hasattr(entity, field):
                        setattr(entity, field, value)
                await repository.update(entity)
                logger.debug(f"Updated {entity_name} {entity_id} fields: {list(fields.keys())}")
                return True
            logger.warning(f"{entity_name} {entity_id} not found for field update")
            return False
        except Exception as e:
            logger.error(f"Failed to update {entity_name} {entity_id} fields: {e}")
            raise RepositoryError(f"Failed to update {entity_name} fields: {e}") from e

    @staticmethod
    async def get_entities_by_field(
        repository: Any, field_name: str, field_value: Any, order_by: str = "-created_at"
    ) -> list[T]:
        """
        Get entities by single field value with consistent pattern.

        Args:
            repository: Repository instance
            field_name: Field name to filter by
            field_value: Field value to match
            order_by: Order by field (default: -created_at)

        Returns:
            list: List of matching entities
        """
        try:
            filters = {field_name: field_value}
            return await repository.get_all(filters=filters, order_by=order_by)
        except Exception as e:
            logger.error(f"Failed to get entities by {field_name}: {e}")
            raise RepositoryError(f"Failed to get entities by {field_name}: {e}") from e

    @staticmethod
    async def get_entities_by_multiple_fields(
        repository: Any, filters: dict[str, Any], order_by: str = "-created_at"
    ) -> list[T]:
        """
        Get entities by multiple fields with consistent pattern.

        Args:
            repository: Repository instance
            filters: Dictionary of field filters
            order_by: Order by field (default: -created_at)

        Returns:
            list: List of matching entities
        """
        try:
            return await repository.get_all(filters=filters, order_by=order_by)
        except Exception as e:
            logger.error(f"Failed to get entities by filters {filters}: {e}")
            raise RepositoryError(f"Failed to get entities by filters: {e}") from e

    @staticmethod
    async def get_recent_entities(
        repository: Any,
        hours: int = 24,
        additional_filters: dict[str, Any] | None = None,
        order_by: str = "-created_at",
        timestamp_field: str = "created_at",
    ) -> list[T]:
        """
        Get recent entities within time range with consistent pattern.

        Args:
            repository: Repository instance
            hours: Number of hours to look back
            additional_filters: Additional filters to apply
            order_by: Order by field (default: -created_at)
            timestamp_field: Field to use for time filtering

        Returns:
            list: List of recent entities
        """
        try:
            return await RepositoryUtils.execute_time_based_query(
                repository.session,
                repository.model,
                timestamp_field=timestamp_field,
                hours=hours,
                additional_filters=additional_filters,
                order_by=order_by,
            )
        except Exception as e:
            logger.error(f"Failed to get recent entities: {e}")
            raise RepositoryError(f"Failed to get recent entities: {e}") from e

    @staticmethod
    async def mark_entity_field(
        repository: Any, entity_id: str, field_name: str, field_value: Any, entity_name: str
    ) -> bool:
        """
        Mark entity field with consistent pattern (e.g., mark_as_read).

        Args:
            repository: Repository instance
            entity_id: Entity ID to update
            field_name: Field name to update
            field_value: New field value
            entity_name: Entity name for logging

        Returns:
            bool: True if marked, False if entity not found
        """
        try:
            entity = await repository.get(entity_id)
            if entity:
                setattr(entity, field_name, field_value)
                await repository.update(entity)
                logger.debug(f"Marked {entity_name} {entity_id} {field_name} as {field_value}")
                return True
            logger.warning(f"{entity_name} {entity_id} not found for marking")
            return False
        except Exception as e:
            logger.error(f"Failed to mark {entity_name} {entity_id}: {e}")
            raise RepositoryError(f"Failed to mark {entity_name}: {e}") from e

    @staticmethod
    async def bulk_mark_entities(
        repository: Any, entities: list[T], field_name: str, field_value: Any, entity_name: str
    ) -> int:
        """
        Bulk mark entities with consistent pattern.

        Args:
            repository: Repository instance
            entities: List of entities to update
            field_name: Field name to update
            field_value: New field value
            entity_name: Entity name for logging

        Returns:
            int: Number of entities updated
        """
        try:
            count = 0
            for entity in entities:
                setattr(entity, field_name, field_value)
                await repository.update(entity)
                count += 1
            logger.debug(
                f"Bulk marked {count} {entity_name} entities {field_name} as {field_value}"
            )
            return count
        except Exception as e:
            logger.error(f"Failed to bulk mark {entity_name} entities: {e}")
            raise RepositoryError(f"Failed to bulk mark {entity_name} entities: {e}") from e

    @staticmethod
    async def get_total_by_field_aggregation(
        session: AsyncSession,
        model: Any,
        field_name: str,
        sum_field: str,
        filters: dict[str, Any] | None = None,
    ) -> Decimal:
        """
        Get total aggregation by field with consistent pattern.

        Args:
            session: Database session
            model: SQLAlchemy model
            field_name: Field to group by
            sum_field: Field to sum
            filters: Additional filters

        Returns:
            Decimal: Total sum
        """
        try:
            from sqlalchemy import select

            stmt = select(func.sum(getattr(model, sum_field))).select_from(model)

            if filters:
                for key, value in filters.items():
                    if hasattr(model, key):
                        column = getattr(model, key)
                        if isinstance(value, list):
                            stmt = stmt.where(column.in_(value))
                        else:
                            stmt = stmt.where(column == value)

            result = await session.execute(stmt)
            total = result.scalar()
            return Decimal(str(total)) if total else Decimal("0")
        except Exception as e:
            logger.error(f"Failed to get total {sum_field} by {field_name}: {e}")
            raise RepositoryError(f"Failed to get total aggregation: {e}") from e

    @staticmethod
    async def get_latest_entity_by_field(
        repository: Any, field_name: str, field_value: Any
    ) -> T | None:
        """
        Get latest entity by field with consistent pattern.

        Args:
            repository: Repository instance
            field_name: Field name to filter by
            field_value: Field value to match

        Returns:
            Entity or None: Latest matching entity
        """
        try:
            entities = await repository.get_all(
                filters={field_name: field_value}, order_by="-created_at", limit=1
            )
            return entities[0] if entities else None
        except Exception as e:
            logger.error(f"Failed to get latest entity by {field_name}: {e}")
            raise RepositoryError(f"Failed to get latest entity by {field_name}: {e}") from e

    @staticmethod
    async def cleanup_old_entities(
        session: AsyncSession,
        model: Any,
        days: int = 30,
        date_field: str = "created_at",
        additional_filters: dict[str, Any] | None = None,
    ) -> int:
        """
        Cleanup old entities with consistent pattern.

        Args:
            session: Database session
            model: SQLAlchemy model
            days: Number of days to keep
            date_field: Date field to check
            additional_filters: Additional filters

        Returns:
            int: Number of entities deleted
        """
        try:
            from sqlalchemy import delete

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

            stmt = delete(model).where(getattr(model, date_field) < cutoff_date)

            if additional_filters:
                for key, value in additional_filters.items():
                    if hasattr(model, key):
                        column = getattr(model, key)
                        if isinstance(value, list):
                            stmt = stmt.where(column.in_(value))
                        else:
                            stmt = stmt.where(column == value)

            result = await session.execute(stmt)
            deleted_count = result.rowcount
            logger.info(f"Cleaned up {deleted_count} old {model.__name__} entities")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old {model.__name__} entities: {e}")
            raise RepositoryError(f"Failed to cleanup old entities: {e}") from e

    @staticmethod
    async def execute_time_based_query(
        session: AsyncSession,
        model: Any,
        timestamp_field: str = "created_at",
        hours: int | None = None,
        days: int | None = None,
        additional_filters: dict[str, Any] | None = None,
        order_by: str = "-created_at",
        limit: int | None = None,
    ) -> list[T]:
        """
        Execute time-based query with consistent pattern.

        Args:
            session: Database session
            model: SQLAlchemy model
            timestamp_field: Field to use for time filtering
            hours: Number of hours to look back (mutually exclusive with days)
            days: Number of days to look back (mutually exclusive with hours)
            additional_filters: Additional filters to apply
            order_by: Order by field
            limit: Limit number of results

        Returns:
            list: List of matching entities
        """
        try:
            from sqlalchemy import asc, desc, select

            if hours is not None and days is not None:
                raise ValueError("Cannot specify both hours and days")

            if hours is None and days is None:
                raise ValueError("Must specify either hours or days")

            # Calculate cutoff time
            if hours is not None:
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            else:
                cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)

            stmt = select(model).where(getattr(model, timestamp_field) >= cutoff_time)

            # Apply additional filters
            if additional_filters:
                for key, value in additional_filters.items():
                    if hasattr(model, key):
                        column = getattr(model, key)
                        if isinstance(value, list):
                            stmt = stmt.where(column.in_(value))
                        else:
                            stmt = stmt.where(column == value)

            # Apply ordering
            if order_by and hasattr(model, order_by.lstrip("-")):
                order_field = getattr(model, order_by.lstrip("-"))
                if order_by.startswith("-"):
                    stmt = stmt.order_by(desc(order_field))
                else:
                    stmt = stmt.order_by(asc(order_field))

            # Apply limit
            if limit:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to execute time-based query: {e}")
            raise RepositoryError(f"Failed to execute time-based query: {e}") from e

    @staticmethod
    async def execute_date_range_query(
        session: AsyncSession,
        model: Any,
        start_date: datetime,
        end_date: datetime,
        timestamp_field: str = "created_at",
        additional_filters: dict[str, Any] | None = None,
        order_by: str = "-created_at",
        limit: int | None = None,
    ) -> list[T]:
        """
        Execute date range query with consistent pattern.

        Args:
            session: Database session
            model: SQLAlchemy model
            start_date: Start date for range
            end_date: End date for range
            timestamp_field: Field to use for date filtering
            additional_filters: Additional filters to apply
            order_by: Order by field
            limit: Limit number of results

        Returns:
            list: List of matching entities
        """
        try:
            from sqlalchemy import asc, desc, select

            stmt = select(model).where(
                getattr(model, timestamp_field).between(start_date, end_date)
            )

            # Apply additional filters
            if additional_filters:
                for key, value in additional_filters.items():
                    if hasattr(model, key):
                        column = getattr(model, key)
                        if isinstance(value, list):
                            stmt = stmt.where(column.in_(value))
                        else:
                            stmt = stmt.where(column == value)

            # Apply ordering
            if order_by and hasattr(model, order_by.lstrip("-")):
                order_field = getattr(model, order_by.lstrip("-"))
                if order_by.startswith("-"):
                    stmt = stmt.order_by(desc(order_field))
                else:
                    stmt = stmt.order_by(asc(order_field))

            # Apply limit
            if limit:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to execute date range query: {e}")
            raise RepositoryError(f"Failed to execute date range query: {e}") from e
