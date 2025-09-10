"""
Database repository implementation using the service layer pattern.

This repository uses the DatabaseService for all data access,
providing a clean separation between service and repository layers.
"""

# Use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Any, TypeVar

from src.core.base.interfaces import HealthStatus
from src.core.base.repository import BaseRepository
from src.core.exceptions import DatabaseConnectionError, DatabaseQueryError
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.database.service import DatabaseService

# Type variables
T = TypeVar("T")
K = TypeVar("K")

logger = get_logger(__name__)


class DatabaseServiceRepository(BaseRepository[T, K]):
    """
    Database repository implementation using the repository pattern.

    This repository uses the DatabaseService for all data access,
    providing a clean separation between service and repository layers.
    """

    def __init__(
        self,
        entity_type: type[T],
        key_type: type[K],
        database_service: "DatabaseService",
        name: str | None = None,
    ):
        """
        Initialize database repository.

        Args:
            entity_type: Type of entities this repository manages
            key_type: Type of primary key
            database_service: Database service instance
            name: Repository name
        """
        super().__init__(
            entity_type=entity_type,
            key_type=key_type,
            name=name or f"{entity_type.__name__}Repository",
        )

        self.database_service = database_service

        # Configure repository for database operations
        self.configure_cache(enabled=True, ttl=300)

        logger.info(f"DatabaseServiceRepository initialized for {entity_type.__name__}")

    async def _create_entity(self, entity: T) -> T:
        """Create entity using database service."""
        return await self.database_service.create_entity(entity)

    async def _get_entity_by_id(self, entity_id: K) -> T | None:
        """Get entity by ID using database service."""
        return await self.database_service.get_entity_by_id(self._entity_type, entity_id)

    async def _update_entity(self, entity: T) -> T | None:
        """Update entity using database service."""
        return await self.database_service.update_entity(entity)

    async def _delete_entity(self, entity_id: K) -> bool:
        """Delete entity using database service."""
        return await self.database_service.delete_entity(self._entity_type, entity_id)

    async def _list_entities(
        self,
        limit: int | None,
        offset: int | None,
        filters: dict[str, Any] | None,
        order_by: str | None,
        order_desc: bool,
    ) -> list[T]:
        """List entities using database service."""
        entities = await self.database_service.list_entities(
            model_class=self._entity_type,
            limit=limit,
            offset=offset or 0,
            filters=filters,
            order_by=order_by,
            order_desc=order_desc,
        )
        return entities

    async def _count_entities(self, filters: dict[str, Any] | None) -> int:
        """Count entities using database service."""
        return await self.database_service.count_entities(self._entity_type, filters)

    async def _bulk_create_entities(self, entities: list[T]) -> list[T]:
        """Bulk create entities using database service."""
        return await self.database_service.bulk_create(entities)

    async def _test_connection(self, connection: Any) -> bool:
        """Test database connection."""
        try:
            health_status = await self.database_service._service_health_check()
            return health_status == HealthStatus.HEALTHY
        except (DatabaseConnectionError, DatabaseQueryError) as e:
            self.logger.debug(f"Database health check failed: {e}")
            return False
        except Exception as e:
            self.logger.warning(f"Unexpected error in database connection test: {e}")
            return False

    async def _repository_health_check(self) -> Any:
        """Repository-specific health check."""
        return await self.database_service._service_health_check()
