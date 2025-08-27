"""
Interface adapter to align database repository with core protocol.

This module provides adapters to ensure database repositories
comply with the core Repository protocol interface.
"""

from typing import Any, TypeVar, Generic
from src.core.base.interfaces import Repository as CoreRepository
from src.database.repository.base import BaseRepository

T = TypeVar("T")
K = TypeVar("K")


class RepositoryAdapter(Generic[T, K]):
    """
    Adapter that wraps a database repository to match core protocol.
    
    This adapter translates between the database layer's get_all() method
    and the core protocol's list() method, ensuring interface compatibility.
    """
    
    def __init__(self, repository: BaseRepository[T]):
        """
        Initialize adapter with a database repository.
        
        Args:
            repository: The database repository to adapt
        """
        self._repository = repository
    
    async def create(self, entity: T) -> T:
        """Create new entity."""
        return await self._repository.create(entity)
    
    async def get_by_id(self, entity_id: K) -> T | None:
        """Get entity by ID."""
        return await self._repository.get(entity_id)
    
    async def update(self, entity: T) -> T:
        """Update existing entity."""
        return await self._repository.update(entity)
    
    async def delete(self, entity_id: K) -> bool:
        """Delete entity by ID."""
        return await self._repository.delete(entity_id)
    
    async def list(
        self,
        limit: int | None = None,
        offset: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[T]:
        """
        List entities with optional pagination and filtering.
        
        This method adapts the parameter order to match core protocol.
        """
        # Note: database layer expects order_by as a parameter,
        # but core protocol doesn't include it in the interface
        return await self._repository.get_all(
            filters=filters,
            order_by=None,  # Default ordering
            limit=limit,
            offset=offset
        )
    
    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count entities with optional filtering."""
        # If the repository doesn't have a count method, 
        # fall back to counting the results
        if hasattr(self._repository, 'count'):
            return await self._repository.count(filters)
        else:
            # Inefficient fallback - get all and count
            results = await self._repository.get_all(filters=filters)
            return len(results)
    
    # Additional database-specific methods can be accessed directly
    def __getattr__(self, name: str) -> Any:
        """Forward any other method calls to the wrapped repository."""
        return getattr(self._repository, name)