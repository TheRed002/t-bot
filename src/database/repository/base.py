"""Base repository pattern implementation."""

from typing import Generic, TypeVar, Type, List, Optional, Dict, Any
from abc import ABC, abstractmethod
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from sqlalchemy.exc import IntegrityError, DataError

from src.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class RepositoryInterface(ABC, Generic[T]):
    """Repository interface."""
    
    @abstractmethod
    async def get(self, id: Any) -> Optional[T]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    async def get_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[T]:
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
    
    def __init__(self, session: Session, model: Type[T]):
        """
        Initialize repository.
        
        Args:
            session: Database session
            model: Model class
        """
        self.session = session
        self.model = model
        self._logger = logger
    
    async def get(self, id: Any) -> Optional[T]:
        """Get entity by ID."""
        try:
            return self.session.query(self.model).filter(
                self.model.id == id
            ).first()
        except Exception as e:
            self._logger.error(f"Error getting {self.model.__name__} by ID {id}: {e}")
            return None
    
    async def get_by(self, **kwargs) -> Optional[T]:
        """Get entity by attributes."""
        try:
            query = self.session.query(self.model)
            for key, value in kwargs.items():
                query = query.filter(getattr(self.model, key) == value)
            return query.first()
        except Exception as e:
            self._logger.error(f"Error getting {self.model.__name__} by {kwargs}: {e}")
            return None
    
    async def get_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[T]:
        """Get all entities with optional filtering."""
        try:
            query = self.session.query(self.model)
            
            # Apply filters
            if filters:
                for key, value in filters.items():
                    if hasattr(self.model, key):
                        if isinstance(value, list):
                            query = query.filter(getattr(self.model, key).in_(value))
                        elif isinstance(value, dict):
                            # Handle complex filters like {'gt': 100, 'lt': 200}
                            column = getattr(self.model, key)
                            if 'gt' in value:
                                query = query.filter(column > value['gt'])
                            if 'gte' in value:
                                query = query.filter(column >= value['gte'])
                            if 'lt' in value:
                                query = query.filter(column < value['lt'])
                            if 'lte' in value:
                                query = query.filter(column <= value['lte'])
                            if 'like' in value:
                                query = query.filter(column.like(f"%{value['like']}%"))
                        else:
                            query = query.filter(getattr(self.model, key) == value)
            
            # Apply ordering
            if order_by:
                if order_by.startswith('-'):
                    query = query.order_by(desc(getattr(self.model, order_by[1:])))
                else:
                    query = query.order_by(asc(getattr(self.model, order_by)))
            
            # Apply pagination
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)
            
            return query.all()
            
        except Exception as e:
            self._logger.error(f"Error getting all {self.model.__name__}: {e}")
            return []
    
    async def create(self, entity: T) -> T:
        """Create new entity."""
        try:
            self.session.add(entity)
            self.session.flush()
            return entity
        except IntegrityError as e:
            self._logger.error(f"Integrity error creating {self.model.__name__}: {e}")
            self.session.rollback()
            raise
        except Exception as e:
            self._logger.error(f"Error creating {self.model.__name__}: {e}")
            self.session.rollback()
            raise
    
    async def create_many(self, entities: List[T]) -> List[T]:
        """Create multiple entities."""
        try:
            self.session.add_all(entities)
            self.session.flush()
            return entities
        except Exception as e:
            self._logger.error(f"Error creating multiple {self.model.__name__}: {e}")
            self.session.rollback()
            raise
    
    async def update(self, entity: T) -> T:
        """Update existing entity."""
        try:
            # Update timestamp if model has it
            if hasattr(entity, 'updated_at'):
                entity.updated_at = datetime.utcnow()
            
            # Increment version if model has it
            if hasattr(entity, 'version'):
                entity.version = (entity.version or 0) + 1
            
            self.session.merge(entity)
            self.session.flush()
            return entity
        except Exception as e:
            self._logger.error(f"Error updating {self.model.__name__}: {e}")
            self.session.rollback()
            raise
    
    async def delete(self, id: Any) -> bool:
        """Delete entity by ID."""
        try:
            entity = await self.get(id)
            if entity:
                self.session.delete(entity)
                self.session.flush()
                return True
            return False
        except Exception as e:
            self._logger.error(f"Error deleting {self.model.__name__} {id}: {e}")
            self.session.rollback()
            return False
    
    async def soft_delete(self, id: Any, deleted_by: str = None) -> bool:
        """Soft delete entity if it supports it."""
        try:
            entity = await self.get(id)
            if entity and hasattr(entity, 'soft_delete'):
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
            return self.session.query(
                self.session.query(self.model).filter(
                    self.model.id == id
                ).exists()
            ).scalar()
        except Exception as e:
            self._logger.error(f"Error checking existence of {self.model.__name__} {id}: {e}")
            return False
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities."""
        try:
            query = self.session.query(self.model)
            
            if filters:
                for key, value in filters.items():
                    if hasattr(self.model, key):
                        query = query.filter(getattr(self.model, key) == value)
            
            return query.count()
        except Exception as e:
            self._logger.error(f"Error counting {self.model.__name__}: {e}")
            return 0
    
    def begin(self):
        """Begin transaction."""
        return self.session.begin()
    
    def commit(self):
        """Commit transaction."""
        self.session.commit()
    
    def rollback(self):
        """Rollback transaction."""
        self.session.rollback()
    
    def refresh(self, entity: T):
        """Refresh entity from database."""
        self.session.refresh(entity)