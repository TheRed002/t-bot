"""Base database models and mixins."""

from datetime import datetime
from typing import Any, Dict
from sqlalchemy import Column, DateTime, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.sql import func

Base = declarative_base()


class TimestampMixin:
    """Mixin for automatic timestamp management."""
    
    @declared_attr
    def created_at(cls):
        return Column(
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False
        )
    
    @declared_attr
    def updated_at(cls):
        return Column(
            DateTime(timezone=True),
            server_default=func.now(),
            onupdate=func.now(),
            nullable=False
        )


class AuditMixin(TimestampMixin):
    """Mixin for audit fields."""
    
    @declared_attr
    def created_by(cls):
        return Column(String(255))
    
    @declared_attr
    def updated_by(cls):
        return Column(String(255))
    
    @declared_attr
    def version(cls):
        return Column(Integer, default=1, nullable=False)


class MetadataMixin:
    """Mixin for metadata storage."""
    
    @declared_attr
    def metadata_json(cls):
        return Column(JSON, default={})
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        if self.metadata_json:
            return self.metadata_json.get(key, default)
        return default
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        if not self.metadata_json:
            self.metadata_json = {}
        self.metadata_json[key] = value
    
    def update_metadata(self, data: Dict[str, Any]) -> None:
        """Update multiple metadata values."""
        if not self.metadata_json:
            self.metadata_json = {}
        self.metadata_json.update(data)


class SoftDeleteMixin:
    """Mixin for soft delete functionality."""
    
    @declared_attr
    def deleted_at(cls):
        return Column(DateTime(timezone=True))
    
    @declared_attr
    def deleted_by(cls):
        return Column(String(255))
    
    @property
    def is_deleted(self) -> bool:
        """Check if record is soft deleted."""
        return self.deleted_at is not None
    
    def soft_delete(self, deleted_by: str = None) -> None:
        """Mark record as deleted."""
        self.deleted_at = datetime.utcnow()
        if deleted_by:
            self.deleted_by = deleted_by
    
    def restore(self) -> None:
        """Restore soft deleted record."""
        self.deleted_at = None
        self.deleted_by = None