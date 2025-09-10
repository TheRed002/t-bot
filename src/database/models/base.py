"""Base database models and mixins."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Integer, MetaData, String
from sqlalchemy.orm import declarative_base, declared_attr
from sqlalchemy.sql import func

# Create metadata with info to prevent redefinition errors
metadata = MetaData()
Base = declarative_base(metadata=metadata)

# Add global table config to prevent redefinition errors during imports
Base.metadata.bind = None


class TimestampMixin:
    """Mixin for automatic timestamp management."""

    @declared_attr
    def created_at(self):
        return Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    @declared_attr
    def updated_at(self):
        return Column(
            DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
        )


class AuditMixin(TimestampMixin):
    """Mixin for audit fields."""

    @declared_attr
    def created_by(self):
        return Column(String(255))

    @declared_attr
    def updated_by(self):
        return Column(String(255))

    @declared_attr
    def version(self):
        return Column(Integer, default=1, nullable=False)


class MetadataMixin:
    """Mixin for metadata storage."""

    @declared_attr
    def metadata_json(self):
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

    def update_metadata(self, data: dict[str, Any]) -> None:
        """Update multiple metadata values."""
        if not self.metadata_json:
            self.metadata_json = {}
        self.metadata_json.update(data)


class SoftDeleteMixin:
    """Mixin for soft delete functionality."""

    @declared_attr
    def deleted_at(self):
        return Column(DateTime(timezone=True))

    @declared_attr
    def deleted_by(self):
        return Column(String(255))

    @property
    def is_deleted(self) -> bool:
        """Check if record is soft deleted."""
        return self.deleted_at is not None

    def soft_delete(self, deleted_by: str | None = None) -> None:
        """Mark record as deleted."""
        self.deleted_at = datetime.now(timezone.utc)
        if deleted_by:
            self.deleted_by = deleted_by

    def restore(self) -> None:
        """Restore soft deleted record."""
        self.deleted_at = None
        self.deleted_by = None
