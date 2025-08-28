"""
State management database models for PostgreSQL persistence.

These models provide ACID-compliant state storage with versioning,
checkpointing, audit trails, and backup/restore capabilities.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, relationship
from sqlalchemy.sql import func

from .base import AuditMixin, Base, MetadataMixin, SoftDeleteMixin, TimestampMixin


class StateSnapshot(Base, AuditMixin, MetadataMixin, SoftDeleteMixin):
    """
    State snapshot table for point-in-time state captures.

    Stores complete system state snapshots for recovery and audit purposes.
    Uses JSONB for efficient querying and indexing of state data.
    """

    __tablename__ = "state_snapshots"

    # Primary key
    snapshot_id = Column(UUID(as_uuid=True), primary_key=True)

    # Snapshot metadata
    name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    snapshot_type: Mapped[str] = Column(
        Enum("manual", "automatic", "checkpoint", "backup", name="snapshot_type_enum"),
        nullable=False,
        default="manual",
    )

    # State data (using JSONB for PostgreSQL)
    state_data = Column(JSONB, nullable=False)

    # Compression and storage info
    is_compressed = Column(Boolean, default=False, nullable=False)
    compression_type = Column(String(50), nullable=True)  # gzip, lz4, etc.
    raw_size_bytes = Column(BigInteger, nullable=False, default=0)
    compressed_size_bytes = Column(BigInteger, nullable=True)

    # Versioning
    schema_version = Column(String(50), nullable=False, default="1.0.0")
    state_checksum = Column(String(128), nullable=False)  # SHA-256 hash

    # Status tracking
    status: Mapped[str] = Column(
        Enum("creating", "ready", "corrupted", "archived", name="snapshot_status_enum"),
        nullable=False,
        default="creating",
    )

    # Retention and cleanup
    retention_days = Column(Integer, nullable=True)  # Auto-delete after N days
    is_protected = Column(Boolean, default=False, nullable=False)  # Prevent auto-deletion

    # Relationships
    checkpoints = relationship(
        "StateCheckpoint", back_populates="snapshot", cascade="all, delete-orphan"
    )

    # Indexes for performance
    __table_args__ = (
        Index("ix_state_snapshots_created_at", "created_at"),
        Index("ix_state_snapshots_type", "snapshot_type"),
        Index("ix_state_snapshots_status", "status"),
        Index("ix_state_snapshots_checksum", "state_checksum"),
        Index("ix_state_snapshots_protected", "is_protected"),
    )

    @hybrid_property
    def compression_ratio(self) -> float:
        """Calculate compression ratio if compressed."""
        if self.is_compressed and self.compressed_size_bytes and self.raw_size_bytes:
            return self.compressed_size_bytes / self.raw_size_bytes
        return 1.0

    def get_state_by_type(self, state_type: str) -> dict[str, Any] | None:
        """Get state data for a specific state type."""
        if self.state_data and isinstance(self.state_data, dict):
            return self.state_data.get("states", {}).get(state_type)
        return None

    def get_state_count(self) -> int:
        """Get total number of states in this snapshot."""
        if self.state_data and isinstance(self.state_data, dict):
            states = self.state_data.get("states", {})
            return sum(len(state_dict) for state_dict in states.values())
        return 0


class StateCheckpoint(Base, AuditMixin, MetadataMixin):
    """
    State checkpoint table for incremental state saves.

    Stores incremental state changes and references to base snapshots
    for efficient storage and fast recovery.
    """

    __tablename__ = "state_checkpoints"

    # Primary key
    checkpoint_id = Column(UUID(as_uuid=True), primary_key=True)

    # Checkpoint metadata
    name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    checkpoint_type: Mapped[str] = Column(
        Enum("incremental", "full", "emergency", name="checkpoint_type_enum"),
        nullable=False,
        default="incremental",
    )

    # Reference to base snapshot
    base_snapshot_id = Column(
        UUID(as_uuid=True), ForeignKey("state_snapshots.snapshot_id"), nullable=True
    )
    snapshot = relationship(
        "StateSnapshot", back_populates="checkpoints", foreign_keys=[base_snapshot_id]
    )

    # Previous checkpoint for chain
    previous_checkpoint_id = Column(
        UUID(as_uuid=True), ForeignKey("state_checkpoints.checkpoint_id"), nullable=True
    )
    previous_checkpoint = relationship(
        "StateCheckpoint", remote_side="StateCheckpoint.checkpoint_id"
    )

    # State changes (delta from base or previous checkpoint)
    state_changes = Column(JSONB, nullable=False)  # Only changed states

    # Change tracking
    changes_count = Column(Integer, nullable=False, default=0)
    states_affected = Column(JSONB, nullable=True)  # List of state IDs affected

    # Storage info
    size_bytes = Column(BigInteger, nullable=False, default=0)
    is_compressed = Column(Boolean, default=False, nullable=False)

    # Validation
    changes_checksum = Column(String(128), nullable=False)

    # Status
    status: Mapped[str] = Column(
        Enum("creating", "ready", "applied", "rolled_back", name="checkpoint_status_enum"),
        nullable=False,
        default="creating",
    )

    # Cleanup
    retention_days = Column(Integer, nullable=True)

    # Indexes
    __table_args__ = (
        Index("ix_state_checkpoints_created_at", "created_at"),
        Index("ix_state_checkpoints_type", "checkpoint_type"),
        Index("ix_state_checkpoints_status", "status"),
        Index("ix_state_checkpoints_base_snapshot", "base_snapshot_id"),
        Index("ix_state_checkpoints_previous", "previous_checkpoint_id"),
    )

    def get_changed_state_ids(self) -> set[str]:
        """Get set of state IDs that changed in this checkpoint."""
        if self.states_affected:
            return set(self.states_affected)
        return set()

    def has_state_type(self, state_type: str) -> bool:
        """Check if this checkpoint contains changes for a state type."""
        if self.state_changes and isinstance(self.state_changes, dict):
            return state_type in self.state_changes
        return False


class StateHistory(Base, TimestampMixin, MetadataMixin):
    """
    State history table for detailed audit trail.

    Records every state change with full before/after data
    for comprehensive audit and debugging capabilities.
    """

    __tablename__ = "state_history"

    # Primary key
    history_id = Column(UUID(as_uuid=True), primary_key=True)

    # State identification
    state_type = Column(String(100), nullable=False, index=True)
    state_id = Column(String(255), nullable=False, index=True)

    # Operation details
    operation: Mapped[str] = Column(
        Enum("create", "update", "delete", "restore", "sync", name="state_operation_enum"),
        nullable=False,
    )
    operation_id = Column(UUID(as_uuid=True), nullable=True)  # Link to persistence operation

    # State data
    old_state = Column(JSONB, nullable=True)  # State before change
    new_state = Column(JSONB, nullable=True)  # State after change
    changed_fields = Column(JSONB, nullable=True)  # List of changed field names

    # Change metadata
    source_component = Column(String(255), nullable=True)
    user_id = Column(String(255), nullable=True)
    reason = Column(Text, nullable=True)
    priority = Column(
        Enum("critical", "high", "medium", "low", name="state_priority_enum"),
        nullable=False,
        default="medium",
    )

    # Validation and checksums
    old_checksum = Column(String(128), nullable=True)
    new_checksum = Column(String(128), nullable=True)

    # Size tracking
    old_size_bytes = Column(Integer, nullable=True)
    new_size_bytes = Column(Integer, nullable=True)

    # Status flags
    applied = Column(Boolean, default=True, nullable=False)
    synchronized = Column(Boolean, default=False, nullable=False)
    persisted = Column(Boolean, default=True, nullable=False)

    # Rollback support
    can_rollback = Column(Boolean, default=True, nullable=False)
    rolled_back = Column(Boolean, default=False, nullable=False)
    rollback_checkpoint_id = Column(UUID(as_uuid=True), nullable=True)

    # Indexes for efficient querying
    __table_args__ = (
        Index("ix_state_history_state_type_id", "state_type", "state_id"),
        Index("ix_state_history_operation", "operation"),
        Index("ix_state_history_created_at", "created_at"),
        Index("ix_state_history_source_component", "source_component"),
        Index("ix_state_history_operation_id", "operation_id"),
        Index("ix_state_history_rollback", "can_rollback", "rolled_back"),
        # Compound index for common queries
        Index("ix_state_history_type_id_time", "state_type", "state_id", "created_at"),
    )

    @hybrid_property
    def size_change_bytes(self) -> int:
        """Calculate size change in bytes."""
        old_size = self.old_size_bytes or 0
        new_size = self.new_size_bytes or 0
        return new_size - old_size

    def get_changed_field_names(self) -> set[str]:
        """Get names of fields that changed."""
        if self.changed_fields and isinstance(self.changed_fields, list):
            return set(self.changed_fields)
        return set()


class StateMetadata(Base, AuditMixin):
    """
    State metadata table for state information and indexing.

    Stores lightweight metadata about states for fast querying
    without loading full state data.
    """

    __tablename__ = "state_metadata"

    # Composite primary key (state_type + state_id)
    state_type = Column(String(100), nullable=False, primary_key=True)
    state_id = Column(String(255), nullable=False, primary_key=True)

    # Current state info
    current_version = Column(Integer, nullable=False, default=1)
    current_checksum = Column(String(128), nullable=False)
    current_size_bytes = Column(Integer, nullable=False, default=0)

    # Storage locations
    in_redis = Column(Boolean, default=False, nullable=False)
    in_postgresql = Column(Boolean, default=True, nullable=False)
    in_influxdb = Column(Boolean, default=False, nullable=False)

    # Access patterns
    last_accessed = Column(DateTime(timezone=True), nullable=True)
    access_count = Column(Integer, default=0, nullable=False)
    cache_priority = Column(Integer, default=0, nullable=False)  # Higher = more important

    # State lifecycle
    first_created = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_modified = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Tags and classification
    tags = Column(JSONB, nullable=True)  # Key-value tags
    category = Column(String(100), nullable=True)
    is_critical = Column(Boolean, default=False, nullable=False)

    # Retention and cleanup
    retention_policy = Column(String(100), nullable=True)  # e.g., "30d", "permanent"
    can_delete = Column(Boolean, default=True, nullable=False)

    # Performance hints
    is_hot = Column(Boolean, default=False, nullable=False)  # Frequently accessed
    should_cache = Column(Boolean, default=True, nullable=False)

    # Validation
    schema_version = Column(String(50), nullable=False, default="1.0.0")
    validation_errors = Column(JSONB, nullable=True)  # Cached validation results

    # Indexes
    __table_args__ = (
        Index("ix_state_metadata_type", "state_type"),
        Index("ix_state_metadata_version", "current_version"),
        Index("ix_state_metadata_last_accessed", "last_accessed"),
        Index("ix_state_metadata_last_modified", "last_modified"),
        Index("ix_state_metadata_critical", "is_critical"),
        Index("ix_state_metadata_hot", "is_hot"),
        Index("ix_state_metadata_cache_priority", "cache_priority"),
        Index("ix_state_metadata_tags", "tags"),
        # Compound indexes for common queries
        Index("ix_state_metadata_type_modified", "state_type", "last_modified"),
        Index("ix_state_metadata_storage_locations", "in_redis", "in_postgresql"),
    )

    def get_tag(self, key: str, default: Any = None) -> Any:
        """Get a tag value."""
        if self.tags and isinstance(self.tags, dict):
            return self.tags.get(key, default)
        return default

    def set_tag(self, key: str, value: Any) -> None:
        """Set a tag value."""
        if not self.tags:
            self.tags = {}
        self.tags[key] = value

    def increment_access_count(self) -> None:
        """Increment access counter and update last_accessed."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)

    def is_in_storage_layer(self, layer: str) -> bool:
        """Check if state is stored in a specific layer."""
        layer_map = {
            "redis": self.in_redis,
            "postgresql": self.in_postgresql,
            "influxdb": self.in_influxdb,
        }
        return layer_map.get(layer.lower(), False)


class StateBackup(Base, AuditMixin, MetadataMixin):
    """
    State backup table for backup operations tracking.

    Tracks backup operations, their status, and metadata for
    disaster recovery and compliance purposes.
    """

    __tablename__ = "state_backups"

    # Primary key
    backup_id = Column(UUID(as_uuid=True), primary_key=True)

    # Backup metadata
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    backup_type = Column(
        Enum("full", "incremental", "differential", name="backup_type_enum"),
        nullable=False,
        default="full",
    )

    # Backup scope
    state_types_included = Column(JSONB, nullable=True)  # List of state types
    total_states = Column(Integer, nullable=False, default=0)
    total_size_bytes = Column(BigInteger, nullable=False, default=0)

    # Storage information
    storage_location = Column(String(500), nullable=True)  # File path or S3 URL
    storage_type = Column(String(50), nullable=False, default="filesystem")  # filesystem, s3, gcs

    # Compression and encryption
    is_compressed = Column(Boolean, default=True, nullable=False)
    compression_type = Column(String(50), nullable=True)
    compression_ratio = Column(String(10), nullable=True)  # "0.3" means 30% of original

    is_encrypted = Column(Boolean, default=False, nullable=False)
    encryption_algorithm = Column(String(50), nullable=True)

    # Integrity checks
    backup_checksum = Column(String(128), nullable=False)  # SHA-256 of backup file
    integrity_verified = Column(Boolean, default=False, nullable=False)

    # Status tracking
    status = Column(
        Enum(
            "creating",
            "completed",
            "failed",
            "verifying",
            "corrupted",
            "archived",
            name="backup_status_enum",
        ),
        nullable=False,
        default="creating",
    )

    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Integer, nullable=True)

    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)

    # Retention
    retention_days = Column(Integer, nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_protected = Column(Boolean, default=False, nullable=False)

    # Restore tracking
    restore_count = Column(Integer, default=0, nullable=False)
    last_restored_at = Column(DateTime(timezone=True), nullable=True)

    # Indexes
    __table_args__ = (
        Index("ix_state_backups_name", "name"),
        Index("ix_state_backups_type", "backup_type"),
        Index("ix_state_backups_status", "status"),
        Index("ix_state_backups_created_at", "created_at"),
        Index("ix_state_backups_completed_at", "completed_at"),
        Index("ix_state_backups_expires_at", "expires_at"),
        Index("ix_state_backups_protected", "is_protected"),
    )

    @hybrid_property
    def is_expired(self) -> bool:
        """Check if backup is expired."""
        if self.expires_at:
            return datetime.now(timezone.utc) > self.expires_at
        return False

    @hybrid_property
    def backup_success_rate(self) -> float:
        """Calculate backup success rate based on retries."""
        if self.retry_count == 0:
            return 100.0 if self.status == "completed" else 0.0

        success_attempts = 1 if self.status == "completed" else 0
        total_attempts = self.retry_count + 1
        return (success_attempts / total_attempts) * 100.0

    def get_included_state_types(self) -> set[str]:
        """Get set of included state types."""
        if self.state_types_included and isinstance(self.state_types_included, list):
            return set(self.state_types_included)
        return set()

    def mark_verified(self, checksum: str) -> bool:
        """Mark backup as integrity verified."""
        if self.backup_checksum == checksum:
            self.integrity_verified = True
            return True
        return False


# Create all indexes and constraints
def create_state_indexes():
    """Create additional performance indexes for state tables."""
    # This would be called during database migration
    pass
