"""
State management database models for PostgreSQL persistence.

These models provide ACID-compliant state storage with versioning,
checkpointing, audit trails, and backup/restore capabilities.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    DECIMAL,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
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
    snapshot_type = Column(
        String(50),
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
    status = Column(
        String(50),
        nullable=False,
        default="creating",
    )

    # Retention and cleanup
    retention_days = Column(Integer, nullable=True)  # Auto-delete after N days
    is_protected = Column(Boolean, default=False, nullable=False)  # Prevent auto-deletion

    # Foreign key relationships for business context
    bot_id = Column(
        UUID(as_uuid=True),
        ForeignKey("bots.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        index=True
    )
    strategy_id = Column(
        UUID(as_uuid=True),
        ForeignKey("strategies.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        index=True
    )
    position_id = Column(
        UUID(as_uuid=True),
        ForeignKey("positions.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        index=True
    )

    # Relationships
    checkpoints = relationship(
        "StateCheckpoint", back_populates="snapshot", cascade="all, delete-orphan"
    )
    bot = relationship("Bot", foreign_keys=[bot_id], back_populates="state_snapshots")
    strategy = relationship("Strategy", foreign_keys=[strategy_id], back_populates="state_snapshots")
    position = relationship("Position", foreign_keys=[position_id], back_populates="state_snapshots")

    # Indexes and constraints for performance and data integrity
    __table_args__ = (
        Index("ix_state_snapshots_created_at", "created_at"),
        Index("ix_state_snapshots_type", "snapshot_type"),
        Index("ix_state_snapshots_status", "status"),
        Index("ix_state_snapshots_checksum", "state_checksum"),
        Index("ix_state_snapshots_protected", "is_protected"),
        Index("ix_state_snapshots_type_status", "snapshot_type", "status"),
        Index("ix_state_snapshots_bot_id", "bot_id"),
        Index("ix_state_snapshots_strategy_id", "strategy_id"),
        Index("ix_state_snapshots_position_id", "position_id"),
        Index("ix_state_snapshots_bot_created", "bot_id", "created_at"),
        Index("ix_state_snapshots_strategy_created", "strategy_id", "created_at"),
        UniqueConstraint("state_checksum", name="uq_state_snapshots_checksum"),
        CheckConstraint("raw_size_bytes >= 0", name="ck_state_snapshots_raw_size_positive"),
        CheckConstraint(
            "compressed_size_bytes >= 0", name="ck_state_snapshots_compressed_size_positive"
        ),
        CheckConstraint("retention_days > 0", name="ck_state_snapshots_retention_positive"),
        CheckConstraint(
            "snapshot_type IN ('manual', 'automatic', 'scheduled', 'emergency')",
            name="ck_state_snapshots_type_valid",
        ),
        CheckConstraint(
            "status IN ('creating', 'active', 'failed', 'expired')",
            name="ck_state_snapshots_status_valid",
        ),
    )

    def compression_ratio(self) -> Decimal:
        """Calculate compression ratio if compressed."""
        if self.is_compressed and self.compressed_size_bytes and self.raw_size_bytes:
            return Decimal(str(self.compressed_size_bytes)) / Decimal(str(self.raw_size_bytes))
        return Decimal("1.0")

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
    checkpoint_type = Column(
        String(50),
        nullable=False,
        default="incremental",
    )

    # Reference to base snapshot
    base_snapshot_id = Column(
        UUID(as_uuid=True),
        ForeignKey("state_snapshots.snapshot_id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=True,
        index=True,
    )
    snapshot = relationship(
        "StateSnapshot", back_populates="checkpoints", foreign_keys=[base_snapshot_id]
    )

    # Previous checkpoint for chain
    previous_checkpoint_id = Column(
        UUID(as_uuid=True),
        ForeignKey("state_checkpoints.checkpoint_id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        index=True,
    )
    previous_checkpoint = relationship(
        "StateCheckpoint", remote_side="StateCheckpoint.checkpoint_id"
    )
    bot = relationship("Bot", foreign_keys="StateCheckpoint.bot_id")
    strategy = relationship("Strategy", foreign_keys="StateCheckpoint.strategy_id")
    order = relationship("Order", foreign_keys="StateCheckpoint.order_id")

    # State changes (delta from base or previous checkpoint)
    state_changes = Column(JSONB, nullable=False)  # Only changed states

    # Change tracking
    changes_count = Column(Integer, nullable=False, default=0)
    states_affected = Column(JSONB, nullable=True)  # List of state IDs affected

    # Foreign key relationships for business context
    bot_id = Column(
        UUID(as_uuid=True),
        ForeignKey("bots.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        index=True
    )
    strategy_id = Column(
        UUID(as_uuid=True),
        ForeignKey("strategies.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        index=True
    )
    order_id = Column(
        UUID(as_uuid=True),
        ForeignKey("orders.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        index=True
    )

    # Storage info
    size_bytes = Column(BigInteger, nullable=False, default=0)
    is_compressed = Column(Boolean, default=False, nullable=False)

    # Validation
    changes_checksum = Column(String(128), nullable=False)

    # Status
    status = Column(
        String(50),
        nullable=False,
        default="creating",
    )

    # Cleanup
    retention_days = Column(Integer, nullable=True)

    # Indexes and constraints
    __table_args__ = (
        Index("ix_state_checkpoints_created_at", "created_at"),
        Index("ix_state_checkpoints_type", "checkpoint_type"),
        Index("ix_state_checkpoints_status", "status"),
        Index("ix_state_checkpoints_type_status", "checkpoint_type", "status"),
        Index("ix_state_checkpoints_bot_id", "bot_id"),
        Index("ix_state_checkpoints_strategy_id", "strategy_id"),
        Index("ix_state_checkpoints_order_id", "order_id"),
        Index("ix_state_checkpoints_bot_created", "bot_id", "created_at"),
        UniqueConstraint("changes_checksum", name="uq_state_checkpoints_checksum"),
        CheckConstraint("changes_count >= 0", name="ck_state_checkpoints_changes_count_positive"),
        CheckConstraint("size_bytes >= 0", name="ck_state_checkpoints_size_positive"),
        CheckConstraint("retention_days > 0", name="ck_state_checkpoints_retention_positive"),
        CheckConstraint(
            "checkpoint_type IN ('incremental', 'full', 'diff')",
            name="ck_state_checkpoints_type_valid",
        ),
        CheckConstraint(
            "status IN ('creating', 'active', 'failed', 'applied')",
            name="ck_state_checkpoints_status_valid",
        ),
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
    operation = Column(
        String(50),
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
        String(50),
        nullable=False,
        default="medium",
    )

    # Foreign key relationships for business context
    bot_id = Column(
        UUID(as_uuid=True),
        ForeignKey("bots.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        index=True
    )
    strategy_id = Column(
        UUID(as_uuid=True),
        ForeignKey("strategies.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        index=True
    )
    order_id = Column(
        UUID(as_uuid=True),
        ForeignKey("orders.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        index=True
    )
    position_id = Column(
        UUID(as_uuid=True),
        ForeignKey("positions.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        index=True
    )
    trade_id = Column(
        UUID(as_uuid=True),
        ForeignKey("trades.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        index=True
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

    # Indexes and constraints for efficient querying and data integrity
    __table_args__ = (
        Index("ix_state_history_state_type_id", "state_type", "state_id"),
        Index("ix_state_history_operation", "operation"),
        Index("ix_state_history_created_at", "created_at"),
        Index("ix_state_history_source_component", "source_component"),
        Index("ix_state_history_operation_id", "operation_id"),
        Index("ix_state_history_rollback", "can_rollback", "rolled_back"),
        Index("ix_state_history_bot_id", "bot_id"),
        Index("ix_state_history_strategy_id", "strategy_id"),
        Index("ix_state_history_order_id", "order_id"),
        Index("ix_state_history_position_id", "position_id"),
        Index("ix_state_history_trade_id", "trade_id"),
        # Compound index for common queries
        Index("ix_state_history_type_id_time", "state_type", "state_id", "created_at"),
        Index("ix_state_history_operation_time", "operation", "created_at"),
        Index("ix_state_history_bot_created", "bot_id", "created_at"),
        Index("ix_state_history_strategy_created", "strategy_id", "created_at"),
        CheckConstraint("old_size_bytes >= 0", name="ck_state_history_old_size_positive"),
        CheckConstraint("new_size_bytes >= 0", name="ck_state_history_new_size_positive"),
        CheckConstraint(
            "operation IN ('create', 'update', 'delete', 'restore', 'sync')",
            name="ck_state_history_operation_valid",
        ),
        CheckConstraint(
            "state_type IN ('bot_state', 'position_state', 'order_state', 'portfolio_state', 'risk_state', 'strategy_state', 'market_state', 'trade_state', 'execution', 'system_state')",
            name="ck_state_history_state_type_valid",
        ),
        CheckConstraint(
            "priority IN ('low', 'medium', 'high', 'critical')",
            name="ck_state_history_priority_valid",
        ),
    )

    def size_change_bytes(self) -> int:
        """Calculate size change in bytes."""
        old_size = int(self.old_size_bytes or 0)
        new_size = int(self.new_size_bytes or 0)
        return new_size - old_size

    def get_changed_field_names(self) -> set[str]:
        """Get names of fields that changed."""
        if self.changed_fields and isinstance(self.changed_fields, list):
            return set(self.changed_fields)
        return set()

    # Relationships
    bot = relationship("Bot", foreign_keys="StateHistory.bot_id")
    strategy = relationship("Strategy", foreign_keys="StateHistory.strategy_id")
    order = relationship("Order", foreign_keys="StateHistory.order_id")
    position = relationship("Position", foreign_keys="StateHistory.position_id")
    trade = relationship("Trade", foreign_keys="StateHistory.trade_id")


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

    # Foreign key relationships for business context
    bot_id = Column(
        UUID(as_uuid=True),
        ForeignKey("bots.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        index=True
    )
    strategy_id = Column(
        UUID(as_uuid=True),
        ForeignKey("strategies.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        index=True
    )

    # Validation
    schema_version = Column(String(50), nullable=False, default="1.0.0")
    validation_errors = Column(JSONB, nullable=True)  # Cached validation results

    # Indexes and constraints
    __table_args__ = (
        Index("ix_state_metadata_type", "state_type"),
        Index("ix_state_metadata_version", "current_version"),
        Index("ix_state_metadata_last_accessed", "last_accessed"),
        Index("ix_state_metadata_last_modified", "last_modified"),
        Index("ix_state_metadata_critical", "is_critical"),
        Index("ix_state_metadata_hot", "is_hot"),
        Index("ix_state_metadata_cache_priority", "cache_priority"),
        Index("ix_state_metadata_tags", "tags"),
        Index("ix_state_metadata_bot_id", "bot_id"),
        Index("ix_state_metadata_strategy_id", "strategy_id"),
        # Compound indexes for common queries
        Index("ix_state_metadata_type_modified", "state_type", "last_modified"),
        Index("ix_state_metadata_storage_locations", "in_redis", "in_postgresql"),
        Index("ix_state_metadata_type_critical", "state_type", "is_critical"),
        Index("ix_state_metadata_bot_type", "bot_id", "state_type"),
        Index("ix_state_metadata_strategy_type", "strategy_id", "state_type"),
        CheckConstraint("current_version > 0", name="ck_state_metadata_version_positive"),
        CheckConstraint("current_size_bytes >= 0", name="ck_state_metadata_size_positive"),
        CheckConstraint("access_count >= 0", name="ck_state_metadata_access_count_positive"),
        CheckConstraint("cache_priority >= 0", name="ck_state_metadata_cache_priority_positive"),
        CheckConstraint(
            "state_type IN ('bot_state', 'position_state', 'order_state', 'portfolio_state', 'risk_state', 'strategy_state', 'market_state', 'trade_state', 'execution', 'system_state')",
            name="ck_state_metadata_state_type_valid",
        ),
    )

    def get_tag(self, key: str, default: Any = None) -> Any:
        """Get a tag value."""
        if self.tags and isinstance(self.tags, dict):
            return self.tags.get(key, default)
        return default

    def set_tag(self, key: str, value: Any) -> None:
        """Set a tag value."""
        if not self.tags:
            self.tags = {}  # type: ignore
        tags_dict = dict(self.tags) if self.tags else {}
        tags_dict[key] = value
        self.tags = tags_dict  # type: ignore

    def increment_access_count(self) -> None:
        """Increment access counter and update last_accessed."""
        self.access_count = int(self.access_count) + 1  # type: ignore
        self.last_accessed = datetime.now(timezone.utc)  # type: ignore

    def is_in_storage_layer(self, layer: str) -> bool:
        """Check if state is stored in a specific layer."""
        layer_map = {
            "redis": bool(self.in_redis),
            "postgresql": bool(self.in_postgresql),
            "influxdb": bool(self.in_influxdb),
        }
        return layer_map.get(layer.lower(), False)

    # Relationships
    bot = relationship("Bot", foreign_keys="StateMetadata.bot_id")
    strategy = relationship("Strategy", foreign_keys="StateMetadata.strategy_id")


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
        String(50),
        nullable=False,
        default="full",
    )

    # Backup scope
    state_types_included = Column(JSONB, nullable=True)  # List of state types
    total_states = Column(Integer, nullable=False, default=0)
    total_size_bytes = Column(BigInteger, nullable=False, default=0)

    # Foreign key relationships for business context
    bot_id = Column(
        UUID(as_uuid=True),
        ForeignKey("bots.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        index=True
    )
    strategy_id = Column(
        UUID(as_uuid=True),
        ForeignKey("strategies.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
        index=True
    )

    # Storage information
    storage_location = Column(String(500), nullable=True)  # File path or S3 URL
    storage_type = Column(String(50), nullable=False, default="filesystem")  # filesystem, s3, gcs

    # Compression and encryption
    is_compressed = Column(Boolean, default=True, nullable=False)
    compression_type = Column(String(50), nullable=True)
    compression_ratio: Column[Decimal] = Column(
        DECIMAL(5, 4), nullable=True
    )  # Precise compression ratio

    is_encrypted = Column(Boolean, default=False, nullable=False)
    encryption_algorithm = Column(String(50), nullable=True)

    # Integrity checks
    backup_checksum = Column(String(128), nullable=False)  # SHA-256 of backup file
    integrity_verified = Column(Boolean, default=False, nullable=False)

    # Status tracking
    status = Column(
        String(50),
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

    # Indexes and constraints
    __table_args__ = (
        Index("ix_state_backups_name", "name"),
        Index("ix_state_backups_type", "backup_type"),
        Index("ix_state_backups_status", "status"),
        Index("ix_state_backups_created_at", "created_at"),
        Index("ix_state_backups_completed_at", "completed_at"),
        Index("ix_state_backups_expires_at", "expires_at"),
        Index("ix_state_backups_protected", "is_protected"),
        Index("ix_state_backups_type_status", "backup_type", "status"),
        Index("ix_state_backups_status_created", "status", "created_at"),
        Index("ix_state_backups_bot_id", "bot_id"),
        Index("ix_state_backups_strategy_id", "strategy_id"),
        Index("ix_state_backups_bot_created", "bot_id", "created_at"),
        Index("ix_state_backups_strategy_created", "strategy_id", "created_at"),
        UniqueConstraint("backup_checksum", name="uq_state_backups_checksum"),
        CheckConstraint("total_states >= 0", name="ck_state_backups_total_states_positive"),
        CheckConstraint("total_size_bytes >= 0", name="ck_state_backups_total_size_positive"),
        CheckConstraint("duration_seconds >= 0", name="ck_state_backups_duration_positive"),
        CheckConstraint("retry_count >= 0", name="ck_state_backups_retry_count_positive"),
        CheckConstraint("retention_days > 0", name="ck_state_backups_retention_positive"),
        CheckConstraint("restore_count >= 0", name="ck_state_backups_restore_count_positive"),
        CheckConstraint(
            "compression_ratio > 0 AND compression_ratio <= 1",
            name="ck_state_backups_compression_ratio_valid",
        ),
        CheckConstraint(
            "backup_type IN ('full', 'incremental', 'differential')",
            name="ck_state_backups_type_valid",
        ),
        CheckConstraint(
            "status IN ('creating', 'completed', 'failed', 'expired')",
            name="ck_state_backups_status_valid",
        ),
        CheckConstraint(
            "storage_type IN ('filesystem', 's3', 'gcs', 'azure')",
            name="ck_state_backups_storage_type_valid",
        ),
    )

    def is_expired(self) -> bool:
        """Check if backup is expired."""
        if self.expires_at:
            return bool(datetime.now(timezone.utc) > self.expires_at)
        return False

    def backup_success_rate(self) -> Decimal:
        """Calculate backup success rate based on retries."""
        if int(self.retry_count) == 0:
            return Decimal("100.0") if self.status == "completed" else Decimal("0.0")

        success_attempts = 1 if self.status == "completed" else 0
        total_attempts = int(self.retry_count) + 1
        return Decimal(str(success_attempts)) / Decimal(str(total_attempts)) * Decimal("100.0")

    def get_included_state_types(self) -> set[str]:
        """Get set of included state types."""
        if self.state_types_included and isinstance(self.state_types_included, list):
            return set(self.state_types_included)
        return set()

    # Relationships
    bot = relationship("Bot", foreign_keys="StateBackup.bot_id")
    strategy = relationship("Strategy", foreign_keys="StateBackup.strategy_id")

    def mark_verified(self, checksum: str) -> bool:
        """Mark backup as integrity verified."""
        if self.backup_checksum == checksum:
            self.integrity_verified = True  # type: ignore
            return True
        return False


# Create all indexes and constraints
def create_state_indexes():
    """Create additional performance indexes for state tables."""
    # This would be called during database migration
    pass
