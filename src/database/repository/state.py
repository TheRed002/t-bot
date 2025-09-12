"""State management repositories implementation."""

from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.state import (
StateBackup,
    StateCheckpoint,
    StateHistory,
    StateMetadata,
    StateSnapshot,
)
from src.database.repository.base import DatabaseRepository


class StateSnapshotRepository(DatabaseRepository):
    """Repository for StateSnapshot entities."""

    def __init__(self, session: AsyncSession):
        """Initialize state snapshot repository."""
        super().__init__(
            session=session,
            model=StateSnapshot,
            entity_type=StateSnapshot,
            key_type=str,
            name="StateSnapshotRepository",
        )

    async def get_by_name_prefix(self, name_prefix: str) -> list[StateSnapshot]:
        """Get state snapshots by name prefix."""
        return await self.get_all(
            filters={"name": {"startswith": name_prefix}}, order_by="-created_at"
        )

    async def get_by_snapshot_type(self, snapshot_type: str) -> list[StateSnapshot]:
        """Get snapshots by type."""
        return await self.get_all(filters={"snapshot_type": snapshot_type}, order_by="-created_at")

    async def get_latest_snapshot(
        self, name_prefix: str, snapshot_type: str | None = None
    ) -> StateSnapshot | None:
        """Get latest snapshot for name prefix."""
        filters = {"name": {"startswith": name_prefix}}
        if snapshot_type:
            filters["snapshot_type"] = snapshot_type

        snapshots = await self.get_all(filters=filters, order_by="-created_at", limit=1)
        return snapshots[0] if snapshots else None

    async def get_by_schema_version(self, schema_version: str) -> list[StateSnapshot]:
        """Get snapshots by schema version."""
        return await self.get_all(
            filters={"schema_version": schema_version}, order_by="-created_at"
        )

    async def cleanup_old_snapshots(self, name_prefix: str, keep_count: int = 10) -> int:
        """Clean up old snapshots, keeping only the most recent."""
        snapshots = await self.get_by_name_prefix(name_prefix)
        if len(snapshots) <= keep_count:
            return 0

        old_snapshots = snapshots[keep_count:]
        count = 0
        for snapshot in old_snapshots:
            await self.delete(snapshot.snapshot_id)
            count += 1
        return count


class StateCheckpointRepository(DatabaseRepository):
    """Repository for StateCheckpoint entities."""

    def __init__(self, session: AsyncSession):
        """Initialize state checkpoint repository."""
        super().__init__(
            session=session,
            model=StateCheckpoint,
            entity_type=StateCheckpoint,
            key_type=str,
            name="StateCheckpointRepository",
        )

    async def get_by_name_prefix(self, name_prefix: str) -> list[StateCheckpoint]:
        """Get checkpoints by name prefix."""
        return await self.get_all(
            filters={"name": {"startswith": name_prefix}}, order_by="-created_at"
        )

    async def get_by_checkpoint_type(self, checkpoint_type: str) -> list[StateCheckpoint]:
        """Get checkpoints by type."""
        return await self.get_all(
            filters={"checkpoint_type": checkpoint_type}, order_by="-created_at"
        )

    async def get_latest_checkpoint(self, name_prefix: str) -> StateCheckpoint | None:
        """Get latest checkpoint for name prefix."""
        checkpoints = await self.get_all(
            filters={"name": {"startswith": name_prefix}}, order_by="-created_at", limit=1
        )
        return checkpoints[0] if checkpoints else None

    async def get_by_status(self, status: str) -> list[StateCheckpoint]:
        """Get checkpoints by status."""
        return await self.get_all(filters={"status": status}, order_by="-created_at")


class StateHistoryRepository(DatabaseRepository):
    """Repository for StateHistory entities."""

    def __init__(self, session: AsyncSession):
        """Initialize state history repository."""
        super().__init__(
            session=session,
            model=StateHistory,
            entity_type=StateHistory,
            key_type=str,
            name="StateHistoryRepository",
        )

    async def get_by_state(self, state_type: str, state_id: str) -> list[StateHistory]:
        """Get history by state."""
        return await self.get_all(
            filters={"state_type": state_type, "state_id": state_id},
            order_by="-created_at",
        )

    async def get_by_operation(self, operation: str) -> list[StateHistory]:
        """Get history by operation type."""
        return await self.get_all(filters={"operation": operation}, order_by="-created_at")

    async def get_recent_changes(
        self, state_type: str, state_id: str, hours: int = 24
    ) -> list[StateHistory]:
        """Get recent changes for state."""
        return await RepositoryUtils.execute_time_based_query(
            self.session,
            self.model,
            timestamp_field="created_at",
            hours=hours,
            additional_filters={"state_type": state_type, "state_id": state_id},
            order_by="-created_at",
        )

    async def get_by_component(self, source_component: str) -> list[StateHistory]:
        """Get history by source component."""
        return await self.get_all(
            filters={"source_component": source_component},
            order_by="-created_at",
        )


class StateMetadataRepository(DatabaseRepository):
    """Repository for StateMetadata entities."""

    def __init__(self, session: AsyncSession):
        """Initialize state metadata repository."""
        super().__init__(
            session=session,
            model=StateMetadata,
            entity_type=StateMetadata,
            key_type=str,
            name="StateMetadataRepository",
        )

    async def get_by_state(self, state_type: str, state_id: str) -> StateMetadata | None:
        """Get metadata by state."""
        return await self.get_by(state_type=state_type, state_id=state_id)

    async def get_by_state_type(self, state_type: str) -> list[StateMetadata]:
        """Get metadata by state type."""
        return await self.get_all(filters={"state_type": state_type})

    async def get_critical_states(self, state_type: str | None = None) -> list[StateMetadata]:
        """Get critical states."""
        filters = {"is_critical": True}
        if state_type:
            filters["state_type"] = state_type
        return await self.get_all(filters=filters)

    async def get_hot_states(self, state_type: str | None = None) -> list[StateMetadata]:
        """Get frequently accessed (hot) states."""
        filters = {"is_hot": True}
        if state_type:
            filters["state_type"] = state_type
        return await self.get_all(filters=filters)


class StateBackupRepository(DatabaseRepository):
    """Repository for StateBackup entities."""

    def __init__(self, session: AsyncSession):
        """Initialize state backup repository."""
        super().__init__(
            session=session,
            model=StateBackup,
            entity_type=StateBackup,
            key_type=str,
            name="StateBackupRepository",
        )

    async def get_by_name_prefix(self, name_prefix: str) -> list[StateBackup]:
        """Get backups by name prefix."""
        return await self.get_all(
            filters={"name": {"startswith": name_prefix}}, order_by="-created_at"
        )

    async def get_by_backup_type(self, backup_type: str) -> list[StateBackup]:
        """Get backups by type."""
        return await self.get_all(filters={"backup_type": backup_type}, order_by="-created_at")

    async def get_latest_backup(self, name_prefix: str) -> StateBackup | None:
        """Get latest backup for name prefix."""
        backups = await self.get_all(
            filters={"name": {"startswith": name_prefix}}, order_by="-created_at", limit=1
        )
        return backups[0] if backups else None

    async def get_verified_backups(self, name_prefix: str) -> list[StateBackup]:
        """Get verified backups for name prefix."""
        return await self.get_all(
            filters={"name": {"startswith": name_prefix}, "integrity_verified": True},
            order_by="-created_at",
        )

    async def cleanup_old_backups(self, name_prefix: str, keep_days: int = 30) -> int:
        """Clean up old backups."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=keep_days)
        old_backups = await self.get_all(
            filters={"name": {"startswith": name_prefix}, "created_at": {"lt": cutoff_date}}
        )

        count = 0
        for backup in old_backups:
            await self.delete(backup.backup_id)
            count += 1
        return count
