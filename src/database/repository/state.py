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

    async def get_by_bot(self, bot_id: str) -> list[StateSnapshot]:
        """Get state snapshots by bot."""
        return await self.get_all(filters={"bot_id": bot_id}, order_by="-timestamp")

    async def get_by_snapshot_type(self, snapshot_type: str) -> list[StateSnapshot]:
        """Get snapshots by type."""
        return await self.get_all(filters={"snapshot_type": snapshot_type}, order_by="-timestamp")

    async def get_latest_snapshot(self, bot_id: str, snapshot_type: str | None = None) -> StateSnapshot | None:
        """Get latest snapshot for bot."""
        filters = {"bot_id": bot_id}
        if snapshot_type:
            filters["snapshot_type"] = snapshot_type

        snapshots = await self.get_all(filters=filters, order_by="-timestamp", limit=1)
        return snapshots[0] if snapshots else None

    async def get_by_version(self, bot_id: str, version: int) -> StateSnapshot | None:
        """Get snapshot by version."""
        return await self.get_by(bot_id=bot_id, state_version=version)

    async def cleanup_old_snapshots(self, bot_id: str, keep_count: int = 10) -> int:
        """Clean up old snapshots, keeping only the most recent."""
        snapshots = await self.get_by_bot(bot_id)
        if len(snapshots) <= keep_count:
            return 0

        old_snapshots = snapshots[keep_count:]
        count = 0
        for snapshot in old_snapshots:
            await self.delete(snapshot.id)
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

    async def get_by_bot(self, bot_id: str) -> list[StateCheckpoint]:
        """Get checkpoints by bot."""
        return await self.get_all(filters={"bot_id": bot_id}, order_by="-timestamp")

    async def get_by_checkpoint_type(self, checkpoint_type: str) -> list[StateCheckpoint]:
        """Get checkpoints by type."""
        return await self.get_all(filters={"checkpoint_type": checkpoint_type}, order_by="-timestamp")

    async def get_latest_checkpoint(self, bot_id: str) -> StateCheckpoint | None:
        """Get latest checkpoint for bot."""
        checkpoints = await self.get_all(filters={"bot_id": bot_id}, order_by="-timestamp", limit=1)
        return checkpoints[0] if checkpoints else None

    async def get_valid_checkpoints(self, bot_id: str) -> list[StateCheckpoint]:
        """Get valid checkpoints for bot."""
        return await self.get_all(filters={"bot_id": bot_id, "is_valid": True}, order_by="-timestamp")


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

    async def get_by_entity(self, entity_type: str, entity_id: str) -> list[StateHistory]:
        """Get history by entity."""
        return await self.get_all(
            filters={"entity_type": entity_type, "entity_id": entity_id},
            order_by="-change_timestamp",
        )

    async def get_by_change_type(self, change_type: str) -> list[StateHistory]:
        """Get history by change type."""
        return await self.get_all(filters={"change_type": change_type}, order_by="-change_timestamp")

    async def get_recent_changes(self, entity_type: str, entity_id: str, hours: int = 24) -> list[StateHistory]:
        """Get recent changes for entity."""
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        return await self.get_all(
            filters={
                "entity_type": entity_type,
                "entity_id": entity_id,
                "change_timestamp": {"gte": since},
            },
            order_by="-change_timestamp",
        )

    async def get_change_summary(
        self, entity_type: str, entity_id: str, start_version: int, end_version: int
    ) -> list[StateHistory]:
        """Get changes between versions."""
        return await self.get_all(
            filters={
                "entity_type": entity_type,
                "entity_id": entity_id,
                "old_version": {"gte": start_version, "lte": end_version},
            },
            order_by="change_timestamp",
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

    async def get_by_entity(self, entity_type: str, entity_id: str) -> StateMetadata | None:
        """Get metadata by entity."""
        return await self.get_by(entity_type=entity_type, entity_id=entity_id)

    async def get_by_entity_type(self, entity_type: str) -> list[StateMetadata]:
        """Get metadata by entity type."""
        return await self.get_all(filters={"entity_type": entity_type})

    async def get_locked_entities(self, entity_type: str | None = None) -> list[StateMetadata]:
        """Get locked entities."""
        filters = {"is_locked": True}
        if entity_type:
            filters["entity_type"] = entity_type
        return await self.get_all(filters=filters)

    async def get_corrupted_entities(self, entity_type: str | None = None) -> list[StateMetadata]:
        """Get entities with corruption detected."""
        filters = {"corruption_detected": True}
        if entity_type:
            filters["entity_type"] = entity_type
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

    async def get_by_bot(self, bot_id: str) -> list[StateBackup]:
        """Get backups by bot."""
        return await self.get_all(filters={"bot_id": bot_id}, order_by="-backup_timestamp")

    async def get_by_backup_type(self, backup_type: str) -> list[StateBackup]:
        """Get backups by type."""
        return await self.get_all(filters={"backup_type": backup_type}, order_by="-backup_timestamp")

    async def get_latest_backup(self, bot_id: str) -> StateBackup | None:
        """Get latest backup for bot."""
        backups = await self.get_all(filters={"bot_id": bot_id}, order_by="-backup_timestamp", limit=1)
        return backups[0] if backups else None

    async def get_verified_backups(self, bot_id: str) -> list[StateBackup]:
        """Get verified backups for bot."""
        return await self.get_all(
            filters={"bot_id": bot_id, "verification_status": "verified"},
            order_by="-backup_timestamp",
        )

    async def cleanup_old_backups(self, bot_id: str, keep_days: int = 30) -> int:
        """Clean up old backups."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=keep_days)
        old_backups = await self.get_all(filters={"bot_id": bot_id, "backup_timestamp": {"lt": cutoff_date}})

        count = 0
        for backup in old_backups:
            await self.delete(backup.id)
            count += 1
        return count
