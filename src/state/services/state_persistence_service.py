"""
State Persistence Service - Handles state data persistence operations.

This service abstracts all persistence operations, providing a clean interface
for state storage and retrieval without exposing infrastructure details.
"""

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol

from src.core.base.service import BaseService
from src.core.exceptions import DataError, ServiceError, StateError

if TYPE_CHECKING:
    from ..state_service import StateMetadata, StateSnapshot, StateType


class StatePersistenceServiceProtocol(Protocol):
    """Protocol defining the state persistence service interface."""

    async def save_state(
        self,
        state_type: "StateType",
        state_id: str,
        state_data: dict[str, Any],
        metadata: "StateMetadata",
    ) -> bool: ...

    async def load_state(self, state_type: "StateType", state_id: str) -> dict[str, Any] | None: ...

    async def delete_state(self, state_type: "StateType", state_id: str) -> bool: ...

    async def list_states(
        self,
        state_type: "StateType",
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]: ...

    async def save_snapshot(self, snapshot: "StateSnapshot") -> bool: ...

    async def load_snapshot(self, snapshot_id: str) -> "StateSnapshot | None": ...


class StatePersistenceService(BaseService):
    """
    State persistence service providing database-agnostic state storage.

    This service handles all state persistence operations through a clean
    interface that abstracts database implementation details.
    """

    def __init__(self, database_service: Any = None):
        """
        Initialize the state persistence service.

        Args:
            database_service: Database service for data operations
        """
        super().__init__(name="StatePersistenceService")
        self.database_service = database_service

        # Use consistent async pattern with event-driven processing
        # Maintain queues for backward compatibility but prefer event-driven patterns
        self._save_queue: asyncio.Queue = asyncio.Queue()
        self._delete_queue: asyncio.Queue = asyncio.Queue()

        # Background processing
        self._processing_task: asyncio.Task | None = None
        self._is_running = False

        self.logger.info("StatePersistenceService initialized")

    async def start(self) -> None:
        """Start the persistence service."""
        try:
            await super().start()

            # Start background processing
            self._is_running = True
            self._processing_task = asyncio.create_task(self._process_operations())

            self.logger.info("StatePersistenceService started")

        except Exception as e:
            self.logger.error(f"Failed to start StatePersistenceService: {e}")
            raise

    async def stop(self) -> None:
        """Stop the persistence service."""
        try:
            self._is_running = False

            # Process remaining operations
            await self._flush_queues()

            # Cancel background task
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass

            await super().stop()
            self.logger.info("StatePersistenceService stopped")

        except Exception as e:
            self.logger.error(f"Error stopping StatePersistenceService: {e}")
            raise

    async def save_state(
        self,
        state_type: "StateType",
        state_id: str,
        state_data: dict[str, Any],
        metadata: "StateMetadata",
    ) -> bool:
        """
        Save state data to persistent storage.

        Args:
            state_type: Type of state
            state_id: State identifier
            state_data: State data to save
            metadata: State metadata

        Returns:
            True if successful

        Raises:
            StateError: If save operation fails
        """
        try:
            if not self.database_service:
                self.logger.warning("No database service available")
                return False

            # Validate inputs
            if not state_id or not state_data:
                raise StateError("Invalid state data provided")

            # Use database service through proper abstraction
            async with self.database_service.transaction() as session:
                # Import repository here to avoid circular imports
                from src.database.repository.state import StateSnapshotRepository

                snapshot_repo = StateSnapshotRepository(session)

                # Check for existing state
                existing = await snapshot_repo.get_by(
                    entity_type=state_type.value, entity_id=state_id
                )

                # Prepare state data for storage
                import json

                snapshot_data = json.dumps(state_data, default=str)

                if existing:
                    # Update existing state
                    existing.snapshot_data = snapshot_data
                    existing.state_version = metadata.version
                    existing.checksum = metadata.checksum
                    existing.updated_at = datetime.now(timezone.utc)
                    await snapshot_repo.update(existing)
                else:
                    # Create new state record
                    from src.database.models.state import StateSnapshot as DBStateSnapshot

                    new_snapshot = DBStateSnapshot(
                        snapshot_id=f"{state_id}_{state_type.value}_{metadata.version}",
                        entity_type=state_type.value,
                        entity_id=state_id,
                        snapshot_data=snapshot_data,
                        state_version=metadata.version,
                        checksum=metadata.checksum,
                        created_at=metadata.created_at,
                        updated_at=datetime.now(timezone.utc),
                    )
                    await snapshot_repo.create(new_snapshot)

                await session.commit()

            self.logger.debug(f"State saved successfully: {state_type.value}:{state_id}")
            return True

        except (DataError, ServiceError) as e:
            self.logger.error(f"Database service error saving state: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            raise StateError(f"State save failed: {e}") from e

    async def load_state(self, state_type: "StateType", state_id: str) -> dict[str, Any] | None:
        """
        Load state data from persistent storage.

        Args:
            state_type: Type of state
            state_id: State identifier

        Returns:
            State data or None if not found

        Raises:
            StateError: If load operation fails
        """
        try:
            if not self.database_service:
                self.logger.warning("No database service available")
                return None

            async with self.database_service.transaction() as session:
                from src.database.repository.state import StateSnapshotRepository

                snapshot_repo = StateSnapshotRepository(session)

                # Get the latest snapshot for this state
                snapshots = await snapshot_repo.get_all(
                    filters={"entity_type": state_type.value, "entity_id": state_id},
                    order_by="-state_version",
                    limit=1,
                )

                if snapshots and snapshots[0].snapshot_data:
                    import json

                    return json.loads(snapshots[0].snapshot_data)

            return None

        except (DataError, ServiceError) as e:
            self.logger.error(f"Database service error loading state: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return None

    async def delete_state(self, state_type: "StateType", state_id: str) -> bool:
        """
        Delete state data from persistent storage.

        Args:
            state_type: Type of state
            state_id: State identifier

        Returns:
            True if successful

        Raises:
            StateError: If delete operation fails
        """
        try:
            if not self.database_service:
                self.logger.warning("No database service available")
                return False

            async with self.database_service.transaction() as session:
                from src.database.repository.state import StateSnapshotRepository

                snapshot_repo = StateSnapshotRepository(session)

                # Find and delete all snapshots for this state
                snapshots = await snapshot_repo.get_all(
                    filters={"entity_type": state_type.value, "entity_id": state_id}
                )

                for snapshot in snapshots:
                    await snapshot_repo.delete(snapshot.id)

                await session.commit()

            self.logger.debug(f"State deleted successfully: {state_type.value}:{state_id}")
            return True

        except (DataError, ServiceError) as e:
            self.logger.error(f"Database service error deleting state: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete state: {e}")
            raise StateError(f"State delete failed: {e}") from e

    async def list_states(
        self,
        state_type: "StateType",
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List states of a given type.

        Args:
            state_type: Type of states to list
            limit: Maximum number of states to return
            offset: Number of states to skip

        Returns:
            List of state data dictionaries
        """
        try:
            if not self.database_service:
                self.logger.warning("No database service available")
                return []

            async with self.database_service.transaction() as session:
                from src.database.repository.state import StateSnapshotRepository

                snapshot_repo = StateSnapshotRepository(session)

                # Get snapshots for this state type
                snapshots = await snapshot_repo.get_all(
                    filters={"entity_type": state_type.value},
                    order_by="-updated_at",
                    limit=limit,
                    offset=offset,
                )

                states = []
                import json

                for snapshot in snapshots:
                    if snapshot.snapshot_data:
                        state_data = json.loads(snapshot.snapshot_data)
                        states.append(
                            {
                                "state_id": snapshot.entity_id,
                                "data": state_data,
                                "version": snapshot.state_version,
                                "updated_at": (
                                    snapshot.updated_at.isoformat() if snapshot.updated_at else None
                                ),
                            }
                        )

                return states

        except (DataError, ServiceError) as e:
            self.logger.error(f"Database service error listing states: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Failed to list states: {e}")
            return []

    async def save_snapshot(self, snapshot: "StateSnapshot") -> bool:
        """
        Save a state snapshot.

        Args:
            snapshot: Snapshot to save

        Returns:
            True if successful
        """
        try:
            if not self.database_service:
                self.logger.warning("No database service available")
                return False

            async with self.database_service.transaction() as session:
                from src.database.models.state import StateSnapshot as DBStateSnapshot
                from src.database.repository.state import StateSnapshotRepository

                snapshot_repo = StateSnapshotRepository(session)

                # Convert snapshot to database model
                import json

                db_snapshot = DBStateSnapshot(
                    snapshot_id=snapshot.snapshot_id,
                    entity_type="system_snapshot",
                    entity_id=snapshot.snapshot_id,
                    snapshot_data=json.dumps(snapshot.__dict__, default=str),
                    description=snapshot.description,
                    created_at=snapshot.timestamp,
                    updated_at=datetime.now(timezone.utc),
                )

                await snapshot_repo.create(db_snapshot)
                await session.commit()

            self.logger.debug(f"Snapshot saved successfully: {snapshot.snapshot_id}")
            return True

        except (DataError, ServiceError) as e:
            self.logger.error(f"Database service error saving snapshot: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to save snapshot: {e}")
            return False

    async def load_snapshot(self, snapshot_id: str) -> "StateSnapshot | None":
        """
        Load a state snapshot.

        Args:
            snapshot_id: Snapshot identifier

        Returns:
            StateSnapshot or None if not found
        """
        try:
            if not self.database_service:
                self.logger.warning("No database service available")
                return None

            async with self.database_service.transaction() as session:
                from src.database.repository.state import StateSnapshotRepository

                snapshot_repo = StateSnapshotRepository(session)

                db_snapshot = await snapshot_repo.get_by(snapshot_id=snapshot_id)

                if db_snapshot and db_snapshot.snapshot_data:
                    import json

                    from ..state_service import StateSnapshot

                    snapshot_data = json.loads(db_snapshot.snapshot_data)
                    return StateSnapshot(**snapshot_data)

            return None

        except (DataError, ServiceError) as e:
            self.logger.error(f"Database service error loading snapshot: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load snapshot: {e}")
            return None

    async def queue_save_operation(
        self,
        state_type: "StateType",
        state_id: str,
        state_data: dict[str, Any],
        metadata: "StateMetadata",
    ) -> None:
        """Queue a save operation for background processing."""
        await self._save_queue.put(
            {
                "operation": "save",
                "state_type": state_type,
                "state_id": state_id,
                "state_data": state_data,
                "metadata": metadata,
            }
        )

    async def queue_delete_operation(self, state_type: "StateType", state_id: str) -> None:
        """Queue a delete operation for background processing."""
        await self._delete_queue.put(
            {
                "operation": "delete",
                "state_type": state_type,
                "state_id": state_id,
            }
        )

    # Private methods

    async def _process_operations(self) -> None:
        """Process queued operations in the background."""
        while self._is_running:
            try:
                # Process save operations
                await self._process_save_operations()

                # Process delete operations
                await self._process_delete_operations()

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error processing operations: {e}")
                await asyncio.sleep(1.0)

    async def _process_save_operations(self) -> None:
        """Process queued save operations."""
        try:
            while not self._save_queue.empty():
                operation = await asyncio.wait_for(self._save_queue.get(), timeout=0.1)

                success = await self.save_state(
                    operation["state_type"],
                    operation["state_id"],
                    operation["state_data"],
                    operation["metadata"],
                )

                if success:
                    self._save_queue.task_done()
                else:
                    # Re-queue failed operations (with limit)
                    retry_count = operation.get("retry_count", 0)
                    if retry_count < 3:
                        operation["retry_count"] = retry_count + 1
                        await self._save_queue.put(operation)

                    self._save_queue.task_done()

        except asyncio.TimeoutError:
            # No operations to process
            pass
        except Exception as e:
            self.logger.error(f"Error processing save operations: {e}")

    async def _process_delete_operations(self) -> None:
        """Process queued delete operations."""
        try:
            while not self._delete_queue.empty():
                operation = await asyncio.wait_for(self._delete_queue.get(), timeout=0.1)

                success = await self.delete_state(
                    operation["state_type"],
                    operation["state_id"],
                )

                if success:
                    self._delete_queue.task_done()
                else:
                    # Re-queue failed operations (with limit)
                    retry_count = operation.get("retry_count", 0)
                    if retry_count < 3:
                        operation["retry_count"] = retry_count + 1
                        await self._delete_queue.put(operation)

                    self._delete_queue.task_done()

        except asyncio.TimeoutError:
            # No operations to process
            pass
        except Exception as e:
            self.logger.error(f"Error processing delete operations: {e}")

    async def _flush_queues(self) -> None:
        """Flush all remaining operations in queues."""
        try:
            # Process remaining save operations
            while not self._save_queue.empty():
                operation = self._save_queue.get_nowait()
                await self.save_state(
                    operation["state_type"],
                    operation["state_id"],
                    operation["state_data"],
                    operation["metadata"],
                )
                self._save_queue.task_done()

            # Process remaining delete operations
            while not self._delete_queue.empty():
                operation = self._delete_queue.get_nowait()
                await self.delete_state(
                    operation["state_type"],
                    operation["state_id"],
                )
                self._delete_queue.task_done()

        except asyncio.QueueEmpty:
            pass
        except Exception as e:
            self.logger.error(f"Error flushing queues: {e}")

    def is_available(self) -> bool:
        """Check if persistence service is available."""
        return self.database_service is not None and self._is_running
