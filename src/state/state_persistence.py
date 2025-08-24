"""
State Persistence module for managing state storage and retrieval.

This module provides database persistence for state management,
handling save/load operations with the database service.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.base import BaseComponent
from src.core.exceptions import DataError, ServiceError, StateError
from src.database.connection import get_async_session
from src.database.repository.state import (
    StateBackupRepository,
    StateMetadataRepository,
    StateSnapshotRepository,
)
from src.error_handling.decorators import with_retry

if TYPE_CHECKING:
    from src.database.service import DatabaseService

    from .state_service import StateMetadata, StateService, StateSnapshot, StateType


class StatePersistence(BaseComponent):
    """
    Handles state persistence operations for the StateService.

    Provides methods for saving, loading, and managing state data
    in persistent storage through the database service.
    """

    def __init__(self, state_service: "StateService"):
        """
        Initialize the state persistence handler.

        Args:
            state_service: Reference to the main state service
        """
        super().__init__()
        self.state_service = state_service

        # PER-07 Fix: Add proper null check before accessing database_service
        if not hasattr(state_service, "database_service") or state_service.database_service is None:
            self.logger.warning("Database service not available on state_service")
            self.database_service: DatabaseService | None = None
        else:
            self.database_service = state_service.database_service

        # Initialize repositories (will be None if database service not available)
        self._snapshot_repo: StateSnapshotRepository | None = None
        self._backup_repo: StateBackupRepository | None = None
        self._metadata_repo: StateMetadataRepository | None = None

        # Queues for async persistence
        self._save_queue: asyncio.Queue = asyncio.Queue()
        self._delete_queue: asyncio.Queue = asyncio.Queue()

        # Background task
        self._persistence_task: asyncio.Task | None = None
        self._running = False

        self.logger.info("StatePersistence initialized")

    async def initialize(self) -> None:
        """Initialize the persistence handler."""
        try:
            # Initialize repositories if database service is available
            if self.database_service is not None:
                try:
                    # Note: Repositories will be created per-session as needed
                    self.logger.info("Database service available for persistence operations")
                except Exception as e:
                    self.logger.error(f"Failed to initialize database repositories: {e}")
                    # Continue without database repositories for graceful degradation

            # Start background persistence task
            self._running = True
            self._persistence_task = asyncio.create_task(self._persistence_loop())

            await super().initialize()
            self.logger.info("StatePersistence initialization completed")

        except Exception as e:
            self.logger.error(f"StatePersistence initialization failed: {e}")
            raise StateError(f"Failed to initialize StatePersistence: {e}") from e

    async def cleanup(self) -> None:
        """Cleanup persistence resources."""
        try:
            self._running = False

            # Process remaining queued items
            await self._flush_queues()

            # Cancel background task
            if self._persistence_task and not self._persistence_task.done():
                self._persistence_task.cancel()
                try:
                    await self._persistence_task
                except asyncio.CancelledError:
                    pass

            super().cleanup()
            self.logger.info("StatePersistence cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during StatePersistence cleanup: {e}")

    async def load_state(self, state_type: "StateType", state_id: str) -> dict[str, Any] | None:
        """
        Load state from persistent storage.

        Args:
            state_type: Type of state to load
            state_id: State identifier

        Returns:
            State data or None if not found
        """
        try:
            # PER-08 Fix: Replace direct null check with proper abstraction
            if not self._is_database_available():
                self.logger.warning("Database not available for load_state operation")
                return None

            # PER-09 Fix: Use repository pattern instead of direct SQL queries
            async with get_async_session() as session:
                # Create temporary repository for this operation
                snapshot_repo = StateSnapshotRepository(session)

                # Search for the latest snapshot matching the criteria
                snapshots = await snapshot_repo.get_all(
                    filters={"entity_type": state_type.value, "entity_id": state_id},
                    order_by="-state_version",
                    limit=1,
                )

                if snapshots and snapshots[0].snapshot_data:
                    # Parse the stored JSON data
                    return json.loads(snapshots[0].snapshot_data)

                return None

        except (DataError, ServiceError) as e:
            self.logger.error(f"Database service error loading state: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error loading state: {e}")
            return None

    async def save_state(
        self,
        state_type: "StateType",
        state_id: str,
        state_data: dict[str, Any],
        metadata: "StateMetadata",
    ) -> bool:
        """
        Save state to persistent storage.

        Args:
            state_type: Type of state
            state_id: State identifier
            state_data: State data to save
            metadata: State metadata

        Returns:
            True if successful
        """
        try:
            if not self._is_database_available():
                self.logger.warning("Database not available for save_state operation")
                return False

            # PER-09 Fix: Use repository pattern instead of direct SQL queries
            async with get_async_session() as session:
                snapshot_repo = StateSnapshotRepository(session)

                # Try to find existing snapshot
                existing = await snapshot_repo.get_by(
                    entity_type=state_type.value, entity_id=state_id
                )

                snapshot_data = json.dumps(state_data, default=str)

                if existing:
                    # Update existing snapshot
                    existing.snapshot_data = snapshot_data
                    existing.state_version = metadata.version
                    existing.checksum = metadata.checksum
                    existing.updated_at = datetime.now(timezone.utc)
                    await snapshot_repo.update(existing)
                else:
                    # Create new snapshot
                    from src.database.models.state import StateSnapshot

                    new_snapshot = StateSnapshot(
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

            return True

        except (DataError, ServiceError) as e:
            self.logger.error(f"Database service error saving state: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error saving state: {e}")
            return False

    async def delete_state(self, state_type: "StateType", state_id: str) -> bool:
        """
        Delete state from persistent storage.

        Args:
            state_type: Type of state
            state_id: State identifier

        Returns:
            True if successful
        """
        try:
            if not self._is_database_available():
                self.logger.warning("Database not available for delete_state operation")
                return False

            # PER-09 Fix: Use repository pattern instead of direct SQL queries
            async with get_async_session() as session:
                snapshot_repo = StateSnapshotRepository(session)

                # Find snapshots to delete
                snapshots = await snapshot_repo.get_all(
                    filters={"entity_type": state_type.value, "entity_id": state_id}
                )

                # Delete all matching snapshots
                for snapshot in snapshots:
                    await snapshot_repo.delete(snapshot.id)

                await session.commit()
                return True

        except (DataError, ServiceError) as e:
            self.logger.error(f"Database service error deleting state: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error deleting state: {e}")
            return False

    async def queue_state_save(
        self,
        state_type: "StateType",
        state_id: str,
        state_data: dict[str, Any],
        metadata: "StateMetadata",
    ) -> None:
        """Queue state for asynchronous saving."""
        await self._save_queue.put(
            {
                "state_type": state_type,
                "state_id": state_id,
                "state_data": state_data,
                "metadata": metadata,
            }
        )

    async def queue_state_delete(self, state_type: "StateType", state_id: str) -> None:
        """Queue state for asynchronous deletion."""
        await self._delete_queue.put(
            {
                "state_type": state_type,
                "state_id": state_id,
            }
        )

    async def get_states_by_type(
        self,
        state_type: "StateType",
        limit: int | None = None,
        include_metadata: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Get all states of a specific type from persistence.

        Args:
            state_type: Type of states to retrieve
            limit: Maximum number of states
            include_metadata: Whether to include metadata

        Returns:
            List of states
        """
        try:
            if not self._is_database_available():
                self.logger.warning("Database not available for get_states_by_type operation")
                return []

            # PER-09 Fix: Use repository pattern instead of direct SQL queries
            async with get_async_session() as session:
                snapshot_repo = StateSnapshotRepository(session)

                snapshots = await snapshot_repo.get_all(
                    filters={"entity_type": state_type.value}, order_by="-updated_at", limit=limit
                )

            states = []
            for snapshot in snapshots:
                if snapshot.snapshot_data:
                    state_data = json.loads(snapshot.snapshot_data)

                    if include_metadata:
                        metadata = {
                            "state_id": snapshot.entity_id,
                            "version": snapshot.state_version,
                            "checksum": snapshot.checksum,
                            "created_at": (
                                snapshot.created_at.isoformat() if snapshot.created_at else None
                            ),
                            "updated_at": (
                                snapshot.updated_at.isoformat() if snapshot.updated_at else None
                            ),
                        }
                        states.append({"data": state_data, "metadata": metadata})
                    else:
                        states.append(state_data)

            return states

        except (DataError, ServiceError) as e:
            self.logger.error(f"Database service error getting states by type: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error getting states by type: {e}")
            return []

    async def search_states(
        self,
        criteria: dict[str, Any],
        state_types: list["StateType"] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Search states based on criteria.

        Args:
            criteria: Search criteria
            state_types: State types to search
            limit: Maximum results

        Returns:
            List of matching states
        """
        try:
            if not self._is_database_available():
                self.logger.warning("Database not available for search_states operation")
                return []

            # PER-09 Fix: Use repository pattern instead of direct SQL queries
            async with get_async_session() as session:
                snapshot_repo = StateSnapshotRepository(session)

                filters = {}
                if state_types:
                    type_values = [st.value for st in state_types]
                    filters["entity_type"] = type_values

                snapshots = await snapshot_repo.get_all(filters=filters, limit=limit)

            states = []
            for snapshot in snapshots:
                if snapshot.snapshot_data:
                    state_data = json.loads(snapshot.snapshot_data)
                    # Apply additional criteria filtering
                    if self._matches_criteria(state_data, criteria):
                        states.append(state_data)

            return states

        except (DataError, ServiceError) as e:
            self.logger.error(f"Database service error searching states: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error searching states: {e}")
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
            if not self._is_database_available():
                self.logger.warning("Database not available for save_snapshot operation")
                return False

            # PER-09 Fix: Use repository pattern instead of direct SQL queries
            async with get_async_session() as session:
                snapshot_repo = StateSnapshotRepository(session)

                # Convert the StateSnapshot to database model
                from src.database.models.state import StateSnapshot as DBStateSnapshot

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

            return True

        except (DataError, ServiceError) as e:
            self.logger.error(f"Database service error saving snapshot: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error saving snapshot: {e}")
            return False

    async def load_snapshot(self, snapshot_id: str) -> "StateSnapshot | None":
        """
        Load a state snapshot.

        Args:
            snapshot_id: Snapshot identifier

        Returns:
            Snapshot or None if not found
        """
        try:
            if not self._is_database_available():
                self.logger.warning("Database not available for load_snapshot operation")
                return None

            # PER-09 Fix: Use repository pattern instead of direct SQL queries
            async with get_async_session() as session:
                snapshot_repo = StateSnapshotRepository(session)

                db_snapshot = await snapshot_repo.get_by(snapshot_id=snapshot_id)

            if db_snapshot and db_snapshot.snapshot_data:
                snapshot_data = json.loads(db_snapshot.snapshot_data)
                # Reconstruct StateSnapshot object
                from .state_service import StateSnapshot

                return StateSnapshot(**snapshot_data)

            return None

        except (DataError, ServiceError) as e:
            self.logger.error(f"Database service error loading snapshot: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error loading snapshot: {e}")
            return None

    async def load_all_states_to_cache(self) -> None:
        """Load all states from persistence to cache on startup."""
        try:
            if not self._is_database_available():
                self.logger.warning("Database not available for load_all_states_to_cache operation")
                return

            # This is a simplified implementation
            # In production, this would be paginated for large datasets
            self.logger.info("Loading states from persistence layer...")

            # Use repository pattern for loading states
            async with get_async_session() as session:
                snapshot_repo = StateSnapshotRepository(session)

                # Load a limited batch of recent snapshots
                snapshots = await snapshot_repo.get_all(
                    order_by="-updated_at",
                    limit=1000,  # Reasonable batch size
                )

                self.logger.info(f"Loaded {len(snapshots)} state snapshots for caching")

        except Exception as e:
            self.logger.error(f"Failed to load states to cache: {e}")

    # Private methods

    async def _persistence_loop(self) -> None:
        """Background loop for processing persistence queue."""
        while self._running:
            try:
                # Process save queue
                try:
                    save_item = await asyncio.wait_for(self._save_queue.get(), timeout=1.0)
                    await self.save_state(
                        save_item["state_type"],
                        save_item["state_id"],
                        save_item["state_data"],
                        save_item["metadata"],
                    )
                    self._save_queue.task_done()
                except asyncio.TimeoutError:
                    pass

                # Process delete queue
                try:
                    delete_item = await asyncio.wait_for(self._delete_queue.get(), timeout=0.1)
                    await self.delete_state(
                        delete_item["state_type"],
                        delete_item["state_id"],
                    )
                    self._delete_queue.task_done()
                except asyncio.TimeoutError:
                    pass

            except Exception as e:
                self.logger.error(f"Persistence loop error: {e}")
                await asyncio.sleep(1.0)

    async def _flush_queues(self) -> None:
        """Flush all pending persistence operations."""
        # Process remaining save operations
        while not self._save_queue.empty():
            try:
                save_item = self._save_queue.get_nowait()
                await self.save_state(
                    save_item["state_type"],
                    save_item["state_id"],
                    save_item["state_data"],
                    save_item["metadata"],
                )
                self._save_queue.task_done()
            except asyncio.QueueEmpty:
                break

        # Process remaining delete operations
        while not self._delete_queue.empty():
            try:
                delete_item = self._delete_queue.get_nowait()
                await self.delete_state(
                    delete_item["state_type"],
                    delete_item["state_id"],
                )
                self._delete_queue.task_done()
            except asyncio.QueueEmpty:
                break

    def _matches_criteria(self, state_data: dict[str, Any], criteria: dict[str, Any]) -> bool:
        """Check if state matches search criteria."""
        for key, value in criteria.items():
            if key not in state_data or state_data[key] != value:
                return False
        return True

    def _is_database_available(self) -> bool:
        """
        PER-08 Fix: Proper abstraction for checking database availability.

        Returns:
            bool: True if database service is available and initialized
        """
        try:
            return (
                self.database_service is not None
                and hasattr(self.database_service, "initialized")
                and getattr(self.database_service, "initialized", False)
            )
        except Exception as e:
            self.logger.warning(f"Error checking database availability: {e}")
            return False

    @with_retry(max_retries=3, base_delay=1.0)
    async def _ensure_repositories_initialized(self) -> bool:
        """
        Ensure database repositories are properly initialized.

        Returns:
            bool: True if repositories are available
        """
        try:
            if not self._is_database_available():
                return False

            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize repositories: {e}")
            return False
