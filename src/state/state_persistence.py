"""
State Persistence module for managing state storage and retrieval.

This module provides database persistence for state management,
handling save/load operations with the database service.
"""

import asyncio
import json
from typing import TYPE_CHECKING, Any

from src.core.base.component import BaseComponent
from src.core.exceptions import DataError, ServiceError, StateError

# Service layer imports instead of direct repository access
from .services import StatePersistenceServiceProtocol

if TYPE_CHECKING:
    from .state_service import StateMetadata, StateService, StateSnapshot, StateType


class StatePersistence(BaseComponent):
    """
    Handles state persistence operations for the StateService.

    This is now a wrapper around the service layer to maintain backward compatibility
    while properly separating concerns.
    """

    def __init__(self, state_service: "StateService"):
        """
        Initialize the state persistence handler.

        Args:
            state_service: Reference to the main state service
        """
        super().__init__()
        self.state_service = state_service

        # Use service layer instead of direct database access
        self._persistence_service: StatePersistenceServiceProtocol | None = None

        # Queues for async persistence (maintained for compatibility)
        self._save_queue: asyncio.Queue = asyncio.Queue()
        self._delete_queue: asyncio.Queue = asyncio.Queue()

        # Background task
        self._persistence_task: asyncio.Task | None = None
        self._running = False

        self.logger.info("StatePersistence initialized as service layer wrapper")

    async def initialize(self) -> None:
        """Initialize the persistence handler."""
        try:
            # Get persistence service from state service
            if hasattr(self.state_service, "_persistence_service"):
                self._persistence_service = self.state_service._persistence_service
                self.logger.info("Using service layer for persistence operations")
            else:
                self.logger.warning("No persistence service available - operations will be limited")

            # Start background persistence task for backward compatibility
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

            await super().cleanup()
            self.logger.info("StatePersistence cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during StatePersistence cleanup: {e}")
            raise

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
            # Use service layer for persistence operations
            if self._persistence_service:
                return await self._persistence_service.load_state(state_type, state_id)
            else:
                self.logger.warning("No persistence service available for load_state operation")
                return None

        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
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
            # Use service layer for persistence operations
            if self._persistence_service:
                return await self._persistence_service.save_state(
                    state_type, state_id, state_data, metadata
                )
            else:
                self.logger.warning("No persistence service available for save_state operation")
                return False

        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
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
            # Use service layer for persistence operations
            if self._persistence_service:
                return await self._persistence_service.delete_state(state_type, state_id)
            else:
                self.logger.warning("No persistence service available for delete_state operation")
                return False

        except Exception as e:
            self.logger.error(f"Failed to delete state: {e}")
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
            # Use service layer for persistence operations
            if self._persistence_service:
                return await self._persistence_service.list_states(
                    state_type, limit=limit, offset=0
                )
            else:
                self.logger.warning(
                    "No persistence service available for get_states_by_type operation"
                )
                return []

        except Exception as e:
            self.logger.error(f"Failed to get states by type: {e}")
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
            if self.database_service is None:
                self.logger.warning("Database service not available, cannot save state")
                return []

            async with self.database_service.transaction() as session:
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
            # Use service layer for persistence operations
            if self._persistence_service:
                return await self._persistence_service.save_snapshot(snapshot)
            else:
                self.logger.warning("No persistence service available for save_snapshot operation")
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
            Snapshot or None if not found
        """
        try:
            # Use service layer for persistence operations
            if self._persistence_service:
                return await self._persistence_service.load_snapshot(snapshot_id)
            else:
                self.logger.warning("No persistence service available for load_snapshot operation")
                return None

        except Exception as e:
            self.logger.error(f"Failed to load snapshot: {e}")
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
            if self.database_service is None:
                self.logger.warning("Database service not available, cannot save state")
                return None

            async with self.database_service.transaction() as session:
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
                # Don't re-raise to keep loop running
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

    def _is_service_available(self) -> bool:
        """
        Check if persistence service is available.

        Returns:
            bool: True if persistence service is available
        """
        try:
            return (
                self._persistence_service is not None
                and hasattr(self._persistence_service, "is_available")
                and self._persistence_service.is_available()
            )
        except Exception as e:
            self.logger.warning(f"Error checking persistence service availability: {e}")
            return False
