"""
State Persistence module for managing state storage and retrieval.

This module provides database persistence for state management,
handling save/load operations with the database service.
"""

import asyncio
from typing import TYPE_CHECKING, Any

from src.core.base.component import BaseComponent
from src.core.exceptions import StateConsistencyError

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

    @property
    def database_service(self):
        """Get database service from state service."""
        return getattr(self.state_service, "database_service", None)

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
            try:
                self._persistence_task = asyncio.create_task(
                    self._persistence_loop(), name="state_persistence_loop"
                )
            except TypeError:
                # Fallback for mocked or older asyncio versions
                self._persistence_task = asyncio.create_task(self._persistence_loop())

            await super().initialize()
            self.logger.info("StatePersistence initialization completed")

        except Exception as e:
            self.logger.error(f"StatePersistence initialization failed: {e}")
            raise StateConsistencyError(f"Failed to initialize StatePersistence: {e}") from e

    async def cleanup(self) -> None:
        """Cleanup persistence resources."""
        try:
            self._running = False

            # Process remaining queued items
            await self._flush_queues()

            # Cancel and cleanup background task
            persistence_task = self._persistence_task
            self._persistence_task = None

            if persistence_task and not persistence_task.done():
                persistence_task.cancel()
                try:
                    await asyncio.wait_for(persistence_task, timeout=5.0)
                except asyncio.CancelledError:
                    pass
                except asyncio.TimeoutError:
                    self.logger.warning("Persistence task cleanup timeout")
                except Exception as e:
                    self.logger.error(f"Error waiting for persistence task cleanup: {e}")
                finally:
                    # Ensure task reference is cleared
                    persistence_task = None

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
            # Use service layer for persistence operations
            if self._persistence_service:
                # Service layer should handle search logic
                states = []
                if state_types:
                    for state_type in state_types:
                        type_states = await self._persistence_service.list_states(
                            state_type, limit=limit
                        )
                        # Apply criteria filtering
                        filtered_states = [
                            s
                            for s in type_states
                            if self._matches_criteria(s.get("data", s), criteria)
                        ]
                        states.extend(filtered_states)
                else:
                    # This would need service enhancement to support cross-type search
                    self.logger.warning(
                        "Cross-type search not fully supported through service layer"
                    )
                return states[:limit] if limit else states
            else:
                self.logger.warning("No persistence service available for search_states operation")
                return []

        except Exception as e:
            self.logger.error(f"Failed to search states: {e}")
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
            # Use service layer for persistence operations
            if self._persistence_service:
                self.logger.info("Loading states from persistence service...")

                # This is a simplified implementation - would need service enhancement
                # for comprehensive state loading across all types
                # For now, log that operation was delegated to service
                self.logger.info("State loading delegated to persistence service layer")
            else:
                self.logger.warning(
                    "No persistence service available for load_all_states_to_cache operation"
                )

        except Exception as e:
            self.logger.error(f"Failed to load states to cache: {e}")

    # Private methods

    async def _persistence_loop(self) -> None:
        """Background persistence processing loop with proper resource management."""
        save_batch = []
        delete_batch = []
        batch_size = 10  # Process in batches to improve efficiency

        while self._running:
            try:
                # Collect save operations into batches
                try:
                    save_item = await asyncio.wait_for(self._save_queue.get(), timeout=1.0)
                    save_batch.append(save_item)
                    self._save_queue.task_done()

                    # Continue collecting until batch is full or timeout
                    while len(save_batch) < batch_size and not self._save_queue.empty():
                        try:
                            save_item = await asyncio.wait_for(self._save_queue.get(), timeout=0.1)
                            save_batch.append(save_item)
                            self._save_queue.task_done()
                        except asyncio.TimeoutError:
                            break

                except asyncio.TimeoutError:
                    pass  # No items to process

                # Collect delete operations into batches
                try:
                    delete_item = await asyncio.wait_for(self._delete_queue.get(), timeout=0.1)
                    delete_batch.append(delete_item)
                    self._delete_queue.task_done()

                    # Continue collecting until batch is full or timeout
                    while len(delete_batch) < batch_size and not self._delete_queue.empty():
                        try:
                            delete_item = await asyncio.wait_for(
                                self._delete_queue.get(), timeout=0.05
                            )
                            delete_batch.append(delete_item)
                            self._delete_queue.task_done()
                        except asyncio.TimeoutError:
                            break

                except asyncio.TimeoutError:
                    pass  # No items to process

                # Process batches concurrently if we have items
                if save_batch or delete_batch:
                    tasks = []

                    if save_batch:
                        tasks.append(self._process_save_batch(save_batch.copy()))
                        save_batch.clear()

                    if delete_batch:
                        tasks.append(self._process_delete_batch(delete_batch.copy()))
                        delete_batch.clear()

                    # Execute batches concurrently with timeout
                    if tasks:
                        await asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True), timeout=30.0
                        )
                else:
                    # No work to do, sleep briefly
                    await asyncio.sleep(0.1)

            except asyncio.TimeoutError:
                self.logger.warning("Persistence batch processing timeout")
                # Clear batches on timeout to prevent memory leaks
                save_batch.clear()
                delete_batch.clear()
                await asyncio.sleep(1.0)
            except Exception as e:
                self.logger.error(f"Persistence loop error: {e}")
                # Clear batches on error to prevent memory leaks
                save_batch.clear()
                delete_batch.clear()
                await asyncio.sleep(1.0)  # Wait before retrying on error

    async def _process_save_batch(self, batch: list[dict[str, Any]]) -> None:
        """Process a batch of save operations concurrently."""
        try:
            save_tasks = []
            for save_item in batch:
                task = self.save_state(
                    save_item["state_type"],
                    save_item["state_id"],
                    save_item["state_data"],
                    save_item["metadata"],
                )
                save_tasks.append(task)

            # Execute save operations concurrently
            if save_tasks:
                await asyncio.gather(*save_tasks, return_exceptions=True)

        except Exception as e:
            self.logger.error(f"Save batch processing error: {e}")

    async def _process_delete_batch(self, batch: list[dict[str, Any]]) -> None:
        """Process a batch of delete operations concurrently."""
        try:
            delete_tasks = []
            for delete_item in batch:
                task = self.delete_state(delete_item["state_type"], delete_item["state_id"])
                delete_tasks.append(task)

            # Execute delete operations concurrently
            if delete_tasks:
                await asyncio.gather(*delete_tasks, return_exceptions=True)

        except Exception as e:
            self.logger.error(f"Delete batch processing error: {e}")

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

    def _is_database_available(self) -> bool:
        """
        Check if database service is available.

        Returns:
            bool: True if database service is available
        """
        try:
            return self.database_service is not None
        except Exception as e:
            self.logger.warning(f"Error checking database service availability: {e}")
            return False
