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
from src.utils.checksum_utilities import calculate_state_checksum
from src.utils.serialization_utilities import serialize_state_data, deserialize_state_data
from src.utils.state_utils import (
    create_state_metadata,
    format_cache_key,
    store_in_redis_with_timeout,
    get_from_redis_with_timeout
)

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
            database_service: Database service for data operations (injected dependency)
        """
        super().__init__(name="StatePersistenceService")
        
        # Injected dependency - use protocol to avoid tight coupling
        self.database_service = database_service
        
        if database_service is None:
            self.logger.warning("StatePersistenceService initialized without database_service - some operations may fail")

        # Use consistent async pattern with event-driven processing
        # Maintain queues for backward compatibility but prefer event-driven patterns
        self._save_queue: asyncio.Queue = asyncio.Queue()
        self._delete_queue: asyncio.Queue = asyncio.Queue()

        # Background processing
        self._processing_task: asyncio.Task | None = None
        self._is_running = False

        self.logger.info(f"StatePersistenceService initialized with database_service: {type(database_service).__name__ if database_service else 'None'}")

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

            # Cancel and cleanup background task
            processing_task = self._processing_task
            self._processing_task = None
            
            if processing_task and not processing_task.done():
                processing_task.cancel()
                try:
                    await processing_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.error(f"Error waiting for processing task cleanup: {e}")

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

            # Use database service with generic operations to avoid direct model access
            import json
            
            # Check for existing state using service layer generic operations
            search_criteria = {
                "entity_type": state_type.value,
                "entity_id": state_id
            }
            
            # Use generic database operations instead of model-specific calls
            existing_records = await self._search_state_records(search_criteria, limit=1)

            # Prepare state data for storage
            snapshot_data = json.dumps(state_data, default=str)

            state_record = {
                "snapshot_id": f"{state_id}_{state_type.value}_{metadata.version}",
                "entity_type": state_type.value,
                "entity_id": state_id,
                "snapshot_data": snapshot_data,
                "state_version": metadata.version,
                "checksum": metadata.checksum,
                "created_at": metadata.created_at,
                "updated_at": datetime.now(timezone.utc),
            }

            if existing_records:
                # Update existing state using generic service operations
                record_id = existing_records[0].get("id") or existing_records[0].get("record_id")
                await self._update_state_record(record_id, state_record)
            else:
                # Create new state record using generic service operations  
                await self._create_state_record(state_record)

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

            # Use generic database operations to avoid direct model access
            import json
            
            search_criteria = {
                "entity_type": state_type.value,
                "entity_id": state_id
            }
            
            # Get the latest snapshot for this state using generic service operations
            records = await self._search_state_records(search_criteria, limit=1)

            if records and records[0].get("snapshot_data"):
                return json.loads(records[0]["snapshot_data"])

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

            # Use generic database operations to avoid direct model access
            search_criteria = {
                "entity_type": state_type.value,
                "entity_id": state_id
            }
            
            # Find all snapshots for this state using generic service operations
            records = await self._search_state_records(search_criteria)

            # Delete each record using generic service operations
            for record in records:
                record_id = record.get("id") or record.get("record_id")
                if record_id:
                    await self._delete_state_record(record_id)

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

            # Use database service properly without bypassing to repositories
            # Use database service interface instead of direct model access
            import json
            
            # Use generic database service operations to avoid direct model access
            query_data = {
                "table": "state_snapshots",
                "filters": {"entity_type": state_type.value},
                "order_by": "updated_at",
                "order_desc": True,
                "limit": limit,
                "offset": offset
            }
            
            snapshots = await self.database_service.query(query_data)

            states = []
            for snapshot in snapshots:
                if snapshot.get("snapshot_data"):
                    state_data = json.loads(snapshot["snapshot_data"])
                    states.append(
                        {
                            "state_id": snapshot.get("entity_id"),
                            "data": state_data,
                            "version": snapshot.get("state_version", 1),
                            "updated_at": snapshot.get("updated_at"),
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

            # Use database service interface instead of direct model access
            import json
            
            # Create snapshot data dictionary
            snapshot_data = {
                "snapshot_id": snapshot.snapshot_id,
                "entity_type": "system_snapshot",
                "entity_id": snapshot.snapshot_id,
                "snapshot_data": json.dumps(snapshot.__dict__, default=str),
                "description": snapshot.description,
                "created_at": snapshot.timestamp,
                "updated_at": datetime.now(timezone.utc),
            }

            await self.database_service.create("state_snapshots", snapshot_data)

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

            # Use database service interface instead of direct model access
            import json
            
            # Query snapshot by snapshot_id using service layer
            query_data = {
                "table": "state_snapshots",
                "filters": {"snapshot_id": snapshot_id},
                "limit": 1
            }
            
            snapshots = await self.database_service.query(query_data)

            if snapshots and snapshots[0].get("snapshot_data"):
                from ..state_service import StateSnapshot
                
                snapshot_data = json.loads(snapshots[0]["snapshot_data"])
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

    # Private helper methods for generic database operations

    async def _search_state_records(
        self, criteria: dict[str, Any], limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Search state records using generic database operations."""
        connection = None
        try:
            if not self.database_service:
                return []

            # Use database service generic search capabilities
            # This avoids direct model imports by using service layer patterns
            return await self.database_service.search_records(
                table_name="state_snapshots",  # Generic table reference
                filters=criteria,
                limit=limit or 100
            )

        except AttributeError:
            # Fallback using generic query interface instead of direct model access
            query_data = {
                "table": "state_snapshots",
                "filters": criteria,
                "limit": limit or 100
            }
            
            try:
                snapshots = await self.database_service.query(query_data)
                
                # Return consistent dict format
                return [
                    {
                        "id": s.get("id"),
                        "snapshot_id": s.get("snapshot_id"),
                        "entity_type": s.get("entity_type"),
                        "entity_id": s.get("entity_id"),
                        "snapshot_data": s.get("snapshot_data"),
                        "state_version": s.get("state_version"),
                        "checksum": s.get("checksum"),
                    }
                    for s in snapshots
                ]
            finally:
                # Connection cleanup handled by database service
                pass

        except Exception as e:
            self.logger.error(f"Failed to search state records: {e}")
            return []
        finally:
            if connection:
                try:
                    await connection.close()
                except (ServiceError, DataError) as e:
                    pass

    async def _create_state_record(self, record_data: dict[str, Any]) -> bool:
        """Create a new state record using generic database operations."""
        try:
            if not self.database_service:
                return False

            # Try generic record creation first
            try:
                return await self.database_service.create_record(
                    table_name="state_snapshots",
                    data=record_data
                )
            except AttributeError:
                # Fallback using generic create interface instead of direct model access
                await self.database_service.create("state_snapshots", record_data)
                return True

        except Exception as e:
            self.logger.error(f"Failed to create state record: {e}")
            return False

    async def _update_state_record(self, record_id: Any, record_data: dict[str, Any]) -> bool:
        """Update an existing state record using generic database operations."""
        try:
            if not self.database_service:
                return False

            # Try generic record update first
            try:
                return await self.database_service.update_record(
                    table_name="state_snapshots",
                    record_id=record_id,
                    data=record_data
                )
            except AttributeError:
                # Fallback using generic update interface instead of direct model access
                await self.database_service.update("state_snapshots", record_id, record_data)
                return True

        except Exception as e:
            self.logger.error(f"Failed to update state record: {e}")
            return False

    async def _delete_state_record(self, record_id: Any) -> bool:
        """Delete a state record using generic database operations."""
        try:
            if not self.database_service:
                return False

            # Try generic record deletion first
            try:
                return await self.database_service.delete_record(
                    table_name="state_snapshots",
                    record_id=record_id
                )
            except AttributeError:
                # Fallback using generic delete interface
                await self.database_service.delete("state_snapshots", record_id)
                return True

        except Exception as e:
            self.logger.error(f"Failed to delete state record: {e}")
            return False
