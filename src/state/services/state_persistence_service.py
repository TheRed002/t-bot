"""
State Persistence Service - Handles state data persistence operations.

This service abstracts all persistence operations, providing a clean interface
for state storage and retrieval without exposing infrastructure details.
"""

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol

from src.core.base.service import BaseService
from src.core.exceptions import DatabaseError, StateConsistencyError

# Database service handles serialization and checksums

if TYPE_CHECKING:
    from src.core.types import StateType

    from ..state_service import StateMetadata, StateSnapshot


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
        self._database_service = database_service

        if database_service is None:
            self.logger.info(
                "StatePersistenceService initialized without database_service - will resolve from DI container when needed"
            )

        # Use consistent pub/sub event-driven processing matching database module
        # Removed queues to align with messaging patterns consistency - using events only
        from src.utils.messaging_patterns import MessagingCoordinator

        self._messaging_coordinator = MessagingCoordinator("StatePersistence")

        # Background processing
        self._processing_task: asyncio.Task | None = None
        self._is_running = False

        self.logger.info(
            f"StatePersistenceService initialized with database_service: {type(database_service).__name__ if database_service else 'None'}"
        )

    def _get_database_service(self):
        """Get database service from DI container if not injected."""
        if self._database_service is not None:
            return self._database_service

        # Try to resolve from DI container
        try:
            from src.core.dependency_injection import get_container
            container = get_container()
            self._database_service = container.get("DatabaseService")
            return self._database_service
        except Exception:
            # Return None if not available
            return None

    @property
    def database_service(self):
        """Lazy-loaded database service property."""
        return self._get_database_service()

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

            # Flush remaining events using consistent pub/sub pattern
            await self._flush_events()

            # Cancel and cleanup background task
            processing_task = self._processing_task
            self._processing_task = None

            if processing_task and not processing_task.done():
                processing_task.cancel()
                try:
                    await asyncio.wait_for(processing_task, timeout=5.0)
                except asyncio.CancelledError:
                    pass
                except asyncio.TimeoutError:
                    self.logger.warning("Processing task cleanup timeout")
                except Exception as e:
                    self.logger.error(f"Error waiting for processing task cleanup: {e}")
                finally:
                    # Ensure task reference is cleared
                    processing_task = None

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
            StateConsistencyError: If save operation fails
        """
        try:
            if not self.database_service:
                self.logger.warning("No database service available")
                return False

            # Validate inputs
            if not state_id or not state_data:
                raise StateConsistencyError("Invalid state data provided")

            # Use database service with proper model
            import json
            import uuid

            from src.database.models.state import StateSnapshot

            # Check for existing snapshot
            existing_snapshots = await self.database_service.list_entities(
                StateSnapshot,
                filters={"snapshot_id": f"{state_id}_{state_type.value}_{metadata.version}"},
                limit=1,
            )

            if existing_snapshots:
                # Update existing snapshot
                snapshot = existing_snapshots[0]
                snapshot.state_data = {"state_data": state_data}
                snapshot.state_checksum = metadata.checksum
                snapshot.updated_at = datetime.now(timezone.utc)
                await self.database_service.update_entity(snapshot)
            else:
                # Create new snapshot
                snapshot = StateSnapshot(
                    snapshot_id=uuid.uuid4(),
                    name=f"{state_type.value}_{state_id}",
                    description=f"State for {state_type.value}:{state_id}",
                    snapshot_type="automatic",
                    state_data={"state_data": state_data},
                    raw_size_bytes=len(json.dumps(state_data)),
                    state_checksum=metadata.checksum,
                    schema_version="1.0.0",
                    status="active",
                )
                await self.database_service.create_entity(snapshot)

            self.logger.debug(f"State saved successfully: {state_type.value}:{state_id}")
            return True

        except DatabaseError as e:
            self.logger.error(f"Database service error saving state: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            raise StateConsistencyError(f"State save failed: {e}") from e

    async def load_state(self, state_type: "StateType", state_id: str) -> dict[str, Any] | None:
        """
        Load state data from persistent storage.

        Args:
            state_type: Type of state
            state_id: State identifier

        Returns:
            State data or None if not found

        Raises:
            StateConsistencyError: If load operation fails
        """
        try:
            if not self.database_service:
                self.logger.warning("No database service available")
                return None

            # Use database service with proper model
            from src.database.models.state import StateSnapshot

            # Find snapshot by name pattern
            snapshots = await self.database_service.list_entities(
                StateSnapshot,
                filters={"name": f"{state_type.value}_{state_id}", "status": "active"},
                order_by="created_at",
                order_desc=True,
                limit=1,
            )

            if snapshots and snapshots[0].state_data:
                snapshot_data = snapshots[0].state_data
                if isinstance(snapshot_data, dict) and "state_data" in snapshot_data:
                    return snapshot_data["state_data"]
                return snapshot_data

            return None

        except DatabaseError as e:
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
            StateConsistencyError: If delete operation fails
        """
        try:
            if not self.database_service:
                self.logger.warning("No database service available")
                return False

            # Use database service with proper model
            from src.database.models.state import StateSnapshot

            # Find all snapshots for this state
            snapshots = await self.database_service.list_entities(
                StateSnapshot, filters={"name": f"{state_type.value}_{state_id}"}
            )

            # Delete each snapshot
            for snapshot in snapshots:
                await self.database_service.delete_entity(StateSnapshot, snapshot.snapshot_id)

            self.logger.debug(f"State deleted successfully: {state_type.value}:{state_id}")
            return True

        except DatabaseError as e:
            self.logger.error(f"Database service error deleting state: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete state: {e}")
            raise StateConsistencyError(f"State delete failed: {e}") from e

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

            # Use database service with proper model
            from src.database.models.state import StateSnapshot

            # List snapshots with the state type prefix
            snapshots = await self.database_service.list_entities(
                StateSnapshot,
                filters={"status": "active"},
                order_by="created_at",
                order_desc=True,
                limit=limit,
                offset=offset,
            )

            states = []
            for snapshot in snapshots:
                # Filter by state type in name
                if snapshot.name and snapshot.name.startswith(f"{state_type.value}_"):
                    if snapshot.state_data:
                        snapshot_data = snapshot.state_data
                        if isinstance(snapshot_data, dict) and "state_data" in snapshot_data:
                            state_data = snapshot_data["state_data"]
                        else:
                            state_data = snapshot_data

                        # Extract state_id from name
                        state_id = snapshot.name.replace(f"{state_type.value}_", "")

                        states.append(
                            {
                                "state_id": state_id,
                                "data": state_data,
                                "version": 1,  # Default version
                                "updated_at": snapshot.updated_at,
                            }
                        )

            return states

        except DatabaseError as e:
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

            # Use database service with proper model
            import json
            import uuid

            from src.database.models.state import StateSnapshot as StateSnapshotModel

            # Create StateSnapshot entity
            snapshot_entity = StateSnapshotModel(
                snapshot_id=uuid.uuid4(),
                name=f"system_snapshot_{snapshot.snapshot_id}",
                description=snapshot.description,
                snapshot_type="manual",
                state_data=snapshot.__dict__,
                raw_size_bytes=len(json.dumps(snapshot.__dict__, default=str)),
                state_checksum=snapshot.snapshot_id,  # Use snapshot_id as checksum for now
                schema_version="1.0.0",
                status="active",
            )

            await self.database_service.create_entity(snapshot_entity)

            self.logger.debug(f"Snapshot saved successfully: {snapshot.snapshot_id}")
            return True

        except DatabaseError as e:
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

            # Use database service with proper model
            from src.database.models.state import StateSnapshot as StateSnapshotModel

            # Find snapshot by name pattern
            snapshots = await self.database_service.list_entities(
                StateSnapshotModel, filters={"name": f"system_snapshot_{snapshot_id}"}, limit=1
            )

            if snapshots and snapshots[0].state_data:
                from ..state_service import StateSnapshot as _StateSnapshot

                return _StateSnapshot(**snapshots[0].state_data)

            return None

        except DatabaseError as e:
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
        """Publish save operation event using consistent pub/sub pattern."""
        await self._messaging_coordinator.publish(
            topic="persistence.save_requested",
            data={
                "operation": "save",
                "state_type": state_type.value,
                "state_id": state_id,
                "state_data": state_data,
                "metadata": metadata.__dict__,
            },
            source="StatePersistenceService",
        )

    async def queue_delete_operation(self, state_type: "StateType", state_id: str) -> None:
        """Publish delete operation event using consistent pub/sub pattern."""
        await self._messaging_coordinator.publish(
            topic="persistence.delete_requested",
            data={
                "operation": "delete",
                "state_type": state_type.value,
                "state_id": state_id,
            },
            source="StatePersistenceService",
        )

    # Private methods

    async def _process_operations(self) -> None:
        """Process operations using event-driven patterns for consistency."""
        # Set up event handlers for save and delete operations
        from src.utils.messaging_patterns import DataTransformationHandler, MessagePattern

        # Register consistent event handlers
        self._messaging_coordinator.register_handler(
            MessagePattern.PUB_SUB, DataTransformationHandler(self._handle_persistence_event)
        )

        # Subscribe to persistence events
        self._messaging_coordinator._event_emitter.on_pattern(
            "persistence.*", self._handle_persistence_event
        )

        while self._is_running:
            try:
                # Event-driven processing - wait for events
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in event-driven processing: {e}")
                await asyncio.sleep(1.0)

    async def _handle_persistence_event(self, event_data: dict[str, Any]) -> None:
        """Handle persistence events using consistent pub/sub pattern."""
        try:
            operation = event_data.get("operation")
            if operation == "save":
                from src.core.types import StateType

                # Convert string back to enum for compatibility
                state_type_str = event_data.get("state_type")
                state_type = StateType(state_type_str) if state_type_str else None

                await self.save_state(
                    state_type,
                    event_data.get("state_id"),
                    event_data.get("state_data"),
                    event_data.get("metadata"),  # Already a dict from event
                )
            elif operation == "delete":
                from src.core.types import StateType

                state_type_str = event_data.get("state_type")
                state_type = StateType(state_type_str) if state_type_str else None

                await self.delete_state(state_type, event_data.get("state_id"))
        except Exception as e:
            self.logger.error(f"Error handling persistence event: {e}")

    async def _flush_events(self) -> None:
        """Flush remaining events using pub/sub pattern for consistency."""
        try:
            # Notify that service is shutting down via event
            await self._messaging_coordinator.publish(
                topic="persistence.service_stopping",
                data={"status": "stopping", "service": "StatePersistenceService"},
                source="StatePersistenceService",
            )
            self.logger.info("Published service stopping event")

        except Exception as e:
            self.logger.error(f"Error flushing events: {e}")

    def is_available(self) -> bool:
        """Check if persistence service is available."""
        return self.database_service is not None and self._is_running

    # Helper methods removed - now using proper database service interface with models
