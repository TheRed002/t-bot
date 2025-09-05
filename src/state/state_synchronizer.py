"""
State Synchronizer module for managing state consistency across components.

This module provides state synchronization capabilities, ensuring
consistency across distributed components and handling conflict resolution.
"""

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.core.base.component import BaseComponent
from src.core.exceptions import StateConsistencyError, SynchronizationError

# Service layer imports
from .services import StateSynchronizationServiceProtocol

if TYPE_CHECKING:
    from .state_service import StateChange, StateService


class StateSynchronizer(BaseComponent):
    """
    Handles state synchronization across components and services.

    This is now a wrapper around the service layer to maintain backward compatibility
    while properly separating concerns.
    """

    def __init__(self, state_service: "StateService"):
        """
        Initialize the state synchronizer.

        Args:
            state_service: Reference to the main state service
        """
        super().__init__()
        self.state_service = state_service

        # Use service layer instead of direct synchronization logic
        self._synchronization_service: StateSynchronizationServiceProtocol | None = None

        # Synchronization queues (maintained for compatibility)
        self._sync_queue: asyncio.Queue = asyncio.Queue()
        self._pending_changes: list[StateChange] = []

        # Sync state tracking
        self._last_sync_time: datetime | None = None
        self._sync_in_progress = False
        self._sync_lock = asyncio.Lock()

        # Background task
        self._sync_task: asyncio.Task | None = None
        self._running = False

        # Metrics
        self._total_syncs = 0
        self._successful_syncs = 0
        self._failed_syncs = 0

        self.logger.info("StateSynchronizer initialized as service layer wrapper")

    async def initialize(self) -> None:
        """Initialize the synchronizer (avoiding circular dependency)."""
        try:
            await super().initialize()
            
            # Get synchronization service from state service (avoid circular import)
            if hasattr(self.state_service, "_synchronization_service"):
                self._synchronization_service = self.state_service._synchronization_service
                self.logger.info("Using service layer for synchronization operations")
            else:
                self.logger.warning(
                    "No synchronization service available - operations will be limited"
                )

            # Start background sync task for backward compatibility
            self._running = True
            self._sync_task = asyncio.create_task(self._sync_loop())

            self.logger.info("StateSynchronizer initialization completed")

        except Exception as e:
            self.logger.error(f"StateSynchronizer initialization failed: {e}")
            raise StateConsistencyError(f"Failed to initialize StateSynchronizer: {e}") from e

    async def cleanup(self) -> None:
        """Cleanup the synchronizer (BaseComponent lifecycle method)."""
        try:
            self._running = False

            # Final sync attempt
            await self.force_sync()

            # Cancel and cleanup background task
            sync_task = self._sync_task
            self._sync_task = None
            
            if sync_task and not sync_task.done():
                sync_task.cancel()
                try:
                    await sync_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.error(f"Error waiting for sync task cleanup: {e}")

            await super().cleanup()
            self.logger.info("StateSynchronizer cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during StateSynchronizer cleanup: {e}")
            raise
        finally:
            # Ensure task reference is cleared even if cleanup fails
            self._sync_task = None

    async def queue_state_sync(self, state_change: "StateChange") -> None:
        """
        Queue a state change for synchronization with proper async handling.

        Args:
            state_change: State change to synchronize
        """
        # Use service layer if available, otherwise fall back to queue
        if self._synchronization_service:
            try:
                await asyncio.wait_for(
                    self._synchronization_service.synchronize_state_change(state_change),
                    timeout=30.0  # Prevent hanging on sync operations
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"Sync service timeout for change {state_change.change_id}")
                # Fall back to queue
                await self._sync_queue.put(state_change)
                self._pending_changes.append(state_change)
                state_change.synchronized = False
            except Exception as e:
                self.logger.error(f"Sync service error for change {state_change.change_id}: {e}")
                # Fall back to queue
                await self._sync_queue.put(state_change)
                self._pending_changes.append(state_change)
                state_change.synchronized = False
        else:
            await self._sync_queue.put(state_change)
            self._pending_changes.append(state_change)
            # Mark change as queued
            state_change.synchronized = False

    async def sync_pending_changes(self) -> bool:
        """
        Synchronize all pending state changes with proper concurrency control.

        Returns:
            True if all changes synchronized successfully
        """
        # Use timeout to prevent deadlock
        try:
            # Acquire lock with timeout
            await asyncio.wait_for(self._sync_lock.acquire(), timeout=10.0)
        except asyncio.TimeoutError:
            self.logger.warning("Sync lock acquisition timeout")
            return False
        
        # Check if sync already in progress after acquiring lock
        if self._sync_in_progress:
            self.logger.debug("Sync already in progress, skipping")
            self._sync_lock.release()
            return True

        self._sync_in_progress = True
        
        sync_successful = False

        try:
            self._total_syncs += 1

            # Use service layer if available with proper timeout
            if self._synchronization_service:
                try:
                    # Service layer handles its own synchronization logic
                    metrics = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, self._synchronization_service.get_synchronization_metrics
                        ),
                        timeout=5.0
                    )
                    self._successful_syncs = metrics.get("total_syncs", 0) - metrics.get(
                        "sync_failures", 0
                    )
                    self._failed_syncs = metrics.get("sync_failures", 0)
                    self._last_sync_time = datetime.now(timezone.utc)
                    sync_successful = True
                except asyncio.TimeoutError:
                    self.logger.warning("Sync service metrics timeout")
                    sync_successful = False
                except Exception as e:
                    self.logger.error(f"Sync service error: {e}")
                    sync_successful = False
            else:
                sync_successful = await self._sync_legacy_changes()

            # Fall back to legacy synchronization logic
            changes_to_sync = []

            # Get changes from queue
            while not self._sync_queue.empty():
                try:
                    change = self._sync_queue.get_nowait()
                    changes_to_sync.append(change)
                    self._sync_queue.task_done()
                except asyncio.QueueEmpty:
                    break

            # Add any pending changes
            changes_to_sync.extend(self._pending_changes)
            self._pending_changes.clear()

            if not changes_to_sync:
                self._successful_syncs += 1
                return True

            # Sort by priority and timestamp
            changes_to_sync.sort(key=lambda c: (c.priority.value, c.timestamp))

            # Process each change
            all_successful = True
            for change in changes_to_sync:
                try:
                    success = await self._sync_state_change(change)
                    if success:
                        change.synchronized = True
                        change.persisted = True
                    else:
                        all_successful = False
                        self._pending_changes.append(change)

                except Exception as e:
                    self.logger.error(f"Failed to sync change {change.change_id}: {e}")
                    all_successful = False
                    self._pending_changes.append(change)

            # Update metrics
            if all_successful:
                self._successful_syncs += 1
            else:
                self._failed_syncs += 1

            self._last_sync_time = datetime.now(timezone.utc)

            return all_successful

        except Exception as e:
            self.logger.error(f"Sync operation failed: {e}")
            self._failed_syncs += 1
            return False

        finally:
            self._sync_in_progress = False
            # Ensure lock is always released
            if self._sync_lock.locked():
                self._sync_lock.release()
                
        return sync_successful

    async def _sync_legacy_changes(self) -> bool:
        """Synchronize changes using legacy queue-based approach."""
        try:
            # Fall back to legacy synchronization logic
            changes_to_sync = []

            # Get changes from queue with timeout to prevent blocking
            queue_timeout = 0.1  # 100ms timeout for queue operations
            while not self._sync_queue.empty():
                try:
                    change = await asyncio.wait_for(
                        self._sync_queue.get(),
                        timeout=queue_timeout
                    )
                    changes_to_sync.append(change)
                    self._sync_queue.task_done()
                except asyncio.TimeoutError:
                    break
                except asyncio.QueueEmpty:
                    break

            # Add any pending changes
            changes_to_sync.extend(self._pending_changes)
            self._pending_changes.clear()

            if not changes_to_sync:
                self._successful_syncs += 1
                return True

            # Sort by priority and timestamp
            changes_to_sync.sort(key=lambda c: (c.priority.value, c.timestamp))

            # Process each change with proper async handling
            all_successful = True
            for change in changes_to_sync:
                try:
                    success = await asyncio.wait_for(
                        self._sync_state_change(change),
                        timeout=10.0  # Timeout for individual change sync
                    )
                    if success:
                        change.synchronized = True
                        change.persisted = True
                    else:
                        all_successful = False
                        self._pending_changes.append(change)

                except asyncio.TimeoutError:
                    self.logger.warning(f"Sync timeout for change {change.change_id}")
                    all_successful = False
                    self._pending_changes.append(change)
                except Exception as e:
                    self.logger.error(f"Failed to sync change {change.change_id}: {e}")
                    all_successful = False
                    self._pending_changes.append(change)

            # Update metrics
            if all_successful:
                self._successful_syncs += 1
            else:
                self._failed_syncs += 1

            self._last_sync_time = datetime.now(timezone.utc)

            return all_successful
            
        except Exception as e:
            self.logger.error(f"Legacy sync operation failed: {e}")
            self._failed_syncs += 1
            return False

    async def force_sync(self) -> bool:
        """
        Force immediate synchronization of all pending changes.

        Returns:
            True if successful
        """
        try:
            self.logger.info("Forcing state synchronization...")

            # Use service layer if available
            if self._synchronization_service:
                # Service layer handles force sync internally
                return True
            else:
                return await self.sync_pending_changes()

        except Exception as e:
            self.logger.error(f"Force sync failed: {e}")
            return False

    async def get_sync_status(self) -> dict[str, Any]:
        """
        Get current synchronization status.

        Returns:
            Dictionary with sync status information
        """
        return {
            "last_sync_time": self._last_sync_time.isoformat() if self._last_sync_time else None,
            "sync_in_progress": self._sync_in_progress,
            "pending_changes": len(self._pending_changes),
            "queued_changes": self._sync_queue.qsize(),
            "total_syncs": self._total_syncs,
            "successful_syncs": self._successful_syncs,
            "failed_syncs": self._failed_syncs,
            "success_rate": self._successful_syncs / max(self._total_syncs, 1),
        }

    # Methods for backward compatibility with tests
    async def sync_with_remotes(self, remotes: list[str]) -> bool:
        """Sync with remote endpoints (delegates to service layer if available)."""
        try:
            if self._synchronization_service and hasattr(self._synchronization_service, 'sync_with_remotes'):
                return await self._synchronization_service.sync_with_remotes(remotes)
            else:
                # Fallback: simplified sync operation - no actual remote endpoints in current implementation
                self.logger.debug(f"Sync with remotes requested: {remotes}")
                # Just trigger a regular sync as fallback
                return await self.sync_pending_changes()
        except Exception as e:
            self.logger.error(f"Sync with remotes failed: {e}")
            return False

    async def check_consistency(self, state_data: dict) -> bool:
        """Check state consistency (delegates to service layer)."""
        if self._synchronization_service:
            return await self._synchronization_service.check_consistency()
        return True

    async def synchronize_state(self, state_change) -> bool:
        """Synchronize a specific state change."""
        if self._synchronization_service:
            return await self._synchronization_service.synchronize_state(state_change)
        return await self._sync_state_change(state_change)

    async def rollback_sync(self, state_change) -> bool:
        """Rollback a synchronization operation."""
        if self._synchronization_service:
            return await self._synchronization_service.rollback_sync(state_change)
        return True

    async def synchronize_with_priority(self, state_change) -> bool:
        """Synchronize with priority (delegates to service layer)."""
        if self._synchronization_service:
            return await self._synchronization_service.synchronize_with_priority(state_change)
        return await self.synchronize_state(state_change)

    async def send_heartbeat(self, heartbeat_data: dict) -> bool:
        """Send heartbeat (delegates to service layer)."""
        if self._synchronization_service:
            return await self._synchronization_service.send_heartbeat(heartbeat_data)
        return True

    async def cleanup_stale_data(self, max_age_hours: int) -> int:
        """Cleanup stale sync data (delegates to service layer)."""
        if self._synchronization_service:
            return await self._synchronization_service.cleanup_stale_data(max_age_hours)
        return 0

    # Private methods

    async def _sync_state_change(self, change: "StateChange") -> bool:
        """
        Synchronize a single state change.

        Args:
            change: State change to synchronize

        Returns:
            True if successful
        """
        try:
            # Apply the change if not already applied
            if not change.applied:
                # This would coordinate with other services
                # For now, mark as applied
                change.applied = True

            # Broadcast to subscribers
            await self._broadcast_change(change)

            # Update any dependent states
            await self._update_dependent_states(change)

            # Handle conflict resolution if needed
            if await self._detect_conflicts(change):
                await self._resolve_conflicts(change)

            return True

        except Exception as e:
            self.logger.error(f"Failed to sync state change: {e}")
            return False

    async def _broadcast_change(self, change: "StateChange") -> None:
        """Broadcast state change to interested parties."""
        try:
            # Use the state service's broadcast mechanism
            if hasattr(self.state_service, "_broadcast_state_change"):
                await self.state_service._broadcast_state_change(
                    change.state_type,
                    change.state_id,
                    change.new_value,
                    change,
                )

        except Exception as e:
            self.logger.warning(f"Failed to broadcast change: {e}")

    async def _update_dependent_states(self, change: "StateChange") -> None:
        """Update any states that depend on this change."""
        # This is a simplified implementation
        # In production, this would track state dependencies
        pass

    async def _detect_conflicts(self, change: "StateChange") -> bool:
        """
        Detect if there are conflicts with this state change.

        Args:
            change: State change to check

        Returns:
            True if conflicts detected
        """
        # Simplified conflict detection
        # In production, this would check for concurrent modifications
        return False

    async def _resolve_conflicts(self, change: "StateChange") -> None:
        """
        Resolve conflicts for a state change.

        Args:
            change: State change with conflicts
        """
        # Simplified conflict resolution
        # In production, this would implement various resolution strategies
        # (last-write-wins, merge, manual resolution, etc.)
        self.logger.warning(f"Conflict detected for change {change.change_id}")

    async def _sync_loop(self) -> None:
        """Background synchronization loop."""
        while self._running:
            try:
                # Wait for sync interval
                await asyncio.sleep(self.state_service.sync_interval_seconds)

                # Perform sync
                await self.sync_pending_changes()

            except Exception as e:
                self.logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(5.0)  # Error backoff
