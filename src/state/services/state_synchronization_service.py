"""
State Synchronization Service - Handles state synchronization operations.

This service provides synchronization capabilities for state changes,
ensuring consistency across distributed components without tight
coupling to infrastructure concerns.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol

from src.core.base.service import BaseService
from src.core.exceptions import StateConsistencyError

if TYPE_CHECKING:
    from ..state_service import StateChange, StateType


class StateSynchronizationServiceProtocol(Protocol):
    """Protocol defining the state synchronization service interface."""

    async def synchronize_state_change(self, state_change: "StateChange") -> bool: ...

    async def broadcast_state_change(
        self,
        state_type: "StateType",
        state_id: str,
        state_data: dict[str, Any] | None,
        change_info: dict[str, Any],
    ) -> None: ...

    async def resolve_conflicts(
        self,
        state_type: "StateType",
        state_id: str,
        conflicting_changes: list["StateChange"],
    ) -> "StateChange": ...


class StateSynchronizationService(BaseService):
    """
    State synchronization service providing distributed state consistency.

    This service handles synchronization of state changes across components
    and provides conflict resolution mechanisms.
    """

    def __init__(self, event_service: Any = None):
        """
        Initialize the state synchronization service.

        Args:
            event_service: Injected event service dependency for broadcasting
        """
        super().__init__(name="StateSynchronizationService")

        # Injected dependency for event broadcasting
        self.event_service = event_service

        if event_service is None:
            self.logger.info(
                "StateSynchronizationService initialized without event_service - broadcasting will be limited"
            )

        # Synchronization configuration
        self.enable_conflict_resolution = True
        self.sync_timeout_seconds = 30
        self.max_retry_attempts = 3

        # Synchronization state
        self._pending_syncs: dict[str, StateChange] = {}
        self._sync_locks: dict[str, asyncio.Lock] = {}
        self._subscribers: dict[str, set[Callable]] = {}

        # Backpressure handling
        self._sync_semaphore = asyncio.Semaphore(50)  # Limit concurrent syncs
        self._sync_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)  # Bounded queue
        self._high_priority_queue: asyncio.Queue = asyncio.Queue(maxsize=500)  # Priority queue

        # Metrics
        self._sync_count = 0
        self._sync_failures = 0
        self._conflicts_resolved = 0
        self._queue_full_errors = 0

        self.logger.info(
            f"StateSynchronizationService initialized with event_service: {type(event_service).__name__ if event_service else 'None'}"
        )

    async def synchronize_state_change(self, state_change: "StateChange") -> bool:
        """
        Synchronize a state change across the system with backpressure handling.

        Args:
            state_change: State change to synchronize

        Returns:
            True if synchronization successful
        """
        try:
            # Apply backpressure control - use semaphore to limit concurrent operations
            async with self._sync_semaphore:
                self._sync_count += 1
                sync_key = f"{state_change.state_type.value}:{state_change.state_id}"

                # Queue state change based on priority to prevent overwhelming
                target_queue = (
                    self._high_priority_queue
                    if state_change.priority.value in ["critical", "high"]
                    else self._sync_queue
                )

                try:
                    target_queue.put_nowait(state_change)
                except asyncio.QueueFull:
                    self.logger.warning(f"Sync queue full for {sync_key} - applying backpressure")
                    self._queue_full_errors += 1
                    return False

                # Get or create lock for this state
                if sync_key not in self._sync_locks:
                    self._sync_locks[sync_key] = asyncio.Lock()

                async with self._sync_locks[sync_key]:
                    # Process from queue
                    try:
                        queued_change = await asyncio.wait_for(
                            target_queue.get(), timeout=self.sync_timeout_seconds
                        )
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Sync queue timeout for {sync_key}")
                        self._sync_failures += 1
                        return False

                    # Check for conflicts
                    if await self._detect_conflicts(queued_change):
                        if self.enable_conflict_resolution:
                            resolved_change = await self._resolve_conflict(queued_change)
                            if resolved_change:
                                queued_change = resolved_change
                                self._conflicts_resolved += 1
                            else:
                                self._sync_failures += 1
                                target_queue.task_done()
                                return False
                        else:
                            self.logger.error(f"Sync conflict detected for {sync_key}")
                            self._sync_failures += 1
                            target_queue.task_done()
                            return False

                    # Perform synchronization steps
                    success = await self._perform_synchronization(queued_change)

                    if success:
                        # Mark as synchronized
                        queued_change.synchronized = True
                        queued_change.persisted = True

                        # Broadcast to subscribers
                        await self._notify_subscribers(queued_change)

                        # Clean up pending sync
                        self._pending_syncs.pop(sync_key, None)

                        self.logger.debug(f"State change synchronized: {sync_key}")
                    else:
                        self._sync_failures += 1

                    target_queue.task_done()
                    return success

        except Exception as e:
            self.logger.error(f"State synchronization failed: {e}")
            self._sync_failures += 1
            return False

    async def broadcast_state_change(
        self,
        state_type: "StateType",
        state_id: str,
        state_data: dict[str, Any] | None,
        change_info: dict[str, Any],
    ) -> None:
        """
        Broadcast state change to interested parties.

        Args:
            state_type: Type of state that changed
            state_id: State identifier
            state_data: New state data
            change_info: Additional change information
        """
        try:
            broadcast_event = {
                "event_type": "state_change",
                "state_type": state_type.value,
                "state_id": state_id,
                "state_data": state_data,
                "change_info": change_info,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Broadcast to type-specific subscribers
            type_key = state_type.value
            await self._send_to_subscribers(type_key, broadcast_event)

            # Broadcast to global subscribers
            await self._send_to_subscribers("*", broadcast_event)

            self.logger.debug(f"State change broadcasted: {state_type.value}:{state_id}")

        except Exception as e:
            self.logger.error(f"State change broadcast failed: {e}")

    async def resolve_conflicts(
        self,
        state_type: "StateType",
        state_id: str,
        conflicting_changes: list["StateChange"],
    ) -> "StateChange":
        """
        Resolve conflicts between multiple state changes.

        Args:
            state_type: Type of state
            state_id: State identifier
            conflicting_changes: List of conflicting changes

        Returns:
            Resolved state change
        """
        try:
            if not conflicting_changes:
                raise StateConsistencyError("No conflicting changes provided")

            if len(conflicting_changes) == 1:
                return conflicting_changes[0]

            # Apply conflict resolution strategy
            resolved_change = await self._apply_conflict_resolution_strategy(
                state_type, state_id, conflicting_changes
            )

            self.logger.info(
                f"Conflict resolved for {state_type.value}:{state_id}",
                extra={"num_conflicts": len(conflicting_changes)},
            )

            return resolved_change

        except Exception as e:
            self.logger.error(f"Conflict resolution failed: {e}")
            # Fall back to most recent change
            return max(conflicting_changes, key=lambda c: c.timestamp)

    def subscribe_to_state_changes(self, state_type: str, callback: Callable) -> None:
        """
        Subscribe to state change notifications.

        Args:
            state_type: State type to subscribe to (or "*" for all)
            callback: Callback function to invoke on changes
        """
        if state_type not in self._subscribers:
            self._subscribers[state_type] = set()

        self._subscribers[state_type].add(callback)
        self.logger.debug(f"Added subscriber for {state_type} state changes")

    def unsubscribe_from_state_changes(self, state_type: str, callback: Callable) -> None:
        """
        Unsubscribe from state change notifications.

        Args:
            state_type: State type to unsubscribe from
            callback: Callback function to remove
        """
        if state_type in self._subscribers:
            self._subscribers[state_type].discard(callback)
            self.logger.debug(f"Removed subscriber for {state_type} state changes")

    def get_synchronization_metrics(self) -> dict[str, Any]:
        """Get synchronization service metrics including backpressure indicators."""
        return {
            "total_syncs": self._sync_count,
            "sync_failures": self._sync_failures,
            "success_rate": (self._sync_count - self._sync_failures) / max(self._sync_count, 1),
            "conflicts_resolved": self._conflicts_resolved,
            "pending_syncs": len(self._pending_syncs),
            "active_subscribers": sum(len(subs) for subs in self._subscribers.values()),
            "queue_full_errors": self._queue_full_errors,
            "sync_queue_size": self._sync_queue.qsize(),
            "high_priority_queue_size": self._high_priority_queue.qsize(),
            "semaphore_available": self._sync_semaphore._value,
        }

    # Private helper methods

    async def _detect_conflicts(self, state_change: "StateChange") -> bool:
        """Detect if there are conflicts with the state change."""
        try:
            sync_key = f"{state_change.state_type.value}:{state_change.state_id}"

            # Check if there's already a pending sync for this state
            if sync_key in self._pending_syncs:
                existing_change = self._pending_syncs[sync_key]

                # Conflict if timestamps are very close but changes differ
                time_diff = abs(
                    (state_change.timestamp - existing_change.timestamp).total_seconds()
                )

                if time_diff < 1.0 and state_change.new_value != existing_change.new_value:
                    self.logger.warning(f"Conflict detected for {sync_key}")
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Conflict detection failed: {e}")
            return False

    async def _resolve_conflict(self, state_change: "StateChange") -> "StateChange | None":
        """Resolve a conflict for a state change."""
        try:
            sync_key = f"{state_change.state_type.value}:{state_change.state_id}"
            existing_change = self._pending_syncs.get(sync_key)

            if not existing_change:
                return state_change

            # Apply resolution strategy
            conflicting_changes = [existing_change, state_change]
            resolved_change = await self.resolve_conflicts(
                state_change.state_type, state_change.state_id, conflicting_changes
            )

            return resolved_change

        except Exception as e:
            self.logger.error(f"Conflict resolution failed: {e}")
            return None

    async def _perform_synchronization(self, state_change: "StateChange") -> bool:
        """Perform the actual synchronization steps."""
        try:
            # Add to pending syncs
            sync_key = f"{state_change.state_type.value}:{state_change.state_id}"
            self._pending_syncs[sync_key] = state_change

            # Simulate synchronization steps
            await self._validate_change(state_change)
            await self._apply_change(state_change)
            await self._confirm_sync(state_change)

            return True

        except Exception as e:
            self.logger.error(f"Synchronization steps failed: {e}")
            return False

    async def _validate_change(self, state_change: "StateChange") -> None:
        """Validate the state change before applying."""
        # Basic validation
        if not state_change.state_id:
            raise StateConsistencyError("State ID is required")

        if not state_change.new_value:
            raise StateConsistencyError("New state value is required")

    async def _apply_change(self, state_change: "StateChange") -> None:
        """Apply the state change."""
        # Mark change as applied
        state_change.applied = True

        self.logger.debug(
            f"State change applied: {state_change.state_type.value}:{state_change.state_id}"
        )

    async def _confirm_sync(self, state_change: "StateChange") -> None:
        """Confirm synchronization completion."""
        # Update change status
        state_change.synchronized = True

        self.logger.debug(
            f"State change sync confirmed: {state_change.state_type.value}:{state_change.state_id}"
        )

    async def _notify_subscribers(self, state_change: "StateChange") -> None:
        """Notify subscribers of the state change."""
        try:
            event_data = {
                "change_id": state_change.change_id,
                "state_type": state_change.state_type.value,
                "state_id": state_change.state_id,
                "operation": state_change.operation.value,
                "priority": state_change.priority.value,
                "timestamp": state_change.timestamp.isoformat(),
                "source_component": state_change.source_component,
                "reason": state_change.reason,
            }

            # Notify type-specific subscribers
            type_key = state_change.state_type.value
            await self._send_to_subscribers(type_key, event_data)

            # Notify global subscribers
            await self._send_to_subscribers("*", event_data)

        except Exception as e:
            self.logger.error(f"Subscriber notification failed: {e}")

    async def _send_to_subscribers(self, subscription_key: str, event_data: dict[str, Any]) -> None:
        """Send event data to subscribers."""
        try:
            subscribers = self._subscribers.get(subscription_key, set())

            for callback in subscribers.copy():  # Copy to avoid modification during iteration
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event_data)
                    else:
                        callback(event_data)
                except Exception as e:
                    self.logger.error(f"Subscriber callback failed: {e}")
                    # Remove failing callback
                    self._subscribers[subscription_key].discard(callback)

        except Exception as e:
            self.logger.error(f"Error sending to subscribers: {e}")

    async def _apply_conflict_resolution_strategy(
        self,
        state_type: "StateType",
        state_id: str,
        conflicting_changes: list["StateChange"],
    ) -> "StateChange":
        """Apply conflict resolution strategy to resolve conflicts."""
        try:
            # Strategy 1: Priority-based resolution
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            highest_priority_change = min(
                conflicting_changes, key=lambda c: priority_order.get(c.priority.value, 4)
            )

            # Strategy 2: Timestamp-based resolution (last-write-wins)
            most_recent_change = max(conflicting_changes, key=lambda c: c.timestamp)

            # Strategy 3: Source-based resolution (trusted sources win)
            trusted_sources = {"StateService", "TradeLifecycleManager", "RiskManager"}
            trusted_changes = [
                c for c in conflicting_changes if c.source_component in trusted_sources
            ]

            # Apply resolution logic
            if highest_priority_change.priority.value == "critical":
                # Critical priority always wins
                resolved_change = highest_priority_change
                strategy = "priority"
            elif trusted_changes:
                # Trusted sources win over untrusted
                resolved_change = max(trusted_changes, key=lambda c: c.timestamp)
                strategy = "trust+timestamp"
            else:
                # Fall back to most recent change
                resolved_change = most_recent_change
                strategy = "timestamp"

            self.logger.info(
                f"Conflict resolved using {strategy} strategy",
                extra={
                    "state_type": state_type.value,
                    "state_id": state_id,
                    "winning_change": resolved_change.change_id,
                    "winning_source": resolved_change.source_component,
                },
            )

            # Create audit record of conflict resolution
            await self._create_conflict_audit_record(
                state_type, state_id, conflicting_changes, resolved_change, strategy
            )

            return resolved_change

        except Exception as e:
            self.logger.error(f"Conflict resolution strategy failed: {e}")
            # Fall back to most recent change
            return max(conflicting_changes, key=lambda c: c.timestamp)

    async def _create_conflict_audit_record(
        self,
        state_type: "StateType",
        state_id: str,
        conflicting_changes: list["StateChange"],
        resolved_change: "StateChange",
        strategy: str,
    ) -> None:
        """Create audit record for conflict resolution."""
        try:
            audit_record = {
                "event": "conflict_resolution",
                "state_type": state_type.value,
                "state_id": state_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "resolution_strategy": strategy,
                "conflicting_changes": [c.change_id for c in conflicting_changes],
                "winning_change": resolved_change.change_id,
                "winning_source": resolved_change.source_component,
            }

            # In a full implementation, this would be written to audit log
            self.logger.info(f"Conflict audit record created: {audit_record}")

        except Exception as e:
            self.logger.error(f"Failed to create conflict audit record: {e}")

    async def cleanup_expired_locks(self) -> None:
        """Clean up expired synchronization locks."""
        try:
            expired_keys = []

            for key, lock in self._sync_locks.items():
                if not lock.locked():
                    expired_keys.append(key)

            # Remove expired locks in batches
            for key in expired_keys[:100]:  # Limit cleanup batch size
                self._sync_locks.pop(key, None)

            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys[:100])} expired sync locks")

        except Exception as e:
            self.logger.error(f"Lock cleanup failed: {e}")
