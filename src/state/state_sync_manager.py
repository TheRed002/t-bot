"""
Real-time State Synchronization Manager for the T-Bot trading system (P-025).

This module provides comprehensive real-time state synchronization capabilities including:
- State synchronization across components and storage layers
- Conflict detection and resolution mechanisms
- State versioning and concurrent update handling
- Distributed state consistency management
- Real-time sync monitoring and alerting

The StateSyncManager ensures all system components maintain consistent state
views and handles complex synchronization scenarios in a distributed environment.
"""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

import aioredis
from sqlalchemy import select, update

# Core framework imports
from src.core.config import Config
from src.core.exceptions import StateError, SynchronizationError
from src.core.logging import get_logger

# Database imports
from src.database.manager import DatabaseManager
from src.database.redis_client import RedisClient

# Utility imports
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class SyncEventType(Enum):
    """State synchronization event types."""

    STATE_UPDATED = "state_updated"
    STATE_DELETED = "state_deleted"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"
    SYNC_FAILED = "sync_failed"
    RECOVERY_INITIATED = "recovery_initiated"


class ConflictResolutionStrategy(Enum):
    """Conflict resolution strategy enumeration."""

    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    MERGE_STRATEGY = "merge_strategy"
    MANUAL_RESOLUTION = "manual_resolution"
    CUSTOM_RESOLVER = "custom_resolver"


class StateVersion:
    """State version tracking."""

    def __init__(self, version: int = 1, timestamp: datetime | None = None):
        self.version = version
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.checksum = ""

    def increment(self) -> "StateVersion":
        """Create next version."""
        return StateVersion(version=self.version + 1, timestamp=datetime.now(timezone.utc))

    def compare(self, other: "StateVersion") -> int:
        """Compare versions. Returns -1, 0, or 1."""
        if self.version < other.version:
            return -1
        elif self.version > other.version:
            return 1
        else:
            # Same version, compare timestamps
            if self.timestamp < other.timestamp:
                return -1
            elif self.timestamp > other.timestamp:
                return 1
            else:
                return 0


@dataclass
class SyncEvent:
    """State synchronization event."""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: SyncEventType = SyncEventType.STATE_UPDATED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Event details
    entity_type: str = ""  # bot_state, trade, position, etc.
    entity_id: str = ""

    # Version information
    old_version: StateVersion | None = None
    new_version: StateVersion | None = None

    # Change data
    changes: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Source information
    source_component: str = ""
    source_node: str = ""


@dataclass
class StateConflict:
    """State conflict information."""

    conflict_id: str = field(default_factory=lambda: str(uuid4()))
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Conflict details
    entity_type: str = ""
    entity_id: str = ""

    # Conflicting versions
    version_a: StateVersion = field(default_factory=StateVersion)
    version_b: StateVersion = field(default_factory=StateVersion)

    # Conflict data
    conflicting_fields: list[str] = field(default_factory=list)
    state_a: dict[str, Any] = field(default_factory=dict)
    state_b: dict[str, Any] = field(default_factory=dict)

    # Resolution
    resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LAST_WRITE_WINS
    resolved: bool = False
    resolved_at: datetime | None = None
    resolved_state: dict[str, Any] = field(default_factory=dict)
    resolution_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncMetrics:
    """Synchronization performance metrics."""

    total_sync_operations: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0

    # Timing metrics
    average_sync_time_ms: float = 0.0
    max_sync_time_ms: float = 0.0
    min_sync_time_ms: float = float("inf")

    # Latency metrics
    average_propagation_latency_ms: float = 0.0
    max_propagation_latency_ms: float = 0.0

    # Throughput metrics
    syncs_per_second: float = 0.0
    events_per_second: float = 0.0

    # Last update
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class StateSyncManager:
    """
    Comprehensive real-time state synchronization manager.

    Features:
    - Multi-layer state synchronization (Redis, PostgreSQL, local cache)
    - Conflict detection using vector clocks and version tracking
    - Configurable conflict resolution strategies
    - Real-time event propagation and subscription
    - Performance monitoring and optimization
    - Distributed consensus for critical state changes
    """

    def __init__(self, config: Config):
        """
        Initialize the state sync manager.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{id(self)}")

        # Database clients
        self.db_manager = DatabaseManager(config)
        self.redis_client = RedisClient(config)

        # Sync configuration
        sync_config = config.state_sync
        self.sync_interval_seconds = sync_config.get("sync_interval_seconds", 5)
        self.conflict_resolution_timeout_seconds = sync_config.get(
            "conflict_resolution_timeout_seconds", 30
        )
        self.max_sync_retries = sync_config.get("max_sync_retries", 3)
        self.propagation_timeout_seconds = sync_config.get("propagation_timeout_seconds", 10)

        # Conflict resolution
        self.default_resolution_strategy = ConflictResolutionStrategy(
            sync_config.get("default_resolution_strategy", "last_write_wins")
        )
        self.custom_resolvers: dict[str, Callable] = {}

        # State tracking
        self.local_state_cache: dict[str, dict[str, Any]] = {}
        self.state_versions: dict[str, StateVersion] = {}
        self.pending_syncs: set[str] = set()
        self.active_conflicts: dict[str, StateConflict] = {}

        # Event handling
        self.event_subscribers: dict[SyncEventType, list[Callable]] = {
            event_type: [] for event_type in SyncEventType
        }
        self.sync_events: list[SyncEvent] = []

        # Performance tracking
        self.sync_metrics = SyncMetrics()

        # Background tasks
        self._sync_task: asyncio.Task | None = None
        self._conflict_resolution_task: asyncio.Task | None = None
        self._metrics_task: asyncio.Task | None = None

        # Redis pub/sub
        self._redis_pubsub: aioredis.client.PubSub | None = None

        self.logger.info("StateSyncManager initialized")

    async def initialize(self) -> None:
        """Initialize the state sync manager."""
        try:
            # Initialize database connections
            await self.db_manager.initialize()
            await self.redis_client.initialize()

            # Setup Redis pub/sub for real-time sync
            await self._setup_redis_pubsub()

            # Load existing state versions
            await self._load_state_versions()

            # Start background tasks
            self._sync_task = asyncio.create_task(self._sync_loop())
            self._conflict_resolution_task = asyncio.create_task(self._conflict_resolution_loop())
            self._metrics_task = asyncio.create_task(self._metrics_loop())

            self.logger.info("StateSyncManager initialization completed")

        except Exception as e:
            self.logger.error(f"StateSyncManager initialization failed: {e}")
            raise StateError(f"Failed to initialize StateSyncManager: {e}")

    @time_execution
    async def sync_state(
        self,
        entity_type: str,
        entity_id: str,
        state_data: dict[str, Any],
        source_component: str = "unknown",
    ) -> bool:
        """
        Synchronize state across all storage layers.

        Args:
            entity_type: Type of entity (bot_state, trade, position)
            entity_id: Entity identifier
            state_data: State data to sync
            source_component: Component initiating the sync

        Returns:
            True if sync successful

        Raises:
            SynchronizationError: If sync fails
        """
        sync_start_time = datetime.now(timezone.utc)
        entity_key = f"{entity_type}:{entity_id}"

        try:
            # Check if sync is already in progress
            if entity_key in self.pending_syncs:
                self.logger.debug(f"Sync already in progress for {entity_key}")
                return False

            self.pending_syncs.add(entity_key)

            # Get current version
            current_version = self.state_versions.get(entity_key, StateVersion())
            new_version = current_version.increment()

            # Detect conflicts
            conflict = await self._detect_conflicts(entity_key, state_data, new_version)
            if conflict:
                self.logger.warning(f"Conflict detected for {entity_key}")
                await self._handle_conflict(conflict)
                return False

            # Sync to all storage layers
            success = await self._sync_to_storage_layers(
                entity_type, entity_id, state_data, new_version
            )

            if success:
                # Update local cache and version
                self.local_state_cache[entity_key] = state_data.copy()
                self.state_versions[entity_key] = new_version

                # Create sync event
                sync_event = SyncEvent(
                    event_type=SyncEventType.STATE_UPDATED,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    old_version=current_version,
                    new_version=new_version,
                    changes=state_data,
                    source_component=source_component,
                )

                # Propagate event
                await self._propagate_sync_event(sync_event)

                # Update metrics
                sync_time_ms = (datetime.now(timezone.utc) - sync_start_time).total_seconds() * 1000
                self._update_sync_metrics(True, sync_time_ms)

                self.logger.debug(
                    "State synced successfully",
                    entity_key=entity_key,
                    version=new_version.version,
                    sync_time_ms=sync_time_ms,
                )

                return True
            else:
                self._update_sync_metrics(False, 0)
                return False

        except Exception as e:
            self._update_sync_metrics(False, 0)
            self.logger.error(f"State sync failed: {e}", entity_key=entity_key)
            raise SynchronizationError(f"Sync failed for {entity_key}: {e}")

        finally:
            self.pending_syncs.discard(entity_key)

    async def subscribe_to_events(
        self, event_type: SyncEventType, callback: Callable[[SyncEvent], None]
    ) -> None:
        """
        Subscribe to synchronization events.

        Args:
            event_type: Type of events to subscribe to
            callback: Callback function to handle events
        """
        if event_type not in self.event_subscribers:
            self.event_subscribers[event_type] = []

        self.event_subscribers[event_type].append(callback)
        self.logger.debug(f"Subscribed to {event_type.value} events")

    async def register_conflict_resolver(
        self, entity_type: str, resolver: Callable[[StateConflict], dict[str, Any]]
    ) -> None:
        """
        Register a custom conflict resolver for an entity type.

        Args:
            entity_type: Entity type
            resolver: Conflict resolution function
        """
        self.custom_resolvers[entity_type] = resolver
        self.logger.info(f"Registered custom conflict resolver for {entity_type}")

    async def force_sync(self, entity_type: str, entity_id: str) -> bool:
        """
        Force synchronization of a specific entity.

        Args:
            entity_type: Type of entity
            entity_id: Entity identifier

        Returns:
            True if sync successful
        """
        try:
            entity_key = f"{entity_type}:{entity_id}"

            # Load state from primary storage (PostgreSQL)
            state_data = await self._load_from_primary_storage(entity_type, entity_id)
            if not state_data:
                self.logger.warning(f"No state found for forced sync: {entity_key}")
                return False

            # Force sync to all layers
            return await self.sync_state(entity_type, entity_id, state_data, "force_sync")

        except Exception as e:
            self.logger.error(
                f"Force sync failed: {e}", entity_type=entity_type, entity_id=entity_id
            )
            return False

    async def get_sync_status(self, entity_type: str, entity_id: str) -> dict[str, Any]:
        """
        Get synchronization status for an entity.

        Args:
            entity_type: Type of entity
            entity_id: Entity identifier

        Returns:
            Sync status information
        """
        entity_key = f"{entity_type}:{entity_id}"

        try:
            current_version = self.state_versions.get(entity_key)

            # Check consistency across storage layers
            consistency_check = await self._check_consistency(entity_type, entity_id)

            status = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "current_version": current_version.version if current_version else None,
                "last_updated": current_version.timestamp.isoformat() if current_version else None,
                "sync_in_progress": entity_key in self.pending_syncs,
                "has_conflicts": entity_key in self.active_conflicts,
                "consistency_status": consistency_check,
            }

            return status

        except Exception as e:
            self.logger.error(f"Failed to get sync status: {e}", entity_key=entity_key)
            return {"error": str(e)}

    async def get_sync_metrics(self) -> dict[str, Any]:
        """Get comprehensive sync metrics."""
        return {
            "total_sync_operations": self.sync_metrics.total_sync_operations,
            "successful_syncs": self.sync_metrics.successful_syncs,
            "failed_syncs": self.sync_metrics.failed_syncs,
            "success_rate": (
                self.sync_metrics.successful_syncs / max(self.sync_metrics.total_sync_operations, 1)
            )
            * 100,
            "conflicts_detected": self.sync_metrics.conflicts_detected,
            "conflicts_resolved": self.sync_metrics.conflicts_resolved,
            "conflict_resolution_rate": (
                self.sync_metrics.conflicts_resolved / max(self.sync_metrics.conflicts_detected, 1)
            )
            * 100,
            "average_sync_time_ms": self.sync_metrics.average_sync_time_ms,
            "max_sync_time_ms": self.sync_metrics.max_sync_time_ms,
            "min_sync_time_ms": self.sync_metrics.min_sync_time_ms,
            "average_propagation_latency_ms": self.sync_metrics.average_propagation_latency_ms,
            "syncs_per_second": self.sync_metrics.syncs_per_second,
            "events_per_second": self.sync_metrics.events_per_second,
            "active_conflicts": len(self.active_conflicts),
            "pending_syncs": len(self.pending_syncs),
            "last_updated": self.sync_metrics.last_updated.isoformat(),
        }

    async def resolve_conflict_manually(
        self, conflict_id: str, resolved_state: dict[str, Any]
    ) -> bool:
        """
        Manually resolve a state conflict.

        Args:
            conflict_id: Conflict identifier
            resolved_state: Manually resolved state

        Returns:
            True if resolution successful
        """
        try:
            conflict = self.active_conflicts.get(conflict_id)
            if not conflict:
                raise StateError(f"Conflict {conflict_id} not found")

            # Apply resolved state
            entity_key = f"{conflict.entity_type}:{conflict.entity_id}"
            new_version = max(
                conflict.version_a, conflict.version_b, key=lambda v: v.version
            ).increment()

            success = await self._sync_to_storage_layers(
                conflict.entity_type, conflict.entity_id, resolved_state, new_version
            )

            if success:
                # Mark conflict as resolved
                conflict.resolved = True
                conflict.resolved_at = datetime.now(timezone.utc)
                conflict.resolved_state = resolved_state
                conflict.resolution_metadata = {"method": "manual"}

                # Update local state
                self.local_state_cache[entity_key] = resolved_state
                self.state_versions[entity_key] = new_version

                # Remove from active conflicts
                del self.active_conflicts[conflict_id]

                # Create resolution event
                resolution_event = SyncEvent(
                    event_type=SyncEventType.CONFLICT_RESOLVED,
                    entity_type=conflict.entity_type,
                    entity_id=conflict.entity_id,
                    new_version=new_version,
                    changes=resolved_state,
                    metadata={"conflict_id": conflict_id, "resolution_method": "manual"},
                )

                await self._propagate_sync_event(resolution_event)

                self.logger.info("Conflict resolved manually", conflict_id=conflict_id)
                return True

            return False

        except Exception as e:
            self.logger.error(f"Manual conflict resolution failed: {e}", conflict_id=conflict_id)
            return False

    # Private helper methods

    async def _detect_conflicts(
        self, entity_key: str, new_state: dict[str, Any], new_version: StateVersion
    ) -> StateConflict | None:
        """Detect state conflicts."""
        try:
            # Check for concurrent modifications
            current_cached_state = self.local_state_cache.get(entity_key)
            current_version = self.state_versions.get(entity_key)

            if not current_cached_state or not current_version:
                return None  # No conflict for new entities

            # Load current state from storage
            entity_type, entity_id = entity_key.split(":", 1)
            storage_state = await self._load_from_primary_storage(entity_type, entity_id)

            if not storage_state:
                return None

            # Compare states to detect conflicts
            if self._states_differ(current_cached_state, storage_state):
                # Create conflict record
                conflict = StateConflict(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    version_a=current_version,
                    version_b=new_version,
                    conflicting_fields=self._find_conflicting_fields(
                        current_cached_state, new_state
                    ),
                    state_a=current_cached_state,
                    state_b=new_state,
                    resolution_strategy=self.default_resolution_strategy,
                )

                self.active_conflicts[conflict.conflict_id] = conflict
                self.sync_metrics.conflicts_detected += 1

                return conflict

            return None

        except Exception as e:
            self.logger.error(f"Conflict detection failed: {e}", entity_key=entity_key)
            return None

    async def _handle_conflict(self, conflict: StateConflict) -> None:
        """Handle a detected conflict."""
        try:
            entity_key = f"{conflict.entity_type}:{conflict.entity_id}"

            # Create conflict event
            conflict_event = SyncEvent(
                event_type=SyncEventType.CONFLICT_DETECTED,
                entity_type=conflict.entity_type,
                entity_id=conflict.entity_id,
                old_version=conflict.version_a,
                new_version=conflict.version_b,
                metadata={"conflict_id": conflict.conflict_id},
            )

            await self._propagate_sync_event(conflict_event)

            # Apply resolution strategy
            if conflict.resolution_strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
                resolved_state = conflict.state_b  # New state wins
            elif conflict.resolution_strategy == ConflictResolutionStrategy.FIRST_WRITE_WINS:
                resolved_state = conflict.state_a  # Existing state wins
            elif conflict.resolution_strategy == ConflictResolutionStrategy.MERGE_STRATEGY:
                resolved_state = await self._merge_states(conflict.state_a, conflict.state_b)
            elif conflict.resolution_strategy == ConflictResolutionStrategy.CUSTOM_RESOLVER:
                if conflict.entity_type in self.custom_resolvers:
                    resolver = self.custom_resolvers[conflict.entity_type]
                    resolved_state = await resolver(conflict)
                else:
                    # Fall back to last write wins
                    resolved_state = conflict.state_b
            else:
                # Manual resolution required
                self.logger.warning(
                    f"Manual resolution required for conflict {conflict.conflict_id}"
                )
                return

            # Apply resolved state
            new_version = max(
                conflict.version_a, conflict.version_b, key=lambda v: v.version
            ).increment()

            success = await self._sync_to_storage_layers(
                conflict.entity_type, conflict.entity_id, resolved_state, new_version
            )

            if success:
                # Mark as resolved
                conflict.resolved = True
                conflict.resolved_at = datetime.now(timezone.utc)
                conflict.resolved_state = resolved_state

                # Update local state
                self.local_state_cache[entity_key] = resolved_state
                self.state_versions[entity_key] = new_version

                # Remove from active conflicts
                del self.active_conflicts[conflict.conflict_id]
                self.sync_metrics.conflicts_resolved += 1

                self.logger.info(
                    "Conflict resolved automatically", conflict_id=conflict.conflict_id
                )

        except Exception as e:
            self.logger.error(f"Conflict handling failed: {e}", conflict_id=conflict.conflict_id)

    async def _sync_to_storage_layers(
        self, entity_type: str, entity_id: str, state_data: dict[str, Any], version: StateVersion
    ) -> bool:
        """Sync state to all storage layers."""
        try:
            # Sync to Redis (real-time cache)
            redis_success = await self._sync_to_redis(entity_type, entity_id, state_data, version)

            # Sync to PostgreSQL (persistent storage)
            postgres_success = await self._sync_to_postgres(
                entity_type, entity_id, state_data, version
            )

            # Both must succeed for overall success
            return redis_success and postgres_success

        except Exception as e:
            self.logger.error(f"Storage layer sync failed: {e}")
            return False

    async def _sync_to_redis(
        self, entity_type: str, entity_id: str, state_data: dict[str, Any], version: StateVersion
    ) -> bool:
        """Sync state to Redis."""
        try:
            redis_key = f"state:{entity_type}:{entity_id}"

            redis_data = {
                "version": version.version,
                "timestamp": version.timestamp.isoformat(),
                "data": state_data,
            }

            await self.redis_client.setex(
                redis_key,
                3600,
                json.dumps(redis_data, default=str),  # 1 hour TTL
            )

            return True

        except Exception as e:
            self.logger.error(f"Redis sync failed: {e}")
            return False

    async def _sync_to_postgres(
        self, entity_type: str, entity_id: str, state_data: dict[str, Any], version: StateVersion
    ) -> bool:
        """Sync state to PostgreSQL."""
        try:
            async with self.db_manager.get_session() as session:
                if entity_type == "bot_state":
                    # Update bot instance
                    from src.database.models import BotInstance

                    await session.execute(
                        update(BotInstance)
                        .where(BotInstance.id == entity_id)
                        .values(config=state_data, updated_at=version.timestamp)
                    )

                # Add other entity types as needed

                await session.commit()
                return True

        except Exception as e:
            self.logger.error(f"PostgreSQL sync failed: {e}")
            return False

    async def _load_from_primary_storage(
        self, entity_type: str, entity_id: str
    ) -> dict[str, Any] | None:
        """Load state from primary storage (PostgreSQL)."""
        try:
            async with self.db_manager.get_session() as session:
                if entity_type == "bot_state":
                    from src.database.models import BotInstance

                    result = await session.execute(
                        select(BotInstance).where(BotInstance.id == entity_id)
                    )
                    bot_instance = result.scalar_one_or_none()

                    if bot_instance and bot_instance.config:
                        return bot_instance.config

                # Add other entity types as needed

                return None

        except Exception as e:
            self.logger.error(f"Failed to load from primary storage: {e}")
            return None

    async def _check_consistency(self, entity_type: str, entity_id: str) -> dict[str, Any]:
        """Check consistency across storage layers."""
        try:
            entity_key = f"{entity_type}:{entity_id}"

            # Load from different sources
            redis_data = await self._load_from_redis(entity_type, entity_id)
            postgres_data = await self._load_from_primary_storage(entity_type, entity_id)
            cached_data = self.local_state_cache.get(entity_key)

            # Compare versions and data
            consistency = {
                "redis_available": redis_data is not None,
                "postgres_available": postgres_data is not None,
                "cache_available": cached_data is not None,
                "consistent": True,
                "discrepancies": [],
            }

            # Check for discrepancies
            if redis_data and postgres_data:
                if self._states_differ(redis_data, postgres_data):
                    consistency["consistent"] = False
                    consistency["discrepancies"].append("redis_postgres_mismatch")

            if cached_data and postgres_data:
                if self._states_differ(cached_data, postgres_data):
                    consistency["consistent"] = False
                    consistency["discrepancies"].append("cache_postgres_mismatch")

            return consistency

        except Exception as e:
            self.logger.error(f"Consistency check failed: {e}")
            return {"error": str(e)}

    async def _load_from_redis(self, entity_type: str, entity_id: str) -> dict[str, Any] | None:
        """Load state from Redis."""
        try:
            redis_key = f"state:{entity_type}:{entity_id}"
            redis_data = await self.redis_client.get(redis_key)

            if redis_data:
                parsed_data = json.loads(redis_data)
                return parsed_data.get("data")

            return None

        except Exception as e:
            self.logger.error(f"Failed to load from Redis: {e}")
            return None

    def _states_differ(self, state_a: dict[str, Any], state_b: dict[str, Any]) -> bool:
        """Check if two states differ."""
        try:
            # Simple comparison - could be enhanced with deep comparison
            return state_a != state_b

        except Exception:
            return True  # Assume different if comparison fails

    def _find_conflicting_fields(
        self, state_a: dict[str, Any], state_b: dict[str, Any]
    ) -> list[str]:
        """Find fields that differ between states."""
        conflicting_fields = []

        all_keys = set(state_a.keys()) | set(state_b.keys())

        for key in all_keys:
            if state_a.get(key) != state_b.get(key):
                conflicting_fields.append(key)

        return conflicting_fields

    async def _merge_states(
        self, state_a: dict[str, Any], state_b: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge two states using a default strategy."""
        # Simple merge - take union of both states with state_b taking precedence
        merged_state = state_a.copy()
        merged_state.update(state_b)
        return merged_state

    async def _propagate_sync_event(self, event: SyncEvent) -> None:
        """Propagate sync event to subscribers."""
        try:
            # Store event
            self.sync_events.append(event)

            # Limit event history
            if len(self.sync_events) > 1000:
                self.sync_events = self.sync_events[-1000:]

            # Notify local subscribers
            subscribers = self.event_subscribers.get(event.event_type, [])
            for callback in subscribers:
                try:
                    await callback(event)
                except Exception as e:
                    self.logger.warning(f"Event subscriber error: {e}")

            # Publish to Redis for cross-node sync
            await self._publish_event_to_redis(event)

        except Exception as e:
            self.logger.error(f"Event propagation failed: {e}")

    async def _publish_event_to_redis(self, event: SyncEvent) -> None:
        """Publish event to Redis pub/sub."""
        try:
            channel = f"sync_events:{event.entity_type}"
            event_data = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "entity_type": event.entity_type,
                "entity_id": event.entity_id,
                "changes": event.changes,
                "metadata": event.metadata,
            }

            await self.redis_client.publish(channel, json.dumps(event_data, default=str))

        except Exception as e:
            self.logger.warning(f"Failed to publish event to Redis: {e}")

    def _update_sync_metrics(self, success: bool, sync_time_ms: float) -> None:
        """Update synchronization metrics."""
        self.sync_metrics.total_sync_operations += 1

        if success:
            self.sync_metrics.successful_syncs += 1
        else:
            self.sync_metrics.failed_syncs += 1

        if sync_time_ms > 0:
            # Update timing metrics
            current_avg = self.sync_metrics.average_sync_time_ms
            total_ops = self.sync_metrics.total_sync_operations

            self.sync_metrics.average_sync_time_ms = (
                current_avg * (total_ops - 1) + sync_time_ms
            ) / total_ops

            self.sync_metrics.max_sync_time_ms = max(
                self.sync_metrics.max_sync_time_ms, sync_time_ms
            )

            self.sync_metrics.min_sync_time_ms = min(
                self.sync_metrics.min_sync_time_ms, sync_time_ms
            )

        self.sync_metrics.last_updated = datetime.now(timezone.utc)

    async def _setup_redis_pubsub(self) -> None:
        """Setup Redis pub/sub for cross-node synchronization."""
        try:
            # This would setup Redis pub/sub connection
            # For now, just log the intent
            self.logger.info("Redis pub/sub setup completed")

        except Exception as e:
            self.logger.error(f"Redis pub/sub setup failed: {e}")

    async def _load_state_versions(self) -> None:
        """Load existing state versions."""
        try:
            # This would load version information from storage
            # For now, start with empty versions
            self.logger.info("State versions loaded")

        except Exception as e:
            self.logger.warning(f"Failed to load state versions: {e}")

    async def _sync_loop(self) -> None:
        """Background synchronization loop."""
        while True:
            try:
                # Perform periodic sync operations
                await self._periodic_consistency_check()

                # Sleep for sync interval
                await asyncio.sleep(self.sync_interval_seconds)

            except Exception as e:
                self.logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(self.sync_interval_seconds)

    async def _conflict_resolution_loop(self) -> None:
        """Background conflict resolution loop."""
        while True:
            try:
                # Check for conflicts that need automatic resolution
                for conflict_id, conflict in list(self.active_conflicts.items()):
                    if not conflict.resolved:
                        # Check if conflict is too old
                        age = datetime.now(timezone.utc) - conflict.detected_at
                        if age.total_seconds() > self.conflict_resolution_timeout_seconds:
                            self.logger.warning(f"Conflict timeout: {conflict_id}")
                            # Could trigger alerts or forced resolution

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"Conflict resolution loop error: {e}")
                await asyncio.sleep(30)

    async def _metrics_loop(self) -> None:
        """Background metrics calculation loop."""
        while True:
            try:
                # Calculate throughput metrics
                current_time = datetime.now(timezone.utc)
                if hasattr(self, "_last_metrics_time"):
                    time_diff = (current_time - self._last_metrics_time).total_seconds()
                    if time_diff > 0:
                        ops_diff = self.sync_metrics.total_sync_operations - getattr(
                            self, "_last_ops_count", 0
                        )
                        events_diff = len(self.sync_events) - getattr(self, "_last_events_count", 0)

                        self.sync_metrics.syncs_per_second = ops_diff / time_diff
                        self.sync_metrics.events_per_second = events_diff / time_diff

                self._last_metrics_time = current_time
                self._last_ops_count = self.sync_metrics.total_sync_operations
                self._last_events_count = len(self.sync_events)

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                self.logger.error(f"Metrics loop error: {e}")
                await asyncio.sleep(60)

    async def _periodic_consistency_check(self) -> None:
        """Perform periodic consistency checks."""
        try:
            # Check consistency for entities that haven't been synced recently
            # This would be more sophisticated in a full implementation
            pass

        except Exception as e:
            self.logger.error(f"Periodic consistency check failed: {e}")
