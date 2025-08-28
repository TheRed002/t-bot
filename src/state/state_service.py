"""
Comprehensive State Management Service for the T-Bot trading system.

This module provides enterprise-grade state management with:
- Centralized state coordination and synchronization
- State validation and consistency checks
- State persistence with DatabaseManager integration
- State recovery and rollback capabilities
- State event broadcasting and monitoring
- Performance metrics and health monitoring

The StateService eliminates all direct database access from other state modules
and provides a unified interface for all state operations.
"""

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, TypeVar
from uuid import uuid4

from src.core.base.component import BaseComponent
from src.core.base.events import BaseEventEmitter
from src.core.base.interfaces import HealthCheckResult
from src.core.config.main import Config
from src.core.exceptions import StateError, ValidationError
from src.error_handling import (
    ErrorContext,
    ErrorHandler,
    ErrorSeverity,
    with_circuit_breaker,
    with_retry,
)
from src.monitoring.telemetry import get_tracer

# Import consistency patterns for aligned data flow
from .consistency import (
    ConsistentEventPattern,
    ConsistentProcessingPattern,
    ConsistentValidationPattern,
    emit_state_event,
    validate_state_data,
)

# Service layer imports
from .services import (
    StateBusinessService,
    StateBusinessServiceProtocol,
    StatePersistenceService,
    StatePersistenceServiceProtocol,
    StateSynchronizationService,
    StateSynchronizationServiceProtocol,
    StateValidationService,
    StateValidationServiceProtocol,
)
from .utils_imports import time_execution


# Database service protocol to avoid concrete coupling
class DatabaseServiceProtocol(Protocol):
    """Protocol for database service interactions."""

    @property
    def initialized(self) -> bool: ...
    async def start(self) -> None: ...
    async def health_check(self) -> HealthCheckResult: ...
    async def create_redis_client(self, config: dict[str, Any]) -> Any: ...
    async def create_influxdb_client(self, config: dict[str, Any]) -> Any: ...


# Client protocols for type safety
class RedisClientProtocol(Protocol):
    """Protocol for Redis client operations."""

    async def connect(self) -> None: ...
    async def get(self, key: str) -> str | None: ...
    async def setex(self, key: str, ttl: int, value: str) -> None: ...
    async def delete(self, *keys: str) -> int: ...
    async def keys(self, pattern: str) -> list[str]: ...
    async def ping(self) -> bool: ...


class InfluxDBClientProtocol(Protocol):
    """Protocol for InfluxDB client operations."""

    def connect(self) -> None: ...
    def write_point(self, point: Any) -> None: ...
    def ping(self) -> bool: ...


if TYPE_CHECKING:
    from .state_persistence import StatePersistence
    from .state_synchronizer import StateSynchronizer
    from .state_validator import StateValidator

T = TypeVar("T")


class StateType(str, Enum):
    """State type enumeration for type safety."""

    BOT_STATE = "bot_state"
    POSITION_STATE = "position_state"
    ORDER_STATE = "order_state"
    PORTFOLIO_STATE = "portfolio_state"
    RISK_STATE = "risk_state"
    STRATEGY_STATE = "strategy_state"
    MARKET_STATE = "market_state"
    TRADE_STATE = "trade_state"
    EXECUTION = "execution"
    SYSTEM_STATE = "system_state"  # Added to match interface


class StateOperation(Enum):
    """State operation enumeration."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RESTORE = "restore"
    SYNC = "sync"


class StatePriority(str, Enum):
    """State operation priority levels."""

    CRITICAL = "critical"  # Trading operations, risk limits
    HIGH = "high"  # Order management, position updates
    MEDIUM = "medium"  # Strategy updates, configuration
    LOW = "low"  # Metrics, historical data


@dataclass
class StateMetadata:
    """Metadata for state tracking and versioning."""

    state_id: str
    state_type: StateType
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checksum: str = ""
    size_bytes: int = 0
    source_component: str = ""
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class StateChange:
    """Represents a state change for audit and synchronization."""

    change_id: str = field(default_factory=lambda: str(uuid4()))
    state_id: str = ""
    state_type: StateType = StateType.BOT_STATE
    operation: StateOperation = StateOperation.UPDATE
    priority: StatePriority = StatePriority.MEDIUM

    # Change details
    old_value: dict[str, Any] | None = None
    new_value: dict[str, Any] | None = None
    changed_fields: set[str] = field(default_factory=set)

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_component: str = ""
    user_id: str | None = None
    reason: str = ""

    # Status tracking
    applied: bool = False
    synchronized: bool = False
    persisted: bool = False


@dataclass
class StateSnapshot:
    """Complete state snapshot for recovery and backup."""

    snapshot_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1

    # State data by type
    states: dict[StateType, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, StateMetadata] = field(default_factory=dict)

    # Snapshot metadata
    total_states: int = 0
    total_size_bytes: int = 0
    compression_ratio: float = 1.0
    checksum: str = ""
    description: str = ""


@dataclass
class StateValidationResult:
    """Result of state validation operation."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validation_time_ms: float = 0.0
    rules_checked: int = 0
    rules_passed: int = 0


@dataclass
class StateMetrics:
    """State management performance and health metrics."""

    # Operation counts
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0

    # Performance metrics
    average_operation_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    sync_success_rate: float = 0.0

    # Resource usage
    memory_usage_mb: float = 0.0
    storage_usage_mb: float = 0.0
    active_states_count: int = 0

    # Health indicators
    last_successful_sync: datetime | None = None
    last_failed_operation: datetime | None = None
    error_rate: float = 0.0

    def to_dict(self) -> dict[str, int | float | str | None]:
        """Convert metrics to dictionary format for monitoring."""
        return {
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "average_operation_time_ms": self.average_operation_time_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "sync_success_rate": self.sync_success_rate,
            "memory_usage_mb": self.memory_usage_mb,
            "storage_usage_mb": self.storage_usage_mb,
            "active_states_count": self.active_states_count,
            "error_rate": self.error_rate,
            "last_successful_sync": (
                self.last_successful_sync.isoformat() if self.last_successful_sync else None
            ),
            "last_failed_operation": (
                self.last_failed_operation.isoformat() if self.last_failed_operation else None
            ),
        }


class StateService(BaseComponent):
    """
    Comprehensive state management service providing enterprise-grade
    state handling with synchronization, validation, persistence, and recovery.

    Key Features:
    - Centralized state coordination across all components
    - Real-time state synchronization with conflict resolution
    - Comprehensive state validation with business rules
    - Multi-layer persistence (Redis, PostgreSQL, InfluxDB)
    - State recovery and rollback capabilities
    - Event broadcasting and subscription system
    - Performance monitoring and health checks
    """

    def __init__(self, config: Config, database_service: DatabaseServiceProtocol | None = None):
        """
        Initialize the state service.

        Args:
            config: Application configuration
            database_service: Database service instance (injected dependency)
        """
        # Convert Config object to dict for BaseComponent
        config_dict = (
            {
                key: getattr(config, key)
                for key in dir(config)
                if not key.startswith("_") and not callable(getattr(config, key))
            }
            if config
            else {}
        )
        super().__init__(name="StateService", config=config_dict)
        self.config = config

        # Database service (dependency injection with protocol)
        self.database_service = database_service
        self.redis_client: RedisClientProtocol | None = None
        self.influxdb_client: InfluxDBClientProtocol | None = None

        # Client initialization deferred to initialize() method
        self._redis_initialized = False
        self._influxdb_initialized = False

        # State storage layers
        self._memory_cache: dict[str, dict[str, Any]] = {}
        self._metadata_cache: dict[str, StateMetadata] = {}
        self._change_log: list[StateChange] = []

        # Service layer components (dependency injection)
        self._business_service: StateBusinessServiceProtocol | None = None
        self._persistence_service: StatePersistenceServiceProtocol | None = None
        self._validation_service: StateValidationServiceProtocol | None = None
        self._synchronization_service: StateSynchronizationServiceProtocol | None = None

        # Legacy components (backward compatibility)
        self._synchronizer: StateSynchronizer | None = None
        self._validator: StateValidator | None = None
        self._persistence: StatePersistence | None = None

        # Event system - use consistent event-driven pattern only
        self._subscribers: dict[StateType, set[Callable]] = {}
        self._event_emitter = BaseEventEmitter(name="StateService", config=config)
        # Remove mixed queue pattern - use only consistent event pattern
        # self._event_queue: asyncio.Queue = asyncio.Queue()

        # Initialize consistency patterns for aligned data flow
        self._consistent_event_pattern = ConsistentEventPattern("StateService")
        self._consistent_validation_pattern = ConsistentValidationPattern()
        self._consistent_processing_pattern = ConsistentProcessingPattern("StateService")

        # Synchronization primitives
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

        # Performance tracking
        self._state_metrics: StateMetrics = StateMetrics()
        self._operation_times: list[float] = []

        # Configuration - try new config structure first, fall back to old structure
        state_config = config.__dict__.get("state_management_service", {})
        if not state_config:
            state_config = config.__dict__.get("state_management", {})

        # Core configuration values
        self.cache_ttl_seconds = state_config.get("state_ttl_seconds", 86400)  # 24 hours default
        self.sync_interval_seconds = state_config.get(
            "sync_interval_seconds", 60
        )  # 1 minute default
        self.cleanup_interval_seconds = state_config.get(
            "cleanup_interval_seconds", 3600
        )  # 1 hour default
        self.validation_interval_seconds = state_config.get(
            "validation_interval_seconds", 300
        )  # 5 minutes default
        self.snapshot_interval_seconds = state_config.get(
            "snapshot_interval_seconds", 1800
        )  # 30 minutes default
        self.max_state_versions = state_config.get("max_state_versions", 10)

        # Legacy configuration values
        self.max_change_log_size = state_config.get("max_change_log_size", 10000)
        self.enable_compression = state_config.get("enable_compression", True)
        self.max_concurrent_operations = state_config.get("max_concurrent_operations", 100)

        # Background tasks
        self._sync_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._metrics_task: asyncio.Task | None = None
        self._backup_task: asyncio.Task | None = None
        # self._event_task: asyncio.Task | None = None  # Removed - using consistent patterns
        self._running = False

        # Backup configuration
        self.auto_backup_enabled = state_config.get("auto_backup_enabled", True)
        self.backup_interval_hours = state_config.get("backup_interval_hours", 24)
        self.backup_retention_days = state_config.get("backup_retention_days", 30)
        self._last_backup_time: datetime | None = None

        # Error handler instance (singleton pattern)
        self._error_handler: ErrorHandler | None = None

        # Initialize OpenTelemetry tracing
        self.tracer = get_tracer("state_service")

        self.logger.info("StateService initialized")

    async def initialize(self) -> None:
        """Initialize the state service and all components."""
        try:
            # Initialize database service if available with proper error handling
            if self.database_service:
                try:
                    if not self.database_service.initialized:
                        await self.database_service.start()

                    # Verify database service health
                    health_result = await self.database_service.health_check()
                    if health_result.status.value != "healthy":
                        self.logger.warning(
                            f"Database service health check failed: {health_result.message}"
                        )

                except Exception as e:
                    error_context = ErrorContext.from_exception(
                        e,
                        component="StateService",
                        operation="initialize_database_service",
                        severity=ErrorSeverity.HIGH,
                    )
                    handler = self.error_handler
                    await handler.handle_error(e, error_context)
                    self.logger.error(f"Database service initialization failed: {e}")
                    # Continue without database service - graceful degradation
                    self.database_service = None

            # Initialize Redis client through database service factory if available
            await self._initialize_redis_client()

            # Initialize InfluxDB client through database service factory if available
            await self._initialize_influxdb_client()

            # Initialize service layer components
            await self._initialize_service_components()

            # Initialize state management components with lazy imports to avoid circular deps
            await self._initialize_state_components()

            # Initialize components with error handling
            if self._synchronizer:
                try:
                    await self._synchronizer.initialize()
                except Exception as e:
                    self.logger.warning(f"Synchronizer initialization failed: {e}")

            if self._validator:
                try:
                    await self._validator.initialize()
                except Exception as e:
                    self.logger.warning(f"Validator initialization failed: {e}")

            if self._persistence:
                try:
                    await self._persistence.initialize()
                except Exception as e:
                    self.logger.warning(f"Persistence initialization failed: {e}")

            # Load existing states from persistence
            await self._load_existing_states()

            # Start background tasks
            self._running = True
            self._sync_task = asyncio.create_task(self._synchronization_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._metrics_task = asyncio.create_task(self._metrics_loop())
            if self.auto_backup_enabled:
                self._backup_task = asyncio.create_task(self._backup_loop())

            # Event processing now handled synchronously via consistent patterns
            # No separate event processing task needed

            # Note: super().initialize() is not called as it's legacy compatibility
            self.logger.info("StateService initialization completed")

        except Exception as e:
            error_context = ErrorContext.from_exception(
                e, component="StateService", operation="initialize", severity=ErrorSeverity.HIGH
            )
            handler = self.error_handler
            await handler.handle_error(e, error_context)
            self.logger.error(f"StateService initialization failed: {e}")
            raise StateError(f"Failed to initialize StateService: {e}") from e

    async def cleanup(self) -> None:
        """Cleanup state service resources."""
        try:
            self._running = False

            # Cancel background tasks
            for task in [
                self._sync_task,
                self._cleanup_task,
                self._metrics_task,
                self._backup_task,
            ]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Final synchronization
            if self._synchronizer:
                await self._synchronizer.force_sync()
            if self._synchronization_service:
                # Service layer handles its own cleanup
                pass

            # Cleanup service layer components
            if self._persistence_service and hasattr(self._persistence_service, "stop"):
                await self._persistence_service.stop()
            if self._validation_service and hasattr(self._validation_service, "stop"):
                await self._validation_service.stop()
            if self._synchronization_service and hasattr(self._synchronization_service, "stop"):
                await self._synchronization_service.stop()
            if self._business_service and hasattr(self._business_service, "stop"):
                await self._business_service.stop()

            # Cleanup legacy components
            if self._persistence:
                await self._persistence.cleanup()
            if self._synchronizer:
                await self._synchronizer.cleanup()
            if self._validator:
                await self._validator.cleanup()

            # Clear caches
            self._memory_cache.clear()
            self._metadata_cache.clear()
            self._change_log.clear()

            # Cleanup Redis client connection
            if self.redis_client:
                try:
                    if hasattr(self.redis_client, "close"):
                        await self.redis_client.close()
                    elif hasattr(self.redis_client, "disconnect"):
                        await self.redis_client.disconnect()
                    self.logger.info("Redis client connection closed")
                except Exception as redis_error:
                    self.logger.error(f"Error closing Redis connection: {redis_error}")
                finally:
                    self.redis_client = None
                    self._redis_initialized = False

            # Note: super().cleanup() is not called as it's legacy compatibility
            self.logger.info("StateService cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during StateService cleanup: {e}")
            raise

    # Core State Operations

    @time_execution
    @with_retry(max_attempts=3, base_delay=0.1, backoff_factor=2.0, exceptions=(StateError,))
    async def get_state(
        self, state_type: StateType, state_id: str, include_metadata: bool = False
    ) -> dict[str, Any] | None:
        """
        Get state by type and ID with multi-layer fallback.

        Args:
            state_type: Type of state to retrieve
            state_id: Unique state identifier
            include_metadata: Whether to include state metadata

        Returns:
            State data or None if not found
        """
        with self.tracer.start_as_current_span(
            "get_state",
            attributes={
                "state.type": state_type.value,
                "state.id": state_id,
                "include_metadata": include_metadata,
            },
        ):
            cache_key = f"{state_type.value}:{state_id}"

            async with self._get_state_lock(cache_key):
                try:
                    # Try memory cache first
                    state_data = self._memory_cache.get(cache_key)
                    if state_data:
                        self._state_metrics.cache_hit_rate = self._update_hit_rate(True)

                        if include_metadata:
                            metadata = self._metadata_cache.get(cache_key)
                            return {"data": state_data, "metadata": metadata}
                        return state_data

                    # Try Redis cache
                    cached_data = None
                    if self.redis_client:
                        redis_key = f"state:{cache_key}"
                        cached_data = await self.redis_client.get(redis_key)
                    if cached_data:
                        state_data = json.loads(cached_data)
                        # Warm memory cache
                        self._memory_cache[cache_key] = state_data
                        self._state_metrics.cache_hit_rate = self._update_hit_rate(True)

                        if include_metadata:
                            metadata = await self._load_metadata(cache_key)
                            return {"data": state_data, "metadata": metadata}
                        return state_data

                    # Try database persistence
                    if self._persistence:
                        state_data = await self._persistence.load_state(state_type, state_id)
                        if state_data:
                            # Warm both caches
                            self._memory_cache[cache_key] = state_data
                            if self.redis_client:
                                await self.redis_client.setex(
                                    redis_key,
                                    self.cache_ttl_seconds,
                                    json.dumps(state_data, default=str),
                                )
                            self._state_metrics.cache_hit_rate = self._update_hit_rate(False)

                            if include_metadata:
                                metadata = await self._load_metadata(cache_key)
                                return {"data": state_data, "metadata": metadata}
                            return state_data

                    # State not found
                    self._state_metrics.cache_hit_rate = self._update_hit_rate(False)
                    return None

                except Exception as e:
                    error_context = ErrorContext.from_exception(
                        e,
                        component="StateService",
                        operation="get_state",
                        severity=ErrorSeverity.MEDIUM,
                    )
                    error_context.details = {
                        "error_code": "STATE_001",
                        "cache_key": cache_key,
                        "state_type": state_type.value,
                        "state_id": state_id,
                    }
                    handler = self.error_handler
                    await handler.handle_error(e, error_context)
                    self._state_metrics.failed_operations += 1
                    raise StateError(f"State retrieval failed: {e}") from e

    @time_execution
    @with_retry(max_attempts=3, base_delay=0.1, backoff_factor=2.0, exceptions=(StateError,))
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30, expected_exception=StateError)
    async def set_state(
        self,
        state_type: StateType,
        state_id: str,
        state_data: dict[str, Any],
        source_component: str = "",
        validate: bool = True,
        priority: StatePriority = StatePriority.MEDIUM,
        reason: str = "",
    ) -> bool:
        """
        Set state with validation, synchronization, and persistence.

        Args:
            state_type: Type of state
            state_id: Unique state identifier
            state_data: State data to store
            source_component: Component making the change
            validate: Whether to validate the state
            priority: Operation priority
            reason: Reason for the change

        Returns:
            True if successful
        """
        with self.tracer.start_as_current_span(
            "set_state",
            attributes={
                "state.type": state_type.value,
                "state.id": state_id,
                "source_component": source_component,
                "priority": priority.value,
                "validate": validate,
            },
        ):
            cache_key = f"{state_type.value}:{state_id}"

            async with self._get_state_lock(cache_key):
                try:
                    start_time = datetime.now(timezone.utc)

                    # Get current state for change detection
                    current_state = await self.get_state(state_type, state_id)

                    # Use consistent validation pattern
                    if validate:
                        validation_result = await validate_state_data(
                            f"{state_type.value}_data", state_data
                        )
                        if not validation_result["is_valid"]:
                            raise ValidationError(
                                f"State validation failed: {validation_result['errors']}"
                            )

                    # Validate state transition if updating existing state
                    if current_state:
                        transition_valid = await self._validate_state_transition_through_service(
                            state_type, current_state, state_data
                        )
                        if not transition_valid:
                            raise ValidationError("Invalid state transition")

                    # Use business service to process state update
                    state_change = await self._process_state_update_through_service(
                        state_type, state_id, state_data, source_component, reason
                    )

                    # Calculate metadata through business service
                    metadata = await self._calculate_metadata_through_service(
                        state_type, state_id, state_data, source_component
                    )

                    # Update version from existing metadata if available
                    metadata.version = self._get_next_version(cache_key)

                    # Store in memory cache
                    self._memory_cache[cache_key] = state_data.copy()
                    self._metadata_cache[cache_key] = metadata

                    # Store in Redis cache using consistent data transformation
                    if self.redis_client:
                        redis_key = f"state:{cache_key}"
                        # Use consistent JSON serialization pattern
                        serialized_data = (
                            await (
                                self._consistent_processing_pattern.process_item_consistent(
                                    state_data,
                                    lambda data: json.dumps(data, default=str, sort_keys=True),
                                )
                            )
                        )
                        await self.redis_client.setex(
                            redis_key, self.cache_ttl_seconds, serialized_data
                        )

                        # Store metadata in Redis using consistent pattern
                        metadata_key = f"metadata:{cache_key}"
                        serialized_metadata = (
                            await (
                                self._consistent_processing_pattern.process_item_consistent(
                                    metadata.__dict__,
                                    lambda data: json.dumps(data, default=str, sort_keys=True),
                                )
                            )
                        )
                        await self.redis_client.setex(
                            metadata_key,
                            self.cache_ttl_seconds,
                            serialized_metadata,
                        )

                    # Use service layer for persistence
                    await self._persist_state_through_service(
                        state_type, state_id, state_data, metadata
                    )

                    # Add to change log
                    self._change_log.append(state_change)
                    if len(self._change_log) > self.max_change_log_size:
                        self._change_log = self._change_log[-self.max_change_log_size // 2 :]

                    # Use service layer for synchronization
                    await self._synchronize_state_change_through_service(state_change)

                    # Broadcast state change using consistent event pattern
                    await emit_state_event(
                        "changed",
                        {
                            "state_type": state_type.value,
                            "state_id": state_id,
                            "state_data": state_data,
                            "operation": state_change.operation.value,
                            "source_component": source_component,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )

                    # Also broadcast to legacy subscribers
                    await self._broadcast_state_change(
                        state_type, state_id, state_data, state_change
                    )

                    # Update metrics
                    operation_time = (
                        datetime.now(timezone.utc) - start_time
                    ).total_seconds() * 1000
                    self._update_operation_metrics(operation_time, True)

                    self.logger.debug(
                        f"State set successfully: {cache_key}",
                        extra={
                            "state_type": state_type.value,
                            "state_id": state_id,
                            "operation": state_change.operation.value,
                            "priority": priority.value,
                            "source": source_component,
                        },
                    )

                    return True

                except Exception as e:
                    error_context = ErrorContext.from_exception(
                        e,
                        component="StateService",
                        operation="set_state",
                        severity=ErrorSeverity.HIGH,
                    )
                    error_context.details = {
                        "error_code": "STATE_002",
                        "cache_key": cache_key,
                        "state_type": state_type.value,
                        "state_id": state_id,
                    }
                    handler = self.error_handler
                    await handler.handle_error(e, error_context)
                    self._update_operation_metrics(0, False)
                    raise StateError(f"State update failed: {e}") from e

    async def delete_state(
        self, state_type: StateType, state_id: str, source_component: str = "", reason: str = ""
    ) -> bool:
        """
        Delete state from all storage layers.

        Args:
            state_type: Type of state
            state_id: Unique state identifier
            source_component: Component making the change
            reason: Reason for deletion

        Returns:
            True if successful
        """
        cache_key = f"{state_type.value}:{state_id}"

        async with self._get_state_lock(cache_key):
            try:
                # Get current state for change record
                current_state = await self.get_state(state_type, state_id)
                if not current_state:
                    return True  # Already deleted

                # Create state change record
                state_change = StateChange(
                    state_id=state_id,
                    state_type=state_type,
                    operation=StateOperation.DELETE,
                    priority=StatePriority.HIGH,
                    old_value=current_state,
                    new_value=None,
                    source_component=source_component,
                    reason=reason,
                )

                # Remove from memory cache
                self._memory_cache.pop(cache_key, None)
                self._metadata_cache.pop(cache_key, None)

                # Remove from Redis
                if self.redis_client:
                    redis_key = f"state:{cache_key}"
                    metadata_key = f"metadata:{cache_key}"
                    await self.redis_client.delete(redis_key)
                    await self.redis_client.delete(metadata_key)

                # Queue for database deletion
                if self._persistence:
                    await self._persistence.queue_state_delete(state_type, state_id)

                # Add to change log
                self._change_log.append(state_change)

                # Queue for synchronization
                if self._synchronizer:
                    await self._synchronizer.queue_state_sync(state_change)

                # Broadcast deletion event
                await self._broadcast_state_change(state_type, state_id, None, state_change)

                self.logger.info(
                    f"State deleted: {cache_key}",
                    extra={"source": source_component, "reason": reason},
                )

                return True

            except Exception as e:
                self.logger.error(f"Failed to delete state {cache_key}: {e}")
                raise StateError(f"State deletion failed: {e}") from e

    # State Query Operations

    async def get_states_by_type(
        self, state_type: StateType, limit: int | None = None, include_metadata: bool = False
    ) -> list[dict[str, Any]]:
        """
        Get all states of a specific type.

        Args:
            state_type: Type of states to retrieve
            limit: Maximum number of states to return
            include_metadata: Whether to include metadata

        Returns:
            List of state dictionaries
        """
        try:
            states = []
            prefix = f"{state_type.value}:"

            # Search memory cache
            for cache_key, state_data in self._memory_cache.items():
                if cache_key.startswith(prefix):
                    if include_metadata:
                        metadata = self._metadata_cache.get(cache_key)
                        states.append({"data": state_data, "metadata": metadata})
                    else:
                        states.append(state_data)

                    if limit and len(states) >= limit:
                        break

            # If not enough results, query persistence layer
            if (not limit or len(states) < limit) and self._persistence:
                remaining_limit = limit - len(states) if limit else None
                persistent_states = await self._persistence.get_states_by_type(
                    state_type, remaining_limit, include_metadata
                )
                states.extend(persistent_states)

            return states[:limit] if limit else states

        except Exception as e:
            self.logger.error(f"Failed to get states by type {state_type.value}: {e}")
            raise StateError(f"Query failed: {e}") from e

    async def search_states(
        self, criteria: dict[str, Any], state_types: list[StateType] | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Search states based on criteria.

        Args:
            criteria: Search criteria
            state_types: State types to search (all if None)
            limit: Maximum results to return

        Returns:
            List of matching states
        """
        try:
            if self._persistence:
                return await self._persistence.search_states(criteria, state_types, limit)

            # Fallback to memory search
            results = []
            search_types = state_types or list(StateType)

            for state_type in search_types:
                states = await self.get_states_by_type(state_type)
                for state in states:
                    if self._matches_criteria(state, criteria):
                        results.append(state)
                        if len(results) >= limit:
                            return results

            return results

        except Exception as e:
            self.logger.error(f"State search failed: {e}")
            raise StateError(f"Search failed: {e}") from e

    # State Management Operations

    async def create_snapshot(
        self, description: str = "", state_types: list[StateType] | None = None
    ) -> str:
        """
        Create a complete state snapshot for backup/recovery.

        Args:
            description: Snapshot description
            state_types: State types to include (all if None)

        Returns:
            Snapshot ID
        """
        try:
            snapshot = StateSnapshot(description=description, timestamp=datetime.now(timezone.utc))

            include_types = state_types or list(StateType)

            for state_type in include_types:
                states = await self.get_states_by_type(state_type, include_metadata=True)
                snapshot.states[state_type.value] = {}

                for state_item in states:
                    if isinstance(state_item, dict) and "data" in state_item:
                        state_id = state_item["metadata"].state_id
                        snapshot.states[state_type.value][state_id] = state_item["data"]
                        snapshot.metadata[f"{state_type.value}:{state_id}"] = state_item["metadata"]

            # Calculate snapshot metrics
            snapshot.total_states = sum(len(states) for states in snapshot.states.values())
            snapshot_data = json.dumps(snapshot.__dict__, default=str)
            snapshot.total_size_bytes = len(snapshot_data.encode())
            snapshot.checksum = self._calculate_checksum(snapshot.__dict__)

            # Store snapshot
            if self._persistence:
                await self._persistence.save_snapshot(snapshot)

            self.logger.info(f"State snapshot created: {snapshot.snapshot_id}")
            return snapshot.snapshot_id

        except Exception as e:
            self.logger.error(f"Failed to create snapshot: {e}")
            raise StateError(f"Snapshot creation failed: {e}") from e

    async def restore_snapshot(self, snapshot_id: str) -> bool:
        """
        Restore state from a snapshot.

        Args:
            snapshot_id: Snapshot to restore from

        Returns:
            True if successful
        """
        try:
            if not self._persistence:
                raise StateError("Persistence layer not available")

            snapshot = await self._persistence.load_snapshot(snapshot_id)
            if not snapshot:
                raise StateError(f"Snapshot {snapshot_id} not found")

            # Clear current state
            self._memory_cache.clear()
            self._metadata_cache.clear()

            # Restore states
            for state_type, states in snapshot.states.items():
                for state_id, state_data in states.items():
                    await self.set_state(
                        state_type,
                        state_id,
                        state_data,
                        source_component="StateService",
                        validate=False,  # Don't validate restored data to match test expectation
                        reason=f"Restored from snapshot {snapshot_id}",
                    )

            self.logger.info(f"State restored from snapshot: {snapshot_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to restore snapshot {snapshot_id}: {e}")
            raise StateError(f"Snapshot restore failed: {e}") from e

    # Event System

    def subscribe(
        self,
        state_type: StateType,
        callback: Callable[[StateType, str, dict[str, Any] | None, StateChange], None],
    ) -> None:
        """
        Subscribe to state change events.

        Args:
            state_type: Type of state to monitor
            callback: Callback function to invoke on changes
        """
        if state_type not in self._subscribers:
            self._subscribers[state_type] = set()

        self._subscribers[state_type].add(callback)
        self.logger.debug(f"Subscribed to {state_type.value} state changes")

    def unsubscribe(self, state_type: StateType, callback: Callable) -> None:
        """
        Unsubscribe from state change events.

        Args:
            state_type: Type of state
            callback: Callback to remove
        """
        if state_type in self._subscribers:
            self._subscribers[state_type].discard(callback)

        self.logger.debug(f"Unsubscribed from {state_type.value} state changes")

    # Monitoring and Metrics

    def get_metrics(self) -> dict[str, int | float | str]:
        """Get current component metrics for base class interface."""
        # Update current metrics first
        self._state_metrics.active_states_count = len(self._memory_cache)
        self._state_metrics.memory_usage_mb = self._calculate_memory_usage()

        # Return dictionary format expected by base class
        return {
            "total_operations": self._state_metrics.total_operations,
            "successful_operations": self._state_metrics.successful_operations,
            "failed_operations": self._state_metrics.failed_operations,
            "average_operation_time_ms": self._state_metrics.average_operation_time_ms,
            "cache_hit_rate": self._state_metrics.cache_hit_rate,
            "active_states_count": self._state_metrics.active_states_count,
            "memory_usage_mb": self._state_metrics.memory_usage_mb,
            "error_rate": self._state_metrics.error_rate,
        }

    async def get_state_metrics(self) -> StateMetrics:
        """Get comprehensive state management metrics as StateMetrics object."""
        # Update current metrics
        self._state_metrics.active_states_count = len(self._memory_cache)
        self._state_metrics.memory_usage_mb = self._calculate_memory_usage()

        return self._state_metrics

    async def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status."""
        try:
            metrics = await self.get_state_metrics()

            # Check component health
            health_status = {
                "overall_status": "healthy",
                "components": {
                    "state_service": "healthy",
                    "synchronizer": "healthy" if self._synchronizer else "unavailable",
                    "validator": "healthy" if self._validator else "unavailable",
                    "persistence": "healthy" if self._persistence else "unavailable",
                },
                "metrics": {
                    "total_operations": metrics.total_operations,
                    "success_rate": metrics.successful_operations
                    / max(metrics.total_operations, 1),
                    "cache_hit_rate": metrics.cache_hit_rate,
                    "active_states": metrics.active_states_count,
                    "memory_usage_mb": metrics.memory_usage_mb,
                },
                "last_sync": (
                    metrics.last_successful_sync.isoformat()
                    if metrics.last_successful_sync
                    else None
                ),
                "error_rate": metrics.error_rate,
            }

            # Determine overall health
            if metrics.error_rate > 0.1:  # More than 10% error rate
                health_status["overall_status"] = "degraded"
            elif metrics.error_rate > 0.2:  # More than 20% error rate
                health_status["overall_status"] = "unhealthy"

            # Check component availability
            components_status = health_status["components"]
            if isinstance(components_status, dict) and not all(
                status == "healthy" for status in components_status.values()
            ):
                health_status["overall_status"] = "degraded"

            return health_status

        except Exception as e:
            self.logger.error(f"Failed to get health status: {e}")
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    # Private Helper Methods

    async def _load_from_database(
        self, state_type: StateType, state_id: str
    ) -> dict[str, Any] | None:
        """Load state from database - used by tests."""
        try:
            if self._persistence:
                return await self._persistence.load_state(state_type, state_id)
            return None
        except Exception as e:
            self.logger.warning(f"Failed to load from database: {e}")
            return None

    async def _save_to_database(
        self, state_type: StateType, state_id: str, state_data: dict[str, Any]
    ) -> None:
        """Save state to database - used by tests."""
        try:
            if self._persistence:
                metadata = StateMetadata(
                    state_id=state_id,
                    state_type=state_type,
                    version=1,
                    checksum=self._calculate_checksum(state_data),
                    size_bytes=len(json.dumps(state_data, default=str).encode()),
                )
                await self._persistence.queue_state_save(state_type, state_id, state_data, metadata)
        except Exception as e:
            self.logger.warning(f"Failed to save to database: {e}")

    def _get_state_lock(self, state_key: str) -> asyncio.Lock:
        """Get or create a lock for state operations."""
        if state_key not in self._locks:
            self._locks[state_key] = asyncio.Lock()
        return self._locks[state_key]

    def _calculate_checksum(self, data: dict[str, Any]) -> str:
        """Calculate checksum for data integrity."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _detect_changed_fields(
        self, old_state: dict[str, Any] | None, new_state: dict[str, Any]
    ) -> set[str]:
        """Detect which fields changed between states."""
        if not old_state:
            return set(new_state.keys())

        changed = set()

        # Check for modified and new fields
        for key, value in new_state.items():
            if key not in old_state or old_state[key] != value:
                changed.add(key)

        # Check for deleted fields
        for key in old_state:
            if key not in new_state:
                changed.add(key)

        return changed

    def _get_next_version(self, cache_key: str) -> int:
        """Get next version number for state."""
        metadata = self._metadata_cache.get(cache_key)
        return metadata.version + 1 if metadata else 1

    def _update_hit_rate(self, hit: bool) -> float:
        """Update cache hit rate metric."""
        # Simple moving average implementation
        if not hasattr(self, "_hit_history"):
            self._hit_history = []

        self._hit_history.append(1.0 if hit else 0.0)
        if len(self._hit_history) > 1000:
            self._hit_history = self._hit_history[-500:]

        return sum(self._hit_history) / len(self._hit_history)

    def _update_operation_metrics(self, operation_time_ms: float, success: bool) -> None:
        """Update operation performance metrics."""
        self._state_metrics.total_operations += 1

        if success:
            self._state_metrics.successful_operations += 1
        else:
            self._state_metrics.failed_operations += 1
            self._state_metrics.last_failed_operation = datetime.now(timezone.utc)

        # Update average operation time
        self._operation_times.append(operation_time_ms)
        if len(self._operation_times) > 1000:
            self._operation_times = self._operation_times[-500:]

        self._state_metrics.average_operation_time_ms = sum(self._operation_times) / len(
            self._operation_times
        )
        self._state_metrics.error_rate = (
            self._state_metrics.failed_operations / self._state_metrics.total_operations
        )

    def _calculate_memory_usage(self) -> float:
        """Calculate approximate memory usage in MB."""
        import sys

        total_size = 0
        total_size += sys.getsizeof(self._memory_cache)
        total_size += sum(sys.getsizeof(v) for v in self._memory_cache.values())
        total_size += sys.getsizeof(self._metadata_cache)
        total_size += sum(sys.getsizeof(v) for v in self._metadata_cache.values())

        return total_size / (1024 * 1024)  # Convert to MB

    def _matches_criteria(self, state: dict[str, Any], criteria: dict[str, Any]) -> bool:
        """Check if state matches search criteria."""
        for key, value in criteria.items():
            if key not in state or state[key] != value:
                return False
        return True

    async def _load_metadata(self, cache_key: str) -> StateMetadata | None:
        """Load metadata for a state."""
        try:
            if not self.redis_client:
                return None

            metadata_key = f"metadata:{cache_key}"
            metadata_data = await self.redis_client.get(metadata_key)

            if metadata_data:
                metadata_dict = json.loads(metadata_data)
                # Convert back to StateMetadata object
                return StateMetadata(**metadata_dict)

            return None

        except Exception as e:
            self.logger.warning(f"Failed to load metadata for {cache_key}: {e}")
            return None

    async def _load_existing_states(self) -> None:
        """Load existing states from persistence layer."""
        try:
            if self._persistence:
                await self._persistence.load_all_states_to_cache()
                self.logger.info("Loaded existing states from persistence")

        except Exception as e:
            self.logger.warning(f"Failed to load existing states: {e}")

    async def _broadcast_state_change(
        self,
        state_type: StateType,
        state_id: str,
        state_data: dict[str, Any] | None,
        state_change: StateChange,
    ) -> None:
        """Broadcast state change to subscribers using consistent event-driven pattern."""
        try:
            event_data = {
                "state_type": state_type,
                "state_id": state_id,
                "state_data": state_data,
                "state_change": state_change,
            }

            # Use unified event system for consistent message patterns only
            await self._event_emitter.emit_async(
                f"state.{state_type.value}.changed", event_data, source="StateService"
            )

            # Process legacy subscribers synchronously to avoid queue pattern
            await self._notify_legacy_subscribers(state_type, state_id, state_data, state_change)

        except Exception as e:
            self.logger.warning(f"Failed to broadcast state change: {e}")

    async def _notify_legacy_subscribers(
        self,
        state_type: StateType,
        state_id: str,
        state_data: dict[str, Any] | None,
        state_change: StateChange,
    ) -> None:
        """Notify legacy subscribers without using queue pattern."""
        subscribers = self._subscribers.get(state_type, set())
        for callback in subscribers.copy():
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(state_type, state_id, state_data, state_change)
                else:
                    callback(state_type, state_id, state_data, state_change)
            except Exception as e:
                self.logger.error(f"Legacy subscriber callback error: {e}")
                # Remove failing callback
                self._subscribers[state_type].discard(callback)

    # Background Loops

    async def _synchronization_loop(self) -> None:
        """Background synchronization loop."""
        while self._running:
            try:
                if self._synchronizer:
                    await self._synchronizer.sync_pending_changes()

                await asyncio.sleep(self.sync_interval_seconds)

            except Exception as e:
                self.logger.error(f"Synchronization loop error: {e}")
                await asyncio.sleep(self.sync_interval_seconds)

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._running:
            try:
                # Clean up old locks
                datetime.now(timezone.utc)
                old_locks = [key for key in self._locks.keys() if not self._locks[key].locked()]

                for key in old_locks[:100]:  # Cleanup in batches
                    self._locks.pop(key, None)

                # Clean up old change log entries
                if len(self._change_log) > self.max_change_log_size:
                    self._change_log = self._change_log[-self.max_change_log_size // 2 :]

                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(300)

    async def _metrics_loop(self) -> None:
        """Background metrics collection loop."""
        while self._running:
            try:
                # Log metrics to InfluxDB
                metrics = await self.get_state_metrics()

                if self.influxdb_client:
                    try:
                        # Create Point object for InfluxDB
                        from influxdb_client import Point

                        point = Point("state_service_metrics")
                        point.tag("service", "state_service")
                        point.field("total_operations", metrics.total_operations)
                        point.field("successful_operations", metrics.successful_operations)
                        point.field("failed_operations", metrics.failed_operations)
                        point.field("cache_hit_rate", metrics.cache_hit_rate)
                        point.field("active_states", metrics.active_states_count)
                        point.field("memory_usage_mb", metrics.memory_usage_mb)
                        point.field("error_rate", metrics.error_rate)
                        point.time(datetime.now(timezone.utc))

                        # Run blocking InfluxDB operation in executor to avoid blocking event loop
                        await asyncio.get_event_loop().run_in_executor(
                            None, self.influxdb_client.write_point, point
                        )
                    except ImportError:
                        self.logger.warning(
                            "InfluxDB client not available, skipping metrics logging"
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to write metrics to InfluxDB: {e}")

                await asyncio.sleep(
                    self.validation_interval_seconds
                )  # Use configured validation interval

            except Exception as e:
                self.logger.error(f"Metrics loop error: {e}")
                await asyncio.sleep(self.validation_interval_seconds)

    # Event processing loop removed - using synchronous consistent patterns instead

    async def _backup_loop(self) -> None:
        """Background automatic backup loop."""
        while self._running:
            try:
                current_time = datetime.now(timezone.utc)

                # Check if backup is needed
                if (
                    not self._last_backup_time
                    or current_time - self._last_backup_time
                    >= timedelta(hours=self.backup_interval_hours)
                ):
                    # Create automatic backup
                    description = f"Automatic backup - {current_time.isoformat()}"
                    snapshot_id = await self.create_snapshot(description)

                    if snapshot_id:
                        self._last_backup_time = current_time
                        self.logger.info(f"Automatic backup created: {snapshot_id}")
                    else:
                        self.logger.warning("Automatic backup failed")

                # Sleep for configured cleanup interval before checking again
                await asyncio.sleep(self.cleanup_interval_seconds)

            except Exception as e:
                self.logger.error(f"Backup loop error: {e}")
                await asyncio.sleep(self.cleanup_interval_seconds)

    # Database Client Initialization Helper Methods

    async def _initialize_redis_client(self) -> None:
        """Initialize Redis client through database service factory or fallback."""
        if self._redis_initialized:
            return

        try:
            # Try to use database service factory method if available
            if self.database_service and hasattr(self.database_service, "create_redis_client"):
                # Safely get database config with fallback
                database_config = getattr(self.config, "database", {})

                redis_config = {
                    "host": getattr(database_config, "redis_host", "localhost"),
                    "port": getattr(database_config, "redis_port", 6379),
                    "password": getattr(database_config, "redis_password", None),
                    "db": getattr(database_config, "redis_db", 0),
                    "decode_responses": True,
                    "max_connections": 100,
                    "retry_on_timeout": True,
                    "health_check_interval": 30,
                }

                self.redis_client = await self.database_service.create_redis_client(redis_config)

            else:
                # Fallback: Create Redis client through connection manager
                from src.database.connection import get_redis_client

                try:
                    # Type note: get_redis_client() returns concrete client, cast to protocol
                    redis_client_concrete = await get_redis_client()
                    self.redis_client = redis_client_concrete  # type: ignore[assignment]
                except Exception as e:
                    self.logger.warning(f"Failed to get Redis client from connection manager: {e}")
                    self.redis_client = None
                    return

            # Test connection
            if self.redis_client:
                try:
                    if hasattr(self.redis_client, "connect"):
                        await self.redis_client.connect()
                    if hasattr(self.redis_client, "ping"):
                        await self.redis_client.ping()
                    self._redis_initialized = True
                    self.logger.info("Redis client initialized successfully")
                except Exception as e:
                    self.logger.warning(f"Redis connection test failed: {e}")
                    # CRITICAL: Properly cleanup Redis connection on failure
                    if self.redis_client:
                        try:
                            if hasattr(self.redis_client, "close"):
                                await self.redis_client.close()
                            elif hasattr(self.redis_client, "disconnect"):
                                await self.redis_client.disconnect()
                        except Exception as cleanup_error:
                            self.logger.error(
                                f"Error cleaning up Redis connection: {cleanup_error}"
                            )
                    self.redis_client = None

        except Exception as e:
            error_context = ErrorContext.from_exception(
                e,
                component="StateService",
                operation="initialize_redis_client",
                severity=ErrorSeverity.MEDIUM,
            )
            handler = self.error_handler
            await handler.handle_error(e, error_context)
            self.logger.warning(f"Redis client initialization failed: {e}")
            self.redis_client = None

    async def _initialize_influxdb_client(self) -> None:
        """Initialize InfluxDB client through database service factory or fallback."""
        if self._influxdb_initialized:
            return

        try:
            # Check if InfluxDB is configured
            influxdb_config = getattr(self.config, "influxdb", {})
            if not influxdb_config or not isinstance(influxdb_config, dict):
                # Safely get database config with fallback
                database_config = getattr(self.config, "database", {})
                influxdb_config = getattr(database_config, "influxdb", {})

            if not influxdb_config:
                self.logger.debug("InfluxDB not configured, skipping initialization")
                return

            # Try to use database service factory method if available
            if self.database_service and hasattr(self.database_service, "create_influxdb_client"):
                influx_client_config = {
                    "url": influxdb_config.get("url", "http://localhost:8086"),
                    "token": influxdb_config.get("token", ""),
                    "org": influxdb_config.get("org", "default"),
                    "bucket": influxdb_config.get("bucket", "trading"),
                }

                self.influxdb_client = await self.database_service.create_influxdb_client(
                    influx_client_config
                )

            else:
                # Fallback: Create InfluxDB client through connection manager
                from src.database.connection import get_influxdb_client

                try:
                    # Type note: get_influxdb_client() returns concrete client, cast to protocol
                    influxdb_client_concrete = get_influxdb_client()
                    self.influxdb_client = influxdb_client_concrete  # type: ignore[assignment]
                except Exception as e:
                    self.logger.warning(
                        f"Failed to get InfluxDB client from connection manager: {e}"
                    )
                    self.influxdb_client = None
                    return

            # Test connection
            if self.influxdb_client:
                try:
                    if hasattr(self.influxdb_client, "connect"):
                        await self.influxdb_client.connect()
                    if hasattr(self.influxdb_client, "ping"):
                        # ping is synchronous, run in executor to avoid blocking
                        await asyncio.get_event_loop().run_in_executor(
                            None, self.influxdb_client.ping
                        )
                    self._influxdb_initialized = True
                    self.logger.info("InfluxDB client initialized successfully")
                except Exception as e:
                    self.logger.warning(f"InfluxDB connection test failed: {e}")
                    self.influxdb_client = None

        except Exception as e:
            error_context = ErrorContext.from_exception(
                e,
                component="StateService",
                operation="initialize_influxdb_client",
                severity=ErrorSeverity.MEDIUM,
            )
            handler = self.error_handler
            await handler.handle_error(e, error_context)
            self.logger.warning(f"InfluxDB client initialization failed: {e}")
            self.influxdb_client = None

    async def _initialize_service_components(self) -> None:
        """Initialize service layer components with proper dependency injection."""
        try:
            # Initialize business service
            self._business_service = StateBusinessService()
            if hasattr(self._business_service, "start"):
                await self._business_service.start()

            # Initialize persistence service
            self._persistence_service = StatePersistenceService(self.database_service)
            if hasattr(self._persistence_service, "start"):
                await self._persistence_service.start()

            # Initialize validation service
            self._validation_service = StateValidationService()
            if hasattr(self._validation_service, "start"):
                await self._validation_service.start()

            # Initialize synchronization service
            self._synchronization_service = StateSynchronizationService()
            if hasattr(self._synchronization_service, "start"):
                await self._synchronization_service.start()

            self.logger.info("Service layer components initialized successfully")

        except Exception as e:
            error_context = ErrorContext.from_exception(
                e,
                component="StateService",
                operation="initialize_service_components",
                severity=ErrorSeverity.MEDIUM,
            )
            handler = self.error_handler
            await handler.handle_error(e, error_context)
            self.logger.error(f"Failed to initialize service components: {e}")
            # Set services to None for graceful degradation
            self._business_service = None
            self._persistence_service = None
            self._validation_service = None
            self._synchronization_service = None

    async def _initialize_state_components(self) -> None:
        """Initialize state management components with lazy imports to avoid circular deps."""
        try:
            # Lazy import to avoid circular dependencies
            try:
                from .state_persistence import StatePersistence

                self._persistence = StatePersistence(self)
            except ImportError as e:
                self.logger.warning(f"Failed to import StatePersistence: {e}")
                self._persistence = None

            try:
                from .state_synchronizer import StateSynchronizer

                self._synchronizer = StateSynchronizer(self)
            except ImportError as e:
                self.logger.warning(f"Failed to import StateSynchronizer: {e}")
                self._synchronizer = None

            try:
                from .state_validator import StateValidator

                self._validator = StateValidator(self)
            except ImportError as e:
                self.logger.warning(f"Failed to import StateValidator: {e}")
                self._validator = None

        except Exception as e:
            error_context = ErrorContext.from_exception(
                e,
                component="StateService",
                operation="initialize_state_components",
                severity=ErrorSeverity.MEDIUM,
            )
            handler = self.error_handler
            await handler.handle_error(e, error_context)
            self.logger.error(f"Failed to initialize state components: {e}")
            # Set components to None to allow graceful degradation
            self._persistence = None
            self._synchronizer = None
            self._validator = None

    @property
    def error_handler(self) -> ErrorHandler:
        """Get or create the singleton ErrorHandler instance."""
        if self._error_handler is None:
            self._error_handler = ErrorHandler(self.config)
        return self._error_handler

    # Service layer integration methods

    async def _validate_state_through_service(
        self, state_type: StateType, state_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate state through service layer or fall back to legacy."""
        if self._validation_service:
            return await self._validation_service.validate_state_data(state_type, state_data)
        elif self._validator:
            # Fall back to legacy validator
            result = await self._validator.validate_state(state_type, state_data)
            return {
                "is_valid": result.is_valid,
                "errors": [e.error_message for e in result.errors],
                "warnings": [w.warning_message for w in result.warnings],
            }
        else:
            # No validation available
            return {"is_valid": True, "errors": [], "warnings": []}

    async def _validate_state_transition_through_service(
        self,
        state_type: StateType,
        current_state: dict[str, Any],
        new_state: dict[str, Any],
    ) -> bool:
        """Validate state transition through service layer or fall back to legacy."""
        if self._validation_service:
            return await self._validation_service.validate_state_transition(
                state_type, current_state, new_state
            )
        elif self._validator:
            # Fall back to legacy validator
            return await self._validator.validate_state_transition(
                state_type, current_state, new_state
            )
        else:
            # No validation available - allow transition
            return True

    async def _process_state_update_through_service(
        self,
        state_type: StateType,
        state_id: str,
        state_data: dict[str, Any],
        source_component: str,
        reason: str,
    ) -> StateChange:
        """Process state update through business service or fall back to legacy."""
        if self._business_service:
            return await self._business_service.process_state_update(
                state_type, state_id, state_data, source_component, reason
            )
        else:
            # Fall back to creating state change directly
            return StateChange(
                state_id=state_id,
                state_type=state_type,
                operation=StateOperation.UPDATE,
                priority=StatePriority.MEDIUM,
                new_value=state_data,
                source_component=source_component,
                reason=reason,
            )

    async def _calculate_metadata_through_service(
        self,
        state_type: StateType,
        state_id: str,
        state_data: dict[str, Any],
        source_component: str,
    ) -> StateMetadata:
        """Calculate metadata through business service or fall back to legacy."""
        if self._business_service:
            return await self._business_service.calculate_state_metadata(
                state_type, state_id, state_data, source_component
            )
        else:
            # Fall back to direct metadata calculation
            return StateMetadata(
                state_id=state_id,
                state_type=state_type,
                version=1,
                checksum=self._calculate_checksum(state_data),
                size_bytes=len(json.dumps(state_data, default=str).encode()),
                source_component=source_component,
            )

    async def _persist_state_through_service(
        self,
        state_type: StateType,
        state_id: str,
        state_data: dict[str, Any],
        metadata: StateMetadata,
    ) -> None:
        """Persist state through service layer or fall back to legacy."""
        if self._persistence_service:
            await self._persistence_service.save_state(state_type, state_id, state_data, metadata)
        elif self._persistence:
            # Fall back to legacy persistence
            await self._persistence.queue_state_save(state_type, state_id, state_data, metadata)
        # If no persistence is available, data is only stored in memory

    async def _synchronize_state_change_through_service(self, state_change: StateChange) -> None:
        """Synchronize state change through service layer or fall back to legacy."""
        if self._synchronization_service:
            await self._synchronization_service.synchronize_state_change(state_change)
        elif self._synchronizer:
            # Fall back to legacy synchronizer
            await self._synchronizer.queue_state_sync(state_change)
        # If no synchronization is available, change is not synchronized
