"""
Common state management utilities.

This module provides shared utilities for state management operations
to eliminate code duplication across state components.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.core.exceptions import StateError, ValidationError
from src.core.logging import get_logger

# Import centralized utilities to avoid duplication
from .checksum_utilities import calculate_state_checksum as _calculate_checksum
from .file_utils import ensure_directory_exists as _ensure_directory_exists
from .serialization_utilities import (
    deserialize_state_data as _deserialize_data,
    serialize_state_data as _serialize_data,
)

logger = get_logger(__name__)


def create_state_metadata(
    state_id: str,
    state_type: "StateType",
    source_component: str,
    state_data: dict[str, Any] | None = None,
    version: int = 1,
    tags: list[str] | None = None
) -> "StateMetadata":
    """
    Create StateMetadata object with consistent patterns.
    
    Args:
        state_id: State identifier
        state_type: Type of state 
        source_component: Component creating the metadata
        state_data: Optional state data for checksum/size calculation
        version: State version (default 1)
        tags: Optional tags for metadata
    
    Returns:
        StateMetadata object with calculated fields
    """
    try:
        # Import here to avoid circular dependencies
        from src.core.types import StateType as StateTypeImport
        from src.state.state_service import StateMetadata

        # Handle both string and enum types
        if isinstance(state_type, str):
            # Convert string to StateType enum if needed
            state_type = getattr(StateTypeImport, state_type.upper(), StateTypeImport.BOT_STATE)

        # Calculate metadata fields if state_data provided
        checksum = None
        size_bytes = 0

        if state_data:
            checksum = calculate_state_checksum(state_data)
            serialized_data = serialize_state_data(state_data)
            size_bytes = len(serialized_data)

        # Generate default tags if not provided
        if tags is None:
            tags = [
                state_type.value,
                source_component.lower(),
                f"version_{version}"
            ]

        now = datetime.now(timezone.utc)

        return StateMetadata(
            state_id=state_id,
            state_type=state_type,
            current_version=version,
            created_at=now,
            updated_at=now,
            current_checksum=checksum,
            current_size_bytes=size_bytes,
            tags=tags
        )

    except Exception as e:
        logger.error(f"Failed to create state metadata: {e}")
        raise StateError(f"Metadata creation failed: {e}") from e


def handle_state_error(
    operation: str,
    state_type: str | None = None,
    state_id: str | None = None,
    error: Exception | None = None,
    error_context: dict[str, Any] | None = None
) -> None:
    """
    Standardized error handling for state operations.
    
    Args:
        operation: Description of the operation that failed
        state_type: Optional state type for context
        state_id: Optional state ID for context
        error: Original exception that was caught
        error_context: Additional context for error reporting
    """
    # Build context for logging
    context_parts = [operation]
    if state_type:
        context_parts.append(f"state_type={state_type}")
    if state_id:
        context_parts.append(f"state_id={state_id}")

    context_str = " | ".join(context_parts)

    # Log the error with context
    error_msg = f"State operation failed: {context_str}"
    if error:
        error_msg += f" | {error}"

    logger.error(error_msg, extra=error_context or {})

    # Raise appropriate exception based on operation type
    if "validation" in operation.lower():
        raise ValidationError(f"{operation} failed: {error}") from error
    elif "consistency" in operation.lower() or "sync" in operation.lower():
        from src.core.exceptions import StateConsistencyError
        raise StateConsistencyError(f"{operation} failed: {error}") from error
    else:
        raise StateError(f"{operation} failed: {error}") from error


def log_state_operation(
    operation: str,
    state_type: str | None = None,
    state_id: str | None = None,
    details: dict[str, Any] | None = None
) -> None:
    """
    Standardized logging for state operations.
    
    Args:
        operation: Description of the operation
        state_type: Optional state type for context
        state_id: Optional state ID for context  
        details: Additional details to log
    """
    # Build log message
    context_parts = [operation]
    if state_type:
        context_parts.append(f"state_type={state_type}")
    if state_id:
        context_parts.append(f"state_id={state_id}")

    message = " | ".join(context_parts)

    # Log with additional details
    logger.info(message, extra=details or {})


class StateOperationLock:
    """
    Centralized lock manager for state operations to eliminate duplication.
    """

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._cleanup_threshold = 1000  # Clean up locks after this many

    def get_lock(self, key: str) -> asyncio.Lock:
        """Get or create a lock for the given key."""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()

            # Periodic cleanup to prevent memory leaks
            if len(self._locks) > self._cleanup_threshold:
                self._cleanup_unused_locks()

        return self._locks[key]

    def _cleanup_unused_locks(self) -> None:
        """Remove unlocked locks to prevent memory leaks."""
        unused_keys = [
            key for key, lock in self._locks.items()
            if not lock.locked()
        ]

        # Remove up to half of unused locks
        cleanup_count = min(len(unused_keys), self._cleanup_threshold // 2)
        for key in unused_keys[:cleanup_count]:
            del self._locks[key]

        logger.debug(f"Cleaned up {cleanup_count} unused state operation locks")

    async def with_lock(self, key: str, operation: callable, *args, **kwargs):
        """Execute operation with lock for the given key."""
        lock = self.get_lock(key)
        async with lock:
            if asyncio.iscoroutinefunction(operation):
                return await operation(*args, **kwargs)
            else:
                return operation(*args, **kwargs)


# Global instance for state operation locks
_state_locks = StateOperationLock()


def get_state_lock(key: str) -> asyncio.Lock:
    """Get a lock for state operations - centralized lock management."""
    return _state_locks.get_lock(key)


async def with_state_lock(key: str, operation: callable, *args, **kwargs):
    """Execute operation with state lock - eliminates duplicate lock patterns."""
    return await _state_locks.with_lock(key, operation, *args, **kwargs)


class StateCache:
    """
    Centralized cache manager for state operations to eliminate duplication.
    """

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 300):
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds

    def get(self, key: str) -> Any | None:
        """Get value from cache if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]

            # Check if expired
            if (datetime.now(timezone.utc) - timestamp).total_seconds() < self._ttl_seconds:
                return value
            else:
                # Remove expired entry
                del self._cache[key]

        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache with timestamp."""
        # Clean up if at max size
        if len(self._cache) >= self._max_size:
            self._cleanup_expired()

            # If still at max size, remove oldest entries
            if len(self._cache) >= self._max_size:
                self._remove_oldest()

        self._cache[key] = (value, datetime.now(timezone.utc))

    def pop(self, key: str, default: Any = None) -> Any:
        """Remove and return value from cache."""
        if key in self._cache:
            value, _ = self._cache.pop(key)
            return value
        return default

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def keys(self) -> list[str]:
        """Get all cache keys."""
        return list(self._cache.keys())

    def _cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        now = datetime.now(timezone.utc)
        expired_keys = []

        for key, (_, timestamp) in self._cache.items():
            if (now - timestamp).total_seconds() >= self._ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    def _remove_oldest(self) -> None:
        """Remove oldest 25% of entries."""
        if not self._cache:
            return

        # Sort by timestamp and remove oldest 25%
        sorted_items = sorted(self._cache.items(), key=lambda x: x[1][1])
        remove_count = max(1, len(sorted_items) // 4)

        for key, _ in sorted_items[:remove_count]:
            del self._cache[key]

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        total_entries = len(self._cache)
        expired_count = 0
        now = datetime.now(timezone.utc)

        for _, (_, timestamp) in self._cache.items():
            if (now - timestamp).total_seconds() >= self._ttl_seconds:
                expired_count += 1

        return {
            "total_entries": total_entries,
            "expired_entries": expired_count,
            "valid_entries": total_entries - expired_count,
            "max_size": self._max_size,
            "ttl_seconds": self._ttl_seconds,
            "fill_ratio": total_entries / self._max_size if self._max_size > 0 else 0.0
        }


# Global cache instances for common state operations
_validation_cache = StateCache(max_size=5000, ttl_seconds=300)  # 5 minutes
_metadata_cache = StateCache(max_size=10000, ttl_seconds=600)   # 10 minutes
_general_cache = StateCache(max_size=10000, ttl_seconds=300)    # 5 minutes


def get_validation_cache() -> StateCache:
    """Get validation cache instance - eliminates duplicate validation cache management."""
    return _validation_cache


def get_metadata_cache() -> StateCache:
    """Get metadata cache instance - eliminates duplicate metadata cache management."""
    return _metadata_cache


def get_general_state_cache() -> StateCache:
    """Get general state cache instance - eliminates duplicate cache management."""
    return _general_cache


# Backward compatibility wrappers for state management utilities
def calculate_state_checksum(data: dict[str, Any]) -> str:
    """Calculate checksum for state data integrity (wrapper for centralized utility)."""
    return _calculate_checksum(data)


def serialize_state_data(
    data: dict[str, Any], compress: bool = False, compression_threshold: int = 1024
) -> bytes:
    """Serialize state data with optional compression (wrapper for centralized utility)."""
    return _serialize_data(data, compress, compression_threshold)


def deserialize_state_data(data: bytes, is_compressed: bool = False) -> dict[str, Any]:
    """Deserialize state data with optional decompression (wrapper for centralized utility)."""
    return _deserialize_data(data, is_compressed)


async def validate_required_fields(data: dict[str, Any], required_fields: list[str]) -> list[str]:
    """
    Validate that required fields are present in state data.

    Args:
        data: State data to validate
        required_fields: List of required field names

    Returns:
        List of missing field names
    """
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    return missing_fields


async def validate_decimal_fields(data: dict[str, Any], decimal_fields: list[str]) -> list[str]:
    """
    Validate that specified fields contain valid decimal values.

    Args:
        data: State data to validate
        decimal_fields: List of field names that should be decimals

    Returns:
        List of validation errors
    """
    errors = []
    for field in decimal_fields:
        if field in data and data[field] is not None:
            try:
                # Apply consistent data transformation for financial fields
                from src.utils.messaging_patterns import MessagingCoordinator

                coordinator = MessagingCoordinator("StateUtils")

                # Transform field value to ensure consistency
                field_data = {field: data[field]}
                transformed = coordinator._apply_data_transformation(field_data)

                # Validate the transformed decimal value
                if isinstance(transformed.get(field), Decimal):
                    continue  # Already valid decimal
                else:
                    # Try standard conversion as fallback
                    Decimal(str(data[field]))
            except (ValueError, TypeError) as e:
                errors.append(f"{field} must be a valid decimal: {e}")
    return errors


async def validate_positive_values(data: dict[str, Any], positive_fields: list[str]) -> list[str]:
    """
    Validate that specified fields contain positive values.

    Args:
        data: State data to validate
        positive_fields: List of field names that should be positive

    Returns:
        List of validation errors
    """
    errors = []
    for field in positive_fields:
        if field in data and data[field] is not None:
            try:
                # Apply consistent data transformation for financial validation
                from src.utils.messaging_patterns import BoundaryValidator

                # Create boundary validation data
                boundary_data = {
                    field: data[field],
                    "processing_mode": "stream",
                    "message_pattern": "pub_sub",
                    "data_format": "validation_data_v1",
                    "boundary_crossed": True,
                    "component": "StateUtils",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                # Apply boundary validation
                BoundaryValidator.validate_database_entity(boundary_data, "validate")

                value = Decimal(str(data[field]))
                if value <= 0:
                    errors.append(f"{field} must be positive")
            except (ValueError, TypeError):
                errors.append(f"{field} must be a valid number")
            except ValidationError as e:
                errors.append(f"{field} boundary validation failed: {e}")
    return errors


async def create_state_metadata_dict(
    state_id: str,
    state_type: str,
    state_data: dict[str, Any],
    source_component: str = "",
    version: int = 1,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Create standard metadata dictionary for state objects with consistent data transformation.

    Note: This is the async version that returns a dict.
    For StateMetadata objects, use the synchronous create_state_metadata() function.

    Args:
        state_id: Unique state identifier
        state_type: Type of state
        state_data: State data
        source_component: Component that created the state
        version: State version number
        tags: Optional list of tags for categorization

    Returns:
        State metadata dictionary
    """
    now = datetime.now(timezone.utc)

    # Apply consistent data transformation before serialization
    from src.utils.messaging_patterns import MessagingCoordinator

    coordinator = MessagingCoordinator("StateUtils")
    transformed_data = coordinator._apply_data_transformation(state_data)

    serialized_data = json.dumps(transformed_data, default=str, sort_keys=True)

    metadata = {
        "state_id": state_id,
        "state_type": state_type,
        "version": version,
        "created_at": now,
        "updated_at": now,
        "checksum": calculate_state_checksum(transformed_data),
        "size_bytes": len(serialized_data.encode()),
        "source_component": source_component,
        "tags": tags if tags is not None else [],
        # Add consistent processing metadata
        "processing_mode": "stream",
        "message_pattern": "pub_sub",
        "data_format": "metadata_v1",
        "boundary_crossed": False,
        "timestamp": now.isoformat(),
    }

    return metadata


def format_cache_key(state_type: str, state_id: str, prefix: str = "state") -> str:
    """
    Generate consistent cache keys for state data.

    Args:
        state_type: Type of state
        state_id: State identifier
        prefix: Cache key prefix

    Returns:
        Formatted cache key
    """
    return f"{prefix}:{state_type}:{state_id}"


async def store_in_redis_with_timeout(
    redis_client, key: str, value: str, ttl: int, timeout: float = 2.0
) -> bool:
    """
    Store data in Redis with timeout handling.

    Args:
        redis_client: Redis client instance
        key: Redis key
        value: Value to store
        ttl: Time to live in seconds
        timeout: Operation timeout

    Returns:
        True if successful
    """
    try:
        await asyncio.wait_for(redis_client.setex(key, ttl, value), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        logger.warning(f"Redis setex timeout for key: {key}")
        return False
    except Exception as e:
        logger.error(f"Redis setex failed for key {key}: {e}")
        return False


async def get_from_redis_with_timeout(redis_client, key: str, timeout: float = 2.0) -> str | None:
    """
    Get data from Redis with timeout handling.

    Args:
        redis_client: Redis client instance
        key: Redis key
        timeout: Operation timeout

    Returns:
        Value from Redis or None if not found/error
    """
    try:
        return await asyncio.wait_for(redis_client.get(key), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Redis get timeout for key: {key}")
        return None
    except Exception as e:
        logger.error(f"Redis get failed for key {key}: {e}")
        return None


def detect_state_changes(old_state: dict[str, Any] | None, new_state: dict[str, Any]) -> set[str]:
    """
    Detect which fields changed between state versions.

    Args:
        old_state: Previous state data
        new_state: New state data

    Returns:
        Set of changed field names
    """
    if not old_state:
        return set(new_state.keys())

    changed_fields = set()

    # Check for modified and new fields
    for key, value in new_state.items():
        if key not in old_state or old_state[key] != value:
            changed_fields.add(key)

    # Check for deleted fields
    for key in old_state:
        if key not in new_state:
            changed_fields.add(key)

    return changed_fields


def calculate_memory_usage(data_structures: list[Any]) -> float:
    """
    Calculate approximate memory usage of data structures in MB.

    Args:
        data_structures: List of data structures to measure

    Returns:
        Memory usage in MB
    """
    import sys

    total_size = 0
    for data in data_structures:
        total_size += sys.getsizeof(data)
        if isinstance(data, dict):
            total_size += sum(sys.getsizeof(v) for v in data.values())
        elif isinstance(data, list):
            total_size += sum(sys.getsizeof(v) for v in data)

    return total_size / (1024 * 1024)  # Convert to MB


def create_state_change_record(
    state_id: str,
    state_type: str,
    operation: str,
    old_value: dict[str, Any] | None,
    new_value: dict[str, Any] | None,
    source_component: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """
    Create a standardized state change record.

    Args:
        state_id: State identifier
        state_type: Type of state
        operation: Operation performed (create, update, delete)
        old_value: Previous state value
        new_value: New state value
        source_component: Component that made the change
        reason: Reason for the change

    Returns:
        State change record
    """
    return {
        "change_id": hashlib.sha256(
            f"{state_id}_{state_type}_{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:12],
        "state_id": state_id,
        "state_type": state_type,
        "operation": operation,
        "old_value": old_value,
        "new_value": new_value,
        "changed_fields": detect_state_changes(old_value, new_value) if new_value else set(),
        "timestamp": datetime.now(timezone.utc),
        "source_component": source_component,
        "reason": reason,
        "applied": False,
        "synchronized": False,
        "persisted": False,
    }


def ensure_directory_exists(directory_path: str | Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    DEPRECATED: Use _ensure_directory_exists from src.utils.file_utils instead.
    This function is kept for backward compatibility and delegates to the centralized version.

    Args:
        directory_path: Path to directory
    """
    try:
        _ensure_directory_exists(str(directory_path))
    except Exception as e:
        # Convert ValidationError to StateError for state context
        raise StateError(f"Cannot create directory {directory_path}: {e}") from e


def validate_state_transition_rules(
    state_type: str,
    current_status: str,
    new_status: str,
    transition_rules: dict[str, dict[str, set[str]]],
) -> bool:
    """
    Validate state transition against defined rules.

    Args:
        state_type: Type of state
        current_status: Current state status
        new_status: Proposed new status
        transition_rules: Rules mapping state types to allowed transitions

    Returns:
        True if transition is valid
    """
    if state_type not in transition_rules:
        return True  # No rules defined, allow transition

    state_rules = transition_rules[state_type]
    if current_status not in state_rules:
        return True  # No rules for current status, allow transition

    allowed_transitions = state_rules[current_status]
    return new_status in allowed_transitions


def update_moving_average(current_average: float, new_value: float, count: int) -> float:
    """
    Update a moving average with a new value.

    Args:
        current_average: Current average value
        new_value: New value to include
        count: Total number of values (including new one)

    Returns:
        Updated average
    """
    if count <= 0:
        return new_value
    return (current_average * (count - 1) + new_value) / count


def format_state_for_logging(state_data: dict[str, Any], max_length: int = 200) -> str:
    """
    Format state data for safe logging (truncate if too long).

    Args:
        state_data: State data to format
        max_length: Maximum string length

    Returns:
        Formatted string for logging
    """
    try:
        formatted = json.dumps(state_data, default=str, sort_keys=True)
        if len(formatted) > max_length:
            return formatted[:max_length] + "..."
        return formatted
    except Exception:
        return f"<State data with {len(state_data)} fields>"
