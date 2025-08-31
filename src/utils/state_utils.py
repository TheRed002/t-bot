"""
Common state management utilities.

This module provides shared utilities for state management operations
to eliminate code duplication across state components.
"""

import asyncio
import hashlib
import json
import pickle
import gzip
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from src.core.exceptions import StateError, ValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)


def calculate_state_checksum(data: Dict[str, Any]) -> str:
    """
    Calculate checksum for state data integrity.
    
    Args:
        data: State data to checksum
        
    Returns:
        SHA256 hexdigest of the data
    """
    try:
        # Ensure consistent serialization
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate state checksum: {e}")
        raise StateError(f"Checksum calculation failed: {e}") from e


def serialize_state_data(
    data: Dict[str, Any], 
    compress: bool = False,
    compression_threshold: int = 1024
) -> bytes:
    """
    Serialize state data with optional compression.
    
    Args:
        data: State data to serialize
        compress: Whether to compress the data
        compression_threshold: Minimum size for compression
        
    Returns:
        Serialized (and optionally compressed) data
    """
    try:
        # Serialize data
        serialized = pickle.dumps(data)
        
        # Apply compression if requested and data is large enough
        if compress and len(serialized) > compression_threshold:
            return gzip.compress(serialized)
            
        return serialized
    except Exception as e:
        logger.error(f"Failed to serialize state data: {e}")
        raise StateError(f"State serialization failed: {e}") from e


def deserialize_state_data(data: bytes, is_compressed: bool = False) -> Dict[str, Any]:
    """
    Deserialize state data with optional decompression.
    
    Args:
        data: Serialized data
        is_compressed: Whether data is compressed
        
    Returns:
        Deserialized state data
    """
    try:
        # Decompress if needed
        if is_compressed:
            data = gzip.decompress(data)
            
        # Deserialize
        return pickle.loads(data)
    except Exception as e:
        logger.error(f"Failed to deserialize state data: {e}")
        raise StateError(f"State deserialization failed: {e}") from e


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
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


def validate_decimal_fields(data: Dict[str, Any], decimal_fields: List[str]) -> List[str]:
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
                Decimal(str(data[field]))
            except (ValueError, TypeError) as e:
                errors.append(f"{field} must be a valid decimal: {e}")
    return errors


def validate_positive_values(data: Dict[str, Any], positive_fields: List[str]) -> List[str]:
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
                value = Decimal(str(data[field]))
                if value <= 0:
                    errors.append(f"{field} must be positive")
            except (ValueError, TypeError):
                errors.append(f"{field} must be a valid number")
    return errors


def create_state_metadata(
    state_id: str,
    state_type: str,
    state_data: Dict[str, Any],
    source_component: str = "",
    version: int = 1
) -> Dict[str, Any]:
    """
    Create standard metadata for state objects.
    
    Args:
        state_id: Unique state identifier
        state_type: Type of state
        state_data: State data
        source_component: Component that created the state
        version: State version number
        
    Returns:
        State metadata dictionary
    """
    now = datetime.now(timezone.utc)
    serialized_data = json.dumps(state_data, default=str, sort_keys=True)
    
    return {
        "state_id": state_id,
        "state_type": state_type,
        "version": version,
        "created_at": now,
        "updated_at": now,
        "checksum": calculate_state_checksum(state_data),
        "size_bytes": len(serialized_data.encode()),
        "source_component": source_component,
        "tags": {}
    }


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
    redis_client,
    key: str,
    value: str,
    ttl: int,
    timeout: float = 2.0
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
        await asyncio.wait_for(
            redis_client.setex(key, ttl, value),
            timeout=timeout
        )
        return True
    except asyncio.TimeoutError:
        logger.warning(f"Redis setex timeout for key: {key}")
        return False
    except Exception as e:
        logger.error(f"Redis setex failed for key {key}: {e}")
        return False


async def get_from_redis_with_timeout(
    redis_client,
    key: str,
    timeout: float = 2.0
) -> Optional[str]:
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
        return await asyncio.wait_for(
            redis_client.get(key),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.warning(f"Redis get timeout for key: {key}")
        return None
    except Exception as e:
        logger.error(f"Redis get failed for key {key}: {e}")
        return None


def detect_state_changes(
    old_state: Optional[Dict[str, Any]], 
    new_state: Dict[str, Any]
) -> set[str]:
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


def calculate_memory_usage(data_structures: List[Any]) -> float:
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
    old_value: Optional[Dict[str, Any]],
    new_value: Optional[Dict[str, Any]],
    source_component: str = "",
    reason: str = ""
) -> Dict[str, Any]:
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
        "persisted": False
    }


def ensure_directory_exists(directory_path: Union[str, Path]) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to directory
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        raise StateError(f"Cannot create directory {directory_path}: {e}") from e


def validate_state_transition_rules(
    state_type: str,
    current_status: str,
    new_status: str,
    transition_rules: Dict[str, Dict[str, set[str]]]
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


def update_moving_average(
    current_average: float,
    new_value: float,
    count: int
) -> float:
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


def format_state_for_logging(state_data: Dict[str, Any], max_length: int = 200) -> str:
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