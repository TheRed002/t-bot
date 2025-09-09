"""
Checksum and integrity utilities for state management.

This module provides centralized checksum calculation and integrity verification
functions used across the state management system to ensure data consistency
and detect corruption.
"""

import hashlib
import json
from typing import Any


def calculate_state_checksum(data: dict[str, Any]) -> str:
    """
    Calculate SHA256 checksum for state data integrity verification.

    This function provides consistent checksum calculation across all
    state management components to ensure data integrity.

    Args:
        data: State data dictionary to checksum

    Returns:
        SHA256 hexadecimal checksum string

    Raises:
        ValueError: If data cannot be serialized to JSON
    """
    if not data:
        return ""

    try:
        # Use consistent JSON serialization with sorted keys
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()
    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot calculate checksum for data: {e}") from e


def verify_state_integrity(data: dict[str, Any], expected_checksum: str) -> bool:
    """
    Verify state data integrity against expected checksum.

    Args:
        data: State data to verify
        expected_checksum: Expected SHA256 checksum

    Returns:
        True if data integrity is valid, False otherwise
    """
    if not expected_checksum:
        return False

    try:
        calculated_checksum = calculate_state_checksum(data)
        return calculated_checksum == expected_checksum
    except ValueError:
        return False


def calculate_metadata_hash(metadata: dict[str, Any]) -> str:
    """
    Calculate checksum for metadata objects.

    This is an alias for calculate_state_checksum but with clearer naming
    for metadata-specific use cases.

    Args:
        metadata: Metadata dictionary to checksum

    Returns:
        SHA256 hexadecimal checksum string
    """
    return calculate_state_checksum(metadata)


def batch_calculate_checksums(data_items: list[dict[str, Any]]) -> list[str]:
    """
    Calculate checksums for multiple data items efficiently.

    Args:
        data_items: List of data dictionaries to checksum

    Returns:
        List of corresponding checksums
    """
    return [calculate_state_checksum(item) for item in data_items]
