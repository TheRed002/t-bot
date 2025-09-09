"""
Serialization utilities for state management.

This module provides centralized serialization, deserialization, and compression
utilities used across the state management system for consistent data handling.
"""

import gzip
import hashlib
import json
from typing import Any


def serialize_state_data(
    data: dict[str, Any], compress: bool = False, compression_threshold: int = 1024
) -> bytes:
    """
    Serialize and optionally compress state data.

    This function provides consistent serialization across all state management
    components with optional compression for large data sets.

    Args:
        data: Data dictionary to serialize
        compress: Whether to compress the data
        compression_threshold: Minimum size in bytes before compression is applied

    Returns:
        Serialized data as bytes

    Raises:
        ValueError: If data cannot be serialized to JSON
    """
    if not data:
        return b""

    try:
        # Serialize to JSON bytes with consistent formatting
        json_data = json.dumps(data, sort_keys=True, default=str).encode("utf-8")

        # Apply compression if requested and data exceeds threshold
        if compress and len(json_data) > compression_threshold:
            return gzip.compress(json_data)
        else:
            return json_data

    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot serialize data: {e}") from e


def deserialize_state_data(data: bytes, is_compressed: bool = False) -> dict[str, Any]:
    """
    Deserialize and optionally decompress state data.

    Args:
        data: Serialized data bytes
        is_compressed: Whether data is compressed

    Returns:
        Deserialized data as dictionary

    Raises:
        ValueError: If data cannot be deserialized
    """
    if not data:
        return {}

    try:
        # Decompress if needed
        if is_compressed:
            try:
                data = gzip.decompress(data)
            except gzip.BadGzipFile:
                # Data might not actually be compressed, try as-is
                pass

        # Deserialize from JSON
        return json.loads(data.decode("utf-8"))

    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
        raise ValueError(f"Cannot deserialize data: {e}") from e


def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
    """
    Calculate compression ratio.

    Args:
        original_size: Original data size in bytes
        compressed_size: Compressed data size in bytes

    Returns:
        Compression ratio (0.0 to 1.0, lower is better compression)
    """
    if original_size == 0:
        return 1.0
    return compressed_size / original_size


def should_compress_data(data_size: int, threshold: int = 1024) -> bool:
    """
    Determine if data should be compressed based on size threshold.

    Args:
        data_size: Size of data in bytes
        threshold: Minimum size threshold for compression

    Returns:
        True if data should be compressed
    """
    return data_size > threshold


def serialize_with_metadata(
    data: dict[str, Any], metadata: dict[str, Any], compress: bool = False
) -> bytes:
    """
    Serialize data with metadata wrapper.

    Args:
        data: Main data to serialize
        metadata: Metadata to include
        compress: Whether to compress

    Returns:
        Serialized data with metadata wrapper
    """
    wrapper = {"data": data, "metadata": metadata, "version": "1.0"}
    return serialize_state_data(wrapper, compress=compress)


def deserialize_with_metadata(
    data: bytes, is_compressed: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Deserialize data with metadata wrapper.

    Args:
        data: Serialized data with metadata wrapper
        is_compressed: Whether data is compressed

    Returns:
        Tuple of (data, metadata)
    """
    wrapper = deserialize_state_data(data, is_compressed)
    return wrapper.get("data", {}), wrapper.get("metadata", {})


class HashGenerator:
    """Utility class for generating consistent hashes across the system."""

    @staticmethod
    def generate_backtest_hash(request: Any) -> str:
        """Generate deterministic hash for backtest request caching."""
        cache_data = {
            "strategy_config": request.strategy_config,
            "symbols": sorted(request.symbols),
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "initial_capital": str(request.initial_capital),
            "timeframe": request.timeframe,
            "commission_rate": str(request.commission_rate),
            "slippage_rate": str(request.slippage_rate),
            "enable_shorting": request.enable_shorting,
            "max_open_positions": request.max_open_positions,
            "risk_config": request.risk_config,
            "position_sizing_method": request.position_sizing_method,
        }

        cache_string = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.sha256(cache_string.encode()).hexdigest()

    @staticmethod
    def generate_data_hash(data: dict[str, Any]) -> str:
        """Generate hash for arbitrary data."""
        data_string = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_string.encode()).hexdigest()
