"""
Capital Management Resource Utilities

This module provides shared resource management and cleanup functions
for capital management services to prevent memory leaks and manage resources efficiently.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from src.core.logging import get_logger

logger = get_logger(__name__)


class ResourceManager:
    """
    Unified resource manager for capital management services.

    Provides common patterns for managing memory usage, cleanup cycles,
    and resource limits across all capital management services.
    """

    def __init__(self, max_history_size: int = 1000):
        """
        Initialize resource manager.

        Args:
            max_history_size: Maximum number of historical records to keep
        """
        self.max_history_size = max_history_size
        self._cleanup_thresholds = {
            "fund_flows": max_history_size,
            "rate_history": 100,
            "slippage_history": 100,
            "performance_data": 30,  # days
        }

    def limit_list_size(
        self, data_list: list[Any], max_size: int | None = None, keep_recent: bool = True
    ) -> list[Any]:
        """
        Limit list size to prevent memory growth.

        Args:
            data_list: List to limit
            max_size: Maximum size (uses default if None)
            keep_recent: Whether to keep recent items (True) or oldest (False)

        Returns:
            Trimmed list
        """
        if not data_list:
            return data_list

        limit = max_size or self.max_history_size

        if len(data_list) <= limit:
            return data_list

        if keep_recent:
            trimmed = data_list[-limit:]
            logger.debug(
                f"Trimmed list from {len(data_list)} to {len(trimmed)} items (keeping recent)"
            )
        else:
            trimmed = data_list[:limit]
            logger.debug(
                f"Trimmed list from {len(data_list)} to {len(trimmed)} items (keeping oldest)"
            )

        return trimmed

    def clean_time_based_data(
        self,
        data_dict: dict[str, list[tuple[datetime, Any]]],
        max_age_days: int = 30,
        max_items_per_key: int = 100,
    ) -> dict[str, list[tuple[datetime, Any]]]:
        """
        Clean time-based data dictionaries (e.g., rate history, performance metrics).

        Args:
            data_dict: Dictionary of time-series data
            max_age_days: Maximum age of data to keep
            max_items_per_key: Maximum items per dictionary key

        Returns:
            Cleaned data dictionary
        """
        if not data_dict:
            return data_dict

        cutoff_time = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        cleaned_dict = {}

        for key, time_series in data_dict.items():
            if not time_series:
                continue

            # Filter by age and limit size
            recent_data = [
                (timestamp, value) for timestamp, value in time_series if timestamp >= cutoff_time
            ]

            # Limit size
            if len(recent_data) > max_items_per_key:
                recent_data = recent_data[-max_items_per_key:]

            if recent_data:
                cleaned_dict[key] = recent_data

        items_removed = sum(len(v) for v in data_dict.values()) - sum(
            len(v) for v in cleaned_dict.values()
        )
        if items_removed > 0:
            logger.debug(f"Cleaned {items_removed} old time-series items from data dictionary")

        return cleaned_dict

    def clean_fund_flows(self, fund_flows: list[Any], max_age_days: int = 30) -> list[Any]:
        """
        Clean fund flow history based on age.

        Args:
            fund_flows: List of fund flow objects
            max_age_days: Maximum age to keep

        Returns:
            Cleaned fund flows list
        """
        if not fund_flows:
            return fund_flows

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)

        # Filter fund flows by timestamp
        cleaned_flows = [
            flow
            for flow in fund_flows
            if hasattr(flow, "timestamp") and flow.timestamp >= cutoff_date
        ]

        # Also apply size limit
        cleaned_flows = self.limit_list_size(cleaned_flows)

        items_removed = len(fund_flows) - len(cleaned_flows)
        if items_removed > 0:
            logger.debug(f"Cleaned {items_removed} old fund flow records")

        return cleaned_flows

    def clean_performance_data(
        self, performance_data: dict[str, dict[str, Any]], max_age_days: int = 30
    ) -> dict[str, dict[str, Any]]:
        """
        Clean performance data dictionary.

        Args:
            performance_data: Performance data by strategy
            max_age_days: Maximum age to keep

        Returns:
            Cleaned performance data
        """
        if not performance_data:
            return performance_data

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        cleaned_data = {}

        for strategy_id, metrics in performance_data.items():
            if not isinstance(metrics, dict):
                continue

            # Check if metrics have timestamp
            if "last_updated" in metrics:
                try:
                    last_updated = metrics["last_updated"]
                    if isinstance(last_updated, str):
                        last_updated = datetime.fromisoformat(last_updated)

                    if last_updated >= cutoff_date:
                        cleaned_data[strategy_id] = metrics
                except (ValueError, TypeError):
                    # Keep if timestamp parsing fails
                    cleaned_data[strategy_id] = metrics
            else:
                # Keep if no timestamp
                cleaned_data[strategy_id] = metrics

        strategies_removed = len(performance_data) - len(cleaned_data)
        if strategies_removed > 0:
            logger.debug(f"Cleaned {strategies_removed} old strategy performance records")

        return cleaned_data

    def cleanup_allocations_data(self, allocations_data: list[dict[str, Any]]) -> None:
        """
        Clean up allocations data after processing to prevent memory leaks.

        Args:
            allocations_data: List of allocation data dictionaries
        """
        try:
            # Clear references in the data
            for allocation_data in allocations_data:
                if isinstance(allocation_data, dict):
                    allocation_data.clear()

            # Force garbage collection for large datasets
            if len(allocations_data) > 100:
                import gc

                gc.collect()

        except Exception as e:
            logger.warning(f"Error during allocation data cleanup: {e}")

    def should_trigger_cleanup(self, current_size: int, data_type: str = "default") -> bool:
        """
        Check if cleanup should be triggered based on current data size.

        Args:
            current_size: Current size of data structure
            data_type: Type of data for threshold lookup

        Returns:
            bool: True if cleanup should be triggered
        """
        threshold = self._cleanup_thresholds.get(data_type, self.max_history_size)

        # Trigger cleanup when size exceeds threshold by 50%
        cleanup_threshold = int(threshold * 1.5)

        return current_size > cleanup_threshold

    def get_memory_usage_info(self) -> dict[str, Any]:
        """
        Get memory usage information for monitoring.

        Returns:
            Dict with memory usage information
        """
        try:
            import gc
            import sys

            # Get garbage collection stats
            gc_stats = gc.get_stats()

            # Get object count by type
            object_counts: dict[str, int] = {}
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                object_counts[obj_type] = object_counts.get(obj_type, 0) + 1

            # Get top 10 object types
            top_objects = dict(sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10])

            return {
                "gc_stats": gc_stats,
                "total_objects": len(gc.get_objects()),
                "top_object_types": top_objects,
                "reference_count": sys.getrefcount,
            }
        except Exception as e:
            logger.warning(f"Failed to get memory usage info: {e}")
            return {"error": str(e)}


# Global resource manager instance
_resource_manager = ResourceManager()


def get_resource_manager() -> ResourceManager:
    """
    Get global resource manager instance.

    Returns:
        ResourceManager instance
    """
    return _resource_manager


def cleanup_large_dict(
    data_dict: dict[Any, Any], max_size: int = 1000, keep_recent_keys: bool = True
) -> dict[Any, Any]:
    """
    Clean up large dictionaries to prevent memory issues.

    Args:
        data_dict: Dictionary to clean up
        max_size: Maximum number of keys to keep
        keep_recent_keys: Whether to keep recently added keys

    Returns:
        Cleaned dictionary
    """
    if len(data_dict) <= max_size:
        return data_dict

    if keep_recent_keys:
        # Keep the last max_size items (assumes dict maintains insertion order)
        keys_to_keep = list(data_dict.keys())[-max_size:]
    else:
        # Keep the first max_size items
        keys_to_keep = list(data_dict.keys())[:max_size]

    cleaned_dict = {key: data_dict[key] for key in keys_to_keep}

    logger.debug(f"Cleaned dictionary from {len(data_dict)} to {len(cleaned_dict)} entries")

    return cleaned_dict


def safe_clear_references(*objects) -> None:
    """
    Safely clear object references to prevent memory leaks.

    Args:
        *objects: Objects to clear references from
    """
    try:
        for obj in objects:
            if obj is not None:
                if hasattr(obj, "clear") and callable(obj.clear):
                    obj.clear()
                elif hasattr(obj, "__dict__"):
                    obj.__dict__.clear()

    except Exception as e:
        logger.warning(f"Error clearing object references: {e}")
