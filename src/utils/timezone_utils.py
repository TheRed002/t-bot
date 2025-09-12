
"""
Timezone Utilities for Trading Systems.

Common utilities for handling timezone conversions and ensuring
proper datetime objects with timezone information.
"""

from datetime import datetime, timezone


def ensure_timezone_aware(dt: datetime | None, default_tz: timezone = timezone.utc) -> datetime:
    """
    Ensure datetime object has timezone information.

    Args:
        dt: Datetime object that may or may not have timezone info
        default_tz: Default timezone to use if dt is timezone-naive

    Returns:
        Timezone-aware datetime object

    Raises:
        TypeError: If dt is None or not a datetime object
    """
    if dt is None:
        raise TypeError("datetime object cannot be None")

    if not isinstance(dt, datetime):
        raise TypeError(f"Expected datetime object, got {type(dt).__name__}")

    # Already timezone-aware
    if dt.tzinfo is not None:
        return dt

    # Make timezone-aware using default timezone
    return dt.replace(tzinfo=default_tz)


def ensure_utc_timezone(dt: datetime | None) -> datetime:
    """
    Convenience function to ensure datetime has UTC timezone.

    Args:
        dt: Datetime object that may or may not have timezone info

    Returns:
        UTC timezone-aware datetime object
    """
    return ensure_timezone_aware(dt, timezone.utc)
