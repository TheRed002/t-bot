"""
Shared validation utilities for backtesting module.

This module contains common validation functions used across backtesting components
to eliminate duplication and ensure consistency.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict

from src.utils.validators import ValidationFramework


def validate_date_range(end_date: datetime, info: Any) -> datetime:
    """
    Validate that end date is after start date.

    Args:
        end_date: The end date to validate
        info: Pydantic field validation context

    Returns:
        Validated end date

    Raises:
        ValueError: If end date is not after start date
    """
    if "start_date" in info.data and end_date <= info.data["start_date"]:
        raise ValueError("End date must be after start date")
    return end_date


def validate_rate(rate: Decimal) -> Decimal:
    """
    Validate commission and slippage rates.

    Args:
        rate: Rate value to validate (as Decimal for precision)

    Returns:
        Validated rate

    Raises:
        ValueError: If rate is negative or greater than 10%
    """
    # Rates can be zero (no commission/slippage) but must be non-negative and <= 10%
    if rate < 0:
        raise ValueError("Rate must be between 0 and 0.1 (10%)")
    if rate > Decimal("0.1"):
        raise ValueError("Rate must be between 0 and 0.1 (10%)")
    return rate


def validate_symbol_list(symbols: list[str]) -> list[str]:
    """
    Validate symbol format for a list of trading symbols.

    Args:
        symbols: List of trading symbols to validate

    Returns:
        Validated symbol list

    Raises:
        ValueError: If any symbol is invalid
    """
    for symbol in symbols:
        try:
            ValidationFramework.validate_symbol(symbol)
        except ValueError as e:
            raise ValueError(f"Invalid symbol {symbol}: {e}") from e
    return symbols


def validate_is_expired(cache_entry: Dict[str, Any]) -> bool:
    """
    Check if a cache entry has expired.

    Args:
        cache_entry: Dict containing cache entry with 'created_at' and 'ttl_hours'

    Returns:
        True if expired, False otherwise
    """
    from datetime import timezone, timedelta

    if "created_at" not in cache_entry or "ttl_hours" not in cache_entry:
        return True

    created_at = cache_entry["created_at"]
    ttl_hours = cache_entry["ttl_hours"]

    # Ensure timezone awareness
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    expiry_time = created_at + timedelta(hours=ttl_hours)
    return datetime.now(timezone.utc) >= expiry_time