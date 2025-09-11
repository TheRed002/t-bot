"""
State Consistency Utilities

Simple utilities for state validation and error handling.
"""

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from src.core.exceptions import StateConsistencyError
from src.core.logging import get_logger

logger = get_logger(__name__)


def validate_state_data(data_type: str, data: Any) -> dict[str, Any]:
    """
    Simple state data validation.

    Args:
        data_type: Type of data being validated
        data: Data to validate

    Returns:
        Validation result
    """
    try:
        # Basic validation based on data type
        is_valid = True
        errors = []

        if data is None:
            is_valid = False
            errors.append("Data cannot be None")
        elif data_type == "trade_state" and isinstance(data, dict):
            required_fields = ["trade_id", "symbol", "side"]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                is_valid = False
                errors.extend([f"Missing required field: {field}" for field in missing_fields])

        return {
            "is_valid": is_valid,
            "data_type": data_type,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "errors": errors,
            "warnings": [],
        }

    except Exception as e:
        return {
            "is_valid": False,
            "data_type": data_type,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "errors": [str(e)],
            "warnings": [],
        }


def raise_state_error(message: str, context: dict[str, Any] | None = None) -> None:
    """
    Raise state error with consistent format.

    Args:
        message: Error message
        context: Additional error context
    """
    enhanced_context = context or {}
    enhanced_context.update({
        "error_source": "state_consistency",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    raise StateConsistencyError(message, context=enhanced_context)


# Simple utility functions for common operations
async def process_state_change(change: Any, processor: Callable) -> Any:
    """
    Process state change using simple async pattern.

    Args:
        change: State change data
        processor: Processing function

    Returns:
        Processed result
    """
    try:
        if callable(processor):
            import asyncio
            if asyncio.iscoroutinefunction(processor):
                return await processor(change)
            else:
                return processor(change)
        else:
            raise StateConsistencyError("Processor must be callable")
    except Exception as e:
        raise StateConsistencyError(f"Processing failed: {e}") from e


async def emit_state_event(event_type: str, event_data: dict[str, Any]) -> None:
    """
    Emit state event for cross-module integration.

    Args:
        event_type: Type of event being emitted
        event_data: Event data to emit
    """
    try:
        logger.debug(f"Emitting state event: {event_type}")
        # Placeholder for future event system integration
        # For now, just log the event

    except Exception as e:
        logger.warning(f"Failed to emit state event {event_type}: {e}")
