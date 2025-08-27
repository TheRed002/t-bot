"""
Exchange module interfaces for state integration.

This module defines the interface contracts for state services
used by the exchange module to avoid circular dependencies.
"""

from enum import Enum
from typing import Any, Protocol


class StateType(str, Enum):
    """State type enumeration (mirror of state module)."""

    ORDER_STATE = "order_state"
    BOT_STATE = "bot_state"
    POSITION_STATE = "position_state"
    SYSTEM_STATE = "system_state"


class StatePriority(str, Enum):
    """State priority enumeration (mirror of state module)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TradeEvent(str, Enum):
    """Trade event enumeration (mirror of state module)."""

    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACCEPTED = "order_accepted"
    ORDER_REJECTED = "order_rejected"
    PARTIAL_FILL = "partial_fill"
    COMPLETE_FILL = "complete_fill"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_EXPIRED = "order_expired"


class IStateService(Protocol):
    """Interface for StateService used by exchanges."""

    async def set_state(
        self,
        state_type: StateType,
        state_id: str,
        state_data: dict[str, Any],
        source_component: str,
        priority: StatePriority,
        reason: str,
    ) -> bool:
        """Set state data."""
        ...

    async def get_state(self, state_type: StateType, state_id: str) -> dict[str, Any] | None:
        """Get state data."""
        ...


class ITradeLifecycleManager(Protocol):
    """Interface for TradeLifecycleManager used by exchanges."""

    async def update_trade_event(
        self, trade_id: str, event: TradeEvent, event_data: dict[str, Any]
    ) -> None:
        """Update trade event."""
        ...
