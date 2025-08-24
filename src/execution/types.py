"""
Execution module specific types.

These types are used internally by the execution module to maintain
backward compatibility while the core types are being refactored.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from src.core.types import (
    ExecutionAlgorithm,
    OrderRequest,
)


@dataclass
class ExecutionInstruction:
    """
    Internal execution instruction format used by execution engine.

    This maintains backward compatibility with the existing execution
    module implementation.
    """

    order: OrderRequest
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.TWAP
    strategy_name: str | None = None

    # Algorithm-specific parameters
    time_horizon_minutes: int | None = None
    participation_rate: float | None = None
    max_slices: int | None = None
    slice_size: Decimal | None = None
    slice_interval_seconds: int | None = None

    # VWAP specific
    volume_profile: dict[str, float] | None = None
    historical_volume_window_days: int | None = None

    # Iceberg specific
    visible_percentage: float | None = None
    randomization_factor: float | None = None

    # Smart routing
    preferred_exchanges: list[str] | None = None
    avoid_exchanges: list[str] | None = None

    # Risk controls
    max_slippage_bps: Decimal | None = None
    max_participation_rate: float | None = None
    stop_on_adverse_selection: bool = False

    # Timing
    start_time: datetime | None = None
    end_time: datetime | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
