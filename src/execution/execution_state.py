"""
Execution state management for tracking execution progress.

This module provides mutable state tracking for execution algorithms
while maintaining compatibility with immutable core types.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from src.core.types import (
    ExecutionAlgorithm,
    ExecutionStatus,
    OrderRequest,
    OrderResponse,
)


@dataclass
class ExecutionState:
    """Mutable state container for tracking execution progress."""

    execution_id: str
    original_order: OrderRequest
    algorithm: ExecutionAlgorithm
    status: ExecutionStatus
    start_time: datetime

    # Mutable tracking fields
    child_orders: list[OrderResponse] = field(default_factory=list)
    total_filled_quantity: Decimal = Decimal("0")
    average_fill_price: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    end_time: datetime | None = None
    execution_duration: float | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Additional tracking
    number_of_trades: int = 0
    expected_price: Decimal | None = None
    price_slippage: Decimal | None = None

    def add_child_order(self, child_order: OrderResponse) -> None:
        """Add a child order and update metrics."""
        self.child_orders.append(child_order)

        # Update metrics based on fills
        if child_order.filled_quantity > 0:
            old_total = self.total_filled_quantity
            old_avg_price = self.average_fill_price or Decimal("0")

            # Update total filled quantity
            self.total_filled_quantity += child_order.filled_quantity

            # Update average fill price (volume-weighted)
            if child_order.price and child_order.filled_quantity > 0:
                if old_total > 0 and old_avg_price > 0:
                    # Weighted average calculation
                    total_value = (old_total * old_avg_price) + (
                        child_order.filled_quantity * child_order.price
                    )
                    self.average_fill_price = total_value / self.total_filled_quantity
                else:
                    self.average_fill_price = child_order.price

        # Update trade count
        self.number_of_trades = len(self.child_orders)

    def set_completed(self, end_time: datetime) -> None:
        """Mark execution as completed."""
        self.end_time = end_time
        self.execution_duration = (end_time - self.start_time).total_seconds()

    def set_failed(self, error_message: str, end_time: datetime) -> None:
        """Mark execution as failed."""
        self.status = ExecutionStatus.FAILED
        self.error_message = error_message
        self.end_time = end_time
        self.execution_duration = (end_time - self.start_time).total_seconds()
