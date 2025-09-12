"""
Wrapper for ExecutionResult to provide backward compatibility.

The core ExecutionResult type uses instruction_id, but the execution module
expects execution_id and other properties. This wrapper provides those properties.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from src.core.types import (
    ExecutionAlgorithm,
    ExecutionResult as CoreExecutionResult,
    ExecutionStatus,
    OrderRequest,
)


class ExecutionResultWrapper:
    """
    Wrapper around core ExecutionResult to provide backward-compatible properties.

    This allows the execution module to use execution_id instead of instruction_id,
    and provides other properties that the module expects.
    """

    def __init__(
        self,
        core_result: CoreExecutionResult | None,
        original_order: OrderRequest | None = None,
        algorithm: ExecutionAlgorithm | None = None,
        state_registry: dict | None = None,
    ) -> None:
        """
        Initialize the wrapper.

        Args:
            core_result: The core ExecutionResult from the adapter (can be None for fallback)
            original_order: The original order request (can be None for fallback)
            algorithm: The algorithm used for execution
            state_registry: Registry to look up current state (for dynamic updates)
        """
        self._core = core_result
        self._original_order = original_order
        self._algorithm = algorithm
        self._state_registry = state_registry or {}

    def _update_core(self, new_core: CoreExecutionResult) -> None:
        """Update the core result for in-place updates."""
        self._core = new_core

    # Proxy core properties
    @property
    def instruction_id(self) -> str:
        return self._core.instruction_id

    @property
    def execution_id(self) -> str:
        """Map instruction_id to execution_id for backward compatibility."""
        return self._core.instruction_id

    @property
    def symbol(self) -> str:
        return self._core.symbol

    @property
    def status(self) -> ExecutionStatus:
        # Check for original status in metadata first
        original_status_str = self._core.metadata.get("original_status")
        if original_status_str:
            try:
                return ExecutionStatus(original_status_str)
            except ValueError:
                pass
        return self._core.status

    @status.setter
    def status(self, value: ExecutionStatus) -> None:
        # For TWAP finalization - we'll handle this differently
        pass

    @property
    def original_order(self) -> OrderRequest | None:
        return self._original_order

    @property
    def result(self) -> CoreExecutionResult:
        """Access to the underlying core result."""
        return self._core

    @property
    def original_request(self) -> OrderRequest | None:
        """Alias for original_order for backward compatibility."""
        return self._original_order

    @property
    def total_filled_quantity(self) -> Decimal:
        return self._core.filled_quantity

    @total_filled_quantity.setter
    def total_filled_quantity(self, value: Decimal) -> None:
        # Allow setting for test compatibility
        if hasattr(self._core, "filled_quantity"):
            self._core.filled_quantity = value

    @property
    def average_fill_price(self) -> Decimal:
        return self._core.average_price

    @property
    def total_fees(self) -> Decimal:
        return self._core.total_fees

    @property
    def number_of_trades(self) -> int:
        return self._core.num_fills

    @property
    def filled_quantity(self) -> Decimal:
        return self._core.filled_quantity

    @property
    def average_price(self) -> Decimal:
        return self._core.average_price

    @average_price.setter
    def average_price(self, value: Decimal) -> None:
        # Allow setting for test compatibility
        if hasattr(self._core, "average_price"):
            self._core.average_price = value

    @property
    def num_fills(self) -> int:
        return self._core.num_fills

    # Proxy methods
    def add_fill(
        self, price: Decimal, quantity: Decimal, timestamp: datetime, order_id: str
    ) -> None:
        """Add a fill - no-op for now as fills are tracked in state."""
        pass

    @property
    def algorithm(self) -> ExecutionAlgorithm | None:
        """Algorithm used for execution."""
        return self._algorithm

    @property
    def error_message(self) -> str | None:
        """Error message if execution failed."""
        return self._core.metadata.get("error_message")

    @property
    def child_orders(self) -> list:
        """Child orders list - needed for backward compatibility."""
        # Return empty list if not available, since this is used in tests
        return self._core.metadata.get("child_orders", [])

    @property
    def execution_duration(self) -> float | None:
        """Duration of execution in seconds."""
        if self._core.started_at and self._core.completed_at:
            return (self._core.completed_at - self._core.started_at).total_seconds()
        return self._core.execution_time if hasattr(self._core, "execution_time") else None

    @property
    def start_time(self) -> datetime:
        """Start time of execution."""
        return self._core.started_at

    @property
    def end_time(self) -> datetime | None:
        """End time of execution."""
        return self._core.completed_at

    # Missing methods expected by tests
    def get_summary(self) -> dict[str, Any]:
        """Get execution summary."""
        return {
            "instruction_id": self.instruction_id,
            "symbol": self.symbol,
            "status": self.status.value,
            "filled_quantity": str(self.filled_quantity),
            "average_price": str(self.average_price),
            "total_fees": str(self.total_fees),
            "execution_time": self.execution_duration,
            "algorithm": self._algorithm.value if self._algorithm else None,
        }

    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.COMPLETED

    def is_partial(self) -> bool:
        """Check if execution was partial."""
        return self.status == ExecutionStatus.PARTIAL

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        return {
            "slippage_bps": self._core.slippage_bps,
            "fill_rate": self._core.fill_rate,
            "execution_time": self.execution_duration,
            "total_fees": str(self.total_fees),
            "average_price": str(self.average_price),
        }

    def calculate_efficiency(self) -> Decimal:
        """Calculate execution efficiency score."""
        # Simple efficiency calculation based on fill rate and slippage
        fill_efficiency = Decimal(str(self._core.fill_rate * 100))  # 0-100 based on fill rate
        slippage_penalty = min(
            self._core.slippage_bps * Decimal("0.5"), Decimal("20")
        )  # Max 20 point penalty
        efficiency = max(fill_efficiency - slippage_penalty, Decimal("0"))
        return efficiency

    # Allow attribute access to core result
    def __getattr__(self, name: str) -> Any:
        """Proxy any other attributes to the core result."""
        return getattr(self._core, name)
