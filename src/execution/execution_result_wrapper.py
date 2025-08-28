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
        core_result: CoreExecutionResult,
        original_order: OrderRequest,
        algorithm: ExecutionAlgorithm | None = None,
    ):
        """
        Initialize the wrapper.

        Args:
            core_result: The core ExecutionResult from the adapter
            original_order: The original order request
            algorithm: The algorithm used for execution
        """
        self._core = core_result
        self._original_order = original_order
        self._algorithm = algorithm

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
        return self._core.status

    @status.setter
    def status(self, value: ExecutionStatus) -> None:
        # For TWAP finalization - we'll handle this differently
        pass

    @property
    def original_order(self) -> OrderRequest:
        return self._original_order

    @property
    def total_filled_quantity(self) -> Decimal:
        return self._core.filled_quantity

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

    # Allow attribute access to core result
    def __getattr__(self, name: str) -> Any:
        """Proxy any other attributes to the core result."""
        return getattr(self._core, name)
