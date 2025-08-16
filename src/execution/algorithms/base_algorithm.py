"""
Base execution algorithm abstract class for T-Bot execution engine.

This module defines the abstract interface that all execution algorithms must
implement, ensuring consistent behavior and integration with the execution engine.

CRITICAL: This integrates with P-001 (types, exceptions, config),
P-002A (error handling), and P-007A (utils) components.
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import (
    ExecutionAlgorithm,
    ExecutionInstruction,
    ExecutionResult,
    ExecutionStatus,
    OrderResponse,
)

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

# MANDATORY: Import from P-007A
from src.utils.decorators import log_calls, time_execution

logger = get_logger(__name__)


class BaseAlgorithm(ABC):
    """
    Abstract base class for all execution algorithms.

    This class defines the interface that all execution algorithms must implement
    to ensure consistent behavior and proper integration with the execution engine.

    All execution algorithms must:
    1. Accept Config in __init__
    2. Implement execute() method returning ExecutionResult
    3. Handle partial fills and order lifecycle properly
    4. Use ErrorHandler for error scenarios
    5. Log all operations using get_logger
    """

    def __init__(self, config: Config):
        """
        Initialize the base execution algorithm.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.error_handler = ErrorHandler(config.error_handling)

        # Algorithm state
        self.is_running = False
        self.current_executions: dict[str, ExecutionResult] = {}

        # Performance tracking
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0

        self.logger.info(f"Initialized {self.__class__.__name__} execution algorithm")

    @abstractmethod
    async def execute(
        self, instruction: ExecutionInstruction, exchange_factory=None, risk_manager=None
    ) -> ExecutionResult:
        """
        Execute an order using this algorithm.

        Args:
            instruction: Execution instruction with order and parameters
            exchange_factory: Factory for creating exchange instances
            risk_manager: Risk manager for order validation

        Returns:
            ExecutionResult: Result of the execution with detailed metrics

        Raises:
            ExecutionError: If execution fails
            ValidationError: If instruction is invalid
        """
        pass

    @abstractmethod
    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an ongoing execution.

        Args:
            execution_id: ID of the execution to cancel

        Returns:
            bool: True if cancellation successful, False otherwise
        """
        pass

    @abstractmethod
    def get_algorithm_type(self) -> ExecutionAlgorithm:
        """
        Get the algorithm type enum.

        Returns:
            ExecutionAlgorithm: The algorithm type
        """
        pass

    @time_execution
    async def validate_instruction(self, instruction: ExecutionInstruction) -> bool:
        """
        Validate an execution instruction for this algorithm.

        Args:
            instruction: Execution instruction to validate

        Returns:
            bool: True if instruction is valid

        Raises:
            ValidationError: If instruction is invalid
        """
        try:
            # Basic instruction validation
            if not instruction or not instruction.order:
                raise ValidationError("Invalid execution instruction: missing order")

            if not instruction.order.symbol:
                raise ValidationError("Invalid order: missing symbol")

            if instruction.order.quantity <= 0:
                raise ValidationError("Invalid order: quantity must be positive")

            # Algorithm-specific validation
            await self._validate_algorithm_parameters(instruction)

            self.logger.debug(
                "Execution instruction validated",
                symbol=instruction.order.symbol,
                algorithm=instruction.algorithm.value,
            )

            return True

        except Exception as e:
            self.logger.error(
                "Instruction validation failed",
                error=str(e),
                symbol=instruction.order.symbol if instruction.order else None,
            )
            raise ValidationError(f"Instruction validation failed: {e}")

    @abstractmethod
    async def _validate_algorithm_parameters(self, instruction: ExecutionInstruction) -> None:
        """
        Validate algorithm-specific parameters.

        Args:
            instruction: Execution instruction to validate

        Raises:
            ValidationError: If parameters are invalid
        """
        pass

    async def get_execution_status(self, execution_id: str) -> ExecutionStatus | None:
        """
        Get the status of an ongoing execution.

        Args:
            execution_id: ID of the execution

        Returns:
            ExecutionStatus: Current status or None if not found
        """
        if execution_id in self.current_executions:
            return self.current_executions[execution_id].status
        return None

    async def get_execution_result(self, execution_id: str) -> ExecutionResult | None:
        """
        Get the result of an execution.

        Args:
            execution_id: ID of the execution

        Returns:
            ExecutionResult: Execution result or None if not found
        """
        return self.current_executions.get(execution_id)

    def _generate_execution_id(self) -> str:
        """
        Generate a unique execution ID.

        Returns:
            str: Unique execution ID
        """
        return f"{self.__class__.__name__.lower()}_{uuid.uuid4().hex[:8]}"

    async def _create_execution_result(
        self, instruction: ExecutionInstruction, execution_id: str | None = None
    ) -> ExecutionResult:
        """
        Create an initial execution result for tracking.

        Args:
            instruction: Execution instruction
            execution_id: Optional execution ID (generated if not provided)

        Returns:
            ExecutionResult: Initial execution result
        """
        if not execution_id:
            execution_id = self._generate_execution_id()

        return ExecutionResult(
            execution_id=execution_id,
            original_order=instruction.order,
            algorithm=self.get_algorithm_type(),
            status=ExecutionStatus.PENDING,
            start_time=datetime.now(timezone.utc),
            metadata={
                "algorithm_class": self.__class__.__name__,
                "instruction_metadata": instruction.metadata,
                "strategy_name": instruction.strategy_name,
            },
        )

    async def _update_execution_result(
        self,
        execution_result: ExecutionResult,
        status: ExecutionStatus | None = None,
        child_order: OrderResponse | None = None,
        error_message: str | None = None,
    ) -> ExecutionResult:
        """
        Update an execution result with new information.

        Args:
            execution_result: Execution result to update
            status: New status (optional)
            child_order: New child order to add (optional)
            error_message: Error message if failed (optional)

        Returns:
            ExecutionResult: Updated execution result
        """
        if status:
            execution_result.status = status

        if child_order:
            execution_result.child_orders.append(child_order)

            # Update metrics based on fills
            if child_order.filled_quantity > 0:
                old_total = execution_result.total_filled_quantity
                old_avg_price = execution_result.average_fill_price or Decimal("0")

                # Update total filled quantity
                execution_result.total_filled_quantity += child_order.filled_quantity

                # Update average fill price (volume-weighted)
                if child_order.price and child_order.filled_quantity > 0:
                    if old_total > 0 and old_avg_price > 0:
                        # Weighted average calculation
                        total_value = (old_total * old_avg_price) + (
                            child_order.filled_quantity * child_order.price
                        )
                        execution_result.average_fill_price = (
                            total_value / execution_result.total_filled_quantity
                        )
                    else:
                        execution_result.average_fill_price = child_order.price

            # Update trade count
            execution_result.number_of_trades = len(execution_result.child_orders)

        if error_message:
            execution_result.error_message = error_message
            execution_result.status = ExecutionStatus.FAILED

        # Update timing if completed or failed
        if status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]:
            end_time = datetime.now(timezone.utc)
            execution_result.end_time = end_time
            execution_result.execution_duration = (
                end_time - execution_result.start_time
            ).total_seconds()

        return execution_result

    async def _calculate_slippage_metrics(
        self, execution_result: ExecutionResult, expected_price: Decimal | None = None
    ) -> None:
        """
        Calculate slippage and cost metrics for the execution.

        Args:
            execution_result: Execution result to update
            expected_price: Expected execution price for slippage calculation
        """
        try:
            if (
                not execution_result.average_fill_price
                or execution_result.total_filled_quantity <= 0
            ):
                return

            # Set expected price if provided
            if expected_price:
                execution_result.expected_price = expected_price

                # Calculate price slippage (positive = adverse)
                if execution_result.original_order.side.value == "buy":
                    # For buy orders, slippage is paying more than expected
                    execution_result.price_slippage = (
                        execution_result.average_fill_price - expected_price
                    )
                else:
                    # For sell orders, slippage is receiving less than expected
                    execution_result.price_slippage = (
                        expected_price - execution_result.average_fill_price
                    )

            # Calculate fees
            total_fees = Decimal("0")
            for child_order in execution_result.child_orders:
                # Estimate fees (can be made more sophisticated)
                if child_order.filled_quantity > 0 and child_order.price:
                    order_value = child_order.filled_quantity * child_order.price
                    estimated_fee = order_value * Decimal("0.001")  # 0.1% estimated fee
                    total_fees += estimated_fee

            execution_result.total_fees = total_fees

            self.logger.debug(
                "Slippage metrics calculated",
                execution_id=execution_result.execution_id,
                price_slippage=float(execution_result.price_slippage),
                total_fees=float(execution_result.total_fees),
            )

        except Exception as e:
            self.logger.warning(
                "Failed to calculate slippage metrics",
                execution_id=execution_result.execution_id,
                error=str(e),
            )

    @log_calls
    async def get_algorithm_summary(self) -> dict[str, Any]:
        """
        Get summary information about the algorithm's performance.

        Returns:
            dict: Algorithm performance summary
        """
        success_rate = 0.0
        if self.total_executions > 0:
            success_rate = self.successful_executions / self.total_executions

        return {
            "algorithm_name": self.__class__.__name__,
            "algorithm_type": self.get_algorithm_type().value,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": success_rate,
            "is_running": self.is_running,
            "active_executions": len(self.current_executions),
        }

    async def cleanup_completed_executions(self, max_history: int = 100) -> None:
        """
        Clean up completed executions to prevent memory growth.

        Args:
            max_history: Maximum number of completed executions to keep
        """
        completed_executions = [
            (execution_id, result)
            for execution_id, result in self.current_executions.items()
            if result.status
            in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]
        ]

        if len(completed_executions) > max_history:
            # Remove oldest completed executions
            completed_executions.sort(key=lambda x: x[1].start_time)
            to_remove = completed_executions[:-max_history]

            for execution_id, _ in to_remove:
                del self.current_executions[execution_id]

            self.logger.debug(
                "Cleaned up completed executions",
                removed_count=len(to_remove),
                remaining_count=len(self.current_executions),
            )
