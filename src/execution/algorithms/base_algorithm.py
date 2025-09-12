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

from src.core.base.component import BaseComponent
from src.core.config import Config
from src.core.exceptions import ExecutionError, ValidationError

# MANDATORY: Import from P-001
from src.core.types import (
    ExecutionAlgorithm,
    ExecutionResult,
    ExecutionStatus,
    OrderResponse,
    OrderSide,
)

# Import adapter for type conversion
from src.execution.adapters import ExecutionResultAdapter

# Import result wrapper for backward compatibility
from src.execution.execution_result_wrapper import ExecutionResultWrapper

# Import execution state management
from src.execution.execution_state import ExecutionState

# Import internal execution instruction type
from src.execution.types import ExecutionInstruction

# MANDATORY: Import from P-007A
from src.utils import log_calls, time_execution


class BaseAlgorithm(BaseComponent, ABC):
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

    def __init__(self, config: Config) -> None:
        """
        Initialize the base execution algorithm.

        Args:
            config: Application configuration
        """
        super().__init__()  # Initialize BaseComponent
        self.config = config

        # Algorithm state
        # is_running is managed by BaseComponent
        self.current_executions: dict[str, ExecutionState] = {}
        self.result_wrappers: dict[str, ExecutionResultWrapper] = {}  # Track wrappers

        # Performance tracking
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0

        self._logger.info(f"Initialized {self.__class__.__name__} execution algorithm")

    async def start(self) -> None:
        """Start the execution algorithm."""
        await super().start()  # This sets _is_running = True
        self._logger.info(f"Started {self.__class__.__name__} execution algorithm")

    async def stop(self) -> None:
        """Stop the execution algorithm."""
        # Cancel any running executions
        for execution_id in list(self.current_executions.keys()):
            await self.cancel_execution(execution_id)
        await super().stop()  # This sets _is_running = False
        self._logger.info(f"Stopped {self.__class__.__name__} execution algorithm")

    @abstractmethod
    async def execute(
        self, instruction: ExecutionInstruction, exchange_factory=None, risk_manager=None
    ) -> ExecutionResultWrapper:
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

            self._logger.debug(
                "Execution instruction validated",
                symbol=instruction.order.symbol,
                algorithm=instruction.algorithm.value,
            )

            return True

        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            self._logger.error(
                "Unexpected error during instruction validation",
                error=str(e),
                symbol=instruction.order.symbol if instruction.order else None,
            )
            raise ValidationError(f"Instruction validation failed: {e}") from e

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
        state = self.current_executions.get(execution_id)
        if state:
            return self._state_to_result(state)
        return None

    def _generate_execution_id(self) -> str:
        """
        Generate a unique execution ID.

        Returns:
            str: Unique execution ID
        """
        return f"{self.__class__.__name__.lower()}_{uuid.uuid4().hex[:8]}"

    async def _create_execution_state(
        self, instruction: ExecutionInstruction, execution_id: str | None = None
    ) -> ExecutionState:
        """
        Create an initial execution state for tracking.

        Args:
            instruction: Execution instruction
            execution_id: Optional execution ID (generated if not provided)

        Returns:
            ExecutionState: Initial execution state
        """
        if not execution_id:
            execution_id = self._generate_execution_id()

        # Create mutable execution state
        state = ExecutionState(
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

        # Store in current executions
        self.current_executions[execution_id] = state

        return state

    def _state_to_result(self, state: ExecutionState) -> ExecutionResult:
        """Convert ExecutionState to core ExecutionResult and wrap for compatibility."""
        core_result = ExecutionResultAdapter.to_core_result(
            execution_id=state.execution_id,
            original_order=state.original_order,
            algorithm=state.algorithm,
            status=state.status,
            start_time=state.start_time,
            child_orders=state.child_orders,
            total_filled_quantity=state.total_filled_quantity,
            average_fill_price=state.average_fill_price,
            total_fees=state.total_fees,
            end_time=state.end_time,
            execution_duration=state.execution_duration,
            error_message=state.error_message,
            metadata=state.metadata,
        )
        # Always wrap the result for backward compatibility
        wrapper = ExecutionResultWrapper(core_result, state.original_order, state.algorithm)
        # Track the wrapper for future updates
        self.result_wrappers[state.execution_id] = wrapper
        return wrapper

    async def _update_execution_state(
        self,
        execution_state: ExecutionState,
        status: ExecutionStatus | None = None,
        child_order: OrderResponse | None = None,
        error_message: str | None = None,
    ) -> ExecutionState:
        """
        Update an execution state with new information.

        Args:
            execution_state: Execution state to update
            status: New status (optional)
            child_order: New child order to add (optional)
            error_message: Error message if failed (optional)

        Returns:
            ExecutionState: Updated execution state
        """
        if status:
            execution_state.status = status

        if child_order:
            execution_state.add_child_order(child_order)

        if error_message:
            execution_state.set_failed(error_message, datetime.now(timezone.utc))

        # Update timing if completed or failed
        if status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]:
            execution_state.set_completed(datetime.now(timezone.utc))

        return execution_state

    async def _create_execution_result(
        self, instruction: ExecutionInstruction, execution_id: str | None = None
    ) -> ExecutionResult:
        """
        Create an initial execution result for tracking.
        This is a convenience method that creates a state and converts it to result.

        Args:
            instruction: Execution instruction
            execution_id: Optional execution ID (generated if not provided)

        Returns:
            ExecutionResult: Initial execution result with execution_id property
        """
        state = await self._create_execution_state(instruction, execution_id)
        # Register the state
        self.current_executions[state.execution_id] = state
        # _state_to_result already returns wrapped result
        return self._state_to_result(state)

    async def _update_execution_result(
        self,
        execution_result: ExecutionResult,
        status: ExecutionStatus | None = None,
        child_order: OrderResponse | None = None,
        error_message: str | None = None,
    ) -> ExecutionResult:
        """
        Update an execution result with new information.
        This method updates the underlying state and returns the updated result.

        Args:
            execution_result: Execution result to update
            status: New status (optional)
            child_order: New child order to add (optional)
            error_message: Error message if failed (optional)

        Returns:
            ExecutionResult: Updated execution result
        """
        # Find the corresponding state
        # Get execution_id from either property or attribute
        execution_id = (
            getattr(execution_result, "execution_id", None) or execution_result.instruction_id
        )
        state = self.current_executions.get(execution_id)
        if not state:
            # If no state found, create one from the result
            raise ExecutionError(f"No execution state found for {execution_id}")

        # Update the state
        await self._update_execution_state(state, status, child_order, error_message)

        # Update the existing wrapper or create a new one
        existing_wrapper = self.result_wrappers.get(execution_id)
        if existing_wrapper:
            # Update the existing wrapper with new core result
            new_core = ExecutionResultAdapter.to_core_result(
                execution_id=state.execution_id,
                original_order=state.original_order,
                algorithm=state.algorithm,
                status=state.status,
                start_time=state.start_time,
                child_orders=state.child_orders,
                total_filled_quantity=state.total_filled_quantity,
                average_fill_price=state.average_fill_price,
                total_fees=state.total_fees,
                end_time=state.end_time,
                execution_duration=state.execution_duration,
                error_message=state.error_message,
                metadata=state.metadata,
            )
            existing_wrapper._update_core(new_core)
            return existing_wrapper
        else:
            # _state_to_result already returns wrapped result
            return self._state_to_result(state)

    async def _calculate_slippage_metrics(
        self, execution_state: ExecutionState, expected_price: Decimal | None = None
    ) -> None:
        """
        Calculate slippage and cost metrics for the execution.

        Args:
            execution_state: Execution state to update
            expected_price: Expected execution price for slippage calculation
        """
        try:
            if not execution_state.average_fill_price or execution_state.total_filled_quantity <= 0:
                return

            # Set expected price if provided
            if expected_price:
                execution_state.expected_price = expected_price

                # Calculate price slippage (positive = adverse)
                if execution_state.original_order.side == OrderSide.BUY:
                    # For buy orders, slippage is paying more than expected
                    execution_state.price_slippage = (
                        execution_state.average_fill_price - expected_price
                    )
                else:
                    # For sell orders, slippage is receiving less than expected
                    execution_state.price_slippage = (
                        expected_price - execution_state.average_fill_price
                    )

            # Calculate fees
            total_fees = Decimal("0")
            for child_order in execution_state.child_orders:
                # Estimate fees (can be made more sophisticated)
                if child_order.filled_quantity > 0 and child_order.price:
                    order_value = child_order.filled_quantity * child_order.price
                    estimated_fee = order_value * Decimal("0.001")  # 0.1% estimated fee
                    total_fees += estimated_fee

            execution_state.total_fees = total_fees

            self._logger.debug(
                "Slippage metrics calculated",
                execution_id=execution_state.execution_id,
                price_slippage=(
                    str(execution_state.price_slippage) if execution_state.price_slippage else 0
                ),
                total_fees=str(execution_state.total_fees),
            )

        except (ValidationError, ExecutionError) as e:
            self._logger.warning(
                "Failed to calculate slippage metrics",
                execution_id=execution_state.execution_id,
                error=str(e),
            )
            raise
        except Exception as e:
            self._logger.error(
                "Unexpected error calculating slippage metrics",
                execution_id=execution_state.execution_id,
                error=str(e),
            )
            raise ExecutionError(f"Slippage calculation failed: {e}") from e

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

            self._logger.debug(
                "Cleaned up completed executions",
                removed_count=len(to_remove),
                remaining_count=len(self.current_executions),
            )

    def _validate_exchange_factory(self, exchange_factory) -> None:
        """
        Validate that exchange factory is provided and accessible.

        Args:
            exchange_factory: Exchange factory to validate

        Raises:
            ExecutionError: If exchange factory is invalid
        """
        if not exchange_factory:
            raise ExecutionError(
                f"Exchange factory is required for {self.__class__.__name__} execution"
            )

    async def _get_exchange_from_factory(self, exchange_factory, instruction: ExecutionInstruction):
        """
        Get exchange instance from factory based on instruction preferences.

        Args:
            exchange_factory: Exchange factory
            instruction: Execution instruction with exchange preferences

        Returns:
            Exchange instance

        Raises:
            ExecutionError: If exchange cannot be obtained
        """
        # Determine which exchange to use
        exchange_name = "binance"  # Default exchange
        if instruction.preferred_exchanges:
            exchange_name = instruction.preferred_exchanges[0]

        try:
            exchange = await exchange_factory.get_exchange(exchange_name)
            if not exchange:
                raise ExecutionError(f"Failed to get exchange: {exchange_name}")
            return exchange
        except Exception as e:
            raise ExecutionError(f"Failed to access exchange {exchange_name}: {e}")

    def _update_execution_statistics(self, status: ExecutionStatus) -> None:
        """
        Update algorithm execution statistics.

        Args:
            status: Final execution status
        """
        if status == ExecutionStatus.COMPLETED:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
        self.total_executions += 1

    async def _handle_execution_error(
        self, e: Exception, execution_id: str = None, algorithm_name: str = None
    ) -> None:
        """
        Handle execution errors with common error handling pattern.

        Args:
            e: Exception that occurred
            execution_id: Execution ID if available
            algorithm_name: Algorithm name for error messages
        """
        algo_name = algorithm_name or self.__class__.__name__

        # Update execution result if we have the execution_id
        if execution_id and execution_id in self.current_executions:
            await self._update_execution_result(
                self.current_executions[execution_id],
                status=ExecutionStatus.FAILED,
                error_message=str(e),
            )
            self.failed_executions += 1
            self.total_executions += 1

        self._logger.error(
            f"{algo_name} execution failed",
            execution_id=execution_id or "unknown",
            error=str(e),
        )

    async def _standard_cancel_execution(
        self, execution_id: str, algorithm_name: str = None
    ) -> bool:
        """
        Standard cancellation logic used by all algorithms.

        Args:
            execution_id: ID of execution to cancel
            algorithm_name: Algorithm name for logging

        Returns:
            bool: True if cancellation successful, False otherwise
        """
        algo_name = algorithm_name or self.__class__.__name__

        try:
            if execution_id not in self.current_executions:
                self._logger.warning(f"Execution not found for cancellation: {execution_id}")
                return False

            execution_result = self.current_executions[execution_id]

            if execution_result.status not in [ExecutionStatus.RUNNING, ExecutionStatus.PENDING]:
                self._logger.warning(
                    f"Cannot cancel execution in status: {execution_result.status.value}"
                )
                return False

            # Update status to cancelled
            await self._update_execution_result(execution_result, status=ExecutionStatus.CANCELLED)

            self._logger.info(f"{algo_name} execution cancelled: {execution_id}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to cancel {algo_name} execution: {e}")
            return False

    # Abstract methods required by BaseComponent
    async def _do_start(self) -> None:
        """Component-specific startup logic."""
        # Override in subclasses if needed
        pass

    async def _do_stop(self) -> None:
        """Component-specific cleanup logic."""
        # Cancel any running executions
        for execution_id in list(self.current_executions.keys()):
            try:
                await self.cancel_execution(execution_id)
            except Exception as e:
                self._logger.error(f"Error cancelling execution {execution_id}: {e}")

    async def _health_check_internal(self) -> Any:
        """Component-specific health checks."""
        # Basic health: check if we have too many stuck executions
        stuck_count = sum(
            1
            for state in self.current_executions.values()
            if state.status == ExecutionStatus.RUNNING
            and (datetime.now(timezone.utc) - state.start_time).total_seconds() > 300  # 5 minutes
        )

        if stuck_count > 5:
            return {"status": "unhealthy", "reason": f"{stuck_count} stuck executions"}

        return {"status": "healthy", "active_executions": len(self.current_executions)}
