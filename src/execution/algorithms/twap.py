"""
Time-Weighted Average Price (TWAP) execution algorithm.

This module implements the TWAP execution strategy that splits large orders
across time intervals to minimize market impact and achieve time-weighted
average price execution.

CRITICAL: This integrates with P-001 (types, exceptions, config),
P-002A (error handling), and P-007A (utils) components.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.core.config import Config
from src.core.exceptions import (
    ExchangeError,
    ExecutionError,
    NetworkError,
    ServiceError,
    ValidationError,
)

# MANDATORY: Import from P-001
from src.core.types import (
    ExecutionAlgorithm,
    ExecutionResult,
    ExecutionStatus,
    OrderRequest,
    OrderType,
)

# Import exchange interfaces
from src.execution.exchange_interface import ExchangeFactoryInterface

# Import execution state for proper type hints
from src.execution.execution_state import ExecutionState

# Import internal execution instruction type
from src.execution.types import ExecutionInstruction

# MANDATORY: Import from P-002A
# MANDATORY: Import from P-007A
from src.utils import log_calls, time_execution

from .base_algorithm import BaseAlgorithm


class TWAPAlgorithm(BaseAlgorithm):
    """
    Time-Weighted Average Price (TWAP) execution algorithm.

    TWAP spreads order execution evenly across a specified time horizon,
    minimizing market impact by avoiding concentration of trading activity.

    Key features:
    - Configurable time slicing with adaptive timing
    - Market condition monitoring for slice adjustment
    - Participation rate controls to limit market impact
    - Intelligent order timing based on volume patterns
    """

    def __init__(self, config: Config):
        """
        Initialize TWAP execution algorithm.

        Args:
            config: Application configuration
        """
        super().__init__(config)

        # TWAP-specific configuration
        self.default_time_horizon_minutes = 60  # 1 hour default
        self.default_participation_rate = 0.2  # 20% of market volume
        self.min_slice_size_pct = 0.01  # Minimum 1% of order per slice
        self.max_slices = 100  # Maximum number of slices

        # Timing controls
        self.slice_interval_buffer = config.execution.twap_slice_interval_buffer_seconds
        self.market_close_buffer_minutes = 30  # Stop 30 minutes before market close

        # Risk validation configuration
        self.default_portfolio_value = Decimal(
            config.execution.get("default_portfolio_value", "100000")
        )

        self._logger.info("TWAP algorithm initialized with default parameters")

    def get_algorithm_type(self) -> ExecutionAlgorithm:
        """Get the algorithm type enum."""
        return ExecutionAlgorithm.TWAP

    async def _validate_algorithm_parameters(self, instruction: ExecutionInstruction) -> None:
        """
        Validate TWAP-specific parameters.

        Args:
            instruction: Execution instruction to validate

        Raises:
            ValidationError: If parameters are invalid
        """
        # Validate time horizon
        if instruction.time_horizon_minutes is not None:
            if instruction.time_horizon_minutes <= 0:
                raise ValidationError("Time horizon must be positive")
            if instruction.time_horizon_minutes > 24 * 60:  # 24 hours
                raise ValidationError("Time horizon cannot exceed 24 hours")

        # Validate participation rate
        if instruction.participation_rate is not None:
            if not 0.0 < instruction.participation_rate <= 1.0:
                raise ValidationError("Participation rate must be between 0 and 1")

        # Validate slice parameters
        if instruction.max_slices is not None:
            if instruction.max_slices <= 0:
                raise ValidationError("Max slices must be positive")
            if instruction.max_slices > self.max_slices:
                raise ValidationError(f"Max slices cannot exceed {self.max_slices}")

        if instruction.slice_size is not None:
            if instruction.slice_size <= 0:
                raise ValidationError("Slice size must be positive")
            if instruction.slice_size > instruction.order.quantity:
                raise ValidationError("Slice size cannot exceed order quantity")

    @time_execution
    @log_calls
    async def execute(
        self,
        instruction: ExecutionInstruction,
        exchange_factory: ExchangeFactoryInterface | None = None,
        risk_manager=None,
    ) -> ExecutionResult:
        """
        Execute an order using TWAP algorithm.

        Args:
            instruction: Execution instruction with TWAP parameters
            exchange_factory: Factory for creating exchange instances
            risk_manager: Risk manager for order validation

        Returns:
            ExecutionResult: Result of the TWAP execution

        Raises:
            ExecutionError: If execution fails
            ValidationError: If instruction is invalid
        """
        try:
            # Validate instruction
            await self.validate_instruction(instruction)

            # Create execution result for tracking (this also registers it)
            execution_result = await self._create_execution_result(instruction)
            execution_id = execution_result.execution_id
            # is_running is managed by BaseComponent/BaseAlgorithm
            execution_result.status = ExecutionStatus.RUNNING

            self._logger.info(
                "Starting TWAP execution",
                execution_id=execution_id,
                symbol=instruction.order.symbol,
                quantity=str(instruction.order.quantity),
                time_horizon=instruction.time_horizon_minutes,
            )

            # Get exchange for execution
            self._validate_exchange_factory(exchange_factory)
            exchange = await self._get_exchange_from_factory(exchange_factory, instruction)

            # Calculate TWAP execution plan
            execution_plan = self._create_execution_plan(instruction)

            # Execute TWAP strategy
            await self._execute_twap_plan(execution_plan, execution_result, exchange, risk_manager)

            # Finalize execution state (not result)
            execution_state = self.current_executions.get(execution_id)
            if execution_state:
                await self._finalize_execution(execution_state)
                # Get updated result
                execution_result = self._state_to_result(execution_state)

            # Update statistics
            self._update_execution_statistics(execution_result.status)

            self._logger.info(
                "TWAP execution completed",
                execution_id=execution_id,
                status=execution_result.status.value,
                filled_quantity=str(execution_result.total_filled_quantity),
                number_of_trades=execution_result.number_of_trades,
            )

            return execution_result

        except Exception as e:
            # Handle execution failure
            execution_id_for_error = execution_id if "execution_id" in locals() else None
            await self._handle_execution_error(e, execution_id_for_error, "TWAP")
            raise ExecutionError(f"TWAP execution failed: {e}")

        finally:
            # is_running is managed by BaseComponent/BaseAlgorithm
            pass

    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an ongoing TWAP execution.

        Args:
            execution_id: ID of the execution to cancel

        Returns:
            bool: True if cancellation successful, False otherwise
        """
        return await self._standard_cancel_execution(execution_id, "TWAP")

    def _create_execution_plan(self, instruction: ExecutionInstruction) -> dict[str, Any]:
        """
        Create detailed execution plan for TWAP strategy.

        Args:
            instruction: Execution instruction

        Returns:
            dict: TWAP execution plan with timing and sizing details
        """
        # Determine time horizon
        time_horizon_minutes = instruction.time_horizon_minutes or self.default_time_horizon_minutes

        # Determine participation rate
        participation_rate = instruction.participation_rate or self.default_participation_rate

        # Calculate number of slices
        if instruction.max_slices:
            num_slices = min(instruction.max_slices, self.max_slices)
        else:
            # Default: one slice per 5 minutes, up to max_slices
            num_slices = min(time_horizon_minutes // 5, self.max_slices)
            num_slices = max(num_slices, 1)  # At least 1 slice

        # Calculate slice size
        if instruction.slice_size:
            slice_size = instruction.slice_size
            # Adjust number of slices based on fixed slice size
            num_slices = int(instruction.order.quantity / slice_size)
            if instruction.order.quantity % slice_size > 0:
                num_slices += 1
        else:
            slice_size = instruction.order.quantity / num_slices

        # Ensure minimum slice size
        min_slice_size = instruction.order.quantity * Decimal(str(self.min_slice_size_pct))
        if slice_size < min_slice_size:
            slice_size = min_slice_size
            num_slices = int(instruction.order.quantity / slice_size)
            if instruction.order.quantity % slice_size > 0:
                num_slices += 1

        # Calculate time intervals
        slice_interval_seconds = (time_horizon_minutes * 60) / num_slices

        # Create slice schedule
        slices = []
        remaining_quantity = instruction.order.quantity
        start_time = instruction.start_time or datetime.now(timezone.utc)

        for i in range(num_slices):
            # Calculate slice quantity (last slice gets remainder)
            if i == num_slices - 1:
                current_slice_size = remaining_quantity
            else:
                current_slice_size = min(slice_size, remaining_quantity)

            # Calculate execution time for this slice
            execution_time = start_time + timedelta(seconds=i * slice_interval_seconds)

            slices.append(
                {
                    "slice_number": i + 1,
                    "quantity": current_slice_size,
                    "execution_time": execution_time,
                    "status": "pending",
                }
            )

            remaining_quantity -= current_slice_size

            if remaining_quantity <= 0:
                break

        execution_plan = {
            "total_quantity": instruction.order.quantity,
            "num_slices": len(slices),
            "slice_size": slice_size,
            "time_horizon_minutes": time_horizon_minutes,
            "slice_interval_seconds": slice_interval_seconds,
            "participation_rate": participation_rate,
            "slices": slices,
            "start_time": start_time,
            "end_time": start_time + timedelta(minutes=time_horizon_minutes),
        }

        self._logger.debug(
            "TWAP execution plan created",
            num_slices=len(slices),
            slice_size=str(slice_size),
            time_horizon=time_horizon_minutes,
        )

        return execution_plan

    async def _execute_twap_plan(
        self,
        execution_plan: dict[str, Any],
        execution_result: ExecutionResult,
        exchange,
        risk_manager,
    ) -> None:
        """
        Execute the TWAP plan by placing orders according to the schedule.

        Args:
            execution_plan: TWAP execution plan
            execution_result: Execution result to update
            exchange: Exchange for order placement
            risk_manager: Risk manager for validation
        """
        try:
            for slice_info in execution_plan["slices"]:
                # Check if execution was cancelled
                if execution_result.status == ExecutionStatus.CANCELLED:
                    self._logger.info("TWAP execution cancelled, stopping slice execution")
                    break

                # Wait until it's time for this slice
                current_time = datetime.now(timezone.utc)
                execution_time = slice_info["execution_time"]

                if current_time < execution_time:
                    wait_seconds = (execution_time - current_time).total_seconds()
                    if wait_seconds > 0:
                        self._logger.debug(
                            f"Waiting {wait_seconds:.1f} seconds for next slice",
                            slice_number=slice_info["slice_number"],
                        )
                        await asyncio.sleep(
                            min(wait_seconds, self.config.execution.twap_max_wait_seconds)
                        )

                # Create order for this slice
                slice_order = OrderRequest(
                    symbol=execution_result.original_order.symbol,
                    side=execution_result.original_order.side,
                    order_type=OrderType.MARKET,  # TWAP typically uses market orders
                    quantity=slice_info["quantity"],
                    price=execution_result.original_order.price,  # May be None for market orders
                    client_order_id=f"{execution_result.execution_id}_slice_{slice_info['slice_number']}",
                )

                # Validate order with risk manager if provided
                if risk_manager:
                    portfolio_value = self.default_portfolio_value
                    try:
                        is_valid = await risk_manager.validate_order(slice_order, portfolio_value)
                        if not is_valid:
                            self._logger.warning(
                                "Risk manager rejected slice order",
                                slice_number=slice_info["slice_number"],
                            )
                            continue
                    except Exception as e:
                        self._logger.warning(
                            f"Risk validation failed for slice {slice_info['slice_number']}: {e}"
                        )
                        continue

                # Place the slice order
                try:
                    slice_info["status"] = "executing"

                    # Place order with error handling
                    try:
                        order_response = await exchange.place_order(slice_order)

                        # Validate order response
                        if not order_response or not hasattr(order_response, "id"):
                            raise ExecutionError("Invalid order response from exchange")

                    except ExchangeError as e:
                        self._logger.error(
                            f"Exchange error placing slice {slice_info['slice_number']}: {e}",
                            slice_num=slice_info["slice_number"],
                            symbol=slice_order.symbol,
                            quantity=str(slice_order.quantity),
                        )
                        # Continue with next slice, don't fail entire execution
                        execution_result.add_fill(
                            price=slice_order.price or Decimal("0"),
                            quantity=Decimal("0"),
                            timestamp=datetime.now(timezone.utc),
                            order_id=f"failed_slice_{slice_info['slice_number']}",
                        )
                        await asyncio.sleep(self.slice_interval_buffer)
                        continue
                    except NetworkError as e:
                        self._logger.error(
                            f"Network error placing slice {slice_info['slice_number']}: {e}",
                            slice_num=slice_info["slice_number"],
                        )
                        # For network errors, might want to retry or abort
                        raise ExecutionError(f"Network error during TWAP execution: {e}")

                    # Update execution result with new child order
                    await self._update_execution_result(
                        execution_result, child_order=order_response
                    )

                    slice_info["status"] = "completed"
                    slice_info["order_response"] = order_response

                    self._logger.info(
                        "TWAP slice executed",
                        slice_number=slice_info["slice_number"],
                        quantity=str(slice_info["quantity"]),
                        filled=str(order_response.filled_quantity),
                        order_id=order_response.id,
                    )

                    # Add small buffer between slices to avoid overwhelming the market
                    if slice_info["slice_number"] < execution_plan["num_slices"]:
                        await asyncio.sleep(self.slice_interval_buffer)

                except Exception as e:
                    slice_info["status"] = "failed"
                    slice_info["error"] = str(e)

                    self._logger.error(
                        "TWAP slice execution failed",
                        slice_number=slice_info["slice_number"],
                        error=str(e),
                    )

                    # Continue with next slice unless it's a critical error
                    if "rate limit" not in str(e).lower():
                        continue
                    else:
                        # For rate limit errors, wait longer before next slice
                        await asyncio.sleep(self.config.execution.twap_error_recovery_delay_seconds)

        except Exception as e:
            self._logger.error(f"TWAP plan execution failed: {e}")
            raise ExecutionError(f"TWAP plan execution failed: {e}")

    async def _finalize_execution(self, execution_state: ExecutionState) -> None:
        """
        Finalize the TWAP execution state with final calculations.

        Args:
            execution_state: Execution state to finalize
        """
        try:
            # Determine final status
            if execution_state.status == ExecutionStatus.CANCELLED:
                # Already set to cancelled
                pass
            elif execution_state.total_filled_quantity >= execution_state.original_order.quantity:
                execution_state.status = ExecutionStatus.COMPLETED
            elif execution_state.total_filled_quantity > 0:
                execution_state.status = ExecutionStatus.PARTIALLY_FILLED
            else:
                execution_state.status = ExecutionStatus.FAILED
                execution_state.error_message = "No fills received"

            # Calculate final metrics
            if execution_state.total_filled_quantity > 0:
                # Calculate participation rate
                total_time_seconds = (
                    (execution_state.end_time - execution_state.start_time).total_seconds()
                    if execution_state.end_time
                    else 0
                )

                if total_time_seconds > 0:
                    # Estimate participation rate (simplified)
                    estimated_market_volume = self.config.execution.get(
                        "default_daily_volume", "1000000"
                    )
                    participation_rate = execution_state.total_filled_quantity / Decimal(
                        estimated_market_volume
                    )
                    # Store in metadata
                    execution_state.metadata["participation_rate"] = participation_rate

                # Calculate slippage metrics
                await self._calculate_slippage_metrics(
                    execution_state, expected_price=execution_state.original_order.price
                )

            self._logger.debug(
                "TWAP execution finalized",
                execution_id=execution_state.execution_id,
                final_status=execution_state.status.value,
                fill_rate=execution_state.total_filled_quantity
                / execution_state.original_order.quantity,
            )

        except (ExecutionError, ServiceError) as e:
            self._logger.error(f"Failed to finalize TWAP execution: {e}")
            execution_state.error_message = f"Finalization failed: {e}"
        except Exception as e:
            self._logger.error(f"Unexpected error finalizing TWAP execution: {e}")
            execution_state.error_message = f"Unexpected finalization error: {e}"
            # Don't re-raise during finalization to allow graceful cleanup
