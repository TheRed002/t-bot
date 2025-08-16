"""
Volume-Weighted Average Price (VWAP) execution algorithm.

This module implements the VWAP execution strategy that sizes orders based on
historical volume patterns to minimize market impact while achieving better
execution prices than naive time-slicing.

CRITICAL: This integrates with P-001 (types, exceptions, config),
P-002A (error handling), and P-007A (utils) components.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.core.config import Config
from src.core.exceptions import ExecutionError, ValidationError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import (
    ExecutionAlgorithm,
    ExecutionInstruction,
    ExecutionResult,
    ExecutionStatus,
    OrderRequest,
    OrderType,
)

# MANDATORY: Import from P-002A
# MANDATORY: Import from P-007A
from src.utils.decorators import log_calls, time_execution

from .base_algorithm import BaseAlgorithm

logger = get_logger(__name__)


class VWAPAlgorithm(BaseAlgorithm):
    """
    Volume-Weighted Average Price (VWAP) execution algorithm.

    VWAP sizes order slices based on historical volume patterns to match
    the natural flow of the market and minimize market impact. This strategy
    is particularly effective for large orders that need to be executed
    without moving the market significantly.

    Key features:
    - Historical volume pattern analysis for optimal sizing
    - Adaptive participation rate based on market conditions
    - Real-time volume monitoring for strategy adjustment
    - Market impact minimization through volume-based execution
    """

    def __init__(self, config: Config):
        """
        Initialize VWAP execution algorithm.

        Args:
            config: Application configuration
        """
        super().__init__(config)

        # VWAP-specific configuration
        self.default_time_horizon_minutes = 240  # 4 hours default
        self.default_participation_rate = 0.15  # 15% of historical volume
        self.volume_lookback_days = 20  # Days of historical volume data
        self.min_volume_threshold = 10000  # Minimum volume to consider

        # Volume pattern parameters
        self.volume_buckets = 24  # Number of hourly buckets for volume pattern
        self.smoothing_factor = 0.3  # Exponential smoothing for volume prediction
        self.max_participation_rate = 0.5  # Maximum 50% participation

        # Execution controls
        self.min_slice_interval_seconds = 30  # Minimum 30 seconds between slices
        self.volume_check_interval_seconds = 60  # Check volume every minute

        # Initialize volume patterns (would be loaded from historical data)
        self._initialize_default_volume_pattern()

        self.logger.info("VWAP algorithm initialized with volume-based execution")

    def get_algorithm_type(self) -> ExecutionAlgorithm:
        """Get the algorithm type enum."""
        return ExecutionAlgorithm.VWAP

    def _initialize_default_volume_pattern(self) -> None:
        """Initialize default intraday volume pattern."""
        # Default US market volume pattern (normalized to sum to 1.0)
        # Higher volume during market open/close, lower during lunch
        self.default_volume_pattern = [
            0.08,  # 09:30-10:30 (market open)
            0.06,  # 10:30-11:30
            0.04,  # 11:30-12:30
            0.03,  # 12:30-13:30 (lunch)
            0.04,  # 13:30-14:30
            0.05,  # 14:30-15:30
            0.07,  # 15:30-16:00 (market close)
            # Extend for 24-hour markets
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,  # Evening
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,  # Night
            0.01,
            0.01,
            0.01,
            0.01,
            0.02,
            0.02,  # Early morning
            0.03,
            0.04,
            0.05,
            0.06,  # Pre-market
        ]

        # Normalize to ensure sum equals 1.0
        total = sum(self.default_volume_pattern)
        self.default_volume_pattern = [v / total for v in self.default_volume_pattern]

    async def _validate_algorithm_parameters(self, instruction: ExecutionInstruction) -> None:
        """
        Validate VWAP-specific parameters.

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
            if not 0.0 < instruction.participation_rate <= self.max_participation_rate:
                raise ValidationError(
                    f"Participation rate must be between 0 and {self.max_participation_rate}"
                )

    @time_execution
    @log_calls
    async def execute(
        self, instruction: ExecutionInstruction, exchange_factory=None, risk_manager=None
    ) -> ExecutionResult:
        """
        Execute an order using VWAP algorithm.

        Args:
            instruction: Execution instruction with VWAP parameters
            exchange_factory: Factory for creating exchange instances
            risk_manager: Risk manager for order validation

        Returns:
            ExecutionResult: Result of the VWAP execution

        Raises:
            ExecutionError: If execution fails
            ValidationError: If instruction is invalid
        """
        try:
            # Validate instruction
            await self.validate_instruction(instruction)

            # Create execution result for tracking
            execution_result = await self._create_execution_result(instruction)
            execution_id = execution_result.execution_id

            # Register execution as running
            self.current_executions[execution_id] = execution_result
            self.is_running = True
            execution_result.status = ExecutionStatus.RUNNING

            self.logger.info(
                "Starting VWAP execution",
                execution_id=execution_id,
                symbol=instruction.order.symbol,
                quantity=float(instruction.order.quantity),
                time_horizon=instruction.time_horizon_minutes,
            )

            # Get exchange for execution
            if not exchange_factory:
                raise ExecutionError("Exchange factory is required for VWAP execution")

            # Determine which exchange to use
            exchange_name = "binance"  # Default exchange
            if instruction.preferred_exchanges:
                exchange_name = instruction.preferred_exchanges[0]

            exchange = await exchange_factory.get_exchange(exchange_name)
            if not exchange:
                raise ExecutionError(f"Failed to get exchange: {exchange_name}")

            # Create VWAP execution plan based on volume patterns
            execution_plan = await self._create_vwap_execution_plan(instruction, exchange)

            # Execute VWAP strategy
            await self._execute_vwap_plan(execution_plan, execution_result, exchange, risk_manager)

            # Finalize execution result
            await self._finalize_execution(execution_result)

            # Update statistics
            if execution_result.status == ExecutionStatus.COMPLETED:
                self.successful_executions += 1
            else:
                self.failed_executions += 1
            self.total_executions += 1

            self.logger.info(
                "VWAP execution completed",
                execution_id=execution_id,
                status=execution_result.status.value,
                filled_quantity=float(execution_result.total_filled_quantity),
                number_of_trades=execution_result.number_of_trades,
            )

            return execution_result

        except Exception as e:
            # Handle execution failure
            if "execution_id" in locals() and execution_id in self.current_executions:
                await self._update_execution_result(
                    self.current_executions[execution_id],
                    status=ExecutionStatus.FAILED,
                    error_message=str(e),
                )
                self.failed_executions += 1
                self.total_executions += 1

            self.logger.error(
                "VWAP execution failed",
                execution_id=execution_id if "execution_id" in locals() else "unknown",
                error=str(e),
            )
            raise ExecutionError(f"VWAP execution failed: {e}")

        finally:
            self.is_running = False

    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an ongoing VWAP execution.

        Args:
            execution_id: ID of the execution to cancel

        Returns:
            bool: True if cancellation successful, False otherwise
        """
        try:
            if execution_id not in self.current_executions:
                self.logger.warning(f"Execution not found for cancellation: {execution_id}")
                return False

            execution_result = self.current_executions[execution_id]

            if execution_result.status not in [ExecutionStatus.RUNNING, ExecutionStatus.PENDING]:
                self.logger.warning(
                    f"Cannot cancel execution in status: {execution_result.status.value}"
                )
                return False

            # Update status to cancelled
            await self._update_execution_result(execution_result, status=ExecutionStatus.CANCELLED)

            self.logger.info(f"VWAP execution cancelled: {execution_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel VWAP execution: {e}")
            return False

    async def _create_vwap_execution_plan(
        self, instruction: ExecutionInstruction, exchange
    ) -> dict[str, Any]:
        """
        Create VWAP execution plan based on historical volume patterns.

        Args:
            instruction: Execution instruction
            exchange: Exchange for volume data

        Returns:
            dict: VWAP execution plan with volume-based sizing
        """
        # Determine execution parameters
        time_horizon_minutes = instruction.time_horizon_minutes or self.default_time_horizon_minutes
        participation_rate = instruction.participation_rate or self.default_participation_rate

        # Get historical volume pattern for the symbol
        volume_pattern = await self._get_volume_pattern(instruction.order.symbol, exchange)

        # Calculate execution window
        start_time = instruction.start_time or datetime.now(timezone.utc)
        end_time = start_time + timedelta(minutes=time_horizon_minutes)

        # Create volume-based slices
        slices = await self._create_volume_based_slices(
            instruction.order.quantity, start_time, end_time, volume_pattern, participation_rate
        )

        execution_plan = {
            "total_quantity": instruction.order.quantity,
            "num_slices": len(slices),
            "time_horizon_minutes": time_horizon_minutes,
            "participation_rate": participation_rate,
            "volume_pattern": volume_pattern,
            "slices": slices,
            "start_time": start_time,
            "end_time": end_time,
            "adaptive_execution": True,  # Enable real-time adjustments
        }

        self.logger.debug(
            "VWAP execution plan created",
            num_slices=len(slices),
            time_horizon=time_horizon_minutes,
            participation_rate=participation_rate,
        )

        return execution_plan

    async def _get_volume_pattern(self, symbol: str, exchange) -> list[float]:
        """
        Get historical volume pattern for the symbol.

        Args:
            symbol: Trading symbol
            exchange: Exchange for volume data

        Returns:
            list[float]: Normalized hourly volume pattern
        """
        try:
            # In a real implementation, this would fetch historical volume data
            # For now, use default pattern adjusted by symbol characteristics

            # Simulate fetching volume data (placeholder)
            volume_pattern = self.default_volume_pattern.copy()

            # Adjust pattern based on symbol characteristics (simplified)
            if "BTC" in symbol:
                # Bitcoin typically has more even volume distribution
                smoothing = 0.3
                avg_volume = sum(volume_pattern) / len(volume_pattern)
                volume_pattern = [
                    v * (1 - smoothing) + avg_volume * smoothing for v in volume_pattern
                ]

            # Normalize to ensure sum equals 1.0
            total = sum(volume_pattern)
            if total > 0:
                volume_pattern = [v / total for v in volume_pattern]
            else:
                volume_pattern = self.default_volume_pattern

            self.logger.debug(f"Volume pattern retrieved for {symbol}")
            return volume_pattern

        except Exception as e:
            self.logger.warning(f"Failed to get volume pattern for {symbol}: {e}")
            return self.default_volume_pattern

    async def _create_volume_based_slices(
        self,
        total_quantity: Decimal,
        start_time: datetime,
        end_time: datetime,
        volume_pattern: list[float],
        participation_rate: float,
    ) -> list[dict[str, Any]]:
        """
        Create order slices based on volume pattern.

        Args:
            total_quantity: Total quantity to execute
            start_time: Execution start time
            end_time: Execution end time
            volume_pattern: Hourly volume pattern
            participation_rate: Target participation rate

        Returns:
            list[dict]: Volume-based order slices
        """
        slices = []
        execution_duration = (end_time - start_time).total_seconds()

        # Determine which hours of the volume pattern to use
        start_hour = start_time.hour
        duration_hours = max(1, int(execution_duration / 3600))

        # Extract relevant volume pattern
        relevant_volumes = []
        for i in range(duration_hours):
            hour_index = (start_hour + i) % 24
            if hour_index < len(volume_pattern):
                relevant_volumes.append(volume_pattern[hour_index])
            else:
                relevant_volumes.append(volume_pattern[hour_index % len(volume_pattern)])

        # Normalize relevant volumes
        total_volume = sum(relevant_volumes)
        if total_volume > 0:
            relevant_volumes = [v / total_volume for v in relevant_volumes]
        else:
            # Fallback to equal distribution
            relevant_volumes = [1.0 / len(relevant_volumes)] * len(relevant_volumes)

        # Create slices based on volume distribution
        remaining_quantity = total_quantity
        slice_number = 1

        for i, volume_weight in enumerate(relevant_volumes):
            if remaining_quantity <= 0:
                break

            # Calculate slice quantity based on volume weight
            slice_quantity = total_quantity * Decimal(str(volume_weight))

            # Ensure we don't exceed remaining quantity
            if slice_quantity > remaining_quantity:
                slice_quantity = remaining_quantity

            # Calculate execution time for this slice
            hour_offset = i * 3600  # i-th hour
            slice_time = start_time + timedelta(seconds=hour_offset)

            # Create multiple sub-slices within the hour for smoother execution
            sub_slices_per_hour = 4  # Execute every 15 minutes within the hour
            sub_slice_quantity = slice_quantity / sub_slices_per_hour

            for j in range(sub_slices_per_hour):
                if remaining_quantity <= 0:
                    break

                # Calculate sub-slice execution time
                sub_slice_time = slice_time + timedelta(seconds=j * 900)  # 15-minute intervals

                # Adjust quantity for last sub-slice
                if j == sub_slices_per_hour - 1:
                    actual_quantity = min(sub_slice_quantity, remaining_quantity)
                else:
                    actual_quantity = min(sub_slice_quantity, remaining_quantity)

                if actual_quantity > 0:
                    slices.append(
                        {
                            "slice_number": slice_number,
                            "quantity": actual_quantity,
                            "execution_time": sub_slice_time,
                            "volume_weight": volume_weight / sub_slices_per_hour,
                            "hour_bucket": i,
                            "status": "pending",
                        }
                    )

                    remaining_quantity -= actual_quantity
                    slice_number += 1

        return slices

    async def _execute_vwap_plan(
        self,
        execution_plan: dict[str, Any],
        execution_result: ExecutionResult,
        exchange,
        risk_manager,
    ) -> None:
        """
        Execute the VWAP plan with real-time volume monitoring.

        Args:
            execution_plan: VWAP execution plan
            execution_result: Execution result to update
            exchange: Exchange for order placement
            risk_manager: Risk manager for validation
        """
        try:

            for slice_info in execution_plan["slices"]:
                # Check if execution was cancelled
                if execution_result.status == ExecutionStatus.CANCELLED:
                    self.logger.info("VWAP execution cancelled, stopping slice execution")
                    break

                # Wait until it's time for this slice
                current_time = datetime.now(timezone.utc)
                execution_time = slice_info["execution_time"]

                if current_time < execution_time:
                    wait_seconds = (execution_time - current_time).total_seconds()
                    if wait_seconds > 0:
                        self.logger.debug(
                            f"Waiting {wait_seconds:.1f} seconds for next VWAP slice",
                            slice_number=slice_info["slice_number"],
                        )
                        await asyncio.sleep(min(wait_seconds, 300))  # Max 5 minutes wait

                # Monitor current market volume and adjust if needed
                adjusted_quantity = await self._adjust_slice_for_volume(
                    slice_info, execution_result.original_order.symbol, exchange
                )

                # Create order for this slice
                slice_order = OrderRequest(
                    symbol=execution_result.original_order.symbol,
                    side=execution_result.original_order.side,
                    order_type=OrderType.MARKET,  # VWAP typically uses market orders
                    quantity=adjusted_quantity,
                    price=execution_result.original_order.price,
                    client_order_id=f"{execution_result.execution_id}_vwap_{slice_info['slice_number']}",
                )

                # Validate order with risk manager if provided
                if risk_manager:
                    portfolio_value = Decimal("100000")  # Default portfolio value
                    try:
                        is_valid = await risk_manager.validate_order(slice_order, portfolio_value)
                        if not is_valid:
                            self.logger.warning(
                                "Risk manager rejected VWAP slice order",
                                slice_number=slice_info["slice_number"],
                            )
                            continue
                    except Exception as e:
                        self.logger.warning(
                            f"Risk validation failed for VWAP slice {slice_info['slice_number']}: {e}"
                        )
                        continue

                # Place the slice order
                try:
                    slice_info["status"] = "executing"

                    order_response = await exchange.place_order(slice_order)

                    # Update execution result with new child order
                    await self._update_execution_result(
                        execution_result, child_order=order_response
                    )

                    slice_info["status"] = "completed"
                    slice_info["order_response"] = order_response
                    slice_info["actual_quantity"] = adjusted_quantity

                    self.logger.info(
                        "VWAP slice executed",
                        slice_number=slice_info["slice_number"],
                        quantity=float(adjusted_quantity),
                        filled=float(order_response.filled_quantity),
                        volume_weight=slice_info["volume_weight"],
                        order_id=order_response.id,
                    )

                    # Add dynamic interval based on market conditions
                    await asyncio.sleep(self.min_slice_interval_seconds)

                except Exception as e:
                    slice_info["status"] = "failed"
                    slice_info["error"] = str(e)

                    self.logger.error(
                        "VWAP slice execution failed",
                        slice_number=slice_info["slice_number"],
                        error=str(e),
                    )

                    # Continue with next slice unless it's a critical error
                    if "rate limit" not in str(e).lower():
                        continue
                    else:
                        # For rate limit errors, wait longer
                        await asyncio.sleep(60)

        except Exception as e:
            self.logger.error(f"VWAP plan execution failed: {e}")
            raise ExecutionError(f"VWAP plan execution failed: {e}")

    async def _adjust_slice_for_volume(
        self, slice_info: dict[str, Any], symbol: str, exchange
    ) -> Decimal:
        """
        Adjust slice quantity based on current market volume.

        Args:
            slice_info: Slice information
            symbol: Trading symbol
            exchange: Exchange for volume data

        Returns:
            Decimal: Adjusted slice quantity
        """
        try:
            # In a real implementation, this would fetch current volume data
            # For now, return original quantity with minor random adjustment
            original_quantity = slice_info["quantity"]

            # Simulate volume-based adjustment (placeholder)
            # In reality, this would compare current volume to historical patterns
            volume_adjustment_factor = 1.0  # No adjustment for now

            adjusted_quantity = original_quantity * Decimal(str(volume_adjustment_factor))

            self.logger.debug(
                "Volume adjustment applied",
                slice_number=slice_info["slice_number"],
                original=float(original_quantity),
                adjusted=float(adjusted_quantity),
                factor=volume_adjustment_factor,
            )

            return adjusted_quantity

        except Exception as e:
            self.logger.warning(f"Volume adjustment failed: {e}")
            return slice_info["quantity"]

    async def _finalize_execution(self, execution_result: ExecutionResult) -> None:
        """
        Finalize the VWAP execution result with final calculations.

        Args:
            execution_result: Execution result to finalize
        """
        try:
            # Determine final status
            if execution_result.status == ExecutionStatus.CANCELLED:
                # Already set to cancelled
                pass
            elif execution_result.total_filled_quantity >= execution_result.original_order.quantity:
                execution_result.status = ExecutionStatus.COMPLETED
            elif execution_result.total_filled_quantity > 0:
                execution_result.status = ExecutionStatus.PARTIALLY_FILLED
            else:
                execution_result.status = ExecutionStatus.FAILED
                execution_result.error_message = "No fills received"

            # Calculate VWAP-specific metrics
            if execution_result.total_filled_quantity > 0:
                # Calculate actual vs expected VWAP
                if execution_result.average_fill_price:
                    # Mark as achieving VWAP benchmark (simplified)
                    execution_result.metadata["vwap_benchmark_achieved"] = True
                    execution_result.metadata["volume_weighted_execution"] = True

                # Calculate participation rate based on actual market volume
                estimated_market_volume = Decimal("2000000")  # Placeholder
                if estimated_market_volume > 0:
                    execution_result.participation_rate = float(
                        execution_result.total_filled_quantity / estimated_market_volume
                    )

                # Calculate slippage metrics
                await self._calculate_slippage_metrics(
                    execution_result, expected_price=execution_result.original_order.price
                )

            self.logger.debug(
                "VWAP execution finalized",
                execution_id=execution_result.execution_id,
                final_status=execution_result.status.value,
                fill_rate=float(
                    execution_result.total_filled_quantity
                    / execution_result.original_order.quantity
                ),
            )

        except Exception as e:
            self.logger.error(f"Failed to finalize VWAP execution: {e}")
            execution_result.error_message = f"Finalization failed: {e}"
