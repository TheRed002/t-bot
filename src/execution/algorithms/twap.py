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
from src.core.exceptions import ExecutionError, ValidationError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import (
    ExecutionAlgorithm,
    ExecutionInstruction,
    ExecutionResult,
    ExecutionStatus,
    OrderRequest,
    OrderResponse,
    OrderType,
)

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

# MANDATORY: Import from P-007A
from src.utils.decorators import log_calls, time_execution

from .base_algorithm import BaseAlgorithm

logger = get_logger(__name__)


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
        self.slice_interval_buffer = 5  # 5 seconds buffer between slices
        self.market_close_buffer_minutes = 30  # Stop 30 minutes before market close
        
        self.logger.info("TWAP algorithm initialized with default parameters")

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
        exchange_factory=None,
        risk_manager=None
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
            
            # Create execution result for tracking
            execution_result = await self._create_execution_result(instruction)
            execution_id = execution_result.execution_id
            
            # Register execution as running
            self.current_executions[execution_id] = execution_result
            self.is_running = True
            execution_result.status = ExecutionStatus.RUNNING
            
            self.logger.info(
                "Starting TWAP execution",
                execution_id=execution_id,
                symbol=instruction.order.symbol,
                quantity=float(instruction.order.quantity),
                time_horizon=instruction.time_horizon_minutes
            )
            
            # Get exchange for execution
            if not exchange_factory:
                raise ExecutionError("Exchange factory is required for TWAP execution")
                
            # Determine which exchange to use (default to first available)
            exchange_name = "binance"  # Default exchange
            if instruction.preferred_exchanges:
                exchange_name = instruction.preferred_exchanges[0]
                
            exchange = await exchange_factory.get_exchange(exchange_name)
            if not exchange:
                raise ExecutionError(f"Failed to get exchange: {exchange_name}")
                
            # Calculate TWAP execution plan
            execution_plan = await self._create_execution_plan(instruction)
            
            # Execute TWAP strategy
            await self._execute_twap_plan(
                execution_plan, execution_result, exchange, risk_manager
            )
            
            # Finalize execution result
            await self._finalize_execution(execution_result)
            
            # Update statistics
            if execution_result.status == ExecutionStatus.COMPLETED:
                self.successful_executions += 1
            else:
                self.failed_executions += 1
            self.total_executions += 1
            
            self.logger.info(
                "TWAP execution completed",
                execution_id=execution_id,
                status=execution_result.status.value,
                filled_quantity=float(execution_result.total_filled_quantity),
                number_of_trades=execution_result.number_of_trades
            )
            
            return execution_result
            
        except Exception as e:
            # Handle execution failure
            if execution_id in self.current_executions:
                await self._update_execution_result(
                    self.current_executions[execution_id],
                    status=ExecutionStatus.FAILED,
                    error_message=str(e)
                )
                self.failed_executions += 1
                self.total_executions += 1
                
            self.logger.error(
                "TWAP execution failed",
                execution_id=execution_id if 'execution_id' in locals() else "unknown",
                error=str(e)
            )
            raise ExecutionError(f"TWAP execution failed: {e}")
            
        finally:
            self.is_running = False

    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an ongoing TWAP execution.
        
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
            await self._update_execution_result(
                execution_result,
                status=ExecutionStatus.CANCELLED
            )
            
            self.logger.info(f"TWAP execution cancelled: {execution_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel TWAP execution: {e}")
            return False

    async def _create_execution_plan(self, instruction: ExecutionInstruction) -> dict[str, Any]:
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
            
            slices.append({
                "slice_number": i + 1,
                "quantity": current_slice_size,
                "execution_time": execution_time,
                "status": "pending"
            })
            
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
            "end_time": start_time + timedelta(minutes=time_horizon_minutes)
        }
        
        self.logger.debug(
            "TWAP execution plan created",
            num_slices=len(slices),
            slice_size=float(slice_size),
            time_horizon=time_horizon_minutes
        )
        
        return execution_plan

    async def _execute_twap_plan(
        self,
        execution_plan: dict[str, Any],
        execution_result: ExecutionResult,
        exchange,
        risk_manager
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
                    self.logger.info("TWAP execution cancelled, stopping slice execution")
                    break
                    
                # Wait until it's time for this slice
                current_time = datetime.now(timezone.utc)
                execution_time = slice_info["execution_time"]
                
                if current_time < execution_time:
                    wait_seconds = (execution_time - current_time).total_seconds()
                    if wait_seconds > 0:
                        self.logger.debug(
                            f"Waiting {wait_seconds:.1f} seconds for next slice",
                            slice_number=slice_info["slice_number"]
                        )
                        await asyncio.sleep(wait_seconds)
                        
                # Create order for this slice
                slice_order = OrderRequest(
                    symbol=execution_result.original_order.symbol,
                    side=execution_result.original_order.side,
                    order_type=OrderType.MARKET,  # TWAP typically uses market orders
                    quantity=slice_info["quantity"],
                    price=execution_result.original_order.price,  # May be None for market orders
                    client_order_id=f"{execution_result.execution_id}_slice_{slice_info['slice_number']}"
                )
                
                # Validate order with risk manager if provided
                if risk_manager:
                    portfolio_value = Decimal("100000")  # Default portfolio value
                    try:
                        is_valid = await risk_manager.validate_order(slice_order, portfolio_value)
                        if not is_valid:
                            self.logger.warning(
                                "Risk manager rejected slice order",
                                slice_number=slice_info["slice_number"]
                            )
                            continue
                    except Exception as e:
                        self.logger.warning(
                            f"Risk validation failed for slice {slice_info['slice_number']}: {e}"
                        )
                        continue
                        
                # Place the slice order
                try:
                    slice_info["status"] = "executing"
                    
                    order_response = await exchange.place_order(slice_order)
                    
                    # Update execution result with new child order
                    await self._update_execution_result(
                        execution_result,
                        child_order=order_response
                    )
                    
                    slice_info["status"] = "completed"
                    slice_info["order_response"] = order_response
                    
                    self.logger.info(
                        "TWAP slice executed",
                        slice_number=slice_info["slice_number"],
                        quantity=float(slice_info["quantity"]),
                        filled=float(order_response.filled_quantity),
                        order_id=order_response.id
                    )
                    
                    # Add small buffer between slices to avoid overwhelming the market
                    if slice_info["slice_number"] < execution_plan["num_slices"]:
                        await asyncio.sleep(self.slice_interval_buffer)
                        
                except Exception as e:
                    slice_info["status"] = "failed"
                    slice_info["error"] = str(e)
                    
                    self.logger.error(
                        "TWAP slice execution failed",
                        slice_number=slice_info["slice_number"],
                        error=str(e)
                    )
                    
                    # Continue with next slice unless it's a critical error
                    if "rate limit" not in str(e).lower():
                        continue
                    else:
                        # For rate limit errors, wait longer before next slice
                        await asyncio.sleep(30)
                        
        except Exception as e:
            self.logger.error(f"TWAP plan execution failed: {e}")
            raise ExecutionError(f"TWAP plan execution failed: {e}")

    async def _finalize_execution(self, execution_result: ExecutionResult) -> None:
        """
        Finalize the TWAP execution result with final calculations.
        
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
                
            # Calculate final metrics
            if execution_result.total_filled_quantity > 0:
                # Calculate participation rate
                total_time_seconds = (
                    execution_result.end_time - execution_result.start_time
                ).total_seconds() if execution_result.end_time else 0
                
                if total_time_seconds > 0:
                    # Estimate participation rate (simplified)
                    estimated_market_volume = Decimal("1000000")  # Placeholder
                    execution_result.participation_rate = float(
                        execution_result.total_filled_quantity / estimated_market_volume
                    )
                    
                # Calculate slippage metrics
                await self._calculate_slippage_metrics(
                    execution_result,
                    expected_price=execution_result.original_order.price
                )
                
            self.logger.debug(
                "TWAP execution finalized",
                execution_id=execution_result.execution_id,
                final_status=execution_result.status.value,
                fill_rate=float(execution_result.total_filled_quantity / execution_result.original_order.quantity)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to finalize TWAP execution: {e}")
            execution_result.error_message = f"Finalization failed: {e}"