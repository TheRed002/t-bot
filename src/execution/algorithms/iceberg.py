"""
Iceberg execution algorithm for T-Bot execution engine.

This module implements the Iceberg execution strategy that hides the full
order size by only showing a small portion (tip of the iceberg) to the market,
automatically refreshing as portions get filled.

CRITICAL: This integrates with P-001 (types, exceptions, config),
P-002A (error handling), and P-007A (utils) components.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

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
    OrderStatus,
    OrderType,
)

# MANDATORY: Import from P-002A
# MANDATORY: Import from P-007A
from src.utils.decorators import log_calls, time_execution

from .base_algorithm import BaseAlgorithm

logger = get_logger(__name__)


class IcebergAlgorithm(BaseAlgorithm):
    """
    Iceberg execution algorithm for stealth trading.

    The Iceberg algorithm conceals large order sizes by only displaying a small
    portion to the market at any given time. As the visible portion gets filled,
    it's automatically replenished from the hidden reserve, maintaining market
    anonymity and reducing market impact.

    Key features:
    - Configurable display quantity to control visibility
    - Automatic order refresh when visible portion is filled
    - Smart timing to avoid detection patterns
    - Price improvement opportunities through limit orders
    - Adaptive display sizing based on market conditions
    """

    def __init__(self, config: Config):
        """
        Initialize Iceberg execution algorithm.

        Args:
            config: Application configuration
        """
        super().__init__(config)

        # Iceberg-specific configuration
        self.default_display_quantity_pct = 0.1  # Show 10% by default
        self.min_display_quantity_pct = 0.01  # Minimum 1% display
        self.max_display_quantity_pct = 0.5  # Maximum 50% display

        # Refresh controls
        self.refresh_delay_seconds = 2  # Delay between refreshes
        self.random_delay_range = 3  # Random delay to avoid patterns
        self.max_refresh_attempts = 50  # Maximum refresh cycles

        # Price improvement
        self.price_improvement_enabled = True
        self.tick_improvement_attempts = 3  # Try to improve price by ticks
        self.price_staleness_seconds = 30  # Refresh price after 30 seconds

        # Monitoring
        self.fill_monitoring_interval = 1  # Check fills every second
        self.order_timeout_minutes = 60  # Cancel unfilled orders after 1 hour

        self.logger.info("Iceberg algorithm initialized with stealth execution")

    def get_algorithm_type(self) -> ExecutionAlgorithm:
        """Get the algorithm type enum."""
        return ExecutionAlgorithm.ICEBERG

    async def _validate_algorithm_parameters(self, instruction: ExecutionInstruction) -> None:
        """
        Validate Iceberg-specific parameters.

        Args:
            instruction: Execution instruction to validate

        Raises:
            ValidationError: If parameters are invalid
        """
        # Validate display quantity
        if instruction.display_quantity is not None:
            if instruction.display_quantity <= 0:
                raise ValidationError("Display quantity must be positive")
            if instruction.display_quantity > instruction.order.quantity:
                raise ValidationError("Display quantity cannot exceed order quantity")

            # Check display percentage
            display_pct = float(instruction.display_quantity / instruction.order.quantity)
            if display_pct < self.min_display_quantity_pct:
                raise ValidationError(
                    f"Display quantity too small (minimum {self.min_display_quantity_pct*100}%)"
                )
            if display_pct > self.max_display_quantity_pct:
                raise ValidationError(
                    f"Display quantity too large (maximum {self.max_display_quantity_pct*100}%)"
                )

        # Iceberg works best with limit orders
        if instruction.order.order_type == OrderType.MARKET:
            self.logger.warning("Iceberg algorithm is more effective with limit orders")

    @time_execution
    @log_calls
    async def execute(
        self, instruction: ExecutionInstruction, exchange_factory=None, risk_manager=None
    ) -> ExecutionResult:
        """
        Execute an order using Iceberg algorithm.

        Args:
            instruction: Execution instruction with Iceberg parameters
            exchange_factory: Factory for creating exchange instances
            risk_manager: Risk manager for order validation

        Returns:
            ExecutionResult: Result of the Iceberg execution

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

            # Calculate display quantity
            display_quantity = await self._calculate_display_quantity(instruction)

            self.logger.info(
                "Starting Iceberg execution",
                execution_id=execution_id,
                symbol=instruction.order.symbol,
                total_quantity=float(instruction.order.quantity),
                display_quantity=float(display_quantity),
            )

            # Get exchange for execution
            if not exchange_factory:
                raise ExecutionError("Exchange factory is required for Iceberg execution")

            # Determine which exchange to use
            exchange_name = "binance"  # Default exchange
            if instruction.preferred_exchanges:
                exchange_name = instruction.preferred_exchanges[0]

            exchange = await exchange_factory.get_exchange(exchange_name)
            if not exchange:
                raise ExecutionError(f"Failed to get exchange: {exchange_name}")

            # Execute Iceberg strategy
            await self._execute_iceberg_strategy(
                instruction, execution_result, exchange, risk_manager, display_quantity
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
                "Iceberg execution completed",
                execution_id=execution_id,
                status=execution_result.status.value,
                filled_quantity=float(execution_result.total_filled_quantity),
                number_of_refreshes=execution_result.number_of_trades,
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
                "Iceberg execution failed",
                execution_id=execution_id if "execution_id" in locals() else "unknown",
                error=str(e),
            )
            raise ExecutionError(f"Iceberg execution failed: {e}")

        finally:
            self.is_running = False

    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an ongoing Iceberg execution.

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

            self.logger.info(f"Iceberg execution cancelled: {execution_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel Iceberg execution: {e}")
            return False

    async def _calculate_display_quantity(self, instruction: ExecutionInstruction) -> Decimal:
        """
        Calculate the display quantity for the iceberg order.

        Args:
            instruction: Execution instruction

        Returns:
            Decimal: Display quantity to show to the market
        """
        if instruction.display_quantity:
            return instruction.display_quantity

        # Calculate based on percentage
        display_pct = self.default_display_quantity_pct

        # Adjust based on order size (larger orders should show smaller percentage)
        order_value = instruction.order.quantity
        if order_value > Decimal("10000"):  # Large order
            display_pct = self.min_display_quantity_pct * 2
        elif order_value > Decimal("1000"):  # Medium order
            display_pct = self.default_display_quantity_pct * 0.5

        display_quantity = instruction.order.quantity * Decimal(str(display_pct))

        # Ensure minimum viable display size
        min_display = instruction.order.quantity * Decimal(str(self.min_display_quantity_pct))
        display_quantity = max(display_quantity, min_display)

        return display_quantity

    async def _execute_iceberg_strategy(
        self,
        instruction: ExecutionInstruction,
        execution_result: ExecutionResult,
        exchange,
        risk_manager,
        display_quantity: Decimal,
    ) -> None:
        """
        Execute the main Iceberg strategy with order refreshing.

        Args:
            instruction: Execution instruction
            execution_result: Execution result to update
            exchange: Exchange for order placement
            risk_manager: Risk manager for validation
            display_quantity: Quantity to display per order
        """
        try:
            remaining_quantity = instruction.order.quantity
            refresh_count = 0
            current_order_id = None
            last_price_update = datetime.now(timezone.utc)

            while remaining_quantity > 0 and refresh_count < self.max_refresh_attempts:
                # Check if execution was cancelled
                if execution_result.status == ExecutionStatus.CANCELLED:
                    if current_order_id:
                        try:
                            await exchange.cancel_order(current_order_id)
                        except Exception as e:
                            self.logger.warning(f"Failed to cancel order during cancellation: {e}")
                    break

                # Determine quantity for this slice
                slice_quantity = min(display_quantity, remaining_quantity)

                # Update price if needed (for limit orders)
                current_price = instruction.order.price
                if (
                    self.price_improvement_enabled
                    and instruction.order.order_type == OrderType.LIMIT
                    and (datetime.now(timezone.utc) - last_price_update).total_seconds()
                    > self.price_staleness_seconds
                ):

                    current_price = await self._get_improved_price(
                        instruction.order.symbol, instruction.order.side, exchange
                    )
                    last_price_update = datetime.now(timezone.utc)

                # Create order for this slice
                slice_order = OrderRequest(
                    symbol=instruction.order.symbol,
                    side=instruction.order.side,
                    order_type=instruction.order.order_type,
                    quantity=slice_quantity,
                    price=current_price,
                    client_order_id=f"{execution_result.execution_id}_ice_{refresh_count + 1}",
                )

                # Validate order with risk manager if provided
                if risk_manager:
                    portfolio_value = Decimal("100000")  # Default portfolio value
                    try:
                        is_valid = await risk_manager.validate_order(slice_order, portfolio_value)
                        if not is_valid:
                            self.logger.warning(
                                f"Risk manager rejected iceberg slice {refresh_count + 1}"
                            )
                            break
                    except Exception as e:
                        self.logger.warning(
                            f"Risk validation failed for iceberg slice {refresh_count + 1}: {e}"
                        )
                        break

                # Place the slice order
                try:
                    order_response = await exchange.place_order(slice_order)
                    current_order_id = order_response.id

                    self.logger.info(
                        "Iceberg slice placed",
                        refresh_count=refresh_count + 1,
                        slice_quantity=float(slice_quantity),
                        remaining=float(remaining_quantity),
                        order_id=order_response.id,
                    )

                    # Monitor the order for fills
                    filled_quantity = await self._monitor_order_fills(
                        order_response, exchange, execution_result
                    )

                    # Update remaining quantity
                    remaining_quantity -= filled_quantity
                    refresh_count += 1

                    # Add delay before next refresh to avoid detection
                    if remaining_quantity > 0:
                        delay = self.refresh_delay_seconds
                        # Add random component to avoid patterns
                        import random

                        delay += random.uniform(0, self.random_delay_range)
                        await asyncio.sleep(delay)

                except Exception as e:
                    self.logger.error(f"Failed to place iceberg slice {refresh_count + 1}: {e}")

                    # For certain errors, stop execution
                    if "insufficient" in str(e).lower() or "rejected" in str(e).lower():
                        break

                    # For others, try again after a delay
                    await asyncio.sleep(10)

            # Cancel any remaining open order
            if current_order_id:
                try:
                    await exchange.cancel_order(current_order_id)
                except Exception as e:
                    self.logger.warning(f"Failed to cancel final iceberg order: {e}")

            self.logger.info(
                "Iceberg strategy completed",
                total_refreshes=refresh_count,
                remaining_quantity=float(remaining_quantity),
            )

        except Exception as e:
            self.logger.error(f"Iceberg strategy execution failed: {e}")
            raise ExecutionError(f"Iceberg strategy failed: {e}")

    async def _monitor_order_fills(
        self, order_response: OrderResponse, exchange, execution_result: ExecutionResult
    ) -> Decimal:
        """
        Monitor an order for fills and update execution result.

        Args:
            order_response: Initial order response
            exchange: Exchange for status checking
            execution_result: Execution result to update

        Returns:
            Decimal: Total filled quantity for this order
        """
        try:
            order_id = order_response.id
            total_filled = Decimal("0")
            monitoring_start = datetime.now(timezone.utc)
            timeout = timedelta(minutes=self.order_timeout_minutes)

            while datetime.now(timezone.utc) - monitoring_start < timeout:
                # Check if execution was cancelled
                if execution_result.status == ExecutionStatus.CANCELLED:
                    break

                try:
                    # Get current order status
                    order_status = await exchange.get_order_status(order_id)

                    if order_status == OrderStatus.FILLED:
                        # Order fully filled
                        total_filled = order_response.quantity

                        # Update execution result
                        await self._update_execution_result(
                            execution_result, child_order=order_response
                        )

                        self.logger.debug(
                            f"Iceberg slice fully filled: {order_id}",
                            filled_quantity=float(total_filled),
                        )
                        break

                    elif order_status == OrderStatus.PARTIALLY_FILLED:
                        # Check for partial fills (simplified - would need actual fill quantity)
                        # In a real implementation, this would get the actual filled amount
                        partial_filled = order_response.quantity * Decimal("0.5")  # Placeholder

                        if partial_filled > total_filled:
                            total_filled = partial_filled

                        self.logger.debug(
                            f"Iceberg slice partially filled: {order_id}",
                            filled_quantity=float(total_filled),
                        )

                    elif order_status in [
                        OrderStatus.CANCELLED,
                        OrderStatus.REJECTED,
                        OrderStatus.EXPIRED,
                    ]:
                        # Order terminated
                        self.logger.info(
                            f"Iceberg slice terminated: {order_id}",
                            status=order_status.value,
                            filled_quantity=float(total_filled),
                        )
                        break

                    # Wait before next status check
                    await asyncio.sleep(self.fill_monitoring_interval)

                except Exception as e:
                    self.logger.warning(f"Failed to check order status {order_id}: {e}")
                    await asyncio.sleep(5)

            # Update execution result with final filled amount
            if total_filled > 0:
                # Update order response with actual filled quantity
                order_response.filled_quantity = total_filled
                await self._update_execution_result(execution_result, child_order=order_response)

            return total_filled

        except Exception as e:
            self.logger.error(f"Order monitoring failed: {e}")
            return Decimal("0")

    async def _get_improved_price(self, symbol: str, side, exchange) -> Decimal | None:
        """
        Get an improved price for the order based on current market conditions.

        Args:
            symbol: Trading symbol
            side: Order side
            exchange: Exchange for market data

        Returns:
            Decimal: Improved price or None if unable to determine
        """
        try:
            # Get current market data
            market_data = await exchange.get_market_data(symbol)

            if not market_data or not market_data.bid or not market_data.ask:
                return None

            # Calculate improved price (inside the spread)
            if side.value == "buy":
                # For buy orders, try to get filled at a better price
                # Start with bid + small improvement
                tick_size = Decimal("0.01")  # Placeholder tick size
                improved_price = market_data.bid + tick_size

                # Don't exceed the ask
                if improved_price >= market_data.ask:
                    improved_price = market_data.ask - tick_size

            else:  # sell
                # For sell orders, try to get filled at a better price
                tick_size = Decimal("0.01")  # Placeholder tick size
                improved_price = market_data.ask - tick_size

                # Don't go below the bid
                if improved_price <= market_data.bid:
                    improved_price = market_data.bid + tick_size

            self.logger.debug(
                "Price improvement calculated",
                symbol=symbol,
                side=side.value,
                bid=float(market_data.bid),
                ask=float(market_data.ask),
                improved_price=float(improved_price),
            )

            return improved_price

        except Exception as e:
            self.logger.warning(f"Failed to calculate improved price: {e}")
            return None

    async def _finalize_execution(self, execution_result: ExecutionResult) -> None:
        """
        Finalize the Iceberg execution result.

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

            # Calculate Iceberg-specific metrics
            if execution_result.total_filled_quantity > 0:
                # Calculate stealth effectiveness
                fill_rate = (
                    execution_result.total_filled_quantity
                    / execution_result.original_order.quantity
                )
                execution_result.metadata["iceberg_fill_rate"] = float(fill_rate)
                execution_result.metadata["stealth_execution"] = True
                execution_result.metadata["number_of_refreshes"] = execution_result.number_of_trades

                # Calculate slippage metrics
                await self._calculate_slippage_metrics(
                    execution_result, expected_price=execution_result.original_order.price
                )

            self.logger.debug(
                "Iceberg execution finalized",
                execution_id=execution_result.execution_id,
                final_status=execution_result.status.value,
                stealth_effectiveness=execution_result.metadata.get("iceberg_fill_rate", 0.0),
            )

        except Exception as e:
            self.logger.error(f"Failed to finalize Iceberg execution: {e}")
            execution_result.error_message = f"Finalization failed: {e}"
