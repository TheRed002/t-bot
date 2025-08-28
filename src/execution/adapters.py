"""
Adapters for converting between execution module internal types and core types.

This module provides adapters to maintain backward compatibility while
transitioning to use core types properly.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from src.core.types import (
    ExecutionAlgorithm,
    ExecutionResult as CoreExecutionResult,
    ExecutionStatus,
    OrderRequest,
    OrderResponse,
    OrderSide,
)


class ExecutionResultAdapter:
    """Adapter for converting between internal and core ExecutionResult types."""

    @staticmethod
    def to_core_result(
        execution_id: str,
        original_order: OrderRequest,
        algorithm: ExecutionAlgorithm,
        status: ExecutionStatus,
        start_time: datetime,
        child_orders: list[OrderResponse] | None = None,
        total_filled_quantity: Decimal | None = None,
        average_fill_price: Decimal | None = None,
        total_fees: Decimal | None = None,
        end_time: datetime | None = None,
        execution_duration: float | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CoreExecutionResult:
        """
        Convert internal execution result format to core ExecutionResult.

        Args:
            execution_id: Unique execution ID
            original_order: Original order request
            algorithm: Execution algorithm used
            status: Current execution status
            start_time: Execution start time
            child_orders: List of child orders created
            total_filled_quantity: Total quantity filled
            average_fill_price: Average fill price
            total_fees: Total fees paid
            end_time: Execution end time
            execution_duration: Duration in seconds
            error_message: Error message if failed
            metadata: Additional metadata

        Returns:
            CoreExecutionResult: Core-compatible execution result
        """
        # Calculate derived values
        child_orders = child_orders or []
        total_filled_quantity = total_filled_quantity or Decimal("0")
        average_fill_price = average_fill_price or Decimal("0")
        total_fees = total_fees or Decimal("0")

        # Calculate fees breakdown (simplified - can be enhanced)
        maker_fees = total_fees * Decimal("0.4")  # 40% maker fees estimate
        taker_fees = total_fees * Decimal("0.6")  # 60% taker fees estimate

        # Calculate slippage if we have expected price
        expected_price = original_order.price or average_fill_price
        if average_fill_price > 0 and expected_price > 0:
            if original_order.side == OrderSide.BUY:
                slippage_amount = (average_fill_price - expected_price) * total_filled_quantity
            else:
                slippage_amount = (expected_price - average_fill_price) * total_filled_quantity

            slippage_bps = float(
                (slippage_amount / (expected_price * total_filled_quantity)) * 10000
            )
        else:
            slippage_amount = Decimal("0")
            slippage_bps = 0.0

        # Calculate costs
        actual_cost = total_filled_quantity * average_fill_price + total_fees
        expected_cost = original_order.quantity * expected_price

        # Build fills list from child orders with null safety
        fills = []
        for child_order in child_orders:
            if (
                child_order
                and hasattr(child_order, "filled_quantity")
                and child_order.filled_quantity > 0
            ):
                fill_data = {
                    "order_id": getattr(child_order, "order_id", "unknown"),
                    "exchange": getattr(child_order, "exchange", "unknown"),
                    "symbol": getattr(child_order, "symbol", original_order.symbol),
                    "side": child_order.side.value
                    if hasattr(child_order, "side")
                    else original_order.side.value,
                    "quantity": float(child_order.filled_quantity),
                    "price": float(child_order.price)
                    if hasattr(child_order, "price") and child_order.price
                    else 0,
                    "timestamp": None,
                }

                # Safely handle timestamp
                if hasattr(child_order, "timestamp") and child_order.timestamp:
                    try:
                        fill_data["timestamp"] = child_order.timestamp.isoformat()
                    except Exception as e:
                        self.logger.warning("Failed to convert timestamp to ISO format", error=str(e))
                        fill_data["timestamp"] = None

                fills.append(fill_data)

        # Map status - handle PENDING which doesn't exist in core
        if status == ExecutionStatus.PENDING:
            mapped_status = ExecutionStatus.RUNNING
        else:
            mapped_status = status

        # Get best and worst prices from fills
        if fills:
            prices = [f["price"] for f in fills if f["price"] > 0]
            if prices:
                if original_order.side == OrderSide.BUY:
                    best_price = Decimal(str(min(prices)))
                    worst_price = Decimal(str(max(prices)))
                else:
                    best_price = Decimal(str(max(prices)))
                    worst_price = Decimal(str(min(prices)))
            else:
                best_price = worst_price = average_fill_price
        else:
            best_price = worst_price = average_fill_price

        return CoreExecutionResult(
            instruction_id=execution_id,
            symbol=original_order.symbol,
            status=mapped_status,
            # Execution details
            target_quantity=original_order.quantity,
            filled_quantity=total_filled_quantity,
            remaining_quantity=original_order.quantity - total_filled_quantity,
            # Pricing
            target_price=original_order.price,
            average_price=average_fill_price,
            worst_price=worst_price,
            best_price=best_price,
            # Slippage analysis
            expected_cost=expected_cost,
            actual_cost=actual_cost,
            slippage_bps=slippage_bps,
            slippage_amount=slippage_amount,
            # Execution quality
            fill_rate=(
                float(total_filled_quantity / original_order.quantity)
                if original_order.quantity > 0
                else 0.0
            ),
            execution_time=int(execution_duration) if execution_duration else 0,
            num_fills=len(fills),
            num_orders=len(child_orders),
            # Fees
            total_fees=total_fees,
            maker_fees=maker_fees,
            taker_fees=taker_fees,
            # Timestamps
            started_at=start_time,
            completed_at=end_time,
            # Detailed fills
            fills=fills,
            metadata=metadata or {},
        )

    @staticmethod
    def from_core_result(core_result: CoreExecutionResult) -> dict[str, Any]:
        """
        Convert core ExecutionResult to internal format expected by execution module.

        Args:
            core_result: Core execution result

        Returns:
            dict: Internal format with execution module expected fields
        """
        # Map status back
        if core_result.status == ExecutionStatus.RUNNING:
            internal_status = ExecutionStatus.PENDING
        else:
            internal_status = core_result.status

        # Calculate execution duration
        if core_result.completed_at and core_result.started_at:
            execution_duration = (core_result.completed_at - core_result.started_at).total_seconds()
        else:
            execution_duration = None

        return {
            "execution_id": core_result.instruction_id,
            "status": internal_status,
            "total_filled_quantity": core_result.filled_quantity,
            "average_fill_price": core_result.average_price,
            "total_fees": core_result.total_fees,
            "start_time": core_result.started_at,
            "end_time": core_result.completed_at,
            "execution_duration": execution_duration,
            "metadata": core_result.metadata,
        }


class OrderAdapter:
    """Adapter for order-related type conversions."""

    @staticmethod
    def order_response_to_child_order(response: OrderResponse) -> dict[str, Any]:
        """Convert OrderResponse to child order format expected by execution module."""
        return {
            "order_id": response.order_id,
            "exchange": response.exchange,
            "symbol": response.symbol,
            "side": response.side,
            "filled_quantity": response.filled_quantity,
            "price": response.price,
            "status": response.status,
            "timestamp": response.timestamp,
        }
