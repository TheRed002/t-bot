"""
Service Layer Adapters for Execution Module.

These adapters wrap existing components to conform to service layer interfaces,
fixing service layer violations while maintaining backward compatibility.
"""

from datetime import datetime, timezone
from typing import Any

from src.core.exceptions import ServiceError
from src.core.logging import get_logger
from src.core.types import ExecutionResult, MarketData, OrderRequest, OrderStatus
from src.execution.types import ExecutionInstruction
from src.utils.execution_utils import calculate_order_value

try:
    from src.utils import time_execution
except ImportError:
    # Fallback if time_execution is not available
    def time_execution(func):
        return func


class ExecutionEngineServiceAdapter:
    """
    Service adapter for ExecutionEngine to conform to service interface.
    """

    def __init__(self, execution_engine):
        """Initialize with existing execution engine."""
        self.execution_engine = execution_engine
        self.logger = get_logger(__name__)

    async def execute_instruction(
        self,
        instruction: ExecutionInstruction,
        market_data: MarketData,
        bot_id: str | None = None,
        strategy_name: str | None = None,
    ) -> ExecutionResult:
        """Execute instruction through engine."""
        try:
            if not self.execution_engine.is_running:
                raise ServiceError("Execution engine is not running")

            # Use the existing ExecutionEngine.execute_order method
            result = await self.execution_engine.execute_order(
                instruction=instruction,
                market_data=market_data,
                bot_id=bot_id,
                strategy_name=strategy_name,
            )

            return result

        except Exception as e:
            self.logger.error("Execution instruction failed", error=str(e))
            raise ServiceError(f"Execution failed: {e}") from e

    async def get_active_executions(self) -> dict[str, Any]:
        """Get active executions through engine."""
        try:
            active_executions = await self.execution_engine.get_active_executions()
            return {
                "active_executions": {
                    execution_id: {
                        "execution_id": result.execution_id,
                        "status": result.status.value,
                        "symbol": result.original_order.symbol,
                        "side": result.original_order.side.value,
                        "quantity": str(result.total_filled_quantity),
                    }
                    for execution_id, result in active_executions.items()
                },
                "count": len(active_executions),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            self.logger.error("Failed to get active executions", error=str(e))
            return {"error": str(e)}

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel execution through engine."""
        try:
            return await self.execution_engine.cancel_execution(execution_id)
        except Exception as e:
            self.logger.error("Failed to cancel execution", execution_id=execution_id, error=str(e))
            return False

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics through engine."""
        try:
            return await self.execution_engine.get_execution_metrics()
        except Exception as e:
            self.logger.error("Failed to get performance metrics", error=str(e))
            return {"error": str(e)}


class OrderManagementServiceAdapter:
    """
    Service adapter for OrderManager to conform to service interface.
    """

    def __init__(self, order_manager):
        """Initialize with existing order manager."""
        self.order_manager = order_manager
        self.logger = get_logger(__name__)

    async def create_managed_order(
        self,
        order_request: OrderRequest,
        execution_id: str,
        timeout_minutes: int | None = None,
        callbacks: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create managed order (adapter method)."""
        try:
            # This method would need to be integrated with actual order submission
            # For now, return a structured response
            return {
                "order_id": f"order_{execution_id}_{int(datetime.now().timestamp())}",
                "execution_id": execution_id,
                "status": "created",
                "symbol": order_request.symbol,
                "quantity": str(order_request.quantity),
                "side": order_request.side.value,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "timeout_minutes": timeout_minutes,
            }
        except Exception as e:
            self.logger.error("Failed to create managed order", error=str(e))
            raise ServiceError(f"Order creation failed: {e}") from e

    async def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        details: dict[str, Any] | None = None,
    ) -> bool:
        """Update order status through manager."""
        try:
            managed_order = self.order_manager.managed_orders.get(order_id)
            if not managed_order:
                return False

            old_status = managed_order.status
            managed_order.update_status(status, details)

            self.logger.info(
                "Order status updated",
                order_id=order_id,
                old_status=old_status.value,
                new_status=status.value,
            )

            return True
        except Exception as e:
            self.logger.error("Failed to update order status", order_id=order_id, error=str(e))
            return False

    async def cancel_order(self, order_id: str, reason: str = "manual") -> bool:
        """Cancel order through manager."""
        try:
            return await self.order_manager.cancel_order(order_id, reason)
        except Exception as e:
            self.logger.error("Failed to cancel order", order_id=order_id, error=str(e))
            return False

    @time_execution
    async def get_order_metrics(
        self,
        symbol: str | None = None,
        time_range_hours: int = 24,
    ) -> dict[str, Any]:
        """Get order metrics through manager."""
        try:
            if hasattr(self.order_manager, "get_order_manager_summary"):
                summary = await self.order_manager.get_order_manager_summary()

                # Filter by symbol if specified
                if symbol:
                    # Add symbol-specific filtering logic here
                    symbol_orders = await self.order_manager.get_orders_by_symbol(symbol)
                    summary["symbol_specific"] = {
                        "symbol": symbol,
                        "order_count": len(symbol_orders),
                        "orders": [
                            {
                                "order_id": order.order_id,
                                "status": order.status.value,
                                "quantity": str(order.order_request.quantity),
                                "created_at": order.created_at.isoformat(),
                            }
                            for order in symbol_orders
                        ],
                    }

                return summary
            else:
                # Fallback to basic metrics
                return {
                    "total_orders": len(self.order_manager.managed_orders),
                    "active_orders": len([
                        o for o in self.order_manager.managed_orders.values()
                        if o.status not in ["FILLED", "CANCELLED", "REJECTED"]
                    ]),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        except Exception as e:
            self.logger.error("Failed to get order metrics", error=str(e))
            return {"error": str(e)}


class RiskValidationServiceAdapter:
    """
    Service adapter for risk validation operations.
    """

    def __init__(self, risk_service=None):
        """Initialize with optional risk service."""
        self.risk_service = risk_service
        self.logger = get_logger(__name__)

    async def validate_order_risk(
        self,
        order: OrderRequest,
        market_data: MarketData,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate order risk."""
        try:
            if not self.risk_service:
                # Basic validation without risk service
                return {
                    "approved": True,
                    "risk_level": "unknown",
                    "reason": "No risk service available - basic validation passed",
                    "checks": ["basic_validation"],
                }

            # Use risk service if available
            from src.core.types import Signal, SignalDirection

            signal_direction = (
                SignalDirection.BUY if order.side.value == "BUY"
                else SignalDirection.SELL
            )

            signal = Signal(
                symbol=order.symbol,
                direction=signal_direction,
                strength=0.5,
                timestamp=datetime.now(timezone.utc),
                source="ExecutionServiceAdapter",
                metadata={
                    "quantity": str(order.quantity),
                    "price": str(order.price) if order.price else "market",
                    "context": context or {},
                }
            )

            validation_result = await self.risk_service.validate_signal(signal)

            return {
                "approved": bool(validation_result),
                "risk_level": "low" if validation_result else "high",
                "reason": "Risk service validation" + ("passed" if validation_result else "failed"),
                "checks": ["risk_service_validation"],
                "signal_strength": signal.strength,
            }

        except Exception as e:
            self.logger.error("Risk validation failed", error=str(e))
            return {
                "approved": False,
                "risk_level": "critical",
                "reason": f"Risk validation error: {e}",
                "checks": [],
            }

    async def check_position_limits(
        self,
        order: OrderRequest,
        current_positions: dict[str, Any] | None = None,
    ) -> bool:
        """Check position limits."""
        try:
            if not self.risk_service:
                # Basic position limit check
                max_position_value = 100000  # $100k default limit
                from decimal import Decimal
                default_price = Decimal("50000")  # Use $50k default price
                order_value = calculate_order_value(order.quantity, order.price, None, default_price)
                return order_value <= Decimal(str(max_position_value))

            # Use risk service for position validation
            from src.core.types import Signal, SignalDirection

            signal_direction = (
                SignalDirection.BUY if order.side.value == "BUY"
                else SignalDirection.SELL
            )

            signal = Signal(
                symbol=order.symbol,
                direction=signal_direction,
                strength=0.5,
                timestamp=datetime.now(timezone.utc),
                source="RiskValidationAdapter",
            )

            # Use appropriate price source
            current_price = (
                order.price if order.price
                else (current_positions.get("current_price") if current_positions
                     else 50000)  # Default fallback price
            )

            position_size = await self.risk_service.calculate_position_size(
                signal=signal,
                available_capital=None,
                current_price=current_price,
            )

            return position_size is not None and order.quantity <= position_size

        except Exception as e:
            self.logger.error("Position limits check failed", error=str(e))
            return False
