"""
Order Management Service.

This service provides order management operations following proper service layer patterns,
wrapping the OrderManager and ensuring only services are called from controllers.
"""

from datetime import datetime, timezone
from typing import Any

from src.core.base.service import BaseService
from src.core.exceptions import ServiceError
from src.core.logging import get_logger
from src.core.types import OrderRequest, OrderStatus
from src.execution.interfaces import OrderManagementServiceInterface
from src.execution.order_manager import ManagedOrder, OrderManager

try:
    from src.utils import time_execution
except ImportError:
    # Fallback if time_execution is not available
    def time_execution(func):
        return func


class OrderManagementService(BaseService, OrderManagementServiceInterface):
    """
    Service layer for order management operations.
    
    This service wraps the OrderManager component and provides proper service
    layer abstraction for order lifecycle management.
    """

    def __init__(
        self,
        order_manager: OrderManager,
        correlation_id: str | None = None,
    ):
        """
        Initialize order management service.
        
        Args:
            order_manager: OrderManager component instance
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="OrderManagementService",
            correlation_id=correlation_id,
        )

        if not order_manager:
            raise ValueError("OrderManager is required")

        self.order_manager = order_manager
        self.logger = get_logger(__name__)

    async def _do_start(self) -> None:
        """Start the order management service."""
        if not self.order_manager.is_running:
            await self.order_manager.start()

        self.logger.info("Order management service started")

    @time_execution
    async def create_managed_order(
        self,
        order_request: OrderRequest,
        execution_id: str,
        timeout_minutes: int | None = None,
        callbacks: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create and manage an order.
        
        Args:
            order_request: Order details
            execution_id: Associated execution ID
            timeout_minutes: Order timeout in minutes
            callbacks: Optional callbacks
            
        Returns:
            Dict containing order details
            
        Raises:
            ServiceError: If order creation fails
            ValidationError: If validation fails
        """
        try:
            if not self.is_running:
                raise ServiceError("OrderManagementService is not running")

            # Create managed order through OrderManager
            managed_order = ManagedOrder(
                order_request=order_request,
                execution_id=execution_id,
                timeout_minutes=timeout_minutes or 30,
            )

            # Set up callbacks if provided
            if callbacks:
                managed_order.on_fill_callback = callbacks.get("on_fill")
                managed_order.on_complete_callback = callbacks.get("on_complete")
                managed_order.on_cancel_callback = callbacks.get("on_cancel")
                managed_order.on_error_callback = callbacks.get("on_error")

            # Register with OrderManager
            await self.order_manager.register_managed_order(managed_order)

            self.logger.info(
                "Managed order created",
                order_id=managed_order.order_id,
                execution_id=execution_id,
                symbol=order_request.symbol,
            )

            return {
                "order_id": managed_order.order_id,
                "execution_id": execution_id,
                "status": managed_order.status.value,
                "symbol": order_request.symbol,
                "quantity": str(order_request.quantity),
                "side": order_request.side.value,
                "created_at": managed_order.created_at.isoformat(),
                "timeout_minutes": timeout_minutes,
            }

        except Exception as e:
            self.logger.error("Failed to create managed order", error=str(e))
            raise ServiceError(f"Order creation failed: {e}") from e

    @time_execution
    async def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        details: dict[str, Any] | None = None,
    ) -> bool:
        """
        Update order status.
        
        Args:
            order_id: Order ID to update
            status: New status
            details: Optional status details
            
        Returns:
            bool: True if update successful
        """
        try:
            if not self.is_running:
                raise ServiceError("OrderManagementService is not running")

            managed_order = self.order_manager.managed_orders.get(order_id)
            if not managed_order:
                self.logger.warning("Order not found for status update", order_id=order_id)
                return False

            # Update status through ManagedOrder
            managed_order.update_status(status, details)

            # Update in OrderManager's tracking
            await self.order_manager.update_order_status_internal(order_id, status, details)

            self.logger.info(
                "Order status updated",
                order_id=order_id,
                new_status=status.value,
            )

            return True

        except Exception as e:
            self.logger.error("Failed to update order status", order_id=order_id, error=str(e))
            return False

    @time_execution
    async def cancel_order(
        self,
        order_id: str,
        reason: str = "manual"
    ) -> bool:
        """
        Cancel a managed order.
        
        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason
            
        Returns:
            bool: True if cancellation successful
        """
        try:
            if not self.is_running:
                raise ServiceError("OrderManagementService is not running")

            # Cancel through OrderManager
            result = await self.order_manager.cancel_order(order_id, reason)

            if result:
                self.logger.info(
                    "Order cancelled",
                    order_id=order_id,
                    reason=reason,
                )
            else:
                self.logger.warning(
                    "Order cancellation failed",
                    order_id=order_id,
                    reason=reason,
                )

            return result

        except Exception as e:
            self.logger.error("Failed to cancel order", order_id=order_id, error=str(e))
            return False

    @time_execution
    async def get_order_metrics(
        self,
        symbol: str | None = None,
        time_range_hours: int = 24,
    ) -> dict[str, Any]:
        """
        Get order management metrics.
        
        Args:
            symbol: Optional symbol filter
            time_range_hours: Time range for metrics
            
        Returns:
            Dict containing metrics
        """
        try:
            if not self.is_running:
                return {"error": "OrderManagementService is not running"}

            # Get summary from OrderManager
            if hasattr(self.order_manager, "get_order_manager_summary"):
                summary = await self.order_manager.get_order_manager_summary()
            else:
                # Fallback to basic metrics
                summary = self._get_basic_metrics()

            # Add symbol-specific filtering if requested
            if symbol:
                symbol_orders = await self._get_orders_by_symbol(symbol)
                summary["symbol_specific"] = {
                    "symbol": symbol,
                    "order_count": len(symbol_orders),
                    "active_orders": len([
                        o for o in symbol_orders
                        if o["status"] not in ["FILLED", "CANCELLED", "REJECTED"]
                    ]),
                }

            # Add time range context
            summary["metrics_context"] = {
                "time_range_hours": time_range_hours,
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "service_name": "OrderManagementService",
            }

            return summary

        except Exception as e:
            self.logger.error("Failed to get order metrics", error=str(e))
            return {"error": str(e)}

    async def get_active_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """
        Get active orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of active orders
        """
        try:
            if not self.is_running:
                return []

            active_orders = []

            for order_id, managed_order in self.order_manager.managed_orders.items():
                if managed_order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    if not symbol or managed_order.order_request.symbol == symbol:
                        active_orders.append({
                            "order_id": order_id,
                            "symbol": managed_order.order_request.symbol,
                            "side": managed_order.order_request.side.value,
                            "quantity": str(managed_order.order_request.quantity),
                            "status": managed_order.status.value,
                            "created_at": managed_order.created_at.isoformat(),
                            "execution_id": managed_order.execution_id,
                        })

            return active_orders

        except Exception as e:
            self.logger.error("Failed to get active orders", error=str(e))
            return []

    def _get_basic_metrics(self) -> dict[str, Any]:
        """Get basic order metrics fallback."""
        total_orders = len(self.order_manager.managed_orders)
        active_orders = len([
            o for o in self.order_manager.managed_orders.values()
            if o.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]
        ])

        return {
            "total_orders": total_orders,
            "active_orders": active_orders,
            "completed_orders": total_orders - active_orders,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def _get_orders_by_symbol(self, symbol: str) -> list[dict[str, Any]]:
        """Get orders filtered by symbol."""
        symbol_orders = []

        for order_id, managed_order in self.order_manager.managed_orders.items():
            if managed_order.order_request.symbol == symbol:
                symbol_orders.append({
                    "order_id": order_id,
                    "status": managed_order.status.value,
                    "quantity": str(managed_order.order_request.quantity),
                    "created_at": managed_order.created_at.isoformat(),
                })

        return symbol_orders

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Dict containing health status
        """
        return {
            "service_name": "OrderManagementService",
            "is_running": self.is_running,
            "order_manager_running": self.order_manager.is_running,
            "managed_orders_count": len(self.order_manager.managed_orders),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
