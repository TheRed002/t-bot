"""
Order Manager for comprehensive order lifecycle management.

This module provides sophisticated order lifecycle management, including
order tracking, status monitoring, partial fill handling, and automated
order management workflows for the execution engine.

CRITICAL: This integrates with P-001 (types, exceptions, config),
P-002A (error handling), and P-007A (utils) components.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable
from uuid import uuid4

from src.core.config import Config
from src.core.exceptions import ExecutionError, ValidationError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import (
    ExecutionResult,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    OrderType,
)

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

# MANDATORY: Import from P-007A
from src.utils.decorators import log_calls, time_execution

logger = get_logger(__name__)


class OrderLifecycleEvent:
    """Represents an event in the order lifecycle."""
    
    def __init__(
        self,
        event_type: str,
        order_id: str,
        timestamp: datetime,
        data: dict[str, Any] | None = None
    ):
        self.event_type = event_type
        self.order_id = order_id
        self.timestamp = timestamp
        self.data = data or {}


class ManagedOrder:
    """Represents a managed order with complete lifecycle tracking."""
    
    def __init__(self, order_request: OrderRequest, execution_id: str):
        self.order_request = order_request
        self.execution_id = execution_id
        self.order_id: str | None = None
        self.status = OrderStatus.PENDING
        self.filled_quantity = Decimal("0")
        self.remaining_quantity = order_request.quantity
        self.average_fill_price: Decimal | None = None
        self.total_fees = Decimal("0")
        
        # Lifecycle tracking
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        self.events: list[OrderLifecycleEvent] = []
        self.fills: list[dict[str, Any]] = []
        
        # Management parameters
        self.timeout_minutes: int | None = None
        self.auto_cancel_on_timeout = True
        self.partial_fills_accepted = True
        self.retry_count = 0
        self.max_retries = 3
        
        # Callbacks
        self.on_fill_callback: Callable | None = None
        self.on_complete_callback: Callable | None = None
        self.on_cancel_callback: Callable | None = None
        self.on_error_callback: Callable | None = None


class OrderManager:
    """
    Comprehensive order lifecycle management system.
    
    This manager provides:
    - Order creation and submission tracking
    - Real-time status monitoring and updates
    - Partial fill handling and aggregation
    - Automatic timeout and retry management
    - Event-driven order lifecycle callbacks
    - Performance metrics and analytics
    - Risk-aware order management workflows
    """

    def __init__(self, config: Config):
        """
        Initialize order manager.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.error_handler = ErrorHandler(config.error_handling)
        
        # Order tracking
        self.managed_orders: dict[str, ManagedOrder] = {}  # order_id -> ManagedOrder
        self.execution_orders: dict[str, list[str]] = {}   # execution_id -> [order_ids]
        
        # Management configuration
        self.default_order_timeout_minutes = 60  # 1 hour default timeout
        self.status_check_interval_seconds = 5   # Check status every 5 seconds
        self.max_concurrent_orders = 100         # Maximum concurrent orders
        self.order_history_retention_hours = 24  # Keep order history for 24 hours
        
        # Monitoring and alerting
        self.monitoring_tasks: dict[str, asyncio.Task] = {}
        self.alert_thresholds = {
            "high_rejection_rate": 0.1,    # 10% rejection rate
            "slow_fill_rate": 0.5,         # 50% of orders taking > 5 minutes
            "high_cancel_rate": 0.15       # 15% cancellation rate
        }
        
        # Performance tracking
        self.order_statistics = {
            "total_orders": 0,
            "completed_orders": 0,
            "cancelled_orders": 0,
            "rejected_orders": 0,
            "partially_filled_orders": 0,
            "average_fill_time_seconds": 0.0,
            "total_volume": Decimal("0"),
            "total_fees": Decimal("0")
        }
        
        # Cleanup task
        self._cleanup_task: asyncio.Task | None = None
        self._cleanup_started = False
        
        self.logger.info("Order manager initialized with lifecycle management")

    async def start(self) -> None:
        """Start the order manager and its background tasks."""
        if not self._cleanup_started:
            self._start_cleanup_task()
            self._cleanup_started = True
            self.logger.info("Order manager started")

    def _start_cleanup_task(self) -> None:
        """Start the periodic cleanup task."""
        async def cleanup_periodically():
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    await self._cleanup_old_orders()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Cleanup task error: {e}")
                    
        self._cleanup_task = asyncio.create_task(cleanup_periodically())

    @time_execution
    async def submit_order(
        self,
        order_request: OrderRequest,
        exchange,
        execution_id: str,
        timeout_minutes: int | None = None,
        callbacks: dict[str, Callable] | None = None
    ) -> ManagedOrder:
        """
        Submit an order with comprehensive lifecycle management.
        
        Args:
            order_request: Order to submit
            exchange: Exchange to submit to
            execution_id: Execution ID for tracking
            timeout_minutes: Optional timeout override
            callbacks: Optional lifecycle callbacks
            
        Returns:
            ManagedOrder: Managed order instance
            
        Raises:
            ExecutionError: If order submission fails
            ValidationError: If order is invalid
        """
        try:
            # Validate order
            if not order_request or not order_request.symbol:
                raise ValidationError("Invalid order request")
                
            if order_request.quantity <= 0:
                raise ValidationError("Order quantity must be positive")
                
            # Check capacity
            if len(self.managed_orders) >= self.max_concurrent_orders:
                raise ExecutionError("Maximum concurrent orders reached")
                
            # Create managed order
            managed_order = ManagedOrder(order_request, execution_id)
            managed_order.timeout_minutes = timeout_minutes or self.default_order_timeout_minutes
            
            # Set callbacks
            if callbacks:
                managed_order.on_fill_callback = callbacks.get("on_fill")
                managed_order.on_complete_callback = callbacks.get("on_complete")
                managed_order.on_cancel_callback = callbacks.get("on_cancel")
                managed_order.on_error_callback = callbacks.get("on_error")
                
            self.logger.info(
                "Submitting order",
                execution_id=execution_id,
                symbol=order_request.symbol,
                quantity=float(order_request.quantity),
                side=order_request.side.value,
                order_type=order_request.order_type.value
            )
            
            # Submit to exchange
            order_response = await exchange.place_order(order_request)
            
            # Update managed order with response
            managed_order.order_id = order_response.id
            managed_order.status = OrderStatus.PENDING
            
            # Register order
            self.managed_orders[order_response.id] = managed_order
            
            if execution_id not in self.execution_orders:
                self.execution_orders[execution_id] = []
            self.execution_orders[execution_id].append(order_response.id)
            
            # Add creation event
            await self._add_order_event(
                managed_order,
                "order_submitted",
                {
                    "exchange": exchange.exchange_name,
                    "order_response": {
                        "id": order_response.id,
                        "status": order_response.status,
                        "timestamp": order_response.timestamp.isoformat()
                    }
                }
            )
            
            # Start monitoring
            await self._start_order_monitoring(managed_order, exchange)
            
            # Update statistics
            self.order_statistics["total_orders"] += 1
            
            self.logger.info(
                "Order submitted successfully",
                order_id=order_response.id,
                execution_id=execution_id
            )
            
            return managed_order
            
        except Exception as e:
            self.logger.error(f"Order submission failed: {e}")
            
            # Update statistics
            self.order_statistics["rejected_orders"] += 1
            
            # Call error callback if provided
            if 'managed_order' in locals() and managed_order.on_error_callback:
                try:
                    await managed_order.on_error_callback(managed_order, str(e))
                except Exception as callback_error:
                    self.logger.error(f"Error callback failed: {callback_error}")
                    
            raise ExecutionError(f"Order submission failed: {e}")

    async def _start_order_monitoring(self, managed_order: ManagedOrder, exchange) -> None:
        """Start monitoring an order's lifecycle."""
        async def monitor_order():
            try:
                order_id = managed_order.order_id
                if not order_id:
                    return
                    
                start_time = datetime.now(timezone.utc)
                timeout = timedelta(minutes=managed_order.timeout_minutes)
                last_status_check = start_time
                
                while True:
                    current_time = datetime.now(timezone.utc)
                    
                    # Check for timeout
                    if current_time - start_time > timeout:
                        self.logger.warning(f"Order timeout reached: {order_id}")
                        
                        if managed_order.auto_cancel_on_timeout:
                            await self.cancel_order(order_id, "timeout")
                        else:
                            await self._add_order_event(managed_order, "timeout_reached", {})
                            
                        break
                        
                    # Check status periodically
                    if (current_time - last_status_check).total_seconds() >= self.status_check_interval_seconds:
                        try:
                            await self._check_order_status(managed_order, exchange)
                            last_status_check = current_time
                            
                            # Stop monitoring if order is in terminal state
                            if managed_order.status in [
                                OrderStatus.FILLED, 
                                OrderStatus.CANCELLED, 
                                OrderStatus.REJECTED, 
                                OrderStatus.EXPIRED
                            ]:
                                break
                                
                        except Exception as e:
                            self.logger.warning(f"Status check failed for {order_id}: {e}")
                            
                    await asyncio.sleep(1)  # Small sleep to prevent busy waiting
                    
            except asyncio.CancelledError:
                self.logger.debug(f"Order monitoring cancelled for {managed_order.order_id}")
            except Exception as e:
                self.logger.error(f"Order monitoring failed: {e}")
                
                if managed_order.on_error_callback:
                    try:
                        await managed_order.on_error_callback(managed_order, str(e))
                    except Exception:
                        pass
                        
        # Start monitoring task
        task = asyncio.create_task(monitor_order())
        if managed_order.order_id:
            self.monitoring_tasks[managed_order.order_id] = task

    async def _check_order_status(self, managed_order: ManagedOrder, exchange) -> None:
        """Check and update order status."""
        try:
            if not managed_order.order_id:
                return
                
            # Get current status from exchange
            current_status = await exchange.get_order_status(managed_order.order_id)
            
            # Check if status changed
            if current_status != managed_order.status:
                old_status = managed_order.status
                managed_order.status = current_status
                managed_order.updated_at = datetime.now(timezone.utc)
                
                await self._add_order_event(
                    managed_order,
                    "status_changed",
                    {
                        "old_status": old_status.value,
                        "new_status": current_status.value
                    }
                )
                
                # Handle specific status changes
                await self._handle_status_change(managed_order, old_status, current_status)
                
        except Exception as e:
            self.logger.warning(f"Status check failed: {e}")

    async def _handle_status_change(
        self, 
        managed_order: ManagedOrder, 
        old_status: OrderStatus, 
        new_status: OrderStatus
    ) -> None:
        """Handle order status changes."""
        try:
            if new_status == OrderStatus.FILLED:
                # Order fully filled
                managed_order.filled_quantity = managed_order.order_request.quantity
                managed_order.remaining_quantity = Decimal("0")
                
                await self._add_order_event(managed_order, "order_filled", {
                    "filled_quantity": float(managed_order.filled_quantity)
                })
                
                # Update statistics
                self.order_statistics["completed_orders"] += 1
                self.order_statistics["total_volume"] += managed_order.filled_quantity
                
                # Calculate fill time
                if managed_order.created_at:
                    fill_time = (datetime.now(timezone.utc) - managed_order.created_at).total_seconds()
                    await self._update_average_fill_time(fill_time)
                    
                # Call completion callback
                if managed_order.on_complete_callback:
                    await managed_order.on_complete_callback(managed_order)
                    
            elif new_status == OrderStatus.PARTIALLY_FILLED:
                # Handle partial fill (simplified - would need actual fill data from exchange)
                # In practice, this would get the actual filled amount from the exchange
                await self._handle_partial_fill(managed_order)
                
            elif new_status == OrderStatus.CANCELLED:
                await self._add_order_event(managed_order, "order_cancelled", {})
                self.order_statistics["cancelled_orders"] += 1
                
                if managed_order.on_cancel_callback:
                    await managed_order.on_cancel_callback(managed_order)
                    
            elif new_status == OrderStatus.REJECTED:
                await self._add_order_event(managed_order, "order_rejected", {})
                self.order_statistics["rejected_orders"] += 1
                
                if managed_order.on_error_callback:
                    await managed_order.on_error_callback(managed_order, "Order rejected by exchange")
                    
        except Exception as e:
            self.logger.error(f"Status change handling failed: {e}")

    async def _handle_partial_fill(self, managed_order: ManagedOrder) -> None:
        """Handle partial fill processing."""
        try:
            # In a real implementation, this would get actual fill details from the exchange
            # For now, simulate partial fill data
            fill_quantity = managed_order.order_request.quantity * Decimal("0.3")  # 30% fill
            fill_price = managed_order.order_request.price or Decimal("50000")  # Mock price
            
            # Update order state
            managed_order.filled_quantity += fill_quantity
            managed_order.remaining_quantity -= fill_quantity
            
            # Calculate average fill price
            if managed_order.average_fill_price:
                # Weighted average
                total_value = (managed_order.average_fill_price * (managed_order.filled_quantity - fill_quantity)) + (fill_price * fill_quantity)
                managed_order.average_fill_price = total_value / managed_order.filled_quantity
            else:
                managed_order.average_fill_price = fill_price
                
            # Record fill
            fill_record = {
                "timestamp": datetime.now(timezone.utc),
                "quantity": fill_quantity,
                "price": fill_price,
                "cumulative_quantity": managed_order.filled_quantity,
                "remaining_quantity": managed_order.remaining_quantity
            }
            managed_order.fills.append(fill_record)
            
            await self._add_order_event(
                managed_order,
                "partial_fill",
                {
                    "fill_quantity": float(fill_quantity),
                    "fill_price": float(fill_price),
                    "cumulative_filled": float(managed_order.filled_quantity),
                    "remaining": float(managed_order.remaining_quantity)
                }
            )
            
            # Update statistics
            if managed_order.filled_quantity >= managed_order.order_request.quantity:
                self.order_statistics["completed_orders"] += 1
            else:
                self.order_statistics["partially_filled_orders"] += 1
                
            # Call fill callback
            if managed_order.on_fill_callback:
                await managed_order.on_fill_callback(managed_order, fill_record)
                
        except Exception as e:
            self.logger.error(f"Partial fill handling failed: {e}")

    async def _add_order_event(
        self, 
        managed_order: ManagedOrder, 
        event_type: str, 
        data: dict[str, Any]
    ) -> None:
        """Add an event to the order's lifecycle history."""
        event = OrderLifecycleEvent(
            event_type=event_type,
            order_id=managed_order.order_id or "unknown",
            timestamp=datetime.now(timezone.utc),
            data=data
        )
        managed_order.events.append(event)
        managed_order.updated_at = event.timestamp
        
        self.logger.debug(
            "Order event added",
            order_id=managed_order.order_id,
            event_type=event_type,
            execution_id=managed_order.execution_id
        )

    @log_calls
    async def cancel_order(self, order_id: str, reason: str = "manual") -> bool:
        """
        Cancel a managed order.
        
        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason
            
        Returns:
            bool: True if cancellation successful
        """
        try:
            if order_id not in self.managed_orders:
                self.logger.warning(f"Order not found for cancellation: {order_id}")
                return False
                
            managed_order = self.managed_orders[order_id]
            
            if managed_order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                self.logger.warning(f"Cannot cancel order in status: {managed_order.status.value}")
                return False
                
            await self._add_order_event(
                managed_order,
                "cancellation_requested",
                {"reason": reason}
            )
            
            # Note: In a real implementation, this would call exchange.cancel_order()
            # For now, simulate cancellation
            managed_order.status = OrderStatus.CANCELLED
            managed_order.updated_at = datetime.now(timezone.utc)
            
            await self._add_order_event(managed_order, "order_cancelled", {"reason": reason})
            
            # Stop monitoring
            if order_id in self.monitoring_tasks:
                self.monitoring_tasks[order_id].cancel()
                del self.monitoring_tasks[order_id]
                
            self.logger.info(f"Order cancelled: {order_id}, reason: {reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Order cancellation failed: {e}")
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus | None:
        """Get current status of a managed order."""
        if order_id in self.managed_orders:
            return self.managed_orders[order_id].status
        return None

    async def get_managed_order(self, order_id: str) -> ManagedOrder | None:
        """Get managed order by ID."""
        return self.managed_orders.get(order_id)

    async def get_execution_orders(self, execution_id: str) -> list[ManagedOrder]:
        """Get all orders for an execution."""
        order_ids = self.execution_orders.get(execution_id, [])
        return [self.managed_orders[oid] for oid in order_ids if oid in self.managed_orders]

    async def _update_average_fill_time(self, fill_time_seconds: float) -> None:
        """Update average fill time statistics."""
        try:
            total_completed = self.order_statistics["completed_orders"]
            if total_completed == 1:
                self.order_statistics["average_fill_time_seconds"] = fill_time_seconds
            else:
                # Running average
                current_avg = self.order_statistics["average_fill_time_seconds"]
                new_avg = ((current_avg * (total_completed - 1)) + fill_time_seconds) / total_completed
                self.order_statistics["average_fill_time_seconds"] = new_avg
                
        except Exception as e:
            self.logger.warning(f"Fill time update failed: {e}")

    async def _cleanup_old_orders(self) -> None:
        """Clean up old completed orders to prevent memory growth."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.order_history_retention_hours)
            orders_to_remove = []
            
            for order_id, managed_order in self.managed_orders.items():
                # Remove old completed orders
                if (managed_order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED] and
                    managed_order.updated_at < cutoff_time):
                    orders_to_remove.append(order_id)
                    
            # Remove orders
            for order_id in orders_to_remove:
                # Cancel monitoring task if exists
                if order_id in self.monitoring_tasks:
                    self.monitoring_tasks[order_id].cancel()
                    del self.monitoring_tasks[order_id]
                    
                # Remove from tracking
                del self.managed_orders[order_id]
                
                # Remove from execution mapping
                for execution_id, order_list in self.execution_orders.items():
                    if order_id in order_list:
                        order_list.remove(order_id)
                        
            if orders_to_remove:
                self.logger.info(f"Cleaned up {len(orders_to_remove)} old orders")
                
        except Exception as e:
            self.logger.error(f"Order cleanup failed: {e}")

    @log_calls
    async def get_order_manager_summary(self) -> dict[str, Any]:
        """Get comprehensive order manager summary."""
        try:
            active_orders = [
                order for order in self.managed_orders.values()
                if order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]
            ]
            
            # Calculate performance metrics
            total_orders = self.order_statistics["total_orders"]
            success_rate = 0.0
            if total_orders > 0:
                success_rate = self.order_statistics["completed_orders"] / total_orders
                
            rejection_rate = 0.0
            if total_orders > 0:
                rejection_rate = self.order_statistics["rejected_orders"] / total_orders
                
            cancel_rate = 0.0
            if total_orders > 0:
                cancel_rate = self.order_statistics["cancelled_orders"] / total_orders
                
            return {
                "active_orders": len(active_orders),
                "total_managed_orders": len(self.managed_orders),
                "monitoring_tasks": len(self.monitoring_tasks),
                "executions_tracked": len(self.execution_orders),
                
                "performance_metrics": {
                    "total_orders": total_orders,
                    "success_rate": success_rate,
                    "rejection_rate": rejection_rate,
                    "cancellation_rate": cancel_rate,
                    "average_fill_time_seconds": self.order_statistics["average_fill_time_seconds"],
                    "total_volume": float(self.order_statistics["total_volume"]),
                    "total_fees": float(self.order_statistics["total_fees"])
                },
                
                "configuration": {
                    "default_timeout_minutes": self.default_order_timeout_minutes,
                    "status_check_interval_seconds": self.status_check_interval_seconds,
                    "max_concurrent_orders": self.max_concurrent_orders,
                    "retention_hours": self.order_history_retention_hours
                },
                
                "alerts": await self._check_alert_conditions()
            }
            
        except Exception as e:
            self.logger.error(f"Order manager summary generation failed: {e}")
            return {"error": str(e)}

    async def _check_alert_conditions(self) -> list[str]:
        """Check for alert conditions in order management."""
        alerts = []
        
        try:
            total_orders = self.order_statistics["total_orders"]
            
            if total_orders >= 10:  # Only check alerts if we have enough data
                # High rejection rate
                rejection_rate = self.order_statistics["rejected_orders"] / total_orders
                if rejection_rate > self.alert_thresholds["high_rejection_rate"]:
                    alerts.append(f"High rejection rate: {rejection_rate:.1%}")
                    
                # High cancellation rate
                cancel_rate = self.order_statistics["cancelled_orders"] / total_orders
                if cancel_rate > self.alert_thresholds["high_cancel_rate"]:
                    alerts.append(f"High cancellation rate: {cancel_rate:.1%}")
                    
                # Slow fill times
                avg_fill_time = self.order_statistics["average_fill_time_seconds"]
                if avg_fill_time > 300:  # 5 minutes
                    alerts.append(f"Slow average fill time: {avg_fill_time:.1f}s")
                    
        except Exception as e:
            self.logger.warning(f"Alert condition check failed: {e}")
            
        return alerts

    async def shutdown(self) -> None:
        """Shutdown order manager and cleanup resources."""
        try:
            # Cancel all monitoring tasks
            for task in self.monitoring_tasks.values():
                task.cancel()
                
            # Cancel cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                
            # Wait for tasks to complete
            if self.monitoring_tasks:
                await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)
                
            self.logger.info("Order manager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Order manager shutdown failed: {e}")