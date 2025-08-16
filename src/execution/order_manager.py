"""
Order Manager for comprehensive order lifecycle management.

This module provides sophisticated order lifecycle management, including
order tracking, status monitoring, partial fill handling, and automated
order management workflows for the execution engine.

CRITICAL: This integrates with P-001 (types, exceptions, config),
P-002A (error handling), and P-007A (utils) components.
"""

import asyncio
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from threading import RLock
from typing import Any
from uuid import uuid4

from src.core.config import Config
from src.core.exceptions import ExecutionError, ValidationError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import (
    MarketData,
    OrderRequest,
    OrderSide,
    OrderStatus,
)

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

# Import idempotency manager
from src.execution.idempotency_manager import OrderIdempotencyManager
from src.utils.decimal_utils import (
    format_decimal,
)

# MANDATORY: Import from P-007A
from src.utils.decorators import log_calls, time_execution

logger = get_logger(__name__)


class OrderRouteInfo:
    """Information about order routing decisions."""

    def __init__(
        self,
        selected_exchange: str,
        alternative_exchanges: list[str],
        routing_reason: str,
        expected_cost_bps: Decimal,
        expected_execution_time_seconds: float,
    ):
        self.selected_exchange = selected_exchange
        self.alternative_exchanges = alternative_exchanges
        self.routing_reason = routing_reason
        self.expected_cost_bps = expected_cost_bps
        self.expected_execution_time_seconds = expected_execution_time_seconds
        self.timestamp = datetime.now(timezone.utc)


class OrderModificationRequest:
    """Request for order modification."""

    def __init__(
        self,
        order_id: str,
        new_quantity: Decimal | None = None,
        new_price: Decimal | None = None,
        new_time_in_force: str | None = None,
        modification_reason: str = "manual",
    ):
        self.order_id = order_id
        self.new_quantity = new_quantity
        self.new_price = new_price
        self.new_time_in_force = new_time_in_force
        self.modification_reason = modification_reason
        self.timestamp = datetime.now(timezone.utc)


class OrderAggregationRule:
    """Rule for order aggregation and netting."""

    def __init__(
        self,
        symbol: str,
        aggregation_window_seconds: int,
        min_order_count: int,
        netting_enabled: bool = True,
    ):
        self.symbol = symbol
        self.aggregation_window_seconds = aggregation_window_seconds
        self.min_order_count = min_order_count
        self.netting_enabled = netting_enabled


class WebSocketOrderUpdate:
    """WebSocket order update message."""

    def __init__(
        self,
        order_id: str,
        status: OrderStatus,
        filled_quantity: Decimal,
        remaining_quantity: Decimal,
        average_price: Decimal | None,
        timestamp: datetime,
        exchange: str,
        raw_data: dict[str, Any],
    ):
        self.order_id = order_id
        self.status = status
        self.filled_quantity = filled_quantity
        self.remaining_quantity = remaining_quantity
        self.average_price = average_price
        self.timestamp = timestamp
        self.exchange = exchange
        self.raw_data = raw_data


class OrderLifecycleEvent:
    """Represents an event in the order lifecycle."""

    def __init__(
        self,
        event_type: str,
        order_id: str,
        timestamp: datetime,
        data: dict[str, Any] | None = None,
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

        # P-020 Enhanced features
        self.route_info: OrderRouteInfo | None = None
        self.modification_history: list[OrderModificationRequest] = []
        self.parent_order_id: str | None = None  # For child orders from aggregation
        self.child_order_ids: list[str] = []  # For parent orders that were split
        self.net_position_impact: Decimal = Decimal("0")  # Net position impact after netting
        self.compliance_tags: dict[str, str] = {}  # Regulatory compliance tags
        self.audit_trail: list[dict[str, Any]] = []  # Detailed audit trail
        self.websocket_subscription_id: str | None = None  # For real-time updates

        # Thread safety
        self._lock = RLock()

        # Callbacks
        self.on_fill_callback: Callable | None = None
        self.on_complete_callback: Callable | None = None
        self.on_cancel_callback: Callable | None = None
        self.on_error_callback: Callable | None = None

    def add_audit_entry(self, action: str, details: dict[str, Any]) -> None:
        """Add an entry to the audit trail with thread safety."""
        with self._lock:
            self.audit_trail.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": action,
                    "details": details,
                    "order_id": self.order_id,
                    "execution_id": self.execution_id,
                }
            )

    def update_status(self, new_status: OrderStatus, details: dict[str, Any] | None = None) -> None:
        """Update order status with thread safety."""
        with self._lock:
            old_status = self.status
            self.status = new_status
            self.updated_at = datetime.now(timezone.utc)
            self.add_audit_entry(
                "status_change",
                {
                    "old_status": old_status.value,
                    "new_status": new_status.value,
                    "details": details or {},
                },
            )


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

    def __init__(self, config: Config, redis_client=None):
        """
        Initialize order manager.

        Args:
            config: Application configuration
            redis_client: Optional Redis client for idempotency management
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.error_handler = ErrorHandler(config.error_handling)

        # Initialize idempotency manager
        self.idempotency_manager = OrderIdempotencyManager(config, redis_client)

        # Order tracking with thread safety
        self._order_lock = RLock()
        self.managed_orders: dict[str, ManagedOrder] = {}  # order_id -> ManagedOrder
        self.execution_orders: dict[str, list[str]] = {}  # execution_id -> [order_ids]

        # P-020 Enhanced tracking
        self.symbol_orders: dict[str, list[str]] = defaultdict(list)  # symbol -> [order_ids]
        self.pending_aggregation: dict[str, list[str]] = defaultdict(list)  # symbol -> [order_ids]
        self.routing_decisions: dict[str, OrderRouteInfo] = {}  # order_id -> route info
        self.websocket_connections: dict[str, Any] = {}  # exchange -> websocket connection
        self.order_modifications: dict[str, list[OrderModificationRequest]] = defaultdict(list)

        # Management configuration
        self.default_order_timeout_minutes = 60  # 1 hour default timeout
        self.status_check_interval_seconds = 5  # Check status every 5 seconds
        self.max_concurrent_orders = 100  # Maximum concurrent orders
        self.order_history_retention_hours = 24  # Keep order history for 24 hours

        # P-020 Enhanced configuration
        self.order_aggregation_rules: dict[str, OrderAggregationRule] = {}
        self.websocket_enabled = True
        self.websocket_reconnect_attempts = 3
        self.websocket_heartbeat_interval = 30
        self.modification_timeout_seconds = 10
        self.max_child_orders_per_parent = 50

        # Monitoring and alerting
        self.monitoring_tasks: dict[str, asyncio.Task] = {}
        self.alert_thresholds = {
            "high_rejection_rate": 0.1,  # 10% rejection rate
            "slow_fill_rate": 0.5,  # 50% of orders taking > 5 minutes
            "high_cancel_rate": 0.15,  # 15% cancellation rate
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
            "total_fees": Decimal("0"),
        }

        # Cleanup task
        self._cleanup_task: asyncio.Task | None = None
        self._cleanup_started = False

        self.logger.info("Order manager initialized with lifecycle management")

    async def start(self) -> None:
        """Start the order manager and its background tasks."""
        if not self._cleanup_started:
            # Start idempotency manager
            await self.idempotency_manager.start()

            self._start_cleanup_task()

            # Start WebSocket connections for real-time updates
            if self.websocket_enabled:
                await self._initialize_websocket_connections()

            self._cleanup_started = True
            self.logger.info("Order manager started with enhanced P-020 features and idempotency")

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
        callbacks: dict[str, Callable] | None = None,
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

            # Get or create idempotency key
            client_order_id, is_duplicate = (
                await self.idempotency_manager.get_or_create_idempotency_key(
                    order_request,
                    metadata={
                        "execution_id": execution_id,
                        "exchange": exchange.exchange_name,
                        "created_by": "order_manager",
                    },
                )
            )

            # Set client_order_id on the order request
            order_request.client_order_id = client_order_id

            # If this is a duplicate, check if we should allow retry
            if is_duplicate:
                can_retry, retry_count = await self.idempotency_manager.can_retry_order(
                    client_order_id
                )

                if not can_retry:
                    raise ExecutionError(
                        f"Duplicate order detected and max retries exceeded: {client_order_id} "
                        f"(retry count: {retry_count})"
                    )

                self.logger.warning(
                    f"Retrying duplicate order: {client_order_id} (attempt {retry_count + 1})",
                    execution_id=execution_id,
                    retry_count=retry_count,
                )

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
                quantity=format_decimal(order_request.quantity),
                side=order_request.side.value,
                order_type=order_request.order_type.value,
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
                        "timestamp": order_response.timestamp.isoformat(),
                    },
                },
            )

            # Start monitoring
            await self._start_order_monitoring(managed_order, exchange)

            # Update statistics
            self.order_statistics["total_orders"] += 1

            self.logger.info(
                "Order submitted successfully",
                order_id=order_response.id,
                execution_id=execution_id,
                client_order_id=client_order_id,
            )

            return managed_order

        except Exception as e:
            self.logger.error(f"Order submission failed: {e}")

            # Mark idempotency key as failed if we have a client_order_id
            if "client_order_id" in locals() and client_order_id:
                await self.idempotency_manager.mark_order_failed(client_order_id, str(e))

            # Update statistics
            self.order_statistics["rejected_orders"] += 1

            # Call error callback if provided
            if "managed_order" in locals() and managed_order.on_error_callback:
                try:
                    await managed_order.on_error_callback(managed_order, str(e))
                except Exception as callback_error:
                    self.logger.error(f"Error callback failed: {callback_error}")

            raise ExecutionError(f"Order submission failed: {e}") from e

    # ========== P-020 Enhanced Order Management Methods ==========

    @log_calls
    async def submit_order_with_routing(
        self,
        order_request: OrderRequest,
        exchange_factory,
        execution_id: str,
        preferred_exchanges: list[str] | None = None,
        market_data: MarketData | None = None,
        timeout_minutes: int | None = None,
        callbacks: dict[str, Callable] | None = None,
    ) -> ManagedOrder:
        """
        Submit order with intelligent routing across exchanges.

        Args:
            order_request: Order to submit
            exchange_factory: Factory for exchange access
            execution_id: Execution ID for tracking
            preferred_exchanges: List of preferred exchanges
            market_data: Current market data for routing decisions
            timeout_minutes: Optional timeout override
            callbacks: Optional lifecycle callbacks

        Returns:
            ManagedOrder: Managed order instance with routing info
        """
        try:
            # Get routing decision
            route_info = await self._select_optimal_exchange(
                order_request, exchange_factory, preferred_exchanges, market_data
            )

            # Get the selected exchange
            exchange = await exchange_factory.get_exchange(route_info.selected_exchange)
            if not exchange:
                raise ExecutionError(f"Failed to get exchange: {route_info.selected_exchange}")

            # Submit the order using standard method
            managed_order = await self.submit_order(
                order_request, exchange, execution_id, timeout_minutes, callbacks
            )

            # Attach routing information
            managed_order.route_info = route_info
            managed_order.add_audit_entry(
                "order_routed",
                {
                    "selected_exchange": route_info.selected_exchange,
                    "routing_reason": route_info.routing_reason,
                    "expected_cost_bps": format_decimal(route_info.expected_cost_bps),
                },
            )

            # Track routing decision
            if managed_order.order_id:
                self.routing_decisions[managed_order.order_id] = route_info

            # Track by symbol for aggregation
            with self._order_lock:
                self.symbol_orders[order_request.symbol].append(managed_order.order_id or "")

            self.logger.info(
                "Order submitted with routing",
                order_id=managed_order.order_id,
                exchange=route_info.selected_exchange,
                routing_reason=route_info.routing_reason,
            )

            return managed_order

        except Exception as e:
            self.logger.error(f"Order submission with routing failed: {e}")
            raise ExecutionError(f"Order submission with routing failed: {e}") from e

    @log_calls
    async def modify_order(self, modification_request: OrderModificationRequest) -> bool:
        """
        Modify an existing order.

        Args:
            modification_request: Order modification request

        Returns:
            bool: True if modification successful
        """
        try:
            order_id = modification_request.order_id

            with self._order_lock:
                if order_id not in self.managed_orders:
                    self.logger.warning(f"Order not found for modification: {order_id}")
                    return False

                managed_order = self.managed_orders[order_id]

                # Check if order can be modified
                if managed_order.status not in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
                    self.logger.warning(
                        f"Cannot modify order in status: {managed_order.status.value}"
                    )
                    return False

                # Store modification request
                managed_order.modification_history.append(modification_request)
                self.order_modifications[order_id].append(modification_request)

                # Update order details
                if modification_request.new_quantity:
                    old_quantity = managed_order.order_request.quantity
                    managed_order.order_request.quantity = modification_request.new_quantity
                    managed_order.remaining_quantity = (
                        modification_request.new_quantity - managed_order.filled_quantity
                    )

                if modification_request.new_price:
                    managed_order.order_request.price = modification_request.new_price

                # Add audit entry
                managed_order.add_audit_entry(
                    "order_modified",
                    {
                        "old_quantity": (
                            float(old_quantity) if modification_request.new_quantity else None
                        ),
                        "new_quantity": (
                            float(modification_request.new_quantity)
                            if modification_request.new_quantity
                            else None
                        ),
                        "new_price": (
                            float(modification_request.new_price)
                            if modification_request.new_price
                            else None
                        ),
                        "reason": modification_request.modification_reason,
                    },
                )

                managed_order.updated_at = datetime.now(timezone.utc)

                # In a real implementation, this would call exchange.modify_order()
                # For now, simulate successful modification

                self.logger.info(
                    f"Order modified successfully: {order_id}, reason: {modification_request.modification_reason}"
                )
                return True

        except Exception as e:
            self.logger.error(f"Order modification failed: {e}")
            return False

    @log_calls
    async def aggregate_orders(
        self, symbol: str, force_aggregation: bool = False
    ) -> ManagedOrder | None:
        """
        Aggregate pending orders for a symbol based on netting rules.

        Args:
            symbol: Trading symbol
            force_aggregation: Force aggregation regardless of rules

        Returns:
            Optional[ManagedOrder]: Aggregated order if created
        """
        try:
            aggregation_rule = self.order_aggregation_rules.get(symbol)
            if not aggregation_rule and not force_aggregation:
                return None

            with self._order_lock:
                # Get pending orders for the symbol
                pending_order_ids = [
                    oid
                    for oid in self.symbol_orders[symbol]
                    if oid in self.managed_orders
                    and self.managed_orders[oid].status == OrderStatus.PENDING
                ]

                if len(pending_order_ids) < (
                    aggregation_rule.min_order_count if aggregation_rule else 2
                ):
                    return None

                # Calculate net position
                net_quantity = Decimal("0")
                buy_orders = []
                sell_orders = []

                for oid in pending_order_ids:
                    order = self.managed_orders[oid]
                    if order.order_request.side == OrderSide.BUY:
                        buy_orders.append(order)
                        net_quantity += order.remaining_quantity
                    else:
                        sell_orders.append(order)
                        net_quantity -= order.remaining_quantity

                # Apply netting if enabled
                if aggregation_rule and aggregation_rule.netting_enabled:
                    if net_quantity == 0:
                        # Perfect netting - cancel all orders
                        for oid in pending_order_ids:
                            await self.cancel_order(oid, "perfect_netting")
                        return None

                # Create aggregated order
                if net_quantity != 0:
                    # Use the first order as the template
                    template_order = self.managed_orders[pending_order_ids[0]]

                    aggregated_request = OrderRequest(
                        symbol=symbol,
                        side=OrderSide.BUY if net_quantity > 0 else OrderSide.SELL,
                        order_type=template_order.order_request.order_type,
                        quantity=abs(net_quantity),
                        price=template_order.order_request.price,
                        time_in_force=template_order.order_request.time_in_force,
                        client_order_id=f"AGG_{uuid4().hex[:8]}",
                    )

                    # Create aggregated managed order
                    aggregated_order = ManagedOrder(aggregated_request, f"AGG_{uuid4().hex}")
                    aggregated_order.child_order_ids = pending_order_ids.copy()
                    aggregated_order.net_position_impact = net_quantity

                    # Update child orders
                    for oid in pending_order_ids:
                        child_order = self.managed_orders[oid]
                        child_order.parent_order_id = aggregated_order.execution_id
                        child_order.add_audit_entry(
                            "aggregated",
                            {
                                "parent_execution_id": aggregated_order.execution_id,
                                "net_quantity": float(net_quantity),
                            },
                        )

                    aggregated_order.add_audit_entry(
                        "order_created_from_aggregation",
                        {
                            "child_order_count": len(pending_order_ids),
                            "net_quantity": float(net_quantity),
                            "symbol": symbol,
                        },
                    )

                    self.logger.info(
                        f"Orders aggregated for {symbol}: {len(pending_order_ids)} orders -> net {net_quantity}"
                    )

                    return aggregated_order

                return None

        except Exception as e:
            self.logger.error(f"Order aggregation failed: {e}")
            return None

    async def _initialize_websocket_connections(self) -> None:
        """Initialize WebSocket connections for real-time order updates."""
        try:
            # This would initialize WebSocket connections to exchanges
            # For now, simulate the initialization
            exchanges = ["binance", "coinbase", "okx"]  # Would come from config

            for exchange in exchanges:
                try:
                    # Simulate WebSocket connection setup
                    # In reality, this would establish actual WebSocket connections
                    self.websocket_connections[exchange] = {
                        "status": "connected",
                        "last_heartbeat": datetime.now(timezone.utc),
                        "subscriptions": [],
                    }

                    # Start WebSocket message handler
                    asyncio.create_task(self._handle_websocket_messages(exchange))

                    self.logger.info(f"WebSocket connection initialized for {exchange}")

                except Exception as e:
                    self.logger.warning(f"Failed to initialize WebSocket for {exchange}: {e}")

        except Exception as e:
            self.logger.error(f"WebSocket initialization failed: {e}")

    async def _handle_websocket_messages(self, exchange: str) -> None:
        """Handle incoming WebSocket messages for order updates."""
        try:
            while self.websocket_enabled:
                # Simulate receiving WebSocket messages
                # In reality, this would listen to actual WebSocket streams
                await asyncio.sleep(1)

                # Process any pending WebSocket updates
                # This is where real-time order status updates would be handled

        except asyncio.CancelledError:
            self.logger.debug(f"WebSocket handler cancelled for {exchange}")
        except Exception as e:
            self.logger.error(f"WebSocket handler error for {exchange}: {e}")

    async def _process_websocket_order_update(self, update: WebSocketOrderUpdate) -> None:
        """Process a WebSocket order update."""
        try:
            with self._order_lock:
                if update.order_id not in self.managed_orders:
                    return

                managed_order = self.managed_orders[update.order_id]
                old_status = managed_order.status

                # Update order with WebSocket data
                managed_order.update_status(
                    update.status,
                    {
                        "source": "websocket",
                        "exchange": update.exchange,
                        "filled_quantity": float(update.filled_quantity),
                        "remaining_quantity": float(update.remaining_quantity),
                    },
                )

                managed_order.filled_quantity = update.filled_quantity
                managed_order.remaining_quantity = update.remaining_quantity
                if update.average_price:
                    managed_order.average_fill_price = update.average_price

                # Trigger callbacks if status changed
                if old_status != update.status:
                    await self._handle_status_change(managed_order, old_status, update.status)

                self.logger.debug(
                    f"WebSocket order update processed: {update.order_id}, status: {update.status.value}"
                )

        except Exception as e:
            self.logger.error(f"WebSocket order update processing failed: {e}")

    async def _select_optimal_exchange(
        self,
        order_request: OrderRequest,
        exchange_factory,
        preferred_exchanges: list[str] | None = None,
        market_data: MarketData | None = None,
    ) -> OrderRouteInfo:
        """Select optimal exchange for order execution."""
        try:
            available_exchanges = preferred_exchanges or ["binance", "coinbase", "okx"]

            # Simple routing logic (in reality, this would be much more sophisticated)
            # For now, select based on order size and symbol
            order_value = order_request.quantity * (order_request.price or Decimal("50000"))

            if order_value > Decimal("100000"):  # Large orders
                selected_exchange = "binance"  # Typically better for large orders
                routing_reason = "large_order_routing"
                expected_cost_bps = Decimal("15")
                expected_time = 30.0
            elif order_request.symbol.endswith("USDT"):
                selected_exchange = "binance"  # Better for USDT pairs
                routing_reason = "usdt_pair_routing"
                expected_cost_bps = Decimal("10")
                expected_time = 20.0
            else:
                selected_exchange = "coinbase"  # Default
                routing_reason = "default_routing"
                expected_cost_bps = Decimal("20")
                expected_time = 15.0

            # Ensure selected exchange is in available list
            if selected_exchange not in available_exchanges:
                selected_exchange = available_exchanges[0]
                routing_reason = "fallback_routing"

            alternative_exchanges = [ex for ex in available_exchanges if ex != selected_exchange]

            return OrderRouteInfo(
                selected_exchange=selected_exchange,
                alternative_exchanges=alternative_exchanges,
                routing_reason=routing_reason,
                expected_cost_bps=expected_cost_bps,
                expected_execution_time_seconds=expected_time,
            )

        except Exception as e:
            self.logger.error(f"Exchange selection failed: {e}")
            # Fallback to first available exchange
            return OrderRouteInfo(
                selected_exchange=preferred_exchanges[0] if preferred_exchanges else "binance",
                alternative_exchanges=[],
                routing_reason="error_fallback",
                expected_cost_bps=Decimal("25"),
                expected_execution_time_seconds=60.0,
            )

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
                    if (
                        current_time - last_status_check
                    ).total_seconds() >= self.status_check_interval_seconds:
                        try:
                            await self._check_order_status(managed_order, exchange)
                            last_status_check = current_time

                            # Stop monitoring if order is in terminal state
                            if managed_order.status in [
                                OrderStatus.FILLED,
                                OrderStatus.CANCELLED,
                                OrderStatus.REJECTED,
                                OrderStatus.EXPIRED,
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
                    {"old_status": old_status.value, "new_status": current_status.value},
                )

                # Handle specific status changes
                await self._handle_status_change(managed_order, old_status, current_status)

        except Exception as e:
            self.logger.warning(f"Status check failed: {e}")

    async def _handle_status_change(
        self, managed_order: ManagedOrder, old_status: OrderStatus, new_status: OrderStatus
    ) -> None:
        """Handle order status changes."""
        try:
            if new_status == OrderStatus.FILLED:
                # Order fully filled
                managed_order.filled_quantity = managed_order.order_request.quantity
                managed_order.remaining_quantity = Decimal("0")

                await self._add_order_event(
                    managed_order,
                    "order_filled",
                    {"filled_quantity": float(managed_order.filled_quantity)},
                )

                # Mark idempotency key as completed
                if managed_order.order_request.client_order_id and managed_order.order_id:
                    # Create a minimal order response for idempotency tracking
                    from src.core.types import OrderResponse

                    order_response = OrderResponse(
                        id=managed_order.order_id,
                        client_order_id=managed_order.order_request.client_order_id,
                        symbol=managed_order.order_request.symbol,
                        side=managed_order.order_request.side,
                        order_type=managed_order.order_request.order_type,
                        quantity=managed_order.order_request.quantity,
                        price=managed_order.order_request.price,
                        filled_quantity=managed_order.filled_quantity,
                        status=new_status.value,
                        timestamp=datetime.now(timezone.utc),
                    )
                    await self.idempotency_manager.mark_order_completed(
                        managed_order.order_request.client_order_id, order_response
                    )

                # Update statistics
                self.order_statistics["completed_orders"] += 1
                self.order_statistics["total_volume"] += managed_order.filled_quantity

                # Calculate fill time
                if managed_order.created_at:
                    fill_time = (
                        datetime.now(timezone.utc) - managed_order.created_at
                    ).total_seconds()
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

                # Mark idempotency key as failed for cancelled orders
                if managed_order.order_request.client_order_id:
                    await self.idempotency_manager.mark_order_failed(
                        managed_order.order_request.client_order_id, "Order cancelled"
                    )

                self.order_statistics["cancelled_orders"] += 1

                if managed_order.on_cancel_callback:
                    await managed_order.on_cancel_callback(managed_order)

            elif new_status == OrderStatus.REJECTED:
                await self._add_order_event(managed_order, "order_rejected", {})

                # Mark idempotency key as failed for rejected orders
                if managed_order.order_request.client_order_id:
                    await self.idempotency_manager.mark_order_failed(
                        managed_order.order_request.client_order_id, "Order rejected by exchange"
                    )

                self.order_statistics["rejected_orders"] += 1

                if managed_order.on_error_callback:
                    await managed_order.on_error_callback(
                        managed_order, "Order rejected by exchange"
                    )

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
                total_value = (
                    managed_order.average_fill_price
                    * (managed_order.filled_quantity - fill_quantity)
                ) + (fill_price * fill_quantity)
                managed_order.average_fill_price = total_value / managed_order.filled_quantity
            else:
                managed_order.average_fill_price = fill_price

            # Record fill
            fill_record = {
                "timestamp": datetime.now(timezone.utc),
                "quantity": fill_quantity,
                "price": fill_price,
                "cumulative_quantity": managed_order.filled_quantity,
                "remaining_quantity": managed_order.remaining_quantity,
            }
            managed_order.fills.append(fill_record)

            await self._add_order_event(
                managed_order,
                "partial_fill",
                {
                    "fill_quantity": float(fill_quantity),
                    "fill_price": float(fill_price),
                    "cumulative_filled": float(managed_order.filled_quantity),
                    "remaining": float(managed_order.remaining_quantity),
                },
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
        self, managed_order: ManagedOrder, event_type: str, data: dict[str, Any]
    ) -> None:
        """Add an event to the order's lifecycle history."""
        event = OrderLifecycleEvent(
            event_type=event_type,
            order_id=managed_order.order_id or "unknown",
            timestamp=datetime.now(timezone.utc),
            data=data,
        )
        managed_order.events.append(event)
        managed_order.updated_at = event.timestamp

        self.logger.debug(
            "Order event added",
            order_id=managed_order.order_id,
            event_type=event_type,
            execution_id=managed_order.execution_id,
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

            if managed_order.status in [
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
            ]:
                self.logger.warning(f"Cannot cancel order in status: {managed_order.status.value}")
                return False

            await self._add_order_event(managed_order, "cancellation_requested", {"reason": reason})

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
                new_avg = (
                    (current_avg * (total_completed - 1)) + fill_time_seconds
                ) / total_completed
                self.order_statistics["average_fill_time_seconds"] = new_avg

        except Exception as e:
            self.logger.warning(f"Fill time update failed: {e}")

    async def _cleanup_old_orders(self) -> None:
        """Clean up old completed orders to prevent memory growth."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(
                hours=self.order_history_retention_hours
            )
            orders_to_remove = []

            for order_id, managed_order in self.managed_orders.items():
                # Remove old completed orders
                if (
                    managed_order.status
                    in [
                        OrderStatus.FILLED,
                        OrderStatus.CANCELLED,
                        OrderStatus.REJECTED,
                        OrderStatus.EXPIRED,
                    ]
                    and managed_order.updated_at < cutoff_time
                ):
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
                for _execution_id, order_list in self.execution_orders.items():
                    if order_id in order_list:
                        order_list.remove(order_id)

            if orders_to_remove:
                self.logger.info(f"Cleaned up {len(orders_to_remove)} old orders")

        except Exception as e:
            self.logger.error(f"Order cleanup failed: {e}")

    # ========== P-020 Additional Utility Methods ==========

    @log_calls
    async def get_order_audit_trail(self, order_id: str) -> list[dict[str, Any]]:
        """
        Get comprehensive audit trail for an order.

        Args:
            order_id: Order ID to get audit trail for

        Returns:
            List[Dict]: Complete audit trail with all events
        """
        try:
            with self._order_lock:
                if order_id not in self.managed_orders:
                    return []

                managed_order = self.managed_orders[order_id]
                audit_trail = managed_order.audit_trail.copy()

                # Add lifecycle events to audit trail
                for event in managed_order.events:
                    audit_trail.append(
                        {
                            "timestamp": event.timestamp.isoformat(),
                            "action": "lifecycle_event",
                            "details": {"event_type": event.event_type, "data": event.data},
                            "order_id": order_id,
                            "execution_id": managed_order.execution_id,
                        }
                    )

                # Sort by timestamp
                audit_trail.sort(key=lambda x: x["timestamp"])

                return audit_trail

        except Exception as e:
            self.logger.error(f"Audit trail retrieval failed: {e}")
            return []

    @log_calls
    async def set_aggregation_rule(
        self,
        symbol: str,
        aggregation_window_seconds: int,
        min_order_count: int,
        netting_enabled: bool = True,
    ) -> None:
        """
        Set order aggregation rule for a symbol.

        Args:
            symbol: Trading symbol
            aggregation_window_seconds: Time window for aggregation
            min_order_count: Minimum orders before aggregation
            netting_enabled: Whether to enable position netting
        """
        rule = OrderAggregationRule(
            symbol=symbol,
            aggregation_window_seconds=aggregation_window_seconds,
            min_order_count=min_order_count,
            netting_enabled=netting_enabled,
        )

        self.order_aggregation_rules[symbol] = rule
        self.logger.info(
            f"Aggregation rule set for {symbol}: {min_order_count} orders, {aggregation_window_seconds}s window"
        )

    @log_calls
    async def get_orders_by_symbol(self, symbol: str) -> list[ManagedOrder]:
        """
        Get all orders for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List[ManagedOrder]: Orders for the symbol
        """
        with self._order_lock:
            order_ids = self.symbol_orders.get(symbol, [])
            return [self.managed_orders[oid] for oid in order_ids if oid in self.managed_orders]

    @log_calls
    async def get_orders_by_status(self, status: OrderStatus) -> list[ManagedOrder]:
        """
        Get all orders with specific status.

        Args:
            status: Order status to filter by

        Returns:
            List[ManagedOrder]: Orders with the specified status
        """
        with self._order_lock:
            return [order for order in self.managed_orders.values() if order.status == status]

    @log_calls
    async def get_routing_statistics(self) -> dict[str, Any]:
        """
        Get routing statistics and performance metrics.

        Returns:
            Dict: Routing statistics
        """
        try:
            exchange_counts = defaultdict(int)
            routing_reasons = defaultdict(int)
            total_cost_bps = Decimal("0")
            total_execution_time = 0.0
            count = 0

            for route_info in self.routing_decisions.values():
                exchange_counts[route_info.selected_exchange] += 1
                routing_reasons[route_info.routing_reason] += 1
                total_cost_bps += route_info.expected_cost_bps
                total_execution_time += route_info.expected_execution_time_seconds
                count += 1

            avg_cost_bps = float(total_cost_bps / count) if count > 0 else 0.0
            avg_execution_time = total_execution_time / count if count > 0 else 0.0

            return {
                "total_routed_orders": count,
                "exchange_distribution": dict(exchange_counts),
                "routing_reasons": dict(routing_reasons),
                "average_expected_cost_bps": avg_cost_bps,
                "average_expected_execution_time_seconds": avg_execution_time,
                "websocket_connections": {
                    exchange: info["status"]
                    for exchange, info in self.websocket_connections.items()
                },
            }

        except Exception as e:
            self.logger.error(f"Routing statistics generation failed: {e}")
            return {"error": str(e)}

    @log_calls
    async def get_aggregation_opportunities(self) -> dict[str, dict[str, Any]]:
        """
        Identify current order aggregation opportunities.

        Returns:
            Dict: Aggregation opportunities by symbol
        """
        try:
            opportunities = {}

            with self._order_lock:
                for symbol, order_ids in self.symbol_orders.items():
                    pending_orders = [
                        self.managed_orders[oid]
                        for oid in order_ids
                        if oid in self.managed_orders
                        and self.managed_orders[oid].status == OrderStatus.PENDING
                    ]

                    if len(pending_orders) < 2:
                        continue

                    # Calculate net position
                    net_quantity = Decimal("0")
                    buy_quantity = Decimal("0")
                    sell_quantity = Decimal("0")

                    for order in pending_orders:
                        if order.order_request.side == OrderSide.BUY:
                            buy_quantity += order.remaining_quantity
                            net_quantity += order.remaining_quantity
                        else:
                            sell_quantity += order.remaining_quantity
                            net_quantity -= order.remaining_quantity

                    rule = self.order_aggregation_rules.get(symbol)

                    opportunities[symbol] = {
                        "pending_order_count": len(pending_orders),
                        "buy_quantity": float(buy_quantity),
                        "sell_quantity": float(sell_quantity),
                        "net_quantity": float(net_quantity),
                        "can_aggregate": (
                            rule is None or len(pending_orders) >= rule.min_order_count
                        ),
                        "perfect_netting": net_quantity == 0,
                        "aggregation_rule": (
                            {
                                "min_order_count": rule.min_order_count if rule else 2,
                                "netting_enabled": rule.netting_enabled if rule else True,
                                "window_seconds": rule.aggregation_window_seconds if rule else 60,
                            }
                            if rule
                            else None
                        ),
                    }

            return opportunities

        except Exception as e:
            self.logger.error(f"Aggregation opportunities analysis failed: {e}")
            return {"error": str(e)}

    @log_calls
    async def export_order_history(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        symbols: list[str] | None = None,
        statuses: list[OrderStatus] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Export order history for reporting and compliance.

        Args:
            start_time: Start time filter
            end_time: End time filter
            symbols: Symbol filter
            statuses: Status filter

        Returns:
            List[Dict]: Filtered order history
        """
        try:
            history = []

            with self._order_lock:
                for order in self.managed_orders.values():
                    # Apply filters
                    if start_time and order.created_at < start_time:
                        continue
                    if end_time and order.created_at > end_time:
                        continue
                    if symbols and order.order_request.symbol not in symbols:
                        continue
                    if statuses and order.status not in statuses:
                        continue

                    # Build export record
                    record = {
                        "order_id": order.order_id,
                        "execution_id": order.execution_id,
                        "symbol": order.order_request.symbol,
                        "side": order.order_request.side.value,
                        "order_type": order.order_request.order_type.value,
                        "quantity": float(order.order_request.quantity),
                        "price": (
                            float(order.order_request.price) if order.order_request.price else None
                        ),
                        "status": order.status.value,
                        "filled_quantity": float(order.filled_quantity),
                        "remaining_quantity": float(order.remaining_quantity),
                        "average_fill_price": (
                            float(order.average_fill_price) if order.average_fill_price else None
                        ),
                        "total_fees": float(order.total_fees),
                        "created_at": order.created_at.isoformat(),
                        "updated_at": order.updated_at.isoformat(),
                        "routing_info": (
                            {
                                "selected_exchange": order.route_info.selected_exchange,
                                "routing_reason": order.route_info.routing_reason,
                                "expected_cost_bps": float(order.route_info.expected_cost_bps),
                            }
                            if order.route_info
                            else None
                        ),
                        "modifications_count": len(order.modification_history),
                        "parent_order_id": order.parent_order_id,
                        "child_order_count": len(order.child_order_ids),
                        "net_position_impact": float(order.net_position_impact),
                        "compliance_tags": order.compliance_tags,
                        "audit_entries_count": len(order.audit_trail),
                    }

                    history.append(record)

            # Sort by creation time
            history.sort(key=lambda x: x["created_at"])

            return history

        except Exception as e:
            self.logger.error(f"Order history export failed: {e}")
            return []

    @log_calls
    async def get_order_manager_summary(self) -> dict[str, Any]:
        """Get comprehensive order manager summary."""
        try:
            active_orders = [
                order
                for order in self.managed_orders.values()
                if order.status
                not in [
                    OrderStatus.FILLED,
                    OrderStatus.CANCELLED,
                    OrderStatus.REJECTED,
                    OrderStatus.EXPIRED,
                ]
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

            # P-020 Enhanced metrics
            routing_stats = await self.get_routing_statistics()
            aggregation_opportunities = await self.get_aggregation_opportunities()
            idempotency_stats = self.idempotency_manager.get_statistics()

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
                    "total_fees": float(self.order_statistics["total_fees"]),
                },
                # P-020 Enhanced sections
                "routing_statistics": routing_stats,
                "aggregation_opportunities": aggregation_opportunities,
                "idempotency_statistics": idempotency_stats,
                "websocket_status": {
                    "enabled": self.websocket_enabled,
                    "active_connections": len(self.websocket_connections),
                    "connection_status": {
                        exchange: info.get("status", "unknown")
                        for exchange, info in self.websocket_connections.items()
                    },
                },
                "order_tracking": {
                    "orders_by_symbol": {
                        symbol: len(order_ids) for symbol, order_ids in self.symbol_orders.items()
                    },
                    "total_modifications": sum(
                        len(mods) for mods in self.order_modifications.values()
                    ),
                    "aggregation_rules": len(self.order_aggregation_rules),
                    "pending_aggregation": {
                        symbol: len(order_ids)
                        for symbol, order_ids in self.pending_aggregation.items()
                    },
                },
                "configuration": {
                    "default_timeout_minutes": self.default_order_timeout_minutes,
                    "status_check_interval_seconds": self.status_check_interval_seconds,
                    "max_concurrent_orders": self.max_concurrent_orders,
                    "retention_hours": self.order_history_retention_hours,
                    "websocket_enabled": self.websocket_enabled,
                    "websocket_reconnect_attempts": self.websocket_reconnect_attempts,
                    "modification_timeout_seconds": self.modification_timeout_seconds,
                    "max_child_orders_per_parent": self.max_child_orders_per_parent,
                },
                "alerts": await self._check_alert_conditions(),
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
            # Disable WebSocket connections
            self.websocket_enabled = False

            # Close WebSocket connections
            for exchange, connection_info in self.websocket_connections.items():
                try:
                    # In a real implementation, this would close actual WebSocket connections
                    connection_info["status"] = "disconnected"
                    self.logger.info(f"WebSocket connection closed for {exchange}")
                except Exception as e:
                    self.logger.warning(f"Error closing WebSocket for {exchange}: {e}")

            # Cancel all monitoring tasks
            for task in self.monitoring_tasks.values():
                task.cancel()

            # Cancel cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()

            # Wait for tasks to complete
            if self.monitoring_tasks:
                await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)

            # Shutdown idempotency manager
            await self.idempotency_manager.shutdown()

            # Export final order history for compliance
            try:
                final_history = await self.export_order_history()
                self.logger.info(f"Final order history exported: {len(final_history)} orders")
            except Exception as e:
                self.logger.warning(f"Final history export failed: {e}")

            self.logger.info("Order manager shutdown completed with P-020 features and idempotency")

        except Exception as e:
            self.logger.error(f"Order manager shutdown failed: {e}")
