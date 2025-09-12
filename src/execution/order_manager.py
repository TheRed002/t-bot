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

from src.core.base.component import BaseComponent

# Caching imports
from src.core.caching import CacheKeys, cached, get_cache_manager
from src.core.config import Config
from src.core.exceptions import (
    ExchangeConnectionError,
    ExchangeError,
    ExchangeInsufficientFundsError,
    ExchangeRateLimitError,
    ExecutionError,
    NetworkError,
    ServiceError,
    StateError,
    ValidationError,
)

# MANDATORY: Import from P-001
# Import state management
from src.core.types import (
    MarketData,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    PositionStatus,
    StateType,
)

# Import error handling decorators
from src.error_handling.decorators import with_circuit_breaker, with_error_context, with_retry

# Import exchange interfaces
from src.execution.exchange_interface import ExchangeFactoryInterface, ExchangeInterface

# Import idempotency manager
from src.execution.idempotency_manager import OrderIdempotencyManager

# Import monitoring components
from src.monitoring import get_tracer

# MANDATORY: Import from P-007A
from src.utils import format_decimal, log_calls, time_execution
from src.utils.execution_utils import validate_order_basic


class OrderRouteInfo:
    """Information about order routing decisions."""

    def __init__(
        self,
        selected_exchange: str,
        alternative_exchanges: list[str],
        routing_reason: str,
        expected_cost_bps: Decimal,
        expected_execution_time_seconds: float,
    ) -> None:
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
    ) -> None:
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


class OrderManager(BaseComponent):
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

    def __init__(
        self,
        config: Config,
        exchange_service=None,  # ExchangeService dependency injection
        redis_client=None,
        state_service: Any | None = None,
        metrics_collector: Any | None = None,
    ):
        """
        Initialize order manager.

        Args:
            config: Application configuration
            exchange_service: ExchangeService for exchange operations (injected)
            redis_client: Optional Redis client for idempotency management
            state_service: Optional StateService for state persistence
            metrics_collector: Optional metrics collector for monitoring
        """
        super().__init__()  # Initialize BaseComponent
        self.config = config
        self.exchange_service = exchange_service  # Store injected service
        self.state_service = state_service
        self.metrics_collector = metrics_collector

        # Initialize tracer for distributed tracing with safety check
        try:
            self._tracer = get_tracer("execution.order_manager")
        except Exception as e:
            self._logger.warning(f"Failed to initialize tracer: {e}")
            self._tracer = None

        # Initialize idempotency manager
        self.idempotency_manager = OrderIdempotencyManager(config, redis_client)

        # Initialize cache manager
        self.cache_manager = get_cache_manager(config=config)

        # Order tracking with thread safety
        self._order_lock = RLock()
        self.managed_orders: dict[str, Any] = {}  # order_id -> ManagedOrder
        self.execution_orders: dict[str, list[str]] = {}  # execution_id -> [order_ids]
        self.positions: dict[str, Position] = {}  # symbol -> Position

        # P-020 Enhanced tracking
        self.symbol_orders: dict[str, list[str]] = defaultdict(list)  # symbol -> [order_ids]
        self.pending_aggregation: dict[str, list[str]] = defaultdict(list)  # symbol -> [order_ids]
        self.routing_decisions: dict[str, Any] = {}  # order_id -> route info
        self.websocket_connections: dict[str, Any] = {}  # exchange -> websocket connection
        self.order_modifications: dict[str, list[Any]] = defaultdict(list)

        # Management configuration
        self.default_order_timeout_minutes = config.execution.get("order_timeout_minutes", 60)
        self.status_check_interval_seconds = config.execution.get(
            "status_check_interval_seconds", 5
        )
        self.max_concurrent_orders = config.execution.get("max_concurrent_orders", 100)
        self.order_history_retention_hours = config.execution.get(
            "order_history_retention_hours", 24
        )

        # Exchange routing configuration
        self.routing_config = config.execution.get(
            "routing",
            {
                "large_order_exchange": None,  # Will be determined dynamically
                "usdt_preferred_exchange": None,  # Will be determined dynamically
                "default_exchange": None,  # Will be determined dynamically
                "large_order_threshold": "100000",
            },
        )

        # P-020 Enhanced configuration
        self.order_aggregation_rules: dict[str, Any] = {}
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

        # Task management
        self._cleanup_task: asyncio.Task | None = None
        self._cleanup_started = False
        self._is_running = False
        self._background_tasks: set[asyncio.Task] = set()
        self._websocket_tasks: dict[str, asyncio.Task] = {}  # exchange -> task

        # Status check tracking
        self._last_status_check: datetime | None = None

        self._logger.info(
            "Order manager initialized with lifecycle management",
            has_state_service=self.state_service is not None,
        )

        # Register cleanup for proper resource management
        import weakref

        weakref.finalize(self, self._cleanup_on_del)

    async def start(self) -> None:
        """Start the order manager and its background tasks."""
        if not self._cleanup_started:
            self._is_running = True

            # Start idempotency manager
            await self.idempotency_manager.start()

            self._start_cleanup_task()

            # Start WebSocket connections for real-time updates
            if self.websocket_enabled:
                await self._initialize_websocket_connections()

            # Restore orders from StateService if available
            if self.state_service:
                await self._restore_orders_from_state()

            self._cleanup_started = True
            self._logger.info("Order manager started with enhanced P-020 features and idempotency")

    def _start_cleanup_task(self) -> None:
        """Start the periodic cleanup task."""

        async def cleanup_periodically() -> None:
            while self._is_running:
                try:
                    await asyncio.sleep(self.config.execution.cleanup_interval_hours * 3600)
                    if self._is_running:  # Check again after sleep
                        await self._cleanup_old_orders()
                except asyncio.CancelledError:
                    self._logger.debug("Cleanup task cancelled")
                    break
                except Exception as e:
                    self._logger.error(f"Cleanup task error: {e}")
                    # Continue running unless explicitly stopped
                    if not self._is_running:
                        break

        self._cleanup_task = asyncio.create_task(cleanup_periodically())
        self._background_tasks.add(self._cleanup_task)

        # Clean up completed tasks
        self._cleanup_task.add_done_callback(self._background_tasks.discard)

    @with_error_context(component="OrderManager", operation="submit_order")
    @with_circuit_breaker(failure_threshold=10, recovery_timeout=60)
    @with_retry(max_attempts=3, exceptions=(NetworkError, ExchangeError, StateError))
    @time_execution
    async def submit_order(
        self,
        order_request: OrderRequest,
        exchange: ExchangeInterface,
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
            # Validate order using shared utilities
            if not order_request:
                raise ValidationError("Invalid order request")

            validation_errors = validate_order_basic(order_request)
            if validation_errors:
                raise ValidationError("; ".join(validation_errors))

            # Check order size limits
            if (
                hasattr(self.config.execution, "max_order_size")
                and self.config.execution.max_order_size
            ):
                if order_request.quantity > self.config.execution.max_order_size:
                    raise ValidationError(
                        f"Order size {order_request.quantity} exceeds maximum allowed size {self.config.execution.max_order_size}"
                    )

            if (
                hasattr(self.config.execution, "min_order_size")
                and self.config.execution.min_order_size
            ):
                if order_request.quantity < self.config.execution.min_order_size:
                    raise ValidationError(
                        f"Order size {order_request.quantity} is below minimum allowed size {self.config.execution.min_order_size}"
                    )

            # Check capacity
            if len(self.managed_orders) >= self.max_concurrent_orders:
                raise ExecutionError("Maximum concurrent orders reached")

            # Get or create idempotency key
            (
                client_order_id,
                is_duplicate,
            ) = await self.idempotency_manager.get_or_create_idempotency_key(
                order_request,
                metadata={
                    "execution_id": execution_id,
                    "exchange": exchange.exchange_name,
                    "created_by": "order_manager",
                },
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

                self._logger.warning(
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

            self._logger.info(
                "Submitting order",
                execution_id=execution_id,
                symbol=order_request.symbol,
                quantity=format_decimal(order_request.quantity),
                side=order_request.side.value,
                order_type=order_request.order_type.value,
            )

            # Submit to exchange service with comprehensive error handling
            order_response = None
            try:
                if self.exchange_service:
                    # Use service layer (proper pattern)
                    order_response = await self.exchange_service.place_order(
                        exchange_name=exchange.exchange_name, order=order_request
                    )
                else:
                    # Fallback to direct exchange call (legacy)
                    order_response = await exchange.place_order(order_request)
            except ExchangeRateLimitError as e:
                self._logger.error(
                    f"Rate limit error placing order: {e}",
                    exchange=exchange.exchange_name,
                    symbol=order_request.symbol,
                    error_type="rate_limit_error",
                )
                managed_order.status = OrderStatus.REJECTED
                await self._add_order_event(
                    managed_order, "order_rejected", {"reason": "rate_limit_error", "error": str(e)}
                )
                raise
            except ExchangeConnectionError as e:
                self._logger.error(
                    f"Connection error placing order: {e}",
                    exchange=exchange.exchange_name,
                    symbol=order_request.symbol,
                    error_type="connection_error",
                )
                managed_order.status = OrderStatus.FAILED
                await self._add_order_event(
                    managed_order, "order_failed", {"reason": "connection_error", "error": str(e)}
                )
                raise
            except ExchangeError as e:
                self._logger.error(
                    f"Exchange error placing order: {e}",
                    exchange=exchange.exchange_name,
                    symbol=order_request.symbol,
                    error_type="exchange_error",
                )
                managed_order.status = OrderStatus.REJECTED
                await self._add_order_event(
                    managed_order, "order_rejected", {"reason": "exchange_error", "error": str(e)}
                )
                raise
            except NetworkError as e:
                self._logger.error(
                    f"Network error placing order: {e}",
                    exchange=exchange.exchange_name,
                    symbol=order_request.symbol,
                    error_type="network_error",
                )
                managed_order.status = OrderStatus.FAILED
                await self._add_order_event(
                    managed_order, "order_failed", {"reason": "network_error", "error": str(e)}
                )
                raise
            finally:
                # Ensure exchange connection is properly handled on all paths
                if hasattr(exchange, "close"):
                    try:
                        await exchange.close()
                    except Exception as e:
                        self._logger.warning(
                            "Failed to close exchange connection cleanly", error=str(e)
                        )

            # Validate order response structure
            if not order_response:
                raise ExecutionError("Exchange returned null order response")

            if not hasattr(order_response, "id") or not order_response.id:
                raise ExecutionError("Order response missing required id field")

            # Update managed order with validated response
            managed_order.order_id = order_response.id
            managed_order.status = OrderStatus.PENDING

            # Register order
            self.managed_orders[order_response.id] = managed_order

            if execution_id not in self.execution_orders:
                self.execution_orders[execution_id] = []
            self.execution_orders[execution_id].append(order_response.id)

            # Persist order state if StateService is available
            if self.state_service:
                await self._persist_order_state(managed_order)

            # Add creation event
            await self._add_order_event(
                managed_order,
                "order_submitted",
                {
                    "exchange": exchange.exchange_name,
                    "order_response": {
                        "id": order_response.id,
                        "status": order_response.status,
                        "timestamp": order_response.created_at.isoformat(),
                    },
                },
            )

            # Start monitoring
            await self._start_order_monitoring(managed_order, exchange)

            # Update statistics
            self.order_statistics["total_orders"] += 1

            self._logger.info(
                "Order submitted successfully",
                order_id=order_response.id,
                execution_id=execution_id,
                client_order_id=client_order_id,
            )

            return managed_order

        except ValidationError as e:
            self._logger.error(f"Order validation failed: {e}")
            # Mark idempotency key as failed if we have a client_order_id
            if "client_order_id" in locals() and client_order_id:
                await self.idempotency_manager.mark_order_failed(client_order_id, str(e))
            raise  # Re-raise validation errors
        except ExchangeInsufficientFundsError as e:
            self._logger.error(f"Insufficient funds for order: {e}")
            if "client_order_id" in locals() and client_order_id:
                await self.idempotency_manager.mark_order_failed(client_order_id, str(e))
            raise  # Re-raise for proper handling
        except ExchangeError as e:
            self._logger.error(f"Exchange error during order submission: {e}")
            if "client_order_id" in locals() and client_order_id:
                await self.idempotency_manager.mark_order_failed(client_order_id, str(e))
            raise ExecutionError(f"Order submission failed: {e}") from e
        except Exception as e:
            self._logger.error(f"Unexpected error during order submission: {e}", exc_info=True)
            if "client_order_id" in locals() and client_order_id:
                await self.idempotency_manager.mark_order_failed(client_order_id, str(e))

            # Update statistics
            self.order_statistics["rejected_orders"] += 1

            # Call error callback if provided
            if "managed_order" in locals() and managed_order.on_error_callback:
                try:
                    await managed_order.on_error_callback(managed_order, str(e))
                except Exception as callback_error:
                    self._logger.error(f"Error callback failed: {callback_error}")

            raise ExecutionError(f"Order submission failed: {e}") from e

    # ========== P-020 Enhanced Order Management Methods ==========

    @log_calls
    async def submit_order_with_routing(
        self,
        order_request: OrderRequest,
        execution_id: str,
        exchange_service=None,  # Now uses ExchangeService instead of ExchangeFactory
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
            # Use exchange service (either injected or provided)
            service_to_use = exchange_service or self.exchange_service
            if not service_to_use:
                raise ExecutionError("No ExchangeService available for order routing")

            # Get routing decision using service
            route_info = await self._select_optimal_exchange_via_service(
                order_request, service_to_use, preferred_exchanges, market_data
            )

            # Submit order directly through service (bypassing direct exchange calls)
            order_response = await service_to_use.place_order(
                exchange_name=route_info.selected_exchange, order=order_request
            )

            # Create managed order and populate it
            managed_order = ManagedOrder(
                order_id=order_response.id,
                order_request=order_request,
                execution_id=execution_id,
                timeout_minutes=timeout_minutes or self.config.execution.default_timeout_minutes,
            )
            managed_order.order_response = order_response
            managed_order.status = order_response.status

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

            self._logger.info(
                "Order submitted with routing",
                order_id=managed_order.order_id,
                exchange=route_info.selected_exchange,
                routing_reason=route_info.routing_reason,
            )

            return managed_order

        except ValidationError as e:
            self._logger.error(f"Order validation failed: {e}")
            raise  # Re-raise validation errors
        except ExchangeInsufficientFundsError as e:
            self._logger.error(f"Insufficient funds for routed order: {e}")
            raise  # Re-raise for proper handling
        except ExchangeError as e:
            self._logger.error(f"Exchange error during routed order submission: {e}")
            raise ExecutionError(f"Order submission with routing failed: {e}") from e
        except Exception as e:
            self._logger.error(
                f"Unexpected error during routed order submission: {e}", exc_info=True
            )
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
                    self._logger.warning(f"Order not found for modification: {order_id}")
                    return False

                managed_order = self.managed_orders[order_id]

                # Check if order can be modified
                if managed_order.status not in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
                    self._logger.warning(
                        f"Cannot modify order in status: {managed_order.status.value}"
                    )
                    return False

                # Store modification request
                managed_order.modification_history.append(modification_request)
                self.order_modifications[order_id].append(modification_request)

                # Update order details
                old_quantity = None
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
                        "old_quantity": (str(old_quantity) if old_quantity is not None else None),
                        "new_quantity": (
                            str(modification_request.new_quantity)
                            if modification_request.new_quantity
                            else None
                        ),
                        "new_price": (
                            str(modification_request.new_price)
                            if modification_request.new_price
                            else None
                        ),
                        "reason": modification_request.modification_reason,
                    },
                )

                managed_order.updated_at = datetime.now(timezone.utc)

                # In a real implementation, this would call exchange.modify_order()
                # For now, simulate successful modification

                self._logger.info(
                    f"Order modified successfully: {order_id}, "
                    f"reason: {modification_request.modification_reason}"
                )
                return True

        except (ExchangeError, NetworkError, ValidationError) as e:
            self._logger.error(f"Order modification failed: {e}")
            return False
        except Exception as e:
            self._logger.error(f"Unexpected error in order modification: {e}")
            raise ExecutionError(f"Order modification failed unexpectedly: {e}") from e

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
                                "net_quantity": str(net_quantity),
                            },
                        )

                    aggregated_order.add_audit_entry(
                        "order_created_from_aggregation",
                        {
                            "child_order_count": len(pending_order_ids),
                            "net_quantity": str(net_quantity),
                            "symbol": symbol,
                        },
                    )

                    self._logger.info(
                        f"Orders aggregated for {symbol}: "
                        f"{len(pending_order_ids)} orders -> net {net_quantity}"
                    )

                    return aggregated_order

                return None

        except (ValidationError, ServiceError) as e:
            self._logger.error(f"Order aggregation failed: {e}")
            return None
        except Exception as e:
            self._logger.error(f"Unexpected error in order aggregation: {e}")
            # For aggregation, return None instead of raising to allow graceful degradation
            return None

    async def _initialize_websocket_connections(self) -> None:
        """Initialize WebSocket connections for real-time order updates."""
        try:
            # This would initialize WebSocket connections to exchanges
            # For now, simulate the initialization
            exchanges = self.config.execution.get("exchanges", ["binance", "coinbase", "okx"])

            # Use asyncio.gather to initialize connections concurrently
            tasks = []
            for exchange in exchanges:
                task = asyncio.create_task(self._initialize_single_websocket(exchange))
                tasks.append(task)

            if tasks:
                # Wait for all connections to initialize with timeout
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True), timeout=30.0
                    )
                except asyncio.TimeoutError:
                    self._logger.error(
                        "WebSocket initialization timeout - some connections may have failed"
                    )

        except Exception as e:
            self._logger.error(f"WebSocket initialization failed: {e}")

    async def _initialize_single_websocket(self, exchange: str) -> None:
        """Initialize a single WebSocket connection with proper async context handling."""
        connection_info = None
        try:
            # Use asyncio.wait_for for timeout handling
            await asyncio.wait_for(self._perform_websocket_connection(exchange), timeout=10.0)

        except asyncio.TimeoutError:
            self._logger.error(f"WebSocket connection timeout for {exchange}")
            if exchange in self.websocket_connections:
                self.websocket_connections[exchange]["status"] = "failed"
        except Exception as e:
            self._logger.warning(f"Failed to initialize WebSocket for {exchange}: {e}")
            if exchange in self.websocket_connections:
                self.websocket_connections[exchange]["status"] = "error"

    async def _perform_websocket_connection(self, exchange: str) -> None:
        """Perform the actual WebSocket connection setup."""
        connection_info = {
            "status": "connecting",
            "last_heartbeat": datetime.now(timezone.utc),
            "subscriptions": [],
            "reconnect_attempts": 0,
            "max_reconnect_attempts": self.websocket_reconnect_attempts,
            "connection_lock": asyncio.Lock(),  # Add connection-level lock
        }

        # Simulate connection establishment with backoff
        await asyncio.sleep(self.config.execution.order_processing_delay_seconds)
        connection_info["status"] = "connected"

        # Thread-safe update to connections dict
        async with asyncio.Lock():
            self.websocket_connections[exchange] = connection_info

        # Start WebSocket message handler and track it
        task = asyncio.create_task(
            self._handle_websocket_messages(exchange), name=f"websocket_handler_{exchange}"
        )
        self._websocket_tasks[exchange] = task
        self._background_tasks.add(task)

        # Clean up completed tasks with proper exception handling
        def cleanup_task(t: asyncio.Task) -> None:
            self._background_tasks.discard(t)
            if t.exception() and not t.cancelled():
                self._logger.warning(f"WebSocket task {exchange} failed: {t.exception()}")

        task.add_done_callback(cleanup_task)

        self._logger.info(f"WebSocket connection initialized for {exchange}")

    async def _handle_websocket_messages(self, exchange: str) -> None:
        """Handle incoming WebSocket messages for order updates with proper async handling."""
        connection_info = self.websocket_connections.get(exchange)
        last_heartbeat = datetime.now(timezone.utc)
        message_queue = asyncio.Queue(maxsize=1000)  # Add backpressure handling

        try:
            while self.websocket_enabled and self._is_running:
                try:
                    # Use connection lock to prevent race conditions
                    async with connection_info.get("connection_lock", asyncio.Lock()):
                        # Check connection status and attempt reconnect if needed
                        if connection_info and connection_info.get("status") != "connected":
                            reconnect_success = await self._attempt_websocket_reconnect(
                                exchange, connection_info
                            )
                            if not reconnect_success:
                                break

                    # Handle heartbeat with proper async timing
                    current_time = datetime.now(timezone.utc)
                    if (
                        current_time - last_heartbeat
                    ).total_seconds() >= self.websocket_heartbeat_interval:
                        await self._send_websocket_heartbeat(exchange, connection_info)
                        last_heartbeat = current_time

                    # Process message queue with timeout to prevent blocking
                    try:
                        # Simulate message processing with timeout
                        await asyncio.wait_for(
                            self._process_message_queue(exchange, message_queue), timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        # Normal timeout for message processing
                        pass

                    # Small async sleep to yield control
                    await asyncio.sleep(self.config.execution.order_processing_delay_seconds)

                    if not self._is_running:
                        break

                except asyncio.CancelledError:
                    self._logger.debug(f"WebSocket message handler cancelled for {exchange}")
                    break
                except Exception as e:
                    self._logger.error(f"Error in WebSocket message loop for {exchange}: {e}")
                    # Update connection status on error with lock
                    if connection_info:
                        async with connection_info.get("connection_lock", asyncio.Lock()):
                            connection_info["status"] = "error"
                    # Exponential backoff delay before retrying
                    await asyncio.sleep(
                        min(
                            self.config.execution.connection_retry_max_delay_seconds,
                            self.config.execution.connection_retry_delay_seconds
                            * (
                                self.config.execution.connection_retry_backoff_factor
                                ** connection_info.get("error_count", 0)
                            ),
                        )
                    )

        except asyncio.CancelledError:
            self._logger.debug(f"WebSocket handler cancelled for {exchange}")
        except Exception as e:
            self._logger.error(f"WebSocket handler error for {exchange}: {e}")
        finally:
            # Proper async cleanup
            await self._cleanup_websocket_connection(exchange, connection_info)

    async def _attempt_websocket_reconnect(self, exchange: str, connection_info: dict) -> bool:
        """Attempt to reconnect WebSocket with proper async handling."""
        if connection_info.get("reconnect_attempts", 0) < self.websocket_reconnect_attempts:
            connection_info["reconnect_attempts"] = connection_info.get("reconnect_attempts", 0) + 1
            self._logger.info(
                f"Attempting to reconnect WebSocket for {exchange} (attempt {connection_info['reconnect_attempts']})"
            )

            backoff_delay = min(
                self.config.execution.connection_retry_max_delay_seconds,
                self.config.execution.connection_retry_backoff_factor
                ** connection_info["reconnect_attempts"],
            )
            await asyncio.sleep(backoff_delay)

            try:
                # Use asyncio.wait_for for reconnection timeout
                await asyncio.wait_for(
                    self._perform_reconnection(exchange, connection_info), timeout=10.0
                )
                return True
            except asyncio.TimeoutError:
                self._logger.warning(f"WebSocket reconnection timeout for {exchange}")
                connection_info["status"] = "disconnected"
                return False
        else:
            self._logger.error(f"Max reconnection attempts reached for {exchange}")
            return False

    async def _perform_reconnection(self, exchange: str, connection_info: dict) -> None:
        """Perform the actual WebSocket reconnection."""
        # Simulate reconnection attempt
        await asyncio.sleep(self.config.execution.order_processing_delay_seconds)
        connection_info["status"] = "connected"
        connection_info["reconnect_attempts"] = 0
        connection_info["error_count"] = 0
        self._logger.info(f"WebSocket reconnected successfully for {exchange}")

    async def _send_websocket_heartbeat(self, exchange: str, connection_info: dict) -> None:
        """Send WebSocket heartbeat with proper async handling."""
        try:
            if connection_info and connection_info.get("status") == "connected":
                # In real implementation, this would send actual heartbeat
                connection_info["last_heartbeat"] = datetime.now(timezone.utc)
                self._logger.debug(f"Heartbeat sent for {exchange}")
        except Exception as e:
            self._logger.warning(f"Failed to send heartbeat for {exchange}: {e}")

    async def _process_message_queue(self, exchange: str, message_queue: asyncio.Queue) -> None:
        """Process WebSocket message queue with backpressure handling."""
        try:
            # In real implementation, this would process actual WebSocket messages
            # For now, just simulate message processing
            if not message_queue.empty():
                try:
                    message = await asyncio.wait_for(message_queue.get(), timeout=0.1)
                    # Process message here
                    message_queue.task_done()
                except asyncio.TimeoutError:
                    pass
        except Exception as e:
            self._logger.warning(f"Error processing message queue for {exchange}: {e}")

    async def _cleanup_websocket_connection(self, exchange: str, connection_info: dict) -> None:
        """Clean up WebSocket connection with proper async handling."""
        try:
            if connection_info:
                async with connection_info.get("connection_lock", asyncio.Lock()):
                    connection_info["status"] = "disconnected"

            self._logger.info(f"WebSocket connection closed for {exchange}")

        except Exception as cleanup_error:
            self._logger.warning(f"Error during WebSocket cleanup for {exchange}: {cleanup_error}")
        finally:
            # Remove from tracking dict
            if exchange in self._websocket_tasks:
                del self._websocket_tasks[exchange]

    async def _shutdown_websocket_connection(self, exchange: str, connection_info: dict) -> None:
        """Shutdown a single WebSocket connection with proper async handling."""
        try:
            if isinstance(connection_info, dict):
                # Use connection lock if available
                lock = connection_info.get("connection_lock", asyncio.Lock())
                async with lock:
                    connection_info["status"] = "disconnecting"

                # Simulate connection close with timeout
                try:
                    await asyncio.wait_for(
                        asyncio.sleep(
                            self.config.execution.order_processing_delay_seconds
                        ),  # Simulate close delay
                        timeout=5.0,
                    )
                    connection_info["status"] = "disconnected"
                    self._logger.info(f"WebSocket connection closed for {exchange}")
                except asyncio.TimeoutError:
                    self._logger.warning(f"WebSocket close timeout for {exchange}")
                    connection_info["status"] = "force_disconnected"

            elif hasattr(connection_info, "close"):
                # Handle actual WebSocket connection objects
                try:
                    if asyncio.iscoroutinefunction(connection_info.close):
                        await asyncio.wait_for(connection_info.close(), timeout=5.0)
                    else:
                        connection_info.close()
                    self._logger.info(f"WebSocket connection closed for {exchange}")
                except asyncio.TimeoutError:
                    self._logger.warning(f"WebSocket close timeout for {exchange}")
                except Exception as close_error:
                    self._logger.warning(
                        f"Error during connection close for {exchange}: {close_error}"
                    )

        except Exception as e:
            self._logger.warning(f"Error closing WebSocket for {exchange}: {e}")
        finally:
            # Ensure connection is removed from dict even if close fails
            try:
                if exchange in self.websocket_connections:
                    self.websocket_connections.pop(exchange, None)
            except Exception as e:
                self._logger.warning(
                    f"Failed to remove websocket connection from tracking for {exchange}: {e}"
                )

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
                        "filled_quantity": str(update.filled_quantity),
                        "remaining_quantity": str(update.remaining_quantity),
                    },
                )

                managed_order.filled_quantity = update.filled_quantity
                managed_order.remaining_quantity = update.remaining_quantity
                if update.average_price:
                    managed_order.average_fill_price = update.average_price

                # Trigger callbacks if status changed
                if old_status != update.status:
                    await self._handle_status_change(managed_order, old_status, update.status)

                self._logger.debug(
                    f"WebSocket order update processed: {update.order_id}, "
                    f"status: {update.status.value}"
                )

        except Exception as e:
            self._logger.error(f"WebSocket order update processing failed: {e}")

    async def _select_optimal_exchange_via_service(
        self,
        order_request: OrderRequest,
        exchange_service,  # ExchangeService instead of factory
        preferred_exchanges: list[str] | None = None,
        market_data: MarketData | None = None,
    ) -> OrderRouteInfo:
        """Select optimal exchange for order execution using ExchangeService."""
        try:
            # Get available exchanges from service if not specified
            if preferred_exchanges:
                available_exchanges = preferred_exchanges
            else:
                try:
                    available_exchanges = exchange_service.get_available_exchanges()
                except Exception as e:
                    # Fallback to common exchanges if service method not available
                    self._logger.warning(
                        "Failed to get available exchanges from service, using fallback",
                        error=str(e),
                    )
                    available_exchanges = ["binance", "coinbase", "okx"]

            # Select based on order size and symbol (same logic as original method)
            order_value = order_request.quantity * (
                order_request.price or market_data.price if market_data else Decimal("50000")
            )

            # Select exchange based on criteria (simplified logic)
            if order_value > Decimal("100000"):  # Large orders
                selected_exchange = next(
                    (ex for ex in available_exchanges if ex == "binance"), available_exchanges[0]
                )
            elif "BTC" in order_request.symbol.upper():  # Bitcoin trades
                selected_exchange = next(
                    (ex for ex in available_exchanges if ex in ["coinbase", "binance"]),
                    available_exchanges[0],
                )
            else:
                # Default to first available
                selected_exchange = available_exchanges[0]

            # Return routing info
            return OrderRouteInfo(
                selected_exchange=selected_exchange,
                alternative_exchanges=[ex for ex in available_exchanges if ex != selected_exchange],
                routing_reason="service_based_routing",
                expected_cost_bps=Decimal("0.1"),  # Default estimate
                expected_execution_time_seconds=2.0,  # Default estimate
            )

        except Exception as e:
            self._logger.error(f"Exchange selection failed: {e}")
            # Fallback to first available exchange
            fallback_exchange = preferred_exchanges[0] if preferred_exchanges else "binance"
            return OrderRouteInfo(
                selected_exchange=fallback_exchange,
                alternative_exchanges=[],
                routing_reason="fallback_selection",
                expected_cost_bps=Decimal("0.1"),
                expected_execution_time_seconds=5.0,
            )

    async def _select_optimal_exchange(
        self,
        order_request: OrderRequest,
        exchange_factory: ExchangeFactoryInterface,
        preferred_exchanges: list[str] | None = None,
        market_data: MarketData | None = None,
    ) -> OrderRouteInfo:
        """Select optimal exchange for order execution."""
        try:
            # Get available exchanges from factory if not specified
            if preferred_exchanges:
                available_exchanges = preferred_exchanges
            else:
                try:
                    available_exchanges = exchange_factory.get_available_exchanges()
                except Exception as e:
                    # Fallback to common exchanges if factory method not available
                    self._logger.warning(
                        "Failed to get available exchanges from factory, using fallback",
                        error=str(e),
                    )
                    available_exchanges = ["binance", "coinbase", "okx"]

            # Select based on order size and symbol
            order_value = order_request.quantity * (
                order_request.price or market_data.price if market_data else Decimal("50000")
            )

            # Determine exchange based on order characteristics
            selected_exchange = None

            if order_value > Decimal(self.routing_config["large_order_threshold"]):
                selected_exchange = self.routing_config.get("large_order_exchange")
                routing_reason = "large_order_routing"
                expected_cost_bps = Decimal("15")
                expected_time = 30.0
            elif order_request.symbol.endswith("USDT"):
                selected_exchange = self.routing_config.get("usdt_preferred_exchange")
                routing_reason = "usdt_pair_routing"
                expected_cost_bps = Decimal("10")
                expected_time = 20.0

            # If no specific preference or configured exchange not available, use first available
            if not selected_exchange or selected_exchange not in available_exchanges:
                selected_exchange = available_exchanges[0] if available_exchanges else "binance"
                routing_reason = "default_routing" if not selected_exchange else "fallback_routing"
                expected_cost_bps = Decimal("20")
                expected_time = 15.0

            alternative_exchanges = [ex for ex in available_exchanges if ex != selected_exchange]

            return OrderRouteInfo(
                selected_exchange=selected_exchange,
                alternative_exchanges=alternative_exchanges,
                routing_reason=routing_reason,
                expected_cost_bps=expected_cost_bps,
                expected_execution_time_seconds=expected_time,
            )

        except Exception as e:
            self._logger.error(f"Exchange selection failed: {e}")
            # Fallback to first available exchange
            return OrderRouteInfo(
                selected_exchange=preferred_exchanges[0] if preferred_exchanges else "binance",
                alternative_exchanges=[],
                routing_reason="error_fallback",
                expected_cost_bps=Decimal("25"),
                expected_execution_time_seconds=60.0,
            )

    async def _start_order_monitoring(
        self, managed_order: ManagedOrder, exchange: ExchangeInterface
    ) -> None:
        """Start monitoring an order's lifecycle."""

        async def monitor_order() -> None:
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
                        self._logger.warning(f"Order timeout reached: {order_id}")

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

                        except ExchangeRateLimitError as e:
                            self._logger.warning(
                                f"Rate limit hit during status check for {order_id}: {e}"
                            )
                            await asyncio.sleep(
                                e.retry_after
                                if hasattr(e, "retry_after")
                                else self.config.execution.rate_limit_retry_delay_seconds
                            )
                        except ExchangeConnectionError as e:
                            self._logger.warning(
                                f"Connection error during status check for {order_id}: {e}"
                            )
                            await asyncio.sleep(
                                self.config.execution.general_error_delay_seconds
                            )  # Brief pause before retry
                        except ExchangeError as e:
                            self._logger.warning(
                                f"Exchange error during status check for {order_id}: {e}"
                            )
                            # Log but continue monitoring
                        except Exception as e:
                            self._logger.error(
                                f"Unexpected error during status check for {order_id}: {e}"
                            )
                            # For truly unexpected errors, log and continue

                    await asyncio.sleep(
                        self.config.execution.busy_wait_prevention_delay_seconds
                    )  # Small sleep to prevent busy waiting

            except asyncio.CancelledError:
                self._logger.debug(f"Order monitoring cancelled for {managed_order.order_id}")
            except Exception as e:
                self._logger.error(f"Order monitoring failed: {e}")

                if managed_order.on_error_callback:
                    try:
                        await managed_order.on_error_callback(managed_order, str(e))
                    except Exception as callback_error:
                        self._logger.warning(
                            "Error callback execution failed", error=str(callback_error)
                        )

        # Start monitoring task and track it
        task = asyncio.create_task(monitor_order())
        if managed_order.order_id:
            self.monitoring_tasks[managed_order.order_id] = task
            self._background_tasks.add(task)

            # Clean up completed tasks
            task.add_done_callback(lambda t: self._background_tasks.discard(t))

    async def _check_order_status(
        self, managed_order: ManagedOrder, exchange: ExchangeInterface
    ) -> None:
        """Check and update order status."""
        try:
            if not managed_order.order_id:
                return

            # Get current status from exchange with error handling and rate limit awareness
            try:
                # Check if we should skip this check due to rate limiting
                if hasattr(self, "_last_status_check"):
                    time_since_last = (
                        datetime.now(timezone.utc) - self._last_status_check
                    ).total_seconds()
                    if (
                        time_since_last < self.config.execution.order_sync_delay_seconds
                    ):  # Minimum delay between status checks
                        await asyncio.sleep(
                            self.config.execution.order_sync_delay_seconds - time_since_last
                        )

                current_status = await exchange.get_order_status(managed_order.order_id)
                self._last_status_check = datetime.now(timezone.utc)

            except ExchangeRateLimitError as e:
                self._logger.warning(
                    f"Rate limit error checking order status: {e}",
                    order_id=managed_order.order_id,
                    error_type="rate_limit_error",
                )
                # Back off for longer on rate limit
                await asyncio.sleep(self.config.execution.connection_error_delay_seconds)
                return
            except ExchangeConnectionError as e:
                self._logger.warning(
                    f"Connection error checking order status: {e}",
                    order_id=managed_order.order_id,
                    error_type="connection_error",
                )
                # Connection issues - retry with backoff
                await asyncio.sleep(self.config.execution.general_error_delay_seconds)
                return
            except ExchangeError as e:
                self._logger.warning(
                    f"Exchange error checking order status: {e}",
                    order_id=managed_order.order_id,
                    error_type="exchange_error",
                )
                # Don't update status on error, will retry next time
                return
            except NetworkError as e:
                self._logger.warning(
                    f"Network error checking order status: {e}",
                    order_id=managed_order.order_id,
                    error_type="network_error",
                )
                # Don't update status on error, will retry next time
                return

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

        except ExchangeRateLimitError as e:
            self._logger.warning(f"Rate limit error during status check: {e}")
            raise  # Re-raise for proper handling upstream
        except ExchangeConnectionError as e:
            self._logger.warning(f"Connection error during status check: {e}")
            raise  # Re-raise for retry logic
        except ExchangeError as e:
            self._logger.warning(f"Exchange error during status check: {e}")
            # Don't raise - status check failure is not critical
        except Exception as e:
            self._logger.error(f"Unexpected error during status check: {e}", exc_info=True)
            # Don't raise - continue with current status

    async def _handle_status_change(
        self, managed_order: ManagedOrder, old_status: OrderStatus, new_status: OrderStatus
    ) -> None:
        """Handle order status changes."""
        try:
            # Persist state change if StateService is available
            if self.state_service:
                await self._persist_order_state(managed_order)

            if new_status == OrderStatus.FILLED:
                # Order fully filled
                managed_order.filled_quantity = managed_order.order_request.quantity
                managed_order.remaining_quantity = Decimal("0")

                await self._add_order_event(
                    managed_order,
                    "order_filled",
                    {"filled_quantity": str(managed_order.filled_quantity)},
                )

                # Mark idempotency key as completed
                if managed_order.order_request.client_order_id and managed_order.order_id:
                    # Create a minimal order response for idempotency tracking
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
            self._logger.error(f"Status change handling failed: {e}")

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
                    "fill_quantity": str(fill_quantity),
                    "fill_price": str(fill_price),
                    "cumulative_filled": str(managed_order.filled_quantity),
                    "remaining": str(managed_order.remaining_quantity),
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
            self._logger.error(f"Partial fill handling failed: {e}")

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

        self._logger.debug(
            "Order event added",
            order_id=managed_order.order_id,
            event_type=event_type,
            execution_id=managed_order.execution_id,
        )

    @log_calls
    @with_error_context(component="OrderManager", operation="cancel_order")
    @with_retry(max_attempts=3, exceptions=(NetworkError, ExchangeError, StateError))
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
                self._logger.warning(f"Order not found for cancellation: {order_id}")
                return False

            managed_order = self.managed_orders[order_id]

            if managed_order.status in [
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
            ]:
                self._logger.warning(f"Cannot cancel order in status: {managed_order.status.value}")
                return False

            await self._add_order_event(managed_order, "cancellation_requested", {"reason": reason})

            # Cancel order via exchange service
            try:
                if self.exchange_service and managed_order.order_response:
                    # Use service layer for cancellation
                    symbol = managed_order.order_request.symbol
                    exchange_name = None
                    if hasattr(managed_order, "route_info") and managed_order.route_info:
                        exchange_name = managed_order.route_info.selected_exchange
                    else:
                        # Try to get from order response or use fallback
                        exchange_name = getattr(managed_order.order_response, "exchange", "binance")

                    cancel_success = await self.exchange_service.cancel_order(
                        exchange_name=exchange_name, order_id=order_id, symbol=symbol
                    )

                    if cancel_success:
                        managed_order.status = OrderStatus.CANCELLED
                    else:
                        managed_order.status = OrderStatus.CANCEL_REJECTED
                else:
                    # Fallback: simulate cancellation if no service available
                    self._logger.warning(
                        "No exchange service available for cancellation, simulating"
                    )
                    managed_order.status = OrderStatus.CANCELLED

                managed_order.updated_at = datetime.now(timezone.utc)
            except ExchangeError as e:
                self._logger.error(
                    f"Exchange error cancelling order: {e}",
                    order_id=order_id,
                    error_type="exchange_error",
                )
                await self._add_order_event(
                    managed_order,
                    "cancellation_failed",
                    {"reason": "exchange_error", "error": str(e)},
                )
                return False
            except NetworkError as e:
                self._logger.error(
                    f"Network error cancelling order: {e}",
                    order_id=order_id,
                    error_type="network_error",
                )
                await self._add_order_event(
                    managed_order,
                    "cancellation_failed",
                    {"reason": "network_error", "error": str(e)},
                )
                return False

            await self._add_order_event(managed_order, "order_cancelled", {"reason": reason})

            # Stop monitoring
            if order_id in self.monitoring_tasks:
                self.monitoring_tasks[order_id].cancel()
                del self.monitoring_tasks[order_id]

            self._logger.info(f"Order cancelled: {order_id}, reason: {reason}")
            return True

        except (ExchangeError, NetworkError, ValidationError) as e:
            self._logger.error(f"Order cancellation failed: {e}")
            return False
        except Exception as e:
            self._logger.error(f"Unexpected error in order cancellation: {e}")
            # For cancellation, return False to indicate failure
            return False

    @cached(
        ttl=5,
        namespace="orders",
        data_type="orders",
        key_generator=lambda self, order_id: CacheKeys.active_orders("single", order_id),
    )
    async def get_order_status(self, order_id: str) -> OrderStatus | None:
        """Get current status of a managed order."""
        if order_id in self.managed_orders:
            return self.managed_orders[order_id].status
        return None

    async def get_managed_order(self, order_id: str) -> ManagedOrder | None:
        """Get managed order by ID."""
        return self.managed_orders.get(order_id)

    @cached(
        ttl=10,
        namespace="orders",
        data_type="orders",
        key_generator=lambda self, execution_id: CacheKeys.execution_state(
            "execution", execution_id
        ),
    )
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
            self._logger.warning(f"Fill time update failed: {e}")

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
                self._logger.info(f"Cleaned up {len(orders_to_remove)} old orders")

        except Exception as e:
            self._logger.error(f"Order cleanup failed: {e}")

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
            self._logger.error(f"Audit trail retrieval failed: {e}")
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
        self._logger.info(
            f"Aggregation rule set for {symbol}: {min_order_count} orders, "
            f"{aggregation_window_seconds}s window"
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
            exchange_counts: dict[str, int] = defaultdict(int)
            routing_reasons: dict[str, int] = defaultdict(int)
            total_cost_bps = Decimal("0")
            total_execution_time = 0.0
            count = 0

            for route_info in self.routing_decisions.values():
                exchange_counts[route_info.selected_exchange] += 1
                routing_reasons[route_info.routing_reason] += 1
                total_cost_bps += route_info.expected_cost_bps
                total_execution_time += route_info.expected_execution_time_seconds
                count += 1

            avg_cost_bps = str(total_cost_bps / count) if count > 0 else "0.0"
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
            self._logger.error(f"Routing statistics generation failed: {e}")
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
                        "buy_quantity": str(buy_quantity),
                        "sell_quantity": str(sell_quantity),
                        "net_quantity": str(net_quantity),
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
            self._logger.error(f"Aggregation opportunities analysis failed: {e}")
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
                        "quantity": str(order.order_request.quantity),
                        "price": (
                            str(order.order_request.price) if order.order_request.price else None
                        ),
                        "status": order.status.value,
                        "filled_quantity": str(order.filled_quantity),
                        "remaining_quantity": str(order.remaining_quantity),
                        "average_fill_price": (
                            str(order.average_fill_price) if order.average_fill_price else None
                        ),
                        "total_fees": str(order.total_fees),
                        "created_at": order.created_at.isoformat(),
                        "updated_at": order.updated_at.isoformat(),
                        "routing_info": (
                            {
                                "selected_exchange": order.route_info.selected_exchange,
                                "routing_reason": order.route_info.routing_reason,
                                "expected_cost_bps": str(order.route_info.expected_cost_bps),
                            }
                            if order.route_info
                            else None
                        ),
                        "modifications_count": len(order.modification_history),
                        "parent_order_id": order.parent_order_id,
                        "child_order_count": len(order.child_order_ids),
                        "net_position_impact": str(order.net_position_impact),
                        "compliance_tags": order.compliance_tags,
                        "audit_entries_count": len(order.audit_trail),
                    }

                    history.append(record)

            # Sort by creation time
            history.sort(key=lambda x: x["created_at"])

            return history

        except Exception as e:
            self._logger.error(f"Order history export failed: {e}")
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
                    "total_volume": str(self.order_statistics["total_volume"]),
                    "total_fees": str(self.order_statistics["total_fees"]),
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
            self._logger.error(f"Order manager summary generation failed: {e}")
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
            self._logger.warning(f"Alert condition check failed: {e}")

        return alerts

    # ========== State Persistence Methods ==========

    async def _persist_order_state(self, managed_order: ManagedOrder) -> None:
        """
        Persist order state to StateService.

        Args:
            managed_order: Order to persist
        """
        if not self.state_service or not managed_order.order_id:
            return

        try:
            # Prepare state data
            state_data = {
                "order_id": managed_order.order_id,
                "execution_id": managed_order.execution_id,
                "order_request": {
                    "symbol": managed_order.order_request.symbol,
                    "side": managed_order.order_request.side.value,
                    "order_type": managed_order.order_request.order_type.value,
                    "quantity": str(managed_order.order_request.quantity),
                    "price": (
                        str(managed_order.order_request.price)
                        if managed_order.order_request.price
                        else None
                    ),
                    "time_in_force": managed_order.order_request.time_in_force,
                    "client_order_id": managed_order.order_request.client_order_id,
                },
                "status": managed_order.status.value,
                "filled_quantity": str(managed_order.filled_quantity),
                "remaining_quantity": str(managed_order.remaining_quantity),
                "average_fill_price": (
                    str(managed_order.average_fill_price)
                    if managed_order.average_fill_price
                    else None
                ),
                "total_fees": str(managed_order.total_fees),
                "created_at": managed_order.created_at.isoformat(),
                "updated_at": managed_order.updated_at.isoformat(),
                "timeout_minutes": managed_order.timeout_minutes,
                "retry_count": managed_order.retry_count,
                "parent_order_id": managed_order.parent_order_id,
                "child_order_ids": managed_order.child_order_ids,
                "audit_trail": managed_order.audit_trail,
                "events": [
                    {
                        "event_type": event.event_type,
                        "timestamp": event.timestamp.isoformat(),
                        "data": event.data,
                    }
                    for event in managed_order.events
                ],
            }

            # Add routing info if available
            if managed_order.route_info:
                state_data["route_info"] = {
                    "selected_exchange": managed_order.route_info.selected_exchange,
                    "routing_reason": managed_order.route_info.routing_reason,
                    "expected_cost_bps": str(managed_order.route_info.expected_cost_bps),
                }

            # Save to StateService
            await self.state_service.set_state(
                state_type=StateType.ORDER_STATE,
                state_id=f"order_{managed_order.order_id}",
                state_data=state_data,
                source_component="OrderManager",
                reason=f"Order state update: {managed_order.status.value}",
            )

            self._logger.debug(
                "Order state persisted",
                order_id=managed_order.order_id,
                status=managed_order.status.value,
            )

        except Exception as e:
            self._logger.error(
                f"Failed to persist order state: {e}", order_id=managed_order.order_id
            )
            # CRITICAL: Propagate state persistence failures to maintain data integrity
            raise ExecutionError(
                f"Failed to persist order state for {managed_order.order_id}: {e!s}"
            ) from e

    async def _restore_orders_from_state(self) -> None:
        """Restore active orders from StateService on startup."""
        if not self.state_service:
            return

        try:
            # Get all execution states
            execution_states = await self.state_service.get_states_by_type(
                state_type=StateType.ORDER_STATE,
                include_metadata=True,
            )

            restored_count = 0
            for state_item in execution_states:
                try:
                    # Extract state data and metadata
                    if isinstance(state_item, dict) and "data" in state_item:
                        state_data = state_item["data"]
                        metadata = state_item.get("metadata", {})
                    else:
                        state_data = state_item
                        metadata = {}

                    # Get state_id from metadata
                    state_id = metadata.get("state_id", "")

                    # Skip if not an order state
                    if not state_id.startswith("order_"):
                        continue

                    # Restore ManagedOrder from state
                    order_id = state_data.get("order_id")
                    if not order_id or order_id in self.managed_orders:
                        continue

                    # Skip if not an active order
                    status = state_data.get("status", "")
                    if status not in ["PENDING", "PARTIAL", "PARTIALLY_FILLED"]:
                        continue

                    # Recreate OrderRequest
                    order_req_data = state_data.get("order_request", {})
                    order_request = OrderRequest(
                        symbol=order_req_data.get("symbol", ""),
                        side=OrderSide(order_req_data.get("side", "BUY")),
                        order_type=OrderType(order_req_data.get("order_type", "MARKET")),
                        quantity=Decimal(str(order_req_data.get("quantity", 0))),
                        price=(
                            Decimal(str(order_req_data.get("price", 0)))
                            if order_req_data.get("price")
                            else None
                        ),
                        time_in_force=order_req_data.get("time_in_force"),
                        client_order_id=order_req_data.get("client_order_id"),
                    )

                    # Recreate ManagedOrder
                    managed_order = ManagedOrder(
                        order_request=order_request,
                        execution_id=state_data.get("execution_id", ""),
                    )

                    # Restore state
                    managed_order.order_id = order_id
                    managed_order.status = OrderStatus(state_data.get("status", "PENDING"))
                    managed_order.filled_quantity = Decimal(
                        str(state_data.get("filled_quantity", 0))
                    )
                    managed_order.remaining_quantity = Decimal(
                        str(state_data.get("remaining_quantity", 0))
                    )
                    managed_order.average_fill_price = (
                        Decimal(str(state_data.get("average_fill_price")))
                        if state_data.get("average_fill_price")
                        else None
                    )
                    managed_order.total_fees = Decimal(str(state_data.get("total_fees", 0)))
                    managed_order.created_at = datetime.fromisoformat(state_data.get("created_at"))
                    managed_order.updated_at = datetime.fromisoformat(state_data.get("updated_at"))
                    managed_order.timeout_minutes = state_data.get("timeout_minutes")
                    managed_order.retry_count = state_data.get("retry_count", 0)
                    managed_order.parent_order_id = state_data.get("parent_order_id")
                    managed_order.child_order_ids = state_data.get("child_order_ids", [])
                    managed_order.audit_trail = state_data.get("audit_trail", [])

                    # Restore events
                    for event_data in state_data.get("events", []):
                        event = OrderLifecycleEvent(
                            event_type=event_data.get("event_type", "unknown"),
                            order_id=order_id,
                            timestamp=datetime.fromisoformat(event_data.get("timestamp")),
                            data=event_data.get("data", {}),
                        )
                        managed_order.events.append(event)

                    # Register restored order
                    self.managed_orders[order_id] = managed_order

                    # Restore execution mapping
                    execution_id = managed_order.execution_id
                    if execution_id not in self.execution_orders:
                        self.execution_orders[execution_id] = []
                    self.execution_orders[execution_id].append(order_id)

                    # Restore symbol mapping
                    self.symbol_orders[managed_order.order_request.symbol].append(order_id)

                    restored_count += 1

                    self._logger.debug(
                        "Order restored from state",
                        order_id=order_id,
                        status=managed_order.status.value,
                        symbol=managed_order.order_request.symbol,
                    )

                except Exception as e:
                    self._logger.warning(f"Failed to restore order {state_id}: {e}")

            if restored_count > 0:
                self._logger.info(f"Restored {restored_count} orders from state")

        except Exception as e:
            self._logger.error(f"Failed to restore orders from state: {e}")

    async def stop(self) -> None:
        """Stop the order manager and cleanup all resources."""
        try:
            self._logger.info("Stopping order manager...")

            # Signal shutdown to all background tasks
            self._is_running = False
            self.websocket_enabled = False
            self._cleanup_started = False

            # Cancel WebSocket tasks first
            websocket_tasks_to_cancel = list(self._websocket_tasks.values())
            for task in websocket_tasks_to_cancel:
                if not task.done():
                    task.cancel()

            # Wait for WebSocket tasks to complete
            if websocket_tasks_to_cancel:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*websocket_tasks_to_cancel, return_exceptions=True),
                        timeout=3.0,
                    )
                except asyncio.TimeoutError:
                    self._logger.warning("WebSocket tasks did not complete within timeout")

            self._websocket_tasks.clear()

            # Close WebSocket connections with proper async cleanup
            if self.websocket_connections:
                cleanup_tasks = []
                for exchange, connection_info in list(self.websocket_connections.items()):
                    cleanup_task = asyncio.create_task(
                        self._shutdown_websocket_connection(exchange, connection_info),
                        name=f"websocket_cleanup_{exchange}",
                    )
                    cleanup_tasks.append(cleanup_task)

                if cleanup_tasks:
                    try:
                        # Use asyncio.gather with timeout for parallel cleanup
                        await asyncio.wait_for(
                            asyncio.gather(*cleanup_tasks, return_exceptions=True), timeout=10.0
                        )
                    except asyncio.TimeoutError:
                        self._logger.warning("WebSocket connection cleanup timeout")

                # Clear the connections dict after cleanup
                self.websocket_connections.clear()

            # Cancel all monitoring tasks
            monitoring_tasks_to_cancel = list(self.monitoring_tasks.values())
            for task in monitoring_tasks_to_cancel:
                if not task.done():
                    task.cancel()

            # Cancel cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()

            # Cancel all background tasks
            if self._background_tasks:
                self._logger.debug(f"Cancelling {len(self._background_tasks)} background tasks")
                for task in list(self._background_tasks):
                    if not task.done():
                        task.cancel()

            # Wait for all tasks to complete with timeout
            all_tasks = list(self._background_tasks) + monitoring_tasks_to_cancel
            if self._cleanup_task:
                all_tasks.append(self._cleanup_task)

            if all_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*all_tasks, return_exceptions=True), timeout=10.0
                    )
                except asyncio.TimeoutError:
                    self._logger.warning("Some tasks did not complete within shutdown timeout")

            # Clear all task tracking
            self.monitoring_tasks.clear()
            self._background_tasks.clear()
            self._cleanup_task = None

            # Shutdown idempotency manager
            if hasattr(self, "idempotency_manager") and self.idempotency_manager:
                try:
                    await self.idempotency_manager.stop()
                except Exception as e:
                    self._logger.warning(f"Error stopping idempotency manager: {e}")
                finally:
                    self.idempotency_manager = None

            # Export final order history for compliance
            try:
                final_history = await self.export_order_history()
                self._logger.info(f"Final order history exported: {len(final_history)} orders")
            except Exception as e:
                self._logger.warning(f"Final history export failed: {e}")
            finally:
                # Clear remaining resources
                try:
                    self.managed_orders.clear()
                    self.execution_orders.clear()
                    self.websocket_connections.clear()
                except Exception as e:
                    self._logger.warning(
                        "Failed to clear order manager collections during shutdown", error=str(e)
                    )

            self._logger.info("Order manager stopped successfully")

        except Exception as e:
            self._logger.error(f"Order manager stop failed: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown order manager (alias for stop)."""
        await self.stop()

    # Position Management Methods
    async def _update_position_on_fill(self, order) -> None:
        """Update position when an order is filled."""
        try:
            # Handle both ManagedOrder and Order types
            if hasattr(order, "order_request"):
                # ManagedOrder
                symbol = order.order_request.symbol
                filled_qty = order.filled_quantity
                fill_price = order.average_fill_price
                order_side = order.order_request.side
            else:
                # Order
                symbol = order.symbol
                filled_qty = order.filled_quantity
                fill_price = order.average_price
                order_side = order.side

            if filled_qty <= 0 or fill_price is None:
                return

            with self._order_lock:
                if symbol in self.positions:
                    # Update existing position
                    position = self.positions[symbol]
                    old_qty = position.quantity
                    old_price = position.entry_price

                    if order_side == OrderSide.BUY:
                        new_qty = old_qty + filled_qty
                        # Calculate weighted average price
                        if new_qty > 0:
                            new_price = (
                                (old_qty * old_price) + (filled_qty * fill_price)
                            ) / new_qty
                        else:
                            new_price = fill_price
                    else:  # SELL
                        new_qty = old_qty - filled_qty
                        new_price = old_price  # Keep same entry price for sells

                    # Update position
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        side=position.side,
                        quantity=new_qty,
                        entry_price=new_price,
                        status=PositionStatus.OPEN if new_qty > 0 else PositionStatus.CLOSED,
                        opened_at=position.opened_at,
                        exchange=getattr(order, "exchange", "unknown"),
                        metadata=position.metadata,
                    )
                else:
                    # Create new position
                    side = PositionSide.LONG if order_side == OrderSide.BUY else PositionSide.SHORT
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        side=side,
                        quantity=filled_qty,
                        entry_price=fill_price,
                        status=PositionStatus.OPEN,
                        opened_at=datetime.now(timezone.utc),
                        exchange=getattr(order, "exchange", "unknown"),
                        metadata={},
                    )

        except Exception as e:
            self._logger.error(f"Failed to update position on fill: {e}")

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol."""
        with self._order_lock:
            return self.positions.get(symbol)

    def get_all_positions(self) -> list[Position]:
        """Get all positions."""
        with self._order_lock:
            return list(self.positions.values())

    def _cleanup_on_del(self) -> None:
        """Emergency cleanup when object is deleted."""
        try:
            # Cancel cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()

            # Cancel all monitoring tasks
            for task in self.monitoring_tasks.values():
                if not task.done():
                    task.cancel()

            # Cancel WebSocket tasks
            for task in self._websocket_tasks.values():
                if not task.done():
                    task.cancel()

            # Cancel all background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
        except Exception as e:
            self._logger.warning("Error during emergency cleanup", error=str(e))
