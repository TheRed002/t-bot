"""
Enhanced Base Exchange with Unified Infrastructure

This module provides a completely refactored BaseExchange with:
- Unified connection pooling and session management
- Advanced rate limiting with local and global enforcement
- Unified WebSocket management with auto-reconnection
- Comprehensive error handling and recovery
- Order management with tracking and callbacks
- Market data caching and optimization
- Health monitoring and automatic recovery
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

try:
    import aiohttp
except ImportError:
    aiohttp = None

from src.core.base import BaseComponent
from src.core.config import Config
from src.core.exceptions import (
    ExchangeConnectionError,
    ExchangeRateLimitError,
    OrderRejectionError,
    ValidationError,
)

# MANDATORY: Import from P-001
from src.core.types import (
    ExchangeInfo,
    MarketData,
    OrderBook,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    Position,
    Ticker,
    Trade,
)

# Database imports - avoiding direct model imports to prevent coupling
# Database operations should be handled by higher-level services
from src.database.redis_client import RedisClient
from src.error_handling.connection_manager import ConnectionManager as ErrorConnectionManager

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

# Recovery scenarios - imported at top level for better performance
from src.error_handling.recovery_scenarios import (
    NetworkDisconnectionRecovery,
    OrderRejectionRecovery,
)

# MANDATORY: Import from P-007 (advanced rate limiting)
from src.exchanges.advanced_rate_limiter import get_global_rate_limiter
from src.exchanges.connection_manager import ConnectionManager

# MANDATORY: Import from P-030 (monitoring)
from src.monitoring import ExchangeMetrics, MetricsCollector, SystemMetrics, get_tracer

# MANDATORY: Import from P-007A (utils)
from src.utils.decorators import UnifiedDecorator, time_execution

# OpenTelemetry imports for tracing
try:
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    # Create dummy classes if OpenTelemetry is not installed
    class StatusCode:
        ERROR = "ERROR"

    class Status:
        def __init__(self, code, description):
            self.code = code
            self.description = description


# State management imports - lazy loading to avoid circular dependency
if TYPE_CHECKING:
    from src.state import StateService, TradeLifecycleManager
else:
    StateService = Any
    TradeLifecycleManager = Any

# Import exchange interfaces to avoid circular dependencies
from src.exchanges.interfaces import (
    IStateService,
    ITradeLifecycleManager,
    StatePriority,
    StateType,
    TradeEvent,
)

# Create compatibility aliases for missing decorators
log_calls = UnifiedDecorator.enhance(log=True)
log_errors = UnifiedDecorator.enhance(log=True, log_level="error")
memory_usage = UnifiedDecorator.enhance(monitor=True)


class EnhancedBaseExchange(BaseComponent, ABC):
    """
    Enhanced base class for all exchange implementations with unified infrastructure.

    This class provides:
    - Unified connection pooling and session management
    - Advanced rate limiting with local and global enforcement
    - Unified WebSocket management with auto-reconnection
    - Comprehensive error handling and recovery
    - Order management with tracking and callbacks
    - Market data caching and optimization
    - Health monitoring and automatic recovery
    """

    def __init__(
        self,
        config: Config,
        exchange_name: str,
        state_service: IStateService | None = None,
        trade_lifecycle_manager: ITradeLifecycleManager | None = None,
        metrics_collector: Any = None,
    ):
        """
        Initialize the enhanced base exchange.

        Args:
            config: System configuration
            exchange_name: Name of the exchange (e.g., 'binance', 'okx')
            state_service: Optional state service for persistence
            trade_lifecycle_manager: Optional trade lifecycle manager
        """
        super().__init__()  # Initialize BaseComponent
        self.config = config
        self.exchange_name = exchange_name
        self.status = "initializing"
        self.connected = False
        self.last_heartbeat = None

        # State management services
        self.state_service = state_service
        self.trade_lifecycle_manager = trade_lifecycle_manager

        # Initialize error handling
        self.error_handler = ErrorHandler(config)
        self.error_connection_manager = ErrorConnectionManager(config)

        # P-007: Advanced rate limiting and connection management integration
        self.advanced_rate_limiter = get_global_rate_limiter(config)
        self.connection_manager = ConnectionManager(config, exchange_name)

        # Initialize database queries handler
        self.db_queries = None

        # Initialize Redis client for real-time data
        self.redis_client = None

        # Data module components removed to avoid circular dependency
        self.market_data_source = None

        # === ENHANCED INFRASTRUCTURE ===

        # Unified connection pool and session manager
        self.connection_pool = None
        self.session_manager = None
        self.request_session = None

        # Unified WebSocket manager
        self.unified_ws_manager = None

        # Unified rate limiter
        self.unified_rate_limiter = None

        # Connection health monitoring
        self.health_check_interval = 30  # seconds
        self.last_health_check = None
        self.connection_retries = 0
        self.max_connection_retries = 3

        # Order management
        self.pending_orders: dict[str, Any] = {}
        self.order_callbacks: dict[str, list[Callable]] = {}

        # Market data cache
        self.market_data_cache: dict[str, Any] = {}
        self.cache_ttl = 5  # seconds

        # WebSocket streams management
        self.active_streams: dict[str, Any] = {}
        self.stream_callbacks: dict[str, list[Callable]] = {}

        # Rate limiting tracking
        self.rate_limit_windows: dict[str, list[float]] = {}
        self.rate_limit_config = getattr(self.config.exchange, "rate_limits", {}).get(
            exchange_name, {}
        )

        # Initialize connection pool and session management (lazy initialization)
        self._init_task = None
        self._monitoring_tasks: list[asyncio.Task] = []
        self._connector_lock = None
        self._rate_limit_lock = None

        # Initialize monitoring components
        if metrics_collector is None:
            from src.monitoring import get_metrics_collector
            self.metrics_collector = get_metrics_collector()
        else:
            self.metrics_collector = metrics_collector
        self.exchange_metrics = ExchangeMetrics(self.metrics_collector)
        self.system_metrics = SystemMetrics(self.metrics_collector)
        self.tracer = get_tracer(__name__)

        # Mark as initialized
        self._initialized = True

        self.logger.info(
            f"Enhanced BaseExchange initialized for {exchange_name} with unified infrastructure"
        )

    # === CONNECTION INFRASTRUCTURE ===

    async def _initialize_connection_infrastructure(self) -> None:
        """
        Initialize connection pooling, session management, and unified components.
        """
        try:
            # Initialize connection pool
            self.connection_pool = await self._create_connection_pool()

            # Initialize session manager
            self.session_manager = await self._create_session_manager()

            # Initialize unified rate limiter
            self.unified_rate_limiter = await self._create_unified_rate_limiter()

            # Initialize unified WebSocket manager
            self.unified_ws_manager = await self._create_unified_websocket_manager()

            # Start health monitoring
            health_task = asyncio.create_task(self._health_monitor_loop())
            self._monitoring_tasks.append(health_task)

            self.logger.debug(f"Connection infrastructure initialized for {self.exchange_name}")
        except asyncio.CancelledError:
            self.logger.debug("Connection infrastructure initialization cancelled")
            raise
        except Exception as e:
            if aiohttp and isinstance(e, aiohttp.ClientError):
                self.logger.error(f"Network error during initialization: {e!s}")
            else:
                self.logger.error(f"Unexpected error during initialization: {e!s}")
            raise

    async def _create_connection_pool(self) -> Any:
        """
        Create connection pool for HTTP requests.
        """
        try:
            if not aiohttp:
                self.logger.warning("aiohttp not available, connection pool disabled")
                return None

            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=30,  # Connections per host
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
                keepalive_timeout=60,
                enable_cleanup_closed=True,
            )

            timeout = aiohttp.ClientTimeout(total=30, connect=10)

            self.request_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": f"T-Bot-{self.exchange_name}/1.0",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )

            # Start connection pool monitoring
            pool_task = asyncio.create_task(self._monitor_connection_pool(connector))
            self._monitoring_tasks.append(pool_task)

            return connector
        except Exception as e:
            self.logger.error(f"Failed to create connection pool: {e!s}")
            return None

    async def _create_session_manager(self) -> Any:
        """
        Create session manager for connection health monitoring.
        """
        return {
            "created_at": datetime.now(timezone.utc),
            "request_count": 0,
            "error_count": 0,
            "last_request": None,
            "health_status": "initializing",
        }

    async def _create_unified_rate_limiter(self) -> Any:
        """
        Create unified rate limiter for all endpoints.
        """
        return {
            "windows": {},
            "config": self.rate_limit_config,
            "violations": 0,
            "last_violation": None,
        }

    async def _create_unified_websocket_manager(self) -> Any:
        """
        Create unified WebSocket manager.
        """
        return {
            "connections": {},
            "subscriptions": {},
            "reconnect_attempts": {},
            "max_reconnect_attempts": 5,
            "reconnect_delay": 1.0,
        }

    async def _health_monitor_loop(self) -> None:
        """
        Continuous health monitoring loop.
        """
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)

                if self.connected:
                    # Perform health check with monitoring
                    start_time = time.time()
                    health_status = await self.health_check()
                    check_duration = time.time() - start_time

                    # Report health check metrics
                    self.exchange_metrics.record_health_check(
                        success=health_status, duration=check_duration, exchange=self.exchange_name
                    )

                    if not health_status:
                        self.logger.warning(f"Health check failed for {self.exchange_name}")
                        await self._handle_health_check_failure()
                    else:
                        self.connection_retries = 0  # Reset on successful health check
                        # Update system health metrics
                        self.system_metrics.set_gauge(
                            "exchange_health_status", 1.0, {"exchange": self.exchange_name}
                        )

                    self.last_health_check = datetime.now(timezone.utc)
                else:
                    # Report disconnected status
                    self.system_metrics.set_gauge(
                        "exchange_health_status", 0.0, {"exchange": self.exchange_name}
                    )

            except Exception as e:
                self.logger.error(f"Error in health monitor loop: {e!s}")
                # Report health check error
                self.exchange_metrics.record_health_check(
                    success=False, duration=0, exchange=self.exchange_name
                )
                await asyncio.sleep(5)  # Brief pause before retry

    async def _monitor_connection_pool(self, connector) -> None:
        """
        Monitor connection pool metrics.
        """
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                if self._connector_lock:
                    async with self._connector_lock:
                        if hasattr(connector, "_acquired"):
                            # Report connection pool metrics
                            self.system_metrics.set_gauge(
                                "connection_pool_size",
                                len(connector._acquired),
                                {"exchange": self.exchange_name, "pool_type": "http"},
                            )

                        if hasattr(connector, "_limit"):
                            self.system_metrics.set_gauge(
                                "connection_pool_limit",
                                connector._limit,
                                {"exchange": self.exchange_name, "pool_type": "http"},
                            )

                        if hasattr(connector, "_limit_per_host"):
                            self.system_metrics.set_gauge(
                                "connection_pool_limit_per_host",
                                connector._limit_per_host,
                                {"exchange": self.exchange_name, "pool_type": "http"},
                            )

            except Exception as e:
                self.logger.error(f"Error monitoring connection pool: {e!s}")
                await asyncio.sleep(30)  # Longer pause on error

    async def _handle_health_check_failure(self) -> None:
        """
        Handle health check failure with automatic recovery.
        """
        self.connection_retries += 1

        if self.connection_retries >= self.max_connection_retries:
            self.logger.error(f"Max connection retries reached for {self.exchange_name}")
            self.status = "connection_failed"
            self.connected = False
            return

        # Attempt reconnection
        self.logger.info(
            f"Attempting reconnection for {self.exchange_name} (attempt {self.connection_retries})"
        )

        try:
            await self.disconnect()
            await asyncio.sleep(2**self.connection_retries)  # Exponential backoff
            await self.connect()
        except Exception as e:
            self.logger.error(f"Reconnection failed: {e!s}")

    async def _cleanup_connection_infrastructure(self) -> None:
        """
        Clean up connection pool and session resources.
        """
        try:
            # Cancel all monitoring tasks
            for task in self._monitoring_tasks:
                if not task.done():
                    task.cancel()

            # Wait for all tasks to complete with timeout
            if self._monitoring_tasks:
                await asyncio.wait(self._monitoring_tasks, timeout=5.0)

            # Clear task list
            self._monitoring_tasks.clear()

            # Close request session
            if self.request_session:
                await self.request_session.close()
                self.request_session = None

            self.connection_pool = None
            self.session_manager = None
            self.unified_rate_limiter = None
            self.unified_ws_manager = None

        except Exception as e:
            self.logger.error(f"Error cleaning up connection infrastructure: {e!s}")

    # === CONNECTION MANAGEMENT ===

    async def connect(self) -> bool:
        """
        Establish connection to the exchange with unified infrastructure.

        Returns:
            bool: True if connection successful, False otherwise
        """
        with self.tracer.start_as_current_span(
            "exchange.connect", attributes={"exchange": self.exchange_name}
        ) as span:
            try:
                self.logger.info(f"Connecting to {self.exchange_name} exchange...")

                # Initialize locks if not already done
                if self._connector_lock is None:
                    self._connector_lock = asyncio.Lock()
                if self._rate_limit_lock is None:
                    self._rate_limit_lock = asyncio.Lock()

                # Ensure infrastructure is ready
                if not self.connection_pool:
                    await self._initialize_connection_infrastructure()

                # Call exchange-specific connection logic
                connection_result = await self._connect_to_exchange()

                if connection_result:
                    self.connected = True
                    self.status = "connected"
                    self.last_heartbeat = datetime.now(timezone.utc)
                    self.connection_retries = 0

                    # Initialize database and Redis connections
                    await self._initialize_database()
                    await self._initialize_redis()

                    self.logger.info(f"Successfully connected to {self.exchange_name}")

                    # Record connection success metric
                    self.exchange_metrics.record_connection(
                        success=True, exchange=self.exchange_name
                    )
                    span.set_attribute("connection.success", True)

                    return True
                else:
                    self.connected = False
                    self.status = "connection_failed"

                    # Record connection failure metric
                    self.exchange_metrics.record_connection(
                        success=False, exchange=self.exchange_name
                    )
                    span.set_attribute("connection.success", False)

                    return False

            except Exception as e:
                self.logger.error(f"Failed to connect to {self.exchange_name}: {e!s}")
                self.connected = False
                self.status = "connection_error"

                # Record connection error metric
                self.exchange_metrics.record_connection(success=False, exchange=self.exchange_name)
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                await self._handle_exchange_error(e, "connect", {"exchange": self.exchange_name})
                return False

    @abstractmethod
    async def _connect_to_exchange(self) -> bool:
        """
        Exchange-specific connection logic (to be implemented by subclasses).

        Returns:
            bool: True if connection successful, False otherwise
        """
        pass

    async def disconnect(self) -> None:
        """Disconnect from the exchange and cleanup resources."""
        try:
            self.logger.info(f"Disconnecting from {self.exchange_name}...")

            # Close all WebSocket streams
            await self._close_all_streams()

            # Call exchange-specific disconnect
            await self._disconnect_from_exchange()

            # Clean up connection infrastructure
            await self._cleanup_connection_infrastructure()

            self.connected = False
            self.status = "disconnected"

            self.logger.info(f"Successfully disconnected from {self.exchange_name}")

        except Exception as e:
            self.logger.error(f"Error during disconnect: {e!s}")

    @abstractmethod
    async def _disconnect_from_exchange(self) -> None:
        """Disconnect from the exchange (to be implemented by subclasses)."""
        pass

    # === RATE LIMITING ===

    async def _check_unified_rate_limit(self, endpoint: str, weight: int = 1) -> bool:
        """
        Unified rate limiting check across all exchanges.

        Args:
            endpoint: API endpoint
            weight: Request weight

        Returns:
            bool: True if request is allowed

        Raises:
            ExchangeRateLimitError: If rate limit is exceeded
            ValidationError: If parameters are invalid
        """
        try:
            # Validate parameters
            if not endpoint:
                raise ValidationError("Endpoint is required")

            if weight <= 0:
                raise ValidationError("Weight must be positive")

            # Check unified rate limiter first
            if not await self._check_local_rate_limit(endpoint, weight):
                # Record rate limit violation metric
                self.exchange_metrics.record_rate_limit_violation(
                    endpoint=endpoint, exchange=self.exchange_name
                )
                raise ExchangeRateLimitError(f"Local rate limit exceeded for {endpoint}")

            # Check advanced rate limiter
            if not await self.advanced_rate_limiter.check_rate_limit(
                self.exchange_name, endpoint, weight
            ):
                # Record rate limit violation metric
                self.exchange_metrics.record_rate_limit_violation(
                    endpoint=endpoint, exchange=self.exchange_name
                )
                raise ExchangeRateLimitError(f"Advanced rate limit exceeded for {endpoint}")

            # Update local tracking
            await self._update_rate_limit_tracking(endpoint, weight)

            # Record successful rate limit check
            self.exchange_metrics.record_rate_limit_check(
                endpoint=endpoint, weight=weight, exchange=self.exchange_name
            )

            return True

        except (ValidationError, ExchangeRateLimitError):
            raise
        except Exception as e:
            self.logger.error(
                "Rate limit check failed",
                exchange=self.exchange_name,
                endpoint=endpoint,
                error=str(e),
            )
            raise ExchangeRateLimitError(f"Rate limit check failed: {e!s}") from e

    async def _check_local_rate_limit(self, endpoint: str, weight: int) -> bool:
        """
        Check local rate limiting windows.
        """
        if not self.unified_rate_limiter:
            return True

        current_time = time.time()

        if self._rate_limit_lock:
            async with self._rate_limit_lock:
                # Clean old requests from window
                if endpoint not in self.rate_limit_windows:
                    self.rate_limit_windows[endpoint] = []

                window = self.rate_limit_windows[endpoint]

                # Remove requests older than 60 seconds
                self.rate_limit_windows[endpoint] = [
                    req_time for req_time in window if current_time - req_time < 60
                ]

                # Check rate limit
                endpoint_config = self.rate_limit_config.get(endpoint, {})
                max_requests = endpoint_config.get("max_requests", 100)

                if len(self.rate_limit_windows[endpoint]) + weight <= max_requests:
                    return True

                return False

        # If no lock available, just return True (don't block)
        return True

    async def _update_rate_limit_tracking(self, endpoint: str, weight: int) -> None:
        """
        Update rate limit tracking after successful request.
        """
        current_time = time.time()

        if self._rate_limit_lock:
            async with self._rate_limit_lock:
                # Add weight number of timestamps
                for _ in range(weight):
                    self.rate_limit_windows[endpoint].append(current_time)

                # Update session manager
                if self.session_manager:
                    self.session_manager["request_count"] += weight
                    self.session_manager["last_request"] = datetime.now(timezone.utc)

    # === ORDER MANAGEMENT ===

    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """
        Execute a trade order on the exchange with unified order management.

        Args:
            order: Order request with all necessary details

        Returns:
            OrderResponse: Order response with execution details

        Raises:
            ExchangeError: If order placement fails
            ValidationError: If order request is invalid
        """
        with self.tracer.start_as_current_span(
            "exchange.place_order",
            attributes={
                "exchange": self.exchange_name,
                "symbol": order.symbol,
                "side": order.side,
                "amount": str(order.quantity),
                "order_type": order.order_type,
            },
        ) as span:
            try:
                # Pre-order validation
                if not await self.pre_trade_validation(order):
                    raise ValidationError("Order validation failed")

                # Check rate limits
                await self._check_unified_rate_limit("place_order", 1)

                # Generate order ID for tracking
                client_order_id = (
                    order.client_order_id or f"{self.exchange_name}_{int(time.time() * 1000)}"
                )

                # Call exchange-specific order placement first
                try:
                    order_response = await self._place_order_on_exchange(order)
                except Exception:
                    # If order placement fails, no state is saved
                    raise

                # Only save to state after successful exchange placement
                order_data = {
                    "order_id": order_response.id,
                    "client_order_id": client_order_id,
                    "exchange": self.exchange_name,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "order_type": order.order_type.value,
                    "quantity": str(order.quantity),
                    "price": str(order.price) if order.price else None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": order_response.status.value,
                    "exchange_order_id": order_response.id,
                    "filled_quantity": (
                        str(order_response.filled_quantity)
                        if order_response.filled_quantity
                        else "0"
                    ),
                    "average_price": (
                        str(order_response.average_price) if order_response.average_price else None
                    ),
                }

                # Save to state service
                await self._save_order_to_state(order_response.id, order_data)

                # Notify trade lifecycle manager
                await self._notify_trade_lifecycle(order_response.id, "order_placed", order_data)

                # Notify trade lifecycle based on status
                if order_response.status == OrderStatus.FILLED:
                    await self._notify_trade_lifecycle(
                        order_response.id, "order_filled", order_data
                    )
                elif order_response.status == OrderStatus.PARTIALLY_FILLED:
                    await self._notify_trade_lifecycle(
                        order_response.id, "order_partially_filled", order_data
                    )

                # Post-trade processing
                await self.post_trade_processing(order_response)

                # Record metrics
                self.exchange_metrics.record_order(
                    order_type=order.order_type,
                    side=order.side,
                    success=True,
                    exchange=self.exchange_name,
                    symbol=order.symbol,
                )

                span.set_attribute("order.id", order_response.id)
                span.set_attribute("order.status", order_response.status)

                return order_response

            except Exception as e:
                # Record failed order metric
                self.exchange_metrics.record_order(
                    order_type=order.order_type,
                    side=order.side,
                    success=False,
                    exchange=self.exchange_name,
                    symbol=order.symbol,
                )
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                await self._handle_exchange_error(
                    e,
                    "place_order",
                    {
                        "symbol": order.symbol,
                        "client_order_id": (
                            str(order.client_order_id) if order.client_order_id else None
                        ),
                    },
                )
                raise

    @abstractmethod
    async def _place_order_on_exchange(self, order: OrderRequest) -> OrderResponse:
        """
        Exchange-specific order placement logic (to be implemented by subclasses).

        Args:
            order: Order request

        Returns:
            OrderResponse: Order response
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: ID of the order to cancel

        Returns:
            bool: True if cancellation successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Check the status of an order.

        Args:
            order_id: ID of the order to check

        Returns:
            OrderStatus: Current status of the order
        """
        pass

    async def get_unified_order_status(self, order_id: str) -> OrderStatus:
        """
        Get order status with local tracking fallback.
        """
        try:
            # Check local tracking first
            if order_id in self.pending_orders:
                local_status = self.pending_orders[order_id].get("status")
                if local_status:
                    return OrderStatus(local_status)

            # Fallback to exchange API
            return await self.get_order_status(order_id)

        except Exception as e:
            self.logger.error(f"Failed to get unified order status for {order_id}: {e!s}")
            return OrderStatus.UNKNOWN

    # === STATE MANAGEMENT HELPERS ===

    async def _save_order_to_state(self, order_id: str, order_data: dict[str, Any]) -> bool:
        """
        Save order data to state service.

        Args:
            order_id: Order ID
            order_data: Order data to save

        Returns:
            True if saved successfully

        Raises:
            ExchangeConnectionError: If state service operation fails
        """
        if not self.state_service:
            # Fallback to in-memory storage if no state service
            self.pending_orders[order_id] = order_data
            return True

        try:
            # Save to state service
            success = await self.state_service.set_state(
                state_type=StateType.ORDER_STATE,
                state_id=order_id,
                state_data=order_data,
                source_component=f"exchange.{self.exchange_name}",
                priority=StatePriority.HIGH,
                reason="Order placed",
            )

            if not success:
                raise ExchangeConnectionError(f"Failed to save order {order_id} to state service")

            # Also update in-memory cache for quick access
            self.pending_orders[order_id] = order_data
            return success

        except ExchangeConnectionError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to save order to state: {e}")
            raise ExchangeConnectionError(
                f"State service error saving order {order_id}: {e!s}"
            ) from e

    async def _update_order_state(
        self, order_id: str, updates: dict[str, Any], reason: str = "Order updated"
    ) -> bool:
        """
        Update order state in state service.

        Args:
            order_id: Order ID
            updates: Fields to update
            reason: Reason for update

        Returns:
            True if updated successfully

        Raises:
            ExchangeConnectionError: If state service operation fails
        """
        if not self.state_service:
            # Fallback to in-memory update
            if order_id in self.pending_orders:
                self.pending_orders[order_id].update(updates)
            return True

        try:
            # Get current order state
            current_state = await self._get_order_from_state(order_id)
            if not current_state:
                raise ExchangeConnectionError(f"Order {order_id} not found in state service")

            # Merge updates
            current_state.update(updates)

            # Save updated state
            success = await self.state_service.set_state(
                state_type=StateType.ORDER_STATE,
                state_id=order_id,
                state_data=current_state,
                source_component=f"exchange.{self.exchange_name}",
                priority=StatePriority.HIGH,
                reason=reason,
            )

            if not success:
                raise ExchangeConnectionError(f"Failed to update order {order_id} in state service")

            # Update in-memory cache
            self.pending_orders[order_id] = current_state
            return success

        except ExchangeConnectionError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to update order state: {e}")
            raise ExchangeConnectionError(
                f"State service error updating order {order_id}: {e!s}"
            ) from e

    async def _get_order_from_state(self, order_id: str) -> dict[str, Any] | None:
        """
        Get order data from state service.

        Args:
            order_id: Order ID

        Returns:
            Order data or None if not found
        """
        # Check in-memory cache first
        if order_id in self.pending_orders:
            return self.pending_orders[order_id]

        if not self.state_service:
            return None

        try:
            # Get from state service
            order_data = await self.state_service.get_state(
                state_type=StateType.ORDER_STATE, state_id=order_id
            )

            if order_data:
                # Update in-memory cache
                self.pending_orders[order_id] = order_data

            return order_data

        except Exception as e:
            self.logger.error(f"Failed to get order from state: {e}")
            return None

    async def _notify_trade_lifecycle(
        self, order_id: str, event: str, order_data: dict[str, Any]
    ) -> None:
        """
        Notify trade lifecycle manager of order events.

        Args:
            order_id: Order ID
            event: Event type (e.g., 'order_placed', 'order_filled')
            order_data: Order data

        Raises:
            ExchangeConnectionError: If trade lifecycle notification fails
        """
        if not self.trade_lifecycle_manager:
            return

        try:
            # Map events to TradeEvent enum
            event_map = {
                "order_placed": TradeEvent.ORDER_SUBMITTED,
                "order_filled": TradeEvent.COMPLETE_FILL,
                "order_partially_filled": TradeEvent.PARTIAL_FILL,
                "order_cancelled": TradeEvent.ORDER_CANCELLED,
                "order_rejected": TradeEvent.ORDER_REJECTED,
            }

            trade_event = event_map.get(event)
            if not trade_event:
                self.logger.warning(f"Unknown trade event: {event}")
                return

            # Update trade lifecycle
            await self.trade_lifecycle_manager.update_trade_event(
                trade_id=order_id,
                event=trade_event,
                event_data={
                    "order_id": order_id,
                    "exchange": self.exchange_name,
                    **order_data,
                },
            )

        except Exception as e:
            self.logger.error(f"Failed to notify trade lifecycle: {e}")
            # Don't fail the order if lifecycle notification fails
            # This is a secondary operation that shouldn't block trading
            # But we should monitor these failures
            if hasattr(self, "metrics_collector") and self.metrics_collector:
                self.metrics_collector.increment_counter(
                    "exchange_errors_total",
                    {"exchange": self.exchange_name, "error_type": "trade_lifecycle_notification"},
                )

    # === MARKET DATA MANAGEMENT ===

    async def get_market_data(self, symbol: str, timeframe: str = "1m") -> MarketData:
        """
        Get OHLCV market data for a symbol with unified caching and rate limiting.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe for the data (e.g., '1m', '1h', '1d')

        Returns:
            MarketData: Market data with price and volume information
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}"
            cached_data = await self._get_cached_market_data(cache_key)

            if cached_data:
                return cached_data

            # Apply rate limiting for market data requests
            await self._check_unified_rate_limit("market_data", 1)

            # Use centralized market data source if available
            if self.market_data_source:
                market_data = await self.market_data_source.get_historical_data(
                    self.exchange_name, symbol, start_time=None, end_time=None, interval=timeframe
                )

                if market_data:
                    result = market_data[-1] if isinstance(market_data, list) else market_data
                    await self._cache_market_data_unified(cache_key, result)
                    return result

            # Fallback to exchange-specific implementation
            market_data = await self._get_market_data_from_exchange(symbol, timeframe)

            # Cache the result
            await self._cache_market_data_unified(cache_key, market_data)

            return market_data

        except Exception as e:
            self.logger.error(f"Failed to get market data: {e!s}")
            await self._handle_exchange_error(
                e, "get_market_data", {"symbol": symbol, "timeframe": timeframe}
            )
            raise

    async def _get_cached_market_data(self, cache_key: str) -> MarketData | None:
        """
        Get market data from cache if still valid.
        """
        try:
            if cache_key not in self.market_data_cache:
                return None

            cached_entry = self.market_data_cache[cache_key]
            cache_time = cached_entry.get("timestamp", datetime.min)

            # Check if cache is still valid
            if (datetime.now(timezone.utc) - cache_time).total_seconds() < self.cache_ttl:
                return cached_entry.get("data")
            else:
                # Remove expired cache entry
                del self.market_data_cache[cache_key]
                return None

        except Exception as e:
            self.logger.error(f"Error reading market data cache: {e!s}")
            return None

    async def _cache_market_data_unified(self, cache_key: str, data: MarketData) -> None:
        """
        Cache market data with timestamp.
        """
        try:
            self.market_data_cache[cache_key] = {
                "data": data,
                "timestamp": datetime.now(timezone.utc),
            }

            # Only cache in Redis if available and connected with proper validation
            if self.redis_client:
                try:
                    # Validate Redis client is connected before attempting cache operation
                    if hasattr(self.redis_client, "ping") and self.redis_client:
                        await self._cache_market_data(
                            cache_key.split("_")[0],  # symbol
                            {
                                "price": str(data.price),
                                "volume": str(data.volume),
                                "timestamp": data.timestamp.isoformat(),
                                "open_price": str(data.open_price) if data.open_price else None,
                                "high_price": str(data.high_price) if data.high_price else None,
                                "low_price": str(data.low_price) if data.low_price else None,
                            },
                            ttl=self.cache_ttl,
                        )
                    else:
                        self.logger.debug("Redis client not connected, skipping cache operation")
                except Exception as redis_error:
                    # Log Redis error but don't fail the main operation
                    self.logger.warning(f"Failed to cache in Redis: {redis_error!s}")

        except Exception as e:
            self.logger.error(f"Error caching market data: {e!s}")

    @abstractmethod
    async def _get_market_data_from_exchange(
        self, symbol: str, timeframe: str = "1m"
    ) -> MarketData:
        """
        Get market data from exchange API (to be implemented by subclasses).

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data

        Returns:
            MarketData: Market data from exchange
        """
        pass

    # === WEBSOCKET MANAGEMENT ===

    async def subscribe_to_stream(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to real-time data stream for a symbol with unified WebSocket management.

        Args:
            symbol: Trading symbol to subscribe to
            callback: Callback function to handle incoming data
        """
        try:
            stream_name = f"{symbol}_stream"

            # Register callback
            if stream_name not in self.stream_callbacks:
                self.stream_callbacks[stream_name] = []
            self.stream_callbacks[stream_name].append(callback)

            # Check if stream already exists
            if stream_name in self.active_streams:
                self.logger.debug(f"Stream {stream_name} already active")
                return

            # Create new stream
            stream_connection = await self._create_websocket_stream(symbol, stream_name)

            if stream_connection:
                self.active_streams[stream_name] = stream_connection
                # Start stream handler task
                stream_task = asyncio.create_task(self._handle_unified_stream(stream_name))
                # Store reference to avoid garbage collection
                self._monitoring_tasks.append(stream_task)

                self.logger.info(f"Subscribed to {stream_name} for {self.exchange_name}")
            else:
                raise ExchangeConnectionError(f"Failed to create stream for {symbol}")

        except Exception as e:
            self.logger.error(f"Failed to subscribe to stream {symbol}: {e!s}")
            raise

    @abstractmethod
    async def _create_websocket_stream(self, symbol: str, stream_name: str) -> Any:
        """
        Create exchange-specific WebSocket stream (to be implemented by subclasses).

        Args:
            symbol: Trading symbol
            stream_name: Name for the stream

        Returns:
            WebSocket connection object
        """
        pass

    async def _handle_unified_stream(self, stream_name: str) -> None:
        """
        Unified stream handler with automatic reconnection.
        """
        reconnect_attempts = 0
        max_attempts = 5

        while stream_name in self.active_streams and reconnect_attempts < max_attempts:
            try:
                stream = self.active_streams[stream_name]

                # Call exchange-specific stream handler
                await self._handle_exchange_stream(stream_name, stream)

            except Exception as e:
                self.logger.error(f"Stream {stream_name} error: {e!s}")
                reconnect_attempts += 1

                if reconnect_attempts < max_attempts:
                    delay = min(2**reconnect_attempts, 30)  # Exponential backoff, max 30s
                    self.logger.info(f"Reconnecting stream {stream_name} in {delay}s...")
                    await asyncio.sleep(delay)

                    # Attempt to recreate stream
                    try:
                        symbol = stream_name.replace("_stream", "")
                        new_stream = await self._create_websocket_stream(symbol, stream_name)
                        if new_stream:
                            self.active_streams[stream_name] = new_stream
                            reconnect_attempts = 0  # Reset on successful reconnection
                    except Exception as reconnect_error:
                        self.logger.error(f"Stream reconnection failed: {reconnect_error!s}")
                else:
                    self.logger.error(f"Max reconnection attempts reached for {stream_name}")
                    break

        # Clean up failed stream
        if stream_name in self.active_streams:
            del self.active_streams[stream_name]
        if stream_name in self.stream_callbacks:
            del self.stream_callbacks[stream_name]

    @abstractmethod
    async def _handle_exchange_stream(self, stream_name: str, stream: Any) -> None:
        """
        Exchange-specific stream handling logic (to be implemented by subclasses).

        Args:
            stream_name: Name of the stream
            stream: Stream connection object
        """
        pass

    async def _close_stream(self, stream_name: str) -> None:
        """
        Close a specific WebSocket stream.
        """
        try:
            if stream_name in self.active_streams:
                stream = self.active_streams[stream_name]

                # Call exchange-specific stream closure
                await self._close_exchange_stream(stream_name, stream)

                # Clean up tracking
                del self.active_streams[stream_name]

                if stream_name in self.stream_callbacks:
                    del self.stream_callbacks[stream_name]

                self.logger.info(f"Closed stream {stream_name}")

        except Exception as e:
            self.logger.error(f"Error closing stream {stream_name}: {e!s}")

    @abstractmethod
    async def _close_exchange_stream(self, stream_name: str, stream: Any) -> None:
        """
        Exchange-specific stream closure logic (to be implemented by subclasses).

        Args:
            stream_name: Name of the stream
            stream: Stream connection object
        """
        pass

    async def _close_all_streams(self) -> None:
        """
        Close all active WebSocket streams.
        """
        try:
            for stream_name in list(self.active_streams.keys()):
                await self._close_stream(stream_name)

            self.active_streams.clear()
            self.stream_callbacks.clear()

        except Exception as e:
            self.logger.error(f"Error closing streams: {e!s}")

    # === DATABASE AND REDIS ===

    async def _initialize_database(self) -> None:
        """
        Initialize database connection and queries.
        """
        try:
            # Database queries should be initialized with proper session management
            # For now, we'll skip direct database initialization to avoid session issues
            # Higher layers (services) should handle database operations
            self.db_queries = None
            self.logger.debug(
                f"Database initialization delegated to service layer for {self.exchange_name}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize database for {self.exchange_name}: {e!s}")
            self.db_queries = None

    async def _initialize_redis(self) -> None:
        """
        Initialize Redis client for real-time data caching.
        """
        try:
            # Initialize RedisClient with config object - it will extract the URL properly
            self.redis_client = RedisClient(self.config, auto_close=True)
            await self.redis_client.connect()

            # Test Redis connection with proper error handling
            try:
                if self.redis_client:
                    await self.redis_client.ping()
                    self.logger.debug(f"Redis initialized for {self.exchange_name}")
                else:
                    raise Exception("Redis client not connected")
            except Exception as ping_error:
                self.logger.warning(f"Redis ping failed: {ping_error}")
                raise Exception("Redis health check failed") from ping_error

        except Exception as e:
            self.logger.warning(f"Failed to initialize Redis for {self.exchange_name}: {e!s}")
            self.redis_client = None

    async def _cache_market_data(self, symbol: str, data: dict, ttl: int = 300) -> None:
        """
        Cache market data in Redis for fast access.

        Args:
            symbol: Trading symbol
            data: Market data to cache
            ttl: Time to live in seconds
        """
        try:
            if not self.redis_client:
                return

            key = f"market_data:{self.exchange_name}:{symbol}"
            await self.redis_client.set(key, data, ttl=ttl)
            self.logger.debug(f"Market data cached for {symbol}")

        except Exception as e:
            self.logger.error(f"Failed to cache market data: {e!s}")

    # === ERROR HANDLING ===

    async def _handle_exchange_error(
        self, error: Exception, operation: str, context: dict | None = None
    ) -> None:
        """
        Handle exchange errors using the error handler with proper context.

        Args:
            error: The exception that occurred
            operation: The operation being performed
            context: Additional context information
        """
        try:
            # Create error context with exchange-specific information
            error_context = self.error_handler.create_error_context(
                error=error,
                component="exchange",
                operation=operation,
                symbol=context.get("symbol") if context else None,
                order_id=context.get("order_id") if context else None,
                details={
                    "exchange_name": self.exchange_name,
                    "operation": operation,
                    **(context if context else {}),
                },
            )

            # Handle the error with appropriate recovery scenario
            if isinstance(error, ExchangeConnectionError):
                recovery_scenario = NetworkDisconnectionRecovery(self.config)
            elif isinstance(error, OrderRejectionError | ValidationError):
                recovery_scenario = OrderRejectionRecovery(self.config)
            else:
                recovery_scenario = None

            # Handle the error
            await self.error_handler.handle_error(error, error_context, recovery_scenario)

        except Exception as e:
            # Fallback to basic logging if error handling fails
            self.logger.error(f"Error handling failed for {operation}: {e!s}")

    # === ABSTRACT METHODS ===

    @abstractmethod
    @time_execution
    @memory_usage
    async def get_account_balance(self) -> dict[str, Decimal]:
        """
        Get all asset balances from the exchange.

        Returns:
            Dict[str, Decimal]: Dictionary mapping asset symbols to balances
        """
        pass

    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 10) -> OrderBook:
        """
        Get order book depth for a symbol.

        Args:
            symbol: Trading symbol
            depth: Number of levels to retrieve

        Returns:
            OrderBook: Order book with bid and ask levels
        """
        pass

    async def get_trade_history(self, symbol: str, limit: int = 100) -> list[Trade]:
        """
        Get historical trades for a symbol.

        Args:
            symbol: Trading symbol
            limit: Maximum number of trades to retrieve

        Returns:
            List[Trade]: List of historical trades
        """
        try:
            # Try to get trades from database first with proper error handling
            if self.db_queries:
                try:
                    trades = await self.db_queries.get_trades_by_symbol(symbol, limit=limit)
                    if trades:
                        self.logger.debug(
                            f"Retrieved {len(trades)} trades from database for {symbol}"
                        )
                        return trades
                except Exception as db_error:
                    self.logger.warning(f"Database query failed for trades: {db_error!s}")
                    # Continue to exchange API fallback

            # Fallback to exchange API if no database data
            return await self._get_trade_history_from_exchange(symbol, limit)

        except Exception as e:
            self.logger.error(f"Failed to get trade history: {e!s}")
            await self._handle_exchange_error(
                e, "get_trade_history", {"symbol": symbol, "limit": limit}
            )
            return []

    @abstractmethod
    async def _get_trade_history_from_exchange(self, symbol: str, limit: int = 100) -> list[Trade]:
        """
        Get trade history from exchange API (to be implemented by subclasses).

        Args:
            symbol: Trading symbol
            limit: Maximum number of trades to retrieve

        Returns:
            List[Trade]: List of trades from exchange
        """
        pass

    # Optional lifecycle/portfolio helpers for higher layers (risk/emergency)
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]:
        """
        Optional: Return open orders. Default implementation returns empty list.

        Args:
            symbol: Optional symbol filter

        Returns:
            List[OrderResponse]: Open orders
        """
        return []

    async def get_positions(self) -> list[Position]:
        """
        Optional: Return current open positions. Default implementation returns empty list.

        Returns:
            List[Position]: Open positions
        """
        return []

    @abstractmethod
    async def get_exchange_info(self) -> ExchangeInfo:
        """
        Get exchange information including supported symbols and features.

        Returns:
            ExchangeInfo: Exchange information
        """
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """
        Get real-time ticker information for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Ticker: Real-time ticker data
        """
        pass

    # === HEALTH AND MONITORING ===

    async def health_check(self) -> bool:
        """
        Perform a health check on the exchange connection.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Basic health check - try to get account balance
            await self.get_account_balance()
            self.last_heartbeat = datetime.now(timezone.utc)

            # Check database health if available with proper error handling
            if self.db_queries:
                try:
                    db_healthy = await self.db_queries.health_check()
                    if not db_healthy:
                        self.logger.warning(
                            "Database health check failed", exchange=self.exchange_name
                        )
                        # Don't fail the entire health check for database issues
                except Exception as db_error:
                    self.logger.warning(
                        f"Database health check error: {db_error}", exchange=self.exchange_name
                    )
                    # Continue with other health checks

            # Check Redis health if available with proper error handling
            if self.redis_client:
                try:
                    await self.redis_client.ping()
                except Exception as redis_error:
                    self.logger.warning(
                        f"Redis health check failed: {redis_error}", exchange=self.exchange_name
                    )
                    # Don't fail the entire health check for Redis issues

            return True
        except Exception as e:
            self.logger.warning("Health check failed", exchange=self.exchange_name, error=str(e))
            return False

    async def get_unified_connection_health(self) -> dict[str, Any]:
        """
        Get comprehensive connection health status.
        """
        try:
            health_data = {
                "exchange": self.exchange_name,
                "connected": self.connected,
                "status": self.status,
                "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
                "connection_retries": self.connection_retries,
                "active_streams": len(self.active_streams),
                "pending_orders": len(self.pending_orders),
                "cache_entries": len(self.market_data_cache),
                "session_info": self.session_manager,
                "rate_limit_violations": (
                    self.unified_rate_limiter.get("violations", 0)
                    if self.unified_rate_limiter
                    else 0
                ),
            }

            # Add exchange-specific health check
            exchange_health = await self.health_check()
            health_data["exchange_specific_health"] = exchange_health

            return health_data

        except Exception as e:
            self.logger.error(f"Failed to get connection health: {e!s}")
            return {
                "exchange": self.exchange_name,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def get_rate_limits(self) -> dict[str, int]:
        """
        Get the rate limits for this exchange.

        Returns:
            Dict[str, int]: Rate limits configuration
        """
        return getattr(self.config.exchange, "rate_limits", {}).get(self.exchange_name, {})

    # === UTILITY METHODS ===

    # Standard methods that can be overridden
    async def pre_trade_validation(self, order: OrderRequest) -> bool:
        """
        Pre-trade validation hook.

        Args:
            order: Order request to validate

        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            # Basic validation
            if not order.symbol or not order.quantity:
                self.logger.warning("Invalid order: missing symbol or quantity")
                return False

            if order.quantity <= 0:
                self.logger.warning("Invalid order: quantity must be positive")
                return False

            self.logger.debug(f"Order validation passed for {order.symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Order validation failed: {e!s}")
            return False

    async def post_trade_processing(self, order_response: OrderResponse) -> None:
        """
        Post-trade processing hook.

        Args:
            order_response: Response from the exchange after order execution
        """
        try:
            # Log the trade
            self.logger.info(
                "Order executed",
                order_id=order_response.id,
                symbol=order_response.symbol,
                side=order_response.side.value,
                filled_quantity=str(order_response.filled_quantity),
            )

            # Delegate trade storage to higher layers (bot management, strategy)
            # This avoids direct database model conflicts and session management issues
            if order_response.filled_quantity > 0:
                self.logger.debug(
                    f"Trade {order_response.id} executed, storage delegated to service layer"
                )

        except Exception as e:
            self.logger.error(f"Post-trade processing failed: {e!s}")

    async def _store_trade_in_database(self, order_response: OrderResponse) -> None:
        """
        Store executed trade in the database.

        Args:
            order_response: Order response with execution details
        """
        try:
            if not self.db_queries:
                return

            # Validate trade data before storage
            if not self._validate_trade_data(order_response):
                self.logger.warning(f"Invalid trade data for {order_response.id}, skipping storage")
                return

            # Trade data should be prepared and stored by higher layers
            # The exchange layer only provides raw execution data
            # Higher layers (bot management, strategy) have context for proper storage
            self.logger.debug(
                f"Trade execution completed for {order_response.id}, "
                f"storage delegated to service layer"
            )

        except Exception as e:
            self.logger.error(f"Failed to store trade in database: {e!s}")

    def _validate_trade_data(self, order_response: OrderResponse) -> bool:
        """
        Validate trade data before database storage.

        Args:
            order_response: Order response to validate

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Basic validation
            if not order_response.id or not order_response.symbol:
                return False

            if order_response.filled_quantity <= 0:
                return False

            if not order_response.timestamp:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Trade data validation failed: {e!s}")
            return False

    async def _update_performance_metrics(self, order_response: OrderResponse) -> None:
        """
        Update performance metrics after trade execution.

        Args:
            order_response: Order response with execution details
        """
        try:
            # Performance metrics should be updated by higher layers that have proper
            # context about the trade (bot, strategy, etc.)
            # Exchange layer only provides raw execution data
            self.logger.debug(f"Trade execution data available for metrics: {order_response.id}")

        except Exception as e:
            self.logger.error(f"Failed to process performance metrics: {e!s}")

    async def _store_performance_metrics(self, metrics: dict) -> None:
        """
        Store performance metrics in the database.

        Args:
            metrics: Dictionary containing performance metrics
        """
        try:
            if not self.db_queries:
                return

            # Skip direct database model creation - should be handled by service layer
            # Performance metrics should be created through a service abstraction
            self.logger.debug(
                f"Performance metrics data ready for {self.exchange_name}, "
                "delegating storage to service layer"
            )

        except Exception as e:
            self.logger.error(f"Failed to store performance metrics: {e!s}")

    async def _store_balance_snapshot(self, balances: dict[str, Decimal]) -> None:
        """
        Store balance snapshot in the database.

        Args:
            balances: Dictionary mapping asset symbols to balances
        """
        try:
            if not self.db_queries:
                return

            # Skip direct database model creation - should be handled by service layer
            # Balance snapshots should be created through a service abstraction
            self.logger.debug(
                f"Balance snapshot data ready for {self.exchange_name} "
                f"with {len(balances)} currencies, delegating storage to service layer"
            )

        except Exception as e:
            self.logger.error(f"Failed to store balance snapshot: {e!s}")

    # === CONTEXT MANAGERS ===

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    def cleanup(self) -> None:
        """Cleanup exchange resources and unified infrastructure."""
        try:
            # Disconnect if connected
            if self.connected:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Store task reference to avoid garbage collection
                    disconnect_task = loop.create_task(self.disconnect())
                    # Best effort - don't wait for completion in cleanup
                    _ = disconnect_task
                else:
                    loop.run_until_complete(self.disconnect())

            # Clear caches and tracking
            self.market_data_cache.clear()
            self.pending_orders.clear()
            self.order_callbacks.clear()
            self.rate_limit_windows.clear()

            self.logger.info(f"Enhanced exchange cleanup completed for {self.exchange_name}")
        except Exception as e:
            self.logger.error(f"Error during exchange cleanup: {e}")
        finally:
            super().cleanup()  # Call parent cleanup

    # === STATUS METHODS ===

    def is_connected(self) -> bool:
        """
        Check if the exchange is connected.

        Returns:
            bool: True if connected, False otherwise
        """
        return self.connected and self.status == "connected"

    def get_status(self) -> str:
        """
        Get the current status of the exchange.

        Returns:
            str: Current status
        """
        return self.status

    def get_exchange_name(self) -> str:
        """
        Get the name of the exchange.

        Returns:
            str: Exchange name
        """
        return self.exchange_name


# Compatibility alias for imports
BaseExchange = EnhancedBaseExchange
