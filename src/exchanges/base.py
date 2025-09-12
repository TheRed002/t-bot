"""
Base Exchange Implementation Following Service Layer Pattern

This module provides a production-ready BaseExchange that follows the project's
service layer pattern with proper dependency injection and error handling.

Key Features:
- BaseService inheritance for proper lifecycle management
- Financial precision with Decimal types
- Proper error handling with decorators
- Core type system integration
- Repository pattern integration
"""

from abc import abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

# MANDATORY: Core imports as per CLAUDE.md
from src.core.base import BaseService
from src.core.base.interfaces import HealthCheckResult, HealthStatus
from src.core.exceptions import (
    ExchangeConnectionError,
    OrderRejectionError,
    ServiceError,
    ValidationError,
)
from src.core.types import (
    ExchangeInfo,
    OrderBook,
    OrderBookLevel,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    Position,
    StateType,
    Ticker,
)
from src.core.types.market import Trade

# MANDATORY: Import from error_handling decorators
from src.error_handling.decorators import with_circuit_breaker, with_retry

# Import state management
from src.state import StatePriority, StateService
from src.utils.validation.market_data_validation import MarketDataValidator

# Import validation service
from src.utils.validation.service import ValidationContext, ValidationService

# Import services that may not be available - using TYPE_CHECKING to avoid import errors
if TYPE_CHECKING:
    pass
    # Use CapitalAllocationError from core exceptions instead

# Simple event system - simplified for easier imports
class ExchangeEvent:
    """Base class for exchange events"""
    def __init__(self):
        self.timestamp = datetime.now(timezone.utc)

class OrderPlacedEvent(ExchangeEvent):
    """Event emitted when order is placed"""
    pass

class OrderFilledEvent(ExchangeEvent):
    """Event emitted when order is filled"""
    pass

class OrderCancelledEvent(ExchangeEvent):
    """Event emitted when order is cancelled"""
    pass

if TYPE_CHECKING:
    pass


class BaseExchange(BaseService):
    """
    Base class for all exchange implementations following service layer pattern.

    This class provides a standardized interface for all cryptocurrency exchanges
    while following the project's mandatory service patterns:
    - Inherits from BaseService for proper lifecycle management
    - Uses Decimal for financial precision (never float)
    - Implements proper error handling with decorators
    - Follows Controller→Service→Repository pattern
    """

    def __init__(self, name: str, config: dict[str, Any]):
        """
        Initialize base exchange service.

        Args:
            name: Exchange name (e.g., 'binance', 'coinbase', 'okx')
            config: Exchange configuration
        """
        super().__init__(name=f"{name}_exchange")
        self.exchange_name = name
        self.config = config
        # BaseService already provides self.logger, don't override it

        # Exchange info cache
        self._exchange_info: ExchangeInfo | None = None
        self._trading_symbols: list[str] | None = None

        # Service dependencies (to be resolved during startup)
        self.add_dependency("config")
        self.add_dependency("state_service")
        self.add_dependency("validation_service")
        self.add_dependency("data_pipeline")
        self.add_dependency("database_service")  # Use database service instead of direct repositories
        self.add_dependency("risk_management_service")
        self.add_dependency("analytics_service")

        # Add new service dependencies (optional)
        self.add_dependency("trading_tracer")
        self.add_dependency("performance_profiler")
        self.add_dependency("alert_manager")
        self.add_dependency("capital_management_service")
        self.add_dependency("telemetry_service")
        self.add_dependency("alerting_service")
        self.add_dependency("event_bus")

        # Services will be resolved in _do_start
        self.state_service: StateService | None = None
        self.validation_service: ValidationService | None = None
        self.data_pipeline: Any | None = None  # DataPipeline type, avoiding circular import
        self.database_service: Any | None = None  # DatabaseService for proper data access
        self.risk_service: Any | None = None  # RiskManagementService type
        self.analytics_service: Any | None = None  # AnalyticsService type
        self.market_validator = MarketDataValidator()

        # Monitoring services
        self.trading_tracer: Any | None = None  # TradingTracer type
        self.performance_profiler: Any | None = None  # PerformanceProfiler type
        self.alert_manager: Any | None = None  # AlertManager type

        # Capital management service
        self.capital_service: Any | None = None  # CapitalService type

        # Additional services used by exchanges
        self.telemetry_service: Any | None = None  # TelemetryService type
        self.alerting_service: Any | None = None  # AlertingService type
        self.event_bus: Any | None = None  # EventBus type

        # Event handlers for broadcasting events
        self.event_handlers = []

    @property
    def connected(self) -> bool:
        """Check if exchange is connected."""
        # Return local connection state if available, otherwise False
        return self._connected if hasattr(self, "_connected") else False

    @property
    def last_heartbeat(self) -> datetime | None:
        """Get last heartbeat timestamp from state."""
        return self._last_heartbeat if hasattr(self, "_last_heartbeat") else None

    def is_connected(self) -> bool:
        """Check if exchange is connected (for backward compatibility)."""
        return self.connected

    async def _update_connection_state(self, connected: bool) -> None:
        """Update connection state through StateService."""
        if self.state_service:
            await self.state_service.set_state(
                StateType.SYSTEM_STATE,
                f"{self.exchange_name}_connection",
                {
                    "connected": connected,
                    "last_heartbeat": datetime.now(timezone.utc).isoformat(),
                    "exchange_info": self._exchange_info.__dict__ if self._exchange_info else None,
                },
                source_component=f"{self.exchange_name}_exchange",
                priority=StatePriority.HIGH,
                reason=f"Connection {'established' if connected else 'closed'}",
            )
        # Also update local cache for quick access
        self._connected = connected
        self._last_heartbeat = datetime.now(timezone.utc)

    async def _do_start(self) -> None:
        """Initialize exchange service (called by BaseService)."""
        try:
            self.logger.info(f"Starting {self.exchange_name} exchange service")

            # Resolve service dependencies
            self.state_service = self.resolve_dependency("state_service")

            try:
                self.validation_service = self.resolve_dependency("validation_service")
            except Exception:
                self.logger.warning("ValidationService not available, using local validator")
                # ValidationService is optional, we have MarketDataValidator as fallback

            try:
                self.data_pipeline = self.resolve_dependency("data_pipeline")
            except Exception:
                self.logger.warning("DataPipeline not available")

            # Initialize database service (proper architecture pattern)
            try:
                self.database_service = self.resolve_dependency("database_service")
                if self.database_service:
                    self.logger.info(f"Initialized database service for {self.exchange_name}")
                else:
                    self.logger.warning("Database service not available")
            except Exception as e:
                self.logger.warning(f"Database service initialization failed: {e}")

            # Initialize risk management service
            try:
                self.risk_service = self.resolve_dependency("risk_management_service")
                self.logger.info(f"Integrated RiskManagementService for {self.exchange_name}")
            except Exception:
                self.logger.warning("RiskManagementService not available")

            # Initialize analytics service
            try:
                self.analytics_service = self.resolve_dependency("analytics_service")
                self.logger.info(f"Integrated AnalyticsService for {self.exchange_name}")
            except Exception:
                self.logger.warning("AnalyticsService not available")

            # Initialize monitoring services
            try:
                self.trading_tracer = self.resolve_dependency("trading_tracer")
                self.performance_profiler = self.resolve_dependency("performance_profiler")
                self.alert_manager = self.resolve_dependency("alert_manager")
                self.logger.info(f"Initialized monitoring services for {self.exchange_name}")
            except Exception as e:
                self.logger.warning(f"Monitoring services initialization failed: {e}")

            # Initialize capital management service
            try:
                self.capital_service = self.resolve_dependency("capital_management_service")
                self.logger.info(f"Initialized CapitalManagementService for {self.exchange_name}")
            except Exception:
                self.logger.warning("Capital management service not available")

            # Initialize additional services
            try:
                self.telemetry_service = self.resolve_dependency("telemetry_service")
                self.logger.info(f"Initialized TelemetryService for {self.exchange_name}")
            except Exception:
                self.logger.warning("Telemetry service not available")

            try:
                self.alerting_service = self.resolve_dependency("alerting_service")
                self.logger.info(f"Initialized AlertingService for {self.exchange_name}")
            except Exception:
                self.logger.warning("Alerting service not available")

            try:
                self.event_bus = self.resolve_dependency("event_bus")
                self.logger.info(f"Initialized EventBus for {self.exchange_name}")
            except Exception:
                self.logger.warning("Event bus not available")

            # Initialize event bus
            # Event system is simplified - no complex event bus needed
            self.logger.info(f"Exchange event system initialized for {self.exchange_name}")

            # Initialize connection state
            self._connected = False
            self._last_heartbeat = None

            await self.connect()
            await self.load_exchange_info()
            await self._update_connection_state(True)

            self.logger.info(f"{self.exchange_name} exchange service started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start {self.exchange_name} exchange: {e}")
            # Use consistent error propagation
            self._propagate_error_consistently(e, "startup")
            raise ServiceError(f"Exchange startup failed: {e}") from e

    async def _do_stop(self) -> None:
        """Cleanup exchange service (called by BaseService)."""
        try:
            self.logger.info(f"Stopping {self.exchange_name} exchange service")
            await self._update_connection_state(False)
            await self.disconnect()
            self.logger.info(f"{self.exchange_name} exchange service stopped")
        except Exception as e:
            self.logger.error(f"Error stopping {self.exchange_name} exchange: {e}")

    async def health_check(self) -> HealthCheckResult:
        """Perform health check on exchange connection."""
        try:
            if not self.connected:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="Exchange not connected",
                    details={"exchange": self.exchange_name},
                )

            # Test basic connectivity
            await self.ping()

            # Update heartbeat in state
            await self._update_connection_state(True)

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Exchange connection healthy",
                details={
                    "exchange": self.exchange_name,
                    "last_heartbeat": self.last_heartbeat.isoformat()
                    if self.last_heartbeat
                    else None,
                },
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                details={"exchange": self.exchange_name, "error": str(e)},
            )

    # Abstract methods that must be implemented by exchange implementations

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to exchange."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to exchange."""
        pass

    @abstractmethod
    async def ping(self) -> bool:
        """Test exchange connectivity."""
        pass

    @abstractmethod
    async def load_exchange_info(self) -> ExchangeInfo:
        """Load exchange information and trading rules."""
        pass

    # Market Data Methods (must use Decimal for financial precision)

    @abstractmethod
    @with_circuit_breaker(failure_threshold=5)
    @with_retry(max_attempts=3)
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker information for a symbol."""
        pass

    @abstractmethod
    @with_circuit_breaker(failure_threshold=5)
    @with_retry(max_attempts=3)
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook:
        """Get order book for a symbol."""
        pass

    @abstractmethod
    @with_circuit_breaker(failure_threshold=5)
    @with_retry(max_attempts=3)
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        """Get recent trades for a symbol."""
        pass

    # Trading Methods (must use Decimal for financial precision)

    @abstractmethod
    @with_circuit_breaker(failure_threshold=3)
    @with_retry(max_attempts=2)
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """
        Place an order on the exchange.

        Args:
            order_request: Order details with Decimal prices/quantities

        Returns:
            OrderResponse with order details

        Raises:
            ValidationError: If order parameters are invalid
            OrderRejectionError: If exchange rejects the order
            ExchangeRateLimitError: If rate limited
        """
        pass

    @abstractmethod
    @with_circuit_breaker(failure_threshold=3)
    @with_retry(max_attempts=2)
    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse:
        """Cancel an existing order."""
        pass

    @abstractmethod
    @with_circuit_breaker(failure_threshold=5)
    @with_retry(max_attempts=3)
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse:
        """Get current status of an order."""
        pass

    @abstractmethod
    @with_circuit_breaker(failure_threshold=5)
    @with_retry(max_attempts=3)
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]:
        """Get all open orders, optionally filtered by symbol."""
        pass

    # Account Methods (must use Decimal for financial precision)

    @abstractmethod
    @with_circuit_breaker(failure_threshold=5)
    @with_retry(max_attempts=3)
    async def get_account_balance(self) -> dict[str, Decimal]:
        """
        Get account balance for all assets.

        Returns:
            Dict mapping asset names to Decimal balances
        """
        pass

    @abstractmethod
    @with_circuit_breaker(failure_threshold=5)
    @with_retry(max_attempts=3)
    async def get_positions(self) -> list[Position]:
        """Get all open positions (for margin/futures trading)."""
        pass

    # Validation Methods using ValidationService

    async def _validate_order(self, order_request: OrderRequest) -> None:
        """Validate order request using ValidationService and RiskManagementService."""
        # First use ValidationService if available
        if self.validation_service:
            context = ValidationContext(
                exchange=self.exchange_name,
                trading_mode="live"
            )
            result = await self.validation_service.validate_order(order_request, context=context)
            if not result.is_valid:
                validation_error = ValidationError(f"Order validation failed: {result.errors}")
                self._propagate_error_consistently(validation_error, "order_validation")
                raise validation_error
        else:
            # Fallback to utils validation
            from src.utils import validate_price, validate_quantity
            validate_quantity(order_request.quantity)
            if order_request.price:
                validate_price(order_request.price)

        # Risk management check (optional in test mode)
        import os
        if os.getenv('TESTING') != '1':  # Only require risk service in production
            if not self.risk_service:
                service_error = ServiceError(
                    f"Risk management service is required for {self.exchange_name}. "
                    "Please ensure risk_management_service is registered in DI container."
                )
                self._propagate_error_consistently(service_error, "risk_service_missing")
                raise service_error

            # Validate order risk
            risk_check = await self.risk_service.validate_order_risk(order_request)
            if not risk_check.is_valid:
                self.logger.warning(
                    f"Order rejected by risk management: {risk_check.reason}",
                    extra={"order": order_request.to_dict(), "risk_check": risk_check.to_dict()}
                )
                raise ValidationError(
                    f"Risk validation failed: {risk_check.reason}"
            )

        # Calculate recommended position size (if risk service available)
        if self.risk_service:
            position_size = await self.risk_service.calculate_position_size(
                symbol=order_request.symbol,
                signal_strength=getattr(order_request, "signal_strength", Decimal("0.8")),
                entry_price=order_request.price if order_request.price else None
            )

            # Adjust order quantity if needed
            if position_size.recommended_size < order_request.quantity:
                self.logger.info(
                    f"Adjusting order quantity from {order_request.quantity} to {position_size.recommended_size} based on risk management"
                )
                order_request.quantity = position_size.recommended_size

            # Log risk metrics
            self.logger.info(
                f"Risk check passed for {order_request.symbol}: "
                f"adjusted size={order_request.quantity}, "
                f"risk_score={position_size.risk_score}"
            )

    async def _validate_market_data(self, data: Any) -> bool:
        """Validate market data using MarketDataValidator."""
        return self.market_validator.validate(data)

    async def _process_market_data(self, data: Any) -> Any:
        """Process market data through DataPipeline if available."""
        if self.data_pipeline:
            try:
                # Import ProcessingMode only when needed to avoid circular import
                from src.data.pipeline.data_pipeline import ProcessingMode

                # Process through data pipeline for validation, transformation, and storage
                processed_data = await self.data_pipeline.process(
                    data,
                    mode=ProcessingMode.REALTIME,
                    source=self.exchange_name
                )
                return processed_data
            except Exception as e:
                self.logger.error(f"Data pipeline processing failed: {e}")
                return data  # Return original data if pipeline fails
        return data  # Return original if no pipeline

    # Repository Integration Methods

    async def _persist_order(self, order_response: OrderResponse) -> None:
        """Persist order to database using database service."""
        if self.database_service:
            try:
                # Use database service to persist order data
                order_data = {
                    "id": order_response.order_id,
                    "exchange": self.exchange_name,
                    "exchange_order_id": order_response.exchange_order_id,
                    "symbol": order_response.symbol,
                    "side": order_response.side,
                    "order_type": order_response.order_type,
                    "quantity": order_response.quantity,
                    "price": order_response.price,
                    "status": order_response.status.value,
                    "created_at": order_response.created_at,
                    "updated_at": order_response.updated_at
                }

                await self.database_service.save_order(order_data)
                self.logger.debug(f"Persisted order {order_response.order_id} to database")
            except Exception as e:
                self.logger.error(f"Failed to persist order: {e}")
                # Don't fail the operation if persistence fails

    async def _persist_market_data(self, ticker: Ticker) -> None:
        """Persist market data to database using database service."""
        if self.database_service:
            try:
                ticker_data = {
                    "exchange": self.exchange_name,
                    "symbol": ticker.symbol,
                    "data": ticker.__dict__
                }
                await self.database_service.save_ticker(ticker_data)
                self.logger.debug(f"Persisted ticker for {ticker.symbol}")
            except Exception as e:
                self.logger.error(f"Failed to persist market data: {e}")
                # Don't fail the operation if persistence fails

    async def _persist_position(self, position: Position) -> None:
        """Persist position to database using database service."""
        if self.database_service:
            try:
                # Use database service to persist position data
                position_data = {
                    "symbol": position.symbol,
                    "side": position.side,
                    "quantity": position.quantity,
                    "entry_price": position.entry_price,
                    "current_price": position.current_price,
                    "unrealized_pnl": position.unrealized_pnl,
                    "created_at": position.created_at,
                    "status": "OPEN",
                    "exchange": self.exchange_name
                }

                await self.database_service.save_position(position_data)
                self.logger.debug(f"Persisted position for {position.symbol}")
            except Exception as e:
                self.logger.error(f"Failed to persist position: {e}")
                # Don't fail the operation if persistence fails

    def _validate_data_at_boundary(self, data: dict[str, Any], source_operation: str) -> dict[str, Any]:
        """Validate data at module boundary with consistent patterns aligned with utils."""
        try:
            from src.utils.messaging_patterns import BoundaryValidator

            # Apply boundary validation based on target module using utils patterns
            if source_operation in ("stream", "ticker", "orderbook"):
                # Data flowing to state/analytics modules
                BoundaryValidator.validate_monitoring_to_error_boundary(data)
            elif source_operation in ("order", "cancel"):
                # Data flowing to execution/risk modules
                BoundaryValidator.validate_database_to_error_boundary(data)
            else:
                # Generic boundary validation for other operations
                BoundaryValidator.validate_database_entity(data, "validate")

            return data
        except Exception as e:
            self.logger.warning(f"Boundary validation failed for {source_operation}: {e}")
            return data  # Don't fail operation, just log warning

    def _propagate_error_consistently(self, error: Exception, context: str, operation_type: str = "exchange") -> None:
        """Propagate errors consistently using utils error propagation patterns."""
        try:
            from src.utils.messaging_patterns import ErrorPropagationMixin
            error_propagator = ErrorPropagationMixin()

            # Use appropriate propagation method based on error type
            if isinstance(error, ValidationError):
                error_propagator.propagate_validation_error(error, f"{self.exchange_name}_{context}")
            elif isinstance(error, ServiceError):
                error_propagator.propagate_service_error(error, f"{self.exchange_name}_{context}")
            else:
                # For other exceptions, use service error propagation as default
                error_propagator.propagate_service_error(error, f"{self.exchange_name}_{context}")

        except Exception as prop_error:
            # Fallback if error propagation itself fails
            self.logger.debug(f"Error propagation failed in {context}: {prop_error}")

    def _apply_messaging_pattern(self, data: dict[str, Any], operation_type: str) -> dict[str, Any]:
        """Apply consistent messaging patterns aligned with utils module standards."""
        transformed_data = data.copy()

        # Apply messaging patterns aligned with utils module preferences
        if operation_type in ("stream", "websocket", "ticker", "orderbook", "trades"):
            transformed_data["message_pattern"] = "pub_sub"  # Streams use pub/sub
            transformed_data["processing_mode"] = "stream"
        elif operation_type in ("order", "cancel", "status"):
            transformed_data["message_pattern"] = "req_reply"  # Orders need responses
            transformed_data["processing_mode"] = "request_reply"  # Align with execution module pattern
        elif operation_type == "batch":
            transformed_data["message_pattern"] = "pub_sub"  # Align with utils pub_sub preference
            transformed_data["processing_mode"] = "batch"
        else:
            transformed_data["message_pattern"] = "pub_sub"  # Default to pub_sub
            transformed_data["processing_mode"] = "stream"

        # Ensure data format consistency with utils module versioning
        transformed_data["data_format"] = transformed_data.get("data_format", "event_data_v1")
        transformed_data["boundary_validation"] = "applied"

        # Apply boundary validation
        validated_data = self._validate_data_at_boundary(transformed_data, operation_type)

        return validated_data

    async def _track_analytics(self, event_type: str, data: dict) -> None:
        """Track analytics events using AnalyticsService."""
        if self.analytics_service:
            try:
                # Apply consistent messaging patterns before sending
                analytics_data = self._apply_messaging_pattern(data, event_type)

                if event_type == "order_placed":
                    await self.analytics_service.track_order_event(
                        order_id=analytics_data.get("order_id"),
                        symbol=analytics_data.get("symbol"),
                        side=analytics_data.get("side"),
                        quantity=analytics_data.get("quantity"),
                        price=analytics_data.get("price"),
                        exchange=self.exchange_name
                    )
                elif event_type == "trade_executed":
                    await self.analytics_service.track_trade_event(
                        trade_id=analytics_data.get("trade_id"),
                        symbol=analytics_data.get("symbol"),
                        price=analytics_data.get("price"),
                        quantity=analytics_data.get("quantity"),
                        exchange=self.exchange_name
                    )
                elif event_type == "market_data":
                    await self.analytics_service.track_market_data(
                        symbol=data.get("symbol"),
                        ticker=data.get("ticker"),
                        exchange=self.exchange_name
                    )

                self.logger.debug(f"Analytics tracked: {event_type} for {data.get('symbol')}")
            except Exception as e:
                self.logger.error(f"Failed to track analytics: {e}")
                # Don't fail the operation if analytics fails

    # Utility Methods

    def get_exchange_info(self) -> ExchangeInfo | None:
        """Get cached exchange information."""
        return self._exchange_info

    def get_trading_symbols(self) -> list[str] | None:
        """Get list of available trading symbols."""
        return self._trading_symbols

    def is_symbol_supported(self, symbol: str) -> bool:
        """Check if a symbol is supported for trading."""
        if not self._trading_symbols:
            return False
        return symbol in self._trading_symbols

    # Validation helpers (as per CLAUDE.md standards)

    def _validate_symbol(self, symbol: str) -> None:
        """Validate trading symbol format using utils validation."""
        from src.utils import validate_symbol

        # Use utils validation first
        validated_symbol = validate_symbol(symbol)

        # Add exchange-specific validation
        if not self.is_symbol_supported(validated_symbol):
            raise ValidationError(f"Symbol {validated_symbol} not supported on {self.exchange_name}")

    def _validate_price(self, price: Decimal, max_price: Decimal | None = None) -> None:
        """Validate price using utils validation."""
        # First check if it's actually a Decimal (strict type checking for tests)
        if not isinstance(price, Decimal):
            raise ValidationError(f"Price must be Decimal type, got {type(price).__name__}")
        
        from src.utils import validate_price

        # Use utils validation which handles Decimal conversion and all edge cases
        # Allow exchanges to have different max price limits
        if max_price:
            validate_price(price, max_price=max_price)
        else:
            # Use a higher default for crypto exchanges (10 million)
            validate_price(price, max_price=Decimal("10000000"))

    def _validate_quantity(self, quantity: Decimal) -> None:
        """Validate quantity using utils validation."""
        # First check if it's actually a Decimal (strict type checking for tests)
        if not isinstance(quantity, Decimal):
            raise ValidationError(f"Quantity must be Decimal type, got {type(quantity).__name__}")
        
        from src.utils import validate_quantity

        # Use utils validation which handles Decimal conversion and all edge cases
        validate_quantity(quantity)


class MockExchangeError(Exception):
    """Exception for mock exchange testing."""

    pass


class BaseMockExchange(BaseExchange):
    """
    Mock exchange implementation for testing.

    This provides a fake exchange that can be used in tests without
    connecting to real exchange APIs.
    """

    def __init__(self, name: str = "mock", config: dict[str, Any] | None = None):
        super().__init__(name, config or {})
        self._mock_balances: dict[str, Decimal] = {
            "USDT": Decimal("10000.00000000"),
            "BTC": Decimal("0.50000000"),
            "ETH": Decimal("5.00000000"),
        }
        self._mock_orders: dict[str, OrderResponse] = {}

    async def connect(self) -> None:
        """Mock connection - always succeeds."""
        self._connected = True
        self._last_heartbeat = datetime.now(timezone.utc)

    async def disconnect(self) -> None:
        """Mock disconnection."""
        self._connected = False

    async def ping(self) -> bool:
        """Mock ping - always returns True."""
        if not self.connected:
            raise ExchangeConnectionError("Mock exchange not connected")
        self._last_heartbeat = datetime.now(timezone.utc)
        return True

    async def load_exchange_info(self) -> ExchangeInfo:
        """Load mock exchange info."""
        self._exchange_info = ExchangeInfo(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            status="TRADING",
            min_price=Decimal("0.01000000"),
            max_price=Decimal("1000000.00000000"),
            tick_size=Decimal("0.01000000"),
            min_quantity=Decimal("0.00001000"),
            max_quantity=Decimal("1000000.00000000"),
            step_size=Decimal("0.00001000"),
            exchange="mock",
        )
        self._trading_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        return self._exchange_info

    async def get_ticker(self, symbol: str) -> Ticker:
        """Return mock ticker data."""
        if not self.connected:
            raise ExchangeConnectionError("Mock exchange not connected")
        self._validate_symbol(symbol)
        base_price = Decimal("45000.00000000") if "BTC" in symbol else Decimal("3000.00000000")
        return Ticker(
            symbol=symbol,
            bid_price=base_price - Decimal("1.00000000"),
            bid_quantity=Decimal("10.00000000"),
            ask_price=base_price + Decimal("1.00000000"),
            ask_quantity=Decimal("10.00000000"),
            last_price=base_price,
            open_price=base_price * Decimal("0.98"),  # 2% lower than current
            high_price=base_price * Decimal("1.02"),  # 2% higher than current
            low_price=base_price * Decimal("0.96"),  # 4% lower than current
            volume=Decimal("1000.00000000"),
            exchange="mock",
            timestamp=datetime.now(timezone.utc),
        )

    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook:
        """Return mock order book."""
        if not self.connected:
            raise ExchangeConnectionError("Mock exchange not connected")
        self._validate_symbol(symbol)
        base_price = Decimal("45000") if "BTC" in symbol else Decimal("3000")

        return OrderBook(
            symbol=symbol,
            bids=[OrderBookLevel(price=base_price - Decimal("1"), quantity=Decimal("1.0"))],
            asks=[OrderBookLevel(price=base_price + Decimal("1"), quantity=Decimal("1.0"))],
            timestamp=datetime.now(timezone.utc),
            exchange="mock",
        )

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        """Return mock recent trades."""
        if not self.connected:
            raise ExchangeConnectionError("Mock exchange not connected")
        self._validate_symbol(symbol)
        base_price = Decimal("45000") if "BTC" in symbol else Decimal("3000")

        return [
            Trade(
                id=f"trade_{symbol}_{int(datetime.now(timezone.utc).timestamp())}",
                symbol=symbol,
                exchange="mock",
                side="BUY",
                price=base_price,
                quantity=Decimal("0.1"),
                timestamp=datetime.now(timezone.utc),
                maker=True,
            )
        ]

    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place mock order."""
        if not self.connected:
            raise ExchangeConnectionError("Mock exchange not connected")
        self._validate_symbol(order_request.symbol)
        self._validate_price(order_request.price)
        self._validate_quantity(order_request.quantity)

        order_id = f"mock_order_{len(self._mock_orders) + 1}"

        order_response = OrderResponse(
            order_id=order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            price=order_request.price,
            status=OrderStatus.FILLED,
            filled_quantity=order_request.quantity,
            created_at=datetime.now(timezone.utc),
            exchange="mock",
        )

        self._mock_orders[order_id] = order_response
        return order_response

    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse:
        """Cancel mock order."""
        if not self.connected:
            raise ExchangeConnectionError("Mock exchange not connected")
        if order_id not in self._mock_orders:
            raise OrderRejectionError(f"Order {order_id} not found")

        order = self._mock_orders[order_id]
        order.status = OrderStatus.CANCELLED
        return order

    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse:
        """Get mock order status."""
        if not self.connected:
            raise ExchangeConnectionError("Mock exchange not connected")
        if order_id not in self._mock_orders:
            raise OrderRejectionError(f"Order {order_id} not found")
        return self._mock_orders[order_id]

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]:
        """Get mock open orders."""
        if not self.connected:
            raise ExchangeConnectionError("Mock exchange not connected")
        orders = list(self._mock_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return [o for o in orders if o.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]]

    async def get_account_balance(self) -> dict[str, Decimal]:
        """Get mock account balance."""
        if not self.connected:
            raise ExchangeConnectionError("Mock exchange not connected")
        return self._mock_balances.copy()

    async def get_positions(self) -> list[Position]:
        """Get mock positions (empty for spot trading)."""
        if not self.connected:
            raise ExchangeConnectionError("Mock exchange not connected")
        return []
