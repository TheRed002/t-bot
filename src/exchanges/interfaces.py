"""
Exchange module interfaces for state integration and service contracts.

This module defines the interface contracts used by the exchange module
to avoid circular dependencies and enforce proper service layer patterns.
"""

from decimal import Decimal
from enum import Enum
from typing import Any, Protocol

# Use shared types from core
from src.core.types import (
    ExchangeInfo,
    MarketData,
    OrderBook,
    OrderRequest,
    OrderResponse,
    Position,
    StateType,
    Ticker,
    Trade,
)
from src.state.state_service import StatePriority


class TradeEvent(str, Enum):
    """Trade event enumeration (mirror of state module)."""

    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACCEPTED = "order_accepted"
    ORDER_REJECTED = "order_rejected"
    PARTIAL_FILL = "partial_fill"
    COMPLETE_FILL = "complete_fill"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_EXPIRED = "order_expired"


class IStateService(Protocol):
    """Interface for StateService used by exchanges."""

    async def set_state(
        self,
        state_type: StateType,
        state_id: str,
        state_data: dict[str, Any],
        source_component: str,
        priority: StatePriority,
        reason: str,
    ) -> bool:
        """Set state data."""
        ...

    async def get_state(self, state_type: StateType, state_id: str) -> dict[str, Any] | None:
        """Get state data."""
        ...


class ITradeLifecycleManager(Protocol):
    """Interface for TradeLifecycleManager used by exchanges."""

    async def update_trade_event(
        self, trade_id: str, event: TradeEvent, event_data: dict[str, Any]
    ) -> None:
        """Update trade event."""
        ...


class IExchange(Protocol):
    """
    Interface contract for exchange implementations.

    This interface defines the contract that all exchange implementations
    must follow, enabling proper dependency injection and service decoupling.
    """

    # Connection Management
    async def connect(self) -> bool:
        """Establish connection to the exchange."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        ...

    async def health_check(self) -> bool:
        """Perform health check on exchange connection."""
        ...

    def is_connected(self) -> bool:
        """Check if exchange is connected."""
        ...

    # Trading Operations
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place a trading order."""
        ...

    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse:
        """Cancel an existing order."""
        ...

    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse:
        """Get status of an order."""
        ...

    # Market Data Operations
    async def get_market_data(self, symbol: str, timeframe: str = "1m") -> MarketData:
        """Get market data for a symbol."""
        ...

    async def get_order_book(self, symbol: str, depth: int = 10) -> OrderBook:
        """Get order book for a symbol."""
        ...

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker data for a symbol."""
        ...

    async def get_trade_history(self, symbol: str, limit: int = 100) -> list[Trade]:
        """Get trade history for a symbol."""
        ...

    # Account Operations
    async def get_account_balance(self) -> dict[str, Decimal]:
        """Get account balances."""
        ...

    async def get_positions(self) -> list[Position]:
        """Get open positions."""
        ...

    # Exchange Information
    async def get_exchange_info(self) -> ExchangeInfo:
        """Get exchange information."""
        ...

    # WebSocket Operations
    async def subscribe_to_stream(self, symbol: str, callback: Any) -> None:
        """Subscribe to real-time data stream."""
        ...

    # Properties
    @property
    def exchange_name(self) -> str:
        """Get exchange name."""
        ...


class IConnectionManager(Protocol):
    """Interface for connection management implementations."""

    async def connect(self) -> bool:
        """Establish connection."""
        ...

    async def disconnect(self) -> None:
        """Close connection."""
        ...

    def is_connected(self) -> bool:
        """Check connection status."""
        ...

    async def request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        signed: bool = False,
    ) -> dict[str, Any]:
        """Make HTTP request."""
        ...


class IRateLimiter(Protocol):
    """Interface for rate limiting implementations."""

    async def acquire(self, weight: int = 1) -> bool:
        """Acquire rate limit tokens."""
        ...

    async def release(self, weight: int = 1) -> None:
        """Release rate limit tokens."""
        ...

    def reset(self) -> None:
        """Reset rate limiter."""
        ...

    def get_statistics(self) -> dict[str, Any]:
        """Get rate limit statistics."""
        ...


class IHealthMonitor(Protocol):
    """Interface for health monitoring implementations."""

    def record_success(self) -> None:
        """Record successful operation."""
        ...

    def record_failure(self) -> None:
        """Record failed operation."""
        ...

    def record_latency(self, latency_ms: float) -> None:
        """Record operation latency."""
        ...

    def get_health_status(self) -> dict[str, Any]:
        """Get current health status."""
        ...

    async def check_health(self) -> bool:
        """Perform health check."""
        ...


class IExchangeAdapter(Protocol):
    """Interface for exchange adapter implementations."""

    async def place_order(self, **kwargs) -> dict[str, Any]:
        """Place order via adapter."""
        ...

    async def cancel_order(self, order_id: str, **kwargs) -> dict[str, Any]:
        """Cancel order via adapter."""
        ...

    async def get_order(self, order_id: str, **kwargs) -> dict[str, Any]:
        """Get order via adapter."""
        ...

    async def get_balance(self, asset: str | None = None) -> dict[str, Any]:
        """Get balance via adapter."""
        ...

    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        """Get ticker via adapter."""
        ...


class IExchangeFactory(Protocol):
    """
    Interface for exchange factory implementations.

    This interface defines the contract for creating and managing
    exchange instances.
    """

    def get_supported_exchanges(self) -> list[str]:
        """Get list of supported exchanges."""
        ...

    def get_available_exchanges(self) -> list[str]:
        """Get list of configured exchanges."""
        ...

    def is_exchange_supported(self, exchange_name: str) -> bool:
        """Check if exchange is supported."""
        ...

    async def get_exchange(
        self, exchange_name: str, create_if_missing: bool = True, force_recreate: bool = False
    ) -> IExchange | None:
        """Get or create exchange instance."""
        ...

    async def create_exchange(self, exchange_name: str) -> IExchange:
        """Create new exchange instance."""
        ...

    async def remove_exchange(self, exchange_name: str) -> bool:
        """Remove exchange instance."""
        ...

    async def health_check_all(self) -> dict[str, Any]:
        """Health check all exchanges."""
        ...

    async def disconnect_all(self) -> None:
        """Disconnect all exchanges."""
        ...


# Sandbox-specific interfaces
from decimal import Decimal
from enum import Enum


class SandboxMode(str, Enum):
    """Sandbox operation modes."""

    PRODUCTION = "production"
    SANDBOX = "sandbox"
    MOCK = "mock"
    HYBRID = "hybrid"  # Mix of sandbox and mock


class ISandboxConnectionManager(Protocol):
    """
    Interface for sandbox connection management.

    Extends basic connection management with sandbox-specific features
    like endpoint switching and credential validation.
    """

    async def connect_to_sandbox(self) -> bool:
        """Connect to sandbox endpoints."""
        ...

    async def connect_to_production(self) -> bool:
        """Connect to production endpoints."""
        ...

    async def switch_environment(self, mode: SandboxMode) -> bool:
        """Switch between environments dynamically."""
        ...

    def get_current_endpoints(self) -> dict[str, str]:
        """Get current API and WebSocket endpoints."""
        ...

    async def validate_environment(self) -> dict[str, Any]:
        """Validate current environment configuration."""
        ...


class ISandboxAdapter(Protocol):
    """
    Interface for sandbox-specific exchange adapters.

    This adapter pattern enables switching between production, sandbox,
    and mock implementations while maintaining the same interface.
    """

    @property
    def sandbox_mode(self) -> SandboxMode:
        """Get current sandbox mode."""
        ...

    async def validate_sandbox_credentials(self) -> bool:
        """Validate sandbox credentials are working."""
        ...

    async def reset_sandbox_account(self) -> bool:
        """Reset sandbox account to initial state (if supported)."""
        ...

    async def get_sandbox_balance(self, asset: str | None = None) -> dict[str, Any]:
        """Get sandbox account balance."""
        ...

    async def place_sandbox_order(self, order: OrderRequest) -> OrderResponse:
        """Place order in sandbox environment."""
        ...

    async def simulate_order_fill(self, order_id: str, fill_percentage: float = 1.0) -> bool:
        """Simulate order fill for testing purposes."""
        ...

    async def inject_market_data(self, symbol: str, data: dict[str, Any]) -> bool:
        """Inject mock market data for testing."""
        ...

    async def simulate_network_error(self, error_type: str, duration_seconds: int = 5) -> None:
        """Simulate network errors for testing."""
        ...
