"""
Protocol interfaces for base classes providing type safety and contract definitions.

These protocols define the contracts that all base classes must follow,
ensuring consistent behavior across the entire system.
"""

from __future__ import annotations

import abc
import builtins
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import ConfigDict

# Type variables for generic protocols
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class HealthStatus(Enum):
    """Health status enumeration for components."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheckResult:
    """Result of a health check operation."""

    def __init__(
        self,
        status: HealthStatus,
        details: dict[str, Any] | None = None,
        message: str | None = None,
        check_time: datetime | None = None,
    ):
        self.status = status
        self.details = details or {}
        self.message = message
        self.check_time = check_time or datetime.now(timezone.utc)

    @property
    def healthy(self) -> bool:
        """Check if the component is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def component(self) -> str:
        """Get component name from details or return default."""
        return self.details.get("component", "Unknown")

    def to_dict(self) -> dict[str, Any]:
        """Convert health check result to dictionary."""
        return {
            "status": self.status.value,
            "details": self.details,
            "message": self.message,
            "check_time": self.check_time.isoformat(),
        }


class Lifecycle(Protocol):
    """Protocol for components with lifecycle management."""

    async def start(self) -> None:
        """Start the component and initialize resources."""
        ...

    async def stop(self) -> None:
        """Stop the component and cleanup resources."""
        ...

    async def restart(self) -> None:
        """Restart the component."""
        ...

    @property
    def is_running(self) -> bool:
        """Check if component is currently running."""
        ...


class HealthCheckable(Protocol):
    """Protocol for components that support health checks."""

    async def health_check(self) -> HealthCheckResult:
        """Perform health check and return status."""
        ...

    async def ready_check(self) -> HealthCheckResult:
        """Check if component is ready to serve requests."""
        ...

    async def live_check(self) -> HealthCheckResult:
        """Check if component is alive and responsive."""
        ...


class Injectable(Protocol):
    """Protocol for dependency injection support."""

    def configure_dependencies(self, container: Any) -> None:
        """Configure component dependencies."""
        ...

    def get_dependencies(self) -> list[str]:
        """Get list of required dependencies."""
        ...


class Loggable(Protocol):
    """Protocol for components with structured logging."""

    @property
    def logger(self) -> Any:  # Logger type from core.logging
        """Get logger instance for this component."""
        ...

    @property
    def correlation_id(self) -> str | None:
        """Get correlation ID for request tracing."""
        ...


class Monitorable(Protocol):
    """Protocol for components with metrics and monitoring."""

    def get_metrics(self) -> dict[str, int | float | str]:
        """Get current component metrics."""
        ...

    def reset_metrics(self) -> None:
        """Reset component metrics."""
        ...


class Configurable(Protocol):
    """Protocol for components with configuration support."""

    def configure(self, config: ConfigDict) -> None:
        """Configure component with provided settings."""
        ...

    def get_config(self) -> ConfigDict:
        """Get current component configuration."""
        ...

    def validate_config(self, config: ConfigDict) -> bool:
        """Validate configuration settings."""
        ...


class Repository(Protocol):
    """Protocol for repository pattern implementation."""

    async def create(self, entity: Any) -> Any:
        """Create new entity."""
        ...

    async def get_by_id(self, entity_id: Any) -> Any | None:
        """Get entity by ID."""
        ...

    async def update(self, entity: Any) -> Any:
        """Update existing entity."""
        ...

    async def delete(self, entity_id: Any) -> bool:
        """Delete entity by ID."""
        ...

    async def list(
        self,
        limit: int | None = None,
        offset: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        """List entities with optional pagination and filtering."""
        ...

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count entities with optional filtering."""
        ...


class Factory(Protocol):
    """Protocol for factory pattern implementation."""

    def register(self, name: str, creator_func: Any) -> None:
        """Register creator function for given name."""
        ...

    def unregister(self, name: str) -> None:
        """Unregister creator function."""
        ...

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Create instance using registered creator."""
        ...

    def list_registered(self) -> list[str]:
        """List all registered creator names."""
        ...


class EventEmitter(Protocol):
    """Protocol for event emission and subscription."""

    def emit(self, event: str, data: Any = None) -> None:
        """Emit event with optional data."""
        ...

    def on(self, event: str, callback: Any) -> Any:
        """Subscribe to event."""
        ...

    def off(self, event: str, callback: Any | None = None) -> None:
        """Unsubscribe from event."""
        ...

    def once(self, event: str, callback: Any) -> Any:
        """Subscribe to event for single execution."""
        ...

    def remove_all_listeners(self, event: str | None = None) -> None:
        """Remove all listeners for event or all events."""
        ...


class DIContainer(Protocol):
    """Protocol for dependency injection container."""

    def register(
        self,
        interface: type,
        implementation: type | Any,
        singleton: bool = False,
    ) -> None:
        """Register service implementation."""
        ...

    def resolve(self, interface: type) -> Any:
        """Resolve service instance."""
        ...

    def is_registered(self, interface: type) -> bool:
        """Check if service is registered."""
        ...

    def register_factory(
        self,
        name: str,
        factory_func: Any,
        singleton: bool = False,
    ) -> None:
        """Register factory function."""
        ...


class AsyncContextManager(Protocol):
    """Protocol for async context managers."""

    async def __aenter__(self) -> Any:
        """Async context enter."""
        ...

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context exit."""
        ...


# Composite protocols for common combinations
class ServiceComponent(Protocol):
    """Combined protocol for service layer components."""

    # Lifecycle methods
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def restart(self) -> None: ...
    @property
    def is_running(self) -> bool: ...

    # Health check methods
    async def health_check(self) -> HealthCheckResult: ...
    async def ready_check(self) -> HealthCheckResult: ...
    async def live_check(self) -> HealthCheckResult: ...

    # Injectable methods
    def configure_dependencies(self, container: Any) -> None: ...
    def get_dependencies(self) -> list[str]: ...

    # Loggable properties
    @property
    def logger(self) -> Any: ...
    @property
    def correlation_id(self) -> str | None: ...

    # Monitorable methods
    def get_metrics(self) -> dict[str, int | float | str]: ...
    def reset_metrics(self) -> None: ...

    # Configurable methods
    def configure(self, config: ConfigDict) -> None: ...
    def get_config(self) -> ConfigDict: ...
    def validate_config(self, config: ConfigDict) -> bool: ...


class RepositoryComponent(Protocol):
    """Combined protocol for repository layer components."""

    # Repository methods
    async def create(self, entity: Any) -> Any: ...
    async def get_by_id(self, entity_id: Any) -> Any | None: ...
    async def update(self, entity: Any) -> Any: ...
    async def delete(self, entity_id: Any) -> bool: ...
    async def list(
        self,
        limit: int | None = None,
        offset: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]: ...
    async def count(self, filters: dict[str, Any] | None = None) -> int: ...

    # Health check methods
    async def health_check(self) -> HealthCheckResult: ...
    async def ready_check(self) -> HealthCheckResult: ...
    async def live_check(self) -> HealthCheckResult: ...

    # Injectable methods
    def configure_dependencies(self, container: Any) -> None: ...
    def get_dependencies(self) -> builtins.list[str]: ...

    # Loggable properties
    @property
    def logger(self) -> Any: ...
    @property
    def correlation_id(self) -> str | None: ...


class FactoryComponent(Protocol):
    """Combined protocol for factory components."""

    # Factory methods
    def register(self, name: str, creator_func: Any) -> None: ...
    def unregister(self, name: str) -> None: ...
    def create(self, name: str, *args: Any, **kwargs: Any) -> Any: ...
    def list_registered(self) -> list[str]: ...

    # Injectable methods
    def configure_dependencies(self, container: Any) -> None: ...
    def get_dependencies(self) -> list[str]: ...

    # Loggable properties
    @property
    def logger(self) -> Any: ...
    @property
    def correlation_id(self) -> str | None: ...


# Web Service Interface Protocols
class WebServiceInterface(Protocol):
    """Base interface for web service implementations."""

    async def initialize(self) -> None:
        """Initialize the service."""
        ...

    async def cleanup(self) -> None:
        """Cleanup service resources."""
        ...


class TradingServiceInterface(WebServiceInterface):
    """Interface for trading operations."""

    @abc.abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,  # Use string to avoid circular dependency
        order_type: str,  # Use string to avoid circular dependency
        amount: Any,  # Use Any to avoid decimal import issues
        price: Any | None = None,
    ) -> str:
        """Place a trading order."""
        ...

    @abc.abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        ...

    @abc.abstractmethod
    async def get_positions(self) -> list[Any]:
        """Get current positions."""
        ...


class BotManagementServiceInterface(WebServiceInterface):
    """Interface for bot management operations."""

    @abc.abstractmethod
    async def create_bot(self, config: Any) -> str:
        """Create a new trading bot."""
        ...

    @abc.abstractmethod
    async def start_bot(self, bot_id: str) -> bool:
        """Start a bot."""
        ...

    @abc.abstractmethod
    async def stop_bot(self, bot_id: str) -> bool:
        """Stop a bot."""
        ...

    @abc.abstractmethod
    async def get_bot_status(self, bot_id: str) -> dict[str, Any]:
        """Get bot status."""
        ...

    @abc.abstractmethod
    async def list_bots(self) -> list[dict[str, Any]]:
        """List all bots."""
        ...

    @abc.abstractmethod
    async def get_all_bots_status(self) -> dict[str, Any]:
        """Get status of all bots."""
        ...

    @abc.abstractmethod
    async def delete_bot(self, bot_id: str, force: bool = False) -> bool:
        """Delete a bot."""
        ...


class MarketDataServiceInterface(WebServiceInterface):
    """Interface for market data operations."""

    @abc.abstractmethod
    async def get_ticker(self, symbol: str) -> Any:
        """Get current ticker data."""
        ...

    @abc.abstractmethod
    async def subscribe_to_ticker(self, symbol: str, callback: Any) -> None:
        """Subscribe to ticker updates."""
        ...

    @abc.abstractmethod
    async def unsubscribe_from_ticker(self, symbol: str) -> None:
        """Unsubscribe from ticker updates."""
        ...


class PortfolioServiceInterface(WebServiceInterface):
    """Interface for portfolio operations."""

    @abc.abstractmethod
    async def get_balance(self) -> dict[str, Any]:
        """Get account balances."""
        ...

    @abc.abstractmethod
    async def get_portfolio_summary(self) -> dict[str, Any]:
        """Get portfolio summary."""
        ...

    @abc.abstractmethod
    async def get_pnl_report(self, start_date: Any, end_date: Any) -> dict[str, Any]:
        """Get P&L report for date range."""
        ...


class RiskServiceInterface(WebServiceInterface):
    """Interface for risk management operations."""

    @abc.abstractmethod
    async def validate_order(
        self, symbol: str, side: str, amount: Any, price: Any | None = None
    ) -> dict[str, Any]:
        """Validate an order against risk rules."""
        ...

    @abc.abstractmethod
    async def get_risk_metrics(self) -> dict[str, Any]:
        """Get current risk metrics."""
        ...

    @abc.abstractmethod
    async def update_risk_limits(self, limits: dict[str, Any]) -> bool:
        """Update risk limits."""
        ...


class StrategyServiceInterface(WebServiceInterface):
    """Interface for strategy operations."""

    @abc.abstractmethod
    async def list_strategies(self) -> list[dict[str, Any]]:
        """List available strategies."""
        ...

    @abc.abstractmethod
    async def get_strategy_config(self, strategy_name: str) -> dict[str, Any]:
        """Get strategy configuration."""
        ...

    @abc.abstractmethod
    async def validate_strategy_config(self, strategy_name: str, config: dict[str, Any]) -> bool:
        """Validate strategy configuration."""
        ...


# Infrastructure Service Interfaces
class CacheClientInterface(Protocol):
    """Interface for cache client implementations (Redis, etc.)."""

    async def connect(self) -> None:
        """Connect to the cache server."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from the cache server."""
        ...

    async def ping(self) -> bool:
        """Ping the cache server."""
        ...

    async def get(self, key: str, namespace: str = "cache") -> Any | None:
        """Get value from cache."""
        ...

    async def set(self, key: str, value: Any, ttl: int | None = None, namespace: str = "cache") -> bool:
        """Set value in cache with optional TTL."""
        ...

    async def delete(self, key: str, namespace: str = "cache") -> bool:
        """Delete key from cache."""
        ...

    async def exists(self, key: str, namespace: str = "cache") -> bool:
        """Check if key exists in cache."""
        ...

    async def expire(self, key: str, ttl: int, namespace: str = "cache") -> bool:
        """Set expiration for existing key."""
        ...

    async def info(self) -> dict[str, Any]:
        """Get cache server information."""
        ...

    def _get_namespaced_key(self, key: str, namespace: str) -> str:
        """Get namespaced key."""
        ...

    @property
    def client(self) -> Any:
        """Get underlying client instance."""
        ...


@runtime_checkable
class DatabaseServiceInterface(Protocol):
    """Interface for database service implementations."""

    async def start(self) -> None:
        """Start database service."""
        ...

    async def stop(self) -> None:
        """Stop database service."""
        ...

    async def health_check(self) -> HealthCheckResult:
        """Perform database health check."""
        ...

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get database performance metrics."""
        ...

    async def execute_query(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a database query."""
        ...

    async def get_connection_pool_status(self) -> dict[str, Any]:
        """Get connection pool status."""
        ...
